"""VRAM-aware head manager with GPU mutex."""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import AsyncIterator
from typing import Any

from .adapters.anthropic_adapter import AnthropicAdapter
from .adapters.base import HeadAdapter
from .adapters.botvibes_adapter import BotVibesAdapter
from .adapters.claude_adapter import ClaudeAdapter
from .adapters.claude_agent_sdk import ClaudeAgentSDKAdapter
from .adapters.claude_session import ClaudeSessionAdapter
from .adapters.embedding import EmbeddingAdapter
from .adapters.mock import MockAdapter
from .adapters.deterministic_adapter import DeterministicAdapter
from .adapters.ollama import OllamaAdapter
from .adapters.openai_adapter import OpenAIAdapter
from .adapters.transformers_adapter import TransformersAdapter
from .adapters.vllm import VLLMAdapter
from .models import AdapterKind, HeadManifest, HeadState
from .observability import MetricsCollector
from .resilience import CircuitBreaker, CircuitBreakerOpen

logger = logging.getLogger(__name__)


def _create_adapter(manifest: HeadManifest, knowledge_store: Any = None) -> HeadAdapter:
    """Factory: create adapter instance from manifest.

    Args:
        manifest: Head manifest with adapter configuration
        knowledge_store: Optional KnowledgeStore for adapters that need it (claude_session)
    """
    match manifest.adapter:
        case AdapterKind.DETERMINISTIC:
            return DeterministicAdapter(model=manifest.model)
        case AdapterKind.OLLAMA:
            return OllamaAdapter(manifest)
        case AdapterKind.OPENAI:
            return OpenAIAdapter(manifest)
        case AdapterKind.VLLM:
            return VLLMAdapter(manifest)
        case AdapterKind.TRANSFORMERS:
            return TransformersAdapter(manifest)
        case AdapterKind.EMBEDDING:
            return EmbeddingAdapter(manifest)
        case AdapterKind.CLAUDE:
            return ClaudeAdapter(manifest)
        case AdapterKind.CLAUDE_SESSION:
            if not knowledge_store:
                raise ValueError(
                    "ClaudeSessionAdapter requires knowledge_store parameter - "
                    "pass it to HeadManager.__init__()"
                )
            # Extract config from manifest.extra or use defaults
            extra = manifest.extra or {}
            return ClaudeSessionAdapter(
                knowledge_store=knowledge_store,
                session_id=extra.get("session_id", "claude-main"),
                project_id=extra.get("project_id", "multihead"),
                timeout_seconds=extra.get("timeout_seconds", 300.0),
                min_responses=extra.get("min_responses", 1),
                poll_interval=extra.get("poll_interval", 2.0),
            )
        case AdapterKind.ANTHROPIC:
            return AnthropicAdapter(manifest)
        case AdapterKind.CLAUDE_AGENT_SDK:
            return ClaudeAgentSDKAdapter(manifest)
        case AdapterKind.MESH:
            from .adapters.mesh_adapter import MeshHeadAdapter
            return MeshHeadAdapter(manifest)
        case AdapterKind.BOTVIBES:
            return BotVibesAdapter(manifest, knowledge_store=knowledge_store)
        case AdapterKind.MOCK:
            return MockAdapter(manifest)
        case _:
            raise ValueError(f"Unknown adapter: {manifest.adapter}")


class HeadManager:
    """Manages model heads with GPU mutex: only one heavy head active at a time.

    Tracks state per head (OFF/LOADING/ACTIVE/SLEEPING/UNLOADING).
    Ensures safe head swapping: unload current before loading next.
    """

    def __init__(
        self,
        manifests: dict[str, HeadManifest],
        breaker_threshold: int = 5,
        breaker_timeout_s: float = 60.0,
        metrics: MetricsCollector | None = None,
        narrative_pipeline: Any | None = None,
        knowledge_store: Any | None = None,  # Gap #1: Enable multi-session voting
    ) -> None:
        self._manifests = manifests
        self._adapters: dict[str, HeadAdapter] = {}
        self._states: dict[str, HeadState] = {}
        self._breakers: dict[str, CircuitBreaker] = {}
        self._active_head: str | None = None
        self._gpu_lock = asyncio.Lock()
        self._streaming_refs: dict[str, int] = {}  # head_id -> active stream count
        self._metrics = metrics or MetricsCollector()
        self._narrative = narrative_pipeline
        self._knowledge_store = knowledge_store

        # Create adapters and circuit breakers for all registered heads
        for head_id, manifest in manifests.items():
            self._adapters[head_id] = _create_adapter(manifest, knowledge_store)
            self._states[head_id] = HeadState.OFF
            self._breakers[head_id] = CircuitBreaker(breaker_threshold, breaker_timeout_s)

    @property
    def active_head(self) -> str | None:
        return self._active_head

    @property
    def manifests(self) -> dict[str, HeadManifest]:
        """All registered head manifests."""
        return self._manifests

    def get_manifest(self, head_id: str) -> HeadManifest | None:
        """Get the manifest for a head (None if not found)."""
        return self._manifests.get(head_id)

    def get_state(self, head_id: str) -> HeadState:
        return self._states.get(head_id, HeadState.OFF)

    def get_states(self) -> dict[str, dict[str, Any]]:
        """Return all head states with metadata."""
        result = {}
        for head_id, manifest in self._manifests.items():
            breaker = self._breakers.get(head_id)
            result[head_id] = {
                "head_id": head_id,
                "name": manifest.name,
                "adapter": manifest.adapter.value,
                "model": manifest.model,
                "kind": manifest.kind,
                "state": self._states.get(head_id, HeadState.OFF).value,
                "is_active": head_id == self._active_head,
                "circuit_breaker": breaker.state if breaker else "unknown",
            }
        return result

    def get_breaker(self, head_id: str) -> CircuitBreaker | None:
        """Get the circuit breaker for a head."""
        return self._breakers.get(head_id)

    def get_adapter(self, head_id: str) -> HeadAdapter:
        if head_id not in self._adapters:
            raise KeyError(f"Unknown head: {head_id}")
        return self._adapters[head_id]

    def set_narrative_pipeline(self, pipeline: Any) -> None:
        """Set narrative pipeline for GPU activity logging."""
        self._narrative = pipeline

    def _narrate_event(self, event_type: str, title: str, summary: str, **extra: Any) -> None:
        """Log a GPU event to the narrative pipeline (fire-and-forget)."""
        if not self._narrative:
            return
        try:
            from .knowledge_models import (
                KnowledgeEvent, EventStatus, EventType, Provenance,
                TimeBlock, TimePrecision,
            )
            from datetime import datetime, timezone

            now = datetime.now(timezone.utc)
            etype_map = {
                "head_swap": EventType.TOOL_RUN,
                "generate": EventType.TOOL_RUN,
                "head_load": EventType.TOOL_RUN,
                "head_error": EventType.TOOL_RUN,
            }
            event = KnowledgeEvent(
                event_status=EventStatus.CONFIRMED,
                event_type=etype_map.get(event_type, EventType.TOOL_RUN),
                title=title,
                summary=summary,
                time=TimeBlock(
                    happened_at=now,
                    ended_at=now,
                    timezone="UTC",
                    time_precision=TimePrecision.SECOND,
                ),
                tags=["gpu", event_type, extra.get("head_id", "")],
                metrics=extra.get("metrics", {}),
                provenance=Provenance(
                    produced_by={"kind": "system", "id": "head_manager"},
                    created_at=now,
                    updated_at=now,
                ),
            )
            self._narrative.store.insert_event(event)
        except Exception as e:
            logger.debug("Narrative event failed: %s", e)

    async def ensure_active(self, head_id: str) -> None:
        """Ensure the specified head is loaded and active. Unloads current if different."""
        if head_id not in self._adapters:
            raise KeyError(f"Unknown head: {head_id}")

        if self._active_head == head_id and self._states[head_id] == HeadState.ACTIVE:
            return  # Already active

        prev_head = self._active_head
        t_wait = time.perf_counter()
        async with self._gpu_lock:
            wait_ms = (time.perf_counter() - t_wait) * 1000
            # Unload current active head if different
            if self._active_head and self._active_head != head_id:
                await self._unload(self._active_head)
                self._narrate_event(
                    "head_swap",
                    f"GPU head swap: {prev_head} → {head_id}",
                    f"Unloaded {prev_head}, loading {head_id}. Queue wait: {wait_ms:.0f}ms.",
                    head_id=head_id,
                    metrics={"queue_wait_ms": round(wait_ms), "from_head": prev_head or "none"},
                )

            # Load the requested head
            await self._load(head_id)

    async def _load(self, head_id: str) -> None:
        """Load a head (internal, must hold gpu_lock). Respects circuit breaker."""
        adapter = self._adapters[head_id]
        manifest = self._manifests[head_id]
        breaker = self._breakers[head_id]

        # Check circuit breaker before attempting load
        if breaker.state == "open":
            raise CircuitBreakerOpen(
                f"Circuit breaker open for head {head_id}. "
                f"Too many recent failures; retry later."
            )

        logger.info("Loading head: %s (%s via %s)", head_id, manifest.model, manifest.adapter.value)
        self._states[head_id] = HeadState.LOADING

        t0 = time.perf_counter()
        try:
            await adapter.load()
            breaker.record_success()
            self._states[head_id] = HeadState.ACTIVE
            self._active_head = head_id
            elapsed = time.perf_counter() - t0
            self._metrics.inc("head_loads_total", labels={"head_id": head_id})
            self._metrics.observe("head_load_seconds", elapsed, labels={"head_id": head_id})
            logger.info("Head active: %s", head_id)
        except Exception as e:
            breaker.record_failure()
            self._states[head_id] = HeadState.ERROR
            self._metrics.inc("head_load_errors_total", labels={"head_id": head_id})
            logger.error("Failed to load head %s: %s", head_id, e)
            raise

    async def _unload(self, head_id: str) -> None:
        """Unload a head (internal, must hold gpu_lock).

        Waits for any active streams to finish before unloading to
        prevent GPU resource corruption mid-stream.
        """
        # Wait for active streams to finish (up to 30s)
        refs = self._streaming_refs.get(head_id, 0)
        if refs > 0:
            logger.info("Waiting for %d active stream(s) on %s before unload", refs, head_id)
            for _ in range(300):  # 30s max
                if self._streaming_refs.get(head_id, 0) == 0:
                    break
                await asyncio.sleep(0.1)
            else:
                logger.warning("Timed out waiting for streams on %s, unloading anyway", head_id)

        adapter = self._adapters[head_id]

        logger.info("Unloading head: %s", head_id)
        self._states[head_id] = HeadState.UNLOADING

        try:
            await adapter.unload()
        except Exception as e:
            logger.warning("Error unloading head %s: %s", head_id, e)
        finally:
            self._states[head_id] = HeadState.OFF
            if self._active_head == head_id:
                self._active_head = None

    async def generate(self, head_id: str, prompt: str, **kwargs: Any) -> dict[str, Any]:
        """Generate through a head with circuit breaker protection.

        Ensures the head is active first, then wraps the generate call.
        """
        await self.ensure_active(head_id)
        adapter = self._adapters[head_id]
        breaker = self._breakers[head_id]

        # Check breaker before creating coroutine to avoid unawaited coroutine warning
        if breaker.state == "open":
            self._states[head_id] = HeadState.ERROR
            raise CircuitBreakerOpen(f"Circuit breaker open for head {head_id}")

        t0 = time.perf_counter()
        try:
            result = await breaker.call(adapter.generate(prompt, **kwargs))
            elapsed = time.perf_counter() - t0
            self._metrics.inc("head_generate_total", labels={"head_id": head_id})
            self._metrics.observe("head_generate_seconds", elapsed, labels={"head_id": head_id})
            tokens = (result.get("tokens_out") or result.get("tokens_generated") or 0) if isinstance(result, dict) else 0
            self._narrate_event(
                "generate",
                f"Generate: {head_id} ({tokens} tokens, {elapsed:.1f}s)",
                f"Head {head_id} generated {tokens} tokens in {elapsed:.1f}s. "
                f"Prompt: {prompt[:80]}{'...' if len(prompt) > 80 else ''}",
                head_id=head_id,
                metrics={"tokens": tokens, "latency_s": round(elapsed, 2)},
            )
            return result
        except CircuitBreakerOpen:
            self._states[head_id] = HeadState.ERROR
            self._metrics.inc("head_generate_errors_total", labels={"head_id": head_id})
            raise
        except Exception as e:
            self._metrics.inc("head_generate_errors_total", labels={"head_id": head_id})
            elapsed = time.perf_counter() - t0
            self._narrate_event(
                "head_error",
                f"Generate FAILED: {head_id} ({elapsed:.1f}s)",
                f"Head {head_id} generation failed after {elapsed:.1f}s: {e}",
                head_id=head_id,
                metrics={"latency_s": round(elapsed, 2), "error": str(e)[:100]},
            )
            raise

    async def generate_stream(self, head_id: str, prompt: str, **kwargs: Any) -> AsyncIterator[str]:
        """Stream tokens from a head with circuit breaker protection.

        Holds a streaming refcount to prevent head unload mid-stream.
        """
        await self.ensure_active(head_id)
        adapter = self._adapters[head_id]
        breaker = self._breakers[head_id]

        if breaker.state == "open":
            self._states[head_id] = HeadState.ERROR
            raise CircuitBreakerOpen(f"Circuit breaker open for head {head_id}")

        self._streaming_refs[head_id] = self._streaming_refs.get(head_id, 0) + 1
        try:
            async for chunk in adapter.generate_stream(prompt, **kwargs):
                yield chunk
            breaker.record_success()
        except Exception:
            breaker.record_failure()
            raise
        finally:
            self._streaming_refs[head_id] = max(0, self._streaming_refs.get(head_id, 1) - 1)

    async def unload_head(self, head_id: str) -> None:
        """Public interface to unload a head (acquires GPU lock)."""
        if head_id not in self._adapters:
            raise KeyError(f"Unknown head: {head_id}")
        async with self._gpu_lock:
            if self._states.get(head_id) in (HeadState.ACTIVE, HeadState.SLEEPING):
                await self._unload(head_id)

    async def sleep_head(self, head_id: str, level: int = 1) -> None:
        """Put a head to sleep (if supported)."""
        if head_id not in self._adapters:
            raise KeyError(f"Unknown head: {head_id}")
        adapter = self._adapters[head_id]
        self._states[head_id] = HeadState.SLEEPING
        await adapter.sleep(level)
        if self._active_head == head_id:
            self._active_head = None

    async def wake_head(self, head_id: str) -> None:
        """Wake a sleeping head."""
        if head_id not in self._adapters:
            raise KeyError(f"Unknown head: {head_id}")
        async with self._gpu_lock:
            if self._active_head and self._active_head != head_id:
                await self._unload(self._active_head)
            adapter = self._adapters[head_id]
            self._states[head_id] = HeadState.LOADING
            await adapter.wake()
            self._states[head_id] = HeadState.ACTIVE
            self._active_head = head_id

    async def shutdown(self, timeout_per_head: float = 30.0) -> None:
        """Unload all heads on shutdown with per-head timeout."""
        for head_id in list(self._adapters.keys()):
            if self._states.get(head_id) in (HeadState.ACTIVE, HeadState.SLEEPING):
                try:
                    await asyncio.wait_for(
                        self._adapters[head_id].unload(),
                        timeout=timeout_per_head,
                    )
                except asyncio.TimeoutError:
                    logger.error("Timeout unloading head %s during shutdown (%.1fs)", head_id, timeout_per_head)
                except Exception as e:
                    logger.warning("Error unloading head %s during shutdown: %s", head_id, e)
                finally:
                    self._states[head_id] = HeadState.OFF
        self._active_head = None
