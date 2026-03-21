"""Night Shift: 20-stage offline memory refinery pipeline (core orchestrator)."""

from __future__ import annotations

import json
import logging
import signal
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from multihead.artifact_store import ArtifactStore
from multihead.extractors.base import BatchPending as _BatchPending
from multihead.chunker import Chunker
from multihead.context_packs import PackBuilder
from multihead.extractors.claim_extractor import ClaimExtractor
from multihead.extractors.consistency_checker import ConsistencyChecker
from multihead.extractors.entity_extractor import EntityExtractor
from multihead.extractors.event_extractor import EventExtractor
from multihead.extractors.topic_assigner import TopicAssigner
from multihead.head_manager import HeadManager
from multihead.knowledge_models import NightShiftConfig, NightShiftReport
from multihead.knowledge_store import KnowledgeStore
from multihead.record_store import RecordStore

from .models import STAGES, StageDefinition, StageGate
from .stages_early import EarlyStagesMixin
from .stages_late import LateStagesMixin
from .stages_advanced import AdvancedStagesMixin

logger = logging.getLogger(__name__)


class NightShift(EarlyStagesMixin, LateStagesMixin, AdvancedStagesMixin):
    """The 19-stage memory refinery pipeline."""

    def __init__(
        self,
        knowledge_store: KnowledgeStore,
        record_store: RecordStore,
        artifact_store: ArtifactStore,
        pack_builder: PackBuilder,
        head_manager: HeadManager,
        config: NightShiftConfig,
        output_dir: Path,
        settings=None,
    ) -> None:
        self.knowledge = knowledge_store
        self.records = record_store
        self.artifacts = artifact_store
        self.packs = pack_builder
        self.heads = head_manager
        self.config = config
        self.output_dir = output_dir
        self.settings = settings
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.chunker = Chunker()
        self.entity_extractor = EntityExtractor()
        self.topic_assigner = TopicAssigner()
        self.event_extractor = EventExtractor()
        self.claim_extractor = ClaimExtractor(
            auto_accept_confidence=config.auto_accept_confidence,
            auto_accept_min_supports=config.auto_accept_min_supports,
        )
        self.consistency_checker = ConsistencyChecker()

        # Narrative pipeline (evidence fusion from git/chat/agent sources)
        from multihead.narrative.pipeline import NarrativePipeline
        self.narrative_pipeline = NarrativePipeline(knowledge_store, project_id="multihead", artifact_store=self.artifacts)

        self._on_progress: Callable[[dict[str, Any]], None] | None = None
        self._current_stage: str | None = None
        self._prev_sigint = None
        self._prev_sigterm = None

    @property
    def on_progress(self) -> Callable[[dict[str, Any]], None] | None:
        return self._on_progress

    @on_progress.setter
    def on_progress(self, cb: Callable[[dict[str, Any]], None] | None) -> None:
        self._on_progress = cb

    def _emit(self, event: dict[str, Any]) -> None:
        """Fire progress callback if set."""
        if self._on_progress:
            try:
                self._on_progress(event)
            except Exception:
                pass  # Never let progress reporting break the pipeline

    # ------------------------------------------------------------------
    # Graceful shutdown (mid-stage checkpoint support)
    # ------------------------------------------------------------------

    def _install_signal_handlers(self) -> None:
        """Install SIGINT/SIGTERM handlers that set the shutdown flag."""
        import multihead.extractors.base as _base

        _base.shutdown_requested = False

        def _handler(signum, frame):
            logger.info("Received signal %s — requesting graceful shutdown", signum)
            _base.shutdown_requested = True

        try:
            self._prev_sigint = signal.signal(signal.SIGINT, _handler)
            self._prev_sigterm = signal.signal(signal.SIGTERM, _handler)
        except (OSError, ValueError):
            pass  # Not in main thread — skip signal handling

    def _restore_signal_handlers(self) -> None:
        """Restore previous signal handlers."""
        import multihead.extractors.base as _base

        _base.shutdown_requested = False
        try:
            if self._prev_sigint is not None:
                signal.signal(signal.SIGINT, self._prev_sigint)
            if self._prev_sigterm is not None:
                signal.signal(signal.SIGTERM, self._prev_sigterm)
        except (OSError, ValueError):
            pass

    # ------------------------------------------------------------------
    # State checkpoint save/load (for --from-stage resume)
    # ------------------------------------------------------------------

    def _state_path(self) -> Path:
        return self.output_dir / "nightshift_state.json"

    def _save_state(self, context: dict) -> None:
        """Persist serialisable context keys to disk after each stage."""
        from dataclasses import asdict, is_dataclass

        def _serial(obj: Any) -> Any:
            if is_dataclass(obj) and not isinstance(obj, type):
                return asdict(obj)
            if isinstance(obj, list):
                return [_serial(v) for v in obj]
            if isinstance(obj, dict):
                return {k: _serial(v) for k, v in obj.items()}
            if hasattr(obj, "model_dump"):
                return obj.model_dump(mode="json")
            if isinstance(obj, datetime):
                return obj.isoformat()
            return obj

        # Keys we know how to restore
        saveable_keys = {
            "record_count", "since", "chunks", "hot_signals",
            "entities", "topics", "extracted_events", "extracted_claims",
            "events_created", "claims_created", "_completed_stages",
            "_all_metrics", "max_chunks",
        }
        state = {k: _serial(v) for k, v in context.items() if k in saveable_keys}
        self._state_path().write_text(
            json.dumps(state, indent=2, default=str), encoding="utf-8"
        )

    def _load_state(self) -> dict[str, Any]:
        """Load previously saved context from disk."""
        from multihead.chunker import Chunk

        path = self._state_path()
        if not path.exists():
            raise FileNotFoundError(f"No saved state at {path}. Run from stage 0 first.")
        raw = json.loads(path.read_text(encoding="utf-8"))

        # Reconstruct Chunk objects
        if "chunks" in raw and isinstance(raw["chunks"], list):
            restored = []
            for c in raw["chunks"]:
                if isinstance(c, dict):
                    restored.append(Chunk(**{k: v for k, v in c.items()}))
                else:
                    restored.append(c)
            raw["chunks"] = restored

        # Reconstruct datetime for "since"
        if "since" in raw and isinstance(raw["since"], str):
            raw["since"] = datetime.fromisoformat(raw["since"])

        if "_all_metrics" not in raw:
            raw["_all_metrics"] = {}
        return raw

    async def run(self, *, max_chunks: int | None = None, from_stage: int = 0, to_stage: int | None = None) -> NightShiftReport:
        """Execute the Night Shift pipeline, optionally resuming from a saved stage.

        Supports mid-stage checkpointing: if interrupted (SIGINT/SIGTERM),
        long-running stages save their progress. On resume with --from-stage,
        the stage picks up from the last checkpoint automatically.

        Args:
            max_chunks: Cap on chunk count (for faster test runs).
            from_stage: Stage index to start from (0 = full run). Requires a
                saved state file from a previous partial run.
            to_stage: Stage index to stop after (inclusive). None = run all.
        """
        if from_stage > 0:
            logger.info("Resuming from stage %d — loading saved state", from_stage)
            context = self._load_state()
            if max_chunks:
                context["max_chunks"] = max_chunks
        else:
            context = {"_all_metrics": {}}
            if max_chunks:
                context["max_chunks"] = max_chunks

        report = NightShiftReport()
        total = len(STAGES)
        self._install_signal_handlers()
        pending_stages: set[int] = set()  # stages with pending batches

        for i, stage in enumerate(STAGES):
            if i < from_stage:
                report.stages_skipped.append(stage.name)
                continue
            if to_stage is not None and i > to_stage:
                report.stages_skipped.append(stage.name)
                continue

            # Skip if any dependency is pending (batch not ready yet)
            blocked_by = pending_stages & set(stage.depends_on)
            if blocked_by:
                blocked_names = [STAGES[s].name for s in blocked_by if s < len(STAGES)]
                logger.info(
                    "Stage %s skipped — blocked by pending: %s",
                    stage.name, ", ".join(blocked_names),
                )
                report.stages_skipped.append(stage.name)
                self._emit({
                    "event": "stage_skip", "stage": stage.name,
                    "elapsed_s": 0.0, "reason": f"blocked by pending: {blocked_names}",
                })
                continue

            stage_name = stage.name
            self._current_stage = stage_name
            self._emit({"event": "stage_start", "stage": stage_name, "index": i, "total": total})
            t0 = time.monotonic()
            try:
                result = await self._run_stage(stage, context)
                elapsed = time.monotonic() - t0
                if result is None:
                    report.stages_skipped.append(stage_name)
                    self._emit({"event": "stage_skip", "stage": stage_name, "elapsed_s": round(elapsed, 1)})
                else:
                    # Accumulate metrics before context.update overwrites them
                    if "metrics" in result:
                        context["_all_metrics"][stage_name] = result["metrics"]
                    context.update(result)
                    context["_completed_stages"] = report.stages_completed + [stage_name]
                    report.stages_completed.append(stage_name)
                    summary = self._stage_summary(stage_name, result)
                    self._save_state(context)
                    self._emit({
                        "event": "stage_done", "stage": stage_name,
                        "elapsed_s": round(elapsed, 1), "summary": summary,
                        "metrics": result.get("metrics", {}),
                    })
            except InterruptedError as e:
                elapsed = time.monotonic() - t0
                logger.info("Stage %s interrupted at %s — checkpoint saved", stage_name, e)
                self._save_state(context)
                report.stages_failed.append(stage_name)
                self._emit({
                    "event": "stage_interrupted", "stage": stage_name,
                    "elapsed_s": round(elapsed, 1), "error": str(e)[:200],
                })
                break  # Exit the loop — resume with --from-stage
            except _BatchPending as e:
                elapsed = time.monotonic() - t0
                pending_stages.add(i)
                logger.info(
                    "Stage %s: batch %s pending — continuing with independent stages",
                    stage_name, e.batch_id,
                )
                self._save_state(context)
                self._emit({
                    "event": "stage_pending", "stage": stage_name,
                    "elapsed_s": round(elapsed, 1), "batch_id": e.batch_id,
                })
                continue  # Don't break — run independent stages
            except Exception as e:
                elapsed = time.monotonic() - t0
                logger.error(f"Stage {stage_name} failed: {e}")
                report.stages_failed.append(stage_name)
                self._emit({
                    "event": "stage_fail", "stage": stage_name,
                    "elapsed_s": round(elapsed, 1), "error": str(e)[:200],
                })
                if stage.gate.on_fail == "abort":
                    break

        self._restore_signal_handlers()
        self._current_stage = None
        report.ended_at = datetime.now(timezone.utc)
        report.records_processed = context.get("record_count", 0)
        report.events_created = context.get("events_created", 0)
        report.claims_created = context.get("claims_created", 0)
        report.packs_built = context.get("packs_built", [])

        self._emit({"event": "complete", "records": report.records_processed,
                     "events": report.events_created, "claims": report.claims_created,
                     "packs": len(report.packs_built)})

        # Update last successful run timestamp if we processed records
        if report.records_processed > 0 and not report.stages_failed:
            # Use the "since" timestamp from select_input_window as the new last run time
            since_time = context.get("since")
            if since_time:
                # The next run should start from NOW (when this run ended)
                self._update_last_successful_run_time(report.ended_at)

        # Save report
        report_path = self.output_dir / f"{report.report_id}.json"
        report_path.write_text(
            json.dumps(report.model_dump(mode="json"), indent=2, default=str),
            encoding="utf-8",
        )

        return report

    @staticmethod
    def _stage_summary(name: str, result: dict) -> str:
        """Extract a human-readable summary from stage result."""
        parts: list[str] = []

        # Check known result keys
        for key in ("record_count", "events_created", "claims_created"):
            if key in result:
                parts.append(f"{key}={result[key]}")
        if "chunks" in result:
            parts.append(f"{len(result['chunks'])} chunks")
        if "entities" in result:
            parts.append(f"{len(result['entities'])} entities")
        if "topics" in result:
            parts.append(f"{len(result['topics'])} topics")
        if "extracted_events" in result:
            parts.append(f"{len(result['extracted_events'])} events extracted")
        if "extracted_claims" in result:
            parts.append(f"{len(result['extracted_claims'])} claims extracted")
        if "contradictions" in result:
            v = result["contradictions"]
            parts.append(f"{v if isinstance(v, int) else len(v)} contradictions")
        if "open_loops" in result:
            v = result["open_loops"]
            parts.append(f"{v if isinstance(v, int) else len(v)} open loops")
        if "packs_built" in result:
            parts.append(f"{len(result['packs_built'])} packs")
        if "links_created" in result:
            parts.append(f"{result['links_created']} links")

        # Always include metrics if present — catches everything stages report
        metrics = result.get("metrics", {})
        if metrics and not parts:
            for k, v in metrics.items():
                parts.append(f"{k}={v}")
        elif metrics:
            # Add any metric not already covered
            covered = set()
            for p in parts:
                for k in metrics:
                    if k in p:
                        covered.add(k)
            for k, v in metrics.items():
                if k not in covered:
                    parts.append(f"{k}={v}")

        return ", ".join(parts) if parts else "no output"

    async def _run_stage(self, stage: StageDefinition, context: dict) -> dict | None:
        """Execute one stage with gating logic."""
        handler = getattr(self, f"_stage_{stage.name}", None)
        if handler is None:
            logger.warning(f"No handler for stage {stage.name}")
            return None

        max_attempts = stage.gate.retry.get("max_attempts", 1)
        last_metrics: dict[str, float] = {}

        for attempt in range(max_attempts):
            result = await handler(context)
            if result is None:
                return None

            last_metrics = result.get("metrics", {})
            gate_result = self._evaluate_gate(stage.gate, last_metrics)

            if gate_result == "accept":
                return result
            elif gate_result == "retry" and attempt < max_attempts - 1:
                logger.info(f"Stage {stage.name}: retrying (attempt {attempt + 2})")
                self._emit({"event": "gate", "stage": stage.name, "decision": "retry",
                             "attempt": attempt + 2, "metrics": last_metrics})
                continue
            elif gate_result == "fallback" and stage.gate.fallback:
                result["fallback_used"] = True
                self._emit({"event": "gate", "stage": stage.name, "decision": "fallback",
                             "metrics": last_metrics})
                return result
            else:
                if stage.gate.on_fail == "continue":
                    return result
                elif stage.gate.on_fail == "skip":
                    return None
                elif stage.gate.on_fail == "abort":
                    raise RuntimeError(f"Stage {stage.name} failed gate: {last_metrics}")

        return result

    @staticmethod
    def _evaluate_gate(gate: StageGate, metrics: dict[str, float]) -> str:
        """Returns 'accept' | 'retry' | 'fallback' | 'fail'."""
        if not gate.accept_if:
            return "accept"

        for condition in gate.accept_if:
            metric = condition.get("metric", "")
            op = condition.get("op", ">=")
            threshold = condition.get("value", 0)
            actual = metrics.get(metric, 0)

            passed = False
            if op == ">=":
                passed = actual >= threshold
            elif op == "<=":
                passed = actual <= threshold
            elif op == ">":
                passed = actual > threshold
            elif op == "<":
                passed = actual < threshold
            elif op == "==":
                passed = actual == threshold

            if not passed:
                if gate.retry.get("max_attempts", 1) > 1:
                    return "retry"
                if gate.fallback:
                    return "fallback"
                return "fail"

        return "accept"

    # -------------------------------------------------------------------
    # Backlog processing: sweep all existing claims through analysis stages
    # -------------------------------------------------------------------

    _BACKLOG_EXCLUDE_PREFIXES = [
        "nightshift.last_run.",
        "nightshift.backlog_cursor.",
        "session_harvest.",
    ]

    def _get_backlog_cursor(self) -> int:
        """Get the current backlog processing offset from knowledge store."""
        try:
            claims = self.knowledge.list_claims(
                status="accepted", scope_id="multihead", limit=100,
            )
            matching = [
                c for c in claims
                if c.canonical.claim_key.startswith("nightshift.backlog_cursor.")
            ]
            if matching:
                latest = max(matching, key=lambda c: c.provenance.created_at)
                return int(latest.canonical.object.value)
        except Exception as e:
            logger.warning(f"Failed to get backlog cursor: {e}")
        return 0

    def _set_backlog_cursor(self, offset: int) -> None:
        """Persist the backlog cursor offset."""
        from multihead.knowledge_models import (
            Claim, ClaimCanonical, ClaimScope, ClaimStatus, ClaimType,
            EntityRef, ScopeType, ValueObject,
        )
        from .models import _prov

        try:
            # Use timestamped key to avoid UNIQUE constraint on insert
            import time
            claim_key = f"nightshift.backlog_cursor.{int(time.time())}"

            claim = Claim(
                claim_type=ClaimType.FACT,
                scope=ClaimScope(scope_type=ScopeType.PROJECT, scope_id="multihead"),
                canonical=ClaimCanonical(
                    claim_key=claim_key,
                    subject=EntityRef(
                        entity_type="pipeline", entity_id="nightshift",
                        entity_label="Night Shift",
                    ),
                    predicate="backlog_offset",
                    object=ValueObject(value_type="number", value=str(offset)),
                ),
                statement=f"Backlog sweep cursor at offset {offset}",
                provenance=_prov(),
            )
            claim.claim_status = ClaimStatus.ACCEPTED
            self.knowledge.insert_claim(claim)
        except Exception as e:
            logger.warning(f"Failed to set backlog cursor: {e}")

    async def run_backlog(self, batch_size: int = 500, reset: bool = False) -> dict:
        """Process all existing claims through consistency/links/open-loops stages.

        Cursor-based: remembers where it left off so it can resume across runs.
        """
        offset = 0 if reset else self._get_backlog_cursor()
        total = self.knowledge.count_claims(status="accepted")
        logger.info("Backlog sweep: %d total accepted claims, starting at offset %d", total, offset)

        contradictions_found = 0
        links_found = 0
        open_loops_found = 0
        batches_processed = 0

        while offset < total:
            claims = self.knowledge.list_claims_paginated(
                offset=offset, limit=batch_size, status="accepted",
                exclude_key_prefixes=self._BACKLOG_EXCLUDE_PREFIXES,
            )
            if not claims:
                break

            self._emit({
                "event": "backlog_batch", "offset": offset,
                "batch_size": len(claims), "total": total,
            })

            # Run analysis stages on this batch
            result = await self._run_backlog_stage("consistency_check", claims)
            contradictions_found += result.get("contradictions", 0)

            result = await self._run_backlog_stage("cross_topic_links", claims)
            links_found += result.get("links", 0)

            result = await self._run_backlog_stage("open_loops", claims)
            open_loops_found += result.get("open_loops", 0)

            offset += len(claims)
            batches_processed += 1
            self._set_backlog_cursor(offset)

            logger.info(
                "Backlog batch %d done: offset=%d/%d, "
                "contradictions=%d, links=%d, open_loops=%d",
                batches_processed, offset, total,
                contradictions_found, links_found, open_loops_found,
            )

        summary = {
            "total_claims": total,
            "claims_processed": offset,
            "batches": batches_processed,
            "contradictions_found": contradictions_found,
            "links_found": links_found,
            "open_loops_found": open_loops_found,
        }

        self._emit({"event": "backlog_complete", **summary})
        return summary

    async def _run_backlog_stage(self, stage_name: str, claims: list) -> dict:
        """Run a single analysis stage on a batch of claims."""
        if stage_name == "consistency_check":
            return await self._backlog_consistency_check(claims)
        elif stage_name == "cross_topic_links":
            return await self._backlog_cross_topic_links(claims)
        elif stage_name == "open_loops":
            return await self._backlog_open_loops(claims)
        return {}

    async def _backlog_consistency_check(self, claims: list) -> dict:
        """Check a batch of claims for contradictions."""
        statements = [c.statement for c in claims]
        result = await self.consistency_checker.extract(
            statements, self._generate_fn(), concurrency=self.config.concurrency,
        )
        contradictions = result.items if hasattr(result, "items") else []

        for contradiction in contradictions:
            try:
                claim_ids = contradiction.get("claim_ids", [])
                reason = contradiction.get("reason", "")
                if len(claim_ids) >= 2:
                    self.knowledge.add_claim_conflict(
                        claim_ids[0], claim_ids[1], reason,
                    )
            except Exception as e:
                logger.warning(f"Failed to record contradiction: {e}")

        return {"contradictions": len(contradictions)}

    async def _backlog_cross_topic_links(self, claims: list) -> dict:
        """Find cross-topic links in a batch of claims."""
        # Group claims by scope_id (topic)
        by_topic: dict[str, list] = {}
        for c in claims:
            topic = c.scope.scope_id
            by_topic.setdefault(topic, []).append(c)

        if len(by_topic) < 2:
            return {"links": 0}

        links_created = 0
        topics = list(by_topic.keys())
        for i, t1 in enumerate(topics):
            for t2 in topics[i + 1:]:
                # Simple keyword overlap heuristic
                words1 = set()
                for c in by_topic[t1]:
                    words1.update(c.statement.lower().split())
                words2 = set()
                for c in by_topic[t2]:
                    words2.update(c.statement.lower().split())

                overlap = words1 & words2
                # Filter stopwords (very basic)
                overlap -= {"the", "a", "an", "is", "are", "was", "were", "be",
                            "been", "being", "have", "has", "had", "do", "does",
                            "did", "will", "would", "could", "should", "may",
                            "might", "shall", "can", "to", "of", "in", "for",
                            "on", "with", "at", "by", "from", "as", "into",
                            "through", "during", "before", "after", "and", "but",
                            "or", "not", "no", "this", "that", "it", "its"}

                if len(overlap) >= 5:
                    # Link representative claims from each topic
                    c1 = by_topic[t1][0]
                    c2 = by_topic[t2][0]
                    try:
                        self.knowledge.add_claim_conflict(
                            c1.claim_id, c2.claim_id,
                            f"cross-topic-link: {t1} <-> {t2}",
                        )
                        links_created += 1
                    except Exception:
                        pass

        return {"links": links_created}

    async def _backlog_open_loops(self, claims: list) -> dict:
        """Detect open loops (unresolved questions/promises) in claims."""
        open_loop_markers = [
            "todo", "fixme", "hack", "needs", "should", "planned",
            "will be", "not yet", "incomplete", "pending", "tbd",
            "open question", "unresolved", "blocked",
        ]
        open_loops = []
        for c in claims:
            text = c.statement.lower()
            if any(marker in text for marker in open_loop_markers):
                open_loops.append({
                    "claim_id": c.claim_id,
                    "statement": c.statement[:200],
                    "marker": next(m for m in open_loop_markers if m in text),
                })

        return {"open_loops": len(open_loops)}

    def _generate_fn(self, max_tokens: int = 2048, temperature: float = 0.3):
        """Return a generate callable that routes through HeadManager.

        Automatically uses reasoning_head_id for reasoning-heavy stages when set,
        falling back to head_id for extraction stages.
        Disables thinking mode (Qwen3 /no_think) for structured JSON extraction.
        """
        stage = self._current_stage
        if (
            self.config.reasoning_head_id
            and stage in self.config.reasoning_stages
        ):
            head_id = self.config.reasoning_head_id
        else:
            head_id = self.config.head_id

        async def _gen(prompt: str) -> dict:
            t0 = time.monotonic()
            result = await self.heads.generate(
                head_id, prompt,
                max_tokens=max_tokens, temperature=temperature,
                thinking=False,
            )
            elapsed = time.monotonic() - t0
            tokens = result.get("tokens_out") or result.get("tokens_generated") or 0
            self._emit({
                "event": "llm_call", "stage": self._current_stage,
                "prompt_chars": len(prompt), "tokens": tokens,
                "elapsed_s": round(elapsed, 1),
            })
            return result

        return _gen

    async def _get_extraction_adapter(self):
        """Return the raw HeadAdapter for the current extraction head.

        Used by batch mode — generate_batch() needs the adapter directly,
        not a wrapped closure. Ensures the head is loaded first.
        """
        stage = self._current_stage
        if (
            self.config.reasoning_head_id
            and stage in self.config.reasoning_stages
        ):
            head_id = self.config.reasoning_head_id
        else:
            head_id = self.config.head_id
        await self.heads.ensure_active(head_id)
        return self.heads.get_adapter(head_id)
