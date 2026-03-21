"""Runtime configuration: mutable settings persisted as JSON.

Separate from Settings (pydantic-settings) which handles static env-based config.
RuntimeConfig is for dynamic, per-session tuning changed via slash commands or MCP.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class GenerationDefaults(BaseModel):
    """Default generation parameters for LLM inference."""

    temperature: float = 0.7
    max_tokens: int = 4096
    top_p: float = 0.9


class ServicesConfig(BaseModel):
    """Background service auto-start settings for the shell."""

    auto_responder: bool = False       # Auto-start auto-responder on shell launch
    worker_daemon: bool = False        # Auto-start Claude worker daemon
    serve: bool = False                # Auto-start API server (via ProcessManager)
    night_shift: bool = False          # Auto-start Night Shift memory refinery
    acp_auto_execute: bool = False     # Auto-execute ACP tasks (False = route to brain)
    responder_interval: int = 30       # Seconds between knowledge.db polls
    responder_strategy: str = "plan-only"  # "plan-only" or "execute"
    worker_mode: str = "sdk"           # "sdk", "headless", or "interactive"
    night_shift_interval: int = 3600   # Seconds between Night Shift runs (default 1hr)
    night_shift_head: str = "qwen-llm"  # Head for Night Shift LLM stages (local GPU, not Claude SDK)
    night_shift_concurrency: int = 1   # Parallel LLM calls per stage
    cloud_marketplace: bool = False    # Auto-start cloud marketplace bridge
    cloud_rfq_interval: int = 600     # Seconds between RFQ scans (10 min)
    cloud_contract_interval: int = 600 # Seconds between contract checks (10 min)
    cloud_auto_quote: bool = True      # Auto-submit quotes on matching RFQs
    cloud_auto_deliver: bool = True    # Auto-deliver on accepted contracts (fulfill immediately)
    cloud_full_pipeline: bool = False   # Route complex tasks through full decompose->DAG->execute
    cloud_pipeline_complexity_threshold: float = 0.7  # Auto-detect threshold (0.0-1.0)
    cloud_pipeline_max_steps: int = 15  # Max decomposition steps per contract
    cloud_pipeline_timeout: float = 180.0  # Pipeline timeout in seconds
    cloud_auto_deliver_capabilities: list[str] = Field(
        default_factory=lambda: [
            "knowledge.rag.query.v1",
            "ai.text.generate.v1",
            "text_generation",
            "ai.task.decompose.v1",
            "ai.reflection.v1",
            "ai.tree_of_thoughts.v1",
            "code.review.v1",
            "image.describe.v1",
            "image.segment.masks.v1",
            "image.detect.objects.v1",
            "vault.data.escrow.v1",
        ]
    )  # Capabilities we can auto-fulfill locally
    cloud_max_contracts: int = 2       # Max parallel contract executions
    session_harvester: bool = False    # Auto-start session harvester
    harvester_interval: int = 300     # Seconds between harvest scans (5 min)
    harvester_max_claims: int = 100   # Max claims per project per harvest


class ConversationConfig(BaseModel):
    """Conversation context persistence settings.

    Controls how conversation history is injected into the system prompt
    to survive Claude SDK context compaction.
    """

    enabled: bool = True              # Master switch for conversation context
    recent_count: int = 6             # Number of recent messages in system prompt
    summary_interval: int = 10        # Rebuild summary every N turns
    max_summary_chars: int = 2000     # Char budget for conversation summary
    max_recent_chars: int = 4000      # Char budget for recent message window


class EventWatcherConfig(BaseModel):
    """Background event watcher settings.

    Controls how the shell monitors for incoming work from
    BotVibes/ACP marketplace and knowledge.db collaboration requests.
    """

    enabled: bool = True              # Master switch for event watching
    poll_interval: int = 15           # Seconds between source checks
    auto_handle: bool = False         # Auto-route events to brain (opt-in)
    watch_acp: bool = True            # Watch BotVibes/ACP for tasks
    watch_knowledge: bool = True      # Watch knowledge.db for collab requests


class PipelineConfig(BaseModel):
    """Dogfooding pipeline settings for the shell.

    Controls which MultiHead infrastructure stages run automatically
    on every message through the ShellPipeline.
    """

    enabled: bool = True              # Master switch for the entire pipeline
    auto_decompose: bool = True       # Route complex tasks through orchestrator
    auto_record: bool = True          # Record key facts as claims in knowledge.db
    knowledge_rag: bool = True        # Inject knowledge context from knowledge.db
    vlm_auto_route: bool = False      # Auto-route images to VLM head
    decompose_threshold: float = 0.7  # Confidence threshold for task classification
    decompose_head: str = ""          # Head for decomposition: "" = auto, "claude-sdk", "core-llm", etc.
    prompt_color: str = "gold1"       # Rich color for the shell prompt
    conversation: ConversationConfig = Field(default_factory=ConversationConfig)
    event_watcher: EventWatcherConfig = Field(default_factory=EventWatcherConfig)


class OperationsConfig(BaseModel):
    """Per-operation model override settings.

    Maps operation names (e.g. 'nightshift.daily_brief') to model IDs
    so the user can redirect specific maintenance operations to different
    heads/models without changing code.
    """

    model_overrides: dict[str, str] = Field(default_factory=dict)


class RuntimeConfig(BaseModel):
    """Mutable runtime settings, persisted as JSON."""

    # Tool enable/disable (list of disabled tool names)
    disabled_tools: list[str] = Field(default_factory=list)

    # Web tools master switch
    web_tools_enabled: bool = True

    # Generation defaults
    generation: GenerationDefaults = Field(default_factory=GenerationDefaults)

    # Strip <think>...</think> tags from model output
    strip_thinking: bool = True

    # VRAM policy
    vram_core_mode: Literal["keep_loaded", "cpu_fallback", "unload_during_batch"] = "keep_loaded"

    # Shell pipeline (dogfooding) settings
    pipeline: PipelineConfig = Field(default_factory=PipelineConfig)

    # Background services auto-start settings
    services: ServicesConfig = Field(default_factory=ServicesConfig)

    # Operations model overrides
    operations: OperationsConfig = Field(default_factory=OperationsConfig)

    def is_tool_enabled(self, tool_name: str) -> bool:
        return tool_name not in self.disabled_tools

    def enable_tool(self, tool_name: str) -> None:
        self.disabled_tools = [t for t in self.disabled_tools if t != tool_name]

    def disable_tool(self, tool_name: str) -> None:
        if tool_name not in self.disabled_tools:
            self.disabled_tools.append(tool_name)

    def set_value(self, key: str, value: str) -> str:
        """Set a config value by dotted key. Returns description of change.

        Supported keys:
          - generation.temperature, generation.max_tokens, generation.top_p
          - web_tools_enabled
          - vram_core_mode
        """
        parts = key.split(".")

        if len(parts) == 2 and parts[0] == "generation":
            field_name = parts[1]
            if field_name not in GenerationDefaults.model_fields:
                raise ValueError(f"Unknown generation field: {field_name}")
            field_info = GenerationDefaults.model_fields[field_name]
            cast_value = field_info.annotation(value)  # type: ignore[operator]
            setattr(self.generation, field_name, cast_value)
            return f"generation.{field_name} = {cast_value}"

        if key == "web_tools_enabled":
            self.web_tools_enabled = value.lower() in ("true", "1", "yes", "on")
            return f"web_tools_enabled = {self.web_tools_enabled}"

        if key == "strip_thinking":
            self.strip_thinking = value.lower() in ("true", "1", "yes", "on")
            return f"strip_thinking = {self.strip_thinking}"

        if key == "vram_core_mode":
            valid = ("keep_loaded", "cpu_fallback", "unload_during_batch")
            if value not in valid:
                raise ValueError(f"vram_core_mode must be one of {valid}")
            self.vram_core_mode = value  # type: ignore[assignment]
            return f"vram_core_mode = {value}"

        # Pipeline conversation config: pipeline.conversation.enabled, etc.
        if len(parts) == 3 and parts[0] == "pipeline" and parts[1] == "conversation":
            field_name = parts[2]
            if field_name not in ConversationConfig.model_fields:
                raise ValueError(f"Unknown conversation field: {field_name}")
            field_info = ConversationConfig.model_fields[field_name]
            if field_info.annotation is bool:
                cast_value = value.lower() in ("true", "1", "yes", "on")
            elif field_info.annotation is int:
                cast_value = int(value)
            else:
                cast_value = value
            setattr(self.pipeline.conversation, field_name, cast_value)
            return f"pipeline.conversation.{field_name} = {cast_value}"

        # Pipeline event_watcher config: pipeline.event_watcher.enabled, etc.
        if len(parts) == 3 and parts[0] == "pipeline" and parts[1] == "event_watcher":
            field_name = parts[2]
            if field_name not in EventWatcherConfig.model_fields:
                raise ValueError(f"Unknown event_watcher field: {field_name}")
            field_info = EventWatcherConfig.model_fields[field_name]
            if field_info.annotation is bool:
                cast_value = value.lower() in ("true", "1", "yes", "on")
            elif field_info.annotation is int:
                cast_value = int(value)
            else:
                cast_value = value
            setattr(self.pipeline.event_watcher, field_name, cast_value)
            return f"pipeline.event_watcher.{field_name} = {cast_value}"

        # Pipeline config: pipeline.enabled, pipeline.auto_decompose, etc.
        if len(parts) == 2 and parts[0] == "pipeline":
            field_name = parts[1]
            if field_name not in PipelineConfig.model_fields:
                raise ValueError(f"Unknown pipeline field: {field_name}")
            field_info = PipelineConfig.model_fields[field_name]
            if field_info.annotation is bool:
                cast_value = value.lower() in ("true", "1", "yes", "on")
            elif field_info.annotation is float:
                cast_value = float(value)
            else:
                cast_value = value
            setattr(self.pipeline, field_name, cast_value)
            return f"pipeline.{field_name} = {cast_value}"

        # Services config: services.auto_responder, services.worker_daemon, etc.
        if len(parts) == 2 and parts[0] == "services":
            field_name = parts[1]
            if field_name not in ServicesConfig.model_fields:
                raise ValueError(f"Unknown services field: {field_name}")
            field_info = ServicesConfig.model_fields[field_name]
            if field_info.annotation is bool:
                cast_value = value.lower() in ("true", "1", "yes", "on")
            elif field_info.annotation is int:
                cast_value = int(value)
            else:
                cast_value = value
            setattr(self.services, field_name, cast_value)
            return f"services.{field_name} = {cast_value}"

        # Operations model overrides: operations.<operation_name> = <model_id>
        # Operation names contain dots (e.g. 'nightshift.daily_brief'),
        # so join everything after the first part.
        if len(parts) >= 2 and parts[0] == "operations":
            op_name = ".".join(parts[1:])
            if value.strip():
                self.operations.model_overrides[op_name] = value.strip()
            else:
                # Empty value clears the override
                self.operations.model_overrides.pop(op_name, None)
            configured = self.operations.model_overrides.get(op_name, "(default)")
            return f"operations.{op_name} = {configured}"

        raise ValueError(f"Unknown config key: {key}")

    @classmethod
    def load(cls, path: Path) -> RuntimeConfig:
        """Load config from JSON file, or return defaults if missing."""
        if path.exists():
            try:
                return cls.model_validate_json(path.read_text())
            except Exception as e:
                logger.warning("Failed to load runtime config from %s: %s", path, e)
        return cls()

    def save(self, path: Path) -> None:
        """Persist config to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(self.model_dump_json(indent=2))
