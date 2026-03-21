"""Configuration generation: heads.yaml, solvers.yaml, .env files."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .capabilities import (
    DEFAULT_LLM_CAPABILITIES,
    DEFAULT_VLM_CAPABILITIES,
    infer_capabilities,
)
from .models import HardwareProfile

logger = logging.getLogger(__name__)


def suggest_config(profile: HardwareProfile) -> dict[str, Any]:
    """Suggest a configuration based on hardware profile."""
    if profile.gpu_vram_mb >= 20000:
        template = "rtx4090"
        core_model = "qwen3:32b"
        worker_model = "qwen3:8b"
    elif profile.gpu_vram_mb >= 10000:
        template = "rtx3060"
        core_model = "qwen3:8b"
        worker_model = "qwen3:4b"
    elif profile.has_cuda:
        template = "rtx3060"
        core_model = "qwen3:4b"
        worker_model = "qwen3:4b"
    elif profile.platform == "Darwin" and profile.cpu_ram_mb >= 16000:
        template = "apple_silicon"
        core_model = "qwen3:8b"
        worker_model = "qwen3:4b"
    else:
        template = "cpu_only"
        core_model = "qwen3:4b"
        worker_model = "qwen3:4b"

    return {
        "template": template,
        "core_model": core_model,
        "worker_model": worker_model,
        "gpu_name": profile.gpu_name,
        "gpu_vram_mb": profile.gpu_vram_mb,
        "cpu_ram_mb": profile.cpu_ram_mb,
    }


def generate_heads_yaml(
    config: dict[str, Any], templates_dir: Path | None = None,
) -> str:
    """Generate heads.yaml from template file or dynamic fallback."""
    # Prefer template file if available
    if templates_dir:
        template_file = templates_dir / f"{config['template']}.yaml"
        if template_file.exists():
            return template_file.read_text(encoding="utf-8")

    # Fallback: generate dynamically
    core_gpu = config["template"] not in ("cpu_only",)
    head_list: list[dict[str, Any]] = [
        # Mock heads always included
        {
            "head_id": "mock-llm",
            "name": "Mock LLM",
            "adapter": "mock",
            "model": "mock-llm-v1",
            "kind": "llm",
            "gpu_required": False,
        },
        {
            "head_id": "core-llm",
            "name": "Core LLM",
            "adapter": "ollama",
            "model": config["core_model"],
            "kind": "llm",
            "gpu_required": core_gpu,
        },
    ]
    heads = {"heads": head_list}
    return yaml.dump(heads, default_flow_style=False, sort_keys=False)


def generate_solvers_yaml(heads_content: str) -> str:
    """Generate solvers.yaml with inferred capabilities from heads.yaml content.

    Args:
        heads_content: Raw YAML string of the heads.yaml file.

    Returns:
        YAML string for solvers.yaml.
    """
    try:
        data = yaml.safe_load(heads_content) or {}
    except yaml.YAMLError:
        logger.warning("Could not parse heads.yaml for solvers generation")
        return ""

    heads_list = data.get("heads", [])
    if not heads_list:
        return ""

    solvers: list[dict[str, Any]] = []
    for head in heads_list:
        head_id = head.get("head_id", "")
        if not head_id:
            continue

        caps, privacy = infer_capabilities(head)

        solver: dict[str, Any] = {
            "solver_id": head_id,
            "name": head.get("name", head_id),
            "adapter": head.get("adapter", "mock"),
            "model": head.get("model", ""),
            "kind": head.get("kind", "llm"),
            "gpu_required": head.get("gpu_required", False),
            "is_local": head.get("adapter", "") in ("mock", "transformers", "ollama", "claude", "claude_agent_sdk"),
            "privacy_level": privacy,
        }

        # Optional fields
        if head.get("vram_hint_mb"):
            solver["vram_hint_mb"] = head["vram_hint_mb"]
        if head.get("quantization"):
            solver["quantization"] = head["quantization"]
        if head.get("endpoint"):
            solver["endpoint"] = head["endpoint"]
        if head.get("extra"):
            solver["extra"] = head["extra"]

        solver["capabilities"] = caps
        solvers.append(solver)

    # Add mock heads for testing (if not already present)
    existing_ids = {s["solver_id"] for s in solvers}
    if "mock-llm" not in existing_ids:
        solvers.append({
            "solver_id": "mock-llm",
            "name": "Mock LLM",
            "adapter": "mock",
            "model": "mock-llm-model",
            "kind": "llm",
            "gpu_required": False,
            "is_local": True,
            "privacy_level": "local",
            "capabilities": dict(DEFAULT_LLM_CAPABILITIES),
        })

    return yaml.dump(
        {"solvers": solvers},
        default_flow_style=False,
        sort_keys=False,
    )


def generate_env_file(profile: HardwareProfile, env_path: Path) -> bool:
    """Generate .env with auto-detected defaults. Returns True if written.

    Never overwrites an existing .env file.
    """
    if env_path.exists():
        logger.info(".env already exists at %s, skipping", env_path)
        return False

    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        f"# MultiHead config (auto-generated {now})",
        f"# Platform: {profile.platform}, GPU: {profile.gpu_name or 'none'}",
        "",
        "# Data directory (runs, artifacts, knowledge DB)",
        "MULTIHEAD_DATA_DIR=~/.multihead",
        "",
        "# Core head for chat (auto-detected from heads.yaml at startup)",
        "# MULTIHEAD_CORE_HEAD_ID=core-llm",
        "",
    ]
    env_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return True
