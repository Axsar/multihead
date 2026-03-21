"""Data models and constants for the init wizard."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class HardwareProfile:
    """Detected hardware profile."""
    gpu_name: str = ""
    gpu_vram_mb: int = 0
    cpu_ram_mb: int = 0
    disk_free_mb: int = 0
    has_cuda: bool = False
    platform: str = ""


@dataclass
class AdapterStatus:
    """Which model adapters are available on this system."""
    ollama_available: bool = False
    ollama_models: list[str] = field(default_factory=list)
    transformers_available: bool = False
    cuda_available: bool = False
    openai_key_set: bool = False
    claude_cli_available: bool = False


SENTINEL_FILENAME = ".multihead_initialized"
