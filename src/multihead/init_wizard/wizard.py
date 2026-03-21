"""InitWizard: orchestrates hardware detection, adapter checks, and config generation."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .capabilities import (
    DEFAULT_LLM_CAPABILITIES,
    DEFAULT_VLM_CAPABILITIES,
    MODEL_CAPABILITIES,
    infer_capabilities,
)
from .config_gen import (
    generate_env_file,
    generate_heads_yaml,
    generate_solvers_yaml,
    suggest_config,
)
from .hardware import check_adapters, detect_hardware
from .models import SENTINEL_FILENAME


class InitWizard:
    """Hardware detection, adapter checking, and config generation."""

    # Expose capability data as class attributes for backward compatibility
    _MODEL_CAPABILITIES = MODEL_CAPABILITIES
    _DEFAULT_LLM_CAPABILITIES = DEFAULT_LLM_CAPABILITIES
    _DEFAULT_VLM_CAPABILITIES = DEFAULT_VLM_CAPABILITIES

    def is_first_run(self, config_dir: Path) -> bool:
        """Return True if this appears to be a first-time setup.

        Detection order (first match wins):
        1. Sentinel file ``<config_dir>/.multihead_initialized`` is absent -> first run
        2. ``<config_dir>/heads.yaml`` is absent -> first run
        3. Otherwise -> returning user
        """
        sentinel = config_dir / SENTINEL_FILENAME
        if not sentinel.exists():
            return True
        if not (config_dir / "heads.yaml").exists():
            return True
        return False

    def mark_initialized(self, config_dir: Path) -> None:
        """Write the sentinel file to record a completed initialization."""
        config_dir.mkdir(parents=True, exist_ok=True)
        sentinel = config_dir / SENTINEL_FILENAME
        now = datetime.now(timezone.utc).isoformat()
        sentinel.write_text(f"initialized: {now}\n", encoding="utf-8")

    def detect_hardware(self):
        """Detect local hardware capabilities."""
        return detect_hardware()

    def check_adapters(self):
        """Check which model adapters are available on this system."""
        return check_adapters()

    def suggest_config(self, profile):
        """Suggest a configuration based on hardware profile."""
        return suggest_config(profile)

    def generate_heads_yaml(self, config, templates_dir=None):
        """Generate heads.yaml from template file or dynamic fallback."""
        return generate_heads_yaml(config, templates_dir)

    def _infer_capabilities(self, head):
        """Infer capability metadata for a head based on model name and adapter."""
        return infer_capabilities(head)

    def generate_solvers_yaml(self, heads_content):
        """Generate solvers.yaml with inferred capabilities from heads.yaml content."""
        return generate_solvers_yaml(heads_content)

    def generate_env_file(self, profile, env_path):
        """Generate .env with auto-detected defaults. Returns True if written."""
        return generate_env_file(profile, env_path)

    def run_interactive(self, config_dir: Path) -> dict[str, Any]:
        """Run the init wizard and write config files."""
        profile = self.detect_hardware()
        suggestion = self.suggest_config(profile)

        # Write heads.yaml
        config_dir.mkdir(parents=True, exist_ok=True)
        (config_dir / "recipes").mkdir(exist_ok=True)
        heads_content = self.generate_heads_yaml(suggestion)
        (config_dir / "heads.yaml").write_text(heads_content, encoding="utf-8")

        # Generate and write solvers.yaml with inferred capabilities
        solvers_content = self.generate_solvers_yaml(heads_content)
        solvers_generated = False
        if solvers_content:
            (config_dir / "solvers.yaml").write_text(solvers_content, encoding="utf-8")
            solvers_generated = True

        first_run = self.is_first_run(config_dir)
        self.mark_initialized(config_dir)

        return {
            "profile": {
                "gpu_name": profile.gpu_name,
                "gpu_vram_mb": profile.gpu_vram_mb,
                "cpu_ram_mb": profile.cpu_ram_mb,
                "has_cuda": profile.has_cuda,
                "platform": profile.platform,
            },
            "suggestion": suggestion,
            "config_dir": str(config_dir),
            "first_run": first_run,
            "solvers_generated": solvers_generated,
        }

    def run_auto(
        self,
        config_dir: Path,
        env_path: Path | None = None,
        templates_dir: Path | None = None,
    ) -> dict[str, Any]:
        """Full automatic setup: detect, check adapters, generate all config.

        Args:
            config_dir: Where to write heads.yaml and recipes/.
            env_path: Where to write .env (default: cwd/.env). Never overwrites.
            templates_dir: Where to find template YAML files. Default: config_dir/templates.
        """
        if env_path is None:
            env_path = Path(".env")
        if templates_dir is None:
            templates_dir = config_dir / "templates"
        # Fall back to bundled templates if config_dir/templates doesn't exist
        if not templates_dir.exists():
            from ..config import _bundled_config_path
            bundled_templates = _bundled_config_path() / "templates"
            if bundled_templates.exists():
                templates_dir = bundled_templates

        profile = self.detect_hardware()
        suggestion = self.suggest_config(profile)
        adapters = self.check_adapters()

        # Write heads.yaml (prefer template)
        config_dir.mkdir(parents=True, exist_ok=True)
        (config_dir / "recipes").mkdir(exist_ok=True)
        heads_content = self.generate_heads_yaml(suggestion, templates_dir)

        # Auto-enable Claude head if CLI is available
        if adapters.claude_cli_available and "# - head_id: claude-sonnet" in heads_content:
            heads_content = heads_content.replace(
                "# - head_id: claude-sonnet\n"
                "  #   name: Claude Sonnet\n"
                "  #   adapter: claude\n"
                "  #   model: sonnet\n"
                "  #   kind: llm\n"
                "  #   gpu_required: false",
                "- head_id: claude-sonnet\n"
                "    name: Claude Sonnet\n"
                "    adapter: claude\n"
                "    model: claude-sonnet-4-6\n"
                "    kind: llm\n"
                "    gpu_required: false",
            )

        (config_dir / "heads.yaml").write_text(heads_content, encoding="utf-8")

        # Generate and write solvers.yaml with inferred capabilities
        solvers_content = self.generate_solvers_yaml(heads_content)
        solvers_generated = False
        if solvers_content:
            (config_dir / "solvers.yaml").write_text(solvers_content, encoding="utf-8")
            solvers_generated = True

        # Write .env (never clobber)
        env_generated = self.generate_env_file(profile, env_path)
        first_run = self.is_first_run(config_dir)
        self.mark_initialized(config_dir)

        return {
            "profile": {
                "gpu_name": profile.gpu_name,
                "gpu_vram_mb": profile.gpu_vram_mb,
                "cpu_ram_mb": profile.cpu_ram_mb,
                "has_cuda": profile.has_cuda,
                "platform": profile.platform,
                "disk_free_mb": profile.disk_free_mb,
            },
            "suggestion": suggestion,
            "adapters": {
                "ollama": adapters.ollama_available,
                "ollama_models": adapters.ollama_models,
                "transformers": adapters.transformers_available,
                "cuda": adapters.cuda_available,
                "openai": adapters.openai_key_set,
                "claude_cli": adapters.claude_cli_available,
            },
            "config_dir": str(config_dir),
            "env_generated": env_generated,
            "solvers_generated": solvers_generated,
            "first_run": first_run,
        }
