"""Diagnostics: health checks, error messages, common problem solutions."""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from multihead.subprocess_utils import no_window_flags


@dataclass
class DiagnosticResult:
    """Result of a single diagnostic check."""
    name: str
    passed: bool
    message: str
    suggestion: str = ""


@dataclass
class DiagnosticReport:
    """Full diagnostic report."""
    checks: list[DiagnosticResult] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return all(c.passed for c in self.checks)

    @property
    def summary(self) -> str:
        passed = sum(1 for c in self.checks if c.passed)
        total = len(self.checks)
        return f"{passed}/{total} checks passed"


class Diagnostics:
    """Run diagnostic checks on the MultiHead installation."""

    def __init__(self, data_dir: Path, config_dir: Path) -> None:
        self.data_dir = data_dir
        self.config_dir = config_dir

    def run_all(self) -> DiagnosticReport:
        """Run all diagnostic checks."""
        report = DiagnosticReport()
        report.checks.append(self._check_data_dir())
        report.checks.append(self._check_config_dir())
        report.checks.append(self._check_heads_yaml())
        report.checks.append(self._check_knowledge_db())
        report.checks.append(self._check_disk_space())
        report.checks.append(self._check_python_deps())
        report.checks.append(self._check_ollama())
        report.checks.append(self._check_torch())
        report.checks.append(self._check_vram_vs_heads())
        report.checks.append(self._check_claude_cli())
        report.checks.append(self._check_env_file())
        report.checks.append(self._check_acp_reachability())
        return report

    def _check_data_dir(self) -> DiagnosticResult:
        if self.data_dir.exists():
            return DiagnosticResult("data_dir", True, f"Data directory exists: {self.data_dir}")
        return DiagnosticResult(
            "data_dir", False, f"Data directory missing: {self.data_dir}",
            "Run 'multihead init' to create it.",
        )

    def _check_config_dir(self) -> DiagnosticResult:
        if self.config_dir.exists():
            return DiagnosticResult("config_dir", True, f"Config directory exists: {self.config_dir}")
        return DiagnosticResult(
            "config_dir", False, f"Config directory missing: {self.config_dir}",
            "Run 'multihead init' to create it.",
        )

    def _check_heads_yaml(self) -> DiagnosticResult:
        heads_file = self.config_dir / "heads.yaml"
        if heads_file.exists():
            return DiagnosticResult("heads_yaml", True, "heads.yaml found")
        return DiagnosticResult(
            "heads_yaml", False, "heads.yaml not found",
            "Run 'multihead init' to generate heads.yaml based on your hardware.",
        )

    def _check_disk_space(self) -> DiagnosticResult:
        try:
            usage = shutil.disk_usage(self.data_dir if self.data_dir.exists() else Path.home())
            free_gb = usage.free / (1024 ** 3)
            if free_gb >= 5:
                return DiagnosticResult("disk_space", True, f"{free_gb:.1f} GB free")
            return DiagnosticResult(
                "disk_space", False, f"Low disk space: {free_gb:.1f} GB free",
                "Models and artifacts need space. Free up at least 10 GB.",
            )
        except Exception as e:
            return DiagnosticResult("disk_space", False, f"Could not check: {e}")

    def _check_python_deps(self) -> DiagnosticResult:
        missing = []
        for mod in ("fastapi", "uvicorn", "yaml", "pydantic", "httpx"):
            try:
                __import__(mod)
            except ImportError:
                missing.append(mod)
        if not missing:
            return DiagnosticResult("python_deps", True, "All required Python packages found")
        return DiagnosticResult(
            "python_deps", False, f"Missing packages: {', '.join(missing)}",
            f"Install with: pip install {' '.join(missing)}",
        )

    def _check_ollama(self) -> DiagnosticResult:
        try:
            import subprocess
            result = subprocess.run(
                ["ollama", "list"], capture_output=True, text=True, timeout=10,
                creationflags=no_window_flags(),
            )
            if result.returncode == 0:
                models = [l.split()[0] for l in result.stdout.strip().split("\n")[1:] if l.strip()]
                return DiagnosticResult("ollama", True, f"Ollama running with {len(models)} models")
            return DiagnosticResult(
                "ollama", False, "Ollama not responding",
                "Start Ollama with 'ollama serve' or install from ollama.ai",
            )
        except FileNotFoundError:
            return DiagnosticResult(
                "ollama", False, "Ollama not installed",
                "Install from https://ollama.ai",
            )
        except Exception as e:
            return DiagnosticResult("ollama", False, f"Check failed: {e}")

    def _check_torch(self) -> DiagnosticResult:
        """Check if PyTorch is installed and CUDA is available."""
        try:
            import torch
            if torch.cuda.is_available():
                name = torch.cuda.get_device_name(0)
                vram = torch.cuda.get_device_properties(0).total_memory // (1024**2)
                return DiagnosticResult(
                    "torch_cuda", True,
                    f"PyTorch {torch.__version__} with CUDA ({name}, {vram} MB)")
            return DiagnosticResult(
                "torch_cuda", True,
                f"PyTorch {torch.__version__} (CPU only)",
                "For GPU inference, install PyTorch with CUDA support")
        except ImportError:
            return DiagnosticResult(
                "torch_cuda", False, "PyTorch not installed",
                "Required for transformers adapter: pip install torch")

    def _check_claude_cli(self) -> DiagnosticResult:
        """Check if claude CLI is available."""
        if shutil.which("claude"):
            return DiagnosticResult("claude_cli", True, "Claude CLI found")
        return DiagnosticResult(
            "claude_cli", False, "Claude CLI not found",
            "Optional: install from https://docs.anthropic.com/en/docs/claude-code")

    def _check_env_file(self) -> DiagnosticResult:
        """Check if .env file exists."""
        env_path = Path(".env")
        if env_path.exists():
            return DiagnosticResult("env_file", True, ".env file found")
        example = Path(".env.example")
        suggestion = "Copy .env.example to .env" if example.exists() else "Run 'multihead init --auto' to generate"
        return DiagnosticResult("env_file", False, ".env file not found", suggestion)

    def _check_knowledge_db(self) -> DiagnosticResult:
        """Check knowledge.db health and claim count."""
        db_path = self.data_dir / "knowledge.db"
        if not db_path.exists():
            return DiagnosticResult(
                "knowledge_db", False, "knowledge.db not found",
                "Run 'multihead shell' to auto-create, or 'multihead init'.",
            )
        try:
            import sqlite3
            conn = sqlite3.connect(str(db_path))
            count = conn.execute("SELECT COUNT(*) FROM claims").fetchone()[0]
            conn.close()
            return DiagnosticResult(
                "knowledge_db", True,
                f"knowledge.db OK ({count:,} claims)",
            )
        except Exception as e:
            return DiagnosticResult(
                "knowledge_db", False, f"knowledge.db error: {e}",
                "Database may be corrupted. Back up and reinitialize.",
            )

    def _check_vram_vs_heads(self) -> DiagnosticResult:
        """Check if configured heads fit in available GPU VRAM."""
        heads_file = self.config_dir / "heads.yaml"
        if not heads_file.exists():
            return DiagnosticResult("vram_heads", True, "No heads.yaml to check")
        try:
            import torch
            if not torch.cuda.is_available():
                return DiagnosticResult(
                    "vram_heads", True, "No GPU — only CPU/API heads usable")
            vram_mb = torch.cuda.get_device_properties(0).total_memory // (1024**2)
        except ImportError:
            return DiagnosticResult("vram_heads", True, "No PyTorch — skipping VRAM check")

        try:
            import yaml
            with open(heads_file) as f:
                data = yaml.safe_load(f)
            heads = data.get("heads", [])
            gpu_heads = [
                h for h in heads
                if h.get("gpu_required", True)
                and h.get("adapter") not in ("mock", "openai", "claude-agent-sdk")
            ]
            if not gpu_heads:
                return DiagnosticResult("vram_heads", True, "No GPU heads configured")

            # Estimate: 4-bit quant ~= params/2 MB
            largest = max(gpu_heads, key=lambda h: h.get("vram_mb", 0))
            largest_vram = largest.get("vram_mb", 0)
            name = largest.get("head_id", "unknown")

            if largest_vram and largest_vram > vram_mb:
                return DiagnosticResult(
                    "vram_heads", False,
                    f"Head '{name}' needs ~{largest_vram} MB but GPU has {vram_mb} MB",
                    f"Disable '{name}' in heads.yaml or use a smaller model.",
                )
            return DiagnosticResult(
                "vram_heads", True,
                f"GPU has {vram_mb} MB — largest head '{name}' fits"
                + (f" (~{largest_vram} MB)" if largest_vram else ""),
            )
        except Exception as e:
            return DiagnosticResult("vram_heads", True, f"Could not check: {e}")

    def _check_acp_reachability(self) -> DiagnosticResult:
        """Check if ACP/BotVibes server is reachable."""
        import os
        acp_url = os.environ.get("ACP_URL", "")
        if not acp_url:
            return DiagnosticResult(
                "acp_server", True, "ACP not configured (optional)",
                "Set ACP_URL in .env for BotVibes integration.",
            )
        try:
            import urllib.request
            url = f"{acp_url.rstrip('/')}/health"
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=5) as resp:
                if resp.status == 200:
                    return DiagnosticResult(
                        "acp_server", True, f"ACP server reachable at {acp_url}")
            return DiagnosticResult(
                "acp_server", False, f"ACP returned status {resp.status}",
                "Check that BotVibes server is running.",
            )
        except Exception as e:
            return DiagnosticResult(
                "acp_server", False, f"ACP unreachable: {e}",
                f"Check that BotVibes is running at {acp_url}",
            )
