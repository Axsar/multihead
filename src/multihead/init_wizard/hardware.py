"""Hardware detection and adapter availability checks."""

from __future__ import annotations

import logging
import os
import platform
import shutil
import subprocess
from pathlib import Path

from .models import AdapterStatus, HardwareProfile

logger = logging.getLogger(__name__)


def detect_hardware() -> HardwareProfile:
    """Detect local hardware capabilities."""
    profile = HardwareProfile(platform=platform.system())

    # CPU RAM
    try:
        import psutil
        profile.cpu_ram_mb = int(psutil.virtual_memory().total / (1024 * 1024))
    except ImportError:
        # Fallback: read /proc/meminfo on Linux
        try:
            with open("/proc/meminfo") as f:
                for line in f:
                    if line.startswith("MemTotal:"):
                        kb = int(line.split()[1])
                        profile.cpu_ram_mb = kb // 1024
                        break
        except (FileNotFoundError, ValueError) as e:
            logger.debug("Could not read /proc/meminfo: %s", e)

    # Disk free
    try:
        usage = shutil.disk_usage(Path.home())
        profile.disk_free_mb = int(usage.free / (1024 * 1024))
    except Exception as e:
        logger.debug("Could not detect disk space: %s", e)

    # GPU detection
    try:
        import torch
        if torch.cuda.is_available():
            profile.has_cuda = True
            profile.gpu_name = torch.cuda.get_device_name(0)
            profile.gpu_vram_mb = int(torch.cuda.get_device_properties(0).total_memory / (1024 * 1024))
    except ImportError:
        # Try nvidia-smi
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=10,
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split(", ")
                if len(parts) >= 2:
                    profile.gpu_name = parts[0]
                    profile.gpu_vram_mb = int(parts[1])
                    profile.has_cuda = True
        except (FileNotFoundError, subprocess.TimeoutExpired, ValueError) as e:
            logger.debug("nvidia-smi fallback failed: %s", e)

    return profile


def check_adapters() -> AdapterStatus:
    """Check which model adapters are available on this system."""
    status = AdapterStatus()

    # Check ollama
    try:
        result = subprocess.run(
            ["ollama", "list"], capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            status.ollama_available = True
            status.ollama_models = [
                line.split()[0]
                for line in result.stdout.strip().split("\n")[1:]
                if line.strip()
            ]
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    # Check transformers + torch
    try:
        import torch
        status.transformers_available = True
        status.cuda_available = torch.cuda.is_available()
    except ImportError:
        pass

    # Check OpenAI API key
    status.openai_key_set = bool(os.environ.get("OPENAI_API_KEY"))

    # Check claude CLI
    status.claude_cli_available = shutil.which("claude") is not None

    return status
