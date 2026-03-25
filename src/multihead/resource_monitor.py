"""Resource monitoring with time-series history.

Tracks GPU VRAM, system RAM, and disk usage over time in circular buffers.
Provides sparkline-ready data for dashboard visualization.
"""

from __future__ import annotations

import logging
import shutil
import time
import threading
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from multihead.subprocess_utils import no_window_flags

logger = logging.getLogger(__name__)

# Default history length (e.g., 60 samples at 10s interval = 10 min window)
DEFAULT_HISTORY_SIZE = 60


@dataclass
class ResourceSnapshot:
    """Point-in-time resource measurement."""

    timestamp: float = 0.0
    # GPU VRAM (MB)
    gpu_vram_used_mb: int = 0
    gpu_vram_free_mb: int = 0
    gpu_vram_total_mb: int = 0
    # System RAM (MB)
    ram_used_mb: int = 0
    ram_available_mb: int = 0
    ram_total_mb: int = 0
    # Disk (MB)
    disk_free_mb: int = 0
    disk_total_mb: int = 0
    # Process
    process_rss_mb: int = 0

    @property
    def gpu_vram_pct(self) -> float:
        if self.gpu_vram_total_mb == 0:
            return 0.0
        return (self.gpu_vram_used_mb / self.gpu_vram_total_mb) * 100

    @property
    def ram_pct(self) -> float:
        if self.ram_total_mb == 0:
            return 0.0
        return (self.ram_used_mb / self.ram_total_mb) * 100

    @property
    def disk_pct(self) -> float:
        if self.disk_total_mb == 0:
            return 0.0
        return ((self.disk_total_mb - self.disk_free_mb) / self.disk_total_mb) * 100

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "gpu_vram_used_mb": self.gpu_vram_used_mb,
            "gpu_vram_free_mb": self.gpu_vram_free_mb,
            "gpu_vram_total_mb": self.gpu_vram_total_mb,
            "gpu_vram_pct": round(self.gpu_vram_pct, 1),
            "ram_used_mb": self.ram_used_mb,
            "ram_available_mb": self.ram_available_mb,
            "ram_total_mb": self.ram_total_mb,
            "ram_pct": round(self.ram_pct, 1),
            "disk_free_mb": self.disk_free_mb,
            "disk_total_mb": self.disk_total_mb,
            "process_rss_mb": self.process_rss_mb,
        }


class ResourceTracker:
    """Collects resource snapshots and maintains time-series history."""

    def __init__(
        self,
        data_dir: str | Path = ".",
        history_size: int = DEFAULT_HISTORY_SIZE,
    ) -> None:
        self.data_dir = str(data_dir)
        self._history: deque[ResourceSnapshot] = deque(maxlen=history_size)
        self._lock = threading.Lock()
        self._running = False
        self._thread: threading.Thread | None = None

    def sample(self) -> ResourceSnapshot:
        """Take a single resource snapshot."""
        snap = ResourceSnapshot(timestamp=time.time())

        # GPU VRAM via torch.cuda
        try:
            import torch
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                snap.gpu_vram_total_mb = props.total_mem // (1024 * 1024)
                allocated = torch.cuda.memory_allocated(0) // (1024 * 1024)
                reserved = torch.cuda.memory_reserved(0) // (1024 * 1024)
                snap.gpu_vram_used_mb = reserved  # reserved is more accurate for actual GPU use
                snap.gpu_vram_free_mb = snap.gpu_vram_total_mb - snap.gpu_vram_used_mb
        except Exception:
            # Fallback: nvidia-smi
            try:
                import subprocess
                result = subprocess.run(
                    ["nvidia-smi", "--query-gpu=memory.used,memory.free,memory.total",
                     "--format=csv,noheader,nounits"],
                    capture_output=True, text=True, timeout=5,
                    creationflags=no_window_flags(),
                )
                if result.returncode == 0:
                    parts = result.stdout.strip().split(",")
                    if len(parts) >= 3:
                        snap.gpu_vram_used_mb = int(parts[0].strip())
                        snap.gpu_vram_free_mb = int(parts[1].strip())
                        snap.gpu_vram_total_mb = int(parts[2].strip())
            except Exception:
                pass

        # System RAM
        try:
            import psutil
            mem = psutil.virtual_memory()
            snap.ram_total_mb = mem.total // (1024 * 1024)
            snap.ram_available_mb = mem.available // (1024 * 1024)
            snap.ram_used_mb = mem.used // (1024 * 1024)
            # Process RSS
            proc = psutil.Process()
            snap.process_rss_mb = proc.memory_info().rss // (1024 * 1024)
        except ImportError:
            # psutil not available — try /proc/meminfo on Linux
            try:
                with open("/proc/meminfo") as f:
                    info = {}
                    for line in f:
                        parts = line.split()
                        if len(parts) >= 2:
                            info[parts[0].rstrip(":")] = int(parts[1])
                    snap.ram_total_mb = info.get("MemTotal", 0) // 1024
                    snap.ram_available_mb = info.get("MemAvailable", 0) // 1024
                    snap.ram_used_mb = snap.ram_total_mb - snap.ram_available_mb
            except Exception:
                pass

        # Disk
        try:
            usage = shutil.disk_usage(self.data_dir)
            snap.disk_free_mb = int(usage.free / (1024 * 1024))
            snap.disk_total_mb = int(usage.total / (1024 * 1024))
        except Exception:
            pass

        with self._lock:
            self._history.append(snap)

        return snap

    @property
    def latest(self) -> ResourceSnapshot | None:
        """Most recent snapshot."""
        with self._lock:
            return self._history[-1] if self._history else None

    @property
    def history(self) -> list[ResourceSnapshot]:
        """Full history as a list (oldest first)."""
        with self._lock:
            return list(self._history)

    def series(self, field: str) -> list[float]:
        """Extract a single metric as a time series for sparklines.

        Args:
            field: Attribute name on ResourceSnapshot (e.g., 'gpu_vram_used_mb')
        """
        with self._lock:
            return [getattr(s, field, 0) for s in self._history]

    def start_background(self, interval: float = 10.0) -> None:
        """Start background sampling thread."""
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._background_loop,
            args=(interval,),
            daemon=True,
            name="resource-tracker",
        )
        self._thread.start()
        logger.info("ResourceTracker started (interval=%.0fs)", interval)

    def stop(self) -> None:
        """Stop background sampling."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
            self._thread = None

    def _background_loop(self, interval: float) -> None:
        while self._running:
            try:
                self.sample()
            except Exception as e:
                logger.debug("Resource sample error: %s", e)
            time.sleep(interval)


def sparkline(values: list[float], width: int = 20) -> str:
    """Render a Unicode sparkline from a list of values.

    Uses block characters ▁▂▃▄▅▆▇█ to represent relative values.

    Args:
        values: Numeric values to plot
        width: Max characters to render (truncates from left if needed)

    Returns:
        Sparkline string, e.g., "▁▂▃▅▇█▇▅▃▂"
    """
    if not values:
        return ""

    blocks = "▁▂▃▄▅▆▇█"
    # Take last `width` values
    try:
        vals = list(values[-width:])
    except (TypeError, IndexError):
        return ""
    if not vals:
        return ""
    try:
        mn, mx = min(vals), max(vals)
    except (TypeError, ValueError):
        return ""

    if mn == mx:
        # Flat line — use middle block
        return blocks[3] * len(vals)

    span = mx - mn
    result = []
    for v in vals:
        idx = int((v - mn) / span * (len(blocks) - 1))
        result.append(blocks[idx])
    return "".join(result)
