"""Resilience patterns: circuit breaker and resource monitoring."""

from __future__ import annotations

import shutil
import time
from typing import Any


# ---------------------------------------------------------------------------
# Circuit Breaker
# ---------------------------------------------------------------------------


class CircuitBreakerOpen(Exception):
    """Raised when the circuit breaker is open and rejecting calls."""


class CircuitBreaker:
    """Simple circuit breaker for adapter calls.

    States:
        closed  — normal operation, failures counted
        open    — all calls rejected immediately
        half_open — next call is a probe; success closes, failure re-opens
    """

    def __init__(self, failure_threshold: int = 5, reset_timeout_s: float = 60.0) -> None:
        self.failure_threshold = failure_threshold
        self.reset_timeout_s = reset_timeout_s
        self._failure_count = 0
        self._state = "closed"
        self._opened_at: float = 0.0

    @property
    def state(self) -> str:
        if self._state == "open" and time.monotonic() - self._opened_at >= self.reset_timeout_s:
            self._state = "half_open"
        return self._state

    def record_success(self) -> None:
        self._failure_count = 0
        self._state = "closed"

    def record_failure(self) -> None:
        self._failure_count += 1
        if self._failure_count >= self.failure_threshold:
            self._state = "open"
            self._opened_at = time.monotonic()

    async def call(self, coro: Any) -> Any:
        """Execute an async callable through the circuit breaker."""
        state = self.state
        if state == "open":
            raise CircuitBreakerOpen(
                f"Circuit open after {self.failure_threshold} failures. "
                f"Retry after {self.reset_timeout_s}s."
            )
        try:
            result = await coro
            self.record_success()
            return result
        except Exception:
            self.record_failure()
            raise


# ---------------------------------------------------------------------------
# Resource Monitor
# ---------------------------------------------------------------------------


class ResourceStatus:
    """Snapshot of system resource usage."""

    def __init__(
        self,
        disk_free_mb: int = 0,
        disk_total_mb: int = 0,
        gpu_vram_free_mb: int = 0,
        gpu_vram_total_mb: int = 0,
        warnings: list[str] | None = None,
    ) -> None:
        self.disk_free_mb = disk_free_mb
        self.disk_total_mb = disk_total_mb
        self.gpu_vram_free_mb = gpu_vram_free_mb
        self.gpu_vram_total_mb = gpu_vram_total_mb
        self.warnings = warnings or []

    def to_dict(self) -> dict[str, Any]:
        return {
            "disk_free_mb": self.disk_free_mb,
            "disk_total_mb": self.disk_total_mb,
            "gpu_vram_free_mb": self.gpu_vram_free_mb,
            "gpu_vram_total_mb": self.gpu_vram_total_mb,
            "warnings": self.warnings,
        }


class ResourceMonitor:
    """Monitor system resources and flag low conditions."""

    def __init__(
        self,
        min_disk_mb: int = 1024,
        min_vram_mb: int = 512,
    ) -> None:
        self.min_disk_mb = min_disk_mb
        self.min_vram_mb = min_vram_mb

    def check(self, data_dir: str = ".") -> ResourceStatus:
        warnings: list[str] = []

        # Disk
        disk_free_mb = 0
        disk_total_mb = 0
        try:
            usage = shutil.disk_usage(data_dir)
            disk_free_mb = int(usage.free / (1024 * 1024))
            disk_total_mb = int(usage.total / (1024 * 1024))
            if disk_free_mb < self.min_disk_mb:
                warnings.append(f"Low disk space: {disk_free_mb}MB free (min {self.min_disk_mb}MB)")
        except Exception as e:
            warnings.append(f"Disk check failed: {e}")

        # GPU VRAM (nvidia-smi)
        gpu_free = 0
        gpu_total = 0
        try:
            import subprocess

            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=memory.free,memory.total", "--format=csv,noheader,nounits"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split(",")
                if len(parts) >= 2:
                    gpu_free = int(parts[0].strip())
                    gpu_total = int(parts[1].strip())
                    if gpu_free < self.min_vram_mb:
                        warnings.append(f"Low VRAM: {gpu_free}MB free (min {self.min_vram_mb}MB)")
        except FileNotFoundError:
            pass  # No nvidia-smi, no GPU monitoring
        except Exception as e:
            warnings.append(f"GPU check failed: {e}")

        return ResourceStatus(
            disk_free_mb=disk_free_mb,
            disk_total_mb=disk_total_mb,
            gpu_vram_free_mb=gpu_free,
            gpu_vram_total_mb=gpu_total,
            warnings=warnings,
        )
