"""Tests for resilience patterns."""

from __future__ import annotations

import time

import pytest

from multihead.resilience import CircuitBreaker, CircuitBreakerOpen, ResourceMonitor, ResourceStatus


class TestCircuitBreaker:
    def test_initial_state_closed(self):
        cb = CircuitBreaker()
        assert cb.state == "closed"

    def test_stays_closed_under_threshold(self):
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        assert cb.state == "closed"

    def test_opens_at_threshold(self):
        cb = CircuitBreaker(failure_threshold=3)
        for _ in range(3):
            cb.record_failure()
        assert cb.state == "open"

    def test_success_resets(self):
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        assert cb.state == "closed"
        # Should need 3 more failures to open
        cb.record_failure()
        cb.record_failure()
        assert cb.state == "closed"

    def test_half_open_after_timeout(self):
        cb = CircuitBreaker(failure_threshold=1, reset_timeout_s=0.01)
        cb.record_failure()
        assert cb.state == "open"
        time.sleep(0.02)
        assert cb.state == "half_open"

    @pytest.mark.asyncio
    async def test_call_success(self):
        cb = CircuitBreaker()

        async def ok():
            return 42

        result = await cb.call(ok())
        assert result == 42
        assert cb.state == "closed"

    @pytest.mark.asyncio
    async def test_call_failure(self):
        cb = CircuitBreaker(failure_threshold=1)

        async def fail():
            raise ValueError("boom")

        with pytest.raises(ValueError, match="boom"):
            await cb.call(fail())
        assert cb.state == "open"

    @pytest.mark.asyncio
    async def test_call_rejected_when_open(self):
        cb = CircuitBreaker(failure_threshold=1, reset_timeout_s=60)
        cb.record_failure()

        assert cb.state == "open"
        # Verify the breaker rejects without needing to create a coroutine
        with pytest.raises(CircuitBreakerOpen):
            await cb.call(None)  # type: ignore[arg-type]

    @pytest.mark.asyncio
    async def test_half_open_success_closes(self):
        cb = CircuitBreaker(failure_threshold=1, reset_timeout_s=0.01)
        cb.record_failure()
        time.sleep(0.02)
        assert cb.state == "half_open"

        async def ok():
            return "recovered"

        result = await cb.call(ok())
        assert result == "recovered"
        assert cb.state == "closed"

    @pytest.mark.asyncio
    async def test_half_open_failure_reopens(self):
        cb = CircuitBreaker(failure_threshold=1, reset_timeout_s=0.01)
        cb.record_failure()
        time.sleep(0.02)
        assert cb.state == "half_open"

        async def fail():
            raise RuntimeError("still broken")

        with pytest.raises(RuntimeError):
            await cb.call(fail())
        assert cb.state == "open"


class TestResourceStatus:
    def test_to_dict(self):
        rs = ResourceStatus(disk_free_mb=500, disk_total_mb=10000, warnings=["low disk"])
        d = rs.to_dict()
        assert d["disk_free_mb"] == 500
        assert d["warnings"] == ["low disk"]


class TestResourceMonitor:
    def test_check_disk(self):
        mon = ResourceMonitor(min_disk_mb=1)
        status = mon.check(".")
        assert status.disk_total_mb > 0
        assert status.disk_free_mb >= 0

    def test_low_disk_warning(self):
        mon = ResourceMonitor(min_disk_mb=999_999_999)  # impossibly high
        status = mon.check(".")
        assert any("Low disk" in w for w in status.warnings)
