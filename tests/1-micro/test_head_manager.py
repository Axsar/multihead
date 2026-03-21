"""Tests for the head manager with mock adapters."""


import pytest

from multihead.head_manager import HeadManager
from multihead.models import AdapterKind, HeadManifest, HeadState
from multihead.observability import MetricsCollector
from multihead.resilience import CircuitBreakerOpen


@pytest.fixture
def manifests():
    return {
        "mock-llm": HeadManifest(
            head_id="mock-llm",
            name="Mock LLM",
            adapter=AdapterKind.MOCK,
            model="mock-v1",
            kind="llm",
            gpu_required=False,
        ),
        "mock-vlm": HeadManifest(
            head_id="mock-vlm",
            name="Mock VLM",
            adapter=AdapterKind.MOCK,
            model="mock-v1",
            kind="vlm",
            gpu_required=False,
        ),
    }


@pytest.fixture
def manager(manifests):
    return HeadManager(manifests)


@pytest.mark.asyncio
async def test_initial_state(manager):
    assert manager.active_head is None
    assert manager.get_state("mock-llm") == HeadState.OFF
    assert manager.get_state("mock-vlm") == HeadState.OFF


@pytest.mark.asyncio
async def test_ensure_active(manager):
    await manager.ensure_active("mock-llm")
    assert manager.active_head == "mock-llm"
    assert manager.get_state("mock-llm") == HeadState.ACTIVE


@pytest.mark.asyncio
async def test_head_swap(manager):
    """Activating a new head should unload the current one."""
    await manager.ensure_active("mock-llm")
    assert manager.active_head == "mock-llm"

    await manager.ensure_active("mock-vlm")
    assert manager.active_head == "mock-vlm"
    assert manager.get_state("mock-llm") == HeadState.OFF
    assert manager.get_state("mock-vlm") == HeadState.ACTIVE


@pytest.mark.asyncio
async def test_ensure_active_idempotent(manager):
    """Calling ensure_active for already active head should be a no-op."""
    await manager.ensure_active("mock-llm")
    await manager.ensure_active("mock-llm")
    assert manager.active_head == "mock-llm"


@pytest.mark.asyncio
async def test_get_states(manager):
    states = manager.get_states()
    assert "mock-llm" in states
    assert "mock-vlm" in states
    assert states["mock-llm"]["state"] == "off"


@pytest.mark.asyncio
async def test_unknown_head(manager):
    with pytest.raises(KeyError):
        await manager.ensure_active("nonexistent")


@pytest.mark.asyncio
async def test_shutdown(manager):
    await manager.ensure_active("mock-llm")
    await manager.shutdown()
    assert manager.active_head is None
    assert manager.get_state("mock-llm") == HeadState.OFF


# ---------------------------------------------------------------------------
# Circuit Breaker Integration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_breaker_initial_closed(manager):
    breaker = manager.get_breaker("mock-llm")
    assert breaker is not None
    assert breaker.state == "closed"


@pytest.mark.asyncio
async def test_breaker_state_in_get_states(manager):
    states = manager.get_states()
    assert states["mock-llm"]["circuit_breaker"] == "closed"


@pytest.mark.asyncio
async def test_breaker_success_on_load(manager):
    await manager.ensure_active("mock-llm")
    breaker = manager.get_breaker("mock-llm")
    assert breaker.state == "closed"
    assert breaker._failure_count == 0


@pytest.mark.asyncio
async def test_generate_through_breaker(manager):
    """HeadManager.generate() wraps adapter call through circuit breaker."""
    result = await manager.generate("mock-llm", "hello")
    assert "text" in result
    breaker = manager.get_breaker("mock-llm")
    assert breaker.state == "closed"


@pytest.mark.asyncio
async def test_breaker_opens_after_failures(manifests):
    """Circuit breaker should open after threshold failures."""
    mgr = HeadManager(manifests, breaker_threshold=2, breaker_timeout_s=60.0)
    await mgr.ensure_active("mock-llm")

    adapter = mgr.get_adapter("mock-llm")
    original_generate = adapter.generate

    # Force generate to fail
    async def failing_generate(prompt, **kw):
        raise RuntimeError("GPU OOM")

    adapter.generate = failing_generate

    # Fail twice to trip the breaker
    for _ in range(2):
        with pytest.raises(RuntimeError, match="GPU OOM"):
            await mgr.generate("mock-llm", "test")

    breaker = mgr.get_breaker("mock-llm")
    assert breaker.state == "open"

    # Next call should be rejected by breaker
    with pytest.raises(CircuitBreakerOpen):
        await mgr.generate("mock-llm", "test")

    # Restore for cleanup
    adapter.generate = original_generate


@pytest.mark.asyncio
async def test_breaker_recovers_after_success(manifests):
    """Circuit breaker resets after a successful call."""
    mgr = HeadManager(manifests, breaker_threshold=2, breaker_timeout_s=60.0)
    await mgr.ensure_active("mock-llm")

    adapter = mgr.get_adapter("mock-llm")
    breaker = mgr.get_breaker("mock-llm")

    # Record one failure (below threshold)
    breaker.record_failure()
    assert breaker._failure_count == 1

    # Successful generate should reset
    await mgr.generate("mock-llm", "test")
    assert breaker._failure_count == 0
    assert breaker.state == "closed"


# ---------------------------------------------------------------------------
# Streaming
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_stream_basic(manager):
    """generate_stream should yield tokens from the adapter."""
    tokens = []
    async for chunk in manager.generate_stream("mock-llm", "hello"):
        tokens.append(chunk)
    assert len(tokens) > 0
    full_text = "".join(tokens)
    assert "Mock LLM response" in full_text


@pytest.mark.asyncio
async def test_generate_stream_records_success(manager):
    """Successful streaming should record success on the circuit breaker."""
    breaker = manager.get_breaker("mock-llm")
    breaker.record_failure()  # put 1 failure on the counter
    assert breaker._failure_count == 1

    async for _ in manager.generate_stream("mock-llm", "test"):
        pass

    assert breaker._failure_count == 0
    assert breaker.state == "closed"


@pytest.mark.asyncio
async def test_generate_stream_rejected_when_open(manifests):
    """generate_stream should raise CircuitBreakerOpen when breaker is open."""
    mgr = HeadManager(manifests, breaker_threshold=1, breaker_timeout_s=60.0)
    await mgr.ensure_active("mock-llm")
    mgr.get_breaker("mock-llm").record_failure()

    with pytest.raises(CircuitBreakerOpen):
        async for _ in mgr.generate_stream("mock-llm", "test"):
            pass


@pytest.mark.asyncio
async def test_generate_stream_records_failure(manifests):
    """Streaming failure should record failure on the circuit breaker."""
    mgr = HeadManager(manifests, breaker_threshold=3, breaker_timeout_s=60.0)
    await mgr.ensure_active("mock-llm")

    adapter = mgr.get_adapter("mock-llm")

    async def failing_stream(prompt, **kw):
        yield "partial"
        raise RuntimeError("stream error")

    adapter.generate_stream = failing_stream

    with pytest.raises(RuntimeError, match="stream error"):
        async for _ in mgr.generate_stream("mock-llm", "test"):
            pass

    breaker = mgr.get_breaker("mock-llm")
    assert breaker._failure_count == 1


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_metrics_on_load(manifests):
    """Load should increment head_loads_total and record latency."""
    metrics = MetricsCollector()
    mgr = HeadManager(manifests, metrics=metrics)
    await mgr.ensure_active("mock-llm")

    assert metrics.counter("head_loads_total", labels={"head_id": "mock-llm"}) == 1.0
    hist = metrics.histogram("head_load_seconds", labels={"head_id": "mock-llm"})
    assert hist["count"] == 1
    assert hist["avg"] > 0


@pytest.mark.asyncio
async def test_metrics_on_generate(manifests):
    """Generate should increment counters and record latency."""
    metrics = MetricsCollector()
    mgr = HeadManager(manifests, metrics=metrics)
    await mgr.generate("mock-llm", "hello")

    assert metrics.counter("head_generate_total", labels={"head_id": "mock-llm"}) == 1.0
    hist = metrics.histogram("head_generate_seconds", labels={"head_id": "mock-llm"})
    assert hist["count"] == 1


@pytest.mark.asyncio
async def test_metrics_on_generate_failure(manifests):
    """Failed generate should increment error counter."""
    metrics = MetricsCollector()
    mgr = HeadManager(manifests, breaker_threshold=5, metrics=metrics)
    await mgr.ensure_active("mock-llm")

    adapter = mgr.get_adapter("mock-llm")
    original = adapter.generate

    async def fail(prompt, **kw):
        raise RuntimeError("boom")

    adapter.generate = fail

    with pytest.raises(RuntimeError):
        await mgr.generate("mock-llm", "test")

    assert metrics.counter("head_generate_errors_total", labels={"head_id": "mock-llm"}) == 1.0
    adapter.generate = original
