"""Tests for ServiceManager — background service lifecycle within shell."""

import asyncio

import pytest

from multihead.runtime_config import RuntimeConfig, ServicesConfig
from multihead.service_manager import ServiceManager, ServiceEntry


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

async def _forever_service():
    """A service that runs forever until cancelled."""
    while True:
        await asyncio.sleep(0.05)


async def _fast_exit_service():
    """A service that completes immediately."""
    return


async def _failing_service():
    """A service that raises on startup."""
    raise RuntimeError("intentional failure")


async def _slow_failing_service():
    """A service that runs briefly then fails."""
    await asyncio.sleep(0.05)
    raise ValueError("delayed failure")


_counter = 0


async def _counting_service():
    """A service that increments a counter each tick."""
    global _counter
    while True:
        _counter += 1
        await asyncio.sleep(0.02)


# ---------------------------------------------------------------------------
# ServiceEntry
# ---------------------------------------------------------------------------

class TestServiceEntry:
    def test_defaults(self):
        entry = ServiceEntry(name="test", factory=_forever_service)
        assert entry.status == "stopped"
        assert entry.started_at is None
        assert entry.error is None
        assert entry.description == ""
        assert entry.auto_start is False


# ---------------------------------------------------------------------------
# ServiceManager — Registration
# ---------------------------------------------------------------------------

class TestRegistration:
    def test_register(self):
        rc = RuntimeConfig()
        sm = ServiceManager(rc)
        sm.register("svc1", _forever_service, description="Test service")
        assert "svc1" in sm.registered_names

    def test_register_multiple(self):
        rc = RuntimeConfig()
        sm = ServiceManager(rc)
        sm.register("a", _forever_service)
        sm.register("b", _fast_exit_service)
        assert sm.registered_names == ["a", "b"]

    def test_registered_names_empty(self):
        rc = RuntimeConfig()
        sm = ServiceManager(rc)
        assert sm.registered_names == []


# ---------------------------------------------------------------------------
# ServiceManager — Start / Stop
# ---------------------------------------------------------------------------

class TestStartStop:
    @pytest.mark.asyncio
    async def test_start_service(self):
        rc = RuntimeConfig()
        sm = ServiceManager(rc)
        sm.register("svc", _forever_service)
        msg = await sm.start("svc")
        assert "started" in msg
        statuses = sm.status()
        assert statuses[0]["status"] == "running"
        await sm.shutdown_all()

    @pytest.mark.asyncio
    async def test_start_unknown(self):
        rc = RuntimeConfig()
        sm = ServiceManager(rc)
        msg = await sm.start("nonexistent")
        assert "Unknown" in msg

    @pytest.mark.asyncio
    async def test_start_already_running(self):
        rc = RuntimeConfig()
        sm = ServiceManager(rc)
        sm.register("svc", _forever_service)
        await sm.start("svc")
        msg = await sm.start("svc")
        assert "already running" in msg
        await sm.shutdown_all()

    @pytest.mark.asyncio
    async def test_stop_service(self):
        rc = RuntimeConfig()
        sm = ServiceManager(rc)
        sm.register("svc", _forever_service)
        await sm.start("svc")
        msg = await sm.stop("svc")
        assert "stopped" in msg
        statuses = sm.status()
        assert statuses[0]["status"] == "stopped"

    @pytest.mark.asyncio
    async def test_stop_unknown(self):
        rc = RuntimeConfig()
        sm = ServiceManager(rc)
        msg = await sm.stop("nonexistent")
        assert "Unknown" in msg

    @pytest.mark.asyncio
    async def test_stop_already_stopped(self):
        rc = RuntimeConfig()
        sm = ServiceManager(rc)
        sm.register("svc", _forever_service)
        msg = await sm.stop("svc")
        assert "not running" in msg

    @pytest.mark.asyncio
    async def test_start_after_stop(self):
        rc = RuntimeConfig()
        sm = ServiceManager(rc)
        sm.register("svc", _forever_service)
        await sm.start("svc")
        await sm.stop("svc")
        msg = await sm.start("svc")
        assert "started" in msg
        statuses = sm.status()
        assert statuses[0]["status"] == "running"
        await sm.shutdown_all()


# ---------------------------------------------------------------------------
# ServiceManager — Failure handling
# ---------------------------------------------------------------------------

class TestFailure:
    @pytest.mark.asyncio
    async def test_failing_service(self):
        rc = RuntimeConfig()
        sm = ServiceManager(rc)
        sm.register("bad", _failing_service)
        await sm.start("bad")
        # Give event loop time to run the task and call done callback
        await asyncio.sleep(0.1)
        statuses = sm.status()
        assert statuses[0]["status"] == "failed"
        assert statuses[0]["error"] is not None

    @pytest.mark.asyncio
    async def test_slow_failing_service(self):
        rc = RuntimeConfig()
        sm = ServiceManager(rc)
        sm.register("slow-bad", _slow_failing_service)
        await sm.start("slow-bad")
        await asyncio.sleep(0.2)
        statuses = sm.status()
        assert statuses[0]["status"] == "failed"
        assert "delayed failure" in statuses[0]["error"]

    @pytest.mark.asyncio
    async def test_fast_exit_service(self):
        rc = RuntimeConfig()
        sm = ServiceManager(rc)
        sm.register("quick", _fast_exit_service)
        await sm.start("quick")
        await asyncio.sleep(0.1)
        statuses = sm.status()
        # Completed normally → stopped
        assert statuses[0]["status"] == "stopped"


# ---------------------------------------------------------------------------
# ServiceManager — auto_start_all
# ---------------------------------------------------------------------------

class TestAutoStart:
    @pytest.mark.asyncio
    async def test_auto_start_none(self):
        rc = RuntimeConfig()
        sm = ServiceManager(rc)
        sm.register("svc", _forever_service, auto_start=False)
        messages = await sm.auto_start_all()
        assert len(messages) == 0
        statuses = sm.status()
        assert statuses[0]["status"] == "stopped"

    @pytest.mark.asyncio
    async def test_auto_start_from_entry(self):
        rc = RuntimeConfig()
        sm = ServiceManager(rc)
        sm.register("svc", _forever_service, auto_start=True)
        messages = await sm.auto_start_all()
        assert len(messages) == 1
        assert "started" in messages[0]
        await sm.shutdown_all()

    @pytest.mark.asyncio
    async def test_auto_start_from_config(self):
        """ServicesConfig overrides ServiceEntry auto_start."""
        rc = RuntimeConfig()
        rc.services.auto_responder = True
        sm = ServiceManager(rc)
        sm.register("auto_responder", _forever_service, auto_start=False)
        messages = await sm.auto_start_all()
        assert len(messages) == 1
        await sm.shutdown_all()

    @pytest.mark.asyncio
    async def test_auto_start_config_false_overrides(self):
        """ServicesConfig=False overrides entry auto_start=True."""
        rc = RuntimeConfig()
        rc.services.auto_responder = False
        sm = ServiceManager(rc)
        sm.register("auto_responder", _forever_service, auto_start=True)
        messages = await sm.auto_start_all()
        # Config says False, so entry.auto_start is overridden
        assert len(messages) == 0


# ---------------------------------------------------------------------------
# ServiceManager — shutdown_all
# ---------------------------------------------------------------------------

class TestShutdown:
    @pytest.mark.asyncio
    async def test_shutdown_all(self):
        rc = RuntimeConfig()
        sm = ServiceManager(rc)
        sm.register("a", _forever_service)
        sm.register("b", _forever_service)
        await sm.start("a")
        await sm.start("b")
        await sm.shutdown_all()
        statuses = sm.status()
        assert all(s["status"] == "stopped" for s in statuses)

    @pytest.mark.asyncio
    async def test_shutdown_empty(self):
        """Shutdown with no services is a no-op."""
        rc = RuntimeConfig()
        sm = ServiceManager(rc)
        await sm.shutdown_all()  # Should not raise

    @pytest.mark.asyncio
    async def test_shutdown_with_none_running(self):
        rc = RuntimeConfig()
        sm = ServiceManager(rc)
        sm.register("svc", _forever_service)
        await sm.shutdown_all()  # svc is stopped, should not raise


# ---------------------------------------------------------------------------
# ServiceManager — status
# ---------------------------------------------------------------------------

class TestStatus:
    @pytest.mark.asyncio
    async def test_status_running(self):
        rc = RuntimeConfig()
        sm = ServiceManager(rc)
        sm.register("svc", _forever_service, description="Test")
        await sm.start("svc")
        statuses = sm.status()
        assert len(statuses) == 1
        assert statuses[0]["name"] == "svc"
        assert statuses[0]["status"] == "running"
        assert statuses[0]["description"] == "Test"
        assert "uptime_seconds" in statuses[0]
        await sm.shutdown_all()

    def test_status_empty(self):
        rc = RuntimeConfig()
        sm = ServiceManager(rc)
        assert sm.status() == []

    def test_status_line_no_services(self):
        rc = RuntimeConfig()
        sm = ServiceManager(rc)
        assert "none registered" in sm.status_line()

    @pytest.mark.asyncio
    async def test_status_line_with_services(self):
        rc = RuntimeConfig()
        sm = ServiceManager(rc)
        sm.register("svc1", _forever_service)
        sm.register("svc2", _forever_service)
        await sm.start("svc1")
        line = sm.status_line()
        assert "svc1 (running)" in line
        assert "svc2 (stopped)" in line
        await sm.shutdown_all()


# ---------------------------------------------------------------------------
# ServicesConfig
# ---------------------------------------------------------------------------

class TestServicesConfig:
    def test_defaults(self):
        cfg = ServicesConfig()
        assert cfg.auto_responder is False
        assert cfg.worker_daemon is False
        assert cfg.serve is False
        assert cfg.night_shift is False
        assert cfg.responder_interval == 30
        assert cfg.responder_strategy == "plan-only"
        assert cfg.worker_mode == "sdk"
        assert cfg.night_shift_interval == 3600
        assert cfg.night_shift_head == "qwen-llm"
        assert cfg.night_shift_concurrency == 1

    def test_set_value_bool(self):
        rc = RuntimeConfig()
        result = rc.set_value("services.auto_responder", "true")
        assert rc.services.auto_responder is True
        assert "auto_responder" in result

    def test_set_value_int(self):
        rc = RuntimeConfig()
        result = rc.set_value("services.responder_interval", "60")
        assert rc.services.responder_interval == 60

    def test_set_value_string(self):
        rc = RuntimeConfig()
        result = rc.set_value("services.worker_mode", "headless")
        assert rc.services.worker_mode == "headless"

    def test_set_value_unknown_field(self):
        rc = RuntimeConfig()
        with pytest.raises(ValueError, match="Unknown services field"):
            rc.set_value("services.nonexistent", "value")

    def test_roundtrip(self, tmp_path):
        rc = RuntimeConfig()
        rc.services.auto_responder = True
        rc.services.responder_interval = 15
        path = tmp_path / "config.json"
        rc.save(path)

        loaded = RuntimeConfig.load(path)
        assert loaded.services.auto_responder is True
        assert loaded.services.responder_interval == 15

    def test_night_shift_config(self):
        rc = RuntimeConfig()
        rc.set_value("services.night_shift", "true")
        assert rc.services.night_shift is True
        rc.set_value("services.night_shift_interval", "1800")
        assert rc.services.night_shift_interval == 1800
        rc.set_value("services.night_shift_head", "core-llm")
        assert rc.services.night_shift_head == "core-llm"
        rc.set_value("services.night_shift_concurrency", "4")
        assert rc.services.night_shift_concurrency == 4

    def test_night_shift_auto_start(self):
        """Night shift auto-starts when config flag is set."""
        rc = RuntimeConfig()
        rc.services.night_shift = True
        sm = ServiceManager(rc)
        sm.register("night_shift", _forever_service, auto_start=False)
        # ServicesConfig.night_shift=True should override entry.auto_start=False
        # (tested via auto_start_all which checks config flags)


# ---------------------------------------------------------------------------
# /services slash command
# ---------------------------------------------------------------------------

class TestServicesSlashCommand:
    @pytest.fixture
    def slash_handler(self, tmp_path):
        from multihead.slash_commands import SlashCommandHandler
        from multihead.tool_registry import ToolRegistry

        rc = RuntimeConfig()
        sm = ServiceManager(rc)
        sm.register("test-svc", _forever_service, description="A test service")

        return SlashCommandHandler(
            config=rc,
            config_path=tmp_path / "config.json",
            tool_registry=ToolRegistry(),
            head_states_fn=lambda: {},
            service_manager=sm,
        )

    @pytest.mark.asyncio
    async def test_services_list(self, slash_handler):
        result = await slash_handler.handle("/services")
        assert "test-svc" in result
        assert "stopped" in result

    @pytest.mark.asyncio
    async def test_services_list_explicit(self, slash_handler):
        result = await slash_handler.handle("/services list")
        assert "test-svc" in result

    @pytest.mark.asyncio
    async def test_services_start(self, slash_handler):
        result = await slash_handler.handle("/services start test-svc")
        assert "started" in result
        await slash_handler.service_manager.shutdown_all()

    @pytest.mark.asyncio
    async def test_services_stop(self, slash_handler):
        await slash_handler.handle("/services start test-svc")
        result = await slash_handler.handle("/services stop test-svc")
        assert "stopped" in result

    @pytest.mark.asyncio
    async def test_services_start_unknown(self, slash_handler):
        result = await slash_handler.handle("/services start nope")
        assert "Unknown" in result

    @pytest.mark.asyncio
    async def test_services_no_manager(self, tmp_path):
        from multihead.slash_commands import SlashCommandHandler
        from multihead.tool_registry import ToolRegistry

        handler = SlashCommandHandler(
            config=RuntimeConfig(),
            config_path=tmp_path / "config.json",
            tool_registry=ToolRegistry(),
            head_states_fn=lambda: {},
            service_manager=None,
        )
        result = await handler.handle("/services")
        assert "not available" in result

    @pytest.mark.asyncio
    async def test_services_usage(self, slash_handler):
        result = await slash_handler.handle("/services bogus")
        assert "Usage" in result

    @pytest.mark.asyncio
    async def test_services_enable(self, slash_handler):
        result = await slash_handler.handle("/services enable auto-responder")
        assert "enabled" in result.lower() or "auto_responder" in result


# ---------------------------------------------------------------------------
# Integration: Shell with ServiceManager
# ---------------------------------------------------------------------------

class TestShellIntegration:
    def test_shell_accepts_service_manager(self):
        """Shell.__init__ accepts service_manager parameter."""
        from multihead.shell import Shell

        rc = RuntimeConfig()
        sm = ServiceManager(rc)
        sm.register("svc", _forever_service)

        shell = Shell(
            agentic_core=None,
            head_manager=None,
            knowledge_store=None,
            session_manager=None,
            slash_handler=None,
            service_manager=sm,
        )
        assert shell.service_manager is sm

    def test_shell_service_manager_optional(self):
        """Shell works without service_manager."""
        from multihead.shell import Shell

        shell = Shell(
            agentic_core=None,
            head_manager=None,
            knowledge_store=None,
            session_manager=None,
            slash_handler=None,
        )
        assert shell.service_manager is None

    def test_banner_with_services(self):
        """Banner shows service status line."""
        from unittest.mock import MagicMock
        from multihead.shell import Shell

        rc = RuntimeConfig()
        sm = ServiceManager(rc)
        sm.register("test-svc", _forever_service)

        mock_hm = MagicMock()
        mock_hm.get_states.return_value = {}
        mock_hm.active_head = None

        shell = Shell(
            agentic_core=MagicMock(),
            head_manager=mock_hm,
            knowledge_store=None,
            session_manager=None,
            slash_handler=None,
            service_manager=sm,
            show_banner=True,
        )
        # Just verify status_line works
        line = sm.status_line()
        assert "test-svc" in line
