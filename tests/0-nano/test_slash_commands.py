"""Tests for slash command handler."""

import pytest

from multihead.runtime_config import RuntimeConfig
from multihead.slash_commands import SlashCommandHandler
from multihead.tool_registry import ToolRegistry


@pytest.fixture
def handler(tmp_path):
    config = RuntimeConfig()
    config_path = tmp_path / "config.json"
    registry = ToolRegistry()
    head_states = lambda: {
        "mock-llm": {"name": "Mock LLM", "adapter": "mock", "state": "OFF"},
        "mock-vlm": {"name": "Mock VLM", "adapter": "mock", "state": "ACTIVE"},
    }
    return SlashCommandHandler(config, config_path, registry, head_states)


class TestSlashDetection:
    def test_is_slash_command(self):
        assert SlashCommandHandler.is_slash_command("/help")
        assert SlashCommandHandler.is_slash_command("/config show")
        assert not SlashCommandHandler.is_slash_command("hello")
        assert not SlashCommandHandler.is_slash_command("")


class TestHelp:
    @pytest.mark.asyncio
    async def test_help(self, handler):
        result = await handler.handle("/help")
        assert "/config" in result
        assert "/tools" in result
        assert "/heads" in result


class TestConfig:
    @pytest.mark.asyncio
    async def test_config_show(self, handler):
        result = await handler.handle("/config show")
        assert "Runtime Config" in result
        assert "web_tools_enabled" in result

    @pytest.mark.asyncio
    async def test_config_show_default(self, handler):
        result = await handler.handle("/config")
        assert "Runtime Config" in result

    @pytest.mark.asyncio
    async def test_config_show_has_sections(self, handler):
        """Rich tree display should have section headers."""
        result = await handler.handle("/config show")
        assert "General" in result
        assert "Generation" in result
        assert "Pipeline" in result
        assert "Services" in result

    @pytest.mark.asyncio
    async def test_config_show_expands_nested(self, handler):
        """Nested configs should be expanded, not raw dicts."""
        result = await handler.handle("/config show")
        assert "temperature" in result
        assert "max_tokens" in result
        # Conversation sub-config should be expanded
        assert "recent_count" in result

    @pytest.mark.asyncio
    async def test_config_show_color_booleans(self, handler):
        """Booleans should have color markup."""
        result = await handler.handle("/config show")
        # web_tools_enabled defaults to True
        assert "True" in result

    @pytest.mark.asyncio
    async def test_config_set(self, handler):
        result = await handler.handle("/config set generation.temperature 0.3")
        assert "0.3" in result
        assert handler.config.generation.temperature == 0.3

    @pytest.mark.asyncio
    async def test_config_set_invalid(self, handler):
        result = await handler.handle("/config set nonexistent value")
        assert "Error" in result

    @pytest.mark.asyncio
    async def test_config_usage(self, handler):
        result = await handler.handle("/config set")
        assert "Usage" in result

    @pytest.mark.asyncio
    async def test_config_interactive_cancel(self, handler):
        """Interactive mode should handle cancel (choice 0)."""
        from unittest.mock import patch
        with patch("multihead.slash_commands._config.Prompt") as MockPrompt:
            MockPrompt.ask.return_value = "0"
            result = await handler.handle("/config i")
        assert "cancelled" in result.lower()

    @pytest.mark.asyncio
    async def test_config_interactive_edits_value(self, handler):
        """Interactive mode should edit values and save."""
        from unittest.mock import patch
        handler.config_path = handler.config_path  # ensure it's set

        with patch("multihead.slash_commands._config.Prompt") as MockPrompt, \
             patch("multihead.slash_commands._config.Confirm") as MockConfirm, \
             patch("multihead.slash_commands._config.FloatPrompt") as MockFloat:
            # Section choice: "1" = General
            MockPrompt.ask.side_effect = [
                "1",          # section choice
                "keep_loaded",  # vram_core_mode (unchanged)
            ]
            MockConfirm.ask.side_effect = [
                False,  # web_tools_enabled: change from True to False
                True,   # strip_thinking: keep True
            ]

            result = await handler.handle("/config interactive")

        assert handler.config.web_tools_enabled is False
        assert "updated" in result.lower() or "change" in result.lower()

    @pytest.mark.asyncio
    async def test_config_interactive_alias(self, handler):
        """'/config i' should work as alias."""
        from unittest.mock import patch
        with patch("multihead.slash_commands._config.Prompt") as MockPrompt:
            MockPrompt.ask.return_value = "0"
            result = await handler.handle("/config i")
        assert "cancelled" in result.lower()

    @pytest.mark.asyncio
    async def test_config_interactive_in_help(self, handler):
        result = await handler.handle("/help")
        assert "/config interactive" in result


class TestTools:
    @pytest.mark.asyncio
    async def test_tools_list(self, handler):
        result = await handler.handle("/tools list")
        assert "files.read" in result
        assert "enabled" in result

    @pytest.mark.asyncio
    async def test_tools_disable(self, handler):
        result = await handler.handle("/tools disable files.read")
        assert "disabled" in result
        spec = handler.tools.get_spec("files.read")
        assert not spec.enabled

    @pytest.mark.asyncio
    async def test_tools_enable(self, handler):
        await handler.handle("/tools disable files.read")
        result = await handler.handle("/tools enable files.read")
        assert "enabled" in result
        spec = handler.tools.get_spec("files.read")
        assert spec.enabled

    @pytest.mark.asyncio
    async def test_tools_unknown(self, handler):
        result = await handler.handle("/tools disable nonexistent")
        assert "Unknown tool" in result

    @pytest.mark.asyncio
    async def test_tools_usage(self, handler):
        result = await handler.handle("/tools")
        assert "files.read" in result


class TestHeads:
    @pytest.mark.asyncio
    async def test_heads(self, handler):
        result = await handler.handle("/heads")
        assert "mock-llm" in result
        assert "mock-vlm" in result
        assert "ACTIVE" in result


class TestModel:
    @pytest.mark.asyncio
    async def test_model_no_shell(self, handler):
        """Without shell reference, returns error."""
        result = await handler.handle("/model")
        assert "not available" in result

    @pytest.mark.asyncio
    async def test_model_show(self, handler):
        """Show current model when adapter exists."""
        class FakeAdapter:
            _model = "claude-opus-4-6"
        class FakeShell:
            _claude_adapter = FakeAdapter()
        handler.shell = FakeShell()
        result = await handler.handle("/model")
        assert "claude-opus-4-6" in result

    @pytest.mark.asyncio
    async def test_model_switch_alias(self, handler):
        """Switch via alias like 'sonnet'."""
        class FakeAdapter:
            _model = "claude-opus-4-6"
        class FakeShell:
            _claude_adapter = FakeAdapter()
        handler.shell = FakeShell()
        result = await handler.handle("/model sonnet")
        assert "claude-sonnet-4-6" in result
        assert FakeShell._claude_adapter._model == "claude-sonnet-4-6"

    @pytest.mark.asyncio
    async def test_model_switch_full_id(self, handler):
        """Switch via full model ID."""
        class FakeAdapter:
            _model = "claude-opus-4-6"
        class FakeShell:
            _claude_adapter = FakeAdapter()
        handler.shell = FakeShell()
        result = await handler.handle("/model claude-haiku-4-5-20251001")
        assert "claude-haiku-4-5-20251001" in result
        assert FakeShell._claude_adapter._model == "claude-haiku-4-5-20251001"

    @pytest.mark.asyncio
    async def test_model_no_adapter(self, handler):
        """Error when no Claude adapter configured."""
        class FakeShell:
            _claude_adapter = None
        handler.shell = FakeShell()
        result = await handler.handle("/model sonnet")
        assert "No Claude adapter" in result


class TestVerbose:
    @pytest.mark.asyncio
    async def test_verbose_toggle(self, handler):
        """Toggle verbose on and off."""
        class FakeShell:
            def __init__(self):
                self._verbose = False
        shell = FakeShell()
        handler.shell = shell
        result = await handler.handle("/verbose")
        assert "ON" in result
        assert shell._verbose is True
        result = await handler.handle("/verbose")
        assert "OFF" in result
        assert shell._verbose is False


class TestDashboard:
    """Test /dashboard observability command."""

    @pytest.mark.asyncio
    async def test_dashboard_no_shell(self, handler):
        """Returns empty string when no shell attached."""
        handler.shell = None
        result = await handler.handle("/dashboard")
        assert result == ""

    @pytest.mark.asyncio
    async def test_dashboard_renders(self, handler):
        """Dashboard should render without errors with mock shell."""
        from unittest.mock import MagicMock

        mock_console = MagicMock()
        mock_shell = MagicMock()
        mock_shell.console = mock_console
        mock_shell._brain = "claude"
        mock_shell.service_manager = MagicMock()
        mock_shell.service_manager.shared_data = {
            "marketplace_stats": {
                "quotes_sent": 3,
                "contracts_won": 1,
                "contracts_done": 1,
                "trust_score": 0.92,
            },
        }
        handler.shell = mock_shell
        handler.service_manager = MagicMock()
        handler.service_manager.status.return_value = [
            {
                "name": "cloud-marketplace", "status": "running",
                "description": "Cloud", "uptime_seconds": 120,
            },
            {"name": "auto-responder", "status": "stopped", "description": "Responder"},
        ]
        handler.process_manager = MagicMock()
        handler.process_manager.list_processes.return_value = []

        result = await handler.handle("/dashboard")
        assert result == ""
        # Console.print should have been called multiple times (layout sections)
        assert mock_console.print.call_count >= 3

    @pytest.mark.asyncio
    async def test_dash_alias(self, handler):
        """The /dash alias should also work."""
        from unittest.mock import MagicMock

        mock_shell = MagicMock()
        mock_shell.console = MagicMock()
        mock_shell._brain = "local"
        mock_shell.service_manager = MagicMock()
        mock_shell.service_manager.shared_data = {}
        handler.shell = mock_shell
        handler.service_manager = None
        handler.process_manager = None

        result = await handler.handle("/dash")
        assert result == ""
        assert mock_shell.console.print.called

    @pytest.mark.asyncio
    async def test_dashboard_in_help(self, handler):
        """Dashboard should appear in /help output."""
        result = await handler.handle("/help")
        assert "/dashboard" in result

    @pytest.mark.asyncio
    async def test_dashboard_with_knowledge(self, handler):
        """Dashboard renders knowledge count when store available."""
        from unittest.mock import MagicMock

        mock_console = MagicMock()
        mock_shell = MagicMock()
        mock_shell.console = mock_console
        mock_shell._brain = "local"
        mock_shell.service_manager = MagicMock()
        mock_shell.service_manager.shared_data = {}
        handler.shell = mock_shell
        handler.service_manager = None
        handler.process_manager = None
        handler.knowledge_store = MagicMock()
        handler.knowledge_store.list_claims.return_value = [MagicMock()] * 100
        handler.knowledge_store.get_presence_peers.return_value = []

        result = await handler.handle("/dashboard")
        assert result == ""


class TestSolve:
    @pytest.mark.asyncio
    async def test_solve_no_args(self, handler):
        result = await handler.handle("/solve")
        assert "Usage" in result

    @pytest.mark.asyncio
    async def test_solve_no_head_manager(self, handler):
        handler.head_manager = None
        result = await handler.handle("/solve build a thing")
        assert "Head manager not available" in result

    @pytest.mark.asyncio
    async def test_solve_no_stores(self, handler):
        from unittest.mock import MagicMock
        handler.head_manager = MagicMock()
        handler.event_store = None
        handler.artifact_store = None
        result = await handler.handle("/solve build a thing")
        assert "Event/artifact store not available" in result

    @pytest.mark.asyncio
    async def test_solve_runs_pipeline(self, handler):
        from unittest.mock import AsyncMock, MagicMock, patch
        from multihead.solve_pipeline import SolveResult

        handler.head_manager = MagicMock()
        handler.event_store = MagicMock()
        handler.artifact_store = MagicMock()
        handler.runs_dir = "/tmp/test"
        handler.knowledge_store = MagicMock()

        mock_result = SolveResult(
            run_id="run-test-123",
            status="done",
            output="success",
            confidence=0.95,
            steps_total=3,
            steps_succeeded=3,
            steps_failed=0,
            duration_seconds=5.0,
            plan_steps=3,
            parallel_steps=1,
        )

        with patch("multihead.solve_pipeline.SolvePipeline") as MockPipeline:
            MockPipeline.return_value.solve = AsyncMock(return_value=mock_result)
            result = await handler.handle("/solve build a thing")

        assert "run-test-123" in result
        assert "done" in result
        assert "0.95" in result
        assert "3/3" in result

    @pytest.mark.asyncio
    async def test_solve_parses_flags(self, handler):
        from unittest.mock import AsyncMock, MagicMock, patch
        from multihead.solve_pipeline import SolveResult

        handler.head_manager = MagicMock()
        handler.event_store = MagicMock()
        handler.artifact_store = MagicMock()
        handler.runs_dir = "/tmp/test"

        mock_result = SolveResult(
            run_id="r1", status="done", output="ok", confidence=0.8,
        )

        with patch("multihead.solve_pipeline.SolvePipeline") as MockPipeline:
            MockPipeline.return_value.solve = AsyncMock(return_value=mock_result)
            result = await handler.handle("/solve --strategy majority --max-steps 5 my task")

        # Verify constraints were passed correctly
        call_kwargs = MockPipeline.return_value.solve.call_args
        constraints = call_kwargs.kwargs.get("constraints") or call_kwargs[1].get("constraints")
        assert constraints.strategy == "majority"
        assert constraints.max_steps == 5

    @pytest.mark.asyncio
    async def test_solve_invalid_max_steps(self, handler):
        from unittest.mock import MagicMock
        handler.head_manager = MagicMock()
        handler.event_store = MagicMock()
        handler.artifact_store = MagicMock()
        result = await handler.handle("/solve --max-steps abc build it")
        assert "Invalid --max-steps" in result

    @pytest.mark.asyncio
    async def test_solve_dry_run(self, handler):
        from unittest.mock import AsyncMock, MagicMock, patch
        from multihead.solve_pipeline import SolveResult

        handler.head_manager = MagicMock()
        handler.event_store = MagicMock()
        handler.artifact_store = MagicMock()
        handler.runs_dir = "/tmp/test"

        mock_result = SolveResult(
            run_id="dry-run-abc", status="dry_run", output="plan here",
            confidence=0.9, dry_run=True,
        )

        with patch("multihead.solve_pipeline.SolvePipeline") as MockPipeline:
            MockPipeline.return_value.solve = AsyncMock(return_value=mock_result)
            result = await handler.handle("/solve --dry-run my task")

        call_kwargs = MockPipeline.return_value.solve.call_args
        assert call_kwargs.kwargs.get("dry_run") is True

    @pytest.mark.asyncio
    async def test_solve_in_help(self, handler):
        result = await handler.handle("/help")
        assert "/solve" in result


class TestUnknown:
    @pytest.mark.asyncio
    async def test_unknown_command(self, handler):
        result = await handler.handle("/foobar")
        assert "Unknown command" in result
        assert "/help" in result
