"""Comprehensive CLI test suite for MultiHead commands."""

from __future__ import annotations

import pytest
from click.testing import CliRunner

from multihead.cli import main


@pytest.fixture
def runner():
    """Create a Click CLI test runner."""
    return CliRunner()


@pytest.fixture
def tmp_data_dir(tmp_path):
    """Create a temporary data directory with required structure."""
    (tmp_path / "runs").mkdir()
    (tmp_path / "artifacts").mkdir()
    return tmp_path


@pytest.fixture
def tmp_config_dir(tmp_path):
    """Create a temporary config directory with heads.yaml."""
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    heads_yaml = config_dir / "heads.yaml"
    heads_yaml.write_text("""
heads:
  - head_id: mock-llm
    name: Mock LLM
    adapter: mock
    model: mock-v1
    kind: llm
    gpu_required: false

  - head_id: mock-vlm
    name: Mock VLM
    adapter: mock
    model: mock-v1
    kind: vlm
    gpu_required: false
""")
    return config_dir


# -----------------------------------------------------------------------
# Main group
# -----------------------------------------------------------------------


class TestMainGroup:
    """Test the main CLI group."""

    def test_help_output(self, runner):
        """multihead --help should show available commands."""
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "MultiHead" in result.output
        assert "run" in result.output
        assert "solve" in result.output
        assert "heads" in result.output

    def test_no_subcommand_shows_help(self, runner, tmp_data_dir):
        """Invoking without subcommand should show help."""
        result = runner.invoke(main, ["--data-dir", str(tmp_data_dir)])
        assert result.exit_code == 0
        assert "MultiHead" in result.output

    def test_debug_flag_accepted(self, runner, tmp_data_dir):
        """--debug flag should be accepted."""
        result = runner.invoke(main, ["--debug", "--data-dir", str(tmp_data_dir), "--help"])
        assert result.exit_code == 0

    def test_data_dir_from_env(self, runner, tmp_data_dir):
        """MULTIHEAD_DATA_DIR env var should set data directory."""
        result = runner.invoke(
            main, ["--help"],
            env={"MULTIHEAD_DATA_DIR": str(tmp_data_dir)},
        )
        assert result.exit_code == 0


# -----------------------------------------------------------------------
# heads command
# -----------------------------------------------------------------------


class TestHeadsCommand:
    """Test the heads command."""

    def test_heads_help(self, runner):
        """multihead heads --help should work."""
        result = runner.invoke(main, ["heads", "--help"])
        assert result.exit_code == 0
        assert "List registered heads" in result.output

    def test_heads_lists_manifests(self, runner, tmp_data_dir, tmp_config_dir):
        """Should list heads from config."""
        result = runner.invoke(main, [
            "--data-dir", str(tmp_data_dir),
            "--config-dir", str(tmp_config_dir),
            "heads",
        ])
        assert result.exit_code == 0
        assert "mock-llm" in result.output
        assert "mock-vlm" in result.output
        assert "Mock LLM" in result.output

    def test_heads_no_config(self, runner, tmp_data_dir, tmp_path):
        """Should handle missing heads.yaml gracefully."""
        empty_config = tmp_path / "empty_config"
        empty_config.mkdir()
        result = runner.invoke(main, [
            "--data-dir", str(tmp_data_dir),
            "--config-dir", str(empty_config),
            "heads",
        ])
        assert result.exit_code == 0
        assert "No heads registered" in result.output


# -----------------------------------------------------------------------
# status command
# -----------------------------------------------------------------------


class TestStatusCommand:
    """Test the status command."""

    def test_status_help(self, runner):
        result = runner.invoke(main, ["status", "--help"])
        assert result.exit_code == 0
        assert "Show run status" in result.output

    def test_status_no_runs(self, runner, tmp_data_dir, tmp_config_dir):
        """Should show 'No runs found' when no runs exist."""
        result = runner.invoke(main, [
            "--data-dir", str(tmp_data_dir),
            "--config-dir", str(tmp_config_dir),
            "status",
        ])
        assert result.exit_code == 0
        assert "No runs found" in result.output

    def test_status_invalid_run_id(self, runner, tmp_data_dir, tmp_config_dir):
        """Should handle non-existent run_id."""
        result = runner.invoke(main, [
            "--data-dir", str(tmp_data_dir),
            "--config-dir", str(tmp_config_dir),
            "status", "nonexistent-run",
        ])
        # Should fail (sys.exit(1))
        assert result.exit_code != 0


# -----------------------------------------------------------------------
# knowledge group
# -----------------------------------------------------------------------


class TestKnowledgeGroup:
    """Test knowledge subcommands."""

    def test_knowledge_help(self, runner):
        result = runner.invoke(main, ["knowledge", "--help"])
        assert result.exit_code == 0
        assert "claims" in result.output
        assert "events" in result.output

    def test_knowledge_claims_empty(self, runner, tmp_data_dir, tmp_config_dir):
        """Should handle empty knowledge store."""
        result = runner.invoke(main, [
            "--data-dir", str(tmp_data_dir),
            "--config-dir", str(tmp_config_dir),
            "knowledge", "claims",
        ])
        assert result.exit_code == 0
        assert "No claims found" in result.output

    def test_knowledge_claims_with_limit(self, runner, tmp_data_dir, tmp_config_dir):
        """--limit option should be accepted."""
        result = runner.invoke(main, [
            "--data-dir", str(tmp_data_dir),
            "--config-dir", str(tmp_config_dir),
            "knowledge", "claims", "--limit", "5",
        ])
        assert result.exit_code == 0

    def test_knowledge_claims_with_status_filter(self, runner, tmp_data_dir, tmp_config_dir):
        """--status filter should be accepted."""
        result = runner.invoke(main, [
            "--data-dir", str(tmp_data_dir),
            "--config-dir", str(tmp_config_dir),
            "knowledge", "claims", "--status", "accepted",
        ])
        assert result.exit_code == 0

    def test_knowledge_events_empty(self, runner, tmp_data_dir, tmp_config_dir):
        """Should handle empty events store."""
        result = runner.invoke(main, [
            "--data-dir", str(tmp_data_dir),
            "--config-dir", str(tmp_config_dir),
            "knowledge", "events",
        ])
        assert result.exit_code == 0
        assert "No events found" in result.output


# -----------------------------------------------------------------------
# consensus group
# -----------------------------------------------------------------------


class TestConsensusGroup:
    """Test consensus subcommands."""

    def test_consensus_help(self, runner):
        result = runner.invoke(main, ["consensus", "--help"])
        assert result.exit_code == 0
        assert "strategies" in result.output
        assert "test" in result.output

    def test_consensus_strategies(self, runner):
        """Should list all consensus strategies."""
        result = runner.invoke(main, ["consensus", "strategies"])
        assert result.exit_code == 0
        assert "majority" in result.output
        assert "weighted" in result.output
        assert "unanimous" in result.output
        assert "threshold" in result.output


# -----------------------------------------------------------------------
# packs group
# -----------------------------------------------------------------------


class TestPacksGroup:
    """Test packs subcommands."""

    def test_packs_help(self, runner):
        result = runner.invoke(main, ["packs", "--help"])
        assert result.exit_code == 0
        assert "list" in result.output
        assert "build" in result.output
        assert "show" in result.output

    def test_packs_list_empty(self, runner, tmp_data_dir, tmp_config_dir):
        """Should handle empty packs."""
        result = runner.invoke(main, [
            "--data-dir", str(tmp_data_dir),
            "--config-dir", str(tmp_config_dir),
            "packs", "list",
        ])
        assert result.exit_code == 0
        assert "No packs" in result.output


# -----------------------------------------------------------------------
# narrative group
# -----------------------------------------------------------------------


class TestNarrativeGroup:
    """Test narrative subcommands."""

    def test_narrative_help(self, runner):
        result = runner.invoke(main, ["narrative", "--help"])
        assert result.exit_code == 0
        assert "ingest" in result.output
        assert "fuse" in result.output
        assert "status" in result.output

    def test_narrative_ingest_requires_source(self, runner, tmp_data_dir, tmp_config_dir):
        """narrative ingest should require --source."""
        result = runner.invoke(main, [
            "--data-dir", str(tmp_data_dir),
            "--config-dir", str(tmp_config_dir),
            "narrative", "ingest",
        ])
        assert result.exit_code != 0

    def test_narrative_status(self, runner, tmp_data_dir, tmp_config_dir):
        """narrative status should work on empty store."""
        result = runner.invoke(main, [
            "--data-dir", str(tmp_data_dir),
            "--config-dir", str(tmp_config_dir),
            "narrative", "status",
        ])
        assert result.exit_code == 0
        assert "Narrative Pipeline Status" in result.output


# -----------------------------------------------------------------------
# mesh group
# -----------------------------------------------------------------------


class TestMeshGroup:
    """Test mesh subcommands."""

    def test_mesh_help(self, runner):
        result = runner.invoke(main, ["mesh", "--help"])
        assert result.exit_code == 0
        assert "discover" in result.output
        assert "status" in result.output
        assert "list-peers" in result.output


# -----------------------------------------------------------------------
# recipes group
# -----------------------------------------------------------------------


class TestRecipesGroup:
    """Test recipes subcommands."""

    def test_recipes_help(self, runner):
        result = runner.invoke(main, ["recipes", "--help"])
        assert result.exit_code == 0
        assert "learn" in result.output


# -----------------------------------------------------------------------
# run command
# -----------------------------------------------------------------------


class TestRunCommand:
    """Test the run command."""

    def test_run_help(self, runner):
        result = runner.invoke(main, ["run", "--help"])
        assert result.exit_code == 0
        assert "RECIPE" in result.output

    def test_run_missing_recipe(self, runner, tmp_data_dir, tmp_config_dir):
        """Should fail if recipe file doesn't exist."""
        result = runner.invoke(main, [
            "--data-dir", str(tmp_data_dir),
            "--config-dir", str(tmp_config_dir),
            "run", "nonexistent_recipe",
        ])
        assert result.exit_code != 0


# -----------------------------------------------------------------------
# solve command
# -----------------------------------------------------------------------


class TestSolveCommand:
    """Test the solve command."""

    def test_solve_help(self, runner):
        result = runner.invoke(main, ["solve", "--help"])
        assert result.exit_code == 0
        assert "--strategy" in result.output
        assert "--gh-issue" in result.output
        assert "--gh-issue" in result.output
        assert "--auto-approve" in result.output

    def test_solve_requires_task_or_gh_issue(self, runner, tmp_data_dir, tmp_config_dir):
        """Should fail without task or --gh-issue."""
        result = runner.invoke(main, [
            "--data-dir", str(tmp_data_dir),
            "--config-dir", str(tmp_config_dir),
            "solve",
        ])
        assert result.exit_code != 0

    def test_solve_strategy_choices(self, runner):
        """Should accept valid strategy choices."""
        result = runner.invoke(main, ["solve", "--help"])
        assert "majority" in result.output
        assert "first_to_ahead" in result.output


# -----------------------------------------------------------------------
# serve command
# -----------------------------------------------------------------------


class TestServeCommand:
    """Test the serve command."""

    def test_serve_help(self, runner):
        result = runner.invoke(main, ["serve", "--help"])
        assert result.exit_code == 0
        assert "--host" in result.output
        assert "--port" in result.output


# -----------------------------------------------------------------------
# mcp command
# -----------------------------------------------------------------------


class TestMcpCommand:
    """Test the MCP command."""

    def test_mcp_help(self, runner):
        result = runner.invoke(main, ["mcp", "--help"])
        assert result.exit_code == 0
        assert "--transport" in result.output
        assert "stdio" in result.output


# -----------------------------------------------------------------------
# export / import commands
# -----------------------------------------------------------------------


class TestExportImportCommands:
    """Test export and import commands."""

    def test_export_help(self, runner):
        result = runner.invoke(main, ["export", "--help"])
        assert result.exit_code == 0
        assert "--output" in result.output
        assert "--project" in result.output

    def test_import_help(self, runner):
        result = runner.invoke(main, ["import", "--help"])
        assert result.exit_code == 0
        assert "ZIP_PATH" in result.output


# -----------------------------------------------------------------------
# doctor command
# -----------------------------------------------------------------------


class TestDoctorCommand:
    """Test the doctor command."""

    def test_doctor_help(self, runner):
        result = runner.invoke(main, ["doctor", "--help"])
        assert result.exit_code == 0
        assert "diagnostic" in result.output.lower()

    def test_doctor_runs(self, runner, tmp_data_dir, tmp_config_dir):
        """Doctor should run diagnostics without crashing."""
        result = runner.invoke(main, [
            "--data-dir", str(tmp_data_dir),
            "--config-dir", str(tmp_config_dir),
            "doctor",
        ])
        assert result.exit_code == 0
        assert "checks passed" in result.output or "OK" in result.output


# -----------------------------------------------------------------------
# init command
# -----------------------------------------------------------------------


class TestInitCommand:
    """Test the init command."""

    def test_init_help(self, runner):
        result = runner.invoke(main, ["init", "--help"])
        assert result.exit_code == 0
        assert "--auto" in result.output
        assert "--mesh" in result.output


# -----------------------------------------------------------------------
# inspect command
# -----------------------------------------------------------------------


class TestInspectCommand:
    """Test the inspect command."""

    def test_inspect_help(self, runner):
        result = runner.invoke(main, ["inspect", "--help"])
        assert result.exit_code == 0
        assert "RUN_ID" in result.output


# -----------------------------------------------------------------------
# nightshift group
# -----------------------------------------------------------------------


class TestNightshiftGroup:
    """Test nightshift subcommands."""

    def test_nightshift_help(self, runner):
        result = runner.invoke(main, ["nightshift", "--help"])
        assert result.exit_code == 0
        assert "run" in result.output


# -----------------------------------------------------------------------
# shell command
# -----------------------------------------------------------------------


class TestShellCommand:
    """Test the shell command."""

    def test_shell_help(self, runner):
        result = runner.invoke(main, ["shell", "--help"])
        assert result.exit_code == 0
        assert "--head" in result.output
        assert "--session" in result.output
        assert "--no-banner" in result.output
        assert "Interactive agent terminal" in result.output

    def test_shell_in_main_help(self, runner):
        """Shell should appear in main help listing."""
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "shell" in result.output

    def test_shell_invalid_head(self, runner, tmp_data_dir, tmp_config_dir):
        """Should fail gracefully with an invalid head ID."""
        result = runner.invoke(main, [
            "--data-dir", str(tmp_data_dir),
            "--config-dir", str(tmp_config_dir),
            "shell", "--head", "nonexistent-head-xyz",
        ])
        # Should print error about head not found (not crash)
        assert "not found" in result.output or result.exit_code == 0


class TestDaemonCommand:
    """Test the daemon command."""

    def test_daemon_help(self, runner):
        result = runner.invoke(main, ["daemon", "--help"])
        assert result.exit_code == 0
        assert "--service" in result.output
        assert "--head" in result.output
        assert "--log-level" in result.output
        assert "background services" in result.output

    def test_daemon_in_main_help(self, runner):
        """Daemon should appear in main help listing."""
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "daemon" in result.output

    def test_daemon_invalid_head(self, runner, tmp_data_dir, tmp_config_dir):
        """Should fail gracefully with an invalid head ID."""
        result = runner.invoke(main, [
            "--data-dir", str(tmp_data_dir),
            "--config-dir", str(tmp_config_dir),
            "daemon", "--head", "nonexistent-head-xyz",
        ])
        assert "not found" in result.output or result.exit_code == 0

    def test_daemon_invalid_service(self, runner, tmp_data_dir, tmp_config_dir):
        """Should report error for unknown service names."""
        result = runner.invoke(main, [
            "--data-dir", str(tmp_data_dir),
            "--config-dir", str(tmp_config_dir),
            "daemon", "-s", "nonexistent-service",
        ])
        assert "No valid services" in result.output or result.exit_code == 0


# -----------------------------------------------------------------------
# Helper functions
# -----------------------------------------------------------------------


class TestHelperFunctions:
    """Test CLI helper functions."""

    def test_get_settings_defaults(self):
        """_get_settings with no args should return valid Settings."""
        from multihead.cli import _get_settings
        settings = _get_settings()
        assert settings.data_dir is not None
        assert settings.config_dir is not None

    def test_get_settings_custom_dirs(self, tmp_path):
        """_get_settings should accept custom directories."""
        from multihead.cli import _get_settings
        data = tmp_path / "data"
        config = tmp_path / "config"
        settings = _get_settings(str(data), str(config))
        assert settings.data_dir == data
        assert settings.config_dir == config

    def test_build_orchestrator(self, tmp_data_dir, tmp_config_dir):
        """_build_orchestrator should return orchestrator and head_manager."""
        from multihead.cli import _build_orchestrator, _get_settings
        settings = _get_settings(str(tmp_data_dir), str(tmp_config_dir))
        orch, hm = _build_orchestrator(settings)
        from multihead.orchestrator import Orchestrator
        from multihead.head_manager import HeadManager
        assert isinstance(orch, Orchestrator)
        assert isinstance(hm, HeadManager)

    def test_build_knowledge_deps(self, tmp_data_dir, tmp_config_dir):
        """_build_knowledge_deps should return knowledge layer tuple."""
        from multihead.cli import _build_knowledge_deps, _get_settings
        settings = _get_settings(str(tmp_data_dir), str(tmp_config_dir))
        ks, rs, artifact_store, pb = _build_knowledge_deps(settings)
        from multihead.knowledge_store import KnowledgeStore
        assert isinstance(ks, KnowledgeStore)

    def test_parse_since_iso(self):
        """_parse_since should parse ISO datetime."""
        from multihead.cli import _parse_since
        result = _parse_since("2026-01-01T00:00:00")
        assert result.year == 2026
        assert result.month == 1

    def test_parse_since_relative_hours(self):
        """_parse_since should parse relative hours."""
        from multihead.cli import _parse_since
        from datetime import datetime, timezone
        result = _parse_since("24h")
        now = datetime.now(timezone.utc)
        diff = (
            now - result.replace(tzinfo=timezone.utc)
            if result.tzinfo is None
            else now - result
        )
        assert 23 * 3600 < diff.total_seconds() < 25 * 3600

    def test_parse_since_relative_days(self):
        """_parse_since should parse relative days."""
        from multihead.cli import _parse_since
        from datetime import datetime, timezone
        result = _parse_since("7d")
        now = datetime.now(timezone.utc)
        diff = now - result.replace(tzinfo=timezone.utc) if result.tzinfo is None else now - result
        assert 6 * 86400 < diff.total_seconds() < 8 * 86400

    def test_setup_logging(self, tmp_data_dir):
        """_setup_logging should not crash."""
        from multihead.cli import _setup_logging, _get_settings
        settings = _get_settings(str(tmp_data_dir))
        _setup_logging(settings, debug=False)
        log_file = tmp_data_dir / "multihead.log"
        assert log_file.exists()


# -----------------------------------------------------------------------
# Integration tests with data
# -----------------------------------------------------------------------


class TestKnowledgeClaimsWithData:
    """Test knowledge commands with actual claim data."""

    def test_knowledge_claims_shows_data(self, runner, tmp_data_dir, tmp_config_dir):
        """Should display claims when data exists."""
        from multihead.knowledge_store import KnowledgeStore
        from multihead.knowledge_models import (
            Claim, ClaimType, ClaimStatus, ClaimScope, ScopeType,
            ClaimCanonical, EntityRef, ValueObject, Provenance,
        )
        from datetime import datetime, timezone

        ks = KnowledgeStore(tmp_data_dir / "knowledge.db")
        ks.insert_claim(Claim(
            claim_type=ClaimType.FACT,
            claim_status=ClaimStatus.ACCEPTED,
            scope=ClaimScope(
                scope_type=ScopeType.PROJECT, scope_id="test",
                visibility="project", valid_from=datetime.now(timezone.utc),
            ),
            canonical=ClaimCanonical(
                claim_key="test.key",
                subject=EntityRef(entity_type="module", entity_id="auth"),
                predicate="status",
                object=ValueObject(value_type="string", value="ok"),
            ),
            statement="Auth module works correctly",
            confidence=0.95,
            provenance=Provenance(produced_by={"id": "test"}),
        ))

        result = runner.invoke(main, [
            "--data-dir", str(tmp_data_dir),
            "--config-dir", str(tmp_config_dir),
            "knowledge", "claims",
        ])
        assert result.exit_code == 0
        assert "Auth module works" in result.output
        assert "0.95" in result.output


class TestKBCommand:
    """Tests for `multihead kb` shortcut and `knowledge search`."""

    def _insert_test_claims(self, ks):
        from multihead.knowledge_models import (
            Claim, ClaimType, ClaimStatus, ClaimScope, ScopeType,
            ClaimCanonical, EntityRef, ValueObject, Provenance,
        )
        from datetime import datetime, timezone

        for i, (key, stmt, scope_id) in enumerate([
            ("test.portability", "Friends can install MultiHead easily", "multihead"),
            ("test.consensus", "Consensus uses majority voting strategy", "multihead"),
            ("test.balloon", "Balloon layout handles text overflow", "h2v"),
        ]):
            ks.insert_claim(Claim(
                claim_type=ClaimType.FACT,
                claim_status=ClaimStatus.ACCEPTED,
                scope=ClaimScope(
                    scope_type=ScopeType.PROJECT, scope_id=scope_id,
                    visibility="project", valid_from=datetime.now(timezone.utc),
                ),
                canonical=ClaimCanonical(
                    claim_key=key,
                    subject=EntityRef(entity_type="module", entity_id=f"m{i}"),
                    predicate="status",
                    object=ValueObject(value_type="string", value="ok"),
                ),
                statement=stmt,
                confidence=0.9,
                provenance=Provenance(produced_by={"id": "test"}),
            ))

    def test_kb_help(self, runner):
        result = runner.invoke(main, ["kb", "--help"])
        assert result.exit_code == 0
        assert "knowledge search" in result.output.lower() or "shortcut" in result.output.lower()

    def test_kb_search_fts(self, runner, tmp_data_dir, tmp_config_dir):
        from multihead.knowledge_store import KnowledgeStore
        ks = KnowledgeStore(tmp_data_dir / "knowledge.db")
        self._insert_test_claims(ks)

        result = runner.invoke(main, [
            "--data-dir", str(tmp_data_dir),
            "--config-dir", str(tmp_config_dir),
            "kb", "consensus",
        ])
        assert result.exit_code == 0
        assert "majority voting" in result.output

    def test_kb_no_results(self, runner, tmp_data_dir, tmp_config_dir):
        from multihead.knowledge_store import KnowledgeStore
        KnowledgeStore(tmp_data_dir / "knowledge.db")  # init empty db

        result = runner.invoke(main, [
            "--data-dir", str(tmp_data_dir),
            "--config-dir", str(tmp_config_dir),
            "kb", "xyznonexistent",
        ])
        assert result.exit_code == 0
        assert "No claims" in result.output

    def test_knowledge_search_subcommand(self, runner, tmp_data_dir, tmp_config_dir):
        from multihead.knowledge_store import KnowledgeStore
        ks = KnowledgeStore(tmp_data_dir / "knowledge.db")
        self._insert_test_claims(ks)

        result = runner.invoke(main, [
            "--data-dir", str(tmp_data_dir),
            "--config-dir", str(tmp_config_dir),
            "knowledge", "search", "balloon",
        ])
        assert result.exit_code == 0
        assert "text overflow" in result.output

    def test_kb_with_limit(self, runner, tmp_data_dir, tmp_config_dir):
        from multihead.knowledge_store import KnowledgeStore
        ks = KnowledgeStore(tmp_data_dir / "knowledge.db")
        self._insert_test_claims(ks)

        result = runner.invoke(main, [
            "--data-dir", str(tmp_data_dir),
            "--config-dir", str(tmp_config_dir),
            "kb", "install consensus balloon", "-n", "1",
        ])
        assert result.exit_code == 0
        assert "1 results" in result.output


class TestAuthCommand:
    """Tests for `multihead auth status`."""

    def test_auth_help(self, runner):
        result = runner.invoke(main, ["auth", "--help"])
        assert result.exit_code == 0
        assert "status" in result.output

    def test_auth_status_no_env(self, runner, tmp_data_dir, tmp_config_dir, monkeypatch):
        monkeypatch.delenv("ACP_URL", raising=False)
        monkeypatch.delenv("ACP_SESSION_KEY", raising=False)
        monkeypatch.delenv("ACP_API_KEY", raising=False)
        monkeypatch.delenv("ACP_CLAUDE_SESSION_KEY", raising=False)
        monkeypatch.delenv("ACP_AGENT_ID", raising=False)

        result = runner.invoke(main, [
            "--data-dir", str(tmp_data_dir),
            "--config-dir", str(tmp_config_dir),
            "auth", "status",
        ])
        assert result.exit_code == 0
        assert "Auth Status" in result.output
        assert "not set" in result.output

    def test_auth_status_with_agent_id(self, runner, tmp_data_dir, tmp_config_dir, monkeypatch):
        monkeypatch.setenv("ACP_AGENT_ID", "my-test-agent")
        monkeypatch.delenv("ACP_URL", raising=False)
        monkeypatch.delenv("ACP_SESSION_KEY", raising=False)
        monkeypatch.delenv("ACP_API_KEY", raising=False)
        monkeypatch.delenv("ACP_CLAUDE_SESSION_KEY", raising=False)

        result = runner.invoke(main, [
            "--data-dir", str(tmp_data_dir),
            "--config-dir", str(tmp_config_dir),
            "auth", "status",
        ])
        assert result.exit_code == 0
        assert "my-test-agent" in result.output


class TestEmbeddingSearch:
    """Tests for embedding_search module."""

    def test_embedding_index_init(self, tmp_path):
        from multihead.embedding_search import EmbeddingIndex
        idx = EmbeddingIndex(tmp_path / "test.db", cache_dir=tmp_path / "cache")
        assert idx.indexed_count == 0
        assert not idx._built

    def test_embedding_index_empty_db(self, tmp_path):
        """Build on empty DB should return 0."""
        from multihead.embedding_search import EmbeddingIndex
        from multihead.knowledge_store import KnowledgeStore

        ks = KnowledgeStore(tmp_path / "knowledge.db")
        idx = EmbeddingIndex(tmp_path / "knowledge.db", cache_dir=tmp_path / "cache")
        count = idx.build()
        assert count == 0

    def test_embedding_index_search_empty(self, tmp_path):
        """Search on empty index returns empty list."""
        from multihead.embedding_search import EmbeddingIndex
        from multihead.knowledge_store import KnowledgeStore

        KnowledgeStore(tmp_path / "knowledge.db")
        idx = EmbeddingIndex(tmp_path / "knowledge.db", cache_dir=tmp_path / "cache")
        idx.build()
        results = idx.search("anything")
        assert results == []

    def test_embedding_index_build_and_search(self, tmp_path):
        """Full build + search with real sentence-transformers."""
        pytest.importorskip("sentence_transformers")

        from multihead.embedding_search import EmbeddingIndex
        from multihead.knowledge_store import KnowledgeStore
        from multihead.knowledge_models import (
            Claim, ClaimType, ClaimStatus, ClaimScope, ScopeType,
            ClaimCanonical, EntityRef, ValueObject, Provenance,
        )
        from datetime import datetime, timezone

        ks = KnowledgeStore(tmp_path / "knowledge.db")
        # Insert claims with varied topics
        for key, stmt in [
            ("test.install", "Users can install the software on their machines easily"),
            ("test.gpu", "CUDA support requires an NVIDIA GPU with 8GB VRAM"),
            ("test.api", "The REST API serves endpoints on port 7337"),
        ]:
            ks.insert_claim(Claim(
                claim_type=ClaimType.FACT,
                claim_status=ClaimStatus.ACCEPTED,
                scope=ClaimScope(
                    scope_type=ScopeType.PROJECT, scope_id="test",
                    visibility="project", valid_from=datetime.now(timezone.utc),
                ),
                canonical=ClaimCanonical(
                    claim_key=key,
                    subject=EntityRef(entity_type="module", entity_id="m"),
                    predicate="status",
                    object=ValueObject(value_type="string", value="ok"),
                ),
                statement=stmt,
                confidence=0.9,
                provenance=Provenance(produced_by={"id": "test"}),
            ))

        idx = EmbeddingIndex(tmp_path / "knowledge.db", cache_dir=tmp_path / "cache")
        count = idx.build()
        assert count == 3

        # Semantic search: "portability" should find "install" claim
        results = idx.search("portability and setup", limit=3)
        assert len(results) > 0
        # The install-related claim should rank highest
        assert "install" in results[0][1].lower()

        # Check cache was saved
        assert (tmp_path / "cache" / "claim_embeddings.npz").exists()

    def test_embedding_index_cache_reuse(self, tmp_path):
        """Second build should reuse cache."""
        pytest.importorskip("sentence_transformers")

        from multihead.embedding_search import EmbeddingIndex
        from multihead.knowledge_store import KnowledgeStore
        from multihead.knowledge_models import (
            Claim, ClaimType, ClaimStatus, ClaimScope, ScopeType,
            ClaimCanonical, EntityRef, ValueObject, Provenance,
        )
        from datetime import datetime, timezone

        ks = KnowledgeStore(tmp_path / "knowledge.db")
        ks.insert_claim(Claim(
            claim_type=ClaimType.FACT,
            claim_status=ClaimStatus.ACCEPTED,
            scope=ClaimScope(
                scope_type=ScopeType.PROJECT, scope_id="test",
                visibility="project", valid_from=datetime.now(timezone.utc),
            ),
            canonical=ClaimCanonical(
                claim_key="test.one",
                subject=EntityRef(entity_type="module", entity_id="m"),
                predicate="status",
                object=ValueObject(value_type="string", value="ok"),
            ),
            statement="This is a test claim about knowledge storage",
            confidence=0.9,
            provenance=Provenance(produced_by={"id": "test"}),
        ))

        idx1 = EmbeddingIndex(tmp_path / "knowledge.db", cache_dir=tmp_path / "cache")
        idx1.build()
        assert idx1.indexed_count == 1

        # Second index should load from cache without re-encoding
        idx2 = EmbeddingIndex(tmp_path / "knowledge.db", cache_dir=tmp_path / "cache")
        count = idx2.build()
        assert count == 1
        assert idx2.indexed_count == 1
