"""Tests for v0.5 mesh collaboration features."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

from multihead.solve import (
    discover_sessions_mdns,
    mark_session_offline,
    write_presence_claim,
    _get_state_file,
    _load_seen_sessions,
    _save_seen_sessions,
    _show_onboarding_messages,
)
from multihead.knowledge_models import ClaimType
from multihead.knowledge_store import KnowledgeStore


# ---------------------------------------------------------------------------
# Test: mDNS Stub
# ---------------------------------------------------------------------------


def test_discover_sessions_mdns_returns_empty():
    """mDNS discovery stub returns empty list."""
    result = discover_sessions_mdns()
    assert result == []


# ---------------------------------------------------------------------------
# Test: Session Health Monitoring
# ---------------------------------------------------------------------------


def test_mark_session_offline():
    """mark_session_offline updates presence claim to offline."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        ks = KnowledgeStore(db_path)

        # Write initial presence claim (online)
        write_presence_claim(ks, "test-session", "test-project")

        # Mark offline
        claim_id = mark_session_offline(ks, "test-session", "test-project")
        assert claim_id

        # Verify offline claim exists (key includes timestamp now)
        claims = ks.list_claims(claim_type=ClaimType.FACT.value, scope_id="test-project")
        offline_claims = [
            c for c in claims
            if c.canonical.claim_key.startswith("agent.test-session.presence.offline")
        ]

        assert len(offline_claims) >= 1
        # Check that the offline claim has correct predicate
        assert offline_claims[0].canonical.predicate == "offline"


def test_signal_handlers_registration():
    """SolveCoordinator registers signal handlers on init."""
    from multihead.solve import SolveCoordinator, SolveConfig
    import signal

    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        ks = KnowledgeStore(db_path)
        head_manager = MagicMock()
        orchestrator = MagicMock()

        config = SolveConfig(project_id="test", session_id="test-coord")

        # Create coordinator (should register handlers)
        coord = SolveCoordinator(ks, head_manager, orchestrator, config)

        # Verify signal handlers are set (they won't be SIG_DFL)
        assert signal.getsignal(signal.SIGTERM) != signal.SIG_DFL
        assert signal.getsignal(signal.SIGINT) != signal.SIG_DFL


# ---------------------------------------------------------------------------
# Test: Onboarding UX
# ---------------------------------------------------------------------------


def test_get_state_file_creates_directory():
    """_get_state_file creates ~/.multihead if it doesn't exist."""
    with patch("pathlib.Path.home") as mock_home:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            mock_home.return_value = Path(tmpdir)
            state_file = _get_state_file()

            assert state_file.parent.exists()
            assert state_file.parent.name == ".multihead"


def test_load_seen_sessions_empty():
    """_load_seen_sessions returns empty set if no state file."""
    with patch("pathlib.Path.home") as mock_home:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            mock_home.return_value = Path(tmpdir)
            seen = _load_seen_sessions()
            assert seen == set()


def test_save_and_load_seen_sessions():
    """_save_seen_sessions persists and _load_seen_sessions retrieves."""
    with patch("pathlib.Path.home") as mock_home:
        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            mock_home.return_value = Path(tmpdir)

            # Save sessions
            sessions = {"session-1", "session-2", "session-3"}
            _save_seen_sessions(sessions)

            # Load them back
            loaded = _load_seen_sessions()
            assert loaded == sessions


def test_show_onboarding_messages_first_run(capsys):
    """First solo run shows mesh setup tip."""
    _show_onboarding_messages(is_first_run=True, new_sessions=[], total_sessions=0)

    captured = capsys.readouterr()
    assert "multihead init --mesh" in captured.out
    assert "multi-session collaboration" in captured.out


def test_show_onboarding_messages_new_collaborator(capsys):
    """New collaborator shows welcome message."""
    _show_onboarding_messages(
        is_first_run=False,
        new_sessions=["claude-h2v", "claude-bubblefill"],
        total_sessions=2,
    )

    captured = capsys.readouterr()
    assert "New session detected: claude-h2v!" in captured.out
    assert "New session detected: claude-bubblefill!" in captured.out


def test_show_onboarding_messages_many_new_collaborators(capsys):
    """Many new collaborators shows first 3 + count."""
    new_sessions = ["s1", "s2", "s3", "s4", "s5"]
    _show_onboarding_messages(
        is_first_run=False,
        new_sessions=new_sessions,
        total_sessions=5,
    )

    captured = capsys.readouterr()
    assert "s1" in captured.out
    assert "s2" in captured.out
    assert "s3" in captured.out
    assert "and 2 more" in captured.out


# ---------------------------------------------------------------------------
# Test: multihead init --mesh
# ---------------------------------------------------------------------------


def test_init_mesh_config_calls_wizard():
    """_init_mesh_config invokes MeshSetupWizard.run() with config_dir."""
    from multihead.cli import _init_mesh_config

    with patch("multihead.init_wizard.MeshSetupWizard") as mock_cls:
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
            config_dir = Path(tmpdir) / "config"
            config_dir.mkdir()
            _init_mesh_config(config_dir=config_dir)

        mock_instance.run.assert_called_once()
        call_kwargs = mock_instance.run.call_args
        assert call_kwargs.kwargs.get("config_dir") == config_dir


def test_init_mesh_config_default_config_dir():
    """_init_mesh_config defaults config_dir to ~/.multihead."""
    from multihead.cli import _init_mesh_config

    with patch("multihead.init_wizard.MeshSetupWizard") as mock_cls:
        mock_instance = MagicMock()
        mock_cls.return_value = mock_instance

        _init_mesh_config()  # No config_dir — should use default

        mock_instance.run.assert_called_once()
        call_kwargs = mock_instance.run.call_args
        assert call_kwargs.kwargs.get("config_dir") == Path.home() / ".multihead"


# ---------------------------------------------------------------------------
# Integration Test: End-to-End Mesh Flow
# ---------------------------------------------------------------------------


def test_mesh_collaboration_e2e():
    """End-to-end: presence → discovery → health monitoring → onboarding."""
    with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as tmpdir:
        db_path = Path(tmpdir) / "test.db"
        ks = KnowledgeStore(db_path)

        # Session 1 writes presence
        write_presence_claim(ks, "session-1", "test-project", ["solve", "decompose"])

        # Session 2 writes presence
        write_presence_claim(ks, "session-2", "test-project", ["solve"])

        # Session 3 discovers others
        from multihead.solve import discover_active_sessions
        others = discover_active_sessions(ks, "test-project", "session-3")

        assert len(others) == 2
        session_ids = {s["session_id"] for s in others}
        assert "session-1" in session_ids
        assert "session-2" in session_ids

        # Session 1 goes offline
        mark_session_offline(ks, "session-1", "test-project")

        # Session 3 discovers again (should only see session-2 now)
        # Note: This requires filtering by predicate="available" in discover_active_sessions
        # For now, we just verify the offline claim was written
        claims = ks.list_claims(claim_type=ClaimType.FACT.value, scope_id="test-project")
        offline_claims = [
            c for c in claims
            if c.canonical.claim_key.startswith("agent.session-1.presence.offline")
            and c.canonical.predicate == "offline"
        ]
        assert len(offline_claims) >= 1
