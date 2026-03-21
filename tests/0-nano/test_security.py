"""Security regression tests.

Verifies SSRF protection, shell injection prevention, and safe serialization.
"""

import json
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# SSRF protection in web tools
# ---------------------------------------------------------------------------

class TestSSRFProtection:
    """Verify _is_safe_url blocks dangerous URLs."""

    def test_blocks_file_scheme(self):
        from multihead.web_tools import _is_safe_url

        safe, reason = _is_safe_url("file:///etc/passwd")
        assert not safe
        assert "scheme" in reason.lower() or "file" in reason.lower()

    def test_blocks_gopher_scheme(self):
        from multihead.web_tools import _is_safe_url

        safe, reason = _is_safe_url("gopher://evil.com")
        assert not safe

    def test_blocks_ftp_scheme(self):
        from multihead.web_tools import _is_safe_url

        safe, reason = _is_safe_url("ftp://server/file")
        assert not safe

    def test_blocks_localhost(self):
        from multihead.web_tools import _is_safe_url

        safe, reason = _is_safe_url("http://127.0.0.1:7337/health")
        assert not safe
        assert "private" in reason.lower() or "127" in reason

    def test_blocks_localhost_name(self):
        from multihead.web_tools import _is_safe_url

        safe, reason = _is_safe_url("http://localhost/secret")
        assert not safe

    def test_blocks_private_10(self):
        from multihead.web_tools import _is_safe_url

        safe, reason = _is_safe_url("http://10.0.0.1/admin")
        assert not safe

    def test_blocks_private_172(self):
        from multihead.web_tools import _is_safe_url

        safe, reason = _is_safe_url("http://172.16.0.1/admin")
        assert not safe

    def test_blocks_private_192(self):
        from multihead.web_tools import _is_safe_url

        safe, reason = _is_safe_url("http://192.168.1.1/admin")
        assert not safe

    def test_allows_public_https(self):
        from multihead.web_tools import _is_safe_url

        safe, reason = _is_safe_url("https://example.com/page")
        assert safe
        assert reason == ""

    def test_allows_public_http(self):
        from multihead.web_tools import _is_safe_url

        safe, reason = _is_safe_url("http://example.com/page")
        assert safe

    def test_blocks_empty_url(self):
        from multihead.web_tools import _is_safe_url

        safe, _ = _is_safe_url("")
        assert not safe

    def test_blocks_no_scheme(self):
        from multihead.web_tools import _is_safe_url

        safe, _ = _is_safe_url("just-a-string")
        assert not safe

    @pytest.mark.asyncio
    async def test_web_fetch_rejects_file_url(self):
        """End-to-end: _tool_web_fetch returns error for file:// URLs."""
        from multihead.web_tools import _tool_web_fetch

        result = await _tool_web_fetch({"url": "file:///etc/passwd"})
        assert not result.success
        assert "scheme" in result.error.lower() or "file" in result.error.lower()

    @pytest.mark.asyncio
    async def test_web_fetch_rejects_private_ip(self):
        """End-to-end: _tool_web_fetch returns error for private IPs."""
        from multihead.web_tools import _tool_web_fetch

        result = await _tool_web_fetch({"url": "http://127.0.0.1:7337"})
        assert not result.success
        assert "private" in result.error.lower() or "blocked" in result.error.lower()


# ---------------------------------------------------------------------------
# Shell injection in process manager
# ---------------------------------------------------------------------------

class TestShellInjection:
    """Verify process manager uses shlex.split (exec mode, no shell interpretation)."""

    def test_no_shell_expansion(self):
        """shlex.split prevents shell metacharacter interpretation."""
        import shlex
        # With shell=True, semicolons would chain commands
        # With shlex.split + exec, they're just arguments
        args = shlex.split("echo safe; echo pwned")
        assert args[0] == "echo"
        assert ";" in args[1]  # semicolon is part of an argument, not a separator

    def test_command_injection_dollar(self):
        """$(command) should not be split into shell expansion."""
        import shlex
        args = shlex.split('echo "$(whoami)"')
        # $(whoami) stays as a literal string, not expanded
        assert "$(whoami)" in args

    def test_pipe_not_interpreted(self):
        """Pipe characters are not interpreted as pipelines."""
        import shlex
        args = shlex.split("echo hello | cat")
        # "|" and "cat" are just arguments to echo, not a pipeline
        assert "|" in args
        assert "cat" in args

    def test_spawn_uses_exec_not_shell(self):
        """ProcessManager.spawn uses create_subprocess_exec, not shell=True."""
        import inspect
        from multihead.process_manager import ProcessManager

        source = inspect.getsource(ProcessManager.spawn)
        assert "create_subprocess_exec" in source
        assert "shell=True" not in source


# ---------------------------------------------------------------------------
# Pickle deserialization in embedding search
# ---------------------------------------------------------------------------

class TestEmbeddingSafeSerialization:
    """Verify embedding cache uses JSON, not pickle."""

    def test_no_allow_pickle_in_source(self):
        """Source code must not contain allow_pickle=True."""
        source_path = (
            Path(__file__).parent.parent.parent
            / "src" / "multihead" / "embedding_search.py"
        )
        source = source_path.read_text()
        assert "allow_pickle" not in source, (
            "allow_pickle found in embedding_search.py — use JSON instead"
        )

    def test_ids_saved_as_json(self, tmp_path):
        """Claim IDs should be saved as JSON, not numpy pickle."""
        # Create a mock DB
        import sqlite3
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE claims (
                claim_id TEXT PRIMARY KEY,
                claim_key TEXT,
                statement TEXT,
                confidence REAL,
                claim_status TEXT DEFAULT 'accepted'
            )
        """)
        for i in range(5):
            conn.execute(
                "INSERT INTO claims VALUES (?, ?, ?, ?, ?)",
                (
                    f"id_{i}", f"key.{i}",
                    f"This is test claim number {i} with enough text",
                    0.9, "accepted",
                ),
            )
        conn.commit()
        conn.close()

        # Mock the model to avoid loading sentence-transformers
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.randn(5, 384).astype(np.float32)

        from multihead.embedding_search import EmbeddingIndex
        idx = EmbeddingIndex(db_path=db_path, cache_dir=tmp_path)
        idx._model = mock_model

        idx.build(force=True)

        # Check that IDs are saved as JSON
        ids_path = tmp_path / "claim_ids.json"
        assert ids_path.exists(), "claim_ids.json not created"

        # Verify it's valid JSON
        loaded = json.loads(ids_path.read_text())
        assert isinstance(loaded, list)
        assert len(loaded) == 5

        # Verify no .npy file exists
        npy_path = tmp_path / "claim_ids.npy"
        assert not npy_path.exists(), "Legacy claim_ids.npy should not be created"

    def test_ids_loaded_from_json(self, tmp_path):
        """Incremental cache load should read from JSON."""
        import sqlite3
        db_path = tmp_path / "test.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("""
            CREATE TABLE claims (
                claim_id TEXT PRIMARY KEY,
                claim_key TEXT,
                statement TEXT,
                confidence REAL,
                claim_status TEXT DEFAULT 'accepted'
            )
        """)
        for i in range(3):
            conn.execute(
                "INSERT INTO claims VALUES (?, ?, ?, ?, ?)",
                (
                    f"id_{i}", f"key.{i}",
                    f"This is test claim number {i} with enough text",
                    0.9, "accepted",
                ),
            )
        conn.commit()
        conn.close()

        # Pre-create cache files in JSON format
        embeddings = np.random.randn(3, 384).astype(np.float32)
        np.savez_compressed(str(tmp_path / "claim_embeddings.npz"), embeddings=embeddings)
        (tmp_path / "claim_ids.json").write_text(json.dumps(["id_0", "id_1", "id_2"]))

        from multihead.embedding_search import EmbeddingIndex
        idx = EmbeddingIndex(db_path=db_path, cache_dir=tmp_path)

        # Build should load from cache without needing model
        count = idx.build()
        assert count == 3
        assert idx._built


# ---------------------------------------------------------------------------
# API binding warning
# ---------------------------------------------------------------------------

class TestAPIBindingWarning:
    """Verify warning when API binds to all interfaces."""

    def test_warns_on_public_binding(self):
        """Settings should warn when api_host is 0.0.0.0."""
        from multihead.config import Settings
        import logging

        with patch.object(logging.getLogger("multihead.config"), "warning") as mock_warn:
            Settings(api_host="0.0.0.0")
            mock_warn.assert_called_once()
            assert (
                "all interfaces" in mock_warn.call_args[0][0].lower()
                or "0.0.0.0" in str(mock_warn.call_args)
            )

    def test_no_warning_on_localhost(self):
        """No warning when api_host is 127.0.0.1."""
        from multihead.config import Settings
        import logging

        with patch.object(logging.getLogger("multihead.config"), "warning") as mock_warn:
            Settings(api_host="127.0.0.1")
            mock_warn.assert_not_called()
