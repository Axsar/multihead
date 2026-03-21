"""Round 14 tests: tool registry validation, artifact store safety,
API error sanitization, dead code cleanup."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from multihead.tool_registry import ToolRegistry


# ---------------------------------------------------------------------------
# Tool registry parameter validation
# ---------------------------------------------------------------------------


class TestToolRegistryValidation:
    def test_validate_missing_required_param(self):
        """Missing required param should produce a validation error."""
        registry = ToolRegistry()
        errors = registry.validate_params("files.read", {})
        assert any("Missing required parameter: path" in e for e in errors)

    def test_validate_unknown_param(self):
        """Unknown params should produce a validation error."""
        registry = ToolRegistry()
        errors = registry.validate_params("files.read", {"path": "/tmp/x", "bogus": 123})
        assert any("Unknown parameter: bogus" in e for e in errors)

    def test_validate_wrong_type(self):
        """Wrong type should produce a validation error."""
        registry = ToolRegistry()
        errors = registry.validate_params("files.read", {"path": 12345})
        assert any("expected type string" in e for e in errors)

    def test_validate_valid_params(self):
        """Valid params should produce no errors."""
        registry = ToolRegistry()
        errors = registry.validate_params("files.read", {"path": "/tmp/test.txt"})
        assert errors == []

    def test_validate_unknown_tool(self):
        """Unknown tool should produce an error."""
        registry = ToolRegistry()
        errors = registry.validate_params("nonexistent.tool", {"x": 1})
        assert any("Unknown tool" in e for e in errors)

    @pytest.mark.asyncio
    async def test_execute_rejects_invalid_params(self):
        """execute() should reject calls with invalid params."""
        registry = ToolRegistry()
        result = await registry.execute("files.read", {})
        assert result.success is False
        assert "Validation errors" in result.error
        assert "Missing required parameter" in result.error

    @pytest.mark.asyncio
    async def test_execute_rejects_wrong_type(self):
        """execute() should reject calls with wrong param type."""
        registry = ToolRegistry()
        result = await registry.execute("shell.run", {"command": 123})
        assert result.success is False
        assert "expected type string" in result.error

    @pytest.mark.asyncio
    async def test_execute_accepts_valid_params(self):
        """execute() should accept valid params and proceed to handler."""
        registry = ToolRegistry()
        result = await registry.execute("files.read", {"path": "/tmp/nonexistent_file_abc123"})
        # Handler runs, file doesn't exist, but validation passed
        assert result.success is False
        assert "File not found" in result.error

    def test_validate_optional_param_with_default(self):
        """Optional params with defaults should not be required."""
        registry = ToolRegistry()
        # shell.run has 'command' (required) and 'timeout' (has default, not required)
        errors = registry.validate_params("shell.run", {"command": "ls"})
        assert errors == []

    def test_validate_multiple_errors(self):
        """Multiple issues should all be reported."""
        registry = ToolRegistry()
        errors = registry.validate_params("files.write", {"bogus": 42})
        # Missing path, missing content, unknown bogus
        assert len(errors) >= 2


# ---------------------------------------------------------------------------
# Artifact store size limit and atomic writes
# ---------------------------------------------------------------------------


class TestArtifactStoreSafety:
    def test_default_max_size(self):
        """ArtifactStore should have a 100MB default max size."""
        from multihead.artifact_store import ArtifactStore

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
            store = ArtifactStore(Path(d) / "art", Path(d) / "db.sqlite")
            assert store.max_size_bytes == 100 * 1024 * 1024

    def test_custom_max_size(self):
        """ArtifactStore should accept a custom max size."""
        from multihead.artifact_store import ArtifactStore

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
            store = ArtifactStore(Path(d) / "art", Path(d) / "db.sqlite", max_size_bytes=1024)
            assert store.max_size_bytes == 1024

    def test_rejects_oversized_artifact(self):
        """store() should reject data exceeding max_size_bytes."""
        from multihead.artifact_store import ArtifactStore

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
            store = ArtifactStore(Path(d) / "art", Path(d) / "db.sqlite", max_size_bytes=100)
            with pytest.raises(ValueError, match="too large"):
                store.store(b"x" * 101)

    def test_accepts_within_limit(self):
        """store() should accept data within max_size_bytes."""
        from multihead.artifact_store import ArtifactStore

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
            store = ArtifactStore(Path(d) / "art", Path(d) / "db.sqlite", max_size_bytes=100)
            ref = store.store(b"x" * 100)
            assert ref.size_bytes == 100

    def test_atomic_write_creates_file(self):
        """store() should atomically write the file (no partial writes)."""
        from multihead.artifact_store import ArtifactStore

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
            store = ArtifactStore(Path(d) / "art", Path(d) / "db.sqlite")
            data = b"test artifact content"
            ref = store.store(data)
            # Verify file exists and has correct content
            fetched = store.fetch(ref.artifact_id)
            assert fetched == data

    def test_store_file_respects_size_limit(self):
        """store_file() should also respect max_size_bytes."""
        from multihead.artifact_store import ArtifactStore

        with tempfile.TemporaryDirectory(ignore_cleanup_errors=True) as d:
            store = ArtifactStore(Path(d) / "art", Path(d) / "db.sqlite", max_size_bytes=50)
            file_path = Path(d) / "big.bin"
            file_path.write_bytes(b"x" * 51)
            with pytest.raises(ValueError, match="too large"):
                store.store_file(file_path)


# ---------------------------------------------------------------------------
# API error sanitization
# ---------------------------------------------------------------------------


class TestAPIErrorSanitization:
    def test_wake_head_error_no_internal_details(self):
        """Wake head error should not expose internal exception details."""
        import inspect
        from multihead.api.routes_heads import wake_head

        source = inspect.getsource(wake_head)
        # Should NOT contain f"Failed to wake head: {e}" (leaks exception)
        assert "f\"Failed to wake head: {e}\"" not in source
        # Should have logger call for internal details
        assert "logger.exception" in source

    def test_consensus_error_no_internal_details(self):
        """Consensus error should not expose internal exception details."""
        import inspect
        from multihead.api.routes_consensus import execute_consensus

        source = inspect.getsource(execute_consensus)
        # Should NOT directly pass str(e) to HTTPException
        assert "HTTPException(422, str(e))" not in source
        # Should log the actual error
        assert "logger.warning" in source

    def test_routes_heads_has_logger(self):
        """routes_heads module should have a logger configured."""
        from multihead.api import routes_heads
        assert hasattr(routes_heads, "logger")

    def test_routes_consensus_has_logger(self):
        """routes_consensus module should have a logger configured."""
        from multihead.api import routes_consensus
        assert hasattr(routes_consensus, "logger")


# ---------------------------------------------------------------------------
# Dead code cleanup
# ---------------------------------------------------------------------------


class TestDeadCodeCleanup:
    def test_no_subprocess_import_in_tool_registry(self):
        """tool_registry should not import unused subprocess module."""
        import inspect
        from multihead import tool_registry

        source = inspect.getsource(tool_registry)
        # asyncio.create_subprocess_shell is used, but 'import subprocess' is not needed
        assert "import subprocess" not in source

    def test_no_unused_hashlib_in_record_store(self):
        """record_store should not import unused hashlib module."""
        import inspect
        from multihead import record_store

        source = inspect.getsource(record_store)
        assert "import hashlib" not in source

    def test_record_store_still_works(self):
        """record_store should still function after import cleanup."""
        from multihead.record_store import RecordStore
        # Just verify it imports cleanly
        assert RecordStore is not None
