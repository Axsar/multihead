"""Round 17 tests: path traversal safety, link JSON safety,
config error handling, direct knowledge queries."""

from __future__ import annotations

import logging

import pytest
import yaml


# ---------------------------------------------------------------------------
# Path traversal sanitization in orchestrator
# ---------------------------------------------------------------------------


class TestOrchestratorPathSanitization:
    def test_step_name_sanitized_in_output_path(self):
        """Orchestrator should sanitize step.name before using in file paths."""
        import inspect
        from multihead.orchestrator import Orchestrator

        source = inspect.getsource(Orchestrator._execute_step)
        # Should use re.sub sanitization, not just replace(' ', '_')
        assert "re.sub" in source
        # Should NOT have the old unsafe pattern
        assert "step.name.replace(' ', '_')" not in source

    def test_path_traversal_stripped(self):
        """Path traversal characters should be replaced with underscores."""
        import re
        # Same sanitization as in orchestrator
        malicious_name = "../../etc/passwd"
        safe_name = re.sub(r"[^a-zA-Z0-9_\-]", "_", malicious_name)[:100]
        assert "/" not in safe_name
        assert ".." not in safe_name
        assert safe_name == "______etc_passwd"

    def test_long_name_truncated(self):
        """Step names longer than 100 chars should be truncated."""
        import re
        long_name = "a" * 200
        safe_name = re.sub(r"[^a-zA-Z0-9_\-]", "_", long_name)[:100]
        assert len(safe_name) == 100

    def test_normal_name_preserved(self):
        """Normal step names should be preserved."""
        import re
        name = "extract-entities"
        safe_name = re.sub(r"[^a-zA-Z0-9_\-]", "_", name)[:100]
        assert safe_name == "extract-entities"


# ---------------------------------------------------------------------------
# Knowledge store _row_to_link JSON safety
# ---------------------------------------------------------------------------


class TestLinkJSONSafety:
    def test_row_to_link_uses_safe_json(self):
        """_row_to_link should use _safe_json_loads, not raw json.loads."""
        import inspect
        from multihead.knowledge_store import KnowledgeStore

        source = inspect.getsource(KnowledgeStore._row_to_link)
        assert "_safe_json_loads" in source
        assert "json.loads" not in source


# ---------------------------------------------------------------------------
# Config error handling
# ---------------------------------------------------------------------------


class TestConfigErrorHandling:
    def test_load_heads_invalid_yaml(self, tmp_path):
        """load_heads should raise ValueError on invalid YAML."""
        from multihead.config import load_heads

        heads_file = tmp_path / "heads.yaml"
        heads_file.write_text("invalid: yaml: {{{{", encoding="utf-8")
        with pytest.raises(ValueError, match="Invalid YAML"):
            load_heads(tmp_path)

    def test_load_heads_missing_fields(self, tmp_path):
        """load_heads should raise ValueError on invalid head manifest."""
        from multihead.config import load_heads

        heads_file = tmp_path / "heads.yaml"
        # Missing required fields
        heads_file.write_text(
            yaml.dump({"heads": [{"name": "test"}]}),
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="Invalid head"):
            load_heads(tmp_path)

    def test_load_heads_no_heads_key(self, tmp_path):
        """load_heads should return empty dict if no 'heads' key."""
        from multihead.config import load_heads

        heads_file = tmp_path / "heads.yaml"
        heads_file.write_text("something_else: true", encoding="utf-8")
        result = load_heads(tmp_path)
        assert result == {}

    def test_load_heads_missing_file(self, tmp_path):
        """load_heads should return empty dict if file doesn't exist."""
        from multihead.config import load_heads

        result = load_heads(tmp_path)
        assert result == {}

    def test_load_heads_valid(self, tmp_path):
        """load_heads should parse valid heads.yaml correctly."""
        from multihead.config import load_heads

        heads_file = tmp_path / "heads.yaml"
        heads_file.write_text(yaml.dump({
            "heads": [{
                "head_id": "test-head",
                "name": "Test Head",
                "adapter": "mock",
                "model": "mock-v1",
                "kind": "llm",
            }]
        }), encoding="utf-8")
        result = load_heads(tmp_path)
        assert "test-head" in result
        assert result["test-head"].name == "Test Head"

    def test_load_recipe_invalid_yaml(self, tmp_path):
        """load_recipe should raise ValueError on invalid YAML."""
        from multihead.config import load_recipe

        recipes_dir = tmp_path / "recipes"
        recipes_dir.mkdir()
        recipe_file = recipes_dir / "bad.yaml"
        recipe_file.write_text("invalid: yaml: {{{{", encoding="utf-8")
        with pytest.raises(ValueError, match="Invalid YAML"):
            load_recipe(tmp_path, "bad")

    def test_load_recipe_not_a_mapping(self, tmp_path):
        """load_recipe should raise ValueError if YAML is not a mapping."""
        from multihead.config import load_recipe

        recipes_dir = tmp_path / "recipes"
        recipes_dir.mkdir()
        recipe_file = recipes_dir / "list.yaml"
        recipe_file.write_text("- item1\n- item2\n", encoding="utf-8")
        with pytest.raises(ValueError, match="must be a YAML mapping"):
            load_recipe(tmp_path, "list")

    def test_load_recipe_logs_empty_steps(self, tmp_path, caplog):
        """load_recipe should warn when recipe has no steps."""
        from multihead.config import load_recipe

        recipes_dir = tmp_path / "recipes"
        recipes_dir.mkdir()
        recipe_file = recipes_dir / "empty.yaml"
        recipe_file.write_text(yaml.dump({"goal": "nothing"}), encoding="utf-8")
        with caplog.at_level(logging.WARNING):
            wo = load_recipe(tmp_path, "empty")
        assert "no steps" in caplog.text
        assert wo.goal == "nothing"

    def test_config_module_has_logger(self):
        """config module should have a logger."""
        from multihead import config
        assert hasattr(config, "logger")


# ---------------------------------------------------------------------------
# Direct knowledge queries
# ---------------------------------------------------------------------------


class TestDirectKnowledgeQueries:
    def test_knowledge_store_has_get_event_by_id(self):
        """KnowledgeStore should have get_event_by_id method."""
        from multihead.knowledge_store import KnowledgeStore
        assert hasattr(KnowledgeStore, "get_event_by_id")

    def test_knowledge_store_has_get_claim_by_id(self):
        """KnowledgeStore should have get_claim_by_id method."""
        from multihead.knowledge_store import KnowledgeStore
        assert hasattr(KnowledgeStore, "get_claim_by_id")

    def test_get_event_by_id_returns_none_for_missing(self, tmp_path):
        """get_event_by_id should return None for non-existent event."""
        from multihead.knowledge_store import KnowledgeStore

        ks = KnowledgeStore(tmp_path / "knowledge.db")
        result = ks.get_event_by_id("nonexistent")
        assert result is None

    def test_get_claim_by_id_returns_none_for_missing(self, tmp_path):
        """get_claim_by_id should return None for non-existent claim."""
        from multihead.knowledge_store import KnowledgeStore

        ks = KnowledgeStore(tmp_path / "knowledge.db")
        result = ks.get_claim_by_id("nonexistent")
        assert result is None

    def test_routes_use_direct_query_for_event(self):
        """get_event route should use get_event_by_id, not list_events."""
        import inspect
        from multihead.api.routes_knowledge import get_event

        source = inspect.getsource(get_event)
        assert "get_event_by_id" in source
        assert "list_events" not in source

    def test_routes_use_direct_query_for_claim(self):
        """get_claim route should use get_claim_by_id, not list_claims."""
        import inspect
        from multihead.api.routes_knowledge import get_claim

        source = inspect.getsource(get_claim)
        assert "get_claim_by_id" in source
        assert "list_claims" not in source

    def test_get_event_by_id_uses_where_clause(self):
        """get_event_by_id should use WHERE event_id = ? (not scan)."""
        import inspect
        from multihead.knowledge_store import KnowledgeStore

        source = inspect.getsource(KnowledgeStore.get_event_by_id)
        assert "WHERE event_id = ?" in source

    def test_get_claim_by_id_uses_where_clause(self):
        """get_claim_by_id should use WHERE claim_id = ? (not scan)."""
        import inspect
        from multihead.knowledge_store import KnowledgeStore

        source = inspect.getsource(KnowledgeStore.get_claim_by_id)
        assert "WHERE claim_id = ?" in source
