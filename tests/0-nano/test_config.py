"""Tests for configuration loading."""

import pytest

from multihead.config import Settings, load_heads, load_recipe
from multihead.models import AdapterKind


class TestSettings:
    def test_defaults(self, monkeypatch):
        monkeypatch.delenv("MULTIHEAD_CORE_HEAD_ID", raising=False)
        monkeypatch.delenv("MULTIHEAD_DATA_DIR", raising=False)
        s = Settings(_env_file=None)
        assert s.api_host == "127.0.0.1"
        assert s.api_port == 7337
        assert s.default_checkpoint_mode == "sync"
        assert s.mesh_secret is None
        assert s.core_head_id == "mock-llm"

    def test_custom_data_dir(self, tmp_path):
        s = Settings(data_dir=tmp_path / "custom")
        assert s.data_dir == tmp_path / "custom"

    def test_derived_paths(self, tmp_path):
        s = Settings(data_dir=tmp_path / "data")
        assert s.runs_dir == tmp_path / "data" / "runs"
        assert s.artifacts_dir == tmp_path / "data" / "artifacts"
        assert s.db_path == tmp_path / "data" / "state.db"
        assert s.knowledge_db_path == tmp_path / "data" / "knowledge.db"
        assert s.packs_dir == tmp_path / "data" / "packs"
        assert s.nightshift_output_dir == tmp_path / "data" / "nightshift"

    def test_ensure_dirs(self, tmp_path):
        s = Settings(data_dir=tmp_path / "data")
        s.ensure_dirs()
        assert s.runs_dir.exists()
        assert s.artifacts_dir.exists()
        assert s.packs_dir.exists()
        assert s.nightshift_output_dir.exists()

    def test_mesh_secret_configurable(self):
        s = Settings(mesh_secret="super-secret-long-enough-token")
        assert s.mesh_secret == "super-secret-long-enough-token"

    def test_ensure_dirs_creates_data_dir(self, tmp_path):
        s = Settings(data_dir=tmp_path / "brand_new")
        s.ensure_dirs()
        assert (tmp_path / "brand_new").exists()


class TestResolveCoreHeadId:
    def _mock_heads(self, *heads):
        """Create a dict of mock head manifests."""
        from types import SimpleNamespace
        return {h[0]: SimpleNamespace(kind=h[1]) for h in heads}

    def test_configured_value_exists(self):
        s = Settings(core_head_id="qwen-llm", _env_file=None)
        heads = self._mock_heads(("qwen-llm", "llm"), ("mock-llm", "llm"))
        assert s.resolve_core_head_id(heads) == "qwen-llm"

    def test_configured_value_missing_falls_to_llm(self):
        s = Settings(core_head_id="nonexistent", _env_file=None)
        heads = self._mock_heads(("mock-vlm", "vlm"), ("my-llm", "llm"))
        assert s.resolve_core_head_id(heads) == "my-llm"

    def test_no_llm_falls_to_first(self):
        s = Settings(core_head_id="nonexistent", _env_file=None)
        heads = self._mock_heads(("only-vlm", "vlm"))
        assert s.resolve_core_head_id(heads) == "only-vlm"

    def test_empty_heads_falls_to_mock(self):
        s = Settings(core_head_id="anything", _env_file=None)
        assert s.resolve_core_head_id({}) == "mock-llm"


class TestLoadHeads:
    def test_load_heads(self, tmp_path):
        heads_yaml = tmp_path / "heads.yaml"
        heads_yaml.write_text("""\
heads:
  - head_id: test-llm
    name: Test LLM
    adapter: mock
    model: mock-v1
    kind: llm
    gpu_required: false
""")
        heads = load_heads(tmp_path)
        assert "test-llm" in heads
        assert heads["test-llm"].adapter == AdapterKind.MOCK

    def test_load_heads_missing_file(self, tmp_path):
        heads = load_heads(tmp_path)
        assert heads == {}

    def test_load_multiple_heads(self, tmp_path):
        heads_yaml = tmp_path / "heads.yaml"
        heads_yaml.write_text("""\
heads:
  - head_id: h1
    name: H1
    adapter: mock
    model: m1
    kind: llm
  - head_id: h2
    name: H2
    adapter: ollama
    model: phi-3
    kind: llm
""")
        heads = load_heads(tmp_path)
        assert len(heads) == 2


class TestLoadRecipe:
    def test_load_recipe(self, tmp_path):
        recipes_dir = tmp_path / "recipes"
        recipes_dir.mkdir()
        (recipes_dir / "test-pipeline.yaml").write_text("""\
goal: "Test pipeline"
steps:
  - name: plan
    head_id: mock-llm
    prompt_template: "Create a plan"
  - name: execute
    head_id: mock-vlm
    prompt_template: "Execute the plan"
""")
        wo = load_recipe(tmp_path, "test-pipeline")
        assert wo.goal == "Test pipeline"
        assert len(wo.steps) == 2
        assert wo.steps[0].name == "plan"

    def test_load_recipe_missing(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_recipe(tmp_path, "nonexistent")

    def test_load_recipe_with_json_schema_validator(self, tmp_path):
        """Validator config in YAML should be hydrated into Validator instance."""
        from multihead.validators import JSONSchemaValidator
        recipes_dir = tmp_path / "recipes"
        recipes_dir.mkdir()
        (recipes_dir / "validated.yaml").write_text("""\
goal: "Validated pipeline"
steps:
  - name: extract
    head_id: mock-llm
    prompt_template: "Extract entities"
    validator:
      type: json_schema
      schema:
        type: object
        properties:
          entities:
            type: array
        required:
          - entities
""")
        wo = load_recipe(tmp_path, "validated")
        assert len(wo.steps) == 1
        assert wo.steps[0].validator is not None
        assert isinstance(wo.steps[0].validator, JSONSchemaValidator)

    def test_load_recipe_with_format_validator(self, tmp_path):
        """Format validator should be hydrated."""
        from multihead.validators import FormatValidator
        recipes_dir = tmp_path / "recipes"
        recipes_dir.mkdir()
        (recipes_dir / "fmt.yaml").write_text("""\
goal: "Format check"
steps:
  - name: gen
    head_id: mock-llm
    prompt_template: "Generate text"
    validator:
      type: format
      min_length: 10
      max_length: 500
""")
        wo = load_recipe(tmp_path, "fmt")
        assert isinstance(wo.steps[0].validator, FormatValidator)

    def test_load_recipe_with_composite_validator(self, tmp_path):
        """Composite validator with nested sub-validators."""
        from multihead.validators import CompositeValidator
        recipes_dir = tmp_path / "recipes"
        recipes_dir.mkdir()
        (recipes_dir / "comp.yaml").write_text("""\
goal: "Composite"
steps:
  - name: analyze
    head_id: mock-llm
    prompt_template: "Analyze"
    validator:
      type: composite
      mode: all
      validators:
        - type: format
          min_length: 5
        - type: confidence
          min_confidence: 0.7
""")
        wo = load_recipe(tmp_path, "comp")
        assert isinstance(wo.steps[0].validator, CompositeValidator)

    def test_load_recipe_without_validator(self, tmp_path):
        """Steps without validator config should have validator=None."""
        recipes_dir = tmp_path / "recipes"
        recipes_dir.mkdir()
        (recipes_dir / "noval.yaml").write_text("""\
goal: "No validator"
steps:
  - name: step1
    head_id: mock-llm
    prompt_template: "Do something"
""")
        wo = load_recipe(tmp_path, "noval")
        assert wo.steps[0].validator is None

    def test_load_recipe_invalid_validator_warns(self, tmp_path, caplog):
        """Invalid validator config should warn but not crash."""
        recipes_dir = tmp_path / "recipes"
        recipes_dir.mkdir()
        (recipes_dir / "badval.yaml").write_text("""\
goal: "Bad validator"
steps:
  - name: step1
    head_id: mock-llm
    prompt_template: "Do something"
    validator:
      type: nonexistent_type
""")
        import logging
        with caplog.at_level(logging.WARNING):
            wo = load_recipe(tmp_path, "badval")
        assert wo.steps[0].validator is None
        assert "Failed to hydrate validator" in caplog.text
