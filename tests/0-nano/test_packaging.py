"""Install and packaging tests for MultiHead."""

from __future__ import annotations

import importlib
from unittest.mock import patch

import pytest
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent.parent


class TestEntryPoint:
    """Test that the CLI entry point works."""

    def test_multihead_help(self):
        """multihead --help should exit 0 and show usage."""
        from click.testing import CliRunner
        from multihead.cli import main

        runner = CliRunner()
        result = runner.invoke(main, ["--help"])
        assert result.exit_code == 0
        assert "MultiHead" in result.output

    def test_multihead_entrypoint_importable(self):
        """The entry point function should be importable."""
        from multihead.cli import main
        assert callable(main)

    def test_multihead_version(self):
        """Should have a version string."""
        import multihead
        assert hasattr(multihead, "__version__")
        assert isinstance(multihead.__version__, str)
        assert len(multihead.__version__) > 0


class TestCoreImports:
    """Test that all core modules are importable."""

    @pytest.mark.parametrize("module_name", [
        "multihead.cli",
        "multihead.orchestrator",
        "multihead.head_manager",
        "multihead.router",
        "multihead.models",
        "multihead.config",
        "multihead.consensus",
        "multihead.knowledge_store",
        "multihead.knowledge_models",
        "multihead.dag_executor",
        "multihead.plan_normalizer",
        "multihead.mcp_server",
        "multihead.acp_bridge",
        "multihead.runtime_config",
        "multihead.web_tools",
        "multihead.reflection",
        "multihead.tree_of_thoughts",
        "multihead.process_reward_models",
        "multihead.auto_decomposition",
        "multihead.recipe_learning",
        "multihead.resilience",
        "multihead.engine",
        "multihead.shell",
        "multihead.process_manager",
    ])
    def test_import_module(self, module_name):
        """Each core module should import without errors."""
        mod = importlib.import_module(module_name)
        assert mod is not None


class TestAdapterImports:
    """Test that adapter modules are importable."""

    @pytest.mark.parametrize("adapter_name", [
        "multihead.adapters.mock",
        "multihead.adapters.ollama",
        "multihead.adapters.openai_adapter",
        "multihead.adapters.transformers_adapter",
        "multihead.adapters.claude_agent_sdk",
    ])
    def test_import_adapter(self, adapter_name):
        """Each adapter module should import without errors."""
        mod = importlib.import_module(adapter_name)
        assert mod is not None


class TestProjectStructure:
    """Test that required project files exist."""

    def test_pyproject_toml_exists(self):
        assert (PROJECT_ROOT / "pyproject.toml").exists()

    def test_readme_exists(self):
        assert (PROJECT_ROOT / "README.md").exists()

    def test_src_layout(self):
        assert (PROJECT_ROOT / "src" / "multihead").is_dir()

    def test_init_py_exists(self):
        assert (PROJECT_ROOT / "src" / "multihead" / "__init__.py").exists()

    def test_heads_yaml_exists(self):
        assert (PROJECT_ROOT / "config" / "heads.yaml").exists()

    def test_tests_directory(self):
        assert (PROJECT_ROOT / "tests").is_dir()


class TestDependencies:
    """Test that required dependencies are installed."""

    @pytest.mark.parametrize("package", [
        "fastapi",
        "uvicorn",
        "pydantic",
        "httpx",
        "click",
        "yaml",
        "rich",
        "aiosqlite",
        "dotenv",
    ])
    def test_required_dependency(self, package):
        """Each required dependency should be importable."""
        importlib.import_module(package)


class TestV11Exports:
    """Test v1.1 SDK exports are accessible."""

    def test_engine_importable(self):
        from multihead import Engine
        assert Engine is not None

    def test_engine_has_expected_methods(self):
        from multihead import Engine
        e = Engine.__new__(Engine)
        assert hasattr(e, "start")
        assert hasattr(e, "stop")
        assert hasattr(e, "generate")
        assert hasattr(e, "chat")
        assert hasattr(e, "route")
        assert hasattr(e, "wake")
        assert hasattr(e, "swap")

    def test_shell_importable(self):
        from multihead.shell import Shell
        assert Shell is not None

    def test_process_manager_importable(self):
        from multihead.process_manager import ProcessManager, ManagedProcess
        assert ProcessManager is not None
        assert ManagedProcess is not None

    def test_version_is_1_1(self):
        import multihead
        assert multihead.__version__ == "1.3.52"


class TestOptionalDeps:
    """Test that optional dependency groups are declared."""

    def test_gpu_deps_declared(self):
        """pyproject.toml should have [gpu] optional deps."""
        content = (PROJECT_ROOT / "pyproject.toml").read_text()
        assert "[project.optional-dependencies]" in content
        assert "gpu" in content
        assert "torch" in content

    def test_mesh_deps_declared(self):
        """pyproject.toml should have [mesh] optional deps."""
        content = (PROJECT_ROOT / "pyproject.toml").read_text()
        assert "mesh" in content
        assert "zeroconf" in content

    def test_dev_deps_declared(self):
        """pyproject.toml should have [dev] optional deps."""
        content = (PROJECT_ROOT / "pyproject.toml").read_text()
        assert "dev" in content
        assert "pytest" in content


# ---------------------------------------------------------------------------
# Bundled config (pip install support)
# ---------------------------------------------------------------------------


class TestBundledConfig:
    """Verify bundled default_config/ ships with the package."""

    def test_bundled_config_path_exists(self):
        from multihead.config import _bundled_config_path
        path = _bundled_config_path()
        assert path.exists(), f"Bundled config not found at {path}"

    def test_bundled_heads_yaml(self):
        import yaml
        from multihead.config import _bundled_config_path
        heads_file = _bundled_config_path() / "heads.yaml"
        assert heads_file.exists()
        data = yaml.safe_load(heads_file.read_text())
        assert "heads" in data
        assert len(data["heads"]) >= 2

    def test_bundled_templates(self):
        from multihead.config import _bundled_config_path
        templates = _bundled_config_path() / "templates"
        assert templates.exists()
        assert (templates / "cpu_only.yaml").exists()
        assert (templates / "rtx4090.yaml").exists()

    def test_bundled_recipes(self):
        from multihead.config import _bundled_config_path
        recipes = _bundled_config_path() / "recipes"
        assert recipes.exists()
        assert any(recipes.glob("*.yaml"))


class TestResolveConfigDir:
    """Config directory resolution for different install scenarios."""

    def test_explicit_path(self, tmp_path):
        from multihead.config import resolve_config_dir
        result = resolve_config_dir(str(tmp_path))
        assert result == tmp_path

    def test_repo_checkout_detected(self, tmp_path, monkeypatch):
        """./config/ with heads.yaml should be found in dev mode."""
        from multihead.config import resolve_config_dir
        monkeypatch.chdir(tmp_path)
        config = tmp_path / "config"
        config.mkdir()
        (config / "heads.yaml").write_text("heads: []")
        result = resolve_config_dir(None)
        assert result == Path("config")

    def test_user_config_detected(self, tmp_path, monkeypatch):
        """~/.multihead/config/ with heads.yaml should be found."""
        from multihead.config import resolve_config_dir
        user_config = tmp_path / "user_config"
        user_config.mkdir(parents=True)
        (user_config / "heads.yaml").write_text("heads: []")

        empty_dir = tmp_path / "workdir"
        empty_dir.mkdir()
        monkeypatch.chdir(empty_dir)

        with patch("multihead.config._USER_CONFIG_DIR", user_config):
            result = resolve_config_dir(None)
        assert result == user_config

    def test_bundled_fallback_copies_to_user(self, tmp_path, monkeypatch):
        """When no config exists, bundled defaults are copied to user dir."""
        from multihead.config import resolve_config_dir
        empty_dir = tmp_path / "workdir"
        empty_dir.mkdir()
        monkeypatch.chdir(empty_dir)

        user_config = tmp_path / "user_config"

        with patch("multihead.config._USER_CONFIG_DIR", user_config):
            result = resolve_config_dir(None)

        assert user_config.exists()
        assert (user_config / "heads.yaml").exists()
        assert result == user_config

    def test_solvers_yaml_also_detected(self, tmp_path, monkeypatch):
        """solvers.yaml (not just heads.yaml) triggers dev mode detection."""
        from multihead.config import resolve_config_dir
        monkeypatch.chdir(tmp_path)
        config = tmp_path / "config"
        config.mkdir()
        (config / "solvers.yaml").write_text("solvers: []")
        result = resolve_config_dir(None)
        assert result == Path("config")


class TestBootstrapUserConfig:
    """First-run auto-copy of bundled config."""

    def test_copies_bundled_to_user_dir(self, tmp_path):
        from multihead.config import _bootstrap_user_config, _bundled_config_path
        user_dir = tmp_path / "config"

        with patch("multihead.config._USER_CONFIG_DIR", user_dir):
            _bootstrap_user_config(_bundled_config_path())

        assert user_dir.exists()
        assert (user_dir / "heads.yaml").exists()
        assert (user_dir / "templates").exists()
        assert (user_dir / "recipes").exists()

    def test_does_not_overwrite_existing(self, tmp_path):
        from multihead.config import _bootstrap_user_config, _bundled_config_path
        user_dir = tmp_path / "config"
        user_dir.mkdir()
        (user_dir / "heads.yaml").write_text("custom: true")

        with patch("multihead.config._USER_CONFIG_DIR", user_dir):
            _bootstrap_user_config(_bundled_config_path())

        assert (user_dir / "heads.yaml").read_text() == "custom: true"
