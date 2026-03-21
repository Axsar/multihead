"""Tests for runtime configuration."""

import pytest

from multihead.runtime_config import RuntimeConfig


class TestRuntimeConfig:
    def test_defaults(self):
        config = RuntimeConfig()
        assert config.disabled_tools == []
        assert config.web_tools_enabled is True
        assert config.generation.temperature == 0.7
        assert config.generation.max_tokens == 4096
        assert config.vram_core_mode == "keep_loaded"

    def test_tool_enable_disable(self):
        config = RuntimeConfig()
        assert config.is_tool_enabled("web.search")
        config.disable_tool("web.search")
        assert not config.is_tool_enabled("web.search")
        assert "web.search" in config.disabled_tools
        config.enable_tool("web.search")
        assert config.is_tool_enabled("web.search")
        assert "web.search" not in config.disabled_tools

    def test_disable_idempotent(self):
        config = RuntimeConfig()
        config.disable_tool("web.search")
        config.disable_tool("web.search")
        assert config.disabled_tools.count("web.search") == 1

    def test_enable_nonexistent(self):
        config = RuntimeConfig()
        config.enable_tool("nonexistent")
        assert config.disabled_tools == []

    def test_save_and_load(self, tmp_path):
        config = RuntimeConfig()
        config.disable_tool("shell.run")
        config.generation.temperature = 0.5
        path = tmp_path / "config.json"
        config.save(path)
        assert path.exists()

        loaded = RuntimeConfig.load(path)
        assert loaded.disabled_tools == ["shell.run"]
        assert loaded.generation.temperature == 0.5

    def test_load_missing_file(self, tmp_path):
        config = RuntimeConfig.load(tmp_path / "nonexistent.json")
        assert config.disabled_tools == []

    def test_load_corrupt_file(self, tmp_path):
        path = tmp_path / "config.json"
        path.write_text("not valid json")
        config = RuntimeConfig.load(path)
        assert config.disabled_tools == []


class TestSetValue:
    def test_set_temperature(self):
        config = RuntimeConfig()
        result = config.set_value("generation.temperature", "0.3")
        assert config.generation.temperature == 0.3
        assert "0.3" in result

    def test_set_max_tokens(self):
        config = RuntimeConfig()
        config.set_value("generation.max_tokens", "2048")
        assert config.generation.max_tokens == 2048

    def test_set_web_tools_enabled(self):
        config = RuntimeConfig()
        config.set_value("web_tools_enabled", "false")
        assert config.web_tools_enabled is False
        config.set_value("web_tools_enabled", "true")
        assert config.web_tools_enabled is True

    def test_set_vram_core_mode(self):
        config = RuntimeConfig()
        config.set_value("vram_core_mode", "unload_during_batch")
        assert config.vram_core_mode == "unload_during_batch"

    def test_set_invalid_vram_mode(self):
        config = RuntimeConfig()
        with pytest.raises(ValueError, match="must be one of"):
            config.set_value("vram_core_mode", "invalid")

    def test_set_unknown_key(self):
        config = RuntimeConfig()
        with pytest.raises(ValueError, match="Unknown config key"):
            config.set_value("nonexistent", "value")

    def test_set_unknown_generation_field(self):
        config = RuntimeConfig()
        with pytest.raises(ValueError, match="Unknown generation field"):
            config.set_value("generation.nonexistent", "1.0")
