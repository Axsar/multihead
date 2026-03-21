"""Tests for init wizard."""

from pathlib import Path

import yaml

from multihead.init_wizard import AdapterStatus, HardwareProfile, InitWizard


class TestHardwareProfile:
    def test_defaults(self):
        p = HardwareProfile()
        assert p.gpu_name == ""
        assert p.gpu_vram_mb == 0
        assert not p.has_cuda


class TestInitWizard:
    def test_detect_hardware(self):
        wizard = InitWizard()
        profile = wizard.detect_hardware()
        assert isinstance(profile, HardwareProfile)
        assert profile.platform  # Should detect platform

    def test_suggest_config_high_vram(self):
        wizard = InitWizard()
        profile = HardwareProfile(gpu_vram_mb=24000, has_cuda=True)
        config = wizard.suggest_config(profile)
        assert config["template"] == "rtx4090"
        assert "32b" in config["core_model"]

    def test_suggest_config_mid_vram(self):
        wizard = InitWizard()
        profile = HardwareProfile(gpu_vram_mb=12000, has_cuda=True)
        config = wizard.suggest_config(profile)
        assert config["template"] == "rtx3060"

    def test_suggest_config_cpu_only(self):
        wizard = InitWizard()
        profile = HardwareProfile(has_cuda=False)
        config = wizard.suggest_config(profile)
        assert config["template"] == "cpu_only"

    def test_suggest_config_apple(self):
        wizard = InitWizard()
        profile = HardwareProfile(platform="Darwin", cpu_ram_mb=32000)
        config = wizard.suggest_config(profile)
        assert config["template"] == "apple_silicon"

    def test_generate_heads_yaml(self):
        wizard = InitWizard()
        config = {"template": "rtx4090", "core_model": "qwen3:32b", "worker_model": "qwen3:8b"}
        content = wizard.generate_heads_yaml(config)
        parsed = yaml.safe_load(content)
        assert len(parsed["heads"]) == 2
        assert parsed["heads"][0]["head_id"] == "mock-llm"
        assert parsed["heads"][1]["head_id"] == "core-llm"

    def test_run_interactive(self, tmp_path):
        wizard = InitWizard()
        result = wizard.run_interactive(tmp_path / "config")
        assert (tmp_path / "config" / "heads.yaml").exists()
        assert "profile" in result
        assert "suggestion" in result

    def test_generate_heads_yaml_from_template(self, tmp_path):
        """Prefer template file when templates_dir provided."""
        templates = tmp_path / "templates"
        templates.mkdir()
        (templates / "rtx4090.yaml").write_text(
            "heads:\n  - head_id: from-template\n"
            "    name: Test\n    adapter: mock\n"
            "    model: t\n    kind: llm\n"
        )
        wizard = InitWizard()
        config = {"template": "rtx4090", "core_model": "qwen3:32b", "worker_model": "qwen3:8b"}
        content = wizard.generate_heads_yaml(config, templates_dir=templates)
        parsed = yaml.safe_load(content)
        assert parsed["heads"][0]["head_id"] == "from-template"

    def test_generate_heads_yaml_fallback(self):
        """Fall back to dynamic generation when no template exists."""
        wizard = InitWizard()
        config = {"template": "nonexistent", "core_model": "qwen3:8b", "worker_model": "qwen3:4b"}
        content = wizard.generate_heads_yaml(config, templates_dir=Path("/tmp/no-such-dir"))
        parsed = yaml.safe_load(content)
        assert parsed["heads"][0]["head_id"] == "mock-llm"
        assert parsed["heads"][1]["head_id"] == "core-llm"


class TestAdapterStatus:
    def test_defaults(self):
        status = AdapterStatus()
        assert not status.ollama_available
        assert not status.transformers_available
        assert not status.claude_cli_available
        assert status.ollama_models == []

    def test_check_adapters_returns_status(self):
        wizard = InitWizard()
        status = wizard.check_adapters()
        assert isinstance(status, AdapterStatus)
        # On this machine torch is available
        assert isinstance(status.transformers_available, bool)
        assert isinstance(status.claude_cli_available, bool)


class TestGenerateEnvFile:
    def test_generates_env(self, tmp_path):
        wizard = InitWizard()
        env_path = tmp_path / ".env"
        profile = HardwareProfile(platform="Linux", gpu_name="RTX 4090")
        result = wizard.generate_env_file(profile, env_path)
        assert result is True
        assert env_path.exists()
        content = env_path.read_text()
        assert "MULTIHEAD_DATA_DIR" in content
        assert "RTX 4090" in content

    def test_never_clobbers_existing(self, tmp_path):
        wizard = InitWizard()
        env_path = tmp_path / ".env"
        env_path.write_text("EXISTING=true\n")
        profile = HardwareProfile(platform="Linux")
        result = wizard.generate_env_file(profile, env_path)
        assert result is False
        assert env_path.read_text() == "EXISTING=true\n"


class TestRunAuto:
    def test_run_auto_writes_config(self, tmp_path):
        wizard = InitWizard()
        config_dir = tmp_path / "config"
        env_path = tmp_path / ".env"
        result = wizard.run_auto(config_dir=config_dir, env_path=env_path)
        assert (config_dir / "heads.yaml").exists()
        assert (config_dir / "recipes").is_dir()
        assert env_path.exists()
        assert "profile" in result
        assert "adapters" in result
        assert "env_generated" in result
        assert result["env_generated"] is True

    def test_run_auto_with_templates(self, tmp_path):
        """Templates dir is used when available."""
        config_dir = tmp_path / "config"
        templates = config_dir / "templates"
        templates.mkdir(parents=True)
        # Create a cpu_only template (will be used since CI has no GPU usually)
        (templates / "cpu_only.yaml").write_text(
            "heads:\n  - head_id: template-head\n"
            "    name: T\n    adapter: mock\n"
            "    model: m\n    kind: llm\n"
        )
        wizard = InitWizard()
        # Force cpu_only suggestion
        wizard.detect_hardware = lambda: HardwareProfile(platform="Linux")
        result = wizard.run_auto(config_dir=config_dir, env_path=tmp_path / ".env")
        content = (config_dir / "heads.yaml").read_text()
        assert "template-head" in content

    def test_run_auto_preserves_env(self, tmp_path):
        """Existing .env is not overwritten."""
        config_dir = tmp_path / "config"
        env_path = tmp_path / ".env"
        env_path.write_text("KEEP=this\n")
        wizard = InitWizard()
        result = wizard.run_auto(config_dir=config_dir, env_path=env_path)
        assert result["env_generated"] is False
        assert env_path.read_text() == "KEEP=this\n"


class TestSolversYamlGeneration:
    """Tests for auto-generating solvers.yaml."""

    def test_generate_solvers_from_heads(self):
        """Should produce solvers.yaml from heads.yaml content."""
        import yaml
        wizard = InitWizard()
        heads_yaml = yaml.dump({
            "heads": [
                {
                    "head_id": "core-llm",
                    "name": "Core LLM",
                    "adapter": "ollama",
                    "model": "qwen3:8b",
                    "kind": "llm",
                    "gpu_required": True,
                },
            ]
        })
        result = wizard.generate_solvers_yaml(heads_yaml)
        parsed = yaml.safe_load(result)

        assert "solvers" in parsed
        solvers = parsed["solvers"]
        # core-llm + auto-added mock-llm
        core = next(s for s in solvers if s["solver_id"] == "core-llm")
        assert core["adapter"] == "ollama"
        assert core["kind"] == "llm"
        assert "capabilities" in core
        assert core["capabilities"]["accuracy_score"] == 0.71
        assert "mmlu" in core["capabilities"]["benchmarks"]
        assert core["privacy_level"] == "local"
        assert core["is_local"] is True

    def test_generate_solvers_includes_mock(self):
        """Mock head is auto-added if not present."""
        import yaml
        wizard = InitWizard()
        heads_yaml = yaml.dump({
            "heads": [
                {"head_id": "my-llm", "adapter": "ollama", "model": "qwen3:8b", "kind": "llm"},
            ]
        })
        result = wizard.generate_solvers_yaml(heads_yaml)
        parsed = yaml.safe_load(result)
        ids = [s["solver_id"] for s in parsed["solvers"]]
        assert "mock-llm" in ids

    def test_generate_solvers_known_cloud_model(self):
        """Cloud models should get encrypted privacy and correct capabilities."""
        import yaml
        wizard = InitWizard()
        heads_yaml = yaml.dump({
            "heads": [
                {"head_id": "claude-sonnet", "adapter": "claude", "model": "sonnet", "kind": "llm"},
            ]
        })
        result = wizard.generate_solvers_yaml(heads_yaml)
        parsed = yaml.safe_load(result)
        claude = next(s for s in parsed["solvers"] if s["solver_id"] == "claude-sonnet")
        assert claude["privacy_level"] == "encrypted"
        assert claude["capabilities"]["accuracy_score"] == 0.92
        assert "code_generation" in claude["capabilities"]["task_types"]

    def test_generate_solvers_unknown_model_defaults(self):
        """Unknown models should get safe defaults."""
        import yaml
        wizard = InitWizard()
        heads_yaml = yaml.dump({
            "heads": [
                {"head_id": "custom-llm", "adapter": "ollama",
                 "model": "custom:latest", "kind": "llm"},
            ]
        })
        result = wizard.generate_solvers_yaml(heads_yaml)
        parsed = yaml.safe_load(result)
        custom = next(s for s in parsed["solvers"] if s["solver_id"] == "custom-llm")
        assert custom["capabilities"]["accuracy_score"] == 0.5
        assert custom["capabilities"]["solver_type"] == "llm"

    def test_generate_solvers_vlm_defaults(self):
        """VLM kind should use VLM defaults for unknown models."""
        import yaml
        wizard = InitWizard()
        heads_yaml = yaml.dump({
            "heads": [
                {"head_id": "my-vlm", "adapter": "transformers",
                 "model": "custom-vlm", "kind": "vlm"},
            ]
        })
        result = wizard.generate_solvers_yaml(heads_yaml)
        parsed = yaml.safe_load(result)
        vlm = next(s for s in parsed["solvers"] if s["solver_id"] == "my-vlm")
        assert vlm["capabilities"]["solver_type"] == "vlm"
        assert "image" in vlm["capabilities"]["input_modalities"]

    def test_generate_solvers_empty_heads(self):
        """Empty heads list returns empty string."""
        wizard = InitWizard()
        assert wizard.generate_solvers_yaml("heads: []") == ""
        assert wizard.generate_solvers_yaml("invalid yaml: [[[") == ""

    def test_run_interactive_generates_solvers(self, tmp_path):
        """run_interactive should also generate solvers.yaml."""
        wizard = InitWizard()
        result = wizard.run_interactive(tmp_path)
        assert result["solvers_generated"] is True
        assert (tmp_path / "solvers.yaml").exists()
        import yaml
        parsed = yaml.safe_load((tmp_path / "solvers.yaml").read_text())
        assert "solvers" in parsed

    def test_run_auto_generates_solvers(self, tmp_path):
        """run_auto should also generate solvers.yaml."""
        config_dir = tmp_path / "config"
        wizard = InitWizard()
        result = wizard.run_auto(config_dir=config_dir, env_path=tmp_path / ".env")
        assert result["solvers_generated"] is True
        assert (config_dir / "solvers.yaml").exists()

    def test_vram_hint_propagated(self):
        """vram_hint_mb from heads should appear in solvers."""
        import yaml
        wizard = InitWizard()
        heads_yaml = yaml.dump({
            "heads": [
                {
                    "head_id": "gpu-llm", "adapter": "transformers",
                    "model": "Qwen/Qwen3-8B", "kind": "llm",
                    "gpu_required": True, "vram_hint_mb": 6000,
                    "quantization": "4bit",
                },
            ]
        })
        result = wizard.generate_solvers_yaml(heads_yaml)
        parsed = yaml.safe_load(result)
        gpu = next(s for s in parsed["solvers"] if s["solver_id"] == "gpu-llm")
        assert gpu["vram_hint_mb"] == 6000
        assert gpu["quantization"] == "4bit"
        assert gpu["capabilities"]["requires_gpu"] is True
        assert gpu["capabilities"]["vram_mb"] == 6000
