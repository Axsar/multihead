"""Tests for diagnostics module."""



from multihead.diagnostics import DiagnosticReport, DiagnosticResult, Diagnostics


class TestDiagnosticResult:
    def test_passed(self):
        r = DiagnosticResult("test", True, "OK")
        assert r.passed is True
        assert r.suggestion == ""

    def test_failed_with_suggestion(self):
        r = DiagnosticResult("test", False, "Missing", "Install it")
        assert r.passed is False
        assert r.suggestion == "Install it"


class TestDiagnosticReport:
    def test_all_passed(self):
        report = DiagnosticReport(checks=[
            DiagnosticResult("a", True, "ok"),
            DiagnosticResult("b", True, "ok"),
        ])
        assert report.all_passed is True

    def test_some_failed(self):
        report = DiagnosticReport(checks=[
            DiagnosticResult("a", True, "ok"),
            DiagnosticResult("b", False, "fail"),
        ])
        assert report.all_passed is False

    def test_summary(self):
        report = DiagnosticReport(checks=[
            DiagnosticResult("a", True, "ok"),
            DiagnosticResult("b", False, "fail"),
            DiagnosticResult("c", True, "ok"),
        ])
        assert report.summary == "2/3 checks passed"

    def test_empty(self):
        report = DiagnosticReport()
        assert report.all_passed is True
        assert report.summary == "0/0 checks passed"


class TestDiagnostics:
    def test_data_dir_exists(self, tmp_path):
        diag = Diagnostics(tmp_path, tmp_path)
        result = diag._check_data_dir()
        assert result.passed is True

    def test_data_dir_missing(self, tmp_path):
        diag = Diagnostics(tmp_path / "nonexistent", tmp_path)
        result = diag._check_data_dir()
        assert result.passed is False
        assert "init" in result.suggestion.lower()

    def test_config_dir_exists(self, tmp_path):
        diag = Diagnostics(tmp_path, tmp_path)
        result = diag._check_config_dir()
        assert result.passed is True

    def test_config_dir_missing(self, tmp_path):
        diag = Diagnostics(tmp_path, tmp_path / "nonexistent")
        result = diag._check_config_dir()
        assert result.passed is False

    def test_heads_yaml_found(self, tmp_path):
        (tmp_path / "heads.yaml").write_text("heads: []")
        diag = Diagnostics(tmp_path, tmp_path)
        result = diag._check_heads_yaml()
        assert result.passed is True

    def test_heads_yaml_missing(self, tmp_path):
        diag = Diagnostics(tmp_path, tmp_path)
        result = diag._check_heads_yaml()
        assert result.passed is False

    def test_disk_space_check(self, tmp_path):
        diag = Diagnostics(tmp_path, tmp_path)
        result = diag._check_disk_space()
        # Should either pass or fail, not crash
        assert isinstance(result.passed, bool)

    def test_python_deps_check(self, tmp_path):
        diag = Diagnostics(tmp_path, tmp_path)
        result = diag._check_python_deps()
        # fastapi, pydantic, httpx should all be installed in test env
        assert result.passed is True

    def test_run_all(self, tmp_path):
        (tmp_path / "heads.yaml").write_text("heads: []")
        diag = Diagnostics(tmp_path, tmp_path)
        report = diag.run_all()
        assert len(report.checks) == 12
        # data_dir, config_dir, heads_yaml should pass
        names = {c.name: c.passed for c in report.checks}
        assert names["data_dir"] is True
        assert names["config_dir"] is True
        assert names["heads_yaml"] is True

    def test_check_torch(self, tmp_path):
        diag = Diagnostics(tmp_path, tmp_path)
        result = diag._check_torch()
        assert isinstance(result.passed, bool)
        assert result.name == "torch_cuda"

    def test_check_claude_cli(self, tmp_path):
        diag = Diagnostics(tmp_path, tmp_path)
        result = diag._check_claude_cli()
        assert isinstance(result.passed, bool)
        assert result.name == "claude_cli"

    def test_check_env_file_missing(self, tmp_path):
        import os
        old_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            diag = Diagnostics(tmp_path, tmp_path)
            result = diag._check_env_file()
            assert result.passed is False
        finally:
            os.chdir(old_cwd)

    def test_check_env_file_present(self, tmp_path):
        import os
        old_cwd = os.getcwd()
        os.chdir(tmp_path)
        try:
            (tmp_path / ".env").write_text("TEST=1\n")
            diag = Diagnostics(tmp_path, tmp_path)
            result = diag._check_env_file()
            assert result.passed is True
        finally:
            os.chdir(old_cwd)
