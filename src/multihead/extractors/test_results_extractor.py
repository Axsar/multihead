"""Local test results extractor — captures pytest output as claims.

Parses pytest JUnit XML or stdout to produce claims with
observation_method='test_result' at 0.95 confidence.

Usage:
  # Run tests with JUnit output, then extract:
  python -m pytest tests/ --junitxml=test-results.xml
  python -m multihead extract-tests test-results.xml

  # Or parse pytest stdout directly:
  python -m pytest tests/ -v 2>&1 | python -m multihead.extractors.test_results_extractor
"""

from __future__ import annotations

import logging
import re
import subprocess
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def extract_from_junit_xml(xml_path: str | Path) -> list[dict[str, Any]]:
    """Extract test result claims from JUnit XML output.

    Each test case produces a claim: pass (0.95) or fail (0.95 negative).
    """
    claims: list[dict[str, Any]] = []
    path = Path(xml_path)
    if not path.exists():
        logger.error("JUnit XML not found: %s", path)
        return []

    try:
        tree = ET.parse(path)
    except ET.ParseError as e:
        logger.error("Failed to parse JUnit XML: %s", e)
        return []

    root = tree.getroot()

    # Get summary — pytest uses <testsuites><testsuite> structure
    summary_elem = root
    if root.tag == "testsuites":
        ts = root.find("testsuite")
        if ts is not None:
            summary_elem = ts

    total_tests = int(summary_elem.get("tests", 0))
    total_failures = int(summary_elem.get("failures", 0))
    total_errors = int(summary_elem.get("errors", 0))
    total_time = float(summary_elem.get("time", 0))

    # Summary claim
    passed = total_tests - total_failures - total_errors
    stmt = (
        f"Local test run: {passed}/{total_tests} tests passed "
        f"({total_failures} failures, {total_errors} errors) in {total_time:.1f}s"
    )
    if len(stmt) >= 50:
        claims.append({
            "claim_type": "fact",
            "claim_key": "test_run.summary.latest",
            "statement": stmt,
            "confidence": 0.95,
            "observation_method": "test_result",
            "speaker": "tool",
            "evidence": [{
                "type": "test_run",
                "text": f"{passed}/{total_tests} passed, {total_failures} failed, {total_errors} errors in {total_time:.1f}s",
            }],
            "durability": "session",
        })

    # Per-test claims for failures (most valuable signal)
    for testsuite in root.iter("testsuite"):
        for testcase in testsuite.iter("testcase"):
            name = testcase.get("name", "")
            classname = testcase.get("classname", "")
            time_taken = float(testcase.get("time", 0))

            failure = testcase.find("failure")
            error = testcase.find("error")

            if failure is not None:
                msg = failure.get("message", "")[:200]
                stmt = f"Test FAILED: {classname}::{name} — {msg}"
                if len(stmt) >= 50:
                    claims.append({
                        "claim_type": "fact",
                        "claim_key": f"test.{classname}.{name}".replace(" ", "_"),
                        "statement": stmt,
                        "confidence": 0.95,
                        "observation_method": "test_result",
                        "speaker": "tool",
                        "evidence": [{
                            "type": "test_failure",
                            "text": f"{classname}::{name}: {msg}",
                        }],
                        "durability": "session",
                    })
            elif error is not None:
                msg = error.get("message", "")[:200]
                stmt = f"Test ERROR: {classname}::{name} — {msg}"
                if len(stmt) >= 50:
                    claims.append({
                        "claim_type": "fact",
                        "claim_key": f"test.{classname}.{name}".replace(" ", "_"),
                        "statement": stmt,
                        "confidence": 0.95,
                        "observation_method": "test_result",
                        "speaker": "tool",
                        "evidence": [{
                            "type": "test_error",
                            "text": f"{classname}::{name}: {msg}",
                        }],
                        "durability": "session",
                    })

    # Per-module pass summary (aggregate passing tests by module)
    module_pass_counts: dict[str, int] = {}
    module_total_counts: dict[str, int] = {}
    for testsuite in root.iter("testsuite"):
        for testcase in testsuite.iter("testcase"):
            classname = testcase.get("classname", "")
            module = classname.rsplit(".", 1)[0] if "." in classname else classname
            module_total_counts[module] = module_total_counts.get(module, 0) + 1
            if testcase.find("failure") is None and testcase.find("error") is None:
                module_pass_counts[module] = module_pass_counts.get(module, 0) + 1

    for module, total in module_total_counts.items():
        passed = module_pass_counts.get(module, 0)
        if total >= 3:  # Only report modules with meaningful test count
            stmt = f"Test module {module}: {passed}/{total} tests passing"
            if len(stmt) >= 50:
                claims.append({
                    "claim_type": "fact",
                    "claim_key": f"test.module.{module}",
                    "statement": stmt,
                    "confidence": 0.95,
                    "observation_method": "test_result",
                    "speaker": "tool",
                    "evidence": [{
                        "type": "test_module_summary",
                        "text": f"{module}: {passed}/{total} passed",
                    }],
                    "durability": "session",
                })

    logger.info(
        "Test results: %d claims from %s (%d tests, %d failures)",
        len(claims), path.name, total_tests, total_failures,
    )
    return claims


def extract_from_pytest_stdout(output: str) -> list[dict[str, Any]]:
    """Parse pytest -v stdout for pass/fail results.

    Fallback when JUnit XML is not available.
    """
    claims: list[dict[str, Any]] = []

    # Parse summary line: "2631 passed, 6 warnings in 482.37s"
    summary_match = re.search(
        r"(\d+) passed(?:, (\d+) failed)?(?:, (\d+) error)?.*?in ([\d.]+)s",
        output,
    )
    if summary_match:
        passed = int(summary_match.group(1))
        failed = int(summary_match.group(2) or 0)
        errors = int(summary_match.group(3) or 0)
        duration = float(summary_match.group(4))
        total = passed + failed + errors

        stmt = (
            f"Local test run: {passed}/{total} tests passed "
            f"({failed} failures, {errors} errors) in {duration:.1f}s"
        )
        if len(stmt) >= 50:
            claims.append({
                "claim_type": "fact",
                "claim_key": "test_run.summary.latest",
                "statement": stmt,
                "confidence": 0.95,
                "observation_method": "test_result",
                "speaker": "tool",
                "evidence": [{
                    "type": "test_run",
                    "total": total,
                    "passed": passed,
                    "failed": failed,
                    "errors": errors,
                    "duration": duration,
                }],
                "durability": "session",
            })

    # Parse individual FAILED lines
    for match in re.finditer(r"FAILED ([\w/._:-]+)", output):
        test_id = match.group(1)
        stmt = f"Test FAILED: {test_id} — failed during local pytest run"
        if len(stmt) >= 50:
            claims.append({
                "claim_type": "fact",
                "claim_key": f"test.{test_id.replace('/', '.').replace('::', '.')}",
                "statement": stmt,
                "confidence": 0.95,
                "observation_method": "test_result",
                "speaker": "tool",
                "evidence": [{"type": "test_failure", "text": test_id}],
                "durability": "session",
            })

    logger.info("Parsed %d claims from pytest stdout", len(claims))
    return claims


def run_and_extract(
    test_path: str = "tests/0-nano",
    extra_args: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Run pytest with JUnit XML output and extract claims.

    Convenience function that runs tests and returns claims.
    """
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as f:
        xml_path = f.name

    args = [
        "python", "-m", "pytest", test_path,
        f"--junitxml={xml_path}",
        "-q", "--tb=no",
    ]
    if extra_args:
        args.extend(extra_args)

    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=600)
        claims = extract_from_junit_xml(xml_path)

        # Also parse stdout for any extra info
        if not claims:
            claims = extract_from_pytest_stdout(result.stdout + result.stderr)

        return claims
    except subprocess.TimeoutExpired:
        logger.error("pytest timed out after 600s")
        return []
    finally:
        Path(xml_path).unlink(missing_ok=True)
