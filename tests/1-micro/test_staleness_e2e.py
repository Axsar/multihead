"""End-to-end staleness test: modify a file, verify claims go stale.

Expert requirement #4: modify a file, verify staleness sweep detects it.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pytest


def _insert_test_claim(ks, claim_id: str, claim_key: str, statement: str, prov_dict: dict):
    """Insert a minimal test claim directly into the DB."""
    now = datetime.now(timezone.utc).isoformat()
    prov = json.dumps(prov_dict)
    with ks._connect() as conn:
        conn.execute(
            """INSERT INTO claims (
                claim_id, claim_status, claim_type, scope_type, scope_id,
                visibility, valid_from, claim_key, predicate, subject_json,
                object_json, statement, confidence, provenance_json,
                derived_from_json, related_json, conflicts_json,
                created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                claim_id, "proposed", "fact", "project", "test",
                "private", now, claim_key, "has_signature", "{}",
                "{}", statement, 0.95, prov,
                "[]", "[]", "[]",
                now, now,
            ),
        )


@pytest.fixture
def git_repo(tmp_path):
    """Create a temporary git repo with a Python file and initial commit."""
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init"], cwd=repo, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@test.com"], cwd=repo, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test"], cwd=repo, capture_output=True)

    # Create initial file and commit
    py_file = repo / "module.py"
    py_file.write_text("def hello():\n    return 'world'\n")
    subprocess.run(["git", "add", "."], cwd=repo, capture_output=True)
    subprocess.run(["git", "commit", "-m", "initial"], cwd=repo, capture_output=True)

    # Record initial SHA
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True,
    )
    initial_sha = result.stdout.strip()

    # Modify the file and commit again
    py_file.write_text("def hello():\n    return 'changed'\n")
    subprocess.run(["git", "add", "."], cwd=repo, capture_output=True)
    subprocess.run(["git", "commit", "-m", "modify"], cwd=repo, capture_output=True)

    return repo, initial_sha


def test_staleness_sweep_detects_changed_file(git_repo):
    """Claims anchored to old SHA should be marked stale after file changes."""
    repo, old_sha = git_repo

    from multihead.knowledge_store import KnowledgeStore

    db_path = repo / "test_knowledge.db"
    ks = KnowledgeStore(db_path)

    _insert_test_claim(
        ks,
        claim_id="clm_test_stale_1",
        claim_key="module.hello.signature",
        statement="Function hello() defined at line 1. Returns 'world' string literal.",
        prov_dict={
            "produced_by": {"kind": "extractor", "id": "code_reader"},
            "observation_method": "code_read",
            "source_anchor": {"file_path": "module.py", "git_sha": old_sha},
        },
    )

    # Verify claim is proposed
    with ks._connect() as conn:
        row = conn.execute(
            "SELECT claim_status FROM claims WHERE claim_id = 'clm_test_stale_1'"
        ).fetchone()
        assert row["claim_status"] == "proposed"

    # Run staleness sweep
    import asyncio
    import os
    from multihead.night_shift.stages_late import LateStagesMixin

    class FakePipeline(LateStagesMixin):
        def __init__(self, knowledge, output_dir):
            self.knowledge = knowledge
            self.output_dir = output_dir

    # Set project roots to include the test repo
    old_roots = os.environ.get("MULTIHEAD_PROJECT_ROOTS", "")
    os.environ["MULTIHEAD_PROJECT_ROOTS"] = str(repo) + "/"
    try:
        output_dir = repo / "output" / "nightshift"
        output_dir.mkdir(parents=True, exist_ok=True)
        pipeline = FakePipeline(ks, output_dir)
        result = asyncio.run(pipeline._stage_staleness_sweep({}))
    finally:
        if old_roots:
            os.environ["MULTIHEAD_PROJECT_ROOTS"] = old_roots
        else:
            os.environ.pop("MULTIHEAD_PROJECT_ROOTS", None)

    assert result["metrics"]["marked_stale"] >= 1
    assert result["metrics"]["checked"] >= 1

    with ks._connect() as conn:
        row = conn.execute(
            "SELECT claim_status FROM claims WHERE claim_id = 'clm_test_stale_1'"
        ).fetchone()
        assert row["claim_status"] == "stale", (
            f"Expected stale after file change, got: {row['claim_status']}"
        )


def test_staleness_sweep_ignores_unchanged_file(git_repo):
    """Claims for unchanged files should NOT be marked stale."""
    repo, _ = git_repo

    # Add another file that wasn't changed between first two commits
    unchanged = repo / "stable.py"
    unchanged.write_text("CONST = 42\n")
    subprocess.run(["git", "add", "."], cwd=repo, capture_output=True)
    subprocess.run(["git", "commit", "-m", "add stable"], cwd=repo, capture_output=True)

    # Get HEAD sha
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, capture_output=True, text=True,
    )
    head_sha = result.stdout.strip()

    from multihead.knowledge_store import KnowledgeStore

    db_path = repo / "test_knowledge2.db"
    ks = KnowledgeStore(db_path)

    _insert_test_claim(
        ks,
        claim_id="clm_test_stable_1",
        claim_key="stable.CONST.definition",
        statement="Module stable.py defines constant CONST with value 42 at module level.",
        prov_dict={
            "produced_by": {"kind": "extractor", "id": "code_reader"},
            "observation_method": "code_read",
            "source_anchor": {"file_path": "stable.py", "git_sha": head_sha},
        },
    )

    import asyncio
    from multihead.night_shift.stages_late import LateStagesMixin

    class FakePipeline(LateStagesMixin):
        def __init__(self, knowledge, output_dir):
            self.knowledge = knowledge
            self.output_dir = output_dir

    # Set project roots to include the test repo
    import os
    old_roots = os.environ.get("MULTIHEAD_PROJECT_ROOTS", "")
    os.environ["MULTIHEAD_PROJECT_ROOTS"] = str(repo) + "/"
    try:
        output_dir = repo / "output" / "nightshift"
        output_dir.mkdir(parents=True, exist_ok=True)
        pipeline = FakePipeline(ks, output_dir)
        result = asyncio.run(pipeline._stage_staleness_sweep({}))
    finally:
        if old_roots:
            os.environ["MULTIHEAD_PROJECT_ROOTS"] = old_roots
        else:
            os.environ.pop("MULTIHEAD_PROJECT_ROOTS", None)

    with ks._connect() as conn:
        row = conn.execute(
            "SELECT claim_status FROM claims WHERE claim_id = 'clm_test_stable_1'"
        ).fetchone()
        assert row["claim_status"] == "proposed", (
            f"Expected proposed (file unchanged), got: {row['claim_status']}"
        )
