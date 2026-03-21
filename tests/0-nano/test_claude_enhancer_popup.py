"""TestEnhanceDocument popup tests — these call enhance_document with real docs."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from multihead.narrative.claude_enhancer import ClaudeEnhancer


@pytest.fixture
def enhancer():
    return ClaudeEnhancer(
        acp_url="http://localhost:8000/api/v1",
        api_key="test-token",
        project_id="test_project",
        poll_interval=0.1,
        max_wait=5.0,
    )


@pytest.fixture
def plan_doc(tmp_path: Path) -> Path:
    doc = tmp_path / "PLAN.md"
    doc.write_text(
        "# Project Expansion Plan\n\n"
        "## Phase 1: Foundation\n\n"
        "- Install dependencies and configure the base environment\n"
        "- Create the database schema for character tracking tables\n"
        "- Set up the VLM inference pipeline for panel detection\n"
        "\n"
        "## Phase 2: Detection Pipeline\n\n"
        "- [x] Implement YOLO-based panel detection model\n"
        "- [ ] Add speech bubble segmentation using SAM\n"
        "- Integrate OCR engine for text extraction from bubbles\n"
        "- Must support both English and Japanese text extraction\n"
        "\n"
        "## Phase 3: Assembly\n\n"
        "- Build narrative assembly pipeline from detected elements\n"
        "- Implement speaker attribution using character embeddings\n"
        "- Create reading order algorithm for panel sequencing\n",
        encoding="utf-8",
    )
    return doc


@pytest.fixture
def short_doc(tmp_path: Path) -> Path:
    doc = tmp_path / "SHORT.md"
    doc.write_text(
        "# Title\n\n"
        "## Section A\n\n"
        "Too short\n"
        "\n"
        "## Section B\n\n"
        "Also very short\n",
        encoding="utf-8",
    )
    return doc


@pytest.fixture
def empty_doc(tmp_path: Path) -> Path:
    doc = tmp_path / "EMPTY.md"
    doc.write_text("", encoding="utf-8")
    return doc


class TestEnhanceDocument:
    """Integration tests with mocked ACP calls — cause popups."""

    @pytest.mark.asyncio
    async def test_full_flow_with_mocked_acp(self, enhancer: ClaudeEnhancer, plan_doc: Path):
        """Full flow: split -> submit -> poll -> parse -> merge."""
        task_counter = {"n": 0}

        async def mock_create_task(prompt: str) -> tuple[str, str]:
            task_counter["n"] += 1
            tid = f"task-{task_counter['n']}"
            response = json.dumps({
                "claims": [
                    {
                        "text": f"Extracted claim from {tid} with enough detail",
                        "claim_type": "plan",
                        "predicate": "planned",
                        "confidence": 0.80,
                        "entities": ["test"],
                        "reasoning": "test",
                    }
                ]
            })
            return tid, response

        enhancer._client.create_task = mock_create_task

        artifacts = await enhancer.enhance_document(
            plan_doc, doc_type="plan", source_project="h2v",
            synthesize=False,
        )

        assert len(artifacts) == 1
        claims = artifacts[0]["claims"]
        assert len(claims) >= 3
        assert artifacts[0]["record"].uri.startswith("markdown+claude://")
        assert artifacts[0]["event"] is not None

    @pytest.mark.asyncio
    async def test_merges_heuristic_and_claude(self, enhancer: ClaudeEnhancer, plan_doc: Path):
        """Heuristic claims are passed through and merged."""
        from multihead.narrative.source_extractors.markdown_extractor import MarkdownExtractor

        md_ext = MarkdownExtractor(project_id="h2v")
        heuristic_arts = md_ext.extract_from_file(plan_doc, doc_type="plan", source_project="h2v")
        heuristic_claims = []
        for art in heuristic_arts:
            heuristic_claims.extend(art.get("claims", []))

        async def mock_create_task(prompt: str) -> tuple[str, str]:
            response = json.dumps({
                "claims": [{
                    "text": "A unique Claude-only claim about architecture decisions",
                    "claim_type": "decision",
                    "predicate": "decided",
                    "confidence": 0.75,
                }]
            })
            return "task-mock", response

        enhancer._client.create_task = mock_create_task

        artifacts = await enhancer.enhance_document(
            plan_doc, doc_type="plan", source_project="h2v",
            heuristic_claims=heuristic_claims,
            synthesize=False,
        )

        merged_claims = artifacts[0]["claims"]
        heuristic_keys = {c.canonical.claim_key for c in heuristic_claims}
        claude_keys = {
            c.canonical.claim_key for c in merged_claims
            if c.canonical.claim_key.startswith("claude.")
        }
        assert len(heuristic_keys & {c.canonical.claim_key for c in merged_claims}) > 0
        assert len(claude_keys) > 0

    @pytest.mark.asyncio
    async def test_handles_empty_responses(self, enhancer: ClaudeEnhancer, plan_doc: Path):
        """Empty responses from _create_task are skipped gracefully."""
        async def mock_create_task(prompt: str) -> tuple[str, str]:
            return "task-empty", ""

        enhancer._client.create_task = mock_create_task

        artifacts = await enhancer.enhance_document(
            plan_doc, doc_type="plan", synthesize=False,
        )
        assert len(artifacts) == 1
        assert len(artifacts[0]["claims"]) == 0

    @pytest.mark.asyncio
    async def test_handles_task_creation_errors(self, enhancer: ClaudeEnhancer, plan_doc: Path):
        """Errors during task creation are handled gracefully."""
        import httpx

        async def mock_create_task(prompt: str) -> tuple[str, str]:
            raise httpx.ConnectError("Connection refused")

        enhancer._client.create_task = mock_create_task

        artifacts = await enhancer.enhance_document(
            plan_doc, doc_type="plan", synthesize=False,
        )
        assert artifacts == []
