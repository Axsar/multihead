"""Fast TestEnhanceDocument tests — no popups, just edge cases."""

from __future__ import annotations

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
    """Fast edge-case tests for enhance_document."""

    @pytest.mark.asyncio
    async def test_missing_file_returns_empty(self, enhancer: ClaudeEnhancer, tmp_path: Path):
        result = await enhancer.enhance_document(tmp_path / "nonexistent.md")
        assert result == []

    @pytest.mark.asyncio
    async def test_empty_file_returns_empty(self, enhancer: ClaudeEnhancer, empty_doc: Path):
        result = await enhancer.enhance_document(empty_doc)
        assert result == []

    @pytest.mark.asyncio
    async def test_short_sections_returns_empty(self, enhancer: ClaudeEnhancer, short_doc: Path):
        result = await enhancer.enhance_document(short_doc)
        assert result == []
