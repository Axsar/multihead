"""Tests for Night Shift mid-stage checkpointing."""

from __future__ import annotations

from pathlib import Path

import pytest

from multihead.night_shift.checkpoint import (
    StageCheckpoint,
    clear_checkpoint,
    load_checkpoint,
    save_checkpoint,
)
from multihead.extractors.base import BaseExtractor
import multihead.extractors.base as base_mod


@pytest.fixture
def tmp_dir(tmp_path: Path) -> Path:
    return tmp_path


class TestCheckpointIO:
    """Phase 1: save/load/clear checkpoint files."""

    def test_save_and_load(self, tmp_dir: Path):
        ckpt = StageCheckpoint(
            stage_name="entity_extraction",
            processed_count=42,
            total_count=100,
            partial_results=[{"text": "hello"}, None, {"text": "world"}],
        )
        save_checkpoint(tmp_dir, ckpt)

        loaded = load_checkpoint(tmp_dir, "entity_extraction")
        assert loaded is not None
        assert loaded.processed_count == 42
        assert loaded.total_count == 100
        assert len(loaded.partial_results) == 3
        assert loaded.updated_at != ""

    def test_load_nonexistent_returns_none(self, tmp_dir: Path):
        assert load_checkpoint(tmp_dir, "no_such_stage") is None

    def test_load_corrupt_returns_none(self, tmp_dir: Path):
        path = tmp_dir / "bad_stage_checkpoint.json"
        path.write_text("not json at all")
        assert load_checkpoint(tmp_dir, "bad_stage") is None

    def test_clear_removes_file(self, tmp_dir: Path):
        ckpt = StageCheckpoint(stage_name="test", processed_count=1, total_count=1)
        save_checkpoint(tmp_dir, ckpt)
        assert (tmp_dir / "test_checkpoint.json").exists()

        clear_checkpoint(tmp_dir, "test")
        assert not (tmp_dir / "test_checkpoint.json").exists()

    def test_clear_nonexistent_is_fine(self, tmp_dir: Path):
        clear_checkpoint(tmp_dir, "nonexistent")  # Should not raise


class TestMapGenerateCheckpoint:
    """Phase 2: map_generate resumes from checkpoint."""

    @pytest.mark.asyncio
    async def test_resumes_from_checkpoint(self, tmp_dir: Path):
        """If checkpoint exists, skip already-processed prompts."""
        # Pre-populate checkpoint: 2 of 4 done
        ckpt = StageCheckpoint(
            stage_name="test_stage",
            processed_count=2,
            total_count=4,
            partial_results=[{"text": "r0"}, {"text": "r1"}],
        )
        save_checkpoint(tmp_dir, ckpt)

        call_count = 0

        async def fake_generate(prompt: str) -> dict:
            nonlocal call_count
            call_count += 1
            return {"text": f"result_{call_count}"}

        prompts = ["p0", "p1", "p2", "p3"]
        results = await BaseExtractor.map_generate(
            fake_generate, prompts,
            checkpoint_dir=tmp_dir, stage_name="test_stage",
        )

        # Only 2 new calls (p2, p3), not 4
        assert call_count == 2
        # But all 4 results are present
        assert len(results) == 4
        # First two are from checkpoint
        assert results[0] == {"text": "r0"}
        assert results[1] == {"text": "r1"}
        # Checkpoint should be cleared on completion
        assert load_checkpoint(tmp_dir, "test_stage") is None

    @pytest.mark.asyncio
    async def test_no_checkpoint_runs_all(self, tmp_dir: Path):
        """Without checkpoint, processes all prompts."""
        call_count = 0

        async def fake_generate(prompt: str) -> dict:
            nonlocal call_count
            call_count += 1
            return {"text": f"r{call_count}"}

        results = await BaseExtractor.map_generate(
            fake_generate, ["a", "b", "c"],
            checkpoint_dir=tmp_dir, stage_name="fresh_stage",
        )
        assert call_count == 3
        assert len(results) == 3

    @pytest.mark.asyncio
    async def test_checkpoint_saved_on_each_call(self, tmp_dir: Path):
        """Checkpoint is updated after each LLM call."""
        call_index = 0

        async def fake_generate(prompt: str) -> dict:
            nonlocal call_index
            call_index += 1
            # Check checkpoint state mid-run (after first call)
            if call_index == 2:
                ckpt = load_checkpoint(tmp_dir, "incremental")
                assert ckpt is not None
                assert ckpt.processed_count == 1
            return {"text": f"r{call_index}"}

        await BaseExtractor.map_generate(
            fake_generate, ["a", "b", "c"],
            checkpoint_dir=tmp_dir, stage_name="incremental",
        )

    @pytest.mark.asyncio
    async def test_stale_checkpoint_ignored(self, tmp_dir: Path):
        """Checkpoint with mismatched total_count is ignored."""
        ckpt = StageCheckpoint(
            stage_name="mismatch",
            processed_count=2,
            total_count=10,  # Different from actual prompt count
            partial_results=[{"text": "old1"}, {"text": "old2"}],
        )
        save_checkpoint(tmp_dir, ckpt)

        call_count = 0

        async def fake_generate(prompt: str) -> dict:
            nonlocal call_count
            call_count += 1
            return {"text": f"r{call_count}"}

        results = await BaseExtractor.map_generate(
            fake_generate, ["a", "b", "c"],  # 3 prompts, not 10
            checkpoint_dir=tmp_dir, stage_name="mismatch",
        )
        # Should process all 3 (checkpoint total_count mismatch)
        assert call_count == 3
        assert len(results) == 3


class TestShutdownFlag:
    """Phase 4: shutdown_requested flag causes clean exit."""

    @pytest.mark.asyncio
    async def test_shutdown_saves_checkpoint_and_raises(self, tmp_dir: Path):
        """Setting shutdown_requested mid-run saves checkpoint and raises InterruptedError."""
        call_count = 0
        original = base_mod.shutdown_requested

        async def fake_generate(prompt: str) -> dict:
            nonlocal call_count
            call_count += 1
            if call_count == 2:
                base_mod.shutdown_requested = True
            return {"text": f"r{call_count}"}

        try:
            with pytest.raises(InterruptedError, match="Shutdown at 2/4"):
                await BaseExtractor.map_generate(
                    fake_generate, ["a", "b", "c", "d"],
                    checkpoint_dir=tmp_dir, stage_name="shutdown_test",
                )

            # Only 2 calls made before shutdown
            assert call_count == 2

            # Checkpoint was saved
            ckpt = load_checkpoint(tmp_dir, "shutdown_test")
            assert ckpt is not None
            assert ckpt.processed_count == 2
            assert len(ckpt.partial_results) == 2
        finally:
            base_mod.shutdown_requested = original

    @pytest.mark.asyncio
    async def test_resume_after_shutdown(self, tmp_dir: Path):
        """After shutdown + resume, completes the remaining work."""
        # Simulate: first run saved checkpoint at 2/4
        ckpt = StageCheckpoint(
            stage_name="resume_test",
            processed_count=2,
            total_count=4,
            partial_results=[{"text": "r1"}, {"text": "r2"}],
        )
        save_checkpoint(tmp_dir, ckpt)

        call_count = 0

        async def fake_generate(prompt: str) -> dict:
            nonlocal call_count
            call_count += 1
            return {"text": f"resumed_{call_count}"}

        results = await BaseExtractor.map_generate(
            fake_generate, ["a", "b", "c", "d"],
            checkpoint_dir=tmp_dir, stage_name="resume_test",
        )

        assert call_count == 2  # Only c and d
        assert len(results) == 4
        assert results[0] == {"text": "r1"}  # From checkpoint
        assert results[2] == {"text": "resumed_1"}  # New
