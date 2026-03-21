"""Tests for ConversationContext — conversation persistence across compaction."""

from __future__ import annotations

from multihead.conversation_context import (
    ConversationContext,
    _extract_first_sentence,
    _get_content,
    _get_role,
    _truncate,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _msg(role: str, content: str) -> dict:
    """Simple message dict for testing."""
    return {"role": role, "content": content}


def _conversation(turns: int) -> list[dict]:
    """Generate a multi-turn conversation."""
    msgs = []
    for i in range(1, turns + 1):
        msgs.append(_msg("user", f"Question {i} about topic {i}"))
        msgs.append(_msg("assistant", f"Answer {i}. Here is more detail about topic {i}."))
    return msgs


# ---------------------------------------------------------------------------
# ConversationContext — init and state
# ---------------------------------------------------------------------------


class TestInit:
    def test_defaults(self):
        ctx = ConversationContext()
        assert ctx.turn_count == 0
        assert ctx.summary == ""
        assert not ctx.needs_summary_refresh()

    def test_custom_params(self):
        ctx = ConversationContext(
            recent_count=10,
            summary_interval=5,
            max_summary_chars=1000,
            max_recent_chars=2000,
        )
        assert ctx._recent_count == 10
        assert ctx._summary_interval == 5


class TestOnTurn:
    def test_increments_turn_count(self):
        ctx = ConversationContext()
        ctx.on_turn("hello", "hi there")
        assert ctx.turn_count == 1
        ctx.on_turn("another", "response")
        assert ctx.turn_count == 2

    def test_needs_summary_refresh_at_interval(self):
        ctx = ConversationContext(summary_interval=3)
        ctx.on_turn("a", "b")
        assert not ctx.needs_summary_refresh()
        ctx.on_turn("c", "d")
        assert not ctx.needs_summary_refresh()
        ctx.on_turn("e", "f")
        assert ctx.needs_summary_refresh()

    def test_summary_refresh_resets_counter(self):
        ctx = ConversationContext(summary_interval=2)
        ctx.on_turn("a", "b")
        ctx.on_turn("c", "d")
        assert ctx.needs_summary_refresh()
        ctx.build_summary(_conversation(2))
        assert not ctx.needs_summary_refresh()


# ---------------------------------------------------------------------------
# build_summary
# ---------------------------------------------------------------------------


class TestBuildSummary:
    def test_empty_messages(self):
        ctx = ConversationContext()
        result = ctx.build_summary([])
        assert result == ""
        assert ctx.summary == ""

    def test_single_turn(self):
        ctx = ConversationContext()
        msgs = [
            _msg("user", "What is MultiHead?"),
            _msg("assistant", "MultiHead is a local multimodal task-runner. It does many things."),
        ]
        result = ctx.build_summary(msgs)
        assert "What is MultiHead?" in result
        assert "MultiHead is a local multimodal task-runner" in result
        assert ctx.summary == result

    def test_multiple_turns_selects_anchors(self):
        ctx = ConversationContext(summary_interval=10)
        msgs = _conversation(20)
        result = ctx.build_summary(msgs)
        # Should include first turn
        assert "Question 1" in result
        # Should include last turn
        assert "Question 20" in result

    def test_respects_char_budget(self):
        ctx = ConversationContext(max_summary_chars=100)
        msgs = _conversation(50)
        result = ctx.build_summary(msgs)
        assert len(result) <= 200  # some slack for line formatting

    def test_resets_turns_since_summary(self):
        ctx = ConversationContext(summary_interval=3)
        ctx.on_turn("a", "b")
        ctx.on_turn("c", "d")
        ctx.on_turn("e", "f")
        assert ctx.needs_summary_refresh()
        ctx.build_summary(_conversation(3))
        assert not ctx.needs_summary_refresh()

    def test_user_only_messages(self):
        ctx = ConversationContext()
        msgs = [_msg("user", "Hello"), _msg("user", "Are you there?")]
        result = ctx.build_summary(msgs)
        assert "Hello" in result

    def test_skips_system_messages(self):
        ctx = ConversationContext()
        msgs = [
            _msg("system", "Context compacted"),
            _msg("user", "What happened?"),
            _msg("assistant", "The context was compacted. Let me continue."),
        ]
        result = ctx.build_summary(msgs)
        assert "Context compacted" not in result
        assert "What happened?" in result


# ---------------------------------------------------------------------------
# build_recent_window
# ---------------------------------------------------------------------------


class TestBuildRecentWindow:
    def test_empty_messages(self):
        ctx = ConversationContext()
        assert ctx.build_recent_window([]) == ""

    def test_returns_last_k_messages(self):
        ctx = ConversationContext(recent_count=4)
        msgs = _conversation(10)  # 20 messages total
        result = ctx.build_recent_window(msgs)
        # Should contain the last 4 messages (turns 9-10)
        assert "Question 10" in result or "Answer 10" in result

    def test_truncates_long_messages(self):
        ctx = ConversationContext(recent_count=2)
        msgs = [
            _msg("user", "x" * 500),
            _msg("assistant", "y" * 500),
        ]
        result = ctx.build_recent_window(msgs)
        # Each message should be truncated
        assert "..." in result
        assert len(result) < 1000

    def test_skips_system_messages(self):
        ctx = ConversationContext(recent_count=4)
        msgs = [
            _msg("user", "Hello"),
            _msg("assistant", "Hi"),
            _msg("system", "Compacted"),
            _msg("user", "Continue"),
            _msg("assistant", "Sure"),
        ]
        result = ctx.build_recent_window(msgs)
        assert "Compacted" not in result
        assert "User:" in result
        assert "Assistant:" in result

    def test_fewer_messages_than_count(self):
        ctx = ConversationContext(recent_count=10)
        msgs = _conversation(2)  # only 4 messages
        result = ctx.build_recent_window(msgs)
        assert "Question 1" in result
        assert "Question 2" in result

    def test_respects_char_budget(self):
        ctx = ConversationContext(recent_count=100, max_recent_chars=50)
        msgs = _conversation(20)
        result = ctx.build_recent_window(msgs)
        assert len(result) <= 200  # budget + some slack


# ---------------------------------------------------------------------------
# build_context_block
# ---------------------------------------------------------------------------


class TestBuildContextBlock:
    def test_empty_messages(self):
        ctx = ConversationContext()
        assert ctx.build_context_block([]) == ""

    def test_recent_only_when_no_summary(self):
        ctx = ConversationContext(recent_count=4)
        msgs = _conversation(3)
        result = ctx.build_context_block(msgs)
        assert "[Recent Conversation]" in result
        assert "[Conversation Summary" not in result

    def test_both_summary_and_recent(self):
        ctx = ConversationContext(recent_count=4, summary_interval=5)
        msgs = _conversation(10)
        # Build summary first
        for i in range(10):
            ctx.on_turn(f"q{i}", f"a{i}")
        ctx.build_summary(msgs)
        result = ctx.build_context_block(msgs)
        assert "[Conversation Summary" in result
        assert "[Recent Conversation]" in result

    def test_summary_only_when_no_recent(self):
        ctx = ConversationContext(recent_count=4)
        msgs = _conversation(5)
        ctx.on_turn("a", "b")
        ctx.build_summary(msgs)
        # Empty messages = no recent
        result = ctx.build_context_block([])
        assert result == ""


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


class TestExtractFirstSentence:
    def test_normal_sentence(self):
        result = _extract_first_sentence("Hello world. This is more text.")
        assert result == "Hello world."

    def test_exclamation(self):
        result = _extract_first_sentence("Great idea! Let me work on that.")
        assert result == "Great idea!"

    def test_question(self):
        result = _extract_first_sentence("What do you think? I have ideas.")
        assert result == "What do you think?"

    def test_newline_fallback(self):
        result = _extract_first_sentence("First line\nSecond line")
        assert result == "First line"

    def test_long_text_truncated(self):
        result = _extract_first_sentence("x" * 200)
        assert len(result) <= 100
        assert result.endswith("...")

    def test_empty(self):
        assert _extract_first_sentence("") == ""

    def test_no_sentence_boundary(self):
        result = _extract_first_sentence("just some text without punctuation")
        assert result == "just some text without punctuation"


class TestTruncate:
    def test_short_text_unchanged(self):
        assert _truncate("hello", 10) == "hello"

    def test_long_text_truncated(self):
        result = _truncate("a" * 50, 20)
        assert len(result) == 20
        assert result.endswith("...")

    def test_strips_newlines(self):
        assert _truncate("line1\nline2", 50) == "line1 line2"


class TestGetRole:
    def test_dict(self):
        assert _get_role({"role": "user"}) == "user"

    def test_object(self):
        class Msg:
            role = "assistant"
        assert _get_role(Msg()) == "assistant"

    def test_missing(self):
        assert _get_role({}) == ""


class TestGetContent:
    def test_dict(self):
        assert _get_content({"content": "hello"}) == "hello"

    def test_object(self):
        class Msg:
            content = "world"
        assert _get_content(Msg()) == "world"

    def test_missing(self):
        assert _get_content({}) == ""
