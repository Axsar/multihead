"""Extracted from test_shell.py — fails on Windows (prompt_toolkit needs real console).

Fix: allow _build_prompt_session to accept output= parameter,
then test with DummyOutput() to avoid Win32Output console requirement.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from multihead.shell import Shell


@pytest.fixture
def shell():
    ac = MagicMock()
    ac.chat = AsyncMock(return_value="Hello!")
    ac.start = AsyncMock()
    ac.stop = AsyncMock()
    ac._detect_peers.return_value = []
    tools = MagicMock()
    tools.list_tools.return_value = []
    ac.tools = tools

    hm = MagicMock()
    hm.active_head = "core-llm"
    hm.get_states.return_value = {}
    hm.shutdown = AsyncMock()

    ks = MagicMock()
    ks.list_claims.return_value = []

    sm = MagicMock()
    sm.create_session.return_value = MagicMock(session_id="ses_test")
    sm.get_session.return_value = MagicMock(session_id="ses_test", messages=[])

    slash = MagicMock()
    slash.is_slash_command.side_effect = lambda t: t.startswith("/")
    slash.handle = AsyncMock(return_value="ok")

    return Shell(
        agentic_core=ac,
        head_manager=hm,
        knowledge_store=ks,
        session_manager=sm,
        slash_handler=slash,
        show_banner=False,
    )


def test_build_prompt_session_has_multiline(shell):
    """PromptSession should have multiline as a Condition."""
    from prompt_toolkit.output import DummyOutput
    session = shell._build_prompt_session(output=DummyOutput())
    assert session.multiline is not None
