"""Tests for action type parsing."""

import json

from multihead.action_types import (
    CallToolAction,
    CreateWorkOrderAction,
    MonitorWorkOrderAction,
    PauseAndAskAction,
    SayAction,
    parse_action,
)


class TestParseAction:
    def test_parse_say(self):
        raw = json.dumps({"action": "SAY", "content": "Hello!"})
        action = parse_action(raw)
        assert isinstance(action, SayAction)
        assert action.content == "Hello!"

    def test_parse_call_tool(self):
        raw = json.dumps({
            "action": "CALL_TOOL", "tool": "files.read",
            "params": {"path": "/tmp/x"},
        })
        action = parse_action(raw)
        assert isinstance(action, CallToolAction)
        assert action.tool == "files.read"
        assert action.params["path"] == "/tmp/x"

    def test_parse_create_workorder(self):
        raw = json.dumps({"action": "CREATE_WORKORDER", "goal": "Summarize docs", "steps": []})
        action = parse_action(raw)
        assert isinstance(action, CreateWorkOrderAction)
        assert action.goal == "Summarize docs"

    def test_parse_monitor_workorder(self):
        raw = json.dumps({"action": "MONITOR_WORKORDER", "run_id": "run_123", "decision": "check"})
        action = parse_action(raw)
        assert isinstance(action, MonitorWorkOrderAction)
        assert action.run_id == "run_123"

    def test_parse_pause_and_ask(self):
        raw = json.dumps({
            "action": "PAUSE_AND_ASK",
            "question": "Which format?",
            "options": ["A", "B"],
        })
        action = parse_action(raw)
        assert isinstance(action, PauseAndAskAction)
        assert action.question == "Which format?"
        assert len(action.options) == 2

    def test_fallback_to_say(self):
        raw = "Just some random text without JSON"
        action = parse_action(raw)
        assert isinstance(action, SayAction)
        assert "random text" in action.content

    def test_parse_markdown_code_block(self):
        raw = 'Here is my action:\n```json\n{"action": "SAY", "content": "Done"}\n```'
        action = parse_action(raw)
        assert isinstance(action, SayAction)
        assert action.content == "Done"

    def test_parse_embedded_json(self):
        raw = 'I will do this: {"action": "SAY", "content": "ok"} and then stop.'
        action = parse_action(raw)
        assert isinstance(action, SayAction)
        assert action.content == "ok"

    def test_parse_case_insensitive(self):
        raw = json.dumps({"action": "say", "content": "hi"})
        action = parse_action(raw)
        assert isinstance(action, SayAction)

    def test_parse_invalid_action_type(self):
        raw = json.dumps({"action": "EXPLODE", "data": "boom"})
        action = parse_action(raw)
        assert isinstance(action, SayAction)  # Fallback

    def test_parse_missing_required_fields(self):
        raw = json.dumps({"action": "CALL_TOOL"})  # Missing 'tool'
        action = parse_action(raw)
        # Should either parse with default or fall back
        assert isinstance(action, (CallToolAction, SayAction))
