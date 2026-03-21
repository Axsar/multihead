"""LLM prompt templates for task decomposition and refinement."""

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

DECOMPOSE_PROMPT = """You are a task decomposer for software engineering.
Given a goal and codebase context, break the goal into a hierarchical execution plan where each leaf step is a single concrete action.

## Goal
{goal}

## Additional Context
{user_context}

## Codebase Knowledge
{knowledge_context}

## Rules
1. Group steps into phases (e.g. understand, diagnose, implement, verify)
2. Leaf steps must be SINGLE concrete actions: read a file, edit specific code, run a test
3. Scale depth to complexity:
   - Simple bug fix: 1-2 phases, 3-6 leaf steps
   - Moderate feature: 2-4 phases, 8-15 leaf steps
   - Complex refactor: 3-6 phases, 15-40 leaf steps
4. Action types: explore, read, edit, create, test, verify, refactor, delete
5. Include target file paths when known from context
6. Each step should describe what success looks like

## Output
Return ONLY valid JSON (no markdown fences, no commentary):
{{
  "complexity": "simple|moderate|complex",
  "phases": [
    {{
      "id": "1",
      "goal": "Phase description",
      "rationale": "Why this phase is needed",
      "children": [
        {{
          "id": "1.1",
          "goal": "Concrete step description",
          "action_type": "read|edit|test|...",
          "target_files": ["path/to/file.py"],
          "expected_output": "What you'll have after this step",
          "rationale": "",
          "children": []
        }}
      ]
    }}
  ]
}}"""

REFINE_PROMPT = """Break this step into smaller sub-steps.

## Step
ID: {node_id}
Goal: {goal}
Action type: {action_type}
Files: {files}

## Exploration Result
{exploration_result}

## Rules
- Each sub-step should be a single concrete action
- Keep IDs hierarchical (e.g. {node_id}.1, {node_id}.2)
- Include target files and expected outputs

## Output
Return ONLY valid JSON — a list of sub-steps:
[
  {{
    "id": "{node_id}.1",
    "goal": "...",
    "action_type": "...",
    "target_files": [],
    "expected_output": "...",
    "rationale": "",
    "children": []
  }}
]"""
