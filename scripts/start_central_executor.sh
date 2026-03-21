#!/bin/bash
# Start the Central Executor — one poller that executes plans for all agents
#
# Watches knowledge.db for:
#   1. DECOMP_REQUEST / TASK DECOMPOSITION REQUEST claims (standard decompose+execute)
#   2. EXECUTION REQUEST claims (any agent can post to trigger execution)
#
# When a request is found:
#   - Decomposes the task (or uses referenced plan)
#   - Posts plan to knowledge.db (so all agents see it)
#   - Spawns claude -p per step with role-specific tools
#   - Posts results back to knowledge.db
#
# Usage:
#   bash scripts/start_central_executor.sh
#   bash scripts/start_central_executor.sh 3.0 claude-sonnet-4-6
#
# Any agent can trigger execution by posting:
#   from multihead.session_poller import post_execution_request
#   post_execution_request(knowledge_store, goal="Fix the bug", session_id="my-agent")

set -euo pipefail

MAX_BUDGET="${1:-2.0}"
CLAUDE_MODEL="${2:-claude-sonnet-4-6}"
CHECK_INTERVAL="${3:-30}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Activate environment (override MULTIHEAD_VENV to use a different virtualenv)
VENV="${MULTIHEAD_VENV:-}"
if [ -n "$VENV" ] && [ -f "$VENV/bin/activate" ]; then
    source "$VENV/bin/activate"
fi

export CLAUDE_WORK_DIR="${CLAUDE_WORK_DIR:-$PROJECT_DIR}"

echo "=============================================="
echo "  Central Executor"
echo "  Model: $CLAUDE_MODEL"
echo "  Budget: \$$MAX_BUDGET/step"
echo "  Interval: ${CHECK_INTERVAL}s"
echo "  Work dir: $CLAUDE_WORK_DIR"
echo "=============================================="

python "$SCRIPT_DIR/auto_responder_poller.py" \
    --session-id "central-executor" \
    --project-id "multihead" \
    --check-interval "$CHECK_INTERVAL" \
    --capabilities "solve,decompose,execute" \
    --strategy execute \
    --max-budget "$MAX_BUDGET" \
    --claude-model "$CLAUDE_MODEL"
