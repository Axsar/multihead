#!/bin/bash
# Start an auto-responder poller for multi-session collaboration
#
# Usage:
#   bash start_auto_responder.sh [SESSION_ID] [PROJECT_ID] [INTERVAL] [CAPABILITIES] [STRATEGY] [MAX_BUDGET] [MODEL]
#
# Examples:
#   # Plan-only (default — posts plans to knowledge.db):
#   bash start_auto_responder.sh "my-agent" "multihead"
#
#   # Autonomous execution (spawns claude -p per step):
#   bash start_auto_responder.sh "my-agent" "multihead" 30 "solve,decompose" "execute" 3.0 "claude-sonnet-4-6"

# Default values
SESSION_ID="${1:-claude-auto-$(date +%s)}"
PROJECT_ID="${2:-multihead}"
CHECK_INTERVAL="${3:-30}"
CAPABILITIES="${4:-solve,decompose}"
STRATEGY="${5:-plan-only}"
MAX_BUDGET="${6:-1.0}"
CLAUDE_MODEL="${7:-claude-sonnet-4-6}"

# Activate environment (override MULTIHEAD_VENV to use a different virtualenv)
VENV="${MULTIHEAD_VENV:-}"
if [ -n "$VENV" ] && [ -f "$VENV/bin/activate" ]; then
    source "$VENV/bin/activate"
fi

# Start poller
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
python "$SCRIPT_DIR/auto_responder_poller.py" \
    --session-id "$SESSION_ID" \
    --project-id "$PROJECT_ID" \
    --check-interval "$CHECK_INTERVAL" \
    --capabilities "$CAPABILITIES" \
    --strategy "$STRATEGY" \
    --max-budget "$MAX_BUDGET" \
    --claude-model "$CLAUDE_MODEL"
