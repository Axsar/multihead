#!/bin/bash
# Start the Claude ACP Worker Daemon
#
# Prerequisites:
#   - ACP_SESSION_KEY must be set (JWT for claude-session-agent)
#   - BotVibes ACP must be running at localhost:8000
#   - Claude Code CLI must be in PATH
#
# Usage:
#   ./scripts/start_claude_worker.sh                    # headless mode (default)
#   ./scripts/start_claude_worker.sh --interactive-only  # tmux keystroke mode

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Activate Python environment (prefer .venv in project, fall back to system)
if [ -f "$PROJECT_DIR/.venv/bin/activate" ]; then
    source "$PROJECT_DIR/.venv/bin/activate"
elif [ -n "${VIRTUAL_ENV:-}" ]; then
    : # already activated
else
    echo "[launcher] Warning: no .venv found; using system Python"
fi

# Configuration (override via environment)
export ACP_URL="${ACP_URL:-http://localhost:8000/api/v1}"
export CLAUDE_WORK_DIR="${CLAUDE_WORK_DIR:-$PROJECT_DIR}"
export CLAUDE_MAX_BUDGET="${CLAUDE_MAX_BUDGET:-2.0}"
export CLAUDE_TIMEOUT="${CLAUDE_TIMEOUT:-300}"

# Check JWT (prefer ACP_CLAUDE_SESSION_KEY, fall back to ACP_SESSION_KEY)
if [ -z "${ACP_CLAUDE_SESSION_KEY:-}" ] && [ -z "${ACP_SESSION_KEY:-}" ]; then
    # Try reading from .env file
    ENV_FILE="$PROJECT_DIR/.env"
    if [ -f "$ENV_FILE" ]; then
        SESSION_KEY="$(grep '^ACP_CLAUDE_SESSION_KEY=' "$ENV_FILE" | cut -d= -f2 | tr -d '\r')"
        if [ -n "$SESSION_KEY" ]; then
            export ACP_CLAUDE_SESSION_KEY="$SESSION_KEY"
        fi
    fi
    # Final fallback: file-based JWT
    JWT_FILE="${MULTIHEAD_DATA_DIR:-$HOME/.multihead}/.claude_worker_jwt"
    if [ -z "${ACP_CLAUDE_SESSION_KEY:-}" ] && [ -f "$JWT_FILE" ]; then
        export ACP_SESSION_KEY="$(cat "$JWT_FILE" | tr -d '\r')"
    fi
    if [ -z "${ACP_CLAUDE_SESSION_KEY:-}" ] && [ -z "${ACP_SESSION_KEY:-}" ]; then
        echo "Error: Set ACP_CLAUDE_SESSION_KEY env var, add it to .env, or create $JWT_FILE"
        exit 1
    fi
fi

echo "[launcher] Starting Claude Worker Daemon"
echo "[launcher] Project: $CLAUDE_WORK_DIR"
echo "[launcher] ACP: $ACP_URL"

exec python "$SCRIPT_DIR/claude_worker.py" "$@"
