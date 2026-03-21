#!/usr/bin/env bash
# MultiHead installer: detect environment, install deps, generate config.
# Usage: bash scripts/install.sh
set -euo pipefail

# --- Colors ---
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BOLD='\033[1m'; NC='\033[0m'
info()  { echo -e "${GREEN}[+]${NC} $1"; }
warn()  { echo -e "${YELLOW}[!]${NC} $1"; }
error() { echo -e "${RED}[x]${NC} $1"; }

echo -e "${BOLD}MultiHead Installer${NC}"
echo "=============================="
echo ""

# --- Detect OS ---
OS="$(uname -s)"
case "$OS" in
    Linux*)
        if grep -qi microsoft /proc/version 2>/dev/null; then
            PLATFORM="wsl"
        else
            PLATFORM="linux"
        fi
        ;;
    Darwin*) PLATFORM="macos" ;;
    *)       error "Unsupported OS: $OS"; exit 1 ;;
esac
info "Platform: $PLATFORM"

# --- Check Python ---
PYTHON=""
for candidate in python3.12 python3.11 python3; do
    if command -v "$candidate" &>/dev/null; then
        ver=$("$candidate" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
        major=$(echo "$ver" | cut -d. -f1)
        minor=$(echo "$ver" | cut -d. -f2)
        if [ "$major" -eq 3 ] && [ "$minor" -ge 11 ]; then
            PYTHON="$candidate"
            break
        fi
    fi
done

if [ -z "$PYTHON" ]; then
    error "Python 3.11+ required but not found."
    echo "  Install from https://www.python.org/downloads/"
    exit 1
fi
info "Python: $PYTHON ($($PYTHON --version 2>&1))"

# --- Create/reuse venv ---
VENV_DIR=".venv"
if [ -d "$VENV_DIR" ]; then
    info "Virtual environment exists at $VENV_DIR"
else
    info "Creating virtual environment..."
    $PYTHON -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

# --- Detect CUDA/GPU ---
HAS_CUDA=false
if command -v nvidia-smi &>/dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits 2>/dev/null || true)
    if [ -n "$GPU_INFO" ]; then
        HAS_CUDA=true
        info "GPU: $GPU_INFO"
    fi
elif [ "$PLATFORM" = "macos" ]; then
    # macOS: no CUDA, suggest Ollama
    info "GPU: Apple Silicon (use Ollama for local models)"
else
    warn "No NVIDIA GPU detected (nvidia-smi not found)"
fi

# --- Install package ---
info "Installing multihead..."
pip install --upgrade pip -q 2>/dev/null

EXTRAS="dev"
if [ "$HAS_CUDA" = true ]; then
    EXTRAS="dev,gpu"
    info "CUDA detected — installing GPU dependencies (torch, transformers, etc.)"
fi

pip install -e ".[$EXTRAS]" -q 2>/dev/null
info "Installed: $(pip show multihead 2>/dev/null | grep Version || echo 'multihead')"

# --- Run auto-init ---
info "Running hardware detection and config generation..."
multihead init --auto

# --- Copy .env.example if no .env ---
if [ ! -f .env ] && [ -f .env.example ]; then
    cp .env.example .env
    info "Created .env from .env.example"
fi

# --- Run diagnostics ---
echo ""
info "Running diagnostics..."
multihead doctor || true

# --- Done ---
echo ""
echo -e "${BOLD}=============================================${NC}"
echo -e "${BOLD}  MultiHead installed successfully!${NC}"
echo -e "${BOLD}=============================================${NC}"
echo ""
echo "  Activate the venv:    source .venv/bin/activate"
echo "  Start the daemon:     multihead serve"
echo "  Interactive chat:     multihead chat"
echo "  Run diagnostics:      multihead doctor"
echo ""

if [ "$PLATFORM" = "macos" ]; then
    warn "For local models, install Ollama: https://ollama.ai"
    echo "  Then: ollama pull qwen3:8b"
fi
