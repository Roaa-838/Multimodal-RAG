#!/usr/bin/env bash
# run_app.sh
# Launch the Multi-Modal RAG Streamlit application.
#
# Usage:
#   bash run_app.sh            # default port 8501
#   bash run_app.sh --port 8080

set -e

# ── Resolve project root (directory containing this script) ───
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Load .env if present ───────────────────────────────────────
if [ -f ".env" ]; then
    echo "[run_app] Loading environment from .env"
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

# ── Check streamlit is installed ───────────────────────────────
if ! command -v streamlit &> /dev/null; then
    echo "[error] streamlit not found. Run:  pip install streamlit"
    exit 1
fi

# ── Pass any extra args through (e.g. --port 8080) ────────────
echo "[run_app] Starting Streamlit app..."
streamlit run app.py \
    --server.headless true \
    --server.fileWatcherType none \
    "$@"