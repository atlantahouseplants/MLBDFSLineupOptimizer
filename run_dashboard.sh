#!/usr/bin/env bash
# Launch the MLB DFS Lineup Optimizer Dashboard
# Usage: ./run_dashboard.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# Check Python
if ! command -v python3 &>/dev/null; then
    echo "Error: python3 not found. Install Python 3.10+."
    exit 1
fi

# Install dependencies if needed
python3 -c "import streamlit" 2>/dev/null || {
    echo "Installing dependencies..."
    pip install -r requirements.txt
}

# Set Python path so Streamlit can find our modules
export PYTHONPATH="${SCRIPT_DIR}/src:${PYTHONPATH:-}"

echo ""
echo "=========================================="
echo "  MLB DFS Lineup Optimizer Dashboard"
echo "=========================================="
echo ""
echo "Opening in your browser..."
echo ""

streamlit run dashboard/daily_workflow.py \
    --server.headless true \
    --browser.gatherUsageStats false \
    "$@"
