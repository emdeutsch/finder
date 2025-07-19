#!/usr/bin/env bash
# Simple helper to launch the Finder UI in a clean virtual environment.
# Usage:  ./run_finder_ui.sh
# (Re-)creates a Python virtualenv named .venv in the current directory,
# installs the required packages, and starts the Streamlit web interface.

set -e  # Exit immediately on any error

# -----------------------------------------------------------------------------
# Virtual environment setup
# -----------------------------------------------------------------------------

# Ensure Poppler (pdf utilities needed by pdf2image) is installed.
if command -v brew >/dev/null 2>&1; then
  if ! command -v pdfinfo >/dev/null 2>&1; then
    echo "[Setup] Installing Poppler via Homebrew (required for PDF image extraction)..."
    brew install poppler
  fi
fi

python3 -m venv .venv
source .venv/bin/activate

# -----------------------------------------------------------------------------
# Install dependencies
# -----------------------------------------------------------------------------

pip install --upgrade pip
pip install -r finder_requirements.txt

# -----------------------------------------------------------------------------
# Launch the UI (Streamlit)
# -----------------------------------------------------------------------------

# Default port 7861 unless STREAMLIT_PORT is set
PORT="${STREAMLIT_PORT:-7861}"

# Note: Streamlit manages the host/port flags; we pass --server.headless=true to
# ensure it starts without opening a browser automatically.

streamlit run finder_ui.py --server.port "$PORT" --server.headless=true
