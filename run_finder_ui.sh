#!/usr/bin/env bash
# Simple helper to launch the Finder UI in a clean virtual environment.
# Usage:  ./run_finder_ui.sh
# (Re-)creates a Python virtualenv named .venv in the current directory,
# installs the required packages, and starts the Gradio web interface.

set -e  # Exit immediately on any error

# -----------------------------------------------------------------------------
# Virtual environment setup
# -----------------------------------------------------------------------------

python3 -m venv .venv
source .venv/bin/activate

# -----------------------------------------------------------------------------
# Install dependencies
# -----------------------------------------------------------------------------

pip install --upgrade pip
pip install -r finder_requirements.txt

# -----------------------------------------------------------------------------
# Launch the UI
# -----------------------------------------------------------------------------

python finder_ui.py
