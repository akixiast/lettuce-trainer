#!/usr/bin/env python3
"""Entry point to launch the Lattice-Grow Gradio dashboard."""

import sys
from pathlib import Path

# Ensure src is on the path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from lattice_grow.dashboard import launch_dashboard

if __name__ == "__main__":
    launch_dashboard()
