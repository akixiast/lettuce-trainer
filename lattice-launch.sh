#!/usr/bin/env bash
# Lattice-Grow launcher: open browser if dashboard is up, else start it.
if curl -s -o /dev/null --max-time 3 http://localhost:7860/ 2>/dev/null; then
  xdg-open http://localhost:7860/
else
  cd "/home/kensagaryadav/Desktop/coding projects/AI Lab/lattice-grow" || exit 1
  source .venv/bin/activate
  exec python run_dashboard.py
fi
