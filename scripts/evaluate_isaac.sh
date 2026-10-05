#!/usr/bin/env bash
# Run the Isaac Lab evaluator with an Isaac Lab Python (Isaac Sim 5.1 / Isaac Lab).
# Override the interpreter with ISAAC_PYTHON=/path/to/python.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
ISAAC_PYTHON=${ISAAC_PYTHON:-python}
export ACCEPT_EULA=${ACCEPT_EULA:-Y} OMNI_KIT_ACCEPT_EULA=${OMNI_KIT_ACCEPT_EULA:-YES}
cd "$ROOT"
exec "$ISAAC_PYTHON" scripts/evaluate_isaac.py --headless "$@"
