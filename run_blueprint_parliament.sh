#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ -n "${BLUEPRINT_PYTHON:-}" ]]; then
  PYTHON_BIN="$BLUEPRINT_PYTHON"
elif [[ -x "$PROJECT_ROOT/.venv/bin/python" ]]; then
  PYTHON_BIN="$PROJECT_ROOT/.venv/bin/python"
else
  PYTHON_BIN="python3"
fi
exec "$PYTHON_BIN" \
  "$PROJECT_ROOT/run_blueprint_parliament.py" "$@"
