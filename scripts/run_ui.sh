#!/usr/bin/env bash
# Usage: scripts/run_ui.sh [offline|live] [port]
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .env ] && set -a && source .env && set +a
python -m splittrust.ui --mode "${1:-offline}" --port "${2:-8765}"
