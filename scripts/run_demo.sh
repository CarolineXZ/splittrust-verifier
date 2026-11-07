#!/usr/bin/env bash
# Usage: scripts/run_demo.sh [offline|live] [scene]
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .env ] && set -a && source .env && set +a
python -m splittrust.demo --mode "${1:-offline}" --scene "${2:-all}" --pause
