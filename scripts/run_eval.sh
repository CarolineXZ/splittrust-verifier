#!/usr/bin/env bash
# Usage: scripts/run_eval.sh [offline|live] [extra args...]
set -euo pipefail
cd "$(dirname "$0")/.."
[ -f .env ] && set -a && source .env && set +a
mode="${1:-offline}"; shift || true
python -m splittrust.eval --mode "$mode" "$@"
