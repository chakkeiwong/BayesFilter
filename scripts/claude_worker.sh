#!/usr/bin/env bash
set -euo pipefail

exec "${HOME}/python/claudecodex/scripts/claude_worker.sh" "$@"
