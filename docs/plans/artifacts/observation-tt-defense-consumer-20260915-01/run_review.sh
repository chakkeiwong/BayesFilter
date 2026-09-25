#!/usr/bin/env bash
set -euo pipefail
attempt="${1:?unique review suffix required}"
packet="${2:-/home/chakwong/BayesFilter/docs/plans/observation-aware-tt-master-amendment-05-defense-consumer-20260915.md}"
review_dir="/home/chakwong/BayesFilter/docs/plans/artifacts/observation-tt-defense-consumer-20260915-01"
if [[ -n "${ANTHROPIC_AUTH_TOKEN:-}" && -n "${ANTHROPIC_API_KEY:-}" ]]; then unset ANTHROPIC_API_KEY; fi
prompt="READ-ONLY BOUNDED REVIEW. Review exactly this path and nothing else unless the file itself explicitly asks you to inspect a cited line: ${packet}. Do not edit, run commands, launch agents, or review the whole repo. Question: Are the mathematical contract, safety calibration and proposed interpretation sound, including wrong baselines, arbitrary defaults, proxy promotion, numerical vetoes, and budget? Identify material blockers separately from advisory improvements. This is a bounded phase-8 diagnostic, not a filter promotion. Keep your response below 600 words. End with VERDICT: AGREE or VERDICT: REVISE."
cd /home/chakwong/BayesFilter
test ! -e "${review_dir}/review-${attempt}.txt"
timeout 180s claude --print --bare --no-session-persistence --output-format text \
  --permission-mode plan --setting-sources '' --tools Read \
  --disallowed-tools Bash,Edit,Write,NotebookEdit,WebFetch,WebSearch \
  -- "$prompt" > "${review_dir}/review-${attempt}.txt" 2> "${review_dir}/review-${attempt}.stderr"
