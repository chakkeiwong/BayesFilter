#!/usr/bin/env bash
set -euo pipefail
attempt="${1:?unique review suffix required}"
packet="/home/chakwong/BayesFilter/docs/plans/observation-aware-tt-independent-filtering-20260915-result.md"
review_dir="/home/chakwong/BayesFilter/docs/plans/artifacts/observation-tt-independent-filtering-20260915-01"
if [[ -n "${ANTHROPIC_AUTH_TOKEN:-}" && -n "${ANTHROPIC_API_KEY:-}" ]]; then unset ANTHROPIC_API_KEY; fi
prompt="READ-ONLY BOUNDED REVIEW. Review exactly this path and nothing else unless the file itself explicitly asks you to inspect a cited line: ${packet}. Do not edit, run commands, launch agents, or review the whole repo. Question: Does the terminal interpretation follow the predeclared independent-sequence contract, including reference validity, conditional heuristic losses, simultaneous uncertainty, fitting-selection limitations and the next justified action? Identify material scientific blockers separately from advisory improvements. Do not demand extra procedural approval for unchanged authorized work. Keep your response below 600 words. End with VERDICT: AGREE or VERDICT: REVISE."
cd /home/chakwong/BayesFilter
test ! -e "${review_dir}/review-${attempt}.txt"
timeout 180s claude --print --bare --no-session-persistence --output-format text \
  --permission-mode plan --setting-sources '' --tools Read \
  --disallowed-tools Bash,Edit,Write,NotebookEdit,WebFetch,WebSearch \
  -- "$prompt" > "${review_dir}/review-${attempt}.txt" 2> "${review_dir}/review-${attempt}.stderr"
