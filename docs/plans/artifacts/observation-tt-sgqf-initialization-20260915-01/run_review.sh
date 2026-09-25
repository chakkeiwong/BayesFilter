#!/usr/bin/env bash
set -euo pipefail
mode="${1:?health or review required}"
attempt="${2:?unique suffix required}"
review_dir="/home/chakwong/BayesFilter/docs/plans/artifacts/observation-tt-sgqf-initialization-20260915-01"
packet="/home/chakwong/BayesFilter/docs/plans/observation-aware-tt-master-amendment-04-sgqf-initialization-20260915.md"
if [[ -n "${ANTHROPIC_AUTH_TOKEN:-}" && -n "${ANTHROPIC_API_KEY:-}" ]]; then unset ANTHROPIC_API_KEY; fi
case "$mode" in
  health) limit=45; prompt='Return exactly CLAUDE_PROBE_OK.' ;;
  read) limit=45; prompt="Read only ${packet}. Return exactly CLAUDE_PACKET_READ_OK if readable." ;;
  review) limit=180; prompt="READ-ONLY BOUNDED REVIEW. Review exactly this path and nothing else unless the file itself explicitly asks you to inspect a cited line: ${packet}. Do not edit, run commands, launch agents, or review the whole repo. Question: Is this mathematically sound and sufficiently controlled to execute the authorized SGQF-initialization diagnostic, including the Hermite recurrence, empirical safeguard, comparison fairness, inherited assumptions and stops? Identify material blockers only, distinguish advisory improvements, and keep the response below 900 words. End with VERDICT: AGREE or VERDICT: REVISE." ;;
  result) limit=180; prompt="READ-ONLY BOUNDED REVIEW. Review exactly this path and nothing else unless the file itself explicitly asks you to inspect a cited line: /home/chakwong/BayesFilter/docs/plans/observation-aware-tt-sgqf-initialization-20260915-result.md. Do not edit, run commands, launch agents, or review the whole repo. Question: Do the recorded comparisons support the initialization and empirical-safeguard conclusions, with correct distinction of density targets, numerical-design versus observation replication, inherited assumptions, promotion versus continuation vetoes, and limits of what was implemented? Review interpretation rather than certify uninspected code. Identify material scientific blockers only, keep the response below 700 words, and end with VERDICT: AGREE or VERDICT: REVISE." ;;
  terminal) limit=120; prompt="READ-ONLY BOUNDED REVIEW. Review exactly this path and nothing else: ${review_dir}/terminal-review-summary.md. Do not edit, run commands, launch agents, or review the whole repo. Answer only the question in that file, in at most 250 words. End with VERDICT: AGREE or VERDICT: REVISE." ;;
  *) exit 2 ;;
esac
cd /home/chakwong/BayesFilter
timeout "${limit}s" claude --print --bare --no-session-persistence \
  --output-format text --permission-mode plan --setting-sources '' \
  --tools Read --disallowed-tools Bash,Edit,Write,NotebookEdit,WebFetch,WebSearch \
  -- "$prompt" > "${review_dir}/${mode}-${attempt}.txt" 2> "${review_dir}/${mode}-${attempt}.stderr"
