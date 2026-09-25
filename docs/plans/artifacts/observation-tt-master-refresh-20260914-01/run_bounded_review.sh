#!/usr/bin/env bash
set -euo pipefail

# Read-only fallback after the ordinary worker timed out. This does not run
# experiments; Claude has only the Read tool and no workspace instruction load.
# Based on the installed claude-readonly-review-probe skill's bare probe ladder.
mode="${1:?health, read, packet, or review required}"
attempt="${2:?unique attempt suffix required}"
review_dir="/home/chakwong/BayesFilter/docs/plans/artifacts/observation-tt-master-refresh-20260914-01"
packet="/home/chakwong/BayesFilter/docs/plans/observation-aware-tt-master-amendment-03-20260914.md"
if [[ -n "${ANTHROPIC_AUTH_TOKEN:-}" && -n "${ANTHROPIC_API_KEY:-}" ]]; then
  unset ANTHROPIC_API_KEY
fi
case "$mode" in
  health)
    limit=45
    prompt='Return exactly CLAUDE_PROBE_OK.'
    ;;
  read)
    limit=45
    prompt="Read only ${packet}. Reply with exactly CLAUDE_PACKET_READ_OK if readable."
    ;;
  packet)
    limit=90
    prompt="READ-ONLY BOUNDED REVIEW. Read only ${packet}. Is this file self-contained enough to review its program reconciliation and authorization boundary, with exact supporting sections named for factual verification? Do not edit, run commands or launch agents. Return PASS_PACKET_SELF_CONTAINED or BLOCK_PACKET_SELF_CONTAINED and at most three short reasons."
    ;;
  review)
    limit=180
    prompt="READ-ONLY BOUNDED REVIEW. Review exactly this path and nothing else unless the file itself explicitly asks you to inspect a cited line: ${packet}. Do not edit, run commands, launch agents, or review the whole repo. Question: Does A03 accurately reconcile the existing master and terminal evidence, preserve scientific and authorization boundaries, and define an unambiguous design-and-review step instead of ad hoc execution? Read the exact supporting sections requested inside the file as needed. This is a program reconciliation, not an executable numerical protocol. State any material error with anchors; keep the review under 500 words. End with VERDICT: AGREE or VERDICT: REVISE."
    ;;
  *) exit 2 ;;
esac
cd /home/chakwong/BayesFilter
timeout "${limit}s" claude --print --bare --no-session-persistence \
  --output-format text --permission-mode plan --setting-sources '' \
  --tools Read --disallowed-tools Bash,Edit,Write,NotebookEdit,WebFetch,WebSearch \
  -- "$prompt" > "${review_dir}/bounded-${mode}-${attempt}.txt" 2> "${review_dir}/bounded-${mode}-${attempt}.stderr"
