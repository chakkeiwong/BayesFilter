# Complete Leaderboard Claude Review Availability

Date: 2026-07-11

Status: `CLAUDE_PRIMARY_UNAVAILABLE_TWO_TRUSTED_PROBES`

Codex remained supervisor/executor. Claude was requested only as a read-only
reviewer.

## Probe 1

- trusted command shape: direct `claude -p` exact-token health probe;
- model: `opus`;
- effort: `low`;
- timeout: 90 seconds;
- expected token: `CLAUDE_PROBE_OK`;
- result: exit 124, no output.

## Probe 2

- trusted command shape:
  `/home/chakwong/python/claudecodex/scripts/claude_worker.sh`;
- deterministic worker settings and credential-conflict handling enabled;
- model: `opus`;
- effort: `low`;
- timeout: 120 seconds;
- expected token: `CLAUDE_PROBE_OK`;
- result: exit 124, no output.

## Classification

Claude is unavailable for the current gate after two trusted health probes.
Fresh Codex agents may perform one-path read-only substitute reviews under the
user-approved fallback. Substitute agreement is weaker evidence, cannot
authorize any boundary crossing, and cannot alone approve Zhao-Cui
source-faithfulness or final release.

