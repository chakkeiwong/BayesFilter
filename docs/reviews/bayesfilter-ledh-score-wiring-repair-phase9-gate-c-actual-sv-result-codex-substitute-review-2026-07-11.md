# Codex Substitute Review: Phase 9 Gate C Actual-SV Result

Date: 2026-07-11

Review scope: exactly
`docs/plans/bayesfilter-ledh-score-wiring-repair-phase9-gate-c-actual-sv-result-2026-07-11.md`.
Result SHA-256 reviewed:
`f95ffee2fd47562ff27548e38b3e8c7154dbfb300f0b701146f69ab3fc05c5de`.

## Review Question

Does the result accurately classify the actual-SV Gate C evidence, apply the
frozen prefix score-memory and same-scalar FD gates, preserve the transformed
target boundary and required scientific nonclaims, and enforce the correct
next-step boundary?

## Findings

No material finding.

The result matches the two cited live shards and their SHA-256 hashes. It
correctly records terminal trusted GPU/XLA/TF32 score execution at
`T=4,N=10000`, a reset score peak of `35.22705078125 MiB`, and the frozen FD
failure: `max_abs=0.00948423147 > 0.005` and
`max_rel=0.0602924675 > 0.005`, driven by `log_beta`.

The result does not promote the short-prefix memory pass into full-time
`T=1000` evidence, does not change the FD step or thresholds after observing
the failure, and does not claim that the compact recurrence rather than
float32 FD resolution caused the disagreement. It correctly states that the
computed target is `transformed_actual_sv_log_y_square`, not an exact native
actual-SV likelihood, and separates row-candidate rejection from
shared-harness validity and from the remaining SV research direction.

The gate boundary is correct: actual-SV `T=50,250,1000`, Gate D, and
aggregation are blocked. Generalized-SV and KSC-SV may continue their
separately authorized Gate C ladders. This review does not authorize any
actual-SV repair or alternative FD arm.

VERDICT: AGREE
