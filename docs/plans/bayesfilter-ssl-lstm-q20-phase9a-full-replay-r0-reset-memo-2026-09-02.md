# Phase 9A R0 Reset Memo

Date: 2026-09-02  
Active subplan: `docs/plans/bayesfilter-ssl-lstm-q20-phase9a-full-replay-performance-subplan-2026-09-01.md`  
Master: `docs/plans/bayesfilter-ssl-lstm-q20-tempered-rkl-transport-ensemble-master-program-2026-09-02.md`

R0 is closed with no continuation blocker. The historical attempt-03/04/05
manifests remain immutable; attempt-05's embedded canonical manifest hash was
recomputed and matched. Their timing records support a descriptive cost model
but predate compile/steady-run fields.

The smallest repair was additive instrumentation in the existing reusable
runner and two disjoint source-owned profiles:

- `phase9a_full_replay_canary_v1`, scope `(3,1)`, cap `1800` seconds, seeds
  `20260902/780xx--785xx`;
- `phase9a_full_replay_v1`, scope `(0,6)`, cap `7800` seconds, seeds
  `20260902/790xx--795xx`.

Focused profile tests, Python compilation, shell syntax, and whitespace checks
passed. The refreshed next subplan is R2 canary. Its output is calibration
only; a mandatory canary closeout must refresh R3 before the full replay.

No posterior, whitening, mode-discovery, convergence, or default-readiness
claim is opened by this reset.
