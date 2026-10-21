# Exact initial-force consumer localization, 2026-10-13

Question: why did the range-bearing HMC initial force fail when nearby standalone
filter evaluations were healthy? This is the planned localization step in the
ten-hour repair campaign, not a new posterior or precision-policy direction.
Budget: at most450 GPU seconds and one attempt, cumulative ledger36000s/60attempts.
Output: artifacts/surrogate-hmc-exact-force-range-20261013-01. One GPU worker.

Replay the failed selection's controls and exact FP64 beta, data/design and four
FP64 starts (-.15,-.05,.05,.15). The saved selection is diagnostic input only:
verify its hash and all non-source scope fields, record source differences, and
do not grant current-source tuning authority. Use the actual runtime.bindings
force endpoint, then a standalone B2 graph with the same chart, recurrence and
inputs that exposes per-observation traces. Save each phase before proceeding.
No sampling, prior change, narrowed start region, threshold relaxation or fallback.

Engineering criterion: all four current consumer forces valid and finite, full
horizon in exposed traces, same validity and final force within the owner's1%
scaled tolerance across call paths. Any disagreement is a call-chain repair
trigger. Invalidity is an HMC promotion veto, not a campaign continuation veto.
Explanatory diagnostics: per-step margins, moment errors, completed steps, source
hashes, trace counts, allocator peak and compile/execution time. They identify
where to investigate; numerical predicates must be checked before naming a cause.
A pass does not reproduce or explain the old failure and does not prove posterior
correctness, score accuracy against the model oracle, or speed superiority.

Assumptions/defaults: starts, controls and beta come from the failed pilot's
documented inputs; N4032/K2016 and FP32 pair arithmetic stay fixed. FP64 recursion
is the previously reviewed optional repair scope. Stable signatures, XLA and
memory growth remain mandatory. The generic scope's source hashes changed with
the covariance and reporting repairs; differences are explicit, never ignored
for admission. Cheap scope-mismatch and failed-row preservation tests precede GPU.

Skeptical audit: nearby rounded inputs were an inadequate comparator for replay;
this diagnostic corrects that. Exposing traces can change XLA optimization, so
the real consumer endpoint is evaluated first and its result saved independently.
The earliest useful distinction is actual consumer invalidity versus standalone
invalidity. Identical healthy outputs only say that the current failure is not
reproduced. No ranking of methods is attempted; same-input implementation checks
do not replace Kalman/integration/bootstrap baselines for later scientific claims.

Implementation audit: actual consumer resolves through runtime.bindings and
FrozenPositionOnlyForce.function; no alternative force is substituted. Exposed
B2 diagnostics call the same shared all-P filter. CPU-only regressions:114
passed with one incomplete mock fixture; corrected diagnostic suite5passed.
Logs: force-health-regression.log and force-health-fixture-check.log.
The exact-input check is ready; wait for the sole SIR worker to finish.
