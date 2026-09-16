# Observation-aware TT active checkpoint, 2026-09-16

## Active question and authorization

Does SGQF-initialized pair TT remain a useful exact-importance-corrected
Zhao-Cui proposal after guide covariance collapse is prevented? Owner explicitly
requests thorough LaTeX propositions/proofs, MathDevMCP audit, a reviewed plan
and execution. A10 is active; A09 is complete. Follow
[master](observation-aware-tt-repair-complete-program-20260913.md) and
[A10](observation-aware-tt-master-amendment-10-robust-guide-20260916.md).

Checkout /home/chakwong/BayesFilter, branch surrogate-hmc; initial commit
50f93709d8a4c0b9481d49b07ec5373b73b23473.
Preserve unrelated changes; no commit/push requested. No subagents authorized.

## Checked evidence and current stage

A09 d4-s04 t8: nine-node SGQF puts essentially all weight on one node;
covariance about 1e-31 passes a self-scaled SPD check. Higher rules are
indefinite. TT defense shares the collapsed chart. Saved reference has
order-one covariance. d4-s10 also has guide failure and reference imprecision.
Evidence: artifacts/observation-tt-warm-improvement-20260916-01/
covariance-collapse-diagnostic.md and observation-aware-tt-warm-improvement-20260916-result.md.

A10 plan skeptical self-review passed. Eight propositions/proofs drafted in
LaTeX Section 17; MathDevMCP audit/dispositions saved in math-review.md.
Build and rendered-page inspection passed. Optional TF implementation and
actual-consumer wiring are complete; 37 focused tests pass. GPU smoke-02
passed numerical mechanism checks, including exposed s04/s10. Repaired-guide
chart minimum on s04 is .27047 versus baseline 3.72e-32; stable-chart minimum
is .80365. These smoke runs have no accuracy reference. MathDevMCP found no
established counterexample but was incomplete; no formal proof certificate.
Protected LaTeX baseline and metadata live under
artifacts/observation-tt-robust-guide-20260916-01/.
Implementations are optional extensions, not source-faithfulness, default or
HMC claims. Finite diagnostic tests do not certify posterior accuracy or ESS.

## Budget and exact next action

Sole ledger: artifacts/observation-tt-continuation-24h-20260915-01/budget.json.
A10 begins conservatively 2026-09-16 12:48:00 UTC, ceiling 28800 active seconds,
from 41020.892609 seconds remaining. Retain previous reservations. Charge elapsed
work/tool waits once at closeout; numerical times are a subset. Smoke-01 failed
in one-repetition MCSE reporting (150 seconds conservatively reserved);
smoke-02 completed in 169.254 seconds. Plan caps numerical work at 18000 seconds.

Calibration-01 COMPLETE in 581.310050 seconds; all six references pass and no
arm failures. Frozen controls: guide L1 d1=1e-5/d4=.001; stable L1=.001 both;
epsilon=.05 both, smallest passing weight. All calibration-admissible.
Confirmation-01 COMPLETE in 1211.579978 seconds, all references valid; stable
and full fail on d4-s05 t19 because a valid negative least-squares amplitude
scale was incorrectly rejected. Signed-scale repair is derived, documented,
reviewed and tested (38 pass); positive branch unchanged. Exact original
sources/results preserved in source-snapshot and comparison.json. MathDevMCP
additional equation audit abstains on formalization, no counterexample.
Next: smoke-03 includes exposed d4-s05 data AND original fitting seed; then
calibration-02 on existing calibration blocks and fresh confirmation-02 with
--confirmation-block-start 48. This consumes remaining A10 attempt slots,
under unchanged ceilings. Run summarize.py attempt-confirmation-02, terminal
result/manuscript update and build/visual check, master/budget closeout once.
Numerical root: ../benchmarks/artifacts/observation_tt_robust_guide_20260916/.
Result: observation-aware-tt-robust-guide-20260916-result.md.
