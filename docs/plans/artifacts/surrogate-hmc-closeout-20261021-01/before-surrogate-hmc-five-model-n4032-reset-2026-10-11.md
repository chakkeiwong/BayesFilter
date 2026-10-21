# Active five-model surrogate-HMC checkpoint, 2026-10-11

User target: LGSSM, KSC SV, predator-prey, SIR at T50 and the saved range
bearing fixture at T20; ordinary FP32, TF32 off, N4032. Branch
ledh-graceful-failure-execution-20260917, HEAD
10884528ca8b1a7463895256d30b9655d69c0955. Master:
scripts/run_surrogate_hmc_five_model_master.py. Runtime:
bayesfilter/testing/surrogate_hmc_five_model.py. Plan:
docs/plans/surrogate-hmc-five-model-n4032-master-2026-10-11.md. Terminal
result:
docs/plans/surrogate-hmc-five-model-n4032-result-2026-10-11.md.

Preparation attempt06 passed all five endpoint and own-scope filter stages in
928.79 seconds. The fresh attempt07 mechanics stage used four chains and two
transitions. KSC, range bearing, LGSSM and predator-prey passed finite
mechanics with zero divergences/fallbacks; SIR stopped at its fair 492.7-second
share with no pilot artifact. Master status is
INCOMPLETE_SEE_MODEL_STATUS and next-phase-plan automatic_continuation=false.
No adaptive warmup or retained posterior sampling has run for any model.
CPU review: 28 passed, two warnings. No active job.

Measured compile-inclusive minimum-2000-warmup extrapolation is 11.1 h KSC,
15.4 h range bearing, 48.8 h predator-prey, 71.1 h LGSSM; SIR is unresolved.
These are feasibility diagnostics, not convergence evidence. The benchmark
q~N(0,I) prior used in mechanics is not the final prior. Do not launch full
inference until the user settles the target prior and total wall budget, and
until the serial score-coordinate cost is either accepted with adequate
allocation or optimized and remeasured.

GPU execution uses memory growth before TensorFlow initialization, GPU/XLA,
TF32 disabled, and tftwogpu Python. CPU checks hide GPU intentionally. No
Claude review gate.

Cost diagnosis update, 2026-10-11: see docs/plans/surrogate-hmc-five-model-cost-result-2026-10-11.md and artifacts/surrogate-hmc-five-model-cost-20261011-01. Both full T50 scalar forces and endpoints compiled and returned finite valid outputs on the RTX 4080 SUPER. Warmed scalar force: LGSSM 15.07/15.59 seconds, SIR 35.21/35.61 seconds. The independent endpoints were only 0.02-0.04 and 0.10-0.11 seconds, respectively. The 71.1-hour LGSSM figure above is compile-inclusive and must not be used as a steady-state forecast. SIR remains a mechanics pilot timeout, not a numerical failure. Current next action within the existing master is to evaluate exact-work reuse and add warmed pilot telemetry, with parity checks before any renewed HMC run. No full HMC is running.
