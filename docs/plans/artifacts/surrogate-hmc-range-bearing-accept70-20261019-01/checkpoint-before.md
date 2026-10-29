# Active surrogate-HMC checkpoint

Updated 2026-10-19 after the 10:44 UTC status check. Goal: complete the five-model
surrogate-HMC campaign. Latest owner direction: target70% acceptance. The
range-specific fresh-launch95% override is now70%; 24 focused tests pass.
The completed range run remains a95%-target result. A fresh70% range run has
not been executed. The active SIR configuration already targets70%.
Range bearing completed the sampling rule; SIR continues. Campaign incomplete.

Checkout: /home/chakwong/BayesFilter-ledh-graceful-failure-execution-20260917
Branch: ledh-graceful-failure-execution-20260917; HEAD 10884528ca8b1a7463895256d30b9655d69c0955.
Preserve unrelated dirty changes. GPU commands require escalation; verified
memory growth is mandatory. Python: /home/chakwong/anaconda3/envs/tftwogpu/bin/python.

Active execution plan: docs/plans/surrogate-hmc-energy-warning-repair-2026-10-19.md.
Latest configuration repair: docs/plans/surrogate-hmc-acceptance-target-repair-2026-10-19.md.
Evidence A: docs/plans/artifacts/surrogate-hmc-acceptance-target-repair-20261019-01.
This changes only future fresh launches in run_surrogate_hmc_48h_extension.py.
Its pre-edit source is in A/source-before, matching the old campaign source pin.
The live supervisor already loaded the prior source; no live settings changed.
Evidence D: docs/plans/artifacts/surrogate-hmc-energy-warning-repair-20261019-01.
Campaign E: docs/plans/artifacts/surrogate-hmc-48h-extension-20261018-01.
E/run/budget.json is the shared ledger: 172800 worker seconds, maximum six
attempts, deadline 2026-10-20 08:00:41 UTC (20 October 16:00:41 Hong Kong).
Six attempts started, one still running. Completed attempts charged 42505.16s;
the active SIR attempt started 2026-10-18 21:08:04 UTC and also consumes budget.
At04:00UTC about28 hours remained until the absolute deadline; recompute from
current time for any action. No additional
attempts remain under the current cap. Never infer budget from stale aliases.

Active supervisor PID 4166669; SIR master PID 78222; both confirmed alive.
SIR log identifies GPU worker PID78224. Recheck PIDs before acting.
Receipt E/run/queue-energy-repair-01/handoff.json is TAKEN_OVER.
Range output E/run/range_bearing-energy-resume-01/range_bearing/inference/hmc.json:
VERIFIED, 2000 warmup and 2000 retained per chain, four chains. Maximum modern
R-hat1.00745857, acceptance93.8–94.9%, bulk ESS672–1523, tail ESS1310–2504.
Zero arithmetic divergences; retained maximum absolute energy error1006.58
remained a warning. Completed2026-10-18 21:08:04 UTC (19 October05:08 Hong Kong).
Restart wall time9134.72s. Posterior/reference agreement is not checked.
Active SIR output E/run/sir_d18-energy-retry-01/sir_d18/inference/public-tuning:
fresh-progress.json reports RUNNING, 2750 warmup and zero retained per chain.
Latest chunk health checks pass; warmup readiness has not yet passed.

Repaired modules: hmc_tensorflow_verified.py (warning-only finite energy),
hmc_tensorflow_resume.py (exact known source migration, energy-only resumption,
empty-phase reader), run_surrogate_hmc_48h_queue_repair.py (idle queue restart).
Numerical proposal/MH kernel, target, data, priors and seeds are unchanged.
84 focused tests pass, including actual MH rejection, plus six-chunk exact GPU
archive replay. D contains source-before/after, patch, logs and execution manifest.
No energy stop remains in the active sampler; historical gates are replay-only.

Predator-prey completed VERIFIED at E/run/predator_prey-retry-01. SIR's prior
empty-phase schema mismatch is repaired; its current retry has advanced beyond
the restored2000 warmup. Do not launch a duplicate. SIR uses the remaining budget.

Frozen settings: N4032/K2016, FP32/TF32off, GPU/XLA, epsilon1/five OT iterations,
literature model priors. T50 except range T20 with16 flow steps. Four chains:
warmup min2000, latest1000 modern R-hat<=1.05, max10000; retained cumulative
modern R-hat<=1.01, max20000 or budget. No extra validation draws.

Exact next action: monitor E/run/state.json, budget.json, next-phase-plan.json,
and active SIR public-tuning/fresh-progress.json. Let the existing master keep
sampling under its readiness and cumulative stopping rules. Do not restart or
retune a live worker. Further attempts require resolving the exhausted attempt cap.
Evidence: E/status-check-20261019-040045.json and D/range-result-20261019.md.

Prior LGSSM and KSC sampling rule passed (R-hat1.00708 and1.00600). Full
posterior agreement remains not checked; no five-model completion or ranking
claim is established. Detailed earlier history is in D/checkpoint-before.md.
