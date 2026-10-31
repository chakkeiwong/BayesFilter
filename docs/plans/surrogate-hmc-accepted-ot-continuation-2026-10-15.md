# Accepted OT continuation of the five-model surrogate-HMC master

## Research question and current decision

The goal remains testing **surrogate HMC** on the five models, including its
corrected target, computational cost, adaptive warmup and cumulative posterior
diagnostics. Filter value/score accuracy and short HMC mechanics are intermediate
evidence; neither completes the goal. The owner has settled the SIR comparison
question and requested an updated executable master and continued execution.

The completed `ot-epsilon1-steps5-01` and `sir-fresh-reference-01` results under
`artifacts/surrogate-hmc-cost-repair-24h-20261013-01` remain the evidence. At the
fresh SIR point the epsilon=1/five-step score is substantially closer to the
independent Monte Carlo reference. The likelihood level is farther away. This
is descriptive evidence at a point, not universal superiority or posterior
accuracy. No new reference or Zhao-Cui run is needed to reopen that decision.

## Executable continuation and evidence contract

Continue the existing `scripts/run_surrogate_hmc_performance_repair.py advance`
controller, rather than another diagnostic-only campaign. The frozen new control
family is `fixed_epsilon1_steps5`. For each model, sequentially:

1. Start from the corrected-source October 14 selected controls, replace only
   epsilon by 1 and reset iterations by 5, then calibrate mixture weights offline
   using fresh calibration/validation partitions (seed offset 100). Preserve the
   old controls as a warm start, not current-source admission.
2. Issue a fresh hashed exact-scope selection only after existing finite-health
   and 1% same-program FP64 arithmetic checks pass on validation data.
3. Check the actual HMC force binding, independent endpoint and short mechanics,
   then measure actual-consumer force and trajectory costs using that selection.
4. Refresh phase-local and campaign-level next-phase state before every worker,
   after each model, and on completion; preserve completed reference evidence.
   Continue independent models if a candidate requires repair.

Primary pass criteria in this computational stage are a valid fresh selection,
the existing 1% scaled FP64 value/score screen, successful actual force binding
and mechanics health, and complete consumer-cost measurements. Difference from
the old OT answer is explanatory, not an accuracy veto. Numerical invalidity,
scope mismatch and missing required diagnostics veto that candidate; source/data
corruption, unavailable GPU/memory policy or exhausted budget veto continuation.
Sinkhorn residuals remain explanatory; no residual threshold is introduced.

Comparators are same-program FP64 arithmetic, reconstructed bound force, and the
actual corrected-HMC endpoint. Equal mixture weights are the cheap calibration
adversary; selected weights cannot be justified by worse heldout loss. Existing
Kalman/KSC references remain diagnostics, evaluated separately by model and local
parameter situation. This stage does not rank algorithms or promote defaults.
It does not establish posterior convergence, model-oracle accuracy across a
posterior, or canonical Contract-E admission. The covariance-proposal force is
still optional. Full posterior execution still needs an explicit target/prior;
the saved normal prior in q is a mechanics fixture, not an inferred owner choice.
This unresolved target does not block the independent computational continuation.

## Scope, assumptions and early checks

N=4032, exact transport chunks K=2016, TF32 off, GPU/XLA, verified memory growth.
LGSSM, KSC SV, predator-prey and SIR have T=50; the saved range-bearing case has
T=20. Uniform FP32 remains the default. SIR retains the measured mixed-precision
exception (FP64 recursion, FP32 pair kernels) because uniform FP32 failed the
arithmetic screen. Contractions remain direct except predator-prey matrix mode.

| Choice | Provenance/status | Failure mode and earliest check |
|---|---|---|
| epsilon=1, five steps | Owner-accepted continuation hypothesis | Validate fresh-data health and arithmetic; do not transfer a superiority claim |
| Other controls | Latest per-model saved selections, warm starts | Check checksum/model/N/T, preserve all other fields; retune mixture weights |
| Fresh seed offset 100 | New deterministic partition, convenience | Check disjointness against prior calibration/validation and claim data |
| FP32 and SIR exception | Owner default plus measured precision repair | Same finite program FP64 comparison, never label it a model oracle |
| Existing mixture solver/floor | Existing scoped calibration protocol | Compare equal weights on validation; retain equal fallback when fit fails |
| Existing chart and nominal probes | Local diagnostic coverage only | Record exact physical parameters; no posterior coverage claim |
| Zero correction steps | Preserved optional covariance-proposal scope | Not a safety calibration or canonical LEDH claim; do not silently add correction |

## Budget and stop conditions

Existing additional 24-hour ledger: 34,730.320010 of 86,400 worker-wall seconds
used; 51,669.679990 seconds (14.352689 h) remain before this continuation.
Compilation, failures and retries count. No additional compute is authorized
or requested. One GPU worker at a time. The administrative attempt limit changes
from 120 to 140 so fifteen planned workers and bounded local repairs fit; this
does not increase the 86,400-second ceiling. Never bypass the time ceiling.

This phase caps wall time at 28,800 seconds (8 h): tuning <=2,400 s/model,
mechanics <=1,200 s/model, consumer cost <=1,200 s/model. Each cap is clipped to
the phase and campaign remainder. Preserve failures and use unique directories
for retries. Do not overwrite or rerun completed prior diagnostics.

Artifacts: `accepted-ot-scope-execution.json`, `accepted-ot-scope-advance-01/`,
`accepted-ot-refresh-review.json`, and campaign `next-phase-plan.json` beneath
the existing campaign root. Worker manifests archive sources, commands, scopes,
memory/device state, seeds, wall times and exact result locations. CPU-only
regressions are engineering evidence only.

Exact launch (from this checkout, escalated GPU permissions):

```sh
/home/chakwong/anaconda3/envs/tftwogpu/bin/python scripts/run_surrogate_hmc_performance_repair.py advance --output docs/plans/artifacts/surrogate-hmc-cost-repair-24h-20261013-01/accepted-ot-scope-advance-01 --campaign-ledger docs/plans/artifacts/surrogate-hmc-cost-repair-24h-20261013-01/budget.json --execution-plan docs/plans/artifacts/surrogate-hmc-cost-repair-24h-20261013-01/accepted-ot-scope-execution.json --precision-results docs/plans/artifacts/surrogate-hmc-cost-repair-24h-20261013-01/fixed-design-full-precision-01/precision.json --controls-warm-start-root docs/plans/artifacts/surrogate-hmc-cost-repair-24h-20261013-01/fixed-design-scope-advance-01 --consumer-cost-after-mechanics --budget-seconds 28800 --per-model-timeout 2400 --mechanics-timeout 1200 --consumer-timeout 1200
```

## Skeptical pre-run audit

The first audit found a material stale-baseline defect: the worker loaded
October 11 controls even though the accepted ablation used the later corrected
selections. Repaired with an explicit checksum-checked controls-only warm start;
fresh current-source selections remain mandatory for consumer execution.
The second stale path was the campaign-level next-phase file; the advance
controller now updates it along with the local phase file. Tests exercise both
wiring paths, numerical control preservation and campaign budget reporting.

The revised plan survives the audit subject to passing focused tests and trusted
GPU checks. The comparison does not use an approximate old filter as an oracle,
change the target silently, or use short mechanics as convergence evidence.
The strongest misleading-pass risk is limited parameter coverage; a later HMC
failure can still expose that. A failed fresh arithmetic screen triggers
local numerical repair, not abandonment of the accepted OT direction.
