# Remaining-budget KSC continuation, 2026-10-13

This is the next independent scope in the authorized five-model shared repair
master. LGSSM, SIR and range now have fresh current-source mechanics passes.
KSC and predator-prey have earlier-snapshot passes, so their source-bound
selections cannot authorize new mechanics. Choose KSC first in the established
model order; this is not a new research direction.

Question: does current-source KSC at T50/N4032 pass fresh arithmetic checks and
short four-chain HMC mechanics? Baseline is the same finite filter in FP64 on
identical inputs. Preserve the original three-candidate family, controls as
warm starts, uniform FP32/noTF32, direct Gaussian contractions, K2016, GPU/XLA,
memory growth, model, benchmark normal prior and fixed1201-node endpoint.
Fresh calibration seeds26101251/52 and validation26101253/54 are disjoint;
offset40 has not previously been evaluated for KSC.

Promotion here is only exploratory mechanics: own-scope final value and every
score component within1% scaled error, finite full T50 recursions, and healthy
cold/warm two-step trajectories with no fallback/divergence. These are not
posterior promotion criteria. Acceptance, energy error and runtime are
descriptive. Independent KSC grid diagnostics remain explanatory and cannot
certify posterior-region reference adequacy. Practical heuristic dominance
and cumulative inference have not been evaluated.

Budget:1311.624 worker-wall seconds remain of the authorized36000. Cap this
advance at1300 seconds: tuning1000, mechanics300, one worker. The previous
KSC tuning consumed1102 and mechanics269 seconds, so completion within this
cap is uncertain. Preserve partial results if time expires; do not drop
candidates, relax precision, borrow stale authority, or exceed the campaign
budget. Only extend the internal attempt counter63 to65 for these two bounded
phases. That count is an implementation limit, not additional GPU authority.

Default/assumption audit: FP32, N4032 and horizon come from the owner. Direct
contractions have paired output evidence but speed remains descriptive.
Baseline controls are warm starts; selection requires current fresh evidence.
L1/epsilon0.002 and two steps are mechanics fixtures, not tuned HMC defaults.
The fixed KSC grid is an approximate endpoint pending posterior-region
refinement. These limitations prevent full posterior or canonical LEDH claims.

Skeptical pre-run audit: wrong-source selection would invalidate mechanics;
the driver checks exact scope. Lower arithmetic error would not establish a
better force or posterior. A timeout diagnoses insufficient bounded compute,
not a scientific failure. Changing no runtime source preserves completed
range/SIR/LG scopes. This plan passes for the limited engineering question.
Local candidate failure triggers further repair only within remaining budget;
invalid data, wrong source, missing authority, numerical mechanics veto and
the total time cap prevent dependent execution.

Command:
    /home/chakwong/anaconda3/envs/tftwogpu/bin/python scripts/run_surrogate_hmc_performance_repair.py advance --model ksc --execution-plan docs/plans/artifacts/surrogate-hmc-fresh-scope-advance-20261012-01/guard-ksc-execution.json --precision-results docs/plans/artifacts/surrogate-hmc-shared-gaussian-precision-20261012-01/precision.json --output docs/plans/artifacts/surrogate-hmc-guard-advance-ksc-20261013-01 --budget-seconds 1300 --mechanics-timeout 300 --campaign-ledger docs/plans/artifacts/surrogate-hmc-repair-continuation-20261012-01/budget.json

Each worker records source, command, seeds, environment, GPU memory policy,
wall time and full output. The driver refreshes next-phase-plan.json between
tuning and mechanics. After this bounded phase, reassess actual remaining
compute; do not call the full posterior master complete.
