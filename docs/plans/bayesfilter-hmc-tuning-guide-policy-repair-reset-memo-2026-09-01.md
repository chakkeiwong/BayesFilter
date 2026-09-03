# Reset Memo: Repaired HMC Tuning Contract

Date: 2026-09-01  
Read with: `docs/plans/bayesfilter-hmc-tuning-guide-policy-repair-plan-2026-09-01.md`  
Result: `docs/plans/bayesfilter-hmc-tuning-guide-policy-repair-result-2026-09-01.md`

## Current state

The active fixed-transport HMC tuner is governed by
`measured_joint_grid_v1`. A claim-bearing configuration must provide a finite
explicit `step_size_candidates` grid, at least two distinct leapfrog counts,
replicated fixed-kernel efficiency selection, and a disjoint held-out check.
Every declared pair is run before selection. A factor-of-two neighbor inferred
from an acceptance observation is never evidence for that neighbor.

The payload distinction is intentional:

- `mechanics_validated` means finite target/score and valid transition health;
- `tuning_candidate` means the replicated efficiency screen also passed; and
- `posterior_ready` is always false in this tuner and requires a separate
  retained-chain assessment.

Mean Metropolis acceptance probability and binary acceptance are separate
fields. Acceptance outside the target band is a repair diagnostic on the
measured route. Missing/non-finite acceptance, missing movement, non-finite
target/score values, divergence, or invalid telemetry remain hard failures.

## Why the old result must not be reused

The q=20 Phase 9A failure showed acceptance near one at two different step
sizes for the same fixed `L`, followed by a cap hit. For a unit harmonic
oscillator,

`theta(epsilon) = acos(1 - epsilon^2 / 2)` and trajectory phase is
`L * theta(epsilon)`.

That phase is not monotone in a way that justifies directional repair, and a
near-return trajectory can have high acceptance with poor retained movement.
The old attempt-05 record therefore remains historical evidence only. Do not
warm-start, baseline, or relabel it under the repaired schema.

## Next valid q=20 action

Prepare a new target-specific subplan for Phase 9A/9B with:

1. an explicit `(epsilon, L)` grid chosen and justified for the exact chart,
   beta, dimension, dtype, and backend;
2. a declared candidate-count and wall-time budget;
3. disjoint tuning, replicated-selection, held-out, and retained-chain seeds;
4. target/value/score parity and transport-identity checks before HMC;
5. movement, energy, divergence/status telemetry, modern R-hat, ESS, and mode
   coverage with their roles declared before execution; and
6. a fresh versioned output root, manifest, result note, and post-run red-team
   review.

Do not reopen Phase 9B merely because a grid command completes. A passing
mechanics screen is only a nomination for the later posterior gates.

## Caller migration rule

The active q=20 preflight and HNN native caller already pass explicit measured
grids. The July LGSSM serious-validation caller is deliberately marked
`legacy_directional_diagnostic_v1`; it may produce diagnostic records but its
result cannot build a verified handoff. Any newly discovered caller must be
migrated to a reviewed target-specific measured grid before it is described as
claim-bearing. Do not silently fill in a missing grid or reuse the old fixed-L
fields as a default.

## Execution boundary

This repair used CPU-hidden local tests only. The user request authorized the
repair without an intermediate click. The narrow optional allow-list rule is
`bash /home/ubuntu/python/BayesFilter/scripts/run_hmc_tuning_policy_tests.sh`
(the relative form is equivalent from the repository root); broad interpreter
or GPU rules are unnecessary. A future GPU campaign is a separate
authorization boundary and should allow exactly
`bash /home/ubuntu/python/BayesFilter/scripts/run_ssl_lstm_q20_tempered_rkl_phase8_gpu_default.sh`
with GPU 0 exposed. Use one bounded trusted launcher, set
`TF_FORCE_GPU_ALLOW_GROWTH=true` before import, verify growth on every visible
physical GPU before initialization, and record device provenance. Do not
request approval for each retry while the scientific contract and bounded
budget are unchanged.

## What remains open

The guide/policy repair is complete, but q=20 target-specific tuning,
long-chain posterior diagnostics, mode exploration, and any HMC/default
promotion remain open research tasks. The missing historical LGSSM comparator
archive is also unresolved and must be restored or the old campaign must stay
diagnostic-only.

The ordinary Gaussian tuner oracle test remains a separate residual: its
verification reached the R-hat cap (`budget_exhausted`, no hard veto) while the
legacy assertion expects `passed=True`. The ordinary tuner was not changed in
this repair. Do not alter the repaired fixed-transport criteria to mask it;
open a separate ordinary-tuner fixture/verification plan if that route is to
be made claim-bearing.
