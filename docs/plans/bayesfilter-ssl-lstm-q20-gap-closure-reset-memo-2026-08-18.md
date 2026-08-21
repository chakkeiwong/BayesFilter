# SSL-LSTM q=20 Gap-Closure Reset Memo (2026-08-18)

## 2026-08-19 Planning Correction

The final line of the original next-step section was too restrictive.  An
independently converged global physical posterior archive is **not** a
prerequisite for NeuTra training; requiring one is circular because NeuTra is
being developed to make global sampling tractable.  The dense physical run is
now an optional diagnostic comparator, not an upstream NeuTra gate.

Likewise, fixed-HMC runs initialized in separate modes are valid only as
overdispersed diagnostics.  If they remain mode-separated, their conditional
draws must not be pooled or assigned equal weights.  A promoted candidate must
use one exact transformed target and demonstrate initialization forgetting and
cross-mode transitions under that kernel.

The corrected plan is:

`docs/plans/bayesfilter-ssl-lstm-q20-neutra-global-mixing-repair-plan-2026-08-19.md`

The material dense result below remains valid historical evidence; this note
corrects the decision boundary, not its measurements.

## Current State

The 2026-08-18 gap-closure campaign completed Phases A--C/H3.

- Focused preflight: `24 passed`.
- Fresh dense-mass canary: passed short-run mechanics and selection screens.
- Fresh dense-mass material: stopped at 500 discarded warm-up transitions with
  modern R-hat `1.3175028642`, so retained draws are empty and the posterior
  archive is invalid.
- Fresh 32-start target-query discovery: no new competing stationary cluster
  found; this is bounded negative evidence, not exhaustive mode discovery.
- NeuTra retraining and predictive validation were not launched.

## Binding Artifacts

- Plan: `docs/plans/bayesfilter-ssl-lstm-q20-gap-closure-plan-2026-08-18.md`
- Result: `docs/plans/bayesfilter-ssl-lstm-q20-gap-closure-result-2026-08-18.md`
- Dense canary: `docs/plans/artifacts/ssl-lstm-q20-physical-replica-travel-repair-2026-08-10/r1-dense-mass-step-0p35/canary.json`
- Dense material: `docs/plans/artifacts/ssl-lstm-q20-gap-closure-2026-08-18/physical/r2-dense-mass-material-retry/material.json`
- Discovery: `docs/plans/artifacts/ssl-lstm-q20-gap-closure-2026-08-18/discovery/r1/discovery.json`

## Do Not Reuse

- The old seed-B NeuTra archive remains local-positive-mode evidence only.
- The dense canary is nomination evidence, not a posterior sample.
- The dense material warm-up states are discarded and must not enter training,
  HMC, or predictive diagnostics.
- Do not relaunch the identity-mass runner under a new name and call it a repair.

## Next Justified Research Step

Design a new target-specific global sampler hypothesis focused on the remaining
between-chain physical-coordinate disagreement. The next plan should inspect the
four-coordinate chain decomposition and compare a global covariance/transport,
longer hot trajectory, or a staged physical-to-NeuTra bridge. It must preserve:

- no NUTS and `L >= 2`;
- CPU/XLA physical target with row-independent workers;
- finite/status/permutation/invalid-path gates;
- warm-up exclusion and modern rank/folded R-hat;
- SMC interval `[0.405731,0.536018]` as a two-known-region comparator only; and
- no NeuTra retraining or predictive test until an eligible global archive exists.

The bounded multistart result means a third-mode search is not the first repair
to repeat. It remains a limitation to test only if the next global sampler
creates evidence inconsistent with the two-region scope.

## Verification

- `py_compile` passed for all new runners.
- Discovery unit tests: `3 passed`.
- Existing SSL physical/SMC/predictive focused suite: `24 passed`.
- No GPU was used by the CPU/XLA material or discovery runs.
