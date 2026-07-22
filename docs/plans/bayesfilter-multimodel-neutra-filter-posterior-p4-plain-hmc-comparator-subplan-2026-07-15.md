# P4 Subplan: Predator-Prey Same-Target Plain HMC

Date: 2026-07-15

Program ID: `multimodel-neutra-filter-posterior-20260715`

Status: `READY_FOR_EXECUTION`

## Phase Objective

Construct separate tuned plain-HMC comparators for the admitted `PP-UKF` and
`PP-SGQF` filter-posterior identities. Preserve warm-up and retained draws and
admit a comparator only when the shared modern convergence, ESS, health, and
target-status gates pass.

## Entry Conditions

- `PP-UKF` typed target signature:
  `036948f0faaf028d159d7b70337214f01514d732112c2d10e9f7eea1e13b8e30`.
- `PP-SGQF` level-2 typed target signature:
  `8e0a9582fd30643b2e77e7615a21c0d44cc6c1827865ea52c841cc6dbfdde1ad`.
- Both identity roots and every artifact listed by their recursive hash ledger
  must verify before sampling.
- `PP-ZC` remains `TARGET_BLOCKED_SOURCE_ROUTE_MISMATCH`; no sampler is run for
  it.
- The shared GPU/XLA sequential HMC controller and P1 canary remain admitted.

## Research Intent And Evidence Contract

| Field | Frozen contract |
| --- | --- |
| Question | Can fixed-kernel plain HMC sample each exact admitted filter posterior with valid modern diagnostics? |
| Candidate | One separately tuned fixed HMC kernel per target identity |
| Comparator target | The exact same adapter, data, prior, chart, Jacobian, filter settings, dtype, and typed identity admitted in P4 R1B |
| Kernel nomination | Among health-valid probes, maximize minimum rank-normalized bulk ESS; ties use grid order |
| Promotion | Warm-up recent-window modern R-hat `<=1.05`; retained modern R-hat `<=1.01`; minimum bulk ESS `>=1000`; minimum tail ESS `>=400`; all health/status vetoes clear |
| Vetoes | Identity/hash drift, nonfinite state or energy, energy-error divergence, invalid target status, no moved chains, warm-up cap, retained cap, or nonfinite diagnostics |
| Explanatory only | Probe/final acceptance, runtime, truth distance, posterior means, SGQF-versus-UKF differences |
| Not concluded | NeuTra quality, filter exactness, SGQF/UKF superiority, calibration, broad robustness, or readiness |
| Result artifact | Fresh per-cell comparator root containing probe ledger, separate tensor archives, diagnostics, result, manifest, and recursive hashes |

Acceptance is not a tuning or admission objective. It is recorded because zero
or pathological acceptance helps explain a failed ESS or health screen.

## Frozen Kernel And Sampling Design

- Four batched chains in six-probit source coordinates.
- Initial states are the truth-chart audit point plus the fixed offsets
  `(0,0,0,0,0,0)`, `(0.10,-0.10,0.08,-0.08,0.06,-0.06)`,
  `(-0.10,0.10,-0.08,0.08,-0.06,0.06)`, and
  `(0.16,0.08,-0.12,-0.10,0.12,0.04)`. Truth is only an initialization
  convenience and cannot support recovery or calibration claims.
- Fixed `8` leapfrog steps.
- Frozen step-size grid: `0.001`, `0.002`, `0.004`, `0.008`, `0.016`, `0.032`.
- Each probe uses `64` burn-in transitions and `128` retained tuning draws with
  a target- and grid-separated stateless seed.
- A probe is nomination-eligible only when all target/status/energy health
  checks pass and its rank-normalized diagnostics are finite. Select maximum
  minimum bulk ESS; do not use probe R-hat or acceptance as a promotion gate.
- Sequential warm-up uses chunks of `1000`, minimum `2000`, recent window
  `1000`, and maximum `10000` draws per chain.
- Retained sampling uses chunks of `2000`, minimum `4000`, and maximum `10000`
  draws per chain.
- Warm-up and retained stages use distinct fixed seeds and are archived
  separately. Warm-up is never pooled into posterior inference.

No post-result grid expansion or threshold relaxation is allowed. A localized
harness/serialization defect may be repaired in a fresh attempt under the same
scientific contract and comparator budget.

## Default And Assumption Audit

| Choice | Provenance | Failure mode | Early diagnostic | Status |
| --- | --- | --- | --- | --- |
| identity-specific raw source coordinates | admitted six-probit target | posterior geometry may be too correlated for identity-mass HMC | frozen logarithmic probe grid and ESS nomination | baseline comparator geometry |
| eight leapfrog steps | bounded shared fixed-kernel design | trajectory too short or too long | all-grid ESS and movement diagnostics | hypothesis, not default promotion |
| logarithmic step grid | conservative six-dimensional source chart scale | true viable region lies outside grid | no eligible probe or uniformly poor ESS | frozen bounded ladder |
| truth-chart initialization | existing audit point | favorable initialization hides modes | four dispersed chains and modern R-hat | convenience only |
| 1000-draw warm-up window | master/runbook contract | local window passes despite earlier drift | preserve all warm-up and report cumulative history | reviewed campaign policy |
| 10000 caps | owner direction and shared controller | difficult posterior remains inconclusive | explicit cap flags | hard budget |

## Required Artifacts And Checks

1. Verify source identity roots and recursive artifact hashes.
2. Reconstruct each adapter, independent posterior recomposition, and
   repository-issued typed identity; require exact signature equality.
3. Run static import/source checks: no NumPy, host callbacks, local sampler
   implementation, or Python sample/time loop in the target or HMC graph.
4. Run one trusted GPU/XLA probe ladder per cell with memory growth configured
   before logical-device initialization.
5. If a probe is eligible, run the shared sequential controller with separate
   warm-up and retained archives and full rank-normalized split/folded R-hat,
   bulk ESS, and tail ESS diagnostics.
6. Write immutable result/run-manifest/hash records and classify the cell as
   `COMPARATOR_ADMITTED` or `COMPARATOR_BLOCKED`.

## Repair, Handoff, And Stop Conditions

- Serialization, reporting, or harness failures trigger a focused regression
  and one fresh attempt without changing the frozen grid, target, or criteria.
- A target/status failure reopens the owning target cell; it cannot be repaired
  by sampler tuning.
- No eligible kernel, warm-up failure at 10,000, or retained failure at 10,000
  yields `COMPARATOR_BLOCKED` for that cell.
- Once both admitted cells have terminal comparator states and `PP-ZC` is
  recorded blocked, write the P4 comparator result and either start R3 for each
  `COMPARATOR_ADMITTED` cell or close the blocked cell honestly.
- Stop program-wide only for shared harness contamination, corrupted common
  evidence, unavailable trusted GPU, or exhausted program budget.

## Skeptical Pre-Execution Audit

Decision: `PASS`.

The exact registered target is the comparator, not the PF diagnostic. Probe
acceptance is not promoted into a tuning criterion. Short-chain diagnostics
nominate only; fresh sequential evidence controls admission. Warm-up and
retained samples are separated, both rank and folded R-hat are required, ESS
thresholds and caps are fixed, and a failed candidate remains cell-local.

