# Contract E--TP Phase 7 Same-Target All-Model Comparison Result

metadata_date: 2026-07-15
status: PHASE7_COMPLETE_HANDOFF_TO_PHASE8_REFINEMENT
plan: `docs/plans/bayesfilter-contract-e-tp-phase7-same-target-all-model-comparison-plan-2026-07-15.md`
controlling_ledger: `docs/benchmarks/artifacts/contract_e_tp_all_models_2026_07_15/phase7_same_target_comparison_20260715/comparison_ledger_v2.json`
controlling_ledger_sha256: `785bf948af8dff728b4087269bac7d965fdaf47ac750253d7b33840d147cbf5e`

## Decision

Phase 7 is complete as a center-only, deterministic, descriptive same-target
comparison.  Contract E--TP is derivative-correct for every populated finite
program.  It remains close to its declared reference on LGSSM, actual SV,
KSC-SV, and predator--prey at the tested prefixes.  Generalized SV is a
row-specific negative result for the tested progressive feature family, with a
failure already visible at the first reset (`T=2`).

The fixed-parameter adjacent-state squared-TT extension is derivative-correct
for its own finite scalar but inaccurate against every scalar-row reference at
the current degree/order/rank.  It is `extension_or_invention`, not a Zhao--Cui
parameter-learning comparator.  No current row has a Zhao--Cui source-route
parameter-learning comparator, and no Contract E--Chol all-row artifact is
admissible in this campaign.

All cross-method gaps below are descriptive.  No equivalence margin or
statistically supported ranking exists.

## Contract E--TP Results

| Row | T | Value difference to reference | Worst componentwise score gap | Sign reversal | Interpretation |
| --- | ---: | ---: | ---: | --- | --- |
| LGSSM | 2 | `2.0463e-4` | `0.2496%` | none | center diagnostic viable |
| LGSSM | 10 | `-4.6388e-4` | `0.3048%` | none | center diagnostic viable |
| LGSSM | 50 | `-8.6753e-4` | `0.7362%` | none | center diagnostic viable |
| actual SV | 1 | `3.3893e-10` | `5.85e-6%` | none | primitive tie-out |
| actual SV | 2 | `8.1542e-7` | `0.00661%` | none | first-reset tie-out |
| actual SV | 10 | `-2.6057e-5` | `0.05212%` | none | short-prefix viable |
| KSC-SV | 1 | `4.9491e-7` | `0.00239%` | none | primitive tie-out |
| KSC-SV | 2 | `2.9660e-7` | `0.00200%` | none | first-reset tie-out |
| KSC-SV | 10 | `-3.1074e-5` | `0.03067%` | none | short-prefix viable |
| generalized SV | 1 | `1.6249e-9` | `0.00143%` | none | primitive tie-out |
| generalized SV | 2 | `1.6410e-3` | `104.5%` | `gamma` | first-reset feature failure |
| generalized SV | 10 | `2.5042e-3` | `9.525%` | none | tested feature family negative |
| predator--prey | 2 | `-3.8017e-8` | `2.36e-5%` | none | semi-analytic tie-out |
| predator--prey | 5 | `-8.0911e-4` | `0.03085%` | none | viable versus approximate SGQF |

The `T=2` generalized score is
`[0.0491503,-0.00342589,-0.00930175]`, while the dense reference is
`[-0.00221471,-0.00462677,-0.00927258]`.  Its same-scalar FD and chart pass.
The claimed target is therefore correct and the computed TP quantity is the
total derivative of that finite TP scalar, but that scalar is a poor
approximation to the dense filtering reference in the `gamma` direction.

## Adjacent-State Extension Results

| Row | T | Value difference to reference | Worst componentwise score gap | Sign reversal | Own-scalar FD |
| --- | ---: | ---: | ---: | --- | --- |
| actual SV | 1 | `-0.01801` | `16.05%` | none | pass |
| actual SV | 2 | `-0.13230` | `13.70%` | none | pass |
| actual SV | 10 | `-1.03106` | `37.79%` | none | pass |
| KSC-SV | 1 | `0.002673` | `26.44%` | none | pass |
| KSC-SV | 2 | `-0.14904` | `37.53%` | none | pass |
| KSC-SV | 10 | `-1.03282` | `39.08%` | none | pass |
| generalized SV | 1 | `-0.12295` | `98.48%` | none | pass |
| generalized SV | 2 | `-0.22392` | `174.4%` | `log_tau` | pass |
| generalized SV | 10 | `-0.96846` | `87.48%` | none | pass |

These gaps do not show a derivative-wiring defect.  They show that degree 8,
17-point per-axis quadrature, rank 2, two fixed sweeps, and the fixed coordinate
box do not accurately approximate these filtering densities.  Phase 8 must
vary capacity one factor at a time from `T=1` before any longer comparison.

## Defects Found And Repaired During Execution Review

### 1. Generalized-SV extension used the wrong first-step time order

The frozen dataset samples `x_{-1}` from stationarity, transitions to `x_0`,
then generates `y_0`.  The extension originally fitted
`p(x_0)g(y_0|x_0)`.  That is wrong relative to the frozen finite target.

The repaired first step fits

\[
 p_0(x_{-1};\theta)f_0(x_0\mid x_{-1};\theta)
 g_0(y_0\mid x_0;\theta)
\]

over `(x_0,x_{-1})`, includes both coordinate Jacobians, integrates axis 1,
and carries the normalized `x_0` marginal.  Its realized two-axis step, mass,
and own-scalar FD pass at `T=1,2,10`.

### 2. Seed-only regeneration did not bind evaluated data bytes

Preparation and execution independently regenerated observations from a seed.
Transcendental float64 operations differed by one ULP across processes, and
historical actual/KSC `T=1` artifacts also bound a stale prefix.  A seed is
provenance, not an evaluated-tensor identity.

Preparations now embed raw, target, and flow observation prefixes and their
TensorFlow serialization hashes.  The TP and extension runners can consume the
same embedded target tensor and fail on a hash mismatch.  The controlling
ledger validates exact observation-prefix hashes for all nine scalar
TP/extension cells.

### 3. Duplicate KSC target-transform implementations rounded differently

Two implementations of `log(y^2+1e-8)` differed at roundoff.  The extension
runner now obtains its model and target observation tensor from the same
Contract E--TP target owner.  Refreshed KSC values and scores changed only at
roundoff, while hashes became identical.

### 4. Artifact builder initially over-applied an SV policy field

The first v2 builder revision required an SV observation-policy field from the
predator schema.  It failed before writing output.  The check is now scoped to
scalar SV; predator retains its separate initial-law/time-order validation.

The unsuffixed ledger predates these repairs and is noncontrolling.

## Mathematical Execution Review

| Check | Verdict |
| --- | --- |
| Same model row and horizon | pass for every populated cell |
| Same parameter vector and coordinate order | pass |
| Same observation bytes | pass for all nine scalar TP/extension cells |
| Initial/time-order convention | pass; generalized repaired to transitioned initial step |
| TP own-scalar derivative | pass everywhere |
| Extension own-scalar derivative | pass everywhere |
| Reference target | exact Kalman, refined dense scalar quadrature, semi-analytic predator `T=2`, approximate SGQF predator `T=5` |
| Zhao--Cui classification | unavailable as a parameter-learning comparator; extension label preserved |
| Cross-method equivalence | unsupported; descriptive only |

## Verification

The final focused command passed:

```text
29 passed, 2 warnings in 10.52s
```

The warnings are TensorFlow Probability `distutils` deprecations.  All 37
available reference/candidate/extension source artifacts rehashed to the
SHA-256 values stored in the ledger.  `git diff --check` passed on Phase 7-owned
files.

All numerical runs were deliberate CPU-hidden TensorFlow float64 diagnostics
with `CUDA_VISIBLE_DEVICES=-1`.  CUDA plugin registration and `cuInit` messages
were initialization noise from the installed TensorFlow build, not GPU use or
GPU health evidence.

## Decision And Inference Status

| Item | Status |
| --- | --- |
| Engineering hard-veto screen | pass for every populated cell |
| Contract E--TP viable rows | LGSSM, actual SV, KSC-SV, predator--prey at tested prefixes |
| Contract E--TP negative row | generalized SV tested progressive feature family |
| Adjacent extension | derivative-correct but inaccurate at current capacity |
| Zhao--Cui source comparator | unavailable all rows |
| SIR observed-data row | blocked target-measure mismatch |
| Statistically supported ranking | none |
| Default/HMC/GPU/full-horizon readiness | false |
| Next action | Phase 8 one-factor feature and TT capacity refinement |

## Post-Run Red Team

The strongest alternative explanation for the extension's poor scores is
finite TT capacity, quadrature, coordinate support, or insufficient fixed
sweeps, not a score-wiring defect.  The smallest discriminating test is a
`T=1` one-factor degree/order ladder.  For generalized TP, the `T=2` failure
with exact feature matching shows that adding more anchors to the same feature
span is not a principled repair; a materially different distributional state
summary is required.

No exact nonlinear filtering, equivalence, superiority, Zhao--Cui parity,
canonical/default status, HMC readiness, leaderboard completeness, or GPU/XLA
readiness is concluded.
