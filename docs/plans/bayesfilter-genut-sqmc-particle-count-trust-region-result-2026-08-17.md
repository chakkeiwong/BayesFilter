# GenUT SQMC Particle-Count and Trust-Region Result and Reset Memo

Date: 2026-08-17  
Plan: `docs/plans/bayesfilter-genut-sqmc-particle-count-trust-region-plan-2026-08-17.md`  
Trust-region artifact: `docs/benchmarks/artifacts/genut-sqmc-particle-count-trust-region-20260817/claim_attempt01/result.json`  
Legacy baseline artifact: `docs/benchmarks/artifacts/genut-sqmc-particle-count-trust-region-20260817/claim_attempt02/result.json`

## Direct Answer

The trust-region repair materially helps numerical validity. At Austria-SIR
`T=20,N=1008`, every legacy dual-cap row was nonfinite and program-invalid:
`0/64` valid across four algorithms and 16 seeds. The repaired claim completed
all `192/192` rows across four algorithms, three particle counts, and 16 seeds.

More particles help the dominant score variance at `N=4032`, but the evidence
does not show a general clean `1/N` law. For repaired permutation, `j0` SD fell
from `599.71` at `N=1008` to `313.35` at `N=4032`, an SD ratio of `0.522`
against the ideal fourfold-count landmark `0.500`. Its total componentwise
score SD ratio was `0.519`. At `N=2016`, however, the `j0` SD ratio was
`0.883`, not the ideal `0.707`.

This weakens the hypothesis that the Austria `j0` problem is only too few
particles. Larger `N` reduces variance, especially at 4x, but `j0` remains
extremely noisy and no exact SIR score oracle exists. The repaired-permutation
mean is relatively stable (`-325.63`, `-307.82`, `-343.90`) compared with its
large MCSEs, but that is not evidence that the score is correct.

## Validity and Trust Repair

| Scope | Legacy dual cap | Trust-region dual cap |
|---|---:|---:|
| `N=1008` smoke, four routes, seed `97701` | `0/4` valid | `4/4` valid |
| `N=1008` baseline, four routes, 16 seeds | `0/64` valid | `64/64` valid |
| Trust-region scaling claim, `N=1008,2016,4032` | ineligible after baseline veto | `192/192` valid |

The executed solver identity was
`genut_column_scaled_lm_smooth_rms_trust_v1`, with LM damping `1e-2`, scale
floor `1e-4`, and trust radius `0.5`. No invalid claim row was deleted.

## Value Results

Values are replicate mean, MCSE, and sample SD over 16 seeds.

| Route | N | Mean | MCSE | SD | Variance / N=1008 |
|---|---:|---:|---:|---:|---:|
| IID dual cap | 1008 | -681.8125 | 0.1347 | 0.5389 | 1.000 |
| IID dual cap | 2016 | -681.6120 | 0.1435 | 0.5739 | 1.134 |
| IID dual cap | 4032 | -681.5253 | 0.0872 | 0.3489 | 0.419 |
| Previous inverse CDF | 1008 | -682.1562 | 0.1450 | 0.5800 | 1.000 |
| Previous inverse CDF | 2016 | -682.0456 | 0.1897 | 0.7589 | 1.712 |
| Previous inverse CDF | 4032 | -682.0504 | 0.1938 | 0.7750 | 1.786 |
| Repaired fixed | 1008 | -682.0488 | 0.2033 | 0.8133 | 1.000 |
| Repaired fixed | 2016 | -681.8648 | 0.1105 | 0.4421 | 0.296 |
| Repaired fixed | 4032 | -681.8779 | 0.1362 | 0.5447 | 0.448 |
| Repaired permutation | 1008 | -682.0496 | 0.1101 | 0.4402 | 1.000 |
| Repaired permutation | 2016 | -682.1409 | 0.1631 | 0.6523 | 2.196 |
| Repaired permutation | 4032 | -681.8192 | 0.1166 | 0.4664 | 1.123 |

Particle growth does not consistently reduce value variance. In particular,
repaired permutation has nearly unchanged 4x value variance and worse 2x
value variance.

## Score Results

Each cell is component mean plus/minus sample SD over 16 seeds.

| Route | N | `j0` | `j1` | `j2` | Total score SD |
|---|---:|---:|---:|---:|---:|
| IID dual cap | 1008 | -153.90 +/- 409.15 | -34.67 +/- 109.83 | 6.76 +/- 1.48 | 423.63 |
| IID dual cap | 2016 | -371.97 +/- 391.53 | 17.87 +/- 118.49 | 5.49 +/- 1.40 | 409.07 |
| IID dual cap | 4032 | -226.93 +/- 287.46 | -31.81 +/- 85.71 | 5.67 +/- 1.36 | 299.96 |
| Previous inverse CDF | 1008 | -246.57 +/- 465.28 | 7.50 +/- 137.22 | 7.45 +/- 1.30 | 485.09 |
| Previous inverse CDF | 2016 | -194.69 +/- 401.37 | -23.27 +/- 122.20 | 6.95 +/- 1.60 | 419.57 |
| Previous inverse CDF | 4032 | -324.78 +/- 277.25 | -2.81 +/- 75.66 | 6.71 +/- 1.73 | 287.39 |
| Repaired fixed | 1008 | -50.68 +/- 519.15 | -37.05 +/- 149.75 | 6.58 +/- 1.91 | 540.32 |
| Repaired fixed | 2016 | -249.08 +/- 385.42 | -7.15 +/- 117.94 | 6.64 +/- 1.02 | 403.06 |
| Repaired fixed | 4032 | -228.36 +/- 307.48 | -19.33 +/- 98.65 | 6.03 +/- 1.47 | 322.92 |
| Repaired permutation | 1008 | -325.63 +/- 599.71 | 20.44 +/- 188.32 | 7.26 +/- 1.18 | 628.59 |
| Repaired permutation | 2016 | -307.82 +/- 529.72 | -3.28 +/- 156.79 | 7.51 +/- 1.87 | 552.44 |
| Repaired permutation | 4032 | -343.90 +/- 313.35 | 0.28 +/- 91.28 | 6.58 +/- 1.34 | 326.37 |

## Particle-Scaling Check

| Route | `j0` SD 2x / baseline | `j0` SD 4x / baseline | `N Var(j0)`: 1x, 2x, 4x |
|---|---:|---:|---:|
| IID dual cap | 0.957 | 0.703 | 1.69e8, 3.09e8, 3.33e8 |
| Previous inverse CDF | 0.863 | 0.596 | 2.18e8, 3.25e8, 3.10e8 |
| Repaired fixed | 0.742 | 0.592 | 2.72e8, 2.99e8, 3.81e8 |
| Repaired permutation | 0.883 | 0.522 | 3.63e8, 5.66e8, 3.96e8 |
| Ideal independent-MC landmark | 0.707 | 0.500 | constant |

Repaired permutation is close to the 4x landmark and its 4x `N Var(j0)` is
only about 9% above baseline. The 2x point is not close, and the other routes
also fail to keep `N Var(j0)` stable across the whole ladder. Thus the result
is endpoint evidence of useful 4x variance reduction, not confirmation of a
general `1/N` scaling law.

## Numerical and Engineering Diagnostics

- Maximum claim TV residual was `1.07e-5`, below the `1e-4` gate.
- State-map saturation was zero in every row.
- Repaired fixed/permutation retained exactly `N` unique ancestors in every
  row. Previous inverse CDF minima were `992`, `1984`, and `3968`.
- Mean minimum ESS increased with `N`, but remained small relative to `N`.
- A dense `N=2016` score smoke attempted a `13.63 GiB` XLA allocation and
  failed before a result artifact was produced. Exact child blocking with
  block size `126` removed the resource blocker without changing the all-parent
  recursion. Dense-versus-blocked Austria parity passed.
- Peak TensorFlow allocation in the terminal process was `8,283,159,040`
  bytes. Memory growth was configured and verified before GPU initialization.

## Run Manifest

| Field | Value |
|---|---|
| Git commit | `dae37183bf4421682b2ad991e2dc0d0f3c53f260` plus recorded dirty source hashes |
| Environment | `/home/chakwong/anaconda3/envs/tftwogpu` |
| Command | `python docs/benchmarks/run_genut_sqmc_particle_trust_austria_20260817.py --stage claim --particle-counts 1008 2016 4032 --resets trust_region` |
| Device | one visible GPU, TensorFlow GPU/XLA, TF32 enabled |
| Seeds | `97701..97716` |
| Wall time | `3465.16 s` (`57.75 min`) |
| Score block | exact child blocks of `126`; all parents retained |
| Trust artifact | `docs/benchmarks/artifacts/genut-sqmc-particle-count-trust-region-20260817/claim_attempt01/result.json` |
| Legacy artifact | `docs/benchmarks/artifacts/genut-sqmc-particle-count-trust-region-20260817/claim_attempt02/result.json` |

All launch-time source hashes in the terminal artifact matched after the run.
The SQMC harness default was changed to `trust_region` only after the artifact
closed; the legacy route remains explicitly selectable.

## Decision Table

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Use trust region for this SQMC test harness | finite validity versus 16-seed legacy baseline | PASS for repaired; legacy `0/64` hard-vetoed | cross-model behavior | retain repaired default in this harness | repository-wide default readiness |
| Larger N helps score variance | `j0` and total-score SD ratios | descriptively supported at 4x | 16 seeds and non-monotone 2x result | use `N=4032` for a focused follow-up if score evidence is needed | exact `1/N` law |
| Too few particles is the sole `j0` cause | residual `j0` SD and missing oracle | NOT SUPPORTED | finite-program bias versus Monte Carlo noise | retain score-correctness investigation | correct SIR score |
| Rank SQMC algorithms | predeclared uncertainty-supported comparison | NOT TESTED | descriptive variance and value differences only | no ranking | repaired permutation is statistically best |

## Inference Status

| Item | Status |
|---|---|
| Hard veto screen | Trust-region claim passed `192/192`; legacy baseline failed `64/64` at `N=1008` |
| Statistically supported ranking | None |
| Descriptive-only differences | Particle scaling, value variance, ESS, weights, and runtimes |
| Default readiness | Only the dedicated SQMC test harness now defaults to trust region |
| Next evidence needed | More independent replications or a predeclared paired uncertainty analysis, cross-model replay, and an admitted score oracle/finite-program correctness gate |

## Post-Run Red Team and Restart State

The strongest alternative explanation is that the apparent 4x scaling is a
16-seed endpoint fluctuation; the non-monotone 2x result supports that caution.
The conclusion would be overturned by a larger replicated ladder whose
repaired-permutation 4x variance ratio is not near one quarter, or by a
same-program correctness failure showing that reduced variance only
concentrates the wrong score. The weakest evidence remains score correctness,
not numerical validity.

Restart from the plan, this memo, `smoke_attempt03`, `smoke_attempt05`,
`smoke_attempt06`, `claim_attempt01`, and `claim_attempt02`. Do not rerun the
legacy route at larger `N` unless a new repair hypothesis specifically
requires it. Do not use the empty/partial smoke attempts as scientific
evidence.
