# SVX-ZC Monograph Fixed-Branch Admission Plan

Date: 2026-07-31
Status: `EXECUTED_TERMINAL_NO_RANK_ADMITTED_2026-07-31`

## Objective

Re-evaluate `SVX-ZC` against the authority in `docs/main.tex` and its included
fixed-branch TT/KR chapters. The target is the BayesFilter-owned scalar
adjacent-state squared-TT approximation, not reproduction of the Zhao-Cui
author MATLAB implementation. The route remains classified as
`extension_or_invention`; that classification is descriptive and is not itself
a veto under the monograph.

## Research Intent Ledger

| Field | Decision |
| --- | --- |
| Main question | Does the monograph-defined fixed adjacent-state TT/KR approximation satisfy its declared numerical and same-scalar derivative contract for exact transformed SV? |
| Candidate | `zhao_cui_fixed_adjacent_state_squared_tt_v1`, coordinate order `(x_t, x_{t-1})`, fixed points/ranks/sweeps, deterministic ridge ALS, positive frozen defensive mass. |
| Baseline | Preserved rank-2, zero-defense historical comparator and independent dense exact transformed-SV sequential reference. |
| Promotion criterion | All hard vetoes below pass for at least one predeclared rank; no ranking among passing ranks is claimed without uncertainty evidence. |
| Promotion veto | Nonfinite value/score, coordinate/Jacobian mismatch, negative/nonfinite density, normalization failure, rank-saturation residual failure, conditioning veto, retained closure failure, branch mismatch, or no decreasing FD window. |
| Repair trigger | A localized implementation or configuration failure with target, data, route family, and bounded budget unchanged. |
| Continuation veto | Target/data corruption, missing independent reference, invalid mathematical contract, or exhausted bounded rank ladder. |
| Explanatory diagnostics | Fit residuals, condition numbers below veto, value/score gaps, runtime, and rank-to-rank differences. |
| Nonclaims | No author-source faithfulness, exact filtering, posterior correctness, HMC convergence, NeuTra readiness, GPU/XLA readiness, superiority, or default readiness. |

## Evidence Contract

The independent reference is `exact_transformed_sv_scalar_dense_reference`
with the same synthetic SV parameters, raw observations, event order, and
horizon. The candidate consumes the derived `log(y^2)` observations; the
reference helper performs that transform internally. This input distinction is
explicit because attempt 02 compared the transformed array to a helper that
transformed it a second time, invalidating only that comparison field.
The candidate value is the cumulative fixed-branch TT/KR scalar. Candidate
score is TensorFlow autodiff through that same finite program; centered finite
differences must reuse the same fixed ledger and recompute fitted cores.

The rank ladder is `(1, 2, 4, 6)` with degree `8`, quadrature order `17`,
horizon `10`, coordinate half-width `8`, two fixed ALS sweeps, and positive
defensive mass `tau=1e-8`. Rank `1` is a diagnostic baseline; any candidate may
pass only if its declared rank-saturation and residual checks pass.

## Hard Veto Definitions

- finite candidate value and every score component;
- affine coordinate forward/inverse and constant log-Jacobian consistency;
- finite, nonnegative density values on fitting and audit points;
- carried one-axis marginal mass within `1e-10` of one;
- maximum reported condition number `<= 1e10`;
- if the candidate is at the rank cap, fit residual `<= 1e-8`;
- retained evaluator closure at the next-step previous-state quadrature points;
- base/plus/minus branch identities equal for every FD row;
- at least one decreasing FD error window per parameter;
- no retained tensor-product grid route.

The dense reference gap is recorded as statistical/descriptive evidence for
this approximation screen, not silently promoted to a theorem. A large gap
does not invalidate the fixed-branch implementation if the structural vetoes
pass; it triggers a numerical-quality repair or leaves the candidate viable
only as a diagnostic approximation.

## Default And Assumption Audit

| Choice | Provenance | Failure mode | Early diagnostic | Status |
| --- | --- | --- | --- | --- |
| Adjacent order `(x_t,x_{t-1})` | Monograph ch36b | Wrong marginal axis or event order | axis/integrated-axis assertions | binding |
| Positive `tau=1e-8` | Monograph defensive-density convention | defense masks a bad fit or changes scalar | record tau and compare zero-defense historical baseline | reviewed admission hypothesis |
| Rank ladder `(1,2,4,6)` | Fixed-rank branch design | rank cap/residual or conditioning failure | per-rank residual and condition diagnostics | reviewed bounded ladder |
| Degree 8/order 17 | Existing SVX-ZC diagnostic scope | quadrature/basis error | independent dense reference and FD ladder | scope-bound hypothesis |
| FP64 CPU | reference/debug execution policy | not GPU evidence | explicit CPU manifest | reference only |

## Skeptical Plan Audit

- Wrong baseline risk: the zero-defense historical route is preserved only as a
  comparator; the independent dense reference is not treated as the candidate.
- Proxy risk: finite value, residual, and FD are engineering/numerical gates;
  none establishes posterior or HMC correctness.
- Missing stop: the rank ladder and one bounded attempt per rank are explicit;
  no unbounded adaptive rank or budget expansion is allowed.
- Unfair comparison risk: all ranks use identical data, parameter point,
  event order, basis degree, quadrature order, coordinate map, and horizon.
- Hidden assumption risk: positive defense, rank cap, and fit tolerance are
  recorded above and appear in each manifest.
- Artifact risk: each output is written under a fresh versioned root and cannot
  overwrite the historical comparator.

Audit verdict: `PASS_FOR_BOUNDED_CPU_ADMISSION_DIAGNOSTIC_ONLY`.

## Exact Commands

Focused regression:

```bash
CUDA_VISIBLE_DEVICES=-1 TF_CPP_MIN_LOG_LEVEL=2 \
python -m pytest -q tests/highdim/test_zhao_cui_fixed_adjacent_tt_tf.py
```

Admission ladder:

```bash
CUDA_VISIBLE_DEVICES=-1 TF_CPP_MIN_LOG_LEVEL=2 \
python docs/benchmarks/run_neutra_svx_zc_monograph_admission_20260731.py \
  --output-root docs/plans/artifacts/bayesfilter-svx-zc-monograph-admission-20260731/attempt01
```

## Artifact Contract

The runner writes `result.json`, `run_manifest.json`, and one per-rank JSON
record under the fresh output root. Every serious artifact records git commit,
command, environment, CPU/GPU choice, seed, horizon, data hash, branch/config
identity, wall time, plan path, and nonclaims.

## Next Step Rules

If one or more ranks pass hard vetoes, update the SVX-ZC result/reset memo to
remove only the obsolete source-route blocker and classify the route as a
monograph fixed-branch approximation. Do not add it to NeuTra training until a
separate batch-native TensorFlow target adapter and target-specific tuning plan
pass review. If no rank passes, preserve the structural failure and repair only
the smallest failed gate under the remaining bounded scope.

## Execution Closeout

Attempt 02 is preserved as a harness-failure artifact: its independent dense
reference was accidentally given already-transformed observations and applied
the log-square transform a second time. Attempt 03 repaired that comparison
while leaving the candidate contract unchanged.

Attempt 03 artifact:
`docs/plans/artifacts/bayesfilter-svx-zc-monograph-admission-20260731/attempt03/result.json`

Results by rank:

| Rank | Hard status | Max fit residual | Max condition | Dense value gap/obs | FD/branch |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | blocked rank-saturation | 0.0663071 | 1.00 | 0.136531 | pass |
| 2 | blocked rank-saturation | 0.0566693 | 4.28 | 0.103106 | pass |
| 4 | blocked rank-saturation | 0.0564391 | 7.82e3 | 0.100878 | pass |
| 6 | blocked rank-saturation | 0.0564383 | 2.69e4 | 0.100853 | pass |

Coordinate/Jacobian consistency, positivity, finite values/scores, carried
marginal closure, condition-number ceilings, same-scalar FD branch identity,
and the no-retained-grid check passed for all four ranks. Every rank failed
the declared rank-saturation veto (`residual <= 1e-8`), so no rank was
admitted. This is a numerical approximation failure, not a source-route
governance failure.

Decision: keep `SVX-ZC` out of the NeuTra executable registry. The registry
blocker is now `TARGET_BLOCKED_FILTER_ADMISSION` with reentry rung
`fixed-branch numerical admission`. No training or HMC was launched.
