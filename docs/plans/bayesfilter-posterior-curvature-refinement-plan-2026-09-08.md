# Fixed-center posterior curvature refinement: implementation and test plan

Date: 2026-09-08
Status: implementation under review; final bounded synthetic validation pending
Requested by: owner, in the MacroFinance discussion of generic BayesFilter geometry

## 1. Objective, scope, and authority

Add `refine_posterior_local_curvature` to BayesFilter. Given an already selected
center and a provisional lower-Cholesky position factor, estimate a regional
quadratic precision from analytical scores at useful standardized widths.
Return validated position geometry, never an HMC mass artifact or MAP certificate.
Keep the center unchanged even if a probe has a higher exact log density.

Runtime classification: accepted TF/TFP implementation direction; bounded eager
orchestration of TensorFlow float64 numerical kernels and batch-native target
callbacks. CPU-hidden execution here is a synthetic unit/integration-test scope,
not a default production or GPU-readiness claim. No NumPy numerical computation
or legacy HMC imports may occur on the new public call path. Test-only NumPy
oracles are permitted. No transport training, HMC, sampler tuning, retained
samples, real data, or MacroFinance target execution is in scope.

Authority is `/home/ubuntu/python/BayesFilter`, resolving to
`/home/ubuntu/workspace/BayesFilter`, base HEAD
`d2124d425b0ea0ae0e3e5f4246bd6b03ff8a2170`. The checkout is already dirty;
preserve all pre-existing edits. Do not use the historical `/tmp` BayesFilter
checkout, clean the tree, create branches, or commit. The owner requests this
opt-in extension, not a change of localization or HMC defaults.

## 2. Existing implementation and reuse boundary

`fixed_center_curvature.fit_fixed_center_curvature` already separates center
location from fitting, but its selection/orchestration uses NumPy and imports
HMC artifacts. Calling it from a newly admitted runtime would inherit that debt.
Extract its existing dense calculation (least squares followed by symmetrizing)
into `score_curvature_tf.py`. Both the legacy dense fitter and the new helper
call that exact shared kernel. Leave historical structured selection and status
semantics unchanged and regression-test them; this is not their full migration.

New module: `posterior_curvature_refinement.py`.
Public exports from `bayesfilter.inference`:
`PosteriorCurvatureRefinementConfig`, `PosteriorCurvatureRefinementResult`, and
`refine_posterior_local_curvature`. Use direct lazy exports so the helper does
not import unrelated HMC modules while resolving its name.

## 3. Mathematical contract

Let the fixed center be `c`, provisional covariance `F F^T`, and `F` a finite
lower-triangular factor with positive diagonal. Row-vector positions satisfy
`theta = c + z @ F.T`. The target callback supplies `ell(theta)` and its exact
matching raw-coordinate score. There is no change to the target measure.

In column notation, `g_z(z) = F.T g_theta(c + F z)` and the model is
`g_z(z) ~= g_z(0) - K z`. In row notation, transform scores with `scores @ F`.
Retain the actual nonzero center-score intercept; never force it to zero.
Fit unrestricted score least squares and symmetrize the fitted coefficient,
matching the legacy dense algorithm. Do not misdescribe this as the exact
symmetry-constrained least-squares optimum for a nonquadratic target.

For accepted SPD `K`, `Sigma_new = F K^-1 F.T`. Compute this through Cholesky
solves, not explicit matrix inversion. Return the lower Cholesky of `Sigma_new`
and verify finite factor reconstruction. For an exact quadratic, recovery is
exact up to numerical error regardless of nonzero center score or pilot widths.
For nonquadratic targets this is a regional score approximation, not the exact
point Hessian, posterior covariance, or globally valid whitening.

Reference method: inverse-Hessian Gaussian approximation (Stan Reference Manual,
Laplace approximation, `https://mc-stan.org/docs/reference-manual/laplace.html`),
and the existing BayesFilter dense score-regression implementation. The custom
extension is a fixed-center, pilot-scaled, independently checked region, not a
new optimizer. MacroFinance monograph `ch20_hmc.tex`, `eq:mass_hessian`, uses
negative log-posterior curvature as momentum mass; position covariance is its
inverse. No mass conversion is performed by this helper.

MathDevMCP gate: `codex mcp list` reports no servers; the prescribed legacy
`/home/chakwong/miniconda3/bin/python` does not exist. The working fallback is
`PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=/home/ubuntu/workspace/MathDevMCP/src
/home/ubuntu/miniforge3/envs/tf-gpu/bin/python -m mathdevmcp.cli doctor`.
The specific `search-latex --root docs/latex-papers/CIP_monograph --file
chapters/ch20_hmc.tex --limit 2 Hessian` lookup finds `eq:mass_hessian`.
Use SymPy for an independent factor-orientation identity and exact Gaussian
tests for the code/equation correspondence; lookup alone is not a proof.

## 4. Concrete API and execution rules

Inputs: batched value/score callback `[B,D] -> ([B],[B,D])`, center `[D]`, pilot
factor `[D,D]`, required batched boolean eligibility callback `[B,D] -> [B]`,
config, and optional caller lineage dictionary. Callbacks must be deterministic,
row-independent, TensorFlow float64, and express the same target/data/chart.
An always-true eligibility callback is appropriate only for a genuinely
unrestricted target, not for targets returning finite rejection sentinels.

Defaults, prospective engineering thresholds (not scientific universal values):

| Setting | Value |
| --- | --- |
| Coordinate half-width | 1.0 |
| Fitting proposal | independent uniform coordinates in `[-width,width]` |
| Replicates | 2 |
| Rows in each train, selection, audit, and final proposal partition | `max(32,4D)` |
| Physical callback batch size | 64 |
| Seed | 20260908, folded with distinct partition IDs |
| Maximum physical evaluated rows | 10000 |
| Maximum design condition number | 1e6 |
| Maximum fitted precision condition number | 1e10 |
| Selection and untouched-audit relative score RMSE cap | 0.20 |
| Pairwise generalized precision eigenvalue interval | `[1/1.5,1.5]` |
| Refined-factor Gaussian proposal relative score RMSE cap | 0.35 |
| Factor reconstruction relative/absolute tolerances | 1e-10 / 1e-12 |

The box half-width is a multiple of pilot standard deviations in decorrelated
coordinates; uniform-coordinate SD is `width/sqrt(3)`. It is not a unit Euclidean
ball, and not a Gaussian probability-content claim. There is no automatic
chi-square radius shrinkage. The final independent proposal cloud uses standard
normal coordinates and the refined factor, checking whether the regional model
and target support still hold at the proposed initialization scale.

Evaluate one padded center batch, independent training/selection partitions,
freeze the mean accepted replicate precision, and only then evaluate the audit.
After audit success, construct the factor and check a separate Gaussian proposal
cloud. Audit/proposal failures veto; they never select another candidate or
silently reduce width. No retries, recentering, resampling invalid rows, automatic
diagonal fallback, or eigenvalue clipping. The caller can plan a separate retry.

The physical batch shape is fixed, padding the final chunk by repeating its last
row. Count padding and center repeats in the hard evaluation budget; exclude
padding from statistics. Record logical rows, physical attempted rows, target
rows, callback counts, seed/role IDs, and failure partition. Precheck the entire
worst-case budget before calling the target. Eligibility is checked before score
evaluation. Any ineligible/nonfinite row stops with no usable factor. Shape/type
contract mistakes raise natural explicit exceptions rather than masquerading as
scientific failure. Failed callbacks are not silently swallowed.

Reject rank-deficient or ill-conditioned designs, non-SPD raw precision,
excessive precision conditioning, inconsistent replicates, failed holdouts,
and nonfinite factor operations. Report normalized center score using the refined
unshrunk factor for interpretation only. The caller's `s_c <= 0.5` localization
policy is separate: this function must not introduce a second center optimizer.

Result contains acceptance/status, immutable tensor center and pilot factor,
refined factor/covariance when accepted, diagnostic precision where available,
primitive diagnostics, a JSON-safe payload, caller lineage, and explicit
nonclaims. Accepted status is `eligible_for_local_position_factor`, never HMC
eligibility. Failure returns no usable covariance/factor.

## 5. Tests and acceptance criteria

### Unit tests

1. Exact dense score kernel, nonzero intercept, graph/eager parity, rank checking.
2. Config/input shape, finite values, triangular positive factor, boolean
   eligibility contract, invalid budgets before target calls.
3. One-dimensional, diagonal, and correlated exact Gaussian recovery.
4. Nonzero-center invariance and no center mutation on improving cloud points.
5. Both fitting and Gaussian-proposal factor orientation and score pullback.
6. Fixed callback shapes, partial batches, exact physical/logical row accounting,
   early-stop counters, seed reproducibility and distinct partitions.
7. Rank deficiency, raw non-SPD/flat precision, conditioning, nonfinite score,
   nonfinite value, and finite rejection sentinel handling.
8. Selection failure avoids audit calls; fresh audit can veto without refitting;
   final proposal can veto without shrinking or restarting.
9. JSON payload round trip, explicit nonclaims, direct lazy public export.

### Regression tests

1. Compare shared TF kernel and new helper with the existing fixed-center dense
   fitter on the same well-conditioned and ill-conditioned exact score clouds.
2. Preserve old `test_fixed_center_curvature.py`,
   `test_posterior_local_initializer.py`, and documentation-contract tests.
3. Gaussian factor must not shrink to fit the pilot probe radius; vary pilot
   widths while checking invariant recovered raw covariance.
4. Correlated non-diagonal factor catches `F` versus `F.T` and score pullback bugs.
5. Nonzero score intercept and large additive log-density offsets do not alter
   covariance or residual normalization.
6. Affine unit changes, padding/partition accounting, no HMC artifact creation,
   and no NumPy/MacroFinance/legacy-HMC import on the new public path.

### Independent integration executable

`scripts/run_posterior_curvature_refinement_integration.py --output <fresh.json>`
runs without MacroFinance imports or data. Standalone analytical TF targets:

- shifted 1D Gaussian;
- correlated Gaussian with six orders of covariance conditioning;
- Gaussian with deliberately too-small/too-large pilot factors;
- 142-dimensional correlated Gaussian using the same generic API;
- mildly nonquadratic strongly log-concave quartic target (pass under declared
  approximation tolerances, without calling the approximation exact);
- strongly curved banana (reject inaccurate regional approximation);
- bimodal mixture at its saddle (reject non-SPD curvature);
- flat direction and bounded-domain/finite-sentinel targets (reject);
- target-only Host-XLA versus eager Gaussian value/score/refined-geometry parity.

Keep exact analytic/reference covariance errors, all pass/fail statuses, seeds,
actual counters, device-hidden environment, source hashes, start/elapsed times,
and per-case reasons in one JSON report. Expected-rejection cases pass only when
the declared rejection class occurs and no factor escapes. No tolerance changes
to rescue an observed result; repair an implementation defect or record failure.
Synthetic target-only XLA evidence does not qualify the eager refiner itself,
GPU, a user target, training, or full-chain XLA.

## 6. Careful self-review and resolved issues

| Risk | Resolution before implementation |
| --- | --- |
| Existing fitter imports NumPy/HMC | Extract shared pure-TF dense kernel; new helper does not import legacy module. |
| Accidental demand for exact MAP | Fixed center, nonzero intercept, no objective-improvement veto. |
| Ambiguous one-SD radius | Coordinate box convention and variance explicit; final Gaussian proposal separate. |
| SPD manufactured by clipping | Require raw SPD; reject instead of silently projecting. |
| Tiny fit scale mistaken for covariance width | No radius-based factor shrinkage; exact-Gaussian regression. |
| Fit lacks observations in all directions | Full-rank design and SVD conditioning checks. |
| Audit leaks into fitting | Deferred evaluation after freezing candidate; callback-order test. |
| Refit expands into invalid support | Independent full-width Gaussian proposal check, no silent truncation. |
| Thin wrapper inherits full runtime debt | New lazy path is TF-only; legacy comparison remains test-only. |
| Stochastic corner tests are flaky | Fixed seeds and wide analytic pass/rejection margins; report actual metrics. |
| Existing user's changes overwritten | Stage edits, compare exact original touched files before first apply. |

Self-review verdict: executable for this bounded opt-in implementation and
synthetic verification scope after platform approval. This is a local review,
not an independent external/model review. No subagents or external code upload.

## 7. Complete command and approval manifest

Staging root: `/tmp/bayesfilter-curvature-refinement-20260908`.
Execution working directory: `/home/ubuntu/workspace/BayesFilter`.
Narrow requested prefix: `[/bin/bash,
/tmp/bayesfilter-curvature-refinement-20260908/bundle.sh]`.
Approval outcome will be recorded in the result note after the platform gate.

| Exact command under that prefix | Duration/resources | Outputs/mutation |
| --- | --- | --- |
| `apply` | seconds, CPU | apply_patch only to the listed implementation/test/docs files; preserve unrelated work |
| `unit` | <=10 min, CPU <=2 TF threads | new unit tests; fresh stdout/JUnit log under staging root |
| `regression` | <=10 min, same | new and selected legacy regressions; logs/JUnit |
| `integration` | <=10 min, same | standalone synthetic report/log under staging root |
| `audit` | <=2 min, same | syntax/lint, scoped import closure, whitespace and documentation checks |

The initial test allocation was three runs per command within 30 minutes total
active test time. The continuation amendment below requests a new bounded round.
Each command produces a fresh unique artifact; do not overwrite prior runs.
The environment is fixed: `CUDA_VISIBLE_DEVICES=-1`,
`TF_FORCE_GPU_ALLOW_GROWTH=true`, `BAYESFILTER_TEST_DEVICE_SCOPE=cpu`,
`BAYESFILTER_PRELOAD_CUSTOM_OP=0`, `TF_NUM_INTRAOP_THREADS=2`,
`TF_NUM_INTEROP_THREADS=1`, `OMP_NUM_THREADS=2`, `OPENBLAS_NUM_THREADS=1`,
`MKL_NUM_THREADS=1`, `RAYON_NUM_THREADS=1`, `TF_CPP_MIN_LOG_LEVEL=2`,
`PYTHONDONTWRITEBYTECODE=1`, `PYTHONUNBUFFERED=1`, and
`PYTHONPATH=/home/ubuntu/workspace/BayesFilter`.
Interpreter: `/home/ubuntu/miniforge3/envs/tf-gpu/bin/python`.
No network, packages, GPU queries, detached work, git mutation, or training.

Allowed write set: the two new inference modules, one legacy dense-kernel call
site, inference lazy exports, new unit/regression test files, the standalone
integration executable, this plan, a same-prefix result note, new
`docs/reference/posterior-local-geometry.md`, and one link in the existing
HMC interface guide. Final selected reports can be published under
`docs/plans/artifacts/posterior-curvature-refinement-20260908/` by `apply`.
Read-only inspection, MathDevMCP/CAS lookup, and staging edits are part of this
manifest and do not mutate the authority checkout.

## 8. Continuation review and prospective verification amendment

The first implementation was not complete despite an early 9-unit/35-regression
pass and five-case integration smoke. Those runs tested an earlier source and
are not final evidence. A second self-review identified and repaired:

- success-only row accounting (failed/ineligible attempts were disappearing);
- comparisons that allowed NaN diagnostics/factors to evade rejection;
- audit evaluation before the candidate was frozen;
- lazy API resolution loading legacy HMC before reaching the new module;
- comparing replicates only to the first, not all pairs;
- naming pilot-chart K as though it were raw position precision;
- reporting centeredness in pilot units rather than the unshrunk refined units;
- permissive JSON NaN/Infinity and fractional integer coercions;
- integration fixtures with a missing mixture value term and no true sentinel;
- smoke assertions that checked acceptance but not exact covariance recovery.

New tests exercise these bugs directly. Strict JSON converts nonfinite diagnostic
numbers to null while preserving an explicit rejection status. Invalid caller
shape/type and callback exceptions propagate. No covariance/factor is returned
on any rejection. Refined covariance is constructed by triangular solves:
`A = solve(chol(K), F.T); Sigma = A.T @ A`, avoiding explicit inversion.

Integration now uses one prospectively fixed configuration (the public default
0.20 selection/audit and 0.35 proposal RMSE caps) for all cases. Earlier driver
threshold expressions were inconsistent and one branch accidentally used 0.20
for quartic proposal validation. This is a fixture/configuration correction
before the complete integration run, not evidence that a rejected scientific
result passed. Exact-Gaussian covariance error must be <=2e-7 even at six-order
conditioning. Fixture value/score consistency is independently checked with
autodiff (<=1e-12 relative), so an inconsistent synthetic target cannot falsely
validate the implementation. Two additional paired eager/XLA cases use exactly
the same batch shape, target, seeds, and configuration; trace count must be one.

Graph boundary: this opt-in one-shot initializer deliberately uses bounded eager
orchestration and a small number of eager dense fits, not a long-running hot
loop. This is the documented exception for the milestone; the reusable dense
kernel also has graph/eager parity tests, and callers should supply a cached
fixed-shape compiled target for expensive evaluations. Full-refiner XLA and
GPU scalability remain unqualified, not silently replaced with NumPy.

Additional verification allocation requested before execution: at most four
additional `unit`, `regression`, `integration`, and `audit` commands each, same
exact bundle prefix, working directory, output root, environment, write set,
CPU/thread class, and 600-second per-command timeout. **The total active test
budget remains 30 minutes including earlier attempts.** At the amendment point
unit had run twice, regression three times, integration twice, and audit twice.
The audit adds AST syntax, recursive first-party import-closure checks, explicit
MacroFinance-local runtime import checks, source line-ending checks, and scoped
Ruff correctness lint (`--select E9,F63,F7,F82 --no-cache`), without a formatter
or dependency install. Approval is requested once for this complete round and
scoped staged publication; local implementation fixes are permitted inside it.

Self-review verdict: executable within this amended synthetic scope after the
platform gate. Review is local, not independent. Acceptance requires final
source hashes, all declared cases/tests passing, and durable result artifacts.
No MacroFinance plan promotion, target execution, guide-default update, sampler,
whitening, GPU, or full-chain XLA claim is authorized by this milestone.
