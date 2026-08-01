# GenUT Antithetic LGSSM And SV Experiment Plan

Date: 2026-07-22

Status: `ATTEMPT01_FD_CONDITIONING_REPAIRED_ATTEMPT02_AUTHORIZED`

## Research Intent Ledger

| Field | Frozen decision |
|---|---|
| Main question | At fixed GenUT controls and equal computation, does averaging complete particle-filter evaluations at innovation clouds `Z` and `-Z` reduce conditional particle value/score variance relative to averaging two independent innovation clouds? |
| Models | A three-state Gaussian LGSSM and the scalar exact transformed-SV model, both at `T=50` |
| Candidate | Complete-run antithetic average `(F(Z)+F(-Z))/2` using the same deterministic GenUT residual design in both runs |
| Equal-cost comparator | Complete-run independent average `(F(Z1)+F(Z2))/2` using the same deterministic GenUT residual design |
| Cost diagnostic only | Single-cloud `F(Z1)`; it is not the primary comparator because it uses half the filter evaluations |
| Expected failure mode | Repeated nonlinear weighting/reset operations may prevent sign reflection from cancelling the dominant score error, or may reduce variance while leaving/increasing finite-filter bias |
| Promotion criterion | Within each model and coordinate, the dataset-level familywise 95% CI for `log(Var_antithetic/Var_independent_pair)` has upper endpoint below zero. This nominates that coordinate only. |
| Promotion veto | Nonfinite output, GPU/XLA/TF32 failure, residual failure, recursive-score/representative-point finite-difference failure, invalid DGP/oracle, incomplete seed pairing, or claim-data leakage into tuning |
| Continuation veto | Invalid DGP/oracle, broken equal-cost coupling, corrupted artifact, exhausted two-attempt budget, or inability to execute the shared recursive-score XLA path |
| Repair trigger | Local harness, compilation, serialization, or resource failure with the scientific contract unchanged |
| Explanatory diagnostics | Single-cloud variance, estimator mean error, MSE ratio, per-dataset variance ratios, exact sign test, wall time, and TensorFlow allocator peak |
| Not concluded | No default change, unbiasedness, general nonlinear-model superiority, MLE/HMC improvement, NAWM result, or broad GenUT promotion |

The outer sampling unit is the independently generated model dataset. Particle
seeds are nested replications used to estimate conditional Monte Carlo
variance; they are not treated as independent scientific datasets.

## Exact Model Scopes

### LGSSM

The state and observation equations are

```text
x_t = diag(phi) x_(t-1) + q_scale eta_t
y_t = H x_t + r_scale epsilon_t,
```

with stationary `x_-1`, `phi=(0.72,0.55,0.35)`, `q_scale=0.35`,
`r_scale=0.45`, and the existing dense `3 x 3` observation matrix. The exact
Kalman value and analytic recursive score are the oracle. Gaussian GenUT has
`s=0`, `k=3`, zero center mass in dimension three, and therefore equals the
six-point cubature measure in this scope. This run tests innovation coupling,
not a GenUT-versus-Cubature difference.

### Exact transformed SV

The data are generated only from

```text
h_t = gamma h_(t-1) + eta_t
z_t = h_t + 2 log(beta) + log(epsilon_t^2),
```

with stationary `h_-1`, `gamma=Phi(0.25)`, `beta=exp(-0.15)`, and independent
standard-normal innovations. The refined one-dimensional dense integration
route supplies the value/score oracle. The revoked direct-Normal transformed
observation fixture is forbidden from this plan, runner, criteria, tables, and
decision.

## Fixed Execution And Replication Design

| Choice | LGSSM | SV | Status and rationale |
|---|---:|---:|---|
| Horizon | 50 | 50 | Target long-horizon score-variance issue |
| Particle count | 1008 | 1002 | Exactly represents the positive Gaussian GenUT weights |
| Claim DGP datasets | 8 | 8 | Outer replication; feasibility-scale evidence only |
| Particle-seed pairs per dataset | 16 | 16 | Conditional variance estimate and pairing |
| Unique evaluations per seed | `F(Z1)`, `F(Z2)`, `F(-Z1)` | same | Standard, independent-pair, and antithetic-pair estimators |
| Arithmetic | FP32 tensors, TF32 enabled | same | Requested production-target arithmetic |
| Backend | TensorFlow GPU/XLA | same | Repository default; no NumPy and no Python/sample loop inside XLA |
| Score | Complete recursive forward sensitivity | same | No autodiff or finite difference in claim score computation |

Each artifact records equations, parameter point, stationary initial law,
transition/observation seeds, observation tensor hash, particle seeds, GenUT
moments, controls, source hashes, GPU memory policy, and allocator peak.

## Offline Scope-Specific Tuning

Each model receives its own tuning artifact. Calibration, validation, and claim
DGP seeds are disjoint. Prior controls are warm starts only.

The grid is the full declared family:

```text
epsilon        in {2, 4}
sinkhorn_steps in {4, 8}
ridge          in {1e-6, 1e-5}
```

Tuning uses the standard GenUT estimator so the later coupling comparison has
one shared finite kernel. Four particle seeds are evaluated on two calibration
and two validation DGP datasets. Selection does not use an oracle score:

1. reject nonfinite, residual-invalid, or non-GPU/XLA candidates;
2. reject candidates whose recursive score disagrees with central finite
   differences of the same finite value program at three statelessly selected
   representative parameter points by more than `5%` relative error. The
   FP32 audit must pass independently at relative steps `0.004` and `0.008`
   (with absolute floors `4e-4` and `8e-4`);
3. minimize the maximum validation conditional variance across scaled value
   and physical-score coordinates;
4. break ties by calibration variance, finite-difference error, fewer Sinkhorn
   steps, larger ridge, then smaller epsilon.

The variance scaling constants are fixed before execution (`50` for value and
`10` for each score coordinate). They are convenience normalization choices,
not scientific defaults. The claim run reports oracle accuracy separately and
cannot retroactively change the selected controls.

## Statistical Analysis

For every dataset `d`, estimator `e`, and coordinate `j`, calculate the sample
variance across the 16 paired particle seeds. The primary per-dataset statistic
is

```text
L_dj = log((Var_antithetic,dj + floor) /
           (Var_independent_pair,dj + floor)).
```

Across eight DGP datasets, report the mean log ratio, geometric variance ratio,
an exact paired sign-test count, and a two-sided familywise 95% Student-t
interval. Multiplicity is controlled separately within each model over value
plus all physical-score coordinates (six LGSSM coordinates, three SV
coordinates). MSE ratios and signed mean-error differences use the same outer
dataset unit but are explanatory diagnostics, not silent promotion criteria.

## Skeptical Plan Audit

| Risk checked | Finding and repair |
|---|---|
| Wrong baseline | A single cloud is half-cost and was rejected as the primary comparator; the primary comparator averages two independent complete runs. |
| Invalid SV fixture | The revoked direct-Normal fixture is absent. Every SV dataset is generated from the declared latent transition and observation equations and hashed. |
| Proxy promoted | Residuals, finite differences, runtime, single-cloud SD, and inner particle seeds are veto/explanatory evidence only. The promotion statistic uses outer DGP replication. |
| Hidden dataset reuse | Calibration, validation, and claim DGP seed sets are disjoint and emitted in the manifest. |
| Unfair tuning | Controls are selected once per model on standard GenUT and frozen across both equal-cost coupling modes. Separate tuning would confound the coupling comparison. |
| Score oracle leakage | Oracle scores are used only after controls freeze. Tuning checks recursive score against finite differences of the same finite value at representative points. |
| Environment mismatch | Both claims use the same GPU, FP32/TF32, XLA, memory-growth policy, particle count, controls, and recursive-score core within model. |
| Gaussian LGSSM overclaim | The plan states that dimension-three Gaussian GenUT equals cubature; no distinct GenUT advantage is claimed there. |
| Bias hidden by lower variance | Dataset-level mean error and MSE are reported beside variance. Lower variance alone cannot establish a default. |
| Underpowered default decision | Eight datasets are enough for a bounded feasibility test, not broad default promotion. Default status remains unchanged regardless of outcome. |
| Artifact cannot answer question | Raw constituent estimates and per-dataset summaries are preserved, so equal cost, pairing, variance ratios, bias, and MSE can be recomputed. |

Audit verdict: `PASS_AFTER_EQUAL_COST_AND_MULTI_DGP_REPAIR`. The commands below
answer the stated feasibility question without authorizing a default change.

### Attempt 1 Repair Note

Attempt 1 stopped during SV tuning before reading any claim DGP. GPU/XLA,
finiteness, reset residuals, marginal residuals, and recursive increment sums
passed, but the original relative finite-difference step `0.002` produced a
`9.28%` gamma error at the positive representative point. A diagnostic ladder
on the same calibration row gave gamma errors `18.66%, 9.28%, 1.64%, 0.88%,
0.29%` at relative steps `0.001, 0.002, 0.004, 0.008, 0.016`; the beta error
was at most `0.70%`. The contraction as the FP32 subtraction interval grows
identifies cancellation in the approximately `-113` scalar value, not a
recursive-score defect. The repair fixes two predeclared audit steps (`0.004`,
`0.008`) and requires both to pass the unchanged `5%` gate. It does not alter
the DGP, candidate, controls, selection objective, claim seeds, comparison,
promotion criterion, or compute budget. Attempt 2 is the single permitted
localized retry.

## Budget, Stop Conditions, And Artifacts

- One tuning plus claim attempt and at most one localized repair/retry.
- Maximum 16 claim datasets total, 16 particle pairs per dataset, and three
  complete filter evaluations per pair.
- Expected wall time below 15 minutes on the visible RTX 4080 SUPER; stop if
  wall time reaches 30 minutes or allocator peak exceeds 14 GiB.
- Versioned output root:
  `docs/benchmarks/artifacts/genut_antithetic_lgssm_sv_20260722/attempt01/`.
- Result note:
  `docs/plans/bayesfilter-genut-antithetic-lgssm-sv-result-2026-07-22.md`.

Planned commands:

```bash
CUDA_VISIBLE_DEVICES=-1 pytest -q \
  tests/highdim/test_genut_antithetic_lgssm_sv.py \
  tests/highdim/test_cubature_genut_adapters.py

TF_FORCE_GPU_ALLOW_GROWTH=true python \
  docs/benchmarks/run_genut_antithetic_lgssm_sv.py \
  --output-root \
  docs/benchmarks/artifacts/genut_antithetic_lgssm_sv_20260722/attempt01
```
