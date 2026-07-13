# Phase 1 Result: Lower-Triangular LGSSM Model Contract

Date: 2026-07-08

## Decision

`PASS_PHASE1_MODEL_CONTRACT_LOWER_TRIANGULAR_V1`

Phase 1 defines a docs-only model contract for the first serious
multidimensional LGSSM NeuTra-HMC estimation target. No data were generated,
no algorithmic code was edited, and no runtime/model execution, NeuTra
training, or HMC sampling was run.

Machine-readable companion:

`docs/plans/artifacts/multidim-triangular-lgssm-neutra-hmc-2026-07-08/lower_triangular_lgssm_contract_v1.json`

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Does the model contract make stationarity and coordinate identification explicit enough for implementation? |
| Baseline/comparator | Phase 0 `lower_triangular_first` result. |
| Primary criterion | Defines lower-triangular `A`, `H=I`, diagonal `Q/R`, stationary `P_inf`, parameter names, transforms, priors, seeds, and nonclaims. |
| Veto diagnostics | Missing `P_inf`, free `H`, dense unconstrained latent similarity, unordered coordinates, ambiguous transforms, or unsupported identifiability claim. |
| Result | Pass for docs/model-contract work; no implementation/runtime gate is passed. |

## Contract Summary

| Field | Value |
| --- | --- |
| Contract id | `lower_triangular_lgssm_v1` |
| Target id | `bayesfilter_multidim_lower_triangular_lgssm_neutra_hmc_v1` |
| State dimension | `4` |
| Observation dimension | `4` |
| Innovation dimension | `4` |
| Horizon for first serious fixture | `256` |
| Observation matrix | `H = I_4` |
| Observation offset | fixed zero |
| Stationary mean | fixed zero |
| Transition matrix | lower triangular |
| Process covariance | diagonal positive |
| Observation covariance | diagonal positive |
| Initial state law | stationary Gaussian `N(0, P_inf)` |
| Stationary covariance | `P_inf = A P_inf A' + Q` |
| Target coordinate convention | unconstrained raw parameter vector |
| Log-Jacobian convention | `included_in_prior` |
| Filter semantics | exact linear Gaussian Kalman likelihood |

## Raw Parameter Order

The raw unconstrained vector has dimension `18`.

| Index | Name | Transform | Constrained Quantity |
| --- | --- | --- | --- |
| 0 | `a11_raw` | `a11 = rho_max * tanh(a11_raw)` | `A[0,0]` |
| 1 | `a22_raw` | `a22 = rho_max * tanh(a22_raw)` | `A[1,1]` |
| 2 | `a33_raw` | `a33 = rho_max * tanh(a33_raw)` | `A[2,2]` |
| 3 | `a44_raw` | `a44 = rho_max * tanh(a44_raw)` | `A[3,3]` |
| 4 | `a21_raw` | `a21 = lower_scale * tanh(a21_raw)` | `A[1,0]` |
| 5 | `a31_raw` | `a31 = lower_scale * tanh(a31_raw)` | `A[2,0]` |
| 6 | `a32_raw` | `a32 = lower_scale * tanh(a32_raw)` | `A[2,1]` |
| 7 | `a41_raw` | `a41 = lower_scale * tanh(a41_raw)` | `A[3,0]` |
| 8 | `a42_raw` | `a42 = lower_scale * tanh(a42_raw)` | `A[3,1]` |
| 9 | `a43_raw` | `a43 = lower_scale * tanh(a43_raw)` | `A[3,2]` |
| 10 | `log_q1` | `q1 = exp(log_q1)` | process std. dev. 1 |
| 11 | `log_q2` | `q2 = exp(log_q2)` | process std. dev. 2 |
| 12 | `log_q3` | `q3 = exp(log_q3)` | process std. dev. 3 |
| 13 | `log_q4` | `q4 = exp(log_q4)` | process std. dev. 4 |
| 14 | `log_r1` | `r1 = exp(log_r1)` | observation std. dev. 1 |
| 15 | `log_r2` | `r2 = exp(log_r2)` | observation std. dev. 2 |
| 16 | `log_r3` | `r3 = exp(log_r3)` | observation std. dev. 3 |
| 17 | `log_r4` | `r4 = exp(log_r4)` | observation std. dev. 4 |

Constants:

- `rho_max = 0.85`
- `lower_scale = 0.35`

This transform guarantees that every diagonal entry of lower-triangular `A`
lies inside `(-0.85, 0.85)`. Because the eigenvalues of a triangular matrix
are its diagonal entries, this enforces stationarity of `A` by construction.

## Matrix Construction

```text
A =
[[a11, 0,   0,   0  ],
 [a21, a22, 0,   0  ],
 [a31, a32, a33, 0  ],
 [a41, a42, a43, a44]]

Q = diag(q1^2, q2^2, q3^2, q4^2)
R = diag(r1^2, r2^2, r3^2, r4^2)
H = I_4
```

The stationary initial covariance is not an optional tuning parameter:

```text
vec(P_inf) = solve(I_16 - kron(A, A), vec(Q))
x_0 ~ N(0, P_inf)
```

Phase 3 must implement or reuse this solve with residual checks:

```text
max_abs(P_inf - A P_inf A' - Q) <= 1e-10
```

## Planned Truth Template

Phase 2 should instantiate this first synthetic fixture unless a pre-run
recoverability audit finds a material issue:

```text
diag(A) = [0.62, 0.48, 0.30, 0.16]
lower(A) =
  a21 =  0.18
  a31 = -0.10
  a32 =  0.14
  a41 =  0.06
  a42 = -0.08
  a43 =  0.11
q = [0.30, 0.26, 0.22, 0.18]
r = [0.12, 0.11, 0.10, 0.09]
```

Rationale:

- diagonal dynamics are distinct and away from zero and the stationarity
  boundary;
- lower entries are nonzero but bounded;
- process noise is larger than observation noise for the first rung to reduce
  the classic process/measurement ridge;
- `H=I` and fixed coordinate order avoid latent permutation/sign relabeling in
  the benchmark definition.

The equivalent raw truth values are determined by inverse transforms:

```text
a_diag_raw_i = atanh(a_ii / rho_max)
a_lower_raw_ij = atanh(a_ij / lower_scale)
log_q_i = log(q_i)
log_r_i = log(r_i)
```

## Prior Family

The first prior is independent Gaussian on raw coordinates:

```text
theta_raw ~ N(mu_raw, diag(prior_scale^2))
```

For Phase 2/3 the prior center should equal the raw truth template, and the
prior scales should be:

```text
diag entries:     0.50
lower entries:    0.60
log process std:  0.35
log obs std:      0.35
```

This is an intentionally friendly synthetic-recovery prior for the first
serious benchmark. It must not be described as objective, weakly informative,
general-purpose, or production-ready.

## Stable Target Signature Inputs

The Phase 2 implementation should compute a stable contract signature from
these fields:

- contract id and schema version;
- parameter names and order;
- static shape;
- transform ids/constants;
- model manifest including `lower_triangular_A`, `H_identity`,
  `diagonal_Q`, `diagonal_R`, and `stationary_initial_covariance`;
- data signature including horizon, seed, observation shape, and data hash;
- prior manifest including raw-coordinate center/scale and hash;
- filter manifest specifying exact TensorFlow Kalman likelihood and score
  authority after implementation.

No process-local object identity, memory address, wall-clock path, or runtime
environment field may enter the signature.

## Boundary Classification

| Topic | Classification |
| --- | --- |
| Stationarity | `correct` if Phase 3 implements the lower-triangular diagonal transform and Lyapunov residual check. |
| Coordinate anchoring | `correct as benchmark design`: `H=I`, fixed order, and diagonal `Q/R` remove obvious latent similarity freedoms in the named fixture. |
| Global identifiability | `not checked`; not claimed. |
| Synthetic recoverability | `not checked`; Phase 2/5 must test. |
| Score correctness | `not checked`; Phase 4 must test. |
| NeuTra/HMC readiness | `not checked`; later phases only. |

## Required Next Phase Handoff

Phase 2 may begin only if it uses this contract without changing:

- dimension `4`;
- raw parameter order;
- `rho_max = 0.85`;
- `lower_scale = 0.35`;
- `H=I`;
- diagonal `Q/R`;
- stationary `P_inf`;
- prior family and nonclaims.

Any change to those fields requires a visible contract amendment before data
generation.

## Local Checks

- Phase 1 subplan was repaired after Codex substitute review requested an
  explicit no-code/no-runtime boundary.
- `git diff --check` passed for the Phase 1 subplan and execution ledger before
  writing this result.
- No code implementation, runtime command, data generation, NeuTra training, or
  HMC command was run in Phase 1.

## Decision Table

| Decision | Primary Criterion Status | Veto Diagnostic Status | Main Uncertainty | Next Justified Action | Not Concluded |
| --- | --- | --- | --- | --- | --- |
| Admit `lower_triangular_lgssm_v1` as docs/model contract | Met | No Phase 1 veto triggered | Finite-sample recoverability remains empirical | Generate/validate Phase 2 synthetic fixture | No implementation correctness, XLA readiness, NeuTra usefulness, HMC convergence, global identifiability, or product/default readiness |
