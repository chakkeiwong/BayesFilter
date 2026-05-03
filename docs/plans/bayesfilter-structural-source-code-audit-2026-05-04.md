# Audit: structural state partition source and code reconciliation

## Date

2026-05-04

## Scope

This note covers Phases 0--2 of
`docs/plans/bayesfilter-structural-state-partition-core-plan-2026-05-04.md`.
It records repository inventory, mathematical source reconciliation, and code
reuse decisions before BayesFilter backend code is added.

## Phase 0 inventory

BayesFilter status at the start of this pass:

- Branch: `main`, ahead of `origin/main`.
- Existing dirty BayesFilter files included the canonical structural plan.
- During the session, related uncommitted linear-Gaussian spine documentation
  was also visible and preserved.
- No `bayesfilter/` package or `tests/` tree existed before this phase.

Client status:

- `/home/chakwong/python` had untracked local tool/planning files and a DSGE
  structural partition plan.  Candidate code paths include
  `src/dsge_hmc/filters/__init__.py`,
  `src/dsge_hmc/filters/_svd_filters.py`,
  `src/dsge_hmc/filters/_svd_core.py`,
  `src/dsge_hmc/filters/_quadrature.py`, DSGE model adapters under
  `src/dsge_hmc/models`, and tests under `tests/numerics`,
  `tests/contracts`, and `tests/extended`.
- `/home/chakwong/MacroFinance` had no dirty status output in the initial
  scan.  Candidate code paths include `domain/types.py`, `filters/kalman.py`,
  differentiated Kalman modules, square-root/QR/SVD modules, TensorFlow
  modules, and backend parity tests under `tests/`.

Primary derivation sources located:

- DSGE: `/home/chakwong/python/docs/monograph.tex`, especially
  `chapters/ch08_kalman_filter.tex`, `chapters/ch09_sr_ukf.tex`,
  `chapters/ch09b_svd_filters.tex`, and `chapters/ch06_perturbation.tex`.
- MacroFinance: `/home/chakwong/latex/CIP_monograph/main.tex`, especially
  `chapters/ch11_state_space_recursions.tex`,
  `chapters/ch16_kalman_filter.tex`,
  `chapters/ch17_nonlinear_filtering.tex`,
  `chapters/ch26_differentiable_pf.tex`, and
  `chapters/ch33_analytical_validation.tex`.
- MacroFinance code/derivative note:
  `/home/chakwong/MacroFinance/analytic_kalman_derivatives.tex` and
  `/home/chakwong/MacroFinance/filters/*.py`.

## Tool-audit limitations

ResearchAssistant local searches for generic Kalman and sigma-point papers did
not return reviewed paper summaries in the local workspace.  A summary-level
claim audit returned `insufficient_evidence` because no reviewed local paper
IDs were available for that query.

MathDevMCP `search_latex`, `extract_latex_context`, and
`typed_obligation_label` calls against the large monograph roots errored during
this pass.  `doctor` reported SymPy/Sage/LaTeXML availability, but Lean version
checking timed out.  Simple matrix obligations using `.T` notation were
inconclusive because the SymPy encoding treated the symbols as scalars.

Interpretation: local monograph labels and code reads are usable provenance for
this implementation pass, but the MCP results are not proof certificates.
Claims below are therefore intentionally conservative.

## Mathematical reconciliation

BayesFilter notation:

```text
x_t = pack(s_t, d_t, a_t)
s_t = stochastic block
d_t = deterministic-completion block
a_t = auxiliary block
eps_t = innovation
```

Exact linear Gaussian path:

```text
x_t = c + F x_{t-1} + G eps_t
Q = G Sigma_eps G'
y_t = a + H x_t + e_t
```

The DSGE monograph labels `eq:kf_xp`--`eq:kf_ll` and the CIP monograph labels
`eq:kf_predict_mean`--`eq:KF_loglik` support the common Kalman prediction,
update, and prediction-error likelihood formulas.  These equations are exact
for collapsed linear Gaussian state-space models, including rank-deficient
`Q`, provided the selected innovation covariance is positive definite.

Structural nonlinear path:

```text
s_t = T_s(x_{t-1}, eps_t; theta)
d_t = T_d(x_{t-1}, s_t; theta)
x_t = pack(s_t, d_t, a_t)
```

The DSGE pruned perturbation source supports a split between first-order
stochastic propagation and deterministic higher-order correction.  It does not
by itself establish exogenous/endogenous state partitions inside every DSGE
first-order state vector.  Therefore BayesFilter may implement a generic
structural contract and toy fixtures now, but DSGE Rotemberg/SGU/EZ adapters
remain blocked until model-specific structural maps are supplied and tested.

Particle-filter path:

The CIP monograph has particle-filter and differentiable-particle-filter
chapters, but this pass did not complete a source-backed proposal/resampling
audit.  Particle filtering must fail closed in BayesFilter until a separate
proposal, correction, and approximation-label audit is written.

## Backend classification

| Backend/path | Exact for LGSSM? | Exact for mixed nonlinear structural models? | Approximation? | Reuse candidate? | Notes |
| --- | --- | --- | --- | --- | --- |
| Covariance Kalman | Yes | No | No for LGSSM | Reimplement small reference | Simple BayesFilter-local reference is justified; MacroFinance versions are richer but entangled with derivative/backend ladders. |
| MacroFinance differentiated Kalman | Yes for LGSSM when tests pass | No | No for LGSSM | Reuse later with audit | Candidate for later extraction/wrapping after derivative obligations are checked. |
| DSGE `KalmanFilter` | Yes for first-order collapsed DSGE LGSSM | No | No for declared linear path | Do not copy now | Tied to DSGE `model.solve`, `sol`, backend registry, and observation-noise conventions. |
| DSGE `SVDKalmanFilter` | Yes in controlled tests | No | No for linear path, spectral backend telemetry needed | Candidate later | Strong reference for singular/near-singular linear value path, but TensorFlow/MKL/XLA plumbing is client-specific. |
| DSGE `SVDSigmaPointFilter` generic SSM path | Exact in affine Gaussian recovery tests, approximate otherwise | Only if structural map is supplied | Yes | Reuse concepts, not code now | Existing generic path supports `transition_points`; default DSGE adapter lacks the structural exogenous/endogenous partition. |
| DSGE default SVD sigma DSGE adapter | Linear SmallNK acceptable; mixed nonlinear DSGE unresolved | No | Must be labeled for mixed models | Do not migrate as structural backend | It propagates the first-order state as a collapsed Gaussian and tracks pruned `x_s`, but does not expose structural timing inside `x_f`. |
| Particle filters | LGSSM recovery only as Monte Carlo/reference limit | Yes only with correct deterministic completion and proposal correction | Usually | Blocked | Requires separate audit of proposal, resampling, estimator variance, and differentiability. |

## Code reuse decisions

1. Implement a BayesFilter-local structural metadata contract now.
2. Implement a small covariance-form Kalman reference now, because the exact
   LGSSM equations are audited enough and the reference keeps tests independent
   of client repos.
3. Implement structural toy fixtures now: AR(2) lag shift and a nonlinear
   deterministic accumulation model.
4. Implement a small NumPy structural sigma-point backend now as an approximate
   reference.  It must return an approximation label and preserve deterministic
   completion pointwise.
5. Do not copy DSGE TensorFlow/MKL SVD code in this phase.
6. Do not copy MacroFinance differentiated Kalman code in this phase.
7. Do not implement particle filters beyond a fail-closed placeholder.

## Unresolved blockers

- DSGE Rotemberg, SGU, EZ, and NAWM adapters need explicit structural maps and
  model-specific deterministic completion tests before nonlinear BayesFilter
  routing is justified.
- MacroFinance derivative migration needs obligation-level audit of score,
  Hessian, QR/SVD factor derivative, and TensorFlow parity paths.
- SVD/eigen gradient promotion to HMC needs spectral-gap telemetry and
  derivative stress tests.
- Particle-filter HMC claims need a separate literature and code audit.

## Gate decision

Phases 3--7 are justified for BayesFilter-local documentation, contracts,
toy fixtures, exact LGSSM reference tests, and approximate structural
sigma-point tests.  Client adapter pilots and HMC readiness gates are not yet
release-ready; they may be audited and partially scaffolded only if they do not
claim structural correctness for unresolved client models.
