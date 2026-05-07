# Independent Audit: BayesFilter Filter Backend Replacement Plan

Date: 2026-05-08

Auditor stance: treat
`docs/plans/bayesfilter-filter-backend-replacement-implementation-plan-2026-05-08.md`
as if another developer wrote it.  The audit checks whether the plan is
complete enough, whether phases are ordered safely, and whether any phase could
silently change the likelihood target used by MacroFinance or DSGE clients.

## Audit Verdict

The plan is directionally sound and phase order is mostly correct: linear value
and derivative infrastructure should be stabilized before TensorFlow, nonlinear
SVD, SVD-CUT, and client switch-over work.  The plan is also appropriately
conservative about structural mixed-state filtering and about GPU/XLA claims.

However, the plan needs four guardrails before broad execution:

1. Packaging and optional dependency policy must be solved before TensorFlow
   phases.
2. The first implementation pass should not attempt all 12 phases in one
   commit; it should stop at the first major dependency or cross-repo boundary.
3. Client switch-over phases require separate write grants for
   `/home/chakwong/python` and `/home/chakwong/MacroFinance`.
4. SVD-CUT analytic Hessians should be gated behind a smaller implemented
   derivative surface, because spectral factor Hessians can dominate the risk.

## Missing Or Under-Specified Points

### Packaging and dependency policy

BayesFilter currently has no visible `pyproject.toml`, `setup.py`, or
requirements file at the repo root.  TensorFlow, TensorFlow Probability,
optional MKL/custom ops, and GPU/XLA tests cannot be made production claims
without package metadata.

Required addition before TensorFlow phases:
- define optional extras, for example `bayesfilter[tf]`, `bayesfilter[dev]`,
  and possibly `bayesfilter[robust-eigh]`;
- keep NumPy-only imports working when TensorFlow is not installed;
- skip TensorFlow tests cleanly when the optional dependency is absent.

### Version and compatibility policy

The plan names compatibility re-exports but does not specify deprecation
duration or import stability.

Required addition:
- public imports remain stable through at least one migration window:
  `bayesfilter.filters.kalman`, `bayesfilter.filters.sigma_points`, and
  `bayesfilter.adapters.*`;
- new modules may be introduced under `bayesfilter.linear` and
  `bayesfilter.nonlinear`, with old modules wrapping them.

### Data masking convention

The plan mentions masked support but should pin down semantics:
- row selection in NumPy backends;
- static-shape dummy-row convention in TensorFlow backends;
- equality of likelihood contributions between conventions;
- metadata describing whether missing rows are omitted or dummy-normalized.

### Initial condition derivatives

Initial mean/covariance derivatives are included in the MacroFinance dataclass,
but the plan should explicitly test:
- fixed initial moments;
- stationary initial covariance;
- parameter-dependent initial mean;
- block-specific initial policies exposed in metadata.

### Time-varying systems

Client state-space systems may be time-invariant today, but BayesFilter should
avoid painting itself into a corner.  The linear contract should either:
- explicitly state Phase 1--4 are time-invariant only; or
- accept per-time arrays/callables in a later phase.

Recommendation: declare time-invariant first, then add time-varying LGSSM as a
separate phase after dense derivative parity.

### Observation dimensions and mixed frequency

The plan should distinguish:
- static observation dimension with masks;
- time-varying observed subsets;
- ragged observation blocks, which are not XLA-friendly unless represented with
  static masks.

### Numerical regularization contract

Every backend needs a common regularization metadata schema:
- jitter added to observation covariance;
- eigenvalue/singular-value floors;
- PSD projection residual;
- failed conditioning guard;
- implemented covariance `P_star`;
- whether derivative is of pre-regularized or implemented object.

The plan mentions these ideas but should make a shared diagnostics dataclass an
early deliverable.

### Randomness and reproducibility

Particle filters are not the target here, but nonlinear tests and benchmark
fixtures may simulate data.  Add a seeded fixture policy so future HMC/CUT
benchmarks are reproducible.

### Performance benchmarks

The GPU/XLA phase should require:
- compile time;
- first-call time;
- steady-state time;
- peak memory if available;
- point dimension `q`, state dimension, observation dimension, parameter count,
  chain count, and time length.

The original plan says this in spirit; the audit recommends a required JSON
schema before benchmark claims are accepted.

## Phase-Order Audit

### Phase 0

Good and necessary.  It should also record:
- Python version;
- NumPy version;
- TensorFlow availability without importing TensorFlow in NumPy-only tests if
  possible;
- current branch and dirty-tree exclusions.

### Phase 1

Good.  It is the right first coding phase.  Add result/diagnostic dataclasses
before moving algorithms, otherwise every backend will invent result metadata.

### Phase 2

Mostly good.  Splitting linear value backends into covariance, solve, masked,
and SVD is safe.  Do not port TensorFlow SVD here unless TensorFlow dependency
policy is already resolved; a NumPy SVD value backend can be done first.

### Phase 3

Good.  The solve-form analytic derivative backend is the highest-value port
from MacroFinance.  It should be implemented before TensorFlow because it gives
the clean oracle.

### Phase 4

Conditionally justified only after packaging/optional dependency policy is
implemented.  Otherwise BayesFilter may become unimportable in NumPy-only
environments.

### Phase 5

Good.  This should be partly parallel with Phase 1 only if write sets are
disjoint, but a single developer can simply sequence it.

### Phase 6

Conditionally justified after Phase 5 and TensorFlow dependency policy.  The
optional robust-eigh plugin boundary is important and should not import
`dsge_hmc` unconditionally.

### Phase 7

Good after Phase 6.  CUT4-G value support can be implemented in NumPy before
TensorFlow if desired, because the quadrature rule itself is dependency-light.

### Phase 8

High risk.  The plan should split this phase:
- 8A: fixed-rule moment derivatives with a fixed factor derivative supplied by
  a fixture;
- 8B: SVD/eigen factor first derivative;
- 8C: SVD/eigen factor second derivative;
- 8D: full score/Hessian integration and branch diagnostics.

This split prevents the spectral Hessian from obscuring simpler moment
derivative bugs.

### Phase 9

Good but benchmark-only.  It should not block NumPy implementation phases.

### Phase 10 and Phase 11

These are client-repo migration phases and are not fully executable from the
BayesFilter writable root alone.  They require:
- BayesFilter parity tests passing;
- explicit user permission to edit client repos;
- separate commits in each client repo;
- rollback-compatible wrappers.

### Phase 12

Good as final policy, not as an implementation phase for the first pass.

## Additional Acceptance Gates

Add these gates before client switch-over:

```text
import_numpy_only_without_tensorflow
result_metadata_schema_consistent_across_backends
regularization_metadata_reports_implemented_covariance
time_invariant_lgssm_declared_or_time_varying_supported
mask_convention_parity_numpy_vs_tf
initial_condition_derivative_policy_parity
macrofinance_one_country_value_score_hessian_parity
dsge_generic_ssm_value_gradient_parity
full_state_mixed_dsge_blocked_by_default
cut4_point_count_static_and_moment_exact
svd_cut_derivative_branch_labels
```

## Recommended Execution Boundaries

For the current BayesFilter-only execution, continue automatically through:
- Phase 0 baseline;
- Phase 1 core types/results if baseline passes;
- Phase 2 NumPy linear value hardening if Phase 1 passes;
- Phase 3 NumPy solve-form derivative backend if Phase 2 passes.

Stop and ask before:
- adding TensorFlow as a dependency or package extra;
- editing `/home/chakwong/python`;
- editing `/home/chakwong/MacroFinance`;
- implementing full SVD-CUT Hessians beyond a fixture-level derivative core;
- making production/HMC readiness claims.

## Conclusion

The plan is a good roadmap, but all 12 phases are not equally executable in one
BayesFilter-only run.  A justified first execution should establish the baseline
and implement the NumPy linear foundation with shared result/diagnostic
contracts.  That foundation is a prerequisite for every later nonlinear,
TensorFlow, SVD-CUT, and client-migration phase.
