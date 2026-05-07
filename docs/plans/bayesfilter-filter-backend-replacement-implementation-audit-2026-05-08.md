# Independent Audit: BayesFilter Consolidation-First TF/TFP Plan

Date: 2026-05-08

Auditor stance: this audit treats
`docs/plans/bayesfilter-filter-backend-replacement-implementation-plan-2026-05-08.md`
as if another developer wrote it.  The objective is to check completeness,
phase order, missing gates, and whether the plan can be executed safely inside
the BayesFilter workspace without silently changing MacroFinance or DSGE
likelihood targets.

## Audit Verdict

The revised direction is correct: BayesFilter should consolidate the two donor
codebases before implementing more filtering code.

- MacroFinance should be the donor for linear Gaussian TensorFlow/TFP value,
  masked, SVD, QR/square-root, analytic score/Hessian, HMC, and performance
  gates.
- `/home/chakwong/python` should be the donor for UKF/cubature quadrature,
  SVD sigma-point primitives, generic nonlinear SSM filtering, DSGE structural
  metadata gates, XLA diagnostics, and CUT4-G experimental material.
- BayesFilter production implementation should be TensorFlow/TFP only.
  Existing NumPy modules are legacy/reference until TF replacements are
  available and public exports can be retired.

The plan is implementable, but not all 12 phases are justified in one automatic
BayesFilter-only run.  The safe initial execution boundary is:

1. Phase 0: produce a consolidation map and run baseline/TF dependency gates.
2. Phase 1: add TF result, diagnostics, and tensor contracts.
3. Phase 2: port dense and masked TF linear value backends.

Stop before Phase 3 unless Phase 2 has passed and the derivative donor choice
is made explicit, because MacroFinance has several derivative variants
(`tf_differentiated`, `tf_solve_differentiated`, `tf_qr_sqrt_differentiated`)
with different trace, factor, and graph properties.

## Critical Findings

1. The current BayesFilter public surface still imports NumPy implementation
   code:
   - `bayesfilter/filters/kalman.py`;
   - `bayesfilter/filters/sigma_points.py`;
   - `bayesfilter/filters/particles.py`;
   - `bayesfilter/linear/types.py`;
   - `bayesfilter/linear/kalman_derivatives_numpy.py`;
   - `bayesfilter/results.py`.

2. Removing those modules immediately would break current tests and public
   imports.  They should be isolated as legacy while TF front doors are added.

3. MacroFinance TF donor files still contain some NumPy usage for constants and
   eager trace conversion.  The port must replace constants with `math` or TF
   constants and keep `.numpy()` conversion outside production filtering paths.

4. DSGE SVD donor files use a custom `robust_eigh` op.  BayesFilter must define
   an eigensolver interface with default `tf.linalg.eigh`; the custom op can be
   an optional plugin but cannot be imported unconditionally.

5. Static-shape masking is a core contract, not an implementation detail.  It
   must be defined early so TF/XLA and HMC paths do not diverge from masked
   NumPy row-selection semantics.

6. The plan needs a concrete consolidation artifact.  Without it, overlapping
   donors could lead to duplicate backends with inconsistent labels and
   diagnostics.

## Missing Gates To Add

Add these gates before client switch-over:

```text
consolidation_map_committed
tf_tfp_import_cpu_only
tf_modules_no_numpy_imports
tf_modules_no_dot_numpy_calls
tf_result_containers_preserve_tensors
tf_diagnostics_schema_has_regularization_target
dense_tf_linear_one_step_tfp_identity
dense_tf_linear_multistep_closed_form_identity
masked_tf_linear_all_true_equals_dense
masked_tf_linear_all_missing_zero_likelihood
masked_tf_linear_static_shape_graph_reuse
legacy_numpy_exports_not_extended
macrofinance_donor_tests_mapped
dsge_donor_tests_mapped
robust_eigh_optional_plugin_boundary
```

## Phase-Order Audit

### Phase 0

Necessary and first.  It should create a durable consolidation map, not only a
memo paragraph.  It should also run:
- BayesFilter baseline tests;
- CPU-only TensorFlow/TensorFlow Probability import probe;
- source inventory for existing NumPy implementation surfaces;
- donor repo cleanliness checks.

### Phase 1

Justified after Phase 0.  Add TF-specific result and type modules rather than
rewriting existing NumPy modules in place.  This keeps legacy tests passing and
creates a clean production path for TF code.

### Phase 2

Justified after Phase 1.  Port MacroFinance dense and masked TF value paths.
Do not port SVD or derivatives into Phase 2; that would blur the first value
gate.

### Phase 3

Conditionally justified after Phase 2.  Before coding, choose the derivative
route:
- dense covariance-form for continuity with MacroFinance HMC;
- solve-form for clearer derivative diagnostics;
- QR/square-root only after factor-derivative tests are in BayesFilter.

### Phase 4

Conditionally justified after dense and masked TF value pass.  SVD value should
use a default TF eig/SVD primitive and record the implemented/floored
covariance.  Derivatives near branch events must be blocked or labeled.

### Phase 5 and Phase 6

Conditionally justified after Phase 1 contracts and Phase 0 consolidation map.
Do not start by copying DSGE full-state filters.  Start with generic structural
protocols and generic nonlinear SSM tests.

### Phase 7 and Phase 8

High risk.  CUT4-G value and moments are justified before analytic
gradient/Hessian.  Full SVD-CUT Hessians need smaller subphases and explicit
branch/floor labels.

### Phase 9

Benchmark phase only.  GPU/XLA probes must use escalated permissions under the
local policy.  CPU-only tests should set `CUDA_VISIBLE_DEVICES=-1`.

### Phase 10 and Phase 11

Not executable in this BayesFilter-only run.  They require explicit cross-repo
edit permission and separate commits in MacroFinance and `/home/chakwong/python`.

### Phase 12

Final cleanup only after TF replacements cover the public surface.  Do not
delete NumPy modules during the initial port.

## Recommended Automatic Boundary

Proceed automatically through:
- Phase 0;
- Phase 1;
- Phase 2.

Stop and ask for direction before Phase 3 if:
- TensorFlow or TFP imports fail;
- baseline tests fail for unrelated reasons;
- the consolidation map reveals an unresolved donor conflict;
- Phase 2 cannot preserve static mask semantics;
- source checks show TF production modules importing NumPy or calling `.numpy()`.

## Conclusion

The plan is sound after the consolidation-first revision.  The safest useful
execution is to establish the consolidation artifact and port the first TF
value backend family.  That gives BayesFilter a clean TF/TFP spine without
pretending the higher-risk derivative, nonlinear, SVD-CUT, GPU, and client
switch-over phases are already settled.
