# Independent Audit: TF/TFP-Only BayesFilter Filter Plan

Date: 2026-05-08

Auditor stance: treat the revised implementation plan as if another developer
wrote it.  The audit checks whether the new TF/TFP-only rule is enforceable and
whether the prior NumPy foundation can accidentally remain the production path.

## Audit Verdict

The user clarification changes the architecture.  The earlier NumPy-first
foundation is no longer an acceptable implementation target.  It may help as
historical reference, but BayesFilter production filtering should be built from
TensorFlow and TensorFlow Probability modules.

I agree with the pivot.  It is especially important for HMC and XLA workflows:
NumPy implementations break graph compilation, force host/device transfers,
hide `.numpy()` conversions in gradients, and do not match the intended
TensorFlow/TFP execution model of MacroFinance and DSGE clients.

## Main Findings

1. The current committed code still imports NumPy in production modules:
   `bayesfilter/filters/kalman.py`, `bayesfilter/linear/types.py`,
   `bayesfilter/linear/kalman_derivatives_numpy.py`,
   `bayesfilter/filters/sigma_points.py`, `bayesfilter/filters/particles.py`,
   `bayesfilter/results.py`, and `bayesfilter/backends.py`.

2. The public package currently re-exports
   `solve_kalman_score_hessian` from the NumPy derivative module.  That is a
   production-surface problem and should be retired once a TF replacement
   exists.

3. The old plan used NumPy as an oracle for TF parity.  Under the new policy,
   BayesFilter correctness should be checked by:
   - closed-form scalar/multivariate Gaussian identities;
   - TF-only finite differences;
   - TFP distribution log-probability checks;
   - MacroFinance TF provider parity;
   - structural moment identities for quadrature rules.

4. Tests may continue to import NumPy for simple assertions during migration,
   but tests should not make NumPy the mathematical oracle for production
   filtering.

5. The first coding phase should not start with nonlinear SVD-CUT.  It should
   first build the TF result, diagnostics, and tensor contracts that every later
   backend will share.

## Required Changes To The Plan

- Replace all NumPy implementation phases with TF/TFP phases.
- Add a source-check gate: production TF modules must not import NumPy or call
  `.numpy()`.
- Treat existing NumPy files as legacy, not as the implementation core.
- Add TF/TFP dependency and version probing before coding.
- Use CPU-only TF probes by default with `CUDA_VISIBLE_DEVICES=-1`; GPU probes
  require escalated permissions under local policy.
- Make TFP part of the value/log-probability contract where it is clearer than
  handwritten Gaussian density code.
- Keep XLA static-shape constraints visible in the public API from the start.
- Delay removal of old public NumPy exports until TF replacements exist, but do
  not extend those exports.

## Revised Phase Boundary

The next justified automatic boundary is now:

1. Update plan and reset memo to record the TF/TFP-only decision.
2. Inventory NumPy implementation surfaces and classify migration status.
3. Add TF result/diagnostics/tensor contract modules.
4. Add dense TF linear Kalman value backend and tests.

Stop before:
- deleting old NumPy modules without TF replacements;
- editing `/home/chakwong/python` or `/home/chakwong/MacroFinance`;
- claiming SVD derivative or HMC readiness before TF branch diagnostics pass;
- running GPU tests without escalation.

## Acceptance Gates To Add

```text
tf_import_and_tfp_import_cpu_only
production_modules_no_numpy_imports
production_modules_no_dot_numpy_calls
tf_result_metadata_schema_consistent
linear_tf_value_closed_form_identity
linear_tf_value_tfp_log_prob_identity
linear_tf_mask_static_shape
linear_tf_score_hessian_tf_finite_difference
legacy_numpy_exports_removed_after_tf_replacement
```

## Conclusion

Yes, the current implementation uses NumPy.  That should be treated as a
mistake in direction, not as the foundation for the rest of BayesFilter.  The
plan should now proceed TF/TFP-first, with NumPy code isolated as legacy until
it can be removed or replaced safely.
