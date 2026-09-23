# Follow-up SVD caller review

This is a partial source review for E6, not a terminal execution verdict.
The verified 28-call inventory is preserved in
`svd-scale-followup-inventory-03549.json` under the master campaign artifact root.
The four reproduced functions already have focused CPU/GPU qualification;
additional tensor kernels and host numerical callers remain distinct work.

The dense-score singular-value helper, COD condition estimator and sequential
score fitter specify binary64 XLA convergence but do not normalize magnitude
before SVD. The block-score fitter uses the TensorFlow default SVD for both
symmetric design rank and offset rank. These sites therefore need actual
scale/tied-mode diagnostics; the corrected rectangular SRUKF does not qualify
them. Any repair must preserve rank cutoffs, caller statuses and gradient
contracts, with independent residual/rank checks and complete endpoint renewal.
Replacing all calls merely because they use SVD would not establish correctness.

The host `_condition_number` and `_solve_scaled_augmented_ridge` helpers in
highdim/fitting.py feed legacy host fitters, while squared_tt_engine_xla_tf.py
has its own scaled CholeskyQR2/condition route. An import of FixedTTFitter is not
proof that the host SVD executes in the XLA path. The remaining caller audit must
trace invoked methods and test actual endpoints before classifying eligibility.
The same distinction applies to highdim/derivatives.py and source-route adapters.
Do not change Zhao-Cui behavior without its paper and author-source anchors.

independent_score/null_calibration_tf.py explicitly documents predictive
consistency diagnostics. A search of repository Python sources under
bayesfilter/, scripts/ and experiments/ found no direct external call to its
fit_svd_geometry. This is useful discovery evidence, not proof against dynamic
imports or external consumers. The whitening-bank qualifier and transport
reliability screen similarly have host numerical boundaries whose actual
admission callers still require examination. The antithetic directional frame
validator is used by two public local-geometry routines and retains NumPy;
its public call chain cannot be certified by repairing SVD alone.

Next bounded unit: probe the remaining tensor SVD operations at separated and
nearly repeated singular values with multiple magnitudes, then reproduce each
failure through its caller before changing runtime. Trace host consumers to
classify genuine diagnostics versus active numerical/admission work. Preserve
these obligations in F19/E6; this review adds no allowlist entry or exception.
