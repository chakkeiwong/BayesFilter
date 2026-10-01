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

Read-only follow-up at9696666e4: `FixedTTFitter.fit` delegates to
`native_fixed_tt_fit` with JIT true by default (fitting.py:225--249).
The old `_fit_core_update` is reached by `fit_reference`; a public `.fit` call
in source_route.py therefore does not show that the old host SVD ran.
`NativeFixedTTFit` instead uses ops/qr_lstsq_tf.py::condition_number. The latter
is an active consumer in the next magnitude diagnostic. The squared-TT XLA
engine's `_build_design_matrix` calls likewise do not execute `.fit_reference`.

Literal calls to the old scaled ridge helper remain in squared_tt_engine_v0_tf.py
(explicit eager engineering reference) and zhao_cui_moment_teacher_als.py
(explicit fixed-ALS value/JVP reference). The latter also calls the host
fixed_design_lsq_derivative. Its other discovered caller is the historical
retained-grid filtering code. This source inspection narrows the remaining
call-chain question; it does not prove runtime isolation or source faithfulness.

The two callable qualification APIs require a separate disposition:
neutra_whitening.py::qualify_frozen_whitening_bank loops over chunks, points and
curvature steps, while tempered_transitions_tf.py::screen_transport_reliability
loops over charts and materializes numerical decisions on the host. Only tests
directly call either name in the searched repository roots. Absence of a literal
caller cannot certify an exported qualification API, and neither is included in
the five-consumer scale unit. Their host numerical defaults remain explicit E6
review debt. No behavior, method, classification waiver or tolerance changed.


Execution follow-up through03698: the scale diagnostic reproduces block-rank,
quadratic-precision and dense/COD condition errors at well-conditioned small
magnitudes. Four consumers now share magnitude-normalized SVD. The first
replacement inlined its custom-gradient closure and failed graph collection in
03685; independent shape-only tracing repairs that failure, passing CPU03687
and GPU03693 ownership checks. CPU03688/GPU03694 pass86 numerical/derivative/
SRUKF checks each. Block and smaller public geometry suites pass on both;
large GPU capacity03698 times out and its bounded900-second retry is active.
The sequential SVD itself remains unchanged because the tested spectra showed
no error; near-cutoff coverage remains to audit.

Source review of the callable whitening/reliability qualifiers confirms that
host materialization participates in numerical acceptance, not just output
formatting. Whitening chunks and curvature points/steps require native tensor
control with the existing evaluation order and invalid-row accounting. Distinct
chart callables in transport reliability need configuration-time dispatch with
native numerical iteration and completed-record formatting. Their test-only
literal call sites do not exempt the exported APIs. This remains planned
migration work; no callable is relabeled diagnostic merely to pass the audit.

The directional module describes its fixed-center result as diagnostic and
returns explicit nonclaims, but its candidate-qualification language and public
NumPy arrays need a call-chain eligibility decision before any runtime reuse.
Its `_validate_antithetic_frame` feeds both local reconstruction and prediction
methods in that module. No external direct runtime caller was found in the
searched roots. This remains partial source evidence rather than certified
isolation. Do not remove all three host obligations from E6 on search absence.
