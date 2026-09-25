# Transitive GenUT import audit, September 26

The run-03993 discovery follows static repository imports from KDM. Its 80 modules are an overapproximation, including conditional and diagnostic imports; this is not executed call-chain coverage. Fifteen previously uncovered modules are now in the exact-source guard. Their 64 syntax sites require 63 distinct AST allowances (two identical serialization comprehensions share one digest with occurrences=2). No numerical recurrence was allow-listed.

| Module | Reviewed iteration sites | Disposition |
| --- | ---: | --- |
| `bayesfilter/diagnostics.py` | 1 | Exact fixed_schema allowances; each reason recorded with the AST digest in the policy. |
| `bayesfilter/highdim/cubature_genut_adapters.py` | 0 | No flagged Python iteration, NumPy, pfor, callback or non-XLA declaration. |
| `bayesfilter/highdim/cubature_genut_filter.py` | 3 | Exact fixed_schema allowances; each reason recorded with the AST digest in the policy. |
| `bayesfilter/highdim/diagnostics.py` | 2 | Exact fixed_schema, host_reporting allowances; each reason recorded with the AST digest in the policy. |
| `bayesfilter/highdim/dual_cap_genut_primal_tf.py` | 0 | Two correction recurrences repaired to native loops; reduced noncanonical status explicit. |
| `bayesfilter/highdim/fixed_branch.py` | 7 | Exact host_reporting, host_validation allowances; each reason recorded with the AST digest in the policy. |
| `bayesfilter/highdim/genut_guided_proposal_tf.py` | 2 | Exact fixed_schema allowances; each reason recorded with the AST digest in the policy. |
| `bayesfilter/highdim/ledh_pfpf_genut_model_callbacks_tf.py` | 0 | Numerical one-hot comprehension replaced by one batched tensor operation. |
| `bayesfilter/highdim/sir_latent_preclip_tf.py` | 1 | Open numerical time loop and host-only scaled-model/time conversion; separate anchored repair required. |
| `bayesfilter/highdim/transport.py` | 13 | Open mixed transport module; see call-chain findings below. |
| `bayesfilter/highdim/validation.py` | 32 | Exact fixed_schema, host_validation allowances; each reason recorded with the AST digest in the policy. |
| `bayesfilter/inference/posterior_adapter.py` | 10 | Exact fixed_schema, host_reporting, host_validation allowances; each reason recorded with the AST digest in the policy. |
| `bayesfilter/nonlinear/cut_tf.py` | 0 | No flagged Python iteration, NumPy, pfor, callback or non-XLA declaration. |
| `bayesfilter/nonlinear/svd_cut_tf.py` | 0 | No flagged Python iteration, NumPy, pfor, callback or non-XLA declaration. |
| `bayesfilter/results_tf.py` | 2 | Exact fixed_schema, host_reporting allowances; each reason recorded with the AST digest in the policy. |
| `bayesfilter/structural.py` | 3 | Exact fixed_schema allowances; each reason recorded with the AST digest in the policy. |
| `bayesfilter/structural_tf.py` | 2 | Exact fixed_schema allowances; each reason recorded with the AST digest in the policy. |

The table records the candidate after correction-loop and one-hot edits. Zero findings means only that the static checker found none of its prohibited constructs; it does not establish a complete enclosing XLA program, valid derivative or source-faithfulness.

The reduced correction is called by `genut_guided_proposal_tf._restore_cloud_primal` only when dual-cap correction is enabled and the trust-region branch is disabled. The trust-region branch uses `higher_moment_shape_jvp`. Complete-record tests call this real reset consumer in its streaming K=N configuration. The reduced module now explicitly records `CANONICAL_LEDH_ADMITTED=False`; existing candidate identities and downstream admission blocks remain necessary.

`FixedTTSIRTTransport` public forward/inverse, conditional and density methods already delegate to default-XLA `ttsirt_native_tf.evaluate_transport`. The native program calls axis-coordinate/density primitives and the static working-set estimate, not the obsolete private bisection helpers. Repository package search found no call to `_inverse_axis_batch`, `_inverse_axis_suffix_batch` or `_reference_measure_density` outside their definitions. `KRTransport` retains host numerical loops and is exercised by independent `tests/highdim/test_transport.py`. Public `potential` and `proposal_log_density` still apply an eager logarithm around the compiled density result; their full enclosing boundary requires review. Do not label the entire mixed module diagnostic or policy-clean. Dynamic/external callers are not disproved by this static search.

`LatentPreclipSIRSSM.simulate_from_standard_normals` retains the time loop, constructs a parameter-scaled model with host `.numpy()` validation, and uses Python integer time conversions. A loop-only edit would therefore fail XLA. Its separate plan must preserve the un-clipped t=0 state, clipping after later process noise and observation timing, while using one shared numerical authority. Author paper section 6.3 and the local sir_austria source must be inspected before that repair.

These are execution and source-discovery findings only. They do not qualify full SIR/TT consumers, canonical LEDH, HMC, production readiness, numerical cross-mode reporting equality or target-scale memory.
