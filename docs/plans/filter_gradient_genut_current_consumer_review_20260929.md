# Current GenUT consumer review checkpoint

Live continuation completed through04939; see
filter_gradient_genut_current_consumer_result_20260929.md. The original static
observations below retain their limited scope; executable evidence now exists
for the declared value and analytical-score fixtures.

Read-only review after dc22722f1 / run04935. No numerical worker was launched
and no finding is closed by this static review. Continue under
filter_gradient_terminal_source_review_20260928.md before another precision
intervention. Adaptive iAPF/KDM and canonical LEDH rebuilding stay excluded.

The current reduced primal implementation
bayesfilter/highdim/dual_cap_genut_primal_tf.py has SHA256
29c992a7eaf118c2a95f0510bac8b10cd4a5c148282ac6ab655f3c99603393d7.
It matches the recorded runtime bytes in04215,04248,04253,04259 and04260.
Thus a blanket stale-source explanation does not remove the saved precision
findings.04215 remains a failed diagnostic run; its saved graph results do not
establish an XLA result. Corrected cost/gradient cohort evidence remains subject
to its actual numerical vetoes and missing report-local digest limitations.

| Inspected caller | Source-level branch | Current implication |
|---|---|---|
|genut_guided_proposal_tf._restore_cloud_primal, lines694/935/1032 |Reduced primal executes only for dual_cap_enabled=True and trust_region_enabled=False |The reduced-primal weight-gradient finding must not be attributed automatically to the full trust-region derivative authority |
|Same helper, lines936--966 |With both flags true it calls higher_moment_shape_jvp using explicit zero tangents for value work |Separate numerical authority; live wiring and derivative-role reconciliation still needed |
|ledh_canonical_value_program_tf.make_canonical_value_program, lines60/348 |Defaults both flags false, forwards configured flags to the reset |The registered value path has an explicit reduced-branch option; the function name does not confer canonical admission |
|genut_guided_proposal_tf.finite_value_standard_score_guided_proposal, line1238; ledh_pfpf_genut_initialization_tf._apply_reset, line248 |Call reset without overriding its false dual-cap default |No direct reduced-primal invocation through those default calls |
|ledh_pfpf_genut_initial_rqmc_tf.finite_value_standard_score_initial_rqmc, lines316/543/791 |Defaults both flags false and forwards configured flags at initial and subsequent resets |Optional reduced branch remains reachable; score role and current users require individual disposition |

Within the inspected library and tf_tfp experiment roots, the reduced function
has one direct importing caller, the shared reset helper. Text search finds
the two finite-value standard-score entry names outside their definition modules
only in scripts/filter_repair_additional_fixtures.py. These are static observations,
not proof about dynamic dispatch, aliases, external callers or admission.

Next evidence: execute bounded branch-wiring checks from the registered value
endpoint through the shared reset, distinguishing disabled, reduced and full
trust-region configurations. Preserve ordinary records if instrumenting runtime
execution. Inspect the registered analytical-score call chain independently;
the saved reverse-gradient stress tests are diagnostic derivative work and do
not establish a defect in an uninspected analytical-score authority. Bind each
remaining cap-report, gradient, execution and cost finding to the actual caller
and its admissible role. Existing noncanonical markers must remain explicit;
neither lack of a default call nor a static nonclaim closes the optional-route
numerical obligation. No new broad benchmark matrix is justified by this review.

Bounded live continuation baseline9d83dc847: at most5 serialized workers,
900 CPU/900 GPU seconds after04935. Use the existing seeded value LGSSM fixture
and the existing seed71 nonlinear analytical-score fixture, with their frozen
controls. The purpose is callable wiring, not tuning or numerical admission.
Compare ordinary complete endpoint records with an instrumented call that adds
resource counters to the reduced correction, full value JVP and analytical
batched JVP. Require unchanged values/decisions under the existing bounds and
actual nonzero runtime counts for each configured branch; tracing alone is not
an execution witness. Use CPU reference and trusted GPU with verified growth.
Block the reduced primal inside the analytical test; reaching it is a failure.
Record any already-refused fixture without treating branch execution as numerical
qualification. No target, RNG, method, controls or accuracy bound changes.

Skeptical review: resource counters can change graph lowering, so failed output
preservation invalidates attribution rather than proving a runtime defect.
Successful wiring only identifies the executed authority; it cannot clear a
precision failure or certify canonical conformance. Source drift, invalid
fixture/reference or exhausted allocation stops the experiment. The failure
and retry policy remains the existing campaign policy. Source-bound reports
and logs use fresh genut_consumer run directories; results are recorded in
filter_gradient_genut_current_consumer_result_20260929.md.

04936 CPU passes the live value/analytical witnesses.04937 GPU fails because
TensorFlow places int32 resource counters on CPU, making them inaccessible to
the GPU/XLA program. This is diagnostic instrumentation failure; use int64
counters placed with the actual observation/initial-state tensors and retry
under the same budget. No algorithm, data, controls or runtime source changes.
Preserve the failure and require unchanged complete ordinary outputs again.
