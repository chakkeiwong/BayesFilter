# Prepared enclosing posterior initializer result

The internal complete initializer now executes initial evaluation, bounded
localization, movement, curvature, candidate accounting and final eigen summaries
inside one XLA program. Nine distinct checks pass on both CPU and GPU 2. The
exported public endpoint is not yet connected to this owner; public integration,
error precedence and matched endpoint costs remain separate gates.

The comparison executes the entire posterior_local_initializer.py module from
031692a0b under an isolated module name and verifies all 690 pinned BayesFilter
Python dependencies byte-for-byte before sharing imports. Every comparison
retains complete public payloads and ordered physical callbacks. Existing
1e-10 absolute/relative tolerances are unchanged; discrete decisions, row counts,
replay and trace/HLO identity agree exactly. Starts and scales change on the same
owner, construction executes zero target rows, and outputs expose no external
start/scale derivatives. The reference already incorporates prior campaign
repairs and does not replace outstanding oldest-original comparisons.

| Case | Observed result on both backends |
| --- | --- |
| D1/D3 stationary Gaussian, batched | Accepted; original payloads and target rows preserved, including 48/148 rows at the stationary start |
| D1 displaced scalar Gaussian | Accepted; 55 physical target rows in the recorded inputs |
| D3 nonlinear target | Original curvature-geometry rejection retained |
| Invalid initial target | Rejected after exactly one target row |
| Invalid third callback | Original recovery retained, with invalid-row accounting and identical calls |
| Four-row evaluation budget | Exact budget rejection and original partial records retained |
| Eligibility mismatch | Original mismatch precedence, accounting and records retained |
| Unsupported XLA string operation | Compiler error with zero target execution and no eager fallback; Python owner/program/scope collected |

CPU pilot 04389 passed D1 with changed scales. Expanded D3 construction 04390
failed because tf.init_scope stops both backward and forward recording; the
factor fitter's internal ForwardAccumulator then produced a missing JVP.
Removing the blanket suppression repairs construction while frozen outputs
retain the intended external derivative boundary. D3 04391 passes under active
construction and invocation tapes. Runtime documentation changed before the
remaining cohort, without numerical changes. Exact intermediate bytes are
reconstructed and checked against the run hash.

CPU 04392--04394 pass stationary D1, displaced D1 and nonlinear D3. Initial-invalid
04395 exposed a diagnostic formatting mismatch: clean() encoded NaN as a string,
whereas the existing public JSON boundary emits null. The retry uses the original
_json_ready formatter, leaving numerical code and criteria unchanged. CPU
04396--04400 pass all four adverse cases plus the compiler/owner check. GPU
04401--04409 pass all nine checks under one frozen source closure. Failed sources
and artifacts remain preserved. Post-merge policy 04388 and final policy 04410
both pass 141 checks; the new numerical source adds no policy allowance.

These fixtures use the default factor_max=2 at D1/D3. Those dimensions do not
identify the two-factor covariance family; an identifiable larger fixture remains
necessary during public qualification. The invalid-third-callback case exercises
locator recovery, not a full-endpoint curvature-partition failure. Independent
prepared-curvature checks cover partial curvature partitions, but do not replace
that explicit limitation. Python owner collection does not prove native compiler
cache eviction. These correctness runs establish no performance ranking.

| Decision | Primary criterion | Veto / uncertainty | Next action | Not concluded |
| --- | --- | --- | --- | --- |
| Retain the internal enclosing controller | Nine complete-reference checks per backend pass | Public dispatch and late error precedence remain untested | Connect the exported endpoint with the reviewed error boundaries | Whole public API compliance |
| Retain frozen derivative behavior | Active tapes, changed operands and replay pass | This is an initializer, not an analytical score | Keep internal fitter derivatives enabled and returned tensors frozen | New gradient correctness claims |
| Continue integration | No numerical mismatch remains in this cohort | Larger identifiable two-factor fit, native memory and costs remain | Execute exported-reference and matched cost checks | Main merge, HMC readiness or campaign completion |

Skeptical review identified and repaired a construction boundary that appeared
safe in D1 while breaking D3 forward derivatives. It also found a reporting-only
failure in the test harness and the limited two-factor coverage. Public review
has identified two additional late-configuration error boundaries; those are
recorded in the enclosing plan and will be tested before public qualification.
Independent terminal review remains pending.

Runs 04388--04410 charge 736.749819 CPU and 1020.928830 GPU seconds in 14 CPU and
nine GPU workers, including both policy checks and all failures. The environment
is tf-gpu, TensorFlow 2.19.1, explicit CPU-hidden reference workers and trusted
GPU 2 workers with verified memory growth, TF32 enabled and float64 fixtures.
Exact commands, source hashes and environment are retained per run under the
campaign artifact root. The evidence archive and its SHA-256 verification receipt
are posterior-enclosing-04410-evidence.tar.gz and
posterior-enclosing-04410-verification.json.
