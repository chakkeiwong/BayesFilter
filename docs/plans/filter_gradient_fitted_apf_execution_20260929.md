# Close fitted-APF consumer execution gaps

Current-source inspection from the Gaussian and nonlinear public score-study
endpoints finds reachable execution-policy violations:

- `adapters.py:86–94` and `nonlinear_adapter.py:84–88` call both fitted adapters.
- `fitted_twist_adapter.py:46–52` executes offline filter/fit iterations in
  Python, materializing validity after each numerical update.
- `fitted_twist_tf.py:186–191` constructs quadratic numerical feature columns
  in Python comprehensions; line219 maps a numerical validity operation over
  fitted outputs in Python. Index-pair construction is configuration metadata
  and must be distinguished from those numerical loops.
- `iapf_adapter.py:112–146` runs the adaptive likelihood/fit recurrence in
  Python, including numerical stopping and particle-count decisions from
  materialized results. Individual compiled subkernels do not enclose it.

These are confirmed remaining F07/F19 consumer obligations, not diagnostic
exceptions. The recent direction-owner repairs did not touch these branches.
Both fit coefficients offline at a nominal parameter and freeze them for the
final finite-program score; preserve that derivative target. No source-
faithfulness, canonical LEDH, posterior-quality or training claim is intended.

First bounded repair: vectorize the shared quadratic features without changing
column order (intercept, linear, squared, lexicographic cross terms), retain
full-rank and positive-precision rejection, and make the three validity
operations explicit. Use one stable enclosing XLA owner for the fixed-iteration
fitted-twist recurrence and final score; return fixed-size diagnostic histories,
completed iteration count and first invalid-fit index. Format logs/artifacts
and raise the existing errors at the host boundary. Preserve exact seed labels,
fit/final stream separation, nominal-fit conditioning, numerical controls and
public result schema. No new numerical loop may enter the allowlist.

Before implementation, freeze a small declared Gaussian and nonlinear fixture
and read the existing normalized-proposal/QR/finite-score checks in
`tests/highdim/test_younis_score_master_fitted_twist_tf.py`. Compare actual
before/after endpoints with identical frozen arrays, every fit history and
final coefficient/value/score/status, changed operands, independent polynomial
fit targets and two-step finite differences of the final scalar at fixed fit
coefficients. Test rank-deficient/negative-precision refusal and failure at an
intermediate iteration, one trace and enclosing HLO without host callbacks.
Qualify CPU reference and GPU, then matched fresh-process costs/memory.

Seed generation is a separate required wiring gate. A host-generated string
seed table is configuration metadata; host loops performing random tensor
draws are numerical execution. Compare the native generator's complete arrays
with the existing seed authority before promoting the seeded endpoint. If
compilation changes draws, localize or preserve compatibility; the geometry
stream migration approval does not authorize a new fitted-APF stream. Frozen-
cloud parity alone cannot qualify the actual seeded endpoint.

Second bounded repair: inspect `iteration_decision` and the declared finite
particle ladder, then enclose the iAPF numerical controller in TensorFlow.
Preserve all stopping windows, convergence/capacity vetoes, fitting precision,
cast validity, independently seeded stages and retained diagnostic histories.
Resolve changing particle shapes with reviewed bounded shape dispatch using
the same shared kernels; do not silently pad/rescale the probability measure,
replace adaptation by a fixed count or waive a convergence veto. Start with
exact controller fixtures, then real healthy/refused endpoints and costs. If
bounded shape dispatch cannot preserve the existing algorithm and RNG law,
record that implementation blocker and keep the route unqualified.

Prepare the first phase after the streaming-layout unit closes. Reserve at
most120 CPU seconds for read-only source/fixture preparation. Finalize exact
frozen fixture/seeds, source hashes, numerical gates and separate numerical
worker allocation before launch under the remaining global56 CPU/52 GPU-hour
caps. One numerical worker at a time. This plan authorizes bounded preparation;
it is not yet an executable numerical matrix. No HMC/NeuTra training, model
change, new ridge/clipping, package/system/cache change, subagent or main merge.

Skeptical review: treating offline fitting as harmless orchestration would
leave a real numerical loop outside XLA. Conversely, compiling random draws or
an adaptive controller without checking its finite program could silently
change the baseline. Separate primitive, frozen-array recurrence, live seeded
wiring and adaptive-shape gates answer those risks. Keep the fixed-fit and
adaptive-fit scopes distinct; neither Gaussian direction nor LEDH-state tests
can close them. This plan passes review for preparation only.
