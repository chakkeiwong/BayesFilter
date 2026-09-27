# Filter and gradient repair resume checkpoint

Checkpoint through 04488. No worker is active. Final public integration and
graph compatibility pass 11 CPU checks (04467--04476) and 11 GPU checks
(04478--04487). Analyzer 04477 passes 14; policy 04488 passes 147. The 276-source
policy guard has 1436 exact allowances; the graph repair adds none.

The public locator option retains original XLA movement/curvature dependencies;
actual fit compilation status is reported. Default enclosing execution remains
XLA. No private TensorFlow workaround or shared-fitter modification remains.
Complete D1/D3 graph results, ordered callbacks, changed inputs, replay, single
trace, frozen gradients and compiler/cache boundaries pass on both backends.
All 13 exported XLA cases previously passed on CPU/GPU through 04434, including
D5 two-factor. Reference imports now execute from isolated Git 031692a0b sources.
Standalone all-graph raw fitter 04454 remains a diagnostic-only failure; the
public graph option is not an all-non-XLA timing baseline. No tolerance waiver.

Evidence archives posterior-public-localization-04466 and
posterior-public-qualification-04488 verify all 326 and 84 members respectively.
Final source snapshot: posterior-public-stage-jit-04467-source. Result:
filter_gradient_posterior_public_result_20260927.md.

Next commit/push this qualified checkpoint, then run installed cost batches
posterior_initializer_cost_cpu and _gpu, repeats 0/1/2, 300 seconds per worker.
Use GPU 2 if preflight still admits it. Sources must remain frozen through all
36 cost arms. Stop on first numerical failure and preserve measured artifacts.
The separate 10800-second cost allocation (18 CPU/18 GPU plus four short
analysis/policy workers) is within unchanged global caps. Costs are unlaunched.

Charged through 04488: 92738.745087 CPU / 85263.964466 GPU seconds; remaining 30.239237
CPU / 28.315565 GPU hours under 56/52 caps. Extra 24 CPU hours are already included.
Enclosing correctness unit used 9522.019909/10800 seconds in 104 CPU/56 GPU
workers (caps144/96). Latest pushed 6bceb767e includes origin/main 06590cb5a;
fetch remains unchanged. Main stays unmerged and all F01--F20 terminal findings
remain open. Broader gaps: actual DZ5 consumers, reporting/precision, native
capacity, repeated terminal costs/review. No subagents. Controlling plan:
filter_gradient_posterior_initializer_enclosing_20260927.md.
