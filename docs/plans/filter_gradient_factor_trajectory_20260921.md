# D5 factor optimizer trajectory attribution

Question: do the remaining original-source D5 record differences arise at a
specific erroneous objective evaluation or from amplification of small arithmetic
differences over the unchanged L-BFGS recurrence? The authority is the complete
`3582b4ac` closure and the four crossed-input records in run 01826. Earlier
initial-state and row-decoder interventions did not repair the complete record.

Capture every objective input, value and gradient with bounded TensorFlow
resources in diagnostic tests only. Use the original and current prepared
arrays separately, the existing two-factor configuration, and unchanged
optimizer settings. For each arm, first compare every instrumented public field
with its archived uninstrumented record at the existing `atol=rtol=1e-10`.
Instrumentation that changes a record invalidates attribution for that arm;
preserve it and localize before drawing conclusions. No runtime injection is
installed. A fixed 4096-evaluation buffer is an observability limit, not a new
optimizer stop rule; overflow invalidates the diagnostic.

Replay the current objective at every captured original input with matching
prepared data and anchors. Record the complete trajectories, per-evaluation
value/gradient errors, iterate divergence and final gradient norm. These are
explanatory diagnostics; only the unchanged complete-record tests can qualify a
repair. An error decreasing, an unchanged iteration count, or a small objective
residual cannot waive a failed field or justify a different optimizer tolerance.

Run the CPU reference first with GPUs hidden, one worker through the approved
campaign driver, 300 seconds per prepared-input case. If instrumentation is
valid, follow with the corresponding GPU3 diagnostic after idle preflight. All
runs use fresh numbered campaign directories, recorded source/input hashes,
verified GPU growth where applicable, and the existing 32 CPU / 52 GPU-hour
caps. Stop this diagnostic on overflow, invalid attribution, nonfinite objective
or gradient, or its deadline. Do not repeat unchanged failures.

Commands use the existing absolute driver prefix and
`test --group factor_trajectory_original --device CPU --test-timeout-seconds 300`
or `factor_trajectory_current`; GPU counterparts add `--device GPU
--test-gpu-index 3` instead. Results are `factor-trajectory-<data>.json` beside
the driver manifest and JUnit record.

Skeptical review: resource writes can change XLA fusion, so reproducing the
uninstrumented public records is mandatory before interpreting the traces.
Replay uses the same finite objective, without numerical differencing, altered
seeds, or a new solver. This diagnostic distinguishes amplification from local
arithmetic error; it does not promise bitwise agreement, justify a baseline
change, or establish whole-lifecycle correctness or performance.

02097 captured both 217-evaluation trajectories and reproduced every archived
public field, then failed at replay tracing because the diagnostic imported
the public barrier wrapper but used the generated-op keyword. Use the same
generated barrier op as the runtime and preserve capture results before replay
so another replay failure cannot discard the successful observations. This is
a harness repair; the runtime and numerical contract are unchanged.

02098/02099 pass both CPU instrumentation checks: each original/current trace
has 217 evaluations and preserves its archived public fields. Original-input
same-state replay gradient error is at most 7.272e-17; the original/current
position trajectories first separate by more than 1e-10 at evaluation 104.
These are explanatory observations, not a tolerance waiver.

GPU follow-up uses fresh uninstrumented same-device comparators on the archived
prepared arrays. Comparing instrumentation directly to CPU records would
confound device arithmetic with instrumentation. Keep both comparisons in the
artifact, but gate instrumentation only on its own device's uninstrumented
record. Use an int64 resource counter because TensorFlow places int32 resources
on CPU; numerical histories remain float64. This changes only diagnostic
accounting. The original full-lifecycle numerical gate is unaffected.

Next diagnostic, independent of instrumented traces: fit the archived original
prepared data with the original fitter and six one-ULP perturbations (training
offsets, training scores, or both, each toward positive/negative infinity).
Keep holdout, weights, center, optimizer, tolerances and runtime unchanged.
These tiny input perturbations represent sensitivity probes, not replacement
campaign data. Compare all fields at the existing tolerance and report the
full differences. Preserve categorical decisions and iteration/evaluation
counts explicitly; any change is evidence to investigate, not to ignore.

Independently reconstruct the known quadratic fixture covariance and loadings
from `tests/test_filter_repair_structured_memory.py::_inputs`, then transform
the covariance by the changed scale. Report covariance, precision, loadings,
score prediction and holdout errors against that analytic geometry for every
arm, including the unmodified current XLA fit. This reference checks the
actual fitted object, not just loss agreement. It does not make the optimizer's
finite stopped result equal to the exact target or authorize a tighter stopping
rule. Confirm the reconstructed precision generates the archived responses to
roundoff before interpreting it.

Run `test --group factor_input_sensitivity --device CPU
--test-timeout-seconds 300`, then a same-device GPU3 counterpart if useful.
The entire seven-original-fit plus one-current-fit experiment must fit its
single worker bound. Save all records and input hashes. The original untouched
fit must reproduce its CPU archive (or fresh GPU control). Results can support
a concrete contract discussion, but cannot loosen the existing 1e-10 gate.
Pre-run review: perturbation directions are declared before measurement; do
not select a direction that makes the candidate look favorable, rank methods,
or reinterpret diagnostic field discrepancies as missing fields.

## Results through 02105

| Run | Device | Observation | Result |
| --- | --- | --- | --- |
| 02097 | CPU | Original-input capture, replay import | Failed at replay; preserved and repaired |
| 02098 | CPU | Original prepared data, both fitter traces | Complete records preserved, 217 evaluations |
| 02099 | CPU | Current prepared data, both fitter traces | Complete records preserved, 217 evaluations |
| 02100 | GPU3 | Original prepared data, fresh controls and both traces | Complete records preserved, 217 evaluations |
| 02101 | CPU | Six one-ULP original sensitivity arms and analytic geometry | Original self-sensitivity exceeds existing field gate |
| 02102 | GPU3 | Current prepared data, fresh controls and both traces | Complete records preserved, 217 evaluations |
| 02103 | GPU3 | Same sensitivity/reference experiment | Original self-sensitivity exceeds existing field gate |
| 02104 | CPU | Campaign and source-policy tests | 72 pass |
| 02105 | CPU | Working-source inventory | 2941 files, 2940 parsed, one unchanged vendor error |

All six numerical diagnostic workers pass their stated observation contracts;
they do not pass or replace the original lifecycle's 1e-10 gate. On CPU/GPU the
original-input replay gradient discrepancies are <=7.272e-17/4.150e-17; on
current inputs they are <=7.253e-17/4.153e-17. Current objective replay agrees
exactly with the observed current gradient on these trajectories. Original
self-sensitivity has maximum scaled field errors 2.1403e-9 CPU / 2.9149e-9 GPU.
The proposed interpretation and exact owner decision are in
[factor equivalence](filter_gradient_factor_equivalence_decision_20260921.md).

No new NumPy runtime use, numerical-loop exception, environment mutation,
external source/pin change, optimizer setting, seed, threshold, or mandatory
tolerance change. The added NumPy use is explicitly independent diagnostic
one-ULP fixture construction inside a test. Hardware remains the same RTX 4090
class on GPU3, with verified memory growth and idle preflight. New numerical
work costs 120.805 CPU / 203.601 GPU seconds, including the failed CPU attempt;
policy/inventory add 39.717 CPU seconds. The cumulative caps remain unchanged.

| Decision | Primary criterion | Veto | Main uncertainty | Next action | Nonclaim |
| --- | --- | --- | --- | --- | --- |
| Preserve diagnostic checkpoint | Instrumentation reproduces complete original/current results and analytic reference is checked | Existing strict lifecycle field gate still fails | General target and conditioning coverage | Resolve the reviewable equivalence decision, then qualify all affected paths | No completed repair, runtime accuracy guarantee or merge readiness |

Post-run review: a local compiler defect away from these finite trajectories
remains possible. The evidence supports amplification at the measured numerical
scale; it does not prove every fitted result equivalent. No performance ranking
is made. This is the primary agent's review; no independent reviewer was used.

Both new diagnostic tests pass focused Ruff. The unchanged driver portions
retain five existing I001/F601/C408 lint findings (import order, repeated
same-valued `sequential_geometry` key, and dict-construction style); this
checkpoint makes no whole-driver or repository-wide lint claim. Whitespace and
ledger consistency checks pass. The fetched remote repair branch has no
divergence and remote main is an ancestor; main remains unmerged.
