# LGSSM NeuTra Training Topology Performance Benchmark Plan

Date: 2026-07-14

## Objective

Determine whether replacing the historical Python loop over individually XLA-
compiled optimization steps with one XLA-compiled `tf.while_loop` makes the
unchanged source-anchor LGSSM NeuTra training computation faster.

This is a diagnostic benchmark. The historical host-stepped route is permitted
only inside the benchmark harness and must not become an active training entry
point.

## Research Intent Ledger

| Item | Predeclared role |
| --- | --- |
| Main question | At matched step count, state, seed stream, target, recipe, batch size, dtype, GPU, and update equations, is graph-native training faster than host-stepped training? |
| Mechanism under test | Removal of one Python invocation and device/host synchronization boundary per optimization step. |
| Baseline | Historical topology: one XLA-compiled step function invoked from Python once per step, with terminal materialization after each step. |
| Candidate | Current topology: the same step body executed by one XLA-compiled `tf.while_loop`, with terminal materialization after the loop. |
| Expected failure mode | Whole-loop compilation may cost more when cold; the graph-native route may only win after enough steps amortize that cost. |
| Promotion criterion | For a matched rung, graph-native warm elapsed time is lower than legacy warm elapsed time and final state/diagnostics match exactly or within the declared numerical tolerance. |
| Promotion veto | Any target-status failure, nonfinite value, GPU/XLA failure, update/state mismatch above tolerance, or comparator semantic mismatch. |
| Continuation veto | Harness invalidity, missing synchronization, inability to restore identical initial state, or aggregate wall budget exhaustion. A slower candidate is not a continuation veto. |
| Repair trigger | Timing does not force device completion; comparator results do not match; compile and execution boundaries are conflated; or an artifact lacks provenance. |
| Explanatory diagnostics | Cold elapsed time, warm elapsed time, compile topology, per-step time, crossover estimate, and historical artifact timings. |
| Forbidden conclusion | No transport-quality, posterior-correctness, HMC-convergence, recipe-ranking, production-readiness, or broad GPU performance claim. |

## Evidence Contract

The exact comparator is the source-anchor recipe: 18 dimensions, three dense
IAF stages, hidden layers `(18, 18)`, batch size `128`, `float64`, constant
learning rate `5e-3`, manual Adam, clip norm `10`, seed `(20260714, 1411)`, the
same affine geometry, and exact target signature
`f47619320ded5f70259c6932eb2436642a02834c7a0249c7c52c20a5a2302f30`.

The primary criterion is synchronized warm wall time for the complete matched
step count. Cold compile-plus-execution time is reported separately and cannot
be substituted for the warm criterion. Each mode must restore the identical
initial trainable and Adam state before its cold and warm measurements. The
benchmark must force device completion inside the timed interval.

Final trainable variables, Adam moments, and selected diagnostics must be
identical between cold and warm repetitions of one mode. Cross-mode values must
have maximum absolute difference at most `1e-12`; exact SHA-256 equality is
reported separately. No timing comparison is admissible if this veto fails.

Artifacts will be written under the fresh versioned root
`docs/benchmarks/artifacts/lgssm_neutra_training_topology_benchmark_2026_07_14/`
and summarized in a result note under `docs/plans`.

## Default And Assumption Audit

| Choice | Provenance | Justification | Failure mode | Early diagnostic | Status |
| --- | --- | --- | --- | --- | --- |
| Source-anchor recipe | Target-specific NeuTra protocol | It is the previously timed route and isolates topology | Other recipes have different compile/runtime behavior | Bind and record every recipe field | fixed baseline, not a promoted default |
| Rungs `5`, `20`, then conditional `100` | Prior five-step timing and bounded-compute policy | Five steps tie to historical evidence; 20 exposes amortization; 100 is used only if crossover remains material and unresolved | Short rungs may mispredict serious runs | Compare cold and warm trends; label extrapolation | benchmark design hypothesis |
| One cold and one warm repetition per mode/rung | Compute cost of exact target is high | Separates compile from execution with bounded cost | No uncertainty estimate; transient load can distort timing | Record process/GPU provenance and describe timing as descriptive | convenience choice, not statistical ranking evidence |
| Separate process per mode/rung | Avoids reusing a compiled function across cold comparisons | Preserves a genuine first XLA execution for each cell | System cache and run order still affect cold time | Warm timing is primary; cold remains descriptive | reviewed benchmark choice |
| Explicit terminal tensor materialization | TensorFlow GPU calls are asynchronous | Includes actual device completion in elapsed time | Missing synchronization yields false speedup | Materialize every returned diagnostic before stopping timer | required validity check |
| Exact shared step body in one diagnostic harness | Isolates only control topology | Prevents optimizer/target drift between comparator and candidate | Harness could drift from active training source | Anchor fields and equations to `neutra_training.py`; compare the graph-native two-step diagnostics to preserved evidence | reviewed diagnostic implementation |

## Skeptical Plan Audit

| Risk | Verdict and mitigation |
| --- | --- |
| Wrong baseline | The earlier 5-step and 2-step artifacts are not compared as performance evidence. Both benchmark modes use the same new harness and exact configuration. |
| Proxy promoted | Only synchronized wall time is the performance criterion. Loss and target telemetry are validity checks, not speed criteria. |
| Missing stop condition | Stop on parity, target, finite-value, GPU/XLA, synchronization, artifact, or wall-budget failure. |
| Unfair comparison | Both modes use the same step function, state initialization, seed indexed by step, output diagnostics, and XLA setting; only host loop versus graph loop differs. |
| Hidden asynchronous timing | The timer stops only after all returned tensors have been materialized. |
| Warm state drift | Initial trainable variables and Adam moments are restored before every repetition. |
| Compile/execution conflation | Cold and warm measurements are separate. `get_concrete_function` trace time is recorded separately from first XLA execution. |
| Stale context | The active code and preserved artifacts were inspected on 2026-07-14; the benchmark records Git commit and dirty status. |
| Environment mismatch | Trusted GPU preflight and TensorFlow device probe precede execution; both modes run in the same `tf-gpu` environment and hardware class. |
| Misleading 100-step estimate | A 100-step claim requires the 100-step rung. If it is not run, any crossover or 100-step number is explicitly an extrapolation. |

Audit verdict: **PASS**. The plan answers the topology question without treating
unmatched historical timings as evidence. The important limitation is one timed
warm repetition per cell; therefore observed timing differences are descriptive
for this machine and configuration, not a statistical or broad superiority
claim.

Preflight repair note: the first host-cell launch stopped before timing or
artifact emission because the harness incorrectly required the complete host-
step graph to contain no `While` operation. The shared exact LGSSM target has
internal TensorFlow control flow in both modes, so operation absence cannot
identify the outer training topology. The check was narrowed to require the
explicit outer `tf.while_loop` for the graph-native mode while recording the
host comparator's outer Python loop directly. This changes no computation or
budget and is a harness-validity repair under the predeclared repair trigger.

The repaired host cell then completed both timed repetitions but stopped before
artifact emission because its relative CLI output path was compared directly
with the absolute repository root. The CLI now normalizes all input and output
paths to absolute repository paths before execution. No timing from the failed
process is promoted because it did not emit the required structured artifact;
the launch still consumes one process from the campaign budget.

## Execution Ladder And Budget

1. Run trusted GPU and TensorFlow device preflight.
2. Compile-check the diagnostic harness and run the matched 5-step cells.
3. If parity passes, run the matched 20-step cells.
4. Run matched 100-step cells only if the 20-step results leave the 100-step
   conclusion unresolved and the remaining wall budget is adequate.
5. Write a comparison artifact and result note. Preserve every attempt.

Budget: at most six benchmark processes, one cold and one warm repetition per
process, no more than 100 steps per repetition, and at most 45 minutes aggregate
wall time. No training artifact from this benchmark is eligible for downstream
NeuTra or HMC use.

## Required Run Manifest

Record Git commit and dirty status, exact command, Python/conda environment,
TensorFlow version, CUDA/GPU device, XLA and TF32 settings, dtype, seed, target
signature, recipe, step count, cold/warm wall time, output path, this plan path,
and the terminal result path.

## Stop Conditions

Stop and classify the result as invalid if the two modes do not execute the
same updates, any state/diagnostic parity tolerance fails, target telemetry is
invalid, a nonfinite value occurs, output is not on GPU, XLA compilation is not
observed, synchronization is not inside the timer, an existing artifact would
be overwritten, or the 45-minute budget is exhausted.
