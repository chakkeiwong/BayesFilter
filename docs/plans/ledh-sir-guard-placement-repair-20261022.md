# LEDH flow guard placement repair — execution plan

## Question and scope

Can the shared analytical LEDH flow retain its existing rejection coverage while
removing determinant factorization from the sequential flow-substep dependency
chain? Work only on `ledh-refactor-with-policy-fix` in the LEDH execution worktree.
Preserve unrelated edits. OT epsilon/iterations remain (1,5), with the existing
terminal balance count. No guide training, numerical-policy change, or new score
accuracy claim is authorized by this engineering repair.

The preceding design is
[guard-placement-repair-design](ledh-sir-guard-placement-repair-design-20261022.md).
Its conditional determinant-positivity theorem is **insufficient to remove the
guard**: `valid_predict` does not certify the newly predicted covariance PSD.
Negative UKF weights make that an actual missing premise. Adding a new covariance
factorization would also change the rejection domain. Neither shortcut is used.

## Mechanism and invariants

1. Keep configuration checks outside tracing and Q/R checks at their existing
   once-per-evaluation boundary. Keep moving innovation Cholesky, state/tangent
   finite checks, and absorbing validity in each flow substep.
2. Store the existing reduced determinant matrices and analytical tangents for
   a bounded group of substeps. Run the **same checked pivoting determinant and
   derivative solve**, batched over group and particles, after that group.
3. Accumulate log determinants and their tangents in chronological native TF
   loops, preserving the old additions and epsilon placement. Check every
   prefix for finite arithmetic. Never replace transient checks by final-only
   checks, a parallel sum, an unchecked determinant, or an unproved PSD argument.
4. Mask rejected particles at each group boundary and preserve first failing
   substep indices. A determinant failure may delay masking within its group;
   that particle is rejected permanently. Accepted trajectories must retain the
   same finite program. Later diagnostic states on already-rejected trajectories
   need not equal the old zero-placeholder trajectory.
5. Use one shared flow implementation for plain, conditional, physical-mixture,
   generic, reduced-coordinate and annealed consumers. No model/lane fork.

The history-buffer cap is 16 MiB for the pair of matrix/tangent histories. Choose
group length `min(L, max(1, floor(cap / single_step_pair_bytes)))` at trace time.
An unknown static particle count uses length one. A single pair exceeding the
cap still uses length one; the cap bounds extra history, not the unavoidable
single-step working set. This is an engineering memory hypothesis, not a
scientific tuning control. Test group boundaries and the final partial group.

## Research intent and evidence contract

| Role | Prespecified interpretation |
|---|---|
| Main question | Same accepted computation and rejection coverage with less serialized guard overhead |
| Primary correctness criterion | Focused fault tests and all existing frozen replay rows pass unchanged tolerances; common validity flags agree exactly |
| Primary speed criterion | Alternating call-paired timings against retained candidate02; paired log-ratio bootstrap intervals, separately for three consumers and zero/nonzero direction |
| Promotion veto | Missed transient singular/nonfinite/value/tangent failure, changed accepted target, parity failure, unsupported consumer fork, or increased live memory beyond bounded history plus normal compiler workspace |
| Continuation veto | Invalid harness/artifact, exhausted remaining allocation, hardware/resource failure preventing valid comparison |
| Repair trigger | Local implementation/test/harness failure; repair under the same target and budget, preserving each attempt |
| Explanatory only | HLO loop placement, operation counts, compilation time, individual timings, GPU contention, first failure indices |
| Nonconclusions | No claim of guide quality, posterior validity, HMC readiness, repaired pre-merge numerical sensitivity, or universal speed improvement |

The exact baseline is frozen
`artifacts/ledh-sir-merge-repair-20261022/candidate-02-final-source`, not the unsafe
determinant ablation or the unretained Cholesky candidate. Pre-merge comparisons
remain historical engineering context for this repair, with their independent
plain-score mismatch explicitly unresolved. Cheap adversaries for this
engineering question are length-one checked execution, the retained checked
flow, and direct individual checked determinant calls. Evaluate separately for
plain, fully conditional and physical-mixture consumers; zero and nonzero
directions; healthy, singular, nonfinite, derivative-only and recovered-later
faults. This is not selection of a learned method against a weak comparator.

### Timing analysis fixed before measurement

Reuse the four preselected exposed datasets 263043001, 263043004, 263043021,
263043052 and eight alternating paired blocks. Preserve pairing within each
dataset, arm and direction. For each of the six arm/direction cells, average
paired log(candidate/baseline) times across the four datasets in each block.
Resample the eight whole blocks 20000 times (standard-library RNG seed 20261022)
and report percentile 95% intervals for the geometric time ratio. Report the same
interval separately for each dataset. Claim an observed speed improvement only
if all six pooled upper limits are below one and no individual regime has a
lower limit above one. Inconclusive comparisons remain inconclusive. These
intervals assume approximately independent blocks and concern these exposed
inputs and this device; eight blocks do not establish broad performance.

Pre-measurement audit found that the inherited analyzer uses Student-t intervals
and allows a 10% physical-mixture slowdown. Those are historical criteria for
the preceding repair, not the present speed criterion. Preserve its parity,
source and HLO checks; use the separate guard-placement bootstrap analysis for
this phase's decision. This correction precedes all current timing measurements.

## Defaults, assumptions and skeptical review

| Choice | Provenance / status | Failure mode and early check |
|---|---|---|
| Existing determinant algorithm and eps scaling | Retained engineering baseline | Batch axes or tangent scaling wrong; compare individual calls and directional finite differences |
| Group cap 16 MiB | New bounded scheduling hypothesis | Workspace/compilation overhead defeats speed; size tests, allocator and actual-consumer timing |
| Fixed FP64 SIR T5/N64 replay | Existing incident comparator, not a scientific default | Narrow scope hides other geometry/dtype bugs; add FP32, order1/2/9, full-state and reduced-coordinate mechanics checks |
| Fixed OT (1,5), schedules, seeds and tolerances | Owner instruction and frozen replay | Silently changed baseline; verify manifests and source/input hashes |
| Delayed determinant rejection | New implementation assumption | Rejected particle contaminates others or becomes accepted later; fault tests, absorbing masks, no particle interaction inside flow |
| XLA and GPU memory growth | Repository execution policy | Compiler rewrite/parity or resource failures; focused GPU checks before full replay |

Skeptical audit before execution: **PASS with the revised batched-check design**.
The original PSD shortcut is rejected. The actual moment endpoint, not a helper
microbenchmark, supplies timing and replay evidence. Proxy/HLO counts cannot
promote speed. Accepted arithmetic, every determinant and prefix, and failure
history remain checked. GPU contention can make speed inconclusive; report that
without weakening correctness criteria or ranking descriptive means.

Pre-mortem: batched elimination may reorder floating operations; accepted replay
will expose that. Delayed masks could hide an intermediate failure; injected
recovering faults will expose that. A larger batch may compile slowly or allocate
more workspace; record compilation, allocator peaks and worker lifetime. Passing
unit tests alone cannot establish endpoint parity or speed.

## Execution, budget and artifacts

Use `/home/chakwong/anaconda3/envs/tftwogpu/bin/python`. CPU tests explicitly hide
CUDA and are mechanics/reference checks. GPU commands use escalated permissions,
verified growth before device initialization, XLA, and the existing recorded
command-buffer exception if required by the frozen harness.

1. Implement the shared grouping and first-failure diagnostics; add focused tests.
2. Run CPU numerical, flow, analytical-score and proposal regressions. Run focused
   GPU/XLA tests and trace the actual endpoint to verify grouping is wired.
3. Freeze the candidate source. Run the unchanged full frozen replay and analyze
   all rows against candidate02. Then run alternating paired endpoint timing,
   retaining optimized HLO and memory/device provenance.
4. Review terminal results; retain only a correct implementation with an honest
   performance verdict. Update checkpoint/master and budget ledger.

Remaining allocation at start: **4424.451901 GPU-process seconds**, inherited
from the unchanged authorized campaign. This phase cap is 4200 seconds including
failed attempts, tests, worker startup and resident worker lifetimes. Target
allocation: 300 tests/trace, 1700 full replay, 1500 paired timing, 700 repairs or
additional precision. Stop before exceeding the smaller remaining balance.
CPU implementation/tests do not consume GPU-process time. No environment change.

Versioned output root:
`docs/plans/artifacts/ledh-sir-guard-placement-repair-20261022/`.
Save exact executed commands, Git/source identities, environment, hardware,
memory policy, seeds/data identity, durations, checks and result paths in a run
manifest. Preserve failed attempts. Result note must include decision and
inference-status tables, uncertainty, strongest alternative explanation and next
justified action. Record commands before launches and append outcomes afterward.

## Execution record

Plan reviewed before implementation. No GPU time charged at plan creation.

Shared grouping and diagnostics implemented. CPU/XLA focused checks passed:
59 tests in `attempt-01-cpu-tests/pytest.log`, plus two reduced-coordinate
checks in `reduced-coordinates.log`. Broader consumer regressions passed 83
tests and skipped one; the remaining failure was a diagnostic recorder that
did not accept the new nested diagnostics. The recorder repair passed its
unchanged nonlinear Jacobian/determinant assertions. No runtime repair was
needed. These are mechanics checks, not endpoint or performance promotion
evidence. Exact launches and outcomes: `artifacts/ledh-sir-guard-placement-repair-20261022/execution.md`.

### Fixed precision continuation, reviewed before launch

The full 2928-row replay is bitwise identical to the retained checked baseline.
The first eight-block timing comparison does **not** pass the speed criterion:
four pooled intervals include one, and plain/zero-direction on dataset 263043004
has ratio 1.0801, interval [1.0100, 1.1637], a supported regression under the
prespecified diagnostic. Preserve this failed speed result; it is not invalidated
by the continuation.

Run one independent, fixed **48-block** comparison in `attempt-05-precision-timing`.
Use the same frozen sources, replay inputs, four datasets, six consumer cells,
alternating pairing, bootstrap procedure and pass/veto thresholds. Do not pool
the eight-block and 48-block results, inspect intermediate timing estimates,
change the implementation, or stop early for significance. The fresh run has a
2000 GPU-process-second cap, within the 2496.7028 seconds remaining in this
phase after Attempt 04. If incomplete, classify it as incomplete, retain the
first result, and stop; no further precision retry is planned.

Skeptical review: **PASS for a precision diagnostic, not a guaranteed repair**.
The baseline is the retained checked implementation, not the pre-merge path with
unresolved parity. The frozen candidate and exact actual consumer remain under
test. Extra blocks address timing uncertainty only; they cannot repair a real
regression. A separate, block-count-aware artifact analyzer must validate all
warm and timed rows, unchanged sources, single traces, and direct output parity.
Runtime and scientific defaults remain frozen. Both attempts must appear in the
terminal decision, with the independence assumption and limited input scope.

### Completion and terminal review

All four implementation/testing stages are complete. The shared checked
grouping passes 145 distinct CPU tests (one skip), 61 overlapping GPU/XLA tests,
and all 2928 frozen replay rows with bitwise equality and exact validity flags.
Actual endpoints emit grouping diagnostics; no reduced consumer fork was added.

Both timing attempts fail the overall speed criterion. The independent fixed
48-block attempt has no individual regression veto, but both conditional-guide
intervals include one. Plain and physical-mixture pointwise pooled intervals lie
below one on the exposed inputs. Preserve the earlier eight-block regression;
do not pool attempts or promote a uniform speed claim. Retain the correct shared
engineering refactor under stage 4's explicit honest-performance requirement.

Terminal skeptical review: correctness conclusion PASS; overall speed promotion
REJECT. Device scheduling/block dependence and the narrow FP64 scope limit timing
inference. This phase charges 3283.027575 of 4200 GPU-process seconds, including
the completed precision continuation; no further run is planned. Full decision,
inference-status tables and limitations are in
[the terminal result](artifacts/ledh-sir-guard-placement-repair-20261022/result.md).
