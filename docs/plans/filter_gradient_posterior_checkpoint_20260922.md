# Posterior curvature and shared solver checkpoint

This is partial execution-repair evidence on
`repair/filter-gradient-xla-validation-20260918`, based on `96e15e9d`.
The native fixed-center posterior controller is implemented and tested but is
not connected to the public endpoint. One rejected ill-conditioned diagnostic
comparison remains unresolved (renewed CPU failure 02456), and the repository-wide campaign is unfinished.
Main remains unmerged.

The controller uses stable TensorFlow signatures and native loops for ordered
partitions, padded batches, replicate fitting and every pairwise precision
comparison. It preserves the original Philox box, ball and Gaussian proposal
streams, fixed center, analytical target callback, fit decisions and untouched
holdouts. Host loops format only completed records. Center, factor and seed stay
runtime operands. A one-entry cache compares callbacks by identity; tests check
changed target resources and release of Python graphs/callbacks.

Two boundary defects were found during qualification. The shared native COD
assigned rank two to an exact 33-by-3 rank-one design because its combined
Householder reduction introduced a pivot above the original threshold. Matching
Eigen's essential-tail dot followed by leading-row addition restores rank one
and the original minimum-norm solution, without changing the threshold or
pullback convention. The separate GPU posterior zero-design mismatch came from
one ULP between the center's vector product and the batch matrix product. Using
the same multiply/reduce contraction preserves the formula and restores the
original zero relative residual. There is no residual floor or special-case
zeroing. Failed alternatives and unmodified records remain in the numbered runs.

CPU shared-solver qualification passes 159 focused checks and 276 renewed
consumer checks (02382--02402). Public uniform, quadratic-center, paired and
batch GPU consumers pass 132 checks (02403--02406). Native posterior renewal
after the projection repair passes 78 numerical checks per device and 72
policy/controller checks per matrix (02421--02428 and 02435--02442). These include
complete original D1/D3/D5 records, rejection/count/order checks, original random
streams, independent Gaussian orientation, malformed callbacks, finite-score
overflow, callback ownership and changed-input HLO. They exclude the separately
mandatory ill-conditioned comparison. Shared COD GPU renewal initially stopped
at GPU3 contention before any worker launched. Contention cleared at recovery; 02477--02479 pass 34 solver/active-row,
28 dense-consumer and 14 boundary checks. The matrix then stopped at renewed
GPU3 contention before the next worker launched. Derivative, factor, lifecycle
and remaining-consumer groups still require renewal.

The original source closure is `3582b4ac`, including its original graph dense
solver. New tests never substitute the recent graph implementation for that
authority. Diagnostic counters use colocated int64 TensorFlow resources because
int32 variables are pinned to the CPU; the initial GPU placement failure is
preserved in 02408. NumPy and mpmath occur only in independent test/reference code.
The static guard remains explicitly partial: 206 sources and 1,298 exact
exceptions; new numerical modules have no Python-loop exemption.

## Native costs and memory

Twelve fresh processes compare original, graph and XLA at D3/D5 on CPU and
GPU3 (02429--02434 and 02443--02448). Every initial and changed-input complete
record passes. Each process includes 20 warm calls and full result formatting;
total cold includes construction, tracing and first execution. Source and input
checksums match across arms. These are single-process descriptive measurements,
not the three-repeat public or terminal timing evidence.

| Device / dimension | Original warm ms | Graph warm ms | XLA warm ms | Original / XLA total cold s | Extra XLA host RSS MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| CPU / 3 | 99.073 | 5.279 | 1.080 | 0.149 / 3.182 | 377.6 |
| CPU / 5 | 96.710 | 5.640 | 1.617 | 0.139 / 3.133 | 381.4 |
| GPU3 / 3 | 165.886 | 47.220 | 5.481 | 1.428 / 7.208 | 141.6 |
| GPU3 / 5 | 173.857 | 55.001 | 9.716 | 1.418 / 7.624 | 140.4 |

The cold-time trigger fires on both devices, and the additional-host-memory
trigger fires on CPU. GPU allocator peaks fall from roughly 8.4 MB original to
48,384/56,832 bytes XLA. That does not cancel the retained compilation and host
costs. Graph mode is an explicit non-default reference, not an alternative
production default.

Six follow-ups (02450--02455) test 3,000 alternating-input full calculations
at 2/4/8 replicates on each device. Full original records, one trace, unchanged
HLO and all three runtime operands pass. Graph counts are 2,733/2,737/2,737;
HLO stays about 1.98 MB. CPU RSS after first execution is 962 MiB across capacities,
with 4--12 KiB growth over the final 1,000 calls. GPU host RSS after first execution
is 1,215--1,217 MiB, with 4--8 KiB final-interval growth. Live GPU allocator use
stays 5,376 bytes; peaks remain stable within each scope at 58,112/69,376/93,184 bytes.
The initially proposed one-replicate fixture was invalid under the existing
configuration; 02449 preserves that pre-execution harness failure. The corrected
capacities obey the existing physical-row budget.

The observations locate most extra host allocation at tracing/first execution
and show near-flat bounded warm reuse, with no capacity-dependent graph
unrolling. They do not establish exact process peak memory, arbitrary target
turnover, native executable eviction or general leak freedom. CPU compilation
cost remains a recorded tradeoff, and larger real-target scope remains open.

Artifacts are under the shared
`docs/plans/artifacts/filter-gradient-repair-20260917/` root. Analyses:
`posterior-native-cpu-gpu-costs-02448.json` and
`posterior-native-capacity-memory-02455.json`. Exact analysis scripts, source
checksums, commands, environment, GPU preflight/growth records and elapsed times
are retained. The cost analyzer also reproduces all six historical CPU arm
summaries from 02346--02363 before analyzing new runs.

## Renewed public uniform CPU costs

After the shared solver correction, 02458--02475 renew all 18 public uniform
cost processes: three fresh-process repeats per original/graph/XLA arm at each
dimension. Every complete initial and changed-input record passes. The source
closures and prepared inputs match across arms. Results below report the median
of the three process medians, followed by their observed range in parentheses.

| Dimension | Original warm ms | Graph warm ms | XLA warm ms | Original / XLA cold s | Extra XLA observed host RSS MiB |
| --- | ---: | ---: | ---: | ---: | ---: |
| 3 | 441.018 (436.606–442.563) | 48.779 (46.777–49.481) | 26.440 (26.263–27.030) | 0.495 / 4.387 | 521.6 |
| 5 | 446.886 (441.725–452.662) | 57.995 (57.830–58.891) | 35.477 (35.369–36.683) | 0.504 / 4.411 | 524.6 |

The cold-time and additional-host-memory triggers persist. These are descriptive
endpoint costs, not statistical timing superiority or whole-DZ5 qualification.
GPU renewal remains required. Exact repeat values and provenance are in
`uniform-public-cpu-costs-02475.json`. Earlier uniform costs remain evidence for
their archived source only.

Recovery audit 02476 passes with 2,969 working Python files, 2,968 parsed and the
unchanged vendor-reference parse error. Updated charges are
51,000.71724355132 CPU and 48,325.1067211973 GPU seconds through 02480,
within the unchanged 32/52 process-hour caps. Policy/controller 02480 passes
72 checks, focused Ruff and whitespace pass, and no numerical worker is active.

## Review and remaining evidence

The ill-conditioned rejected precision discrepancy remains a strict failure.
The [comparison proposal](filter_gradient_rejected_dense_precision_decision_20260922.md)
is not installed. Its existing 100/160-digit reference covers the captured CPU
design only; CPU and GPU sine generation differ in 48 binary64 entries. Within
each run the original and candidate use identical inputs, but a CPU reference
cannot silently qualify the separate GPU design. Public admission remains
blocked until its exact comparison contract and applicable references pass.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Retain internal candidate and continue qualification | Complete ordinary/boundary records and bounded reuse pass | Rejected ill-conditioned comparison remains open | GPU captured-input reference; complete public integration and costs | Renew shared-solver GPU consumers, resolve the rejected diagnostic, then qualify the public endpoint | No complete repair, merge, posterior or HMC readiness |
| Preserve compilation tradeoff for final review | Matching numerical records and stable compiled inputs | CPU host and both cold-time triggers remain recorded | Larger targets and native executable retention on target changes | Test actual target scope and final repeated public costs | No general leak freedom or timing superiority |

Primary-agent review only. The strongest alternative explanation for apparent
speed gains is omitted setup/reporting; measurements include completed reporting
and separately preserve full cold costs. The weakest evidence is workload
coverage: small fixed Gaussian fixtures and a bounded reuse horizon do not
qualify actual DZ5 targets or all repository consumers. Public sequential and
block/iterative control, external deadlines, F01--F20 dispositions and final
source-frozen tests/comparisons remain in the master program. Budgets remain
32 CPU / 52 GPU process-hours; this checkpoint grants no new compute or tolerance.
