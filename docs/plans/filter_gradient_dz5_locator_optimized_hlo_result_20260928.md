# Optimized locator compiler comparison result

All three CPU/XLA exports reproduce every callback and complete short result
from their saved controls exactly. The positive replay-int32 arm also matches
the original complete short result; the candidate differs at47 scalar leaves.
This is stronger than the earlier first-score check, but the121 differences
in the full unconverged trajectories remain unresolved.

The optimized compiler comparison identifies a concrete floating-point lead.
Original and replay-int32 each contain26 instances of a214-instruction gradient
fusion with an embedded constant printed as2.4. The candidate contains22 such
instances plus four217-instruction variants. Each variant accepts two separate
copies of that constant as loop inputs and computes two products separately;
the214-instruction version embeds the constant and reuses one product.

The caller reader traces all eight candidate inputs through optimizer-loop
tuple entries143--150. Every entry is initialized from the same printed
`f64[1]{0} constant({2.4})` and returned unchanged by the loop body. The four
variant fusions occur in that body. Their metadata points to
`PartitionedCall_3/gradient_tape/Mul_11/pfor/Mul` and
`PartitionedCall_3/gradient_tape/Mul_17/pfor/Mul` in the frozen historical r1
target. This work adds no derivative engine or pfor authorization, and does
not qualify the old target as the current implementation.

This establishes different optimized floating structure accompanying the
observed score change. It does **not** prove which compiler pass caused it,
which product caused the observed rounding, or which emitted machine code ran.
The installed XLA header `include/xla/service/while_loop_constant_sinking.h`
describes the relevant class of transformation, but its description is not a
pass trace of these executions. A targeted intervention is still necessary
before changing runtime code or compiler settings. Further unrelated counter
dtype trials and unchanged full trajectories are not justified by this result.

| Comparison | Computations matched by printed content and multiplicity | Unmatched left/right | Short-record differing leaves |
|---|---:|---:|---:|
| Original / candidate |17491|89 /86|47|
| Original / replay-int32 |17495|85 /82|0|
| Candidate / replay-int32 |17568|9 /9|47|

Original has17580 computations; candidate and replay-int32 each have17577.
SSA identifiers and source metadata are removed for matching, while operand
relations, printed constants, shapes/layouts, attributes, instruction order and
callee content remain. All three files have3033 opaque constant sites printed
with ellipses. A structural match therefore never certifies hidden literal
equality. Declaration order and unmatched counts alone are not arithmetic
evidence; the inspected caller/constant bindings supply the narrower finding.

| Export | Run / exact saved control | Export seconds | Peak RSS before/after export, KiB |
|---|---|---:|---:|
| Original |04656 /04635|155.388634|13454488 /14653048|
| Candidate |04657 /04636|170.202284|13447552 /14663744|
| Replay-int32 |04658 /04651|155.367761|13435936 /14670140|

Every arm has one trace, three optimizer objective batches and no host
callbacks. Source/snapshot/dispatch, callback bytes and IR hashes pass the saved
controls.04659 passes175 controls/parser/policy checks;04660 passes four caller
reader tests, including misleading tuple-index comments and changed/opaque
invariants.04661 adds complete cross-arm short-record reporting and passes all
179 final checks. The structural and caller readers pass in5.965118 and
1.239885 seconds. Ruff and whitespace checks pass.

The observed export RSS increases are1170.469,1187.688 and1205.277MiB. They are
instrumentation overhead after numerical execution, never production memory
measurements. Export may compile independently; other users' CPU jobs also
ran on the host. Neither export time nor these peaks support performance
ranking, allocator-live-memory claims or production-capacity conclusions.

Evidence is under the existing raw root:

- `run-04656` through `run-04661`, including exact commands, environment,
  frozen dispatch/child source, source hashes, callbacks, short records and IR;
- `terminal-locator-optimized-20260928-r1`, with the tested structural reader,
  computation inventory and three comparisons bound to controls04659;
- `terminal-locator-fusion-20260928-r1`, with the tested caller reader,
  checksum-bound excerpts and all eight invariant traces;
- final complete record comparisons in04661, preserving the earlier04659
  report unchanged; and
- checkpoint archive/receipt `locator-optimized-04661-evidence.tar.gz` and
  `locator-optimized-04661-verification.json` in the branch artifact directory.

This unit closes at6/6 workers and1180.751759/4200 CPU process-seconds,
including both static readers. Global charges are108153.762961 CPU and
96071.347964 GPU seconds;25.957288 CPU and25.313514 GPU hours remain inside
the unchanged56/52-hour caps. No worker remains active. The extra24 CPU hours
are already included.

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Accept diagnostic comparison | All callbacks/short results reproduce exact controls | No source/input/control drift | Independently exported IR may differ from executed lowering | Isolate invariant-product propagation in a bounded intervention | Causal pass or numerical algorithm defect |
| Keep locator repair open | Candidate differs at47 short and121 full-record leaves | Both full optimizers remain unconverged | No portable runtime remedy | Preserve strict failures and test the specific fusion mechanism | Equivalence, convergence, GPU-compatible int32 storage |
| Keep master open | Consumer, numerical and measurement gaps remain | Main merge remains blocked | Registered LEDH migration and terminal dispositions unfinished | Continue the current terminal repair queue | Whole-program completion or scientific admission |

Skeptical result review: the strongest alternative is that broader compiler
context, hidden constants or a different export compilation explains the
association. A controlled intervention that changes these four fusions without
moving the observed score would weaken the proposed mechanism. Exact controls
eliminate accidental replay drift in this experiment; they do not settle that
causal question. The primary agent reviewed the plan, reader and results; no
independent reviewer was used. Runtime source, numerical tolerances, admission
rules, canonical NeuTra architecture and GPU policy remain unchanged.
