# Optimized locator compiler comparison

The master is current through04655, but its next optimized-lowering action
needs an executable evidence contract. All new work stays within56 CPU and52
GPU process-hours.26.285275 CPU hours and25.313514 GPU hours remain at entry;
the owner's extra24 CPU hours are already included. One numerical worker.

Question: which optimized computation differences accompany the reproducible
first-score difference in the real one-iteration CPU/XLA locator? Replay
resource storage changed to int32 restores the original first score04651;
int64 storage with int32 increments04652 and removing the resource04654 do
not. No floating-point formula changed, and no runtime remedy is qualified.

Use three isolated copies of the established frozen r1 consumer:
`original` equals the04635 control, `candidate` equals04636, and
`replay_int32` equals04651. The only intervention in the third arm remains the
exact one-resource dtype edit. All use real TFP L-BFGS with max_iterations=1,
the same input/settings/observer and no new compiler flags. Capture text from
`experimental_get_compiler_ir(...)(stage='optimized_hlo')` after the measured
execution and existing unoptimized-HLO capture. Record size/hash/export seconds
and pre/post export host peak RSS. Export overhead is instrumentation, never
production timing or memory evidence.

Required controls before interpretation: each arm's entire callback archive
and complete short result reproduce its own saved control exactly, including
float bit patterns. Verify source/snapshot/dispatch, callback and both HLO
hashes, one trace, three objective batches, no host callbacks and CPU hiding.
Report the original/candidate full-record difference without a tolerance waiver.
If optimized export changes the measured result or a control fails, stop and
diagnose reproducibility; do not attribute differences to the chosen dtype.

Compare optimized computations by content with SSA names/source metadata
removed, preserving operand relations, constants, shapes, layout, instruction
order, operation attributes and called-computation content. Save unmatched
records and opcode/shape summaries. Declaration order alone is not a numerical
difference. Inspect relevant unmatched floating computations and graph callers
before identifying a cause. If the parser meets unsupported syntax, fail and
repair the reader; never silently drop instructions or entire computations.
Parser invariance/change tests must include renaming, callee changes, constant
and operand-order changes, dtype/layout, and unsupported references.

Allocate6 workers/4200 CPU process-seconds: three export probes with900-second
parent/870-second child limits, one300-second saved-evidence/policy readback,
and up to two localized harness/reader checks. Reserve an additional180 seconds
for a standard-library structural read inside this same4200-second unit.
Every actual duration counts once. No full optimizer trajectory or GPU job.
Groups `dz5_locator_optimized_{original,candidate,replay_int32}_cpu` and
`dz5_locator_optimized_readback_cpu`; use the stable campaign runner with
explicit `--device CPU`. Artifacts are new run directories under the existing
raw root; structural results get a unique `terminal-locator-optimized` root.
Existing Python/runner approval signatures suffice; no new allowance required.

Assumption audit: r1 and one iteration are historical-localization baselines,
not current-source admission or optimization success. Their justification is
the exact reproduction of the first historical score difference; full saved
callback/record controls catch context drift. TensorFlow's installed optimized
IR export is diagnostic, and may compile separately; its output alone cannot
certify the machine code used by the observed call. Counters are small and
recorded, excluding overflow in this fixture. CPU is a reference exception.

Skeptical review before execution: the experiment is explanatory, not a
promotion test. Matching normalized HLO does not establish identical compiler
lowering or hardware behavior. Different HLO does not establish which operation
caused the observed rounding. The strongest misleading outcome would be an
apparent integer difference hiding a changed floating fusion; preserve both.
Source/input drift, lost artifacts, timeout and exhausted budget are continuation
vetoes; no localized remedy is a promotion veto only. A credible floating
operation lead triggers a new small intervention plan, never threshold changes
or another unqualified full trajectory. No numerical runtime, canonical NeuTra,
LEDH method, package, HMC, training, global cache or main-merge change. The
primary agent reviewed this plan; no independent reviewer was used.

Prelaunch correction: the established runner accepts900 seconds but not600.
The initial CLI invocation rejected that argument before allocating a run or
starting TensorFlow. Use the existing900-second option and870-second child
limit, with the unit cap explicitly revised to4200 seconds before launch.
This changes only the local diagnostic allocation within the unchanged global
cap. No new timeout option or approval signature is introduced.

Export-format limitation found in04656 before comparison: TensorFlow prints
3033 optimized constants as `constant({...})`. The text therefore cannot prove
literal-payload identity. Preserve/count every opaque constant and label matches
as printed-structure matches only, including callees containing opaque literals.
The reader must test this reporting explicitly. This narrows the structural
claim, not a numerical gate. If a causal question depends on those payloads,
use a separately bounded serialized-IR capture; do not infer their equality
from identical ellipses or silently discard them. The existing frozen-input
and numerical-reproduction controls remain mandatory and unchanged.

Through04659, all three exports reproduce every saved callback and complete
short record exactly;175 readback/parser/policy checks pass. The structural
reader finds only nine unmatched computations on each side of candidate versus
replay-int32. Four are gradient fusions with217 versus214 instructions: two
loop-carried operands replace the shared printed constant2.4. Initial source
inspection traces those operands to duplicated constant tuple entries149/150,
unchanged by the loop. This is a mechanism lead, not yet causal attribution.
Use one of the two already allocated harness checks to verify a small
standard-library caller reader (including misleading tuple-index comments).
Record all affected fusion sites, source excerpts, loop/initial-value bindings
and hashes in a new `terminal-locator-fusion-20260928-r1` directory. Its120-second
limit shares the original180-second structural allowance; the first read used
less than15 seconds. Do not run another numerical trajectory or edit runtime
constants from printed IR. Require a separate bounded intervention before
claiming that changed constant propagation explains observed rounding.
