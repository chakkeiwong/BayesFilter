# Accounting-width context result

The CPU-only intervention04638 passes in218.497 seconds. Its isolated candidate
differs from frozen source only in11 `tf.int64` to `tf.int32` attributes,
verified by AST. Its first two input, value, score and validity rows match
original04584 and the one-iteration original04635 exactly. It differs from
candidate04585/04636 by the same historical1.5276668818842154e-13 score amount.
Readback/policy04639 passes161 checks in13.996 seconds. This establishes an
accounting-width/compiler-context effect for the controlled CPU fixture.

The narrower intervention04640 passes execution in218.042 seconds. It keeps
all resources and indices int64, changing only the invalid-row reduction to
an int32 sum cast back to int64. The AST check verifies exactly that expression
replacement. Its inputs, values and masks remain unchanged, but its scores
still match the unmodified candidate exactly; it does not restore the original
score. Readback04641, including both interventions and policy, passes162 checks
in18.599 seconds. The four-worker unit closes at469.133/1200 CPU seconds,
with no failed execution and no remaining worker slot in this allocation.

The negative narrower test rejects the explanation that the reduction width
alone suffices. Resource/index widths or their broader compiled context remain
live explanations. A diagnostic test passing means its intervention and
readback were valid; it does not turn a negative repair result into a pass.
The actual runtime module is unchanged. In particular, no int32 GPU resource
path was installed and no full optimizer trajectory was rerun.

| Decision | Primary criterion | Veto/uncertainty | Next action | Not concluded |
|---|---|---|---|---|
| Retain the broad dtype intervention as causal localization evidence | Exact frozen inputs and first scores; AST-only dtype changes | CPU-only;11 attributes change together | Separate resource/index effects or inspect optimized lowering before proposing a runtime change | A GPU-compatible fix or exact compiler operation |
| Reject the single-reduction change as the repair | Controlled inputs/AST and readback pass | Original score is not restored | Preserve the negative result; do not promote the expression change | Full-record equivalence or readiness |
| Keep the master numerical gate open | The initial discrepancy has a smaller positive reproduction | Full trajectories and other strict fields remain unresolved | Continue a newly bounded, reviewed localization/repair unit | Whole-program completion or permission to merge |

Post-run skeptical review: dtype interventions also change allocation and
compiler representation. The evidence identifies their effect, not a wrong
analytical formula. The same first scores across independent historical and
one-iteration controls strengthen the finding; changed input bytes, overflow
or other source edits would undermine it. The weakest boundary is extrapolating
this CPU fixture to GPU or the complete initializer. No independent reviewer
was used and no numerical or scientific criterion was relaxed.
