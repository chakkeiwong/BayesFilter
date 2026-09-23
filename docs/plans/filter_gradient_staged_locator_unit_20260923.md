# Staged locator numerical enclosure unit

Implement and qualify an internal two-stage XLA program for the existing
`locate_joint_center_staged` algorithm. The original3582b4ac function and its
numerical import closure are the independent authority. Preserve standardized
coordinates, TFP L-BFGS settings, its complete continuation state, global
callback cap, callback incumbent, checkpoint and endpoint replay, exact
selection order and complete records. The checkpoint validator remains one
host call after numerical checkpoint acceptance and before continuation.

Each stage takes initial/scale tensors and, for continuation, the complete
checkpoint/optimizer/accounting state as explicit operands. Resources required
by TFP's callback interface are owned by the program and reset or restored in
each stage. Hold one owner lock across both stages and the validator. Numerical
status, finite checks, incumbent selection and score summaries belong in the
compiled stages. Host result construction only translates completed tensors.
Preserve separately rounded endpoint multiply/add using the qualified
`rounded_affine_position` helper; do not change optimizer arithmetic.

The public endpoint remains unchanged until this internal unit and public
consumer/error/ownership checks pass. The explicit non-JIT wall-clock diagnostic
cannot be embedded; reject that configuration in the internal factory, retain
its existing public route, and later integrate the normal default stages.
Python construction errors and native compilation/runtime errors must remain
distinguishable; no eager retry. This is execution repair, not optimizer tuning.

First bounded evidence unit: at most32 workers/2400 charged seconds within the
unchanged32CPU/52GPU-hour caps,120s per focused case or300s for one combined
check. This covers ten paired cases per device, two existing-consumer groups,
policy checks and localized retries. Cover D1/D3 quadratic/nonquadratic and constant/invalid/cap cases, exact
callback order/counts, validator accepted/rejected/exception behavior,
continuation from the same optimizer state, changed starts/scales, one trace
per stage, unchanged HLO, and actual owner/callback/graph collection. Renew on
GPU when admitted by the existing selector. Every original mismatch, missing
state term or callback discrepancy is a repair trigger at unchanged criteria.

Artifacts use fresh numbered campaign directories and the stable runner's
registered `staged_center_*` groups. Preserve original source hashes and full
records before asserting. GPU memory growth/provenance is mandatory; shared
GPU timing is not cost evidence. This unit cannot close public integration,
costs, native residency, actual DZ5 integration or main-merge gates.

Skeptical review: restarting L-BFGS from checkpoint position is not equivalent
to continuing its history; pass the full namedtuple. Resetting accounting at
continuation would defeat the global cap; restore it from explicit state.
The original incumbent selector uses candidate order, not a fresh sort by
evaluation index; retain initial/callback/checkpoint/endpoint order. The
validator must never be traced or run after numerical checkpoint rejection.
No source edits occur during a worker or matrix.

Continuation after03319: all nine initial CPU comparisons pass, with the
03310 shared-reference lifetime failure preserved and isolated in03311. Add
cap3 at D3, whose original checkpoint uses exactly3 target rows, to force
global-cap exhaustion in continuation. Also execute the seven existing staged
consumer assertion sets through the internal adapter with JIT enabled,
including synthetic better-interior and finite-sentinel endpoints, and the
internal wall-guard rejection. Extend the attempt ceiling to28 within the same
2400-second limit for these two-device checks and local retries; no cumulative
budget, arithmetic, scientific criterion or public route change.

The same reservation also includes one two-case worker per backend for injected
checkpoint/continuation optimizer construction failures. Compare complete
original exception records and exact target counts. Native compilation/runtime
failures still propagate; this cannot justify an eager fallback or claim
arbitrary Python callback compatibility.

Source review also binds the completed `jit_compile` report to the actual
checkpoint function specification. A caller flag cannot label an explicitly
non-JIT diagnostic as compiled. The two-case CPU configuration smoke03323
exercises graph/XLA reporting; the non-JIT arm is a tiny diagnostic only.

Final explicit-state check resumes an original checkpoint on a different owner
whose resources first run an unrelated start/scale. Require complete original
results and exact joined checkpoint/continuation target order. This tests that
no stale implicit counter/incumbent/optimizer state leaks into continuation.
Reserve one worker per backend and raise the attempt ceiling to32 under the
same2400-second ceiling; this includes prior failed/diagnostic/policy workers
and remaining GPU checks. No expansion of cumulative compute is authorized.

Continuation after03327:23 distinct CPU checks pass and129 policy checks renew.
Seventeen workers used426.276688 seconds. Before public wiring, exercise two
remaining failure/ownership risks in one bounded group per backend: reject a
real XLA-incompatible operation in each stage without executing a Python
fallback or proceeding past the failed checkpoint; and run a nested invocation
on the same owner inside the external validator, then compare the outer result
and exact target order against the untouched original. The validator is allowed
to invoke other numerical work; the outer continuation must restore its supplied
optimizer and accounting state after that work.

The injected incompatible operation exists only in the diagnostic test; no
runtime callback or policy waiver is added. A compiler error must propagate,
release the invocation lock, and permit later healthy execution with a fresh
program. This is an internal failure boundary, not a claim of compatibility
with the public legacy optimizer-exception record. Public integration still owes
an explicit runtime-error disposition. The nested original and candidate arms
must execute the same target work and preserve both complete records, validator
calls and ordering. These checks use two additional workers within32/2400;
no numerical tolerance, optimizer setting, source authority or public default
changes. A discrepancy triggers localized diagnosis before integration.
