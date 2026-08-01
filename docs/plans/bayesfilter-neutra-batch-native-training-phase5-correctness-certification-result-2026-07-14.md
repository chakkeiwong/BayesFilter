# Phase 5 Result: Batch-Native Correctness Certification

Date: 2026-07-14

## Outcome

**PASS_PHASE5_AND_CONTINUE_TO_TRUSTED_GPU_PERFORMANCE.** The exact batch-native
target is numerically/status consistent with the scalar authority, correctly
connected to the reverse-KL objective, deterministic under fixed seeds, and
reproducibly identified through repository-derived dependency closure.

## Certification Matrix

| Ledger | Evidence | Status |
| --- | --- | --- |
| Batch capability identity | v2 binding hashes direct method, direct repository callables, owning modules, and live helper identity | pass |
| Adversarial identity | forged closure and live helper replacement rejected before helper execution | pass |
| Eager mathematical parity | value/score/status near machine precision | pass |
| CPU-XLA execution parity | same-regime scalar/batch near machine precision | pass |
| Invalid rows | status `2`, NaN posterior value/score, valid rows isolated | pass |
| Objective gradient | reviewed exact score bridge matches central finite difference | pass |
| One-step update | real exact target reaches generic optimizer with valid status | pass |
| Five-step state | fresh identical-seed runs have identical variables and Adam moments | pass |
| Graph topology | training and target use TensorFlow loops; no sample map/callback in active path | pass |
| Checkpoint provenance | JSON-round-tripped binding payload equals in-memory payload | pass |
| Final focused matrix | `34 passed`, `7 passed`, and `19 passed` in isolated processes | pass |

## Repairs During Phase 5

1. TensorFlow-decorated batch kernels were initially omitted from helper
   discovery because they are not plain Python functions. The closure now
   accepts inspectable repository-owned callables and records callable/module
   sources plus live identity.
2. The manual XLA target initially received a watched tensor before score
   injection, causing TensorFlow to construct an unwanted internal backward
   graph and fail XLA conversion. The target is now invoked on
   `tf.stop_gradient(x)` inside `@tf.custom_gradient`; the reviewed score is the
   only derivative with respect to the original input.
3. Nonfinite raw proposals initially aborted the stationary Lyapunov solve
   before graph-status handling. The adapter now substitutes a benign finite
   row only for internal completion, forces status `2`, and NaN-gates the final
   posterior value/score while leaving valid rows unchanged.
4. Tuple/list normalization made checkpoint provenance compare unequal in
   memory versus JSON. Public binding payload collections are now JSON-native.

## Decision Table

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Admit correctness boundary | all numerical, status, gradient, state, graph, and identity ledgers pass | no correctness veto | trusted GPU FP64 performance and memory | run bounded GPU ladder | no transport quality or HMC claim |
| Keep training recipe unpromoted | five-step run is mechanics only | no engineering failure | optimization stability and downstream quality unknown | Phase 6 timing, then Phase 7 protocol | no recipe ranking/default |

## Inference Status

| Item | Status |
| --- | --- |
| Hard veto screen | no correctness hard veto remains |
| Statistically supported ranking | not applicable; no stochastic method ranking |
| Descriptive-only differences | CPU eager/XLA ordering and run time |
| Default-readiness | correctness-ready only; performance and downstream evidence outstanding |
| Next evidence needed | trusted GPU target/training performance and stability ladder |

## Handoff

Phase 6 starts under
`docs/plans/bayesfilter-neutra-batch-native-training-phase6-trusted-gpu-performance-subplan-2026-07-14.md`.

