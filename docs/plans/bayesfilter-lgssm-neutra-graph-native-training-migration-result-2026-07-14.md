# LGSSM NeuTra Graph-Native Training Migration Result

Date: 2026-07-14

Plan:
`docs/plans/bayesfilter-lgssm-neutra-graph-native-training-migration-plan-2026-07-14.md`

## Verdict

**PASS for the active LGSSM NeuTra training process.**

The claimed target was a repository-owned training path with no per-step Python
loop and no repository-owned NumPy import/call in its live closure. The quantity
actually executed was a two-step exact-target reverse-KL dense-IAF optimization
inside one `tf.function(jit_compile=True)` program containing a TensorFlow
`While` operation. The active CLI and the legacy public training entry point now
both route to this strict implementation.

This establishes engineering mechanics only. It does not establish 500-step or
5,000-step training success, transport quality, recipe ranking, posterior
correctness, HMC convergence, speedup, or production/default readiness.

## Implementation

- `bayesfilter/inference/neutra_training.py` now invokes one compiled training
  program per call. `tf.while_loop` advances every requested optimization step;
  stateless seeds, reverse-KL objective, clipping, and manual Adam equations are
  unchanged.
- Per-step `.numpy()` synchronization and the Python loop over optimization
  steps were removed. Cadence-selected diagnostics are collected in tensors and
  materialized once after the compiled program returns.
- Checkpoint semantics are now
  `terminal_only_graph_native_v1`. JSON file I/O occurs only after the XLA
  program completes; no `tf.py_function` or `tf.numpy_function` is used.
- `bayesfilter/testing/lgssm_neutra_strict_training_tf.py` isolates the exact
  target, geometry validation, training, frozen transport checks, held-out
  evaluation, and repository import-closure audit from the legacy NumPy-heavy
  HMC campaign.
- The CLI dispatches `train` before importing the legacy campaign. The legacy
  public `run_gpu_training_job` delegates to the strict route, closing the
  alternate-entry bypass.
- Inference/testing public package exports are lazy so focused TensorFlow modules
  do not eagerly import unrelated NumPy-backed reference/HMC modules.

## Attempt Ledger

| Attempt | Classification | Evidence | Repair |
| --- | --- | --- | --- |
| 1 | Training graph completed and terminal checkpoint/frozen payload were written, but the process ended before a result artifact during post-training exact-target parity compilation | `docs/benchmarks/artifacts/lgssm_neutra_graph_native_training_migration_2026_07_14/` | Write a durable training-completion record before validation and replace redundant exact-target parity recompilation with transport-only forward/logdet/pullback parity |
| 2 | Passed | `docs/benchmarks/artifacts/lgssm_neutra_graph_native_training_migration_2026_07_14_attempt2/smoke/candidates/source_anchor_lr5e3/attempt_1_graph_native/result.json` | None |

Attempt 1 is preserved and is not promoted. Attempt 2 used a fresh versioned
root and the same target, recipe, seed family, hardware class, and two-step
debug budget.

## Passing Evidence

Attempt-2 result SHA-256:
`6753614f77115368ba401cf3cca72638f7b77249ae915128fccf73b95c72f338`

| Check | Result |
| --- | --- |
| Training-program host invocations | `1` |
| Compiled control flow | `tf_while_loop`; graph operation inventory contains `While` |
| Program steps | `2`, records `[1, 2]` |
| Checkpoint policy | `terminal_only_graph_native_v1` |
| Terminal checkpoints | Exactly `checkpoint_step_000002.json` plus `checkpoint_latest.json`; no periodic step checkpoint |
| Target value/status | Both steps finite, status valid, nonvalid count `0`, floor count `0` |
| Device/XLA | TensorFlow `2.19.1`, RTX 4080 SUPER `/GPU:0`, float64, XLA compilation logged |
| Frozen transport parity | transport `0.0`, logdet `0.0`, pullback score `0.0`, logdet score `0.0` max absolute differences |
| Repository closure | `19` imported `bayesfilter.*` modules, zero NumPy import/call or TensorFlow host-callback violations |
| Wall time | `41.34 s` compile plus training and post-training validation |

TensorFlow itself depends internally on NumPy. That third-party implementation
detail is not a repository-owned NumPy computation and is explicitly outside
the source-policy claim.

## Local Checks

- `25 passed` in
  `tests/test_lgssm_neutra_target_specific_protocol.py`.
- `57 passed` in `tests/test_common_inference_runtime_contracts.py` and
  `tests/test_fixed_trajectory_hmc_tuning.py` after lazy-export compatibility
  repair.
- `35 passed` in the exact-target, dense-IAF artifact-loader, and batched
  value/score dependency suites.
- All `13` tests in `tests/test_neutra_training.py` passed when split across
  bounded XLA processes: `4 + 4 + 3 + 2`. Running all cases in one process was
  terminated after eight dots without a pytest summary, consistent with
  cumulative CPU-XLA compilation memory; no assertion failure was observed.
- Python compilation passed for all changed Python modules and tests.
- `git diff --check` passed.
- Static search found no per-step Python loop, `compiled_step`, direct NumPy
  import, `tf.py_function`, or `tf.numpy_function` in the strict training engine,
  harness, or CLI.
- A fully loaded strict closure audit passed for all imported repository modules.

## Review Status

The local skeptical audit found and repaired three material plan flaws before
and during execution:

1. package initializers could import NumPy transitively;
2. periodic host-side checkpointing was incompatible with one uninterrupted
   XLA call; and
3. requiring `numpy` to be absent from `sys.modules` was impossible because
   TensorFlow itself imports NumPy.

Claude's fixed-token health probe returned `CLAUDE_PROBE_OK`, but two bounded
single-path substantive review prompts returned empty output. No Claude verdict
is claimed. Execution proceeded under the repository's review-proportionality
rule after the local audit was revised to address the material issues.

## Decision Table

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Admit strict training mechanics for the next target-specific screen | Passed: one XLA call, `While`, no repository NumPy closure | No loop, NumPy, target, finite-state, parity, device, or XLA veto fired | Terminal-only checkpointing sacrifices mid-call crash recovery; only two live GPU steps tested | Run a fresh full five-step strict smoke for each recipe, then the authorized 500-step screen arms if all pass | Training quality, HMC readiness, posterior correctness, speedup, or recipe superiority |

## Post-Run Red Team

Strongest alternative explanation: the two-step smoke may pass while a 500- or
5,000-step graph fails from compile size, device memory, target instability, or
the absence of mid-run checkpoint recovery. The result that would overturn this
engineering admission is any full five-step recipe smoke that imports a
NumPy-backed repository module, requires a host callback, lacks graph control
flow, leaves GPU/XLA, produces invalid target status, or cannot write a terminal
artifact. The weakest evidence is long-budget resilience; no long job was run
under this migration plan.
