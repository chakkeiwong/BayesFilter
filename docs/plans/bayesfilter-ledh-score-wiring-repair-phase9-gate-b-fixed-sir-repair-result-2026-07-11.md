# Phase 9 Gate B Result: Fixed-SIR XLA Extraction Repair

Date: 2026-07-11

Status: `REPAIR_VALIDATED_REVIEW_AGREED_FIXED_SIR_GPU_RETRY_AUTHORIZED`

## Decision Table

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Attempt 1 found a fixable fixed-SIR value-adapter XLA extraction defect. The tensor-only covariance repair passes focused CPU-hidden XLA and eager-parity checks. | Repair validation passed locally. Trusted fixed-SIR Gate B has not passed because the post-repair score and FD shards have not run. | Attempt-1 FD emitted a terminal failed artifact before computing any FD value. The score shard is superseded for retry because the reachable source hash changed. | Actual GPU/XLA execution of the repaired FD route and the frozen same-scalar FD tolerance remain unchecked. | Obtain a fresh bounded local substitute-review `VERDICT: AGREE`; only then rerun both frozen fixed-SIR Gate B commands. | No fixed-SIR FD pass, full-row score admission, Gate B completion, Gate C authorization, HMC readiness, posterior correctness, runtime ranking, or scientific claim. |

## Evidence Contract Result

| Field | Result |
| --- | --- |
| Question | Did attempt 1 expose a local XLA extraction defect that can be repaired without changing fixed-SIR target math, transport policy, score recurrence, frozen command settings, or FD thresholds? |
| Exact baseline/comparator | The prepared `transition_covariance` tensor built by `_build_actual_sir_tensors`, which tiles the fixed covariance returned by the same `_dpf_sir_callbacks` callback used by the removed graph-time constructor. Eager value-objective output is the behavioral comparator. |
| Primary criterion | Passed locally: the repaired fixed-SIR value adapter compiles and executes inside `tf.function(jit_compile=True)` with GPU deliberately hidden and matches the eager objective at `atol=rtol=1e-10`. |
| Promotion vetoes | No trusted retry before review; any post-repair source mismatch, non-GPU/non-XLA output, nonfinite value, wrong prepared fingerprint, or FD tolerance failure still vetoes the fixed-SIR Gate B row. |
| Continuation vetoes | The terminal-artifact harness remained valid. The observed defect was bounded and locally repaired, so no shared-harness continuation veto fired. |
| Explanatory only | Attempt-1 score value, score vector, runtime, and tiny reset-memory peak. They are not post-repair evidence and cannot rank or admit the row. |
| Artifact | This result, archived attempt-1 shards/logs, scoped code/test changes, and the pending substitute-review artifact. |

## Claimed And Computed Quantities

| Item | Classification |
| --- | --- |
| Claimed Gate B quantity | Compact fixed-SIR score and central finite differences of the same realized finite-`N` value scalar at `T=1,N=4`, seed `81120`, under trusted GPU/XLA/TF32 execution. |
| Quantity actually computed in attempt 1 | A finite compact score on GPU/XLA. No finite-difference value was computed because value tracing failed in `SpatialSIRSSM.__post_init__` before the first perturbation evaluation. |
| Relationship | The score portion is valid only for the pre-repair source identity. The FD claim is not checked. It would be wrong to classify the failed FD process as a numerical FD mismatch. |
| Repair quantity checked locally | Repaired tensor-only value output inside CPU-hidden XLA compared with the eager objective on identical prepared tensors. They agree at `atol=rtol=1e-10`. |
| Remaining gap | The repaired GPU/XLA score and all-coordinate FD shards have not run and therefore have no trusted post-repair verdict. |

## Attempt 1 Classification

The trusted GPU preflight passed before attempt 1. The fixed-SIR score-only
command then completed with:

- score `[-9.46016788482666, 3.5756328105926514, 5.445666313171387]`;
- objective `-36.35597229003906`;
- reset score peak `80.04736328125 MiB`;
- prepared-input aggregate SHA-256
  `1999831a78622d6de1cec12dfda87d066502399cbe7d605dd257d79745388716`;
- finite output on `/GPU:0`, `float32`, TF32 enabled, and XLA hard-coded.

The matching FD-only command emitted a terminal `failed` artifact. During
`tf.function` tracing, `_make_sir_callbacks_from_scaled_parameters` called
`_dpf_sir_callbacks()` only to reconstruct the fixed process covariance. That
constructor reached `SpatialSIRSSM.__post_init__`, whose eager validation called
`.numpy()` on a `SymbolicTensor`. This is a harness/XLA extraction failure, not
a fixed-SIR FD numerical result and not evidence against the compact score
recurrence or another nonlinear row.

## Evidence Preservation

Attempt 1 was copied before retry to:

- `docs/plans/artifacts/ledh-score-wiring-repair-phase9-gpu-xla/gate-b/attempt-1-fixed-sir-pre-repair/`
- `docs/plans/logs/ledh-score-wiring-repair-phase9-gpu-xla/attempt-1-fixed-sir-pre-repair/`

Binding archived hashes:

| File | SHA-256 |
| --- | --- |
| `fixed-sir-t1-n4-seed81120-score.json` | `36747beaaf728f524501d808333170d05061cee013c5a8cda57400b66645fadd` |
| `fixed-sir-t1-n4-seed81120-score.md` | `7d79733be3d8051ceb4d56f03a026c0066cc950a9cd07194cea1154fdbc23c0f` |
| `fixed-sir-t1-n4-seed81120-fd.json` | `284bc382c45aa1a4bd10faecd3910ea83f8a6c742e35c594c1152ff7ff5592f6` |
| `gate-b-fixed-sir-t1-n4-seed81120-score.log` | `477ea3b112ac1fe7b34a00284dab7617e4cfdddf06f8f2042845a3782a50beda` |
| `gate-b-fixed-sir-t1-n4-seed81120-fd.log` | `2d7d1c9dc4dc1db9de55e9cfc890036b96fa3f12aef3a29cfc60477250d7e871` |

The live exact-command paths intentionally remain unchanged. Both live shards
must be rerun after review because reachable fixed-SIR source content changed;
mixing the old score with a new FD shard is forbidden by the source-hash and
score-reference validators.

## Repair

Changed only
`docs/benchmarks/benchmark_p8p_parameterized_sir_gradient.py::_make_sir_callbacks_from_scaled_parameters`:

```python
transition_covariance = tf.cast(tensors["transition_covariance"], DTYPE)
process_chol = tf.linalg.cholesky(transition_covariance[0])
```

`_build_actual_sir_tensors` constructs this tensor before XLA tracing by
tiling the fixed covariance from `process_noise_covariance_fn`. Selecting the
first batch entry therefore preserves the existing shared fixed covariance
expected by `tf.einsum("bnd,ed->bne", noise_tensor, process_chol)`. No target,
parameter transformation, state transition, process-noise policy, transport
setting, score equation, command, threshold, or public API changed.

`SpatialSIRSSM.__post_init__` was not modified because its eager validation is
broader repository behavior and is unnecessary inside the prepared-tensor
adapter.

## Regression Coverage

Added two focused tests in
`tests/highdim/test_ledh_compact_score_gpu_xla_harness.py`:

- a source guard requiring the callback helper to contain neither
  `_dpf_sir_callbacks` nor `.numpy`, and to consume prepared
  `transition_covariance`;
- an actual CPU-hidden `tf.function(jit_compile=True)` execution of the tiny
  fixed-SIR value adapter, with eager-objective and log-likelihood-mean parity
  at `atol=rtol=1e-10`.

## Local Checks

All TensorFlow checks intentionally set `CUDA_VISIBLE_DEVICES=-1` before import.
They are engineering/wiring evidence only, not trusted GPU Gate B evidence.

| Check | Result |
| --- | --- |
| Focused new repair tests | `2 passed, 76 deselected, 2 warnings in 13.80s` |
| Shared harness plus fixed-SIR contracts | `97 passed, 2 warnings in 92.99s` |
| Combined harness, Phase 8 cross-model schedule, and shared score contract | `151 passed, 2 warnings in 27.58s` |
| LGSSM/fixed-SIR model-specific shard | `53 passed, 2 warnings in 92.11s` |
| `py_compile` for repaired helper, shared runner, and focused tests | Passed |
| Frozen exact-command manifest `--check` | Passed; manifest remains current |
| `git diff --check` | Passed |
| Final review-bound combined suite | `152 passed, 2 warnings in 35.93s` |
| Final repair-review path/hash suite | `34 passed, 45 deselected, 2 warnings in 12.70s` |

## Run Manifest

| Field | Value |
| --- | --- |
| Git commit | `d269f5bbd8531b878d4f25897a357fbc8f172488` plus the scoped uncommitted repair and pre-existing dirty worktree |
| Commands | CPU-hidden pytest, `py_compile`, generator `--check`, and `git diff --check` commands summarized above |
| Environment | `/home/chakwong/anaconda3/envs/tf-gpu/bin/python`; TensorFlow `2.19.1` |
| CPU/GPU status | Repair validation deliberately CPU-only with `CUDA_VISIBLE_DEVICES=-1`; attempt-1 trusted GPU provenance is preserved in its archived artifacts |
| Data version | Prepared fixed-SIR tensors from the admitted 2026-07-07 source value route |
| Random seeds | Tiny seed `81120` |
| Wall time | `13.80s`, `92.99s`, `27.58s`, and `92.11s` for the four pytest runs |
| Output artifacts | This result plus archived attempt-1 JSON/Markdown/log artifacts |
| Plan file | `docs/plans/bayesfilter-ledh-score-wiring-repair-phase9-gpu-score-memory-subplan-2026-07-10.md` |
| Result file | This file |

## Inference Status

| Item | Status |
| --- | --- |
| Hard veto screen | Attempt-1 FD failed engineering extraction and cannot pass Gate B. The bounded defect is repaired locally, but the trusted retry remains unchecked. |
| Statistically supported ranking | None; no ranking was attempted or supported. |
| Descriptive-only differences | Attempt-1 score, runtime, and tiny peak are descriptive and superseded for post-repair admission. |
| Default-readiness | No new conclusion. The owner-directed production target remains policy, while this row has not passed Gate B. |
| Next evidence needed | Fresh repair review, then both exact fixed-SIR Gate B score-only and FD-only commands under trusted GPU/XLA. |

## Post-Run Red Team

- Strongest alternative explanation: the CPU-hidden XLA parity test may miss a
  GPU-only compile or numerical failure.
- Result that would overturn the repair decision: the exact trusted FD retry
  reaches another graph-unsafe constructor, emits nonfinite values, mismatches
  the score shard's prepared-input fingerprint, or fails the frozen FD rule.
- Weakest evidence: no post-repair GPU/XLA shard exists yet.

## Review Boundary

Claude remains policy-blocked as external repository disclosure; do not retry
or work around that boundary. The first local substitute review returned
`VERDICT: REVISE` because post-repair shards would have bound only the older
Gate A authorization. The runner now records and validates the path and SHA-256
of the dedicated repair review; adversarial path/hash tests pass. Fresh
iteration-2 re-review returned `VERDICT: AGREE`.

That verdict authorizes rerunning both exact fixed-SIR Gate B commands, score
first and FD only if score passes its hard checks. If both pass, the existing
Gate A authorization resumes for the remaining nonlinear Gate B rows. Gate C,
Gate D, aggregation, and LGSSM remain blocked pending the complete Gate B result
and its separate review.
