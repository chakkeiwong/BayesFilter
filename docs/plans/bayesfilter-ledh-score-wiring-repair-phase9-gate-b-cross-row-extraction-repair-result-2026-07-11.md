# Phase 9 Gate B Result: Cross-Row XLA Extraction Repair

Date: 2026-07-11

Status: `REPAIR_VALIDATED_REVIEW_AGREED_GATE_B_RETRY_AUTHORIZED`

## Decision Table

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Two Gate B attempts exposed graph-time eager-model construction in fixed-SIR value extraction and predator-prey score extraction. Both are repaired without changing target or score math, and all five nonlinear score/value adapters now compile under CPU-hidden XLA. | Local repair criterion passed. A common post-repair trusted Gate B set has not run. | Fixed-SIR attempt 1 FD failed before computing FD; predator-prey attempt 1 score failed before computing a score. The successful intermediate fixed-SIR retry is superseded for final aggregation because runner/review identity changed. | GPU-specific compilation and frozen FD behavior for the final common source/review identity remain unchecked. | Obtain a fresh bounded substitute-review `VERDICT: AGREE`, then rerun all ten exact nonlinear Gate B commands sequentially by row. | No complete Gate B pass, full-row score admission, Gate C authorization, HMC readiness, posterior correctness, runtime ranking, or scientific claim. |

## Evidence Contract Result

| Field | Result |
| --- | --- |
| Question | Can the repeated eager-constructor extraction defect be closed across all nonlinear Gate B adapters without changing baselines, target scalars, compact recurrences, frozen commands, or thresholds? |
| Exact baselines/comparators | Fixed-SIR uses the prepared covariance built from the same fixed callback covariance. Predator-prey uses its established `PredatorPreySSM` schedule, `delta=2.0` and `20` RK4 substeps, obtained once at eager module initialization. Every compiled score/value output is compared with the same prepared-input eager adapter output. |
| Primary criterion | Passed locally: all five nonlinear score and value adapters execute inside `tf.function(jit_compile=True)` with GPU intentionally hidden and match eager outputs at `atol=rtol=1e-10`. |
| Promotion vetoes | No GPU retry before review; failed terminal artifact, source/review mismatch, non-GPU/non-XLA output, nonfinite value, prepared-input mismatch, memory failure, or frozen FD failure still vetoes the affected row. |
| Continuation vetoes | Neither failure invalidated terminal artifacts, prepared input construction, target identity, or shared runner logic. Both were bounded extraction repair triggers. |
| Explanatory only | Intermediate fixed-SIR score/FD values, runtimes, and tiny peak. These do not admit a row and cannot rank methods. |
| Artifact | This result, archived attempt artifacts/logs, scoped source/tests, and the pending cross-row substitute review. |

## Claimed And Computed Quantities

| Item | Classification |
| --- | --- |
| Claimed Gate B quantity | For each nonlinear row, a compact score and central finite differences of the same realized finite-`N` scalar under trusted GPU/XLA/TF32 at the frozen tiny shape. |
| Fixed-SIR attempt 1 | Score computed; FD not computed because value tracing reached eager `.numpy()` validation. Extraction failure, not numerical FD evidence. |
| Fixed-SIR intermediate retry | Score and FD computed on trusted GPU/XLA. Frozen OR rule passed by relative error: `max_abs_error=0.15147781372070312 > 0.01`, `max_relative_error=0.017033180221915245 <= 0.05`. This proves the fixed-SIR repair worked under that intermediate identity but is not part of the final common Gate B set. |
| Predator-prey attempt 1 | No score computed because score JVP tracing constructed `PredatorPreySSM` and reached eager `.numpy()` validation. Its FD process did not run. |
| Quantity checked after both repairs | CPU-hidden XLA score/value outputs for fixed-SIR, predator-prey, actual-SV, generalized-SV, and KSC-SV, each compared with its eager prepared-input adapter. |
| Relationship | Compiled and eager adapter outputs agree at `atol=rtol=1e-10`. Trusted final Gate B evidence remains not checked until all live shards are rerun. |

## Failure And Repair Ledger

| Row/attempt | Failure | Repair | Classification |
| --- | --- | --- | --- |
| fixed-SIR attempt 1 | `_make_sir_callbacks_from_scaled_parameters` called `_dpf_sir_callbacks()` during tracing only to reconstruct fixed process covariance. | Cholesky the already-prepared `transition_covariance[0]`. | Tensor-only fixed adaptation; no target or transition-math change. |
| predator-prey attempt 1 | `_predator_prey_transition_mean_jvp_tf` constructed `PredatorPreySSM` during tracing only to read its fixed RK4 schedule. | Instantiate once during eager module initialization and freeze `delta` plus substep count for the score, value-only, and historical transition helpers. | Extraction timing repair; schedule remains exactly `2.0 / 20 = 0.1`. |
| remaining three rows | No failure in proactive CPU-hidden XLA audit. | None. | Actual-SV, generalized-SV, and KSC-SV score/value adapters already compile. |

`SpatialSIRSSM.__post_init__` and `PredatorPreySSM.__post_init__` were not
modified. Their eager validation remains broader repository behavior.

## Evidence Preservation

Fixed-SIR attempt 1 remains under `attempt-1-fixed-sir-pre-repair`. Predator-prey
attempt 1 is archived under `attempt-1-predator-prey-pre-repair`:

| Archived artifact | SHA-256 |
| --- | --- |
| predator-prey failed score JSON | `d3c374454611791b8fd674bca02510fe0ac53a79692e5128805a524c115fcf5c` |
| predator-prey failed score log | `70ab8b2e79933a0c5a8851950fd5584e11662dbb4db2a1246685e43250858d37` |

The successful intermediate fixed-SIR retry is archived under
`attempt-2-fixed-sir-post-repair-pre-cross-row-review`:

| Archived artifact | SHA-256 |
| --- | --- |
| fixed-SIR score JSON | `48a3d49b0929245e0b4f833c511c997d072613e512d2f048bc891fece84ee7bc` |
| fixed-SIR FD JSON | `c4fcb85207d3b27085e648468019095bf0432512df5cabb0c1d06a7d4ff77d3d` |
| fixed-SIR score log | `eef8b5b7120198a89d00cee91be04a4dc5b01d40b3f11759f42d12d474bd842b` |
| fixed-SIR FD log | `19aa22fa20ac4519e652ab6e15f0767531087fd873e6cd072680aa31a8877cae` |

All live Gate B paths must be regenerated after review so the final set shares
one runner source and repair-review hash. No archived shard may be mixed into
the final set.

## Regression Coverage

- Fixed-SIR graph-safe callback source guard and compiled value/eager parity.
- Predator-prey source guard covering score JVP, value-only transition, and
  historical transition helpers.
- Parameterized all-five-row compiled score/value parity under
  `tf.function(jit_compile=True)`.
- Existing per-row eager score/value and same-scalar finite-difference checks.
- Existing governance, exact-command, device, trust, memory, and artifact
  adversarial tests.

## Local Checks

All TensorFlow commands intentionally set `CUDA_VISIBLE_DEVICES=-1` before
import. They are engineering evidence only.

| Check | Result |
| --- | --- |
| Proactive actual-SV score/value XLA probe | Passed |
| Proactive generalized-SV score/value XLA probe | Passed |
| Proactive KSC-SV score/value XLA probe | Passed |
| New graph-safe plus all-row XLA tests | `6 passed, 79 deselected, 2 warnings in 73.21s` |
| Predator-prey model contract | `22 passed, 2 warnings in 128.77s` |
| Binding combined harness/cross-model/shared contract | `158 passed, 2 warnings in 98.73s` |
| Syntax check | Passed |
| Frozen exact-command manifest `--check` | Passed; literal commands unchanged |
| `git diff --check` | Passed |
| Final review-bound all-row XLA/governance subset | `40 passed, 45 deselected, 2 warnings in 77.51s` |

## Run Manifest

| Field | Value |
| --- | --- |
| Git commit | `d269f5bbd8531b878d4f25897a357fbc8f172488` plus scoped uncommitted repairs and the pre-existing dirty worktree |
| Commands | CPU-hidden XLA probes/pytest, `py_compile`, manifest `--check`, and `git diff --check` summarized above |
| Environment | `/home/chakwong/anaconda3/envs/tf-gpu/bin/python`; TensorFlow `2.19.1` |
| CPU/GPU status | Repair checks intentionally CPU-only; trusted intermediate fixed-SIR and failed predator-prey provenance preserved in archived artifacts |
| Data version | Admitted nonlinear source value artifacts dated 2026-07-07 |
| Random seeds | Tiny seed `81120` |
| Wall time | `73.21s`, `128.77s`, and `98.73s` for the binding pytest runs |
| Output artifacts | This result plus archived Gate B JSON/Markdown/log artifacts |
| Plan file | `docs/plans/bayesfilter-ledh-score-wiring-repair-phase9-gpu-score-memory-subplan-2026-07-10.md` |
| Result file | This file |

## Inference Status

| Item | Status |
| --- | --- |
| Hard veto screen | Failed attempts are invalid Gate B shards. Both bounded extraction defects pass local repair checks; final trusted row screens remain pending. |
| Statistically supported ranking | None; no ranking was attempted or supported. |
| Descriptive-only differences | Intermediate scores, FD errors, runtimes, and tiny memory are descriptive only. |
| Default-readiness | No new conclusion beyond the existing owner-directed production policy. |
| Next evidence needed | Fresh cross-row review, then all ten exact Gate B commands under one source/review identity. |

## Post-Run Red Team

- Strongest alternative explanation: CPU XLA may accept an operation that GPU
  XLA rejects or evaluates differently.
- Result that would overturn the repair decision: any final exact score or FD
  process reaches another graph-unsafe constructor, emits nonfinite/wrong-device
  output, mismatches prepared inputs, or fails its frozen FD rule.
- Weakest evidence: only fixed-SIR has an intermediate trusted post-repair pass;
  the final common Gate B set has not run.

## Review Boundary

Claude remains policy-blocked as external repository disclosure. Fresh local
cross-row substitute review returned `VERDICT: AGREE`. Its exact bytes are now
governance-hashed, and the final all-row XLA/governance subset passed `40`
tests. Execute score then FD for each row in frozen order, stopping that row on
any hard failure. Gate C, Gate D, aggregation, and LGSSM remain blocked until a
complete Gate B result receives a separate review.
