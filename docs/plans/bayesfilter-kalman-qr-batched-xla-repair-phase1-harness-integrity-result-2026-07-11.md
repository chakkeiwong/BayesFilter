# Phase 1 Result: Harness Failure Isolation And Artifact Integrity

Date: 2026-07-11
Status: `PHASE_CLOSED_PHASE2_REVIEW_AGREED`

## Outcome

Phase 1 implemented a fail-closed v2 benchmark contract and method-isolated
supervisor without changing Kalman/QR algorithmic math, vectorization,
batched-autodiff semantics, or the historical timing boundary.

Each scheduled unit is one `(case_id, method_id)` subprocess. The supervisor
persists the expected schedule before launch, binds resume eligibility to
source/config/runtime/fixture/schedule fingerprints, records durable
attempt-specific stage journals, rejects non-strict JSON, atomically replaces
artifacts, preserves passed sibling records, and returns zero only for a fully
valid `complete` schedule.

## Lane Isolation

Git HEAD changed from the Phase 0 opening commit
`52ee244498988e046a6356f926003b581103083b` to
`a644d29c5c2fd09a0deb3a7b5212799ff1fcb163` because another authorized agent
committed unrelated documentation. Per owner instruction, this lane ignores
other-lane HEAD movement and gates on exact relevant-path hashes. All Phase 1
work-set hashes, deferred algorithmic-source hashes, and 15 historical hashes
were unchanged across that movement.

## Evidence Contract Result

| Field | Result |
| --- | --- |
| Question | One method can fail, time out, or crash without erasing sibling evidence or being mislabeled resumable. |
| Exact baseline | Phase 0 v1 all-method coupling, unconditional `complete`, weak resume, permissive JSON, and GPU-exit defect. |
| Primary criterion | Passed: 51 focused tests plus one GPU-hidden, non-JIT, one-method integration smoke and exact resume. |
| Promotion vetoes | No silent skip, unselected dispatch, stale reuse, invalid JSON, identity/stage loss, or GPU-exit omission remains in the tested v2 contract. |
| Explanatory only | Smoke stage durations and exception text. |
| Not concluded | Numerical parity, vectorization correctness, XLA viability, runtime ranking, GPU readiness, HMC/posterior/default/production/scientific validity. |

## Defect-To-Test Map

| Phase 0 defect | Repair | Focused evidence |
| --- | --- | --- |
| All methods executed in one child | Closed builder registry, `--method`, singleton dispatch ledger | Selected builder called once; all unselected builders/callables remain zero |
| Failure stage lost on crash/timeout | Exclusive attempt journal; strict identity-bound, flushed, `fsync`ed stage events | Real hard-exit and timeout after every one of seven stage markers |
| `complete` despite missing/failed rows | Predeclared schedule and exact identity/status classifier | Missing, extra, duplicate, nonterminal, failed, and interrupted cases |
| Stale v1 artifact reuse | Five fingerprints plus exact method/case resume key | v1, malformed, source/config/runtime/fixture/schedule mismatch, failed/nonfinite/incomplete rejection |
| Permissive NaN/Infinity JSON | `allow_nan=False`, strict decoder, duplicate-key rejection | Raw and nested nonfinite tokens plus duplicate keys rejected |
| Partial artifact replacement | Temp file, flush, `fsync`, `os.replace` | Existing target survives encoding failure and real hard exit before replace |
| GPU failure could return zero | Only top-level `complete` maps to zero | Synthetic GPU method failure yields `complete_with_failures`, exit 1 |
| Failed sibling erased passed evidence | Per-method artifacts and exact passed-record reuse | Synthetic failure/crash/timeout preserves sibling; exact passed record reused |

## Artifacts And Hashes

| Path | SHA-256 |
| --- | --- |
| `scripts/kalman_qr_benchmark_contract.py` | `7d897101de7e02a80857dd785fb893a0e053c11656826856adb7f2b296f91f43` |
| `scripts/benchmark_kalman_qr_parameter_count_scaling.py` | `be6e494b3bc5246d3d89acfca9b112273013c399098f5b3e7030738c4a9cefaf` |
| `docs/benchmarks/run_kalman_qr_batched_xla_repair_2026_07_11.py` | `5bd779d8edbb5d78e9ae4104f1594ac8ecee443094012ede16480ee77897767a` |
| `tests/test_kalman_qr_benchmark_contract.py` | `62a65ebe795f6a95a58949cd92f9c547cf62741d1b6902193a0fc9ea41d92987` |
| `tests/test_kalman_qr_parameter_count_scaling_harness.py` | `607baf19f1df85846a3e89255ae4672b22a6742d004078bec91aac6237cc40d7` |

The historical 2026-07-09 runner and all historical JSON/Markdown outputs were
not edited. Deferred algorithmic hashes remained:

- `kalman_qr_derivatives_tf.py`: `9434c3e0...`
- `kalman_qr_tf.py`: `cc99674d...`
- `qr_factor_tf.py`: `bfde07b5...`

## Checks Actually Run

```bash
/home/ubuntu/anaconda3/envs/tfgpu/bin/python -m py_compile \
  scripts/kalman_qr_benchmark_contract.py \
  scripts/benchmark_kalman_qr_parameter_count_scaling.py \
  docs/benchmarks/run_kalman_qr_batched_xla_repair_2026_07_11.py \
  tests/test_kalman_qr_benchmark_contract.py \
  tests/test_kalman_qr_parameter_count_scaling_harness.py
```

Passed.

```bash
CUDA_VISIBLE_DEVICES=-1 /home/ubuntu/anaconda3/envs/tfgpu/bin/python -m pytest -q \
  tests/test_kalman_qr_benchmark_contract.py \
  tests/test_kalman_qr_parameter_count_scaling_harness.py
```

Result: `51 passed in 10.73s`.

```bash
git diff --check -- <five Phase 1 implementation/test paths>
```

Passed.

All 15 historical anchor/row/preflight paths in the Phase 0 inventory were
rehashed with zero mismatches. The closing benchmark-worker check was empty.

## Optional Integration Smoke

Command class:

- GPU hidden before TensorFlow import with `CUDA_VISIBLE_DEVICES=-1`.
- `dim=2,P=2,T=2,B=1,float32`, CPU thread request 1.
- One selected `batch_native_analytical_qr_score` method.
- Explicit `--no-jit-compile`; Phase 1 debug exception only.
- Artifacts under `/tmp/kalman_qr_phase1_harness/integration/`.

The fresh current-source run produced `status=complete`, one passed method,
finite `[1]` value and `[1,2]` score metadata, singleton invocation ledger,
empty GPU lists, all seven progress stages, and stable schedule identity. The
second identical command produced `reusable_exact_match` without creating a
new progress journal. No `NaN`/`Infinity` token occurred.

The smoke's trace/compile/warm numbers are explanatory debugging values only;
the Phase 1 boundary still includes host materialization and cannot support a
speed or XLA claim.

## Repair Loop Record

- Pure test run 1: one test mutated `method_id` to an invalid value; the contract
  correctly failed closed. The test was repaired to use another valid method.
- Harness test run 1: the Python 3.13 dynamic test loader omitted
  `sys.modules` registration for a dataclass module. The test loader was fixed.
- Subsequent audits tightened multi-case schedule fingerprints, duplicate JSON
  keys, stale current-attempt output handling, whole-schedule drift,
  passed-record stage integrity, resume rejection reasons, and structural
  record validation. The final 51-test suite passed.

## Decision Record

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Close Phase 1 after Phase 2 review agrees | Passed | No Phase 1 harness veto remains | Fixture semantics and graph structure are still historical and non-nested | Review and execute Phase 2 fixture-only repair | No numerical/XLA/runtime/GPU/scientific promotion |

## Handoff

Phase 2 was authorized to start after the refreshed
`docs/plans/bayesfilter-kalman-qr-batched-xla-repair-phase2-batched-fixture-subplan-2026-07-11.md`
received exact `VERDICT: AGREE` in substitute-review round 3. Phase 2 inherits the v2 supervisor and must
refresh fixture contract versions so old Phase 1 method records fail resume.
