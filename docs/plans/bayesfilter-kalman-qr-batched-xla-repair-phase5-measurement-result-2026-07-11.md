# Phase 5 Result: Compile And Runtime Measurement Separation

Date: 2026-07-11
Status: `LOCAL_GATE_PASSED_PHASE6_REVIEW_PENDING`

## Result

Phase 5 replaced the selected-method child's historical materializing timer with
the closed v4 timing contract
`separated-trace-execution-materialization-phase5-v1`. Trace, first executable
call, each synchronized warm call, one complete output materialization,
post-measurement parity, immutable payload encoding, and immutable payload write
are now separate stages. The outer envelope write is explicitly untimed.

The final GPU-hidden CPU-XLA smoke completed both primary methods and the strict
repository artifact passed all 42 independently recomputed gates:

- artifact:
  `docs/benchmarks/kalman_qr_batched_xla_repair_phase5_measurement_smoke_2026-07-11.json`;
- artifact SHA-256:
  `a74be199826f12b2c7931e7bb8d82d510826b69fcb7232ca3ad0b255b90ce74d`;
- raw status: `complete`;
- exported state: `passed`;
- failed gates: none;
- embedded immutable payload sidecars: two;
- source fingerprint:
  `56f0a447f1a12516a78ae5c98d64ed2f5f2c6f611d8f0e2c5d83f67d95b5fbc6`.

This closes measurement mechanics only. It does not rank methods or establish
target-scale CPU/GPU viability.

## Implementation

- Migrated the method/schedule contract to
  `bayesfilter.kalman_qr_batched_xla_repair.v4` and
  `measurement-boundaries-phase5-v1`; prior v3 timing artifacts are not
  reusable.
- Added closed, monotonic nanosecond stage events and cross-checks between raw
  intervals and reported durations.
- Added scalar-sentinel synchronization when
  `tf.experimental.async_wait()` is unavailable. Timed-call synchronization,
  one packed full-output transfer, and one packed parity-residual transfer have
  separate exact counts.
- Enforced zero selected-method invocations before the timed first call, one
  first call, two warm calls, and one post-measurement reference call.
- Replaced self-referential write timing with an immutable strict-JSON payload
  sidecar and an untimed outer envelope. Resume/export validate exact path,
  hash, strict content, and equality with the envelope.
- Added a strict Phase 5 evaluator/export that reconstructs the expected
  schedule and checks source/runtime/config/argv/path/device/thread/JIT/TF32,
  method records, events, durations, counts, parity, sidecars, aggregate
  comparator parity, and nonclaims from raw evidence. Stored `checks=true`
  cannot rescue a mutation.
- Preserved the old materializing timer only as an explicitly named historical
  helper for the untouched historical broad-grid route. It is not used by the
  v4 selected-method path.
- Added pure boundary, mutation, failure-injection, resume-integrity, and real
  GPU-hidden non-JIT supervisor tests.

No `bayesfilter/linear/*.py` algorithm source was edited in Phase 5.

## Evidence Contract Assessment

| Field | Observed result |
| --- | --- |
| Question | Each declared duration has a distinct raw interval; full output serialization is excluded from first/warm calls. |
| Baseline | The historical `_time_call` included complete host materialization in every timed call. |
| Primary criterion | Passed: local boundary/failure tests, method isolation, strict raw evaluator, and tiny JIT-on CPU smoke. |
| Promotion vetoes | None remains. No hidden full transfer, unsynchronized call, ambiguous first-call label, overlapping interval, invalid duration, sibling construction, stale reuse, or corrupt sidecar passed. |
| Explanatory only | Tiny-smoke trace/first/warm/write durations and GraphDef sizes below. |
| Not concluded | No method ranking, pure compilation estimate, CPU/GPU scalability, GPU readiness, HMC/posterior/default/production/scientific validity. |

## Local Checks

### Final compile

```bash
/home/ubuntu/anaconda3/envs/tfgpu/bin/python -m py_compile \
  scripts/kalman_qr_benchmark_contract.py \
  scripts/benchmark_kalman_qr_parameter_count_scaling.py \
  docs/benchmarks/run_kalman_qr_batched_xla_repair_2026_07_11.py \
  tests/test_kalman_qr_measurement_boundaries.py \
  tests/test_kalman_qr_parameter_count_scaling_harness.py \
  tests/test_kalman_qr_benchmark_contract.py
```

Result: passed.

### Final GPU-hidden non-XLA suite

```bash
CUDA_VISIBLE_DEVICES=-1 /home/ubuntu/anaconda3/envs/tfgpu/bin/python -m pytest -q \
  tests/test_kalman_qr_measurement_boundaries.py \
  tests/test_kalman_qr_parameter_count_scaling_harness.py \
  tests/test_kalman_qr_benchmark_contract.py \
  tests/test_kalman_qr_batch_native_autodiff.py \
  --deselect=tests/test_kalman_qr_batch_native_autodiff.py::test_batch_native_autodiff_cpu_xla_preserves_dtype_signature_and_value
```

Result: `207 passed, 1 deselected, 6634 warnings in 50.27s`.

- log: `/tmp/kalman_qr_phase5_measurement/pre_xla_pytest.log`;
- log SHA-256:
  `3ddb89c43839be3c190cb1026754baa27340271e7fbe4ff3fdd6181f9523459e`.

Warnings are TensorFlow AutoGraph/Gast deprecations. They are not promotion
evidence and did not hide a failed node.

### Final tiny CPU-XLA smoke

```bash
CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 TF_NUM_INTRAOP_THREADS=1 TF_NUM_INTEROP_THREADS=1 \
  timeout 210s /home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/run_kalman_qr_batched_xla_repair_2026_07_11.py \
  --dimensions 2 --parameter-counts 3 --timesteps 4 --batch-size 4 \
  --dtype float32 --device cpu --cpu-threads 1 --repeats 2 \
  --timeout-seconds 90 \
  --methods batch_native_analytical_qr_score batch_native_autodiff_qr_score \
  --output-dir /tmp/kalman_qr_phase5_measurement --no-resume \
  --jit-compile --tf32-enabled \
  > /tmp/kalman_qr_phase5_measurement/smoke.log 2>&1
```

Result: supervisor `complete`; both child return codes zero; all aggregate gates
true.

Strict export:

```bash
/home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/run_kalman_qr_batched_xla_repair_2026_07_11.py \
  --evaluate-phase5-smoke \
  --phase5-input /tmp/kalman_qr_phase5_measurement/status.json \
  --phase5-log /tmp/kalman_qr_phase5_measurement/smoke.log \
  --phase5-output docs/benchmarks/kalman_qr_batched_xla_repair_phase5_measurement_smoke_2026-07-11.json
```

Result: zero exit and `state=passed`, 42/42 gates true.

### Other checks

- Scoped `git diff --check`: passed.
- Strict JSON parse: passed.
- No active benchmark/pytest worker remained; the bounded `pgrep` check matched
  only its own sandbox wrapper when unanchored and was empty under the anchored
  worker pattern.
- The source/test/harness write set matched the reviewed Phase 5 set.

## Tiny-Smoke Descriptive Measurements

These are single-process descriptive mechanics only. No ranking is supported.

| Method | Trace s | First executable call s | Warm calls s | Materialization s | GraphDef nodes | GraphDef bytes |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| batch-native analytical | 1.735003176 | 0.885488860 | 0.000833817, 0.000574608 | 0.001073356 | 884 | 201148 |
| batch-native autodiff | 1.413144810 | 0.979238754 | 0.000853607, 0.000547807 | 0.001141296 | 638 | 291502 |

Both records used `scalar_sentinel`, exactly three scalar synchronization
materializations, exactly one packed full-output materialization, and exactly
one packed parity-residual materialization. Both direct value and score maximum
absolute residuals were exactly zero.

`first_executable_call_seconds` may include XLA compilation and first execution.
No subtraction is reported as compilation time.

## Artifact Hashes

| Artifact | SHA-256 |
| --- | --- |
| strict Phase 5 repository JSON | `a74be199826f12b2c7931e7bb8d82d510826b69fcb7232ca3ad0b255b90ce74d` |
| raw status JSON | `578bd7ace518d066bc2401b1ff4cae102fb3729ca8355b94e68d2966f55f26ff` |
| raw schedule JSON | `f6771b77e452a49baa25156aeb444c19587da162a1af75ada60fe9ded5aaaf7e` |
| smoke log | `9304b7fe24b4bb942b666fed18522f314474039180995a21f3b7f5f43a377d38` |
| final non-XLA pytest log | `3ddb89c43839be3c190cb1026754baa27340271e7fbe4ff3fdd6181f9523459e` |
| pre-edit path ledger | `9ce09f29859c4fb3f0ed7064587a4409b9d98ecc854f5963b3e02765ad608ef7` |

Final implementation/test hashes:

| Path | SHA-256 |
| --- | --- |
| `scripts/kalman_qr_benchmark_contract.py` | `e06f5686a13cd2075745e87d540b1dcf3c491a6845b1d2811fb7b58017d8457d` |
| `scripts/benchmark_kalman_qr_parameter_count_scaling.py` | `dc0a7deccb943ace3842a54f5219610896633ac7d03e97d8c3a8b2b9bff8ee48` |
| `docs/benchmarks/run_kalman_qr_batched_xla_repair_2026_07_11.py` | `5b255de88574da45f1b8b229a36ec2f64d1a58a49a7858f854e333dae7ed71f8` |
| `tests/test_kalman_qr_measurement_boundaries.py` | `759bd4e8709748fc3c1301498ff6826db52a0bc793443c32b2a7322fd994cb55` |
| `tests/test_kalman_qr_parameter_count_scaling_harness.py` | `00cc34b105fdc8cb88b768f8ff7a60d620a604e6aff4d0bfe3fe011c75a1f99c` |
| `tests/test_kalman_qr_benchmark_contract.py` | `402c21f8c09d1befacd6a7bcc7189b8adfb28217821d0745f701e4a7b54240bf` |

Read-only algorithm hashes remained exactly:

| Path | Opening/closing SHA-256 |
| --- | --- |
| `bayesfilter/linear/kalman_qr_tf.py` | `ad1fc869ce0be2aaffa18c1762d44b39c86de19ee0752e77cdce1c4d9c9fd06b` |
| `bayesfilter/linear/kalman_qr_derivatives_tf.py` | `d24ae4363d4bf14a08149c81cf018b36fe9a3ca85a3c5cb7d6064ce4915bfb57` |
| `bayesfilter/linear/qr_factor_tf.py` | `bfde07b558e6c900a51f888d83ece817f562c06cf393c0dfdc76959adc087401` |

## Opening/Closing Path Ledger

Opening ledger SHA-256:
`9ce09f29859c4fb3f0ed7064587a4409b9d98ecc854f5963b3e02765ad608ef7`.

- The three implementation paths and two existing test paths changed within the
  reviewed write set.
- `tests/test_kalman_qr_measurement_boundaries.py`, the strict smoke JSON, and
  this result were opening-absent and created within the reviewed write set.
- The Phase 6 subplan opening hash was
  `fab59afd676d78cfc53479b1976a511ba0751d5649e7588bd9ea75f42a700f68`;
  it is refreshed only after this result.
- Phase 5 review-record hashes were preserved:
  round 1 `a897288a...`, round 2 `4c9ffa90...`, round 3 `3fdf6a6b...`,
  round 4 `7bfcdd26...`.
- Repository-wide status remains descriptive because another authorized lane is
  active. Exact declared paths/hashes are this lane's boundary.

## Run Manifest

| Field | Value |
| --- | --- |
| Git commit recorded by run | `a644d29c5c2fd09a0deb3a7b5212799ff1fcb163` |
| Working state | Dirty shared worktree; exact source manifest embedded; no commit/reset/clean/revert performed |
| Interpreter | `/home/ubuntu/anaconda3/envs/tfgpu/bin/python3.13` |
| Python | CPython 3.13.13 |
| TensorFlow | 2.20.0 |
| TensorFlow Probability | `tfp-nightly` 0.25.0 |
| NumPy | 2.1.3 |
| Device | GPU-hidden `/CPU:0`; physical/logical GPU lists empty in child records |
| GPU-hidden setting | `CUDA_VISIBLE_DEVICES=-1` before TensorFlow import |
| Requested/effective TensorFlow threads | 1/1 intra-op, 1/1 inter-op |
| JIT/XLA | `jit_compile=true`; Host XLA compilation logged |
| TF32 | Enabled setting recorded; not a tensor-dtype claim on CPU |
| XLA flags | `UNSET` |
| Fixture | deterministic `dimension=2,T=4,P=3,B=4,float32`; no random seed applies |
| Child wall times | analytical 5.015893712 s; autodiff 4.913590333 s |
| Plan | `docs/plans/bayesfilter-kalman-qr-batched-xla-repair-phase5-measurement-subplan-2026-07-11.md` |
| Result | this file |
| Output/log | strict JSON and `/tmp/kalman_qr_phase5_measurement/smoke.log` |

TensorFlow emitted a failed `cuInit` message despite GPU hiding. This is an
import-side effect and not GPU evidence. The child manifests show no visible
physical or logical GPU and selected `/CPU:0`.

## Decision Table

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Accept Phase 5 local gate | Passed | No Phase 5 measurement veto remains | Tiny `T=4,P=3,B=4` mechanics do not establish target-scale graph/compile feasibility | Review and execute Phase 6 GPU-hidden CPU trace/XLA ladder | No speed ranking, compilation-time estimate, scalability, GPU/default/production/scientific claim |

## Inference Status

| Evidence class | Status |
| --- | --- |
| Hard veto screen | Passed for the tiny Phase 5 measurement smoke |
| Statistically supported ranking | None; no ranking procedure or replication evidence was run |
| Descriptive-only differences | Trace, first-call, warm-call, materialization, write, and GraphDef differences |
| Default readiness | Not evaluated |
| Next evidence needed | Reviewed target-grid structural trace and bounded method-isolated CPU-XLA gates, followed by lane-local GPU gates |

## Evidence Ledgers

| Ledger | Phase 5 status |
| --- | --- |
| Engineering correctness | Passed for v4 measurement/event/artifact mechanics and tiny child/supervisor execution |
| Numerical validity | Phase 4 parity remains valid; tiny Phase 5 direct and cross-method parity passed |
| Scientific interpretation | Not evaluated; no evidence promoted across this boundary |

## Negative/Repair Evidence

The local repair loop found schema-transition fixture failures and increasingly
strict evaluator gaps. These were implementation/artifact-contract defects, not
evidence against Kalman QR math or either primary method. Repairs added closed
schemas, interval-duration consistency, full `rtol+atol` parity recomputation,
exact sidecar equality, resume-sidecar checks, observed child runtime identity,
and export-level nonzero mutation tests. Final focused and aggregate checks pass.

## Post-Run Red Team

- Strongest alternative explanation: the tiny case is easy enough that both
  methods compile despite target-scale graph/codegen failure. Phase 6 is designed
  to discriminate this.
- Result that would overturn the Phase 5 conclusion: a raw one-field mutation
  accepted as passed, a timed full-output transfer, a missing synchronization,
  or a final-source smoke failing strict re-export. None occurred.
- Weakest evidence: single-run wall durations and GraphDef sizes. They are
  explanatory only and are not used for ranking or timeout extrapolation beyond
  choosing conservative prospective Phase 6 caps.

## Review Trail

Claude Opus remained platform-policy-blocked before its liveness probe; no
repository content was sent. Fresh bounded Codex substitute review was used and
is explicitly weaker than Claude review.

- Round 1: `REVISE` for self-referential write timing, cold-call order, exact
  commands/hashes, fail-closed stages, raw ordering, and synchronization counts.
- Round 2: `REVISE` for evaluator command, stronger time ordering, path ledger,
  journal/envelope failures, timeout budget, process-entry invocation count, and
  durable sidecar evidence.
- Round 3: `REVISE` for exact runtime paths/opening hashes and honest emergency
  outer-timeout semantics.
- Round 4: exact `VERDICT: AGREE`.

## Handoff

Phase 6 may start only after its refreshed dedicated subplan receives exact
`VERDICT: AGREE` from the available bounded reviewer. It inherits v4 artifacts,
source fingerprint `56f0a447...`, exact read-only algorithm hashes, the final
`207 passed, 1 deselected` local suite, the 42/42 tiny JIT smoke, method-isolated
fresh children, and the rule that target-scale compile/runtime outcomes are
lane-local while common parity/harness invalidity is a continuation veto.
