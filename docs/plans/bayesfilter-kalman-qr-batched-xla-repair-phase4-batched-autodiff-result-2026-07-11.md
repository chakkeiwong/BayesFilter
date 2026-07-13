# Phase 4 Result: Correct True-Batched Autodiff Comparator

Date: 2026-07-11
Status: `LOCAL_GATE_PASSED_PHASE5_REVIEW_PENDING`

## Outcome

Phase 4 replaced the broken outside-tape diagnostic probe with a true-batched
reverse-mode comparator. The production callable differentiates the vector
batch likelihood with one VJP:

```text
tape.gradient(value, params, output_gradients=tf.ones_like(value))
```

It fails closed if the gradient is disconnected, forms no full Jacobian in the
production path, and has no scalar-row, finite-difference, NaN, zero, or stopped-
gradient fallback. The method contract is now v3 with the analytical/autodiff
batch-native pair as the exact default.

The full diagnostic Jacobian was `[B,B,P]=[4,4,3]` in both dtypes. All off-
diagonal blocks were exactly zero, diagonal blocks matched the production VJP,
and perturbing row 2 left rows 0, 1, and 3 unchanged within the locked
tolerances.

## Root Cause And Repair

The original probe created `tf.reduce_sum(value)` after leaving
`GradientTape`, so the reduction was not recorded and its gradient was `None`.
The repaired comparator uses the recorded vector target directly with all-one
output cotangents.

The first CPU-XLA producer then failed non-timeout with:

```text
InvalidArgumentError: XLA compilation requires a fixed tensor list size
```

The error arose in a reverse-mode accumulator under the dynamic batched
likelihood `tf.while_loop`. The loop already terminated at `n_timesteps` but,
unlike its scalar sibling, did not supply that value as `maximum_iterations`.
After a visible subplan repair and bounded review, exactly
`maximum_iterations=n_timesteps` was added. The dynamic loop and recursion math
were retained; the static Python time-loop route was not substituted. The
repaired CLI and exact pytest XLA gates both passed without changing any
tolerance.

## Evidence Contract Result

| Field | Result |
| --- | --- |
| Question | The true-batched QR likelihood produces the intended finite, row-independent `[B,P]` reverse-mode score and is now the primary autodiff comparator. |
| Exact baseline | Scalar-row autodiff, Phase 3 batch-native analytical score, and the same dynamic batch value backend. |
| Primary criterion | Passed: root-cause, fail-closed, dtype/shape/finite, scalar/analytical parity, full Jacobian, perturbation isolation, v3 method migration, strict artifact, and two independent CPU-XLA gates. |
| Promotion vetoes | None remains after the reviewed TensorList-bound repair. |
| Repair triggers exercised | Outside-tape target defect; pre-XLA test-order defect; dynamic-loop XLA TensorList bound. |
| Explanatory only | Residual magnitudes and tiny smoke wall times. |
| Not concluded | No warm-runtime improvement, CPU/GPU scalability, method ranking, HMC/posterior correctness, default, production, or scientific validity. |

## Numerical Evidence

Strict non-JIT artifact:
`docs/benchmarks/kalman_qr_batched_xla_repair_phase4_autodiff_2026-07-11.json`.
Schema `bayesfilter.kalman_qr_batched_xla_repair.phase4.autodiff.v1` passed all
22 independently recomputed gates.

| dtype | B | Batch-vs-scalar value max abs | Batch-vs-scalar score max abs | Batch-vs-analytical score max abs |
| --- | ---: | ---: | ---: | ---: |
| float32 | 1 | 0 | 0 | `2.3283064e-10` |
| float32 | 4 | 0 | `4.6566129e-10` | `6.9849193e-10` |
| float64 | 1 | 0 | `4.3368087e-19` | `2.1684043e-19` |
| float64 | 4 | 0 | `4.3368087e-19` | `1.7347235e-18` |

| dtype | Jacobian shape | Off-diagonal max abs | Diagonal-vs-VJP max abs | Unaffected-row value/score max abs |
| --- | --- | ---: | ---: | --- |
| float32 | `[4,4,3]` | 0 | `2.3283064e-10` | `0 / 0` |
| float64 | `[4,4,3]` | 0 | `2.1684043e-19` | `0 / 0` |

Locked tolerances remained:

- float32 value/score: `rtol=atol=2e-4`;
- float64 value: `rtol=atol=1e-10`;
- float64 score: `rtol=1e-8, atol=1e-9`;
- off-diagonal Jacobian: `2e-6` float32, `2e-12` float64.

## Method And Artifact Contract

```text
PRIMARY_METHOD_IDS = (
    "batch_native_analytical_qr_score",
    "batch_native_autodiff_qr_score",
)
REFERENCE_METHOD_IDS = (
    "scalar_analytical_row_loop",
    "autodiff_row_loop_qr_score",
)
METHOD_CONTRACT_VERSION = "batch-native-autodiff-phase4-v1"
```

The supervisor defaults exactly to the primary pair. A complete primary-pair
schedule requires directed, dtype-specific comparator parity. Incomplete-primary
schedules are explicitly `method_local_only`, record null/not-applicable parity,
and cannot satisfy the Phase 4 handoff. Stale v2, removed-method, method-version,
source, config, and schedule records fail reuse with named reasons and no
artifact overwrite.

## Repair Loop Record

- The reviewed pre-XLA command originally included the later XLA test. Bounded
  review round 1 found that `-k` was substring-based and post-collection. The
  plan was repaired to use the exact node with `--deselect`, require exactly one
  deselection, and describe prevention of execution rather than collection.
  Round 2 returned `VERDICT: AGREE`.
- The first CPU-XLA CLI producer failed with a dynamic-loop TensorList-size
  error. Its artifact and log were preserved under `/tmp` with hashes below.
- A visible subplan repair admitted only the existing timestep bound on the
  dynamic batch loop and explicitly forbade static-horizon substitution.
  Bounded review round 1 returned `VERDICT: AGREE`.
- After adding the bound, all non-JIT gates passed again, the CLI XLA producer
  passed, and the separate exact-node pytest XLA regression passed.

Claude Opus was requested through the narrow review gate. Platform policy
blocked execution before the liveness probe; no repository content was sent.
All substitute reviews are explicitly weaker than Claude review, and local
tests/artifacts carry the evidence burden.

## Checks Actually Run

Final post-repair pre-XLA suite:

```bash
CUDA_VISIBLE_DEVICES=-1 \
/home/ubuntu/anaconda3/envs/tfgpu/bin/python -m pytest -q \
  --deselect=tests/test_kalman_qr_batch_native_autodiff.py::test_batch_native_autodiff_cpu_xla_preserves_dtype_signature_and_value \
  tests/test_kalman_qr_batch_native_autodiff.py \
  tests/test_kalman_qr_benchmark_contract.py \
  tests/test_kalman_qr_parameter_count_scaling_harness.py \
  tests/test_linear_qr_batched_parameter_vectorization_tf.py \
  tests/test_kalman_qr_batched_fixture.py
```

Result: `236 passed, 1 deselected in 57.07s`. The one deselected node is the
separately executed XLA regression. Pycompile and scoped `git diff --check`
passed.

Strict non-JIT producer exited 0 in 21.11 outer seconds. An independent strict
reader recomputed all 22 gates and returned `state=passed`.

### CPU-XLA outer command manifests

| Field | CLI producer | Exact pytest regression |
| --- | --- | --- |
| Command role | Durable JSON producer | Independent exact test node |
| Timeout | 120 seconds | 120 seconds |
| Outer exit | 0, not timeout | 0, not timeout |
| Outer wall | 7.05 seconds | 6.36 seconds |
| Result | 22/22 raw gates true | `1 passed in 4.92s` |
| Log | `/tmp/kalman_qr_phase4_autodiff/cpu_xla_smoke.log` | `/tmp/kalman_qr_phase4_autodiff/cpu_xla_pytest.log` |
| Log SHA-256 | `e8f9858909caef3f668b3c51d68b964924559c594544bfce16f7181f6153bd66` | `d69fa8f8b31897eb0e111b7442ed56f7dfe2c8e7c9a4dd72d725331846da1100` |
| Durable JSON | `docs/benchmarks/kalman_qr_batched_xla_repair_phase4_autodiff_cpu_xla_smoke_2026-07-11.json` | N/A; cannot inherit CLI success |
| JSON SHA-256 | `bb3365288ae2d5b8c952a80bff5274439acf5a4f43aaf7deafe4e7b50141a21b` | N/A |
| Strict read | Passed independently | Exact pytest summary/hash preserved |

The passing CPU-XLA JSON has one concrete function, internal wall time
`4.7693s`, value max-absolute residual `4.7684e-7`, and score max-absolute
residual `1.1642e-9`.

First failed CPU-XLA evidence, retained rather than erased:

| Artifact | SHA-256 |
| --- | --- |
| `/tmp/kalman_qr_phase4_autodiff/cpu_xla_smoke_first_failure.json` | `6cca0ea1162e4bd435a0bc73a5248896e09d13e0a636c3afd19b7bd2d7caea70` |
| `/tmp/kalman_qr_phase4_autodiff/cpu_xla_smoke_first_failure.log` | `33c886ae0e3f6c66f2a4a3b35dcdfbd1d34fb285ad18d4543168c4bcfc057a05` |

## Run Manifest

| Field | Value |
| --- | --- |
| Git commit observed | `a644d29c5c2fd09a0deb3a7b5212799ff1fcb163` |
| Git policy | Other authorized lane ignored; exact declared paths/hashes gated |
| Interpreter | `/home/ubuntu/anaconda3/envs/tfgpu/bin/python` |
| Conda environment | `tfgpu` |
| Python | 3.13.13 |
| TensorFlow | 2.20.0 |
| TensorFlow Probability | `tfp-nightly` 0.25.0 |
| CPU/GPU status | Deliberate GPU-hidden CPU reference and CPU-XLA compatibility; no GPU evidence |
| JIT | Off for correctness/Jacobian; on for the two exact CPU-XLA gates |
| TF32 | Not queried; irrelevant to CPU Phase 4 evidence |
| Randomness | Deterministic fixture, seed N/A |
| Plan | `docs/plans/bayesfilter-kalman-qr-batched-xla-repair-phase4-batched-autodiff-subplan-2026-07-11.md` |
| Result | This file |
| Strict JSON | `docs/benchmarks/kalman_qr_batched_xla_repair_phase4_autodiff_2026-07-11.json` |
| XLA JSON | `docs/benchmarks/kalman_qr_batched_xla_repair_phase4_autodiff_cpu_xla_smoke_2026-07-11.json` |
| Declared source fingerprint | `3c8f75ca2e4aa648a99a9ede6483ec78d252e82ebef020ef6a90de282a47f679` |

TensorFlow emitted failed `cuInit` during GPU-hidden imports. This is an import
side effect, not harness device enumeration, GPU evidence, or evidence that the
trusted GPU environment is unavailable.

## Artifact Hashes

| Path | SHA-256 |
| --- | --- |
| `bayesfilter/linear/kalman_qr_tf.py` | `ad1fc869ce0be2aaffa18c1762d44b39c86de19ee0752e77cdce1c4d9c9fd06b` |
| `scripts/kalman_qr_benchmark_contract.py` | `b4e6d33bf9a3ff67d94d72226c9653b92f6b5ca4037a7b701d4778d69471991a` |
| `scripts/benchmark_kalman_qr_parameter_count_scaling.py` | `5539c070eddc10c14566d18bc49e36cd1db052816af3beb69a3028861aca7eed` |
| `docs/benchmarks/run_kalman_qr_batched_xla_repair_2026_07_11.py` | `927cb139e020372a186f8d77b4710abf9a1fba77ef4d7b8c79298289f9b7d824` |
| `tests/test_kalman_qr_batch_native_autodiff.py` | `ea6241979ceefd6db1a240d3605d63b2ba7158e2ce81f0377c9710cfa59b4488` |
| Strict non-JIT JSON | `987815216c6919ee52de69e1511cba4dd9a1827bb24a8099747907fb8134ba4e` |
| CPU-XLA JSON | `bb3365288ae2d5b8c952a80bff5274439acf5a4f43aaf7deafe4e7b50141a21b` |
| Final pre-XLA pytest log | `847e4b58b6f8a91536e4c02dd764dc13ff8e74ef9b592df83c4a30ae9ff61802` |

Read-only Phase 3 sources remained unchanged:

- analytical source `d24ae4363d4bf14a08149c81cf018b36fe9a3ca85a3c5cb7d6064ce4915bfb57`;
- QR factor source `bfde07b558e6c900a51f888d83ece817f562c06cf393c0dfdc76959adc087401`.

All 14 unique Phase 0 historical anchors matched. No Python/pytest benchmark
worker remained at close.

## Decision Record

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Accept Phase 4 local gate | Passed | No Phase 4 veto remains | Timing still includes host materialization and first-call labels remain ambiguous | Execute reviewed Phase 5 measurement-boundary separation | No runtime ranking, CPU/GPU scalability, HMC, posterior, default, production, or scientific claim |

## Inference Status

| Evidence class | Status |
| --- | --- |
| Hard veto screen | Passed for the declared tiny comparator/Jacobian/XLA cases |
| Statistically supported ranking | Not assessed; no stochastic or timing comparison |
| Descriptive-only differences | Residuals and smoke wall times only |
| Default-readiness | Not assessed |
| Next evidence needed | Measurement-boundary correctness, then reviewed CPU/GPU method-isolated ladders |

## Engineering, Numerical, And Interpretation Ledgers

| Ledger | Status |
| --- | --- |
| Engineering correctness | One-VJP batch-native method, v3 closed contract, stale-resume rejection, strict raw evaluators, and dynamic-loop XLA bound pass. |
| Numerical validity | Scalar/analytical parity, full Jacobian row independence, perturbation isolation, and tiny CPU-XLA parity pass. |
| Scientific interpretation | Not checked and not claimed. |

## Post-Run Red Team

The strongest alternative explanation is that the tiny `T=4,P=3,B=4` CPU-XLA
success does not extrapolate to the target `T=120,P=50/150,B=1/4/16` grid. The
dynamic loop avoids static time unrolling, but XLA lowering, compile memory, or
GPU layout may still fail at target scale. Phases 6 and 7 own those gates.

The weakest evidence is the single CPU-XLA configuration. A target-scale
method-isolated compile failure would overturn any broader XLA viability claim
but would not invalidate the non-JIT score correctness and row-independence
evidence established here.

## Handoff

Phase 5 may start only after its refreshed dedicated subplan receives bounded
read-only `VERDICT: AGREE`. It inherits the exact v3 method pair, dynamic-loop
bound, strict artifacts/hashes, final test counts, review weakness, and the rule
that timing evidence must exclude serialization/materialization ambiguity.
