# Phase 3 Result: Analytical Parameter-Axis Vectorization

Date: 2026-07-11
Status: `LOCAL_GATE_PASSED_PHASE4_REVIEW_PENDING`

## Outcome

Phase 3 removed the three Python parameter-axis loops from the true-batched
analytical QR score route. The QR, Cholesky, and factor-to-covariance first-
derivative helpers now treat `P` as TensorFlow batch algebra while preserving
`[B,P,...]` order, dtype, QR positive-diagonal convention, jitter semantics,
and the existing first-derivative formulas.

The vectorized route passed helper-reference, dynamic-P trace-reuse, source,
end-to-end analytical, reverse-mode diagnostic, shape/dtype, GraphDef, and tiny
CPU-XLA compatibility gates. No scalar, masked, Hessian, QR-factor reference,
or value backend was edited.

## Evidence Contract Result

| Field | Result |
| --- | --- |
| Question | The current first-derivative formulas can evaluate `P` as TensorFlow batch algebra without static P graph duplication. |
| Exact baseline | Unchanged scalar first-order QR-factor helpers, scalar analytical score rows, current batch-native score, and the Phase 2 nested fixtures. |
| Primary criterion | Passed: formula parity, one dynamic-P trace, source gate, end-to-end parity, exact P=50/150 node/op/constant structure, strict artifact validity, and bounded CPU-XLA smoke. |
| Promotion vetoes | None fired after the bounded implementation and test-sequencing repairs. |
| Explanatory only | Serialized GraphDef bytes, trace duration, XLA-smoke duration, and historical graph sizes. |
| Not concluded | Warm-runtime improvement, method ranking, CPU/GPU scalability, HMC/posterior/default/production/scientific validity. |

## Formula And Test Map

| Helper | Vectorized operation | Focused evidence |
| --- | --- | --- |
| `_batched_stack_qr_lower_factor_first_derivatives` | Broadcast positive-diagonal `Q/R` across P, right triangular solve, batched QR differential | Scalar helper parity at `B=1/3`, `P=1/4`, both dtypes; dynamic-P one-trace reuse; static/dynamic `K<N` rejection |
| `_batched_cholesky_factor_first_derivatives` | Batched triangular solves with `L[:,None,...]` and lower-half differential | Scalar helper parity at `B=1/3`, `P=1/4`, both dtypes; dynamic-P one-trace reuse |
| `_batched_factor_covariance_first_derivatives` | `dL L.T + L dL.T` with explicit P-axis broadcasting | Scalar helper parity at `B=1/3`, `P=1/4`, both dtypes; dynamic-P one-trace reuse |
| Full batched analytical score | Existing TensorFlow time loop using the three helpers | Scalar analytical parity, float32 reverse-mode diagnostic, distinct `B/P/state` axes, default-jitter dtype, malformed-rank rejection, no scalar wrapper |

The QR helper now fails closed when `K<N`, both for statically known shapes and
for dynamic TensorFlow signatures. No alternate wide-matrix derivative was
introduced.

## Structural Evidence

The strict artifact is
`docs/benchmarks/kalman_qr_batched_xla_repair_phase3_parameter_graphdef_2026-07-11.json`.
It passed schema `bayesfilter.kalman_qr_batched_xla_repair.phase3.v1` with every
gate true.

| P | B | dtype | Nodes | Serialized bytes | Ordered-op digest | Constants | Output shapes |
| ---: | ---: | --- | ---: | ---: | --- | ---: | --- |
| 50 | 4 | float32 | 884 | 204149 | `c53a60bc537b62f429d03c6976e968e70428ac31fb3b491bd027ea74666606ee` | 573 | `[4]`, `[4,50]` |
| 150 | 4 | float32 | 884 | 204397 | same | 573 | `[4]`, `[4,150]` |

The op histograms also match exactly. The byte difference is expected from
shape metadata and derivative payloads and is explanatory only. The trace did
not execute the score, enumerate devices, or invoke XLA.

## Repair Loop Record

Plan review converged at the user-specified maximum of five rounds:

1. Round 1 required fail-closed graph-evaluator mutation tests and binding
   `K>=N` validation.
2. Round 2 agreed after those repairs.
3. Round 3 found that the plan allowed checks in a historical test file that
   the pre-GraphDef command excluded.
4. Round 4 agreed after all Phase 3 non-JIT checks moved to the focused file and
   the historical file became read-only.
5. Round 5 agreed after an execution-discovered sequencing repair removed an
   out-of-scope scalar/Hessian suite containing explicit XLA from the pre-
   GraphDef command.

Claude Opus was policy-blocked before liveness probe, so all five records are
explicitly weaker bounded Codex substitute reviews. No content was sent to
Claude and no Claude agreement or liveness claim is made.

Execution repairs:

- Added focused float32 reverse-mode, default-jitter dtype, malformed-rank,
  source-contract, and distinct `B/P/state` tests.
- A first distinct-axis test requested unsupported fixture `B=3`; it was
  repaired using a deterministic three-row prefix of the declared `B=4` cloud,
  without changing production fixture batch support.
- A combined 94-node command completed 63 nodes then hit a prospective
  180-second cap. It is incomplete evidence, not a pass. Collection showed the
  read-only legacy scalar/Hessian file included explicit XLA and expensive
  out-of-scope graph checks; round 5 approved excluding it from this phase.
- The corrected 77-test command initially found a source-test interaction:
  Phase 3's literal forbidden-call strings triggered a Phase 2 whole-file text
  assertion. The checker was changed to inspect AST call names, which is more
  semantic and does not weaken either gate. Eight focused tests then passed.

No tolerance, numerical formula, structural equality gate, or XLA timeout was
weakened after observing results.

## Checks Actually Run

Compile:

```bash
/home/ubuntu/anaconda3/envs/tfgpu/bin/python -m py_compile \
  bayesfilter/linear/kalman_qr_derivatives_tf.py \
  scripts/benchmark_kalman_qr_parameter_count_scaling.py \
  tests/test_linear_qr_batched_parameter_vectorization_tf.py
```

Passed.

Final GPU-hidden non-XLA suite:

```bash
CUDA_VISIBLE_DEVICES=-1 timeout 180 \
/home/ubuntu/anaconda3/envs/tfgpu/bin/python -m pytest -q --disable-warnings \
  tests/test_linear_qr_batched_parameter_vectorization_tf.py \
  tests/test_linear_qr_factor_tf.py \
  tests/test_kalman_qr_batched_fixture.py
```

Result: `77 passed in 28.25s`.

Scoped `git diff --check` passed for the algorithm, harness, contract, and
focused test write set.

Trace-only diagnostic:

```bash
CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 TF_NUM_INTRAOP_THREADS=1 \
TF_NUM_INTEROP_THREADS=1 \
/home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  scripts/benchmark_kalman_qr_parameter_count_scaling.py \
  --phase3-parameter-graph-diagnostic --device cpu --cpu-threads 1 \
  --no-jit-compile \
  --output-json \
  docs/benchmarks/kalman_qr_batched_xla_repair_phase3_parameter_graphdef_2026-07-11.json \
  --phase3-log-path \
  /tmp/kalman_qr_phase3_vectorization/phase3_parameter_graphdef.log
```

Exited zero in 5.09 seconds. An independent strict reader confirmed `state=passed`,
all seven checks true, `jit_compile=false`, `xla_execution=not_run`, and
`gpu_detection_by_harness=not_called`.

Bounded CPU-XLA smoke:

```bash
CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 TF_NUM_INTRAOP_THREADS=1 \
TF_NUM_INTEROP_THREADS=1 timeout 120 \
/home/ubuntu/anaconda3/envs/tfgpu/bin/python -m pytest -q --disable-warnings \
  tests/test_linear_qr_batched_analytical_score_tf.py::test_batched_qr_score_cpu_xla_preserves_dtype_and_signature
```

Result: `1 passed in 4.61s`.

## Run Manifest

| Field | Value |
| --- | --- |
| Git commit observed | `a644d29c5c2fd09a0deb3a7b5212799ff1fcb163` |
| Git policy | Unrelated authorized HEAD/worktree movement ignored; declared paths gated by hashes |
| Interpreter | `/home/ubuntu/anaconda3/envs/tfgpu/bin/python` |
| Conda environment | `tfgpu` |
| Python | 3.13.13 |
| TensorFlow | 2.20.0 |
| TensorFlow Probability | `tfp-nightly` 0.25.0 |
| CPU/GPU status | Deliberate GPU-hidden CPU reference/trace/smoke; no GPU evidence |
| JIT | Off for correctness/GraphDef; on only for the exact bounded smoke |
| TF32 | Not queried; not relevant to CPU-only Phase 3 evidence |
| Randomness | Deterministic fixture, seed N/A |
| Plan | `docs/plans/bayesfilter-kalman-qr-batched-xla-repair-phase3-parameter-vectorization-subplan-2026-07-11.md` |
| Result | This file |
| JSON | `docs/benchmarks/kalman_qr_batched_xla_repair_phase3_parameter_graphdef_2026-07-11.json` |
| Logs | `/tmp/kalman_qr_phase3_vectorization/` |

TensorFlow emitted `failed call to cuInit: CUDA_ERROR_NO_DEVICE` during a
GPU-hidden import. This is recorded as an import side effect, not GPU detection
or evidence that the trusted GPU environment is unavailable.

## Artifact Hashes

| Path | SHA-256 |
| --- | --- |
| `bayesfilter/linear/kalman_qr_derivatives_tf.py` | `d24ae4363d4bf14a08149c81cf018b36fe9a3ca85a3c5cb7d6064ce4915bfb57` |
| `scripts/benchmark_kalman_qr_parameter_count_scaling.py` | `17f03ab7ed22d30f5e5d08612cf7b87180034d01c866c9da4f2f32ba68cc5689` |
| `scripts/kalman_qr_benchmark_contract.py` | `7644045a3e9f5e69a18d327f15022ae60ab0b810c1d57eb626a4981e736ec623` |
| `tests/test_linear_qr_batched_parameter_vectorization_tf.py` | `d13e62a28fa01fa74e4a41c578c533f0609bfc6c1c4d0d93f5b2561438f1db3d` |
| Phase 3 strict JSON | `c610b90563f8ee0c4e6d3233dbd56ace20060ca41e8527400131cc16547df2a5` |
| Final non-XLA log | `c2a6f4df3cec2cada9819b715e8b9680f86f2c5b25b60eb55d923a9d5f969b7e` |
| CPU-XLA smoke log | `cf9044cf716b038c6a228f61f49e3341b29f59739607da2a1ac46c1e5170a0d1` |

Read-only hashes remained:

- `bayesfilter/linear/kalman_qr_tf.py`: `cc99674d...`;
- `bayesfilter/linear/qr_factor_tf.py`: `bfde07b5...`;
- historical batched-score test: `b8525aaa...`;
- scalar/Hessian derivative test: `d7ae40b7...`;
- scalar QR-factor test: `14bd2995...`;
- Phase 2 fixture test: `5278e282...`.

All 14 unique Phase 0 historical anchors matched their recorded SHA-256 values.
No Python or pytest worker remained at close.

## Decision Record

| Decision | Primary criterion status | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Accept Phase 3 local gate | Passed | No Phase 3 veto remains | The true-batched autodiff builder still reduces outside its tape and is not a valid comparator | Refresh and review Phase 4 true-batched autodiff subplan | No timing, GPU, ranking, HMC, posterior, default, production, or scientific promotion |

## Engineering, Numerical, And Interpretation Ledgers

| Ledger | Status |
| --- | --- |
| Engineering correctness | Three P-axis loops removed; source, trace, strict artifact, and CPU-XLA compatibility gates pass. |
| Numerical validity | Vectorized helpers and end-to-end analytical score match unchanged references within prospective tolerances. |
| Scientific interpretation | Not checked; this is an engineering/numerical implementation result only. |

## Post-Run Red Team

The strongest alternative explanation for equal P graph structure is that
TensorFlow still hides P-dependent compiler work in shape specialization or
later XLA lowering. Phase 3 proves GraphDef structural invariance and one tiny
CPU-XLA compatibility case only. Phases 6 and 7 must still test method-isolated
XLA compilation on their declared CPU/GPU ladders.

The weakest evidence is the one small CPU-XLA smoke. It cannot establish CPU
compile scalability, GPU support, or runtime improvement. A later P-dependent
HLO/codegen failure would overturn any broader compiler claim but would not
invalidate the helper formula parity established here.

## Handoff

Phase 4 may start only after its refreshed dedicated subplan receives bounded
read-only `VERDICT: AGREE`. The handoff carries the exact Phase 3 artifact,
hashes, local checks, review weakness, and the unresolved true-batched autodiff
target bug.
