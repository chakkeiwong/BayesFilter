# Phase 2 Result: True-Batched And Nested Fixture Tensor Algebra

Date: 2026-07-11
Status: `LOCAL_GATE_PASSED_PHASE3_REVIEW_PENDING`

## Outcome

Phase 2 replaced Python batch-row construction of all 16 model/derivative
tensors with TensorFlow batch algebra and established one nested deterministic
fixture family:

- `theta[j] = -0.2 + 0.4*j/149` for `P<=150`;
- `P=50` parameters and derivative bases are exact prefixes of `P=150`;
- `B=1/4` select locked row IDs from one canonical `B=16` proposal cloud;
- base-model tensors and observations are independent of `P` and `B`;
- observations are generated from the parameter-independent base model;
- the trace-only fixture graph has no B-dependent node growth or normalized
  structural change at `B=1/4/16`.

No QR/Kalman algorithmic helper was edited in this phase. No score, XLA, GPU,
timing, or autodiff-repair run was performed.

## Evidence Contract Result

| Field | Result |
| --- | --- |
| Question | True-batched TensorFlow fixture algebra reproduces stacked scalar tensors while enforcing nested `P/B` identity and removing B-axis graph duplication. |
| Exact baseline | Phase 1 v2 harness plus the historical Python row loop, `tf.linspace(...,P)`, requested-B offsets, and parameterized observation construction. |
| Primary criterion | Passed: all 16 tensor parity rows, nested fixture identity, source-structure gate, strict named stale-resume rejection, and exact GraphDef gate. |
| Promotion vetoes | None fired after the bounded implementation repair loop. |
| Explanatory only | Serialized GraphDef bytes, trace wall time, and floating-point residuals. |
| Not concluded | Analytical score correctness, autodiff correctness, XLA viability, warm runtime, GPU readiness, method ranking, HMC/posterior/default/production/scientific validity. |

## Defect-To-Evidence Map

| Historical defect | Repair | Focused evidence |
| --- | --- | --- |
| `_batched_model_tensors` looped over `range(B)` | `bp` einsums, batched matrix products, and input-dependent broadcast algebra | AST/source gate plus stacked-scalar parity for every one of 16 tensors |
| `tf.linspace(-0.2,0.2,P)` changed common coordinates with `P` | Fixed denominator 149 | Exact parameter and all eight derivative-basis prefix checks |
| Proposal rows changed with requested `B` | Canonical 16-row cloud and locked integer row map | Exact row-ID, cross-B subset, and cross-P prefix checks |
| Observation generation was attached after derivative parameterization | Generate observations from base tensors only | Common base-model and observation hashes across all six `P/B` cells per dtype |
| Trace structure could hide B unrolling or B-shaped constants | Input-dependent empty-slice broadcast zeros, deterministic named fixture constants, and strict leading-B-only normalizer | Equal node count/digest/constant inventory plus mutation tests |
| Phase 1 records could otherwise resume after fixture semantic changes | Centralized Phase 2 version strings in the v2 contract | Synthetic valid prior-version record rejected with exact `config_fingerprint_mismatch` |

## Numerical And Structural Evidence

The strict diagnostic is
`docs/benchmarks/kalman_qr_batched_xla_repair_phase2_graphdef_2026-07-11.json`.
It records `state=passed` under schema
`bayesfilter.kalman_qr_batched_xla_repair.phase2.v1`.

Parity coverage:

- dtypes: `float32`, `float64`;
- parameter counts: `P=3,50`;
- batch sizes: `B=1,4`;
- 8 fixture rows and 16 tensors per row;
- exact shapes and dtypes in every row;
- exact equality required for parameter-independent derivative-basis outputs;
- declared near tolerances `2e-6` for float32 and `2e-13` for float64;
- maximum observed absolute residual across all rows/tensors:
  `2.9802322387695312e-08`.

Nested-identity coverage at `dimension=10,T=8`, both dtypes, `P=50/150`, and
`B=1/4/16` passed all of:

- common base-model hash;
- common observation hash;
- exact parameter and derivative-basis prefixes;
- exact proposal-cloud prefix;
- exact locked row IDs;
- exact selected-row identity across B;
- exact selected-row prefix across P.

GraphDef evidence at `dimension=10,P=50,float32`:

| B | Nodes | Serialized bytes | Normalized digest | Constants | Constant inventory digest |
| --- | ---: | ---: | --- | ---: | --- |
| 1 | 174 | 131754 | `09575a25f543fd100faaab32312a74253931ed6cf48acd0cf78b402fcd4be1bc` | 93 | `c23f4b6c95884427898f17ec6982d6bb94875ab4eb9ff15b430f62d8f0ecf344` |
| 4 | 174 | 131754 | same | 93 | same |
| 16 | 174 | 131754 | same | 93 | same |

The exact byte equality is explanatory only. The passing structural gates are
equal node count, equal strict normalized digest, equal constant inventory,
and exact leading output batch shapes.

## Repair Loop Record

Initial full focused run: `73 passed, 2 failed`. Both failures were confined to
the GraphDef normalization gate:

1. eager-captured fixture tensors received trace-order-dependent numeric node
   names, changing exact names and consumer edges despite equal topology;
2. `tf.zeros_like(parameters_batch)` was folded into B-shaped constants, which
   violated the no-varying-constant-payload/count contract.

The repair did not relax the normalizer. The trace wrapper now embeds the same
fixture tensors as explicitly named graph constants, and derivative-basis
broadcasting derives a zero per row from an empty input slice instead of a
B-shaped constant. The positive normalizer test was also corrected to mutate
all parameter descendants, not only the input placeholder.

Focused graph rerun: `11 passed`. Final full focused run after adding an
explicit constant-role mutation case: `76 passed`.

Mutation coverage rejects changes to op, edge, attribute, duplicate node,
constant role, constant payload, constant element count, constant dtype,
constant rank, and a non-batch dimension. Only leading-B shape metadata on the
parameter input and its descendants is normalized.

## Checks Actually Run

```bash
/home/ubuntu/anaconda3/envs/tfgpu/bin/python -m py_compile \
  scripts/kalman_qr_benchmark_contract.py \
  scripts/benchmark_kalman_qr_parameter_count_scaling.py \
  docs/benchmarks/run_kalman_qr_batched_xla_repair_2026_07_11.py \
  tests/test_kalman_qr_batched_fixture.py
```

Passed.

```bash
CUDA_VISIBLE_DEVICES=-1 /home/ubuntu/anaconda3/envs/tfgpu/bin/python -m pytest -q \
  tests/test_kalman_qr_batched_fixture.py \
  tests/test_kalman_qr_benchmark_contract.py \
  tests/test_kalman_qr_parameter_count_scaling_harness.py
```

Result: `76 passed in 13.96s`.

```bash
git diff --check -- \
  scripts/kalman_qr_benchmark_contract.py \
  scripts/benchmark_kalman_qr_parameter_count_scaling.py \
  docs/benchmarks/run_kalman_qr_batched_xla_repair_2026_07_11.py \
  tests/test_kalman_qr_batched_fixture.py \
  tests/test_kalman_qr_parameter_count_scaling_harness.py
```

Passed.

Diagnostic command:

```bash
CUDA_VISIBLE_DEVICES=-1 OMP_NUM_THREADS=1 TF_NUM_INTRAOP_THREADS=1 \
TF_NUM_INTEROP_THREADS=1 \
/home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  scripts/benchmark_kalman_qr_parameter_count_scaling.py \
  --phase2-fixture-diagnostic --device cpu --cpu-threads 1 \
  --no-jit-compile \
  --output-json \
  docs/benchmarks/kalman_qr_batched_xla_repair_phase2_graphdef_2026-07-11.json \
  --phase2-log-path \
  /tmp/kalman_qr_phase2_fixture/phase2_graphdef.log
```

The command exited zero in 1.59 seconds and the independent strict JSON reader
accepted the artifact. All 15 historical hashes matched the Phase 0 inventory,
the three read-only algorithmic hashes matched Phase 1, and the closing
benchmark-worker check was empty.

## Environment Caveat

`CUDA_VISIBLE_DEVICES=-1` was set before TensorFlow import and the diagnostic
did not call the harness GPU-enumeration helper. TensorFlow nevertheless emitted
a failed `cuInit` message during import. This is recorded as an import/runtime
side effect, not as a GPU probe result or evidence about GPU health. The run is
an explicitly CPU-only, non-JIT reference/structure diagnostic and makes no GPU
claim.

## Artifacts And Hashes

| Path | SHA-256 |
| --- | --- |
| `scripts/kalman_qr_benchmark_contract.py` | `7644045a3e9f5e69a18d327f15022ae60ab0b810c1d57eb626a4981e736ec623` |
| `scripts/benchmark_kalman_qr_parameter_count_scaling.py` | `5163bec4ad9816cabc5495f6c84b787720364aeb9e3e73452d50846726303a96` |
| `docs/benchmarks/run_kalman_qr_batched_xla_repair_2026_07_11.py` | `c18b6be3c156e55088bf8aa58567036c071f21879d6eb103073d263f2d4a38e3` |
| `tests/test_kalman_qr_batched_fixture.py` | `5278e28224f5e113daa34abfd1e12af2e68730b383fdd058a9f02b96432fa8d2` |
| `tests/test_kalman_qr_benchmark_contract.py` | `62a65ebe795f6a95a58949cd92f9c547cf62741d1b6902193a0fc9ea41d92987` |
| `tests/test_kalman_qr_parameter_count_scaling_harness.py` | `607baf19f1df85846a3e89255ae4672b22a6742d004078bec91aac6237cc40d7` |
| Phase 2 strict JSON | `1bbfe1796cb29136f996a8ddb7667b9589896dd4de0531dbadd7063afb22affa` |
| Phase 2 full log | `e4404822e49f5290e81d30dbb6928998d99ccba206cb6e1ed39b561893bf2674` |

Read-only algorithmic hashes remained:

- `kalman_qr_derivatives_tf.py`: `9434c3e0...`;
- `kalman_qr_tf.py`: `cc99674d...`;
- `qr_factor_tf.py`: `bfde07b5...`.

## Decision Record

| Decision | Primary criterion status | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Accept Phase 2 local fixture/structure gate | Passed | No Phase 2 veto remains | Parameter-axis analytical helpers still unroll P and have not been changed or tested here | Refresh and review Phase 3 analytical parameter-vectorization subplan | No score, XLA, timing, GPU, HMC, posterior, default, production, or scientific promotion |

## Engineering, Numerical, And Interpretation Ledgers

| Ledger | Status |
| --- | --- |
| Engineering correctness | Fixture-only tests and strict artifact pass; v2 identity/version invalidation is enforced. |
| Numerical validity | Batched fixture tensors match stacked scalar construction within declared roundoff tolerances; no score was evaluated. |
| Scientific interpretation | Not checked. This phase establishes fixture semantics and graph structure only. |

## Post-Run Red Team

The strongest alternative explanation for equal graph structure is that the
trace-only wrapper, rather than the production score wrapper, removes the
observed B-axis duplication. That is why this result claims only fixture-call-
graph structure. Phase 3 and later end-to-end graph/XLA gates must still test
the score wrappers. A B-dependent node or constant-inventory change in an
end-to-end trace would overturn any broader graph-scaling interpretation.

The weakest evidence is that GraphDef equality is one TensorFlow-version and
fixture-shape observation. It is a hard structural gate for this environment,
not evidence of runtime improvement or general compiler behavior.

## Handoff

Phase 3 may start only after its refreshed dedicated subplan incorporates the
actual Phase 2 formulas, shapes, graph evidence, current hashes, exact test/run
commands, and receives a bounded `VERDICT: AGREE`. Until then, Phase 2 is local-
gate complete but the authoritative phase handoff is pending.
