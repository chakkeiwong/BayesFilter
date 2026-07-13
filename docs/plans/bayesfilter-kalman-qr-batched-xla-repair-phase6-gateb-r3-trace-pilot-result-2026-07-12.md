# Phase 6 Gate B R3 Trace Census And Pilot Result

Date: 2026-07-12

Status: `GATE_B_R3_VALID_STRUCTURAL_TRACE_REJECTION_PILOT_NOT_LAUNCHED_GATE_C_BLOCKED`

Parent repair subplan:
`docs/plans/bayesfilter-kalman-qr-batched-xla-repair-phase6-gateb-r3-mixed-format-bindings-repair-subplan-2026-07-12.md`.

Authority ID:
`7d630ff42cc759c02d3e6618c90b97923ec9a9e8cba5b99dd41ee94e09347a33`.

## Result

The single authorized Gate B R3 runtime closed as a **valid structural trace
rejection**. All 36 GPU-hidden, non-XLA trace children passed their child
evidence contracts, but every fixed-dimension/method cohort failed the strict
cross-`P/B` GraphDef normalization predicate. Therefore
`trace_common_valid=false`, and the supervisor correctly wrote exactly two
pilot rows as `not_launched:trace_gate_not_passed`.

No pilot child was launched. There were zero XLA invocations and zero Kalman
XLA invocations. This result validates the repaired R3 authority and runtime
harness and establishes a real Gate B structural promotion veto. It does not
yet distinguish an evaluator alignment artifact from expected axis-dependent
graph data or genuine TensorFlow/autodiff topology specialization.

Gate C, GPU work, Phase 7, memory benchmarking, and performance benchmarking
remain blocked. The original memory and performance problems are not proven
fixed.

## Classification

| Layer | Observed result | Interpretation |
| --- | --- | --- |
| Authority and harness | Valid | Proposal, attestation, source/runtime identity, child evidence, ledgers, budget, and lease passed their closed checks. |
| Trace child execution | 36 of 36 passed | All scheduled non-XLA concrete-function traces produced valid bounded evidence. |
| Cross-lattice structure gate | Rejected | All six cohorts contain rejected typed GraphDef differences and a canonical-byte mismatch. |
| Pilot XLA | Not launched | Both rows were prospectively pruned by `trace_gate_not_passed`. |
| Numerical correctness | Not tested here | A structural rejection is not a Kalman numerical failure. |
| Memory and performance | Not tested here | No XLA compile/runtime or benchmark row ran. |

The current candidate is the existing positional-token normalization rule, not
the Kalman QR research direction. The candidate gate rejected the observed
graphs. That does not reject the algorithm, TensorFlow implementation,
CPU-XLA backend, GPU-XLA backend, or the broader repair direction.

## Trace Evidence

Trace ledger:
`docs/benchmarks/kalman_qr_batched_xla_repair_phase6_gateb_r3_trace_census_2026-07-12.json`.

| Field | Value |
| --- | --- |
| SHA-256 | `7444fb41ef9d125990dee93a5370227c4b9ec0987ee37cb9ab7dfd362281d2b6` |
| Byte count | `221375005` |
| Schema state | `passed` |
| Update index | `72` |
| Record count | `36` |
| Record states | `36 passed` |
| Record classifications | `trace_pass: 36` |
| Common invalidity | `false` |
| Final ledger checks | all `true` |
| Declared/decoded GraphDef bytes | `8921463` |
| Global decoded-byte cap | passed |
| Final structural predicate | `trace_common_valid=false` |

Raw rejection counts are descriptive coordinates produced by the current
positional protobuf-token evaluator. They are not counts of independent graph
or algorithm defects.

| Method | Dimension | Accepted differences | Rejected differences | Main rejected rules |
| --- | ---: | ---: | ---: | --- |
| `batch_native_analytical_qr_score` | 10 | 692 | 47 | 46 unclassified plus 1 canonical mismatch |
| `batch_native_analytical_qr_score` | 20 | 692 | 47 | 46 unclassified plus 1 canonical mismatch |
| `batch_native_analytical_qr_score` | 30 | 692 | 47 | 46 unclassified plus 1 canonical mismatch |
| `batch_native_autodiff_qr_score` | 10 | 255 | 18998 | 18997 unclassified plus 1 canonical mismatch |
| `batch_native_autodiff_qr_score` | 20 | 357 | 12252 | 12251 unclassified plus 1 canonical mismatch |
| `batch_native_autodiff_qr_score` | 30 | 357 | 12252 | 12251 unclassified plus 1 canonical mismatch |

The directly supported diagnostic hypotheses are deliberately unresolved:

- protobuf paths contain positional node/function indices, so a small insertion
  or reordering can cascade into many raw coordinate differences;
- expected `B`/`P` values stored in `Const` tensors are currently rejected
  rather than axis-normalized;
- `B=1` versus `B>1` may induce genuine TensorFlow or reverse-mode graph
  specialization; and
- the large autodiff rejection totals may combine alignment cascades with true
  differences.

None of those hypotheses is established by this result.

## Pilot Evidence

Pilot ledger:
`docs/benchmarks/kalman_qr_batched_xla_repair_phase6_gateb_r3_cpu_xla_pilot_2026-07-12.json`.

| Field | Value |
| --- | --- |
| SHA-256 | `1344f701eabfbec56e447b2cf40f3a8a4dd6cb79ef195659a50dfa4f03fb8ea2` |
| Byte count | `543644337` |
| Schema state | `complete_with_failures` |
| Update index | `2` |
| Records | exactly two `not_launched:trace_gate_not_passed` rows |
| Common invalidity | none |
| Final ledger checks | all `true` |
| Exact trace predecessors | one, bound to the trace ledger above |
| XLA invocations | `0` |
| Kalman XLA invocations | `0` |

The pilot file is large because its binding embeds the complete trace
predecessor. This is governance-artifact duplication. It is not evidence of
algorithm runtime, compiler, host, or device memory consumption.

## Authority And Run Manifest

| Field | Value |
| --- | --- |
| Git commit | `a644d29c5c2fd09a0deb3a7b5212799ff1fcb163` |
| Environment | `/home/ubuntu/anaconda3/envs/tfgpu/bin/python`; Linux x86_64; TensorFlow/TFP repository environment |
| Device policy | deliberate CPU diagnostic with `CUDA_VISIBLE_DEVICES=-1` before TensorFlow import |
| Requested threads | `OMP_NUM_THREADS=1`, `TF_NUM_INTRAOP_THREADS=1`, `TF_NUM_INTEROP_THREADS=1`; not a physical-core-pinning claim |
| Dtype / target | `float32`; `T=120`; dimensions 10/20/30; `P=50/150`; `B=1/4/16` |
| Trace JIT | disabled by the trace-only child contract |
| Pilot JIT | prospectively enabled, but no pilot child launched |
| Outer authority | 3000-second TERM deadline plus 45-second KILL grace; 3045-second hard ceiling |
| Supervisor exit | `1`, expected for the valid structural-rejection branch |
| Budget elapsed | `1758.622338253` seconds |
| Trust basis | deliberate GPU-hidden CPU diagnostic under the reviewed Gate B authority; not production-target evidence |
| Seeds/data version | `N/A`; deterministic fixture identities are proposal-bound and this was not a stochastic comparison |
| Plan | parent repair subplan named above, SHA-256 `3af2959c719e62b4fb02d9e7c78b3be86521d7e62b757d35d2e4acede679ba1a` |
| Result | this file |

The executed environment, outer timeout, Python executable, and complete Python
argv matched the immutable proposal command binding at
`docs/benchmarks/kalman_qr_batched_xla_repair_phase6_gateb_r3_budget_2026-07-12.json`,
SHA-256
`dd3a9495585a2f4b2995f1910da7d7b733f68467a285fa1856b5acf881f3886d`:

```bash
env CUDA_VISIBLE_DEVICES=-1 \
  OMP_NUM_THREADS=1 \
  TF_NUM_INTRAOP_THREADS=1 \
  TF_NUM_INTEROP_THREADS=1 \
  timeout --signal=TERM --kill-after=45s 3000s \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/run_kalman_qr_batched_xla_repair_2026_07_11.py \
  --phase6-pilot \
  --dimensions 10 20 30 \
  --parameter-counts 50 150 \
  --batch-sizes 1 4 16 \
  --timesteps 120 \
  --dtype float32 \
  --device cpu \
  --cpu-threads 1 \
  --jit-compile \
  --trace-child-timeout-seconds 60 \
  --xla-child-timeout-seconds 60 \
  --xla-cell-timeout-seconds 160 \
  --budget-contract \
    docs/benchmarks/kalman_qr_batched_xla_repair_phase6_gateb_r3_budget_2026-07-12.json \
  --budget-attestation \
    docs/benchmarks/kalman_qr_batched_xla_repair_phase6_gateb_r3_budget_attestation_2026-07-12.json \
  --trace-output-json \
    docs/benchmarks/kalman_qr_batched_xla_repair_phase6_gateb_r3_trace_census_2026-07-12.json \
  --output-json \
    docs/benchmarks/kalman_qr_batched_xla_repair_phase6_gateb_r3_cpu_xla_pilot_2026-07-12.json
```

The command used one supervisor process, 36 fresh trace children, and the
predeclared branch-pruning rule. The `--jit-compile` option applied only to the
prospective pilot branch; trace children were proposal-bound non-XLA work. The
command did not request GPU work or launch either proposed XLA child.

## Authority Ledger

| Artifact | SHA-256 / state |
| --- | --- |
| R3 proposal | `dd3a9495585a2f4b2995f1910da7d7b733f68467a285fa1856b5acf881f3886d` |
| R3 proposal review | `4d9b5ece79f50e1a1b98de8e1fc89391154947b401e52f97f4e7a322efe0bc5e`; `codex_substitute_weaker`; `VERDICT: AGREE` |
| R3 detached attestation | `9399be89c2263b0898c2a1c7718ca8484a81cb3fea31e8740c31750a0147e60f` |
| R3 authority ID | `7d630ff42cc759c02d3e6618c90b97923ec9a9e8cba5b99dd41ee94e09347a33` |
| R3 budget state | `a44904233d54b5906a96c0ba26c6e75e8ef33980e531ea7546d43099cbdefc28`; `closed`; update index 1 |
| R3 lease | `8005baec0a329003b46049b2aa21cfaf887e1934f8b60f8499a1f3aeb10ac14b`; `released`; generation 1 |
| Import discovery | `6db78f6a610e10681a10f4231ba0b32f798264065b12fe6f67290b8e5663719c` |
| Final skeptical runtime audit | `9e4a9fc3cb17358abf05d8749e29b76ab6e9a381e0969dcac371cf7b79eb30ec`; `AUDIT: PASS` |

Before launch, all proposal checks, all ten attestation checks, and independent
roster/budget checks passed. At close, both ledger bindings revalidated against
the current protected source and immutable authority bytes.

## Closure

- budget state is `closed` and the lease is `released`;
- no target supervisor or trace/XLA child survives;
- the R3 work root contains only `import_discovery.json`, `budget_state/`, and
  `trace/` with the preserved 108 trace files;
- no `pilot/`, `children/`, or `progress/` directory exists;
- the three protected algorithm hashes remain
  `ad1fc869...`, `d24ae436...`, and `bfde07b5...`;
- the immutable R2 archive/import/state/lease bytes remain preserved; and
- no runtime artifact was deleted, rewritten, compacted, resumed, or imported.

## Decision Table

| Decision | Primary criterion status | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Reject Gate B promotion and keep Gate C blocked | Failed: `trace_common_valid=false` | Structural promotion veto fired; no continuation invalidity fired | Alignment artifact versus expected axis data versus true topology specialization | Offline, preserved-GraphDef structural-difference diagnostic under a dedicated reviewed subplan | No numerical, XLA, memory, performance, ranking, GPU, production, HMC, or scientific conclusion |

## Inference-Status Table

| Evidence class | Status |
| --- | --- |
| Hard veto screen | Supported only for Gate B structural promotion: all six cohorts were rejected. No crash, divergence, non-finite result, or numerical veto was tested. |
| Statistically supported ranking | None; no timing samples or stochastic comparison ran. |
| Descriptive-only differences | Accepted/rejected coordinate counts and artifact sizes are descriptive diagnostics only. |
| Default readiness | Not established; Gate C and all GPU/Phase 7/default gates remain blocked. |
| Next evidence needed | Stable-identity offline GraphDef alignment, axis-constant consumer analysis, topology fingerprints, and negative mutation controls. |

## Post-Run Red Team

The strongest alternative explanation is that the current evaluator turns a
small number of node/function insertions, reorderings, or axis-dependent
constants into thousands of positional token differences. Conversely, a
name-aligned diagnostic could misleadingly erase a genuine dependency,
control-edge, operation, function-body, or numeric-constant specialization.
Therefore neither raw rejection counts nor a smaller keyed-diff count may be
used alone to weaken the gate.

Evidence that would overturn this close decision is a separately reviewed
offline diagnostic that proves the current rejection is evaluator-generated,
detects adversarial topology/value mutations, and preserves all raw graphs.
Even then it would justify only a prospective evaluator-repair phase and a new
runtime authority generation, not immediate XLA launch. The weakest current
evidence is semantic attribution of the rejected coordinates; the authority,
ledger, process, and branch-pruning evidence is strong.

## Forbidden Conclusions

This result does not establish:

- a Kalman numerical failure or success at the Gate B target lattice;
- a TensorFlow or XLA compile/runtime failure or success;
- memory reduction, absence of OOM, or repair of compile/codegen memory use;
- performance improvement, speedup, method superiority, or any ranking;
- CPU or GPU scalability;
- GPU, Phase 7, HMC, posterior, default, production, release, or scientific
  readiness.

## Handoff

The exact next artifact is
`docs/plans/bayesfilter-kalman-qr-batched-xla-repair-phase6-gatec-r3-trace-rejection-blocker-subplan-2026-07-12.md`.
Despite `gatec` in its filename, it authorizes only an offline diagnostic over
preserved GraphDefs. Gate C runtime remains blocked.
