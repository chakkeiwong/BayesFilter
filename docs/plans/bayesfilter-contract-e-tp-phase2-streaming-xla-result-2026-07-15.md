# Contract E--TP Phase 2 Streaming And XLA Result

metadata_date: 2026-07-15
phase: 2
status: PASS_PHASE2_STREAMING_XLA
master_plan: `docs/plans/bayesfilter-contract-e-tp-all-model-gradient-comparison-master-plan-2026-07-15.md`

## Outcome

The experimental Contract E--TP core now supports a two-pass streaming teacher
reduction and fixed-anchor square projection. A model-owned traced block
program receives compact source tensors and emits one padded block of teacher
points, unnormalized log weights, and feature values. The core retains only
the feature reduction, frozen anchors, student weights, diagnostics, source
tensors, and one block. It does not accept or return a dense teacher tensor.

Forward JVP recomputes teacher blocks. Reverse VJP also recomputes blocks and
accumulates one adjoint per compact source, so no dense candidate or reverse
history tensor is retained by the owned core. Factories bind block program and
block size into `tf.function(jit_compile=True)` evaluators.

## Correctness Evidence

The CPU-hidden float64 suite compares dense and streaming execution over block
sizes 2, 4, 7, and 16, including a non-divisible final block. It also checks:

- candidate-permutation invariance with correspondingly remapped fixed anchors;
- streaming primal, manual JVP, and manual VJP parity with the dense program;
- streaming manual JVP/VJP parity with TensorFlow autodiff; and
- fixed XLA-default factory construction.

Together with Phase 1, the focused suite reports `19 passed`.

## Compiled Fail-Closed Repair

The first successful GPU/XLA compilation emitted TensorFlow warnings that XLA
ignores `Assert` operators inside compiled clusters. That made eager execution
fail closed but left the compiled path unable to support the same claim. The
first artifact under `phase2_gpu_xla_smoke_20260715/` is therefore an incomplete
diagnostic and is superseded; its `status: PASS` must not be used as the Phase 2
gate.

The repair binds validation into the numerical graph:

- input, mass-feature, fixed-index, anchor-hit, row-scale, finite, rank,
  condition-roundoff, residual, and positive-weight predicates contribute to
  `valid_chart`;
- invalid charts NaN-poison all carried student points and weights; and
- eager assertions remain as immediate diagnostics where TensorFlow executes
  them.

The repaired XLA smoke includes an invalid block program whose mass feature is
2 rather than 1. On compiled GPU execution it produced `valid_chart=false` and
NaN-poisoned both student points and weights. This explicitly closes the XLA
assertion gap for the carried outputs.

## GPU/XLA Artifact

The controlling artifact is:

`docs/benchmarks/artifacts/contract_e_tp_all_models_2026_07_15/phase2_gpu_xla_smoke_repair1_20260715/phase2_gpu_xla_smoke.json`.

| Field | Result |
| --- | --- |
| GPU | NVIDIA GeForce RTX 4080 SUPER, 16376 MiB, driver 591.86 |
| TensorFlow | 2.19.1, `/physical_device:GPU:0` |
| XLA | Forward, manual JVP, and manual VJP compiled and executed |
| Dtype / TF32 | float64 / TF32 enabled but irrelevant to float64 matrix arithmetic |
| Graph | 430 operations, two while operations, zero TensorList operations |
| Dense teacher output | Absent |
| Tiny-fixture allocator peak | 49,664 bytes |
| Valid fixture | `valid_chart=true`, minimum weight `0.0987638731990897` |
| Invalid compiled fixture | `valid_chart=false`, carried outputs NaN-poisoned |

This is an engineering composition smoke, not a scaling result. Measured peak
memory on 15 candidates cannot establish asymptotic behavior by itself; the
API shape and graph inspection support the no-dense-retention claim, while
larger measured scaling remains Phase 9 work.

## Attempt And Repair Record

1. Direct script launch failed before execution because the repository root was
   absent from `sys.path`.
2. The next launch initialized the real GPU, then failed because memory growth
   was set after package import had initialized TensorFlow.
3. Moving allocator policy before TensorFlow import allowed XLA execution, but
   the log exposed ignored assertions. That artifact is noncontrolling.
4. Explicit validity predicates and NaN poisoning were added. The full CPU
   suite passed, and the repaired GPU/XLA artifact passed its positive and
   negative compiled fixtures.

These were harness and compiled-validation repairs. Candidate construction,
feature target, fixed anchors, projection equations, and evidence thresholds
were unchanged.

## Decision And Handoff

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Close Phase 2 | Pass | No dense/stream derivative or compiled fail-closed veto | Recursive time order and chart preparation not tested | Implement Phase 3 recursive LGSSM oracle ladder | No full-horizon scaling, nonlinear validity, canonical/default/leaderboard/HMC admission |

Phase 2 gate: `PASS`. Phase 3 must distinguish the bootstrap witness from the
corrected LEDH finite teacher, preserve nonuniform carried weights, and compare
to the differentiated Kalman oracle only after same-program derivatives pass.
