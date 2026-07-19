# Focused Native Review: SSL-LSTM A4 HMC Acquisition

Date: 2026-07-14 (Asia/Shanghai)

Review type: `FOCUSED_NATIVE_READ_ONLY_REVIEW`

Status: `AGREE_AFTER_REPAIR`

## Scope

Reviewed the bounded A4 HMC acquisition lane:

- `docs/plans/bayesfilter-ssl-lstm-completion-phase-a4-calibration-design-freeze-subplan-2026-07-11.md`;
- `docs/benchmarks/run_ssl_lstm_a4_hmc_acquisition_2026_07_14.py`;
- `tests/test_ssl_lstm_a4_hmc_acquisition.py`;
- the six public A4 HMC JSON artifacts and relevant private manifests/tensor
  hashes under
  `docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a4/hmc-acquisition/`;
- `docs/plans/bayesfilter-ssl-lstm-completion-phase-a4-hmc-acquisition-blocker-result-2026-07-14.md`; and
- `docs/plans/bayesfilter-ssl-lstm-completion-phase-a4-hmc-acquisition-repair-plan-2026-07-14.md`.

The review checked target/geometry identity, affine orientation and chain rule,
full-chain authority scope, source and budget lineage, trusted GPU/XLA
placement, archive readback, movement/acceptance semantics, divergence
qualification, sequential stopping, result claims, and the repair boundary.
Claude was not used or required.

## Findings And Repairs

| Severity | Finding | Repair | Recheck |
| --- | --- | --- | --- |
| Material engineering | The first adapter implementation used TensorFlow matrix multiplication for a rank-one latent vector, causing a scalar-path dimension error while the batch path was correct. | Replaced the scalar/batch affine and score maps with rank-agnostic `tf.tensordot` contractions and added exact scalar orientation/chain-rule tests. | Real A1 scalar/batch delegation and finite-difference checks passed; full focused suite passed `11/11`. |
| Material canary-role | The original eight-retained-draw canary required every chain to move. One finite GPU/XLA canary had movement `[false,true,true,true]`, so the mechanics canary was conflating compilation/archive validity with tuning evidence. | Preserved the failed artifact and its `424.92757005395833`-second budget charge; amended only the canary role so partial movement is a tuning repair trigger while all-chain movement remains mandatory for tuning and acquisition; ran a distinct seed/label retry. | Repaired canary passed with movement `[true,true,true,true]` and acceptance `[0.875,0.875,0.5,0.625]`; no sampler-admission threshold was weakened. |
| Moderate claim precision | The first blocker draft stated that the 64-draw tuning screen was “too short” to reveal the retained failure. One realized path does not identify screen length as the cause. | Rephrased the result to state only that the tuning screen selected a kernel that later failed, and classified screen length, warmup path, and start-specific geometry as unresolved hypotheses. | Result now separates observation, repair hypothesis, and nonconclusion. |

## Verification

| Check | Result |
| --- | --- |
| Focused test suite | `11 passed`; warnings are TensorFlow/Gast deprecations, not test failures |
| Compile | Harness and tests compile under the `tfgpu` interpreter |
| Diff whitespace | `git diff --check` passed for the lane files |
| A1 target identity | Scope and semantic/adapter SHA-256 values match the locked A1 contract |
| Geometry | `factor_z @ factor_z.T` residual `8.881784197001252e-16` below tolerance `1.1465297583454372e-13` |
| CPU target/transform | Passed; value/score transform residuals `0.0`; finite-difference residual `3.951489690682646e-09` |
| GPU placement | Repaired canary, tuning, and retained-rung state/target outputs report `GPU:0`; two RTX 4080 SUPER devices recorded |
| Tuning | First predeclared balanced candidate selected; four chains moved; per-chain acceptance within `[0.20,0.95]` |
| Retained admission | Correctly failed: chain movement `[false,true,true,true]`; acceptance `[0.0,0.492,0.652,0.608]` |
| Finiteness | Retained samples, final state, target values, and log-accept ratios finite |
| Native divergence | `not_exposed_by_kernel`; no zero-divergence claim made |
| Budget | All four GPU attempts charged; `1333.7487312000012` seconds consumed of `28800` |

The private retained shard hash in the result matches the manifest and file:
`d39c1d198171cb0d0b9ec3d234f3193f9addd5654fc8298941ebcedc72ba5667`.
The public result hashes match the current public JSON artifacts.

## Interpretation Review

The blocker is correctly scoped. The evidence supports these hard statements:

- no existing artifact qualified;
- the target/transform/archive route is engineering-valid for the reviewed
  fixtures;
- the balanced kernel failed the serious retained admission gate because one
  chain accepted no transitions; and
- no HMC draws are currently admissible for A4 forecast calibration.

It does not support posterior incorrectness, an HMC-direction rejection,
sampler superiority, predictive equivalence, NeuTra readiness, model adequacy,
or default readiness. The native-divergence limitation is stated correctly.

Stopping was required for this acquisition attempt. The observed result
invalidated the current four-chain calibration artifact, not the target,
harness, or research direction. Because the plan made an unmoved acquisition
chain a continuation veto, extending `segment_0` or silently restarting under a
different kernel would have violated the prospective contract. The separate
repair plan is the correct handoff.

## Repair Plan Review

The half-step/eight-leapfrog repair is technically coherent and was present in
the original predeclared kernel ladder. It preserves trajectory length, target,
geometry, starts, thresholds, and four-chain admission. It also prohibits
reusing the invalid archive or selecting seeds post hoc.

The repair is not authorized merely because `7.6295` GPU-hours remain. It is a
fresh acquisition attempt after a fired continuation veto and must use new
artifacts, full prior budget lineage, focused tests, and a visible execution
decision. This is a boundary condition, not a scientific objection to running
the repair next.

## Verdict

`VERDICT: AGREE`

The current A4 HMC acquisition result is correctly
`BLOCKED_INVALID_CALIBRATION_INPUT_REPAIR_REQUIRED`. No forecast calibration,
A5, or NeuTra work should proceed from the failed archive. The separately
prospective smaller-step repair is the next technically justified experiment.

