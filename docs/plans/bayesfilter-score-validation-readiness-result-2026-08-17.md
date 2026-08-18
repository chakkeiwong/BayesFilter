# GenUT Score Validation Readiness Result

Date: 2026-08-17
Plan: `docs/plans/bayesfilter-genut-score-validation-readiness-plan-2026-08-16.md`
Artifact root: `docs/benchmarks/artifacts/genut-score-validation-readiness-20260817/`
Git checkout: `dae37183bf4421682b2ad991e2dc0d0f3c53f260`

## Decision

`KALMAN_CLOSE_ORACLE_GATE_REMOVED_UNSPECIFIED; NEUTRA_HMC_REMAINS_CLOSED`

The prior exact-Kalman closeness gate is withdrawn from the decision record:
the plan and runner did not specify or enforce a numerical value/score
tolerance, so it was not a reproducible scientific gate. Its reported oracle
errors remain descriptive diagnostics only. Phase 1 infrastructure passed and
the legacy dual-cap route is finite/residual-valid, but NeuTra/HMC remains
closed for the independent blockers listed below.

## Evidence Contract Status

| Gate | Status | Evidence |
|---|---|---|
| GPU visibility and memory growth | PASS | Trusted probe: RTX 4080 SUPER and RTX 5080; growth verified before logical devices |
| Python compilation | PASS | `python -m compileall -q bayesfilter docs/benchmarks scripts tests` |
| Sylvester custom-op regression | PASS | `tests/test_symmetric_sylvester_tf.py`: 11 passed |
| LGSSM finite execution | PASS | `phase2_lgssm_oracle/result.json`, both diagonal and dual-cap arms |
| Exact Kalman value/score agreement | NOT A GATE | No reviewed tolerance was declared or enforced; errors retained descriptively |
| Same-program derivative regression | IN PROGRESS | Single-step FD replaced by an h^2 regression diagnostic; current-scope rerun pending |
| Austria endpoint identity | BLOCKED | Tangent-free and tangent-carrying routes do not represent the same finite scalar |
| NeuTra/HMC readiness | CLOSED | Austria blocker, scope-tuning gaps, and no completed target-specific training/HMC evidence |

## Executed Phases

### Phase 1: Regression Closure

Passed on 2026-08-17:

- trusted `nvidia-smi` probe: RTX 5080 (`sm_120`) and RTX 4080 SUPER (`sm_89`);
- TensorFlow `2.20.0-dev0+selfbuilt` GPU probe with memory growth verified on
  both physical devices before logical-device creation;
- Python compilation for `bayesfilter`, `docs/benchmarks`, `scripts`, and
  `tests`;
- `tests/test_symmetric_sylvester_tf.py`: 11 passed, including eager, graph,
  CPU-XLA, custom-op layout, residual, and validation checks.

### Phase 3: Four-Model Current-Scope Diagnostics

Artifact: `phase3_four_model/result.json` and `result.md`.
All 16 cells (four models times four legacy arms) passed finite-value,
program-valid, and residual hard gates over the 16 claim seeds. This is
viability evidence only. Same-program finite-difference diagnostics passed for
KSC-SV and failed for LGSSM, predator-prey, and Austria-SIR. Austria-SIR had
large score variability and normalized same-program residuals up to about
`51.5`; this is not a numerical promotion result.

### Phase 4: LM/Trust-Region Diagnostic

Artifact: `phase4_lm_trust_region/result.json`.
The explicit route identifier
`genut_column_scaled_lm_smooth_rms_trust_v1` was exercised for its existing
LGSSM `T=10`, `N=1008` replay with damping `0.01`, scale floor `1e-4`, and
trust radius `0.5`. It returned `PASS_FINITE` with all program-valid flags,
but the route is labeled `local_repair_diagnostic_not_admission`, is not
tuned for the current `T=50` claim scope, and has no exact Kalman agreement.
It therefore does not rescue the readiness gate.

## Phase 2 Summary

Scope: LGSSM `T=50`, `N=1008`, seeds `98201..98204`, FP32/XLA/no-TF32.
The Kalman value oracle is `-136.07597463460453` with score
`[5.6554462, -3.8350569, 0.3023619, -1.9171764, 4.3542756]`.

| Arm | Finite | Mean absolute value error | Maximum absolute value error | Maximum absolute score error |
|---|---:|---:|---:|---:|
| diagonal | yes | 0.3120 | 0.5419 | 5.2275 |
| dual-cap | yes | 0.3298 | 0.6248 | 5.4009 |

The previous central finite-difference ladder was a single-step diagnostic in
the four-model runner and did not provide a regression criterion. It is being
replaced by the declared quadratic-in-step regression diagnostic in
`docs/benchmarks/genut_fd_regression.py`. The Kalman distances above remain
descriptive and are not used as a pass/fail gate.

## Interpretation

The harness, GPU backend, custom op, and finite-value program are not invalidated
by the descriptive Kalman distances. They do not establish exact filtering or
particle-approximation accuracy. The current hard scientific blocker is the
Austria tangent-free/tangent-carrying endpoint mismatch; the LGSSM, predator-
prey, and Austria same-program derivative diagnostics require the regression
rerun before any score-consistency conclusion.

The dual-cap arm is not promoted over diagonal: the observed differences are
descriptive, with four seeds and no uncertainty analysis supporting a ranking.

## Remaining Issues

1. Complete the h^2 regression diagnostic on current-scope LGSSM, predator-prey,
   and Austria runs; branch changes or nonfinite endpoints remain hard failures.
2. Repair the Austria endpoint mismatch before Austria NeuTra training.
3. Produce fresh scope-specific tuning artifacts for regenerated LGSSM/KSC
   targets; inherited controls remain warm starts only.
4. Validate the repaired trust-region route on the actual dual-cap `T=50` scope
   before treating it as an option for training.
5. Only then run target-specific batch-native NeuTra training, heldout checks,
   and sequential HMC diagnostics.
