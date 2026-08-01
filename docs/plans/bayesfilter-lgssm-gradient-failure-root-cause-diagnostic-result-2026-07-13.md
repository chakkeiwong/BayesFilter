# LGSSM Gradient Failure Root-Cause Diagnostic Result

Date: 2026-07-13

Status: `ROOT_CAUSES_LOCALIZED_REPAIR_REQUIRED`

## Bottom Line

The LGSSM result contains two separate failures.

1. The reported production `same_scalar_finite_difference` comparison is not a
   same-numerical-program comparison. The score shard evaluates the primal
   returned by the compact forward-sensitivity graph, while the FD shard calls a
   separately compiled value graph with a duplicated, algebraically equivalent
   LEDH-flow implementation. With GPU TF32 enabled, those graphs already return
   different center values. Therefore the reported `q_scale` comparison
   `11.2701368` versus `5.5357637` cannot diagnose the derivative of one scalar.
2. Independently of that harness defect, the active deterministic barycentric
   OT reset makes the fixed-noise LEDH scalar's `q_scale` derivative positive and
   far from the exact Kalman likelihood gradient. Tape and FD agree on that
   positive derivative in the completed CPU float64 same-function check.
   Independently compiled GPU graphs also remain positive with TF32 disabled.
   Removing the reset at production particle count changes both GPU derivatives
   back to the correct sign and near the Kalman magnitude.

The leading mechanism within the reset is loss of second moments: deterministic
barycentric projection contracts the particle covariance before the next time
step. A moment-restoration intervention strongly reduces both historical value
bias and the new gradient error, but it is diagnostic code at small particle
count, not a production repair. A second transport defect also exists: the
streaming path measures row-mass residual but applies the transported rows
without normalizing them. Row normalization alone did not repair the dominant
historical LGSSM error, so it is not established as the primary gradient cause.

## Claimed And Computed Quantities

| Item | Quantity | Verdict | Evidence anchor |
| --- | --- | --- | --- |
| Scientific target | Gradient of the transition-first observed-data LGSSM log likelihood with respect to `(phi1, phi2, phi3, q_scale, r_scale)` | `correctly represented by the Kalman oracle` | `bayesfilter/linear/kalman_tf.py::tf_kalman_log_likelihood`; model construction in `docs/benchmarks/benchmark_ledh_same_target_lgssm_m3_t50_value.py::_lgssm_components` |
| Kalman `q_scale` result | Float64 TensorFlow autodiff: `-1.9171802706` | `correct for the stated target` | Every completed precision diagnostic records the same oracle; independent central FD at the production-sized step was approximately `-1.90764` |
| Original compact score | Manual compact JVP of the active-reset compact numerical graph: `+11.2701368` | `wrong relative to the Kalman-gradient claim` | `docs/plans/artifacts/complete-highdim-leaderboard/phase2-ledh-repair1/lgssm/lgssm-t50-n10000-seed81120-score.json` |
| Original reported FD | Central FD of the separately compiled active-reset value graph: `+5.5357637` | `not the derivative oracle for the compact score graph under TF32` | `docs/plans/artifacts/complete-highdim-leaderboard/phase2-ledh-repair1/lgssm/lgssm-t50-n10000-seed81120-fd.json` |
| Active-reset differentiated value | Derivative of the deterministic finite-`N` LEDH scalar including OT reset | `different from the Kalman target` | At CPU float64 `N=16,T=50`, tape and central FD of the same value function both give `+12.768496`; separate compact/value GPU graphs are also positive at `N=256` with TF32 on and off |
| No-reset derivative | Derivative of a fixed-noise LEDH importance recursion without carried OT reset | `approximately related to the Kalman target at N=10000, but not a production replacement` | With TF32 disabled, compact JVP/FD are `-1.750659/-1.737774`, versus Kalman `-1.917180` |

The accepted `d=3,T=50` scalar relative value bias of approximately `0.0852%`
is not a gradient criterion. Its acceptance does not alter any verdict above.
The owner-directed `0.05 * sqrt(p)` relative tolerance applies only to an FD
check of the same scalar; it does not define acceptable LEDH-versus-Kalman
gradient error.

## Root Cause 1: False Same-Scalar Wiring

The unified harness compiles two independent functions:

- score: `benchmark_ledh_compact_score_gpu_xla.py` calls
  `_compact_score_tensor_outputs`, which reaches
  `_compact_value_and_score_from_components`;
- FD: the same harness calls `_value_tensor_outputs`, which reaches
  `_same_target_value_from_components`.

The compact routine contains its own primal LEDH-flow algebra alongside the
tangent propagation. The value routine calls the shared
`_batched_ledh_linearized_flow_with_aux_tf` primitive. These are mathematically
equivalent in exact arithmetic but are not the same compiled numerical program.
There is also a callable-contract mismatch at stationary initialization:
`_compact_score_tensor_outputs` passes the center-prepared `initial_particles`
unchanged for every candidate, while the compact routine injects the nonzero
candidate derivative `initial_noise * d_initial_std`. The value/FD adapter does
rebuild endpoint particles from `initial_noise`. Thus the reported compact
tangent includes the derivative of an operation that is not in the literal
compact callable's perturbed primal. This does not cause the measured center
gap, but it means a repaired compact-self-FD callable must reconstruct initial
particles inside the same graph before its returned primal can be the derivative
target.
The artifact fields `same_route_value_score` and
`same_scalar_finite_difference` are therefore false for production TF32.

### Causal Evidence

| Case | Compact center | Value-route center | Compact minus value |
| --- | ---: | ---: | ---: |
| `N=10000,T=1`, TF32 enabled, no reset | `-7.0097599` | `-7.0053306` | `-0.0044293` |
| `N=10000,T=50`, TF32 enabled, no reset | `-135.6805420` | `-135.7762909` | `+0.0957489` |
| `N=10000,T=1`, TF32 disabled, no reset | `-7.0077415` | `-7.0077415` | `0.0` |
| `N=10000,T=50`, TF32 disabled, no reset | `-135.6945953` | `-135.6945953` | `0.0` |

Reconstructing candidate-dependent initial particles at the center changes
neither center value; the measured contribution is exactly `0.0`. That center
identity does not clear the endpoint callable mismatch described above. At
`T=1`, operation-level
decomposition locates the first TF32 graph-context differences in contractions:

- observation projection `H*x`: maximum difference `9.787e-4`;
- pseudo-observation: maximum difference `9.787e-4`;
- information vector: maximum difference `5.614e-3`;
- post-flow particles: maximum difference `3.759e-4`.

This is a harness/graph-identity bug, not evidence that TF32 alone makes the
active-reset Kalman gradient positive. The active-reset sign failure survives
with TF32 disabled.

## Root Cause 2: Carried Deterministic OT Reset

After every active time step, `_manual_forward_transport_tf` replaces the
weighted post-flow cloud with

```text
transported = transport_matrix @ post_flow
next_particles = transported
next_log_weights = uniform
```

The next LGSSM transition starts from these transported particles. Thus the
reset changes all future likelihood increments. The exact Kalman gradient is
the oracle for the intended observed-data likelihood; a derivative of this
modified finite-`N` recursion need not equal it.

### Necessary-Cause Evidence

| Intervention | Precision/shape | Compact JVP or value-tape derivative | FD | Graph relation | Kalman |
| --- | --- | ---: | ---: | --- | ---: |
| Active reset | GPU/XLA, TF32 on, `N=256,T=50` | compact `+11.247657` | value `+11.397069` | separate compiled graphs; sign evidence only | `-1.917180` |
| Active reset | GPU/XLA, TF32 off, `N=256,T=50` | compact `+11.341722` | value `+11.352114` | separate compiled graphs; center values coincide | `-1.917180` |
| No reset | GPU/XLA, TF32 off, `N=10000,T=50` | compact `-1.750659` | value `-1.737774` | separate compiled graphs; center values coincide | `-1.917180` |
| Active reset | CPU float64, `N=16,T=50` | value tape `+12.768496` | value `+12.768496` | identical value function | `-1.917180` |
| Moment-restored reset | CPU float64, `N=16,T=50` | value tape `+1.552604` | value `+1.552604` | identical value function | `-1.917180` |

At `N=16`, the no-reset one-seed derivative is itself noisy (`+17.67`), so that
small case is not evidence that removing reset is sufficient. The decisive
no-reset sign repair is the completed `N=10000,T=50` arm. The CPU float64
active-reset tape/FD identity proves that a positive derivative can be a real
derivative of the reset-modified value scalar rather than only a manual-JVP
implementation error. The completed `N=256` GPU arms show that the positive
behavior persists in both separately compiled production-style graphs and is
not removed by disabling TF32; they are not claimed as same-program FD evidence.

A full active-reset `N=10000,T=50` compact-self-FD diagnostic was attempted but
did not produce a terminal artifact. This result does not claim that exact
comparison completed.

## Leading Internal Mechanism: Covariance Contraction

Historical reset interventions already isolated second-moment loss:

| Historical `d=2` reset arm | Covariance trace ratio at the first reset | Absolute value gap to Kalman |
| --- | ---: | ---: |
| Current OT | `0.3571` | `0.8549` |
| Row-normalized OT | `0.3584` | `0.8546` |
| Moment-restored OT | `0.9999` | `0.1003` |

The reset is a deterministic barycentric map. Even when its transport weights
represent the desired weighted measure, replacing each target particle by a
conditional mean removes conditional variance. The resulting under-dispersed
cloud is then carried into later transition and observation corrections. Since
`q_scale` controls transition noise and the stationary initial covariance, this
repeated covariance loss creates a particularly large distortion in its local
slope.

The new gradient intervention is consistent with that mechanism: at
`N=16,T=50`, moment restoration moves `q_scale` from `+12.7685` to `+1.5526`,
with tape and FD agreeing within about `4e-8`. That is strong causal support,
but it is not proof that covariance restoration is the complete production-scale
repair. The intervention is one seed, small `N`, CPU float64, non-XLA, and uses
diagnostic moment-restoration code.

## Secondary Transport Defect

`_filterflow_streaming_transport_from_potentials` accumulates both transported
rows and `row_mass`, then returns the transported rows unchanged. The caller
discards `_row_residual` and installs those rows directly. Historical
`N=10000,T=50` runs reported maximum row residual about `0.96355`, so this is a
real transport-contract defect.

It is not established as the dominant LGSSM gradient mechanism. In the
historical controlled reset diagnostic, explicit row normalization drove the
row residual to approximately `2.4e-7` but left the `d=2` value gap and
covariance contraction essentially unchanged. A production repair still must
normalize rows and propagate the quotient-rule tangent/VJP; it must not be sold
as sufficient evidence that the Kalman gradient is repaired.

## Ruled-Out Hypotheses

| Hypothesis | Verdict | Basis |
| --- | --- | --- |
| Kalman likelihood implementation is wrong | `ruled out` | Float64 autodiff agrees with independent central FD; model, observations, time convention, and stationary initial covariance match the LEDH target |
| `q_scale` is parameterized differently | `ruled out` | Both paths use `Q=q_scale^2 I`, transition noise multiplier `q_scale`, and stationary standard deviation `q_scale/sqrt(1-phi^2)` |
| Candidate initial particles are not rescaled in the existing value-FD endpoints | `ruled out` | The value/FD adapter reconstructs them from fixed initial noise |
| Compact callable primal includes candidate-dependent initialization | `false; latent graph-contract bug` | It freezes prepared center particles while the reported tangent includes `initial_noise * d_initial_std`; reconstruction must move inside the canonical graph |
| TF32 causes the Kalman sign reversal | `ruled out` | Active-reset score remains about `+11.34` with TF32 disabled; no-reset production-scale score is negative |
| Generic compact JVP algebra causes the Kalman sign reversal | `ruled out as a sufficient explanation` | Focused primitive tests pass, and the independently autodifferentiated value function has the same positive-sign failure; full production compact-self-FD remains uncompleted |
| Clipped pairwise-distance active set is the dominant cause | `not supported` | No causal intervention tied the sign reversal to clipping; reset and moment interventions are discriminating |
| Small accepted value bias validates the gradient | `wrong` | A scalar value at one point does not constrain its local slope |

## Decision Table

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Reject the existing LGSSM same-scalar FD artifact | `FAIL`: center primal identity fails under TF32 | No input/randomness mismatch; failure is graph identity | A true production-scale compact-self-FD artifact is still missing | Return the compact graph's primal through the FD callable and require exact center-primal identity before FD | No statement that the compact JVP itself is wrong by `50.9%` |
| Reject the active-reset LGSSM score as an estimator of the Kalman likelihood gradient | `FAIL`: sign and magnitude disagree | Same-program tape/FD confirm the positive active-reset slope | Multi-seed production-scale uncertainty is not yet measured | Repair reset semantics, then run prefix and multi-seed Kalman-gradient gates | No HMC, posterior, or leaderboard readiness |
| Treat covariance contraction as the leading repair target | `SUPPORTED`, not production-proven | Moment restoration is diagnostic only | Completeness of mechanism and scalable differentiable implementation | Design a reviewed second-moment-preserving reset and quotient-correct row normalization | No approval of the diagnostic restoration code |

## Inference-Status Table

| Evidence class | Status |
| --- | --- |
| Hard veto screen | Existing LGSSM gradient admission is vetoed by false same-scalar labeling and material Kalman-gradient disagreement. Values are finite and artifacts are otherwise reproducible. |
| Statistically supported ranking | None. No method ranking was attempted, and one-seed interventions do not support one. |
| Descriptive-only differences | Exact per-arm slopes, value gaps, covariance ratios, and row residuals are descriptive outside their deterministic same-program/causal-intervention roles. |
| Default-readiness | Not ready. The repository's TF32/GPU default remains a direction, but this LGSSM gradient route has an unresolved correctness veto. |
| Next evidence needed | Canonical one-graph primal/JVP/FD identity; row-normalized quotient JVP; reviewed moment-preserving reset; `T=1,10,50` checks; then `N=10000,T=50` multi-seed Kalman-gradient comparison with uncertainty. |

## Repair Order

1. Make objective and gradient originate from one canonical numerical graph.
   The FD callable must return the compact score graph's primal, not recompute an
   algebraically equivalent value in a separate function. Candidate-dependent
   initial particles must be reconstructed inside that graph before both primal
   and tangent propagation.
2. Add a mandatory center-primal identity gate. If score-center and FD-center
   values differ at all, the artifact must be invalid and must not say
   `same_scalar`.
3. Keep TF32 as the production default, but use a disclosed TF32-disabled
   correctness arm until the true same-program production check is available.
4. Normalize transport rows before applying them and implement the corresponding
   quotient-rule JVP/VJP.
5. Design and review a differentiable second-moment-preserving reset. Do not
   promote the small diagnostic `_restore_moments` helper directly.
6. Re-run fixed-noise same-program checks at `T=1,10,50`, then
   `N=10000,T=50`, followed by multi-seed comparison to the exact Kalman gradient
   with uncertainty intervals.

## Run Manifest

| Field | Value |
| --- | --- |
| Git commit recorded by GPU diagnostics | `d269f5bbd8531b878d4f25897a357fbc8f172488` |
| Environment | `/home/chakwong/anaconda3/envs/tf-gpu`; TensorFlow/TFP repository environment |
| Production-target diagnostics | Visible GPU, XLA JIT, float32, TF32 enabled and disabled as recorded per JSON |
| Reference interventions | CPU explicitly selected with `CUDA_VISIBLE_DEVICES=-1`, float64, non-XLA |
| Seed | `81120` |
| Shapes | `N=16`, `N=256`, and `N=10000`; `T=1` and `T=50` as recorded per artifact |
| Plan | `docs/plans/bayesfilter-lgssm-gradient-failure-root-cause-diagnostic-plan-2026-07-13.md` |
| Artifacts | `docs/plans/artifacts/lgssm-gradient-root-cause-2026-07-13/` |
| Result | This file |
| Focused test command | `CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp pytest -q tests/test_ledh_compact_transport_jvp.py tests/test_ledh_lgssm_manual_score_phase4.py` |
| Focused test result | `20 passed, 2 warnings in 13.36s` |

Exact GPU commands and wall times are preserved in each JSON artifact's
`run_manifest`. The CPU reset artifacts record their device, dtype, seed,
particle count, horizon, FD step, and per-variant wall time. Full
`N=10000,T=50` compact-self-FD attempts that ended without a terminal artifact
are deliberately excluded from positive evidence.

## Post-Run Red Team

The strongest alternative explanation is finite-`N`, one-seed pathwise Monte
Carlo variability. It clearly matters at `N=16`, where even the no-reset slope is
poor. It does not explain the whole result: the no-reset `N=10000` slope has the
correct sign and near-oracle magnitude, while both active-reset `N=256`
production-style graphs have positive slopes near `+11` with TF32 on and off.
Still, a statistically supported
claim about residual bias after repair requires multiple production-scale seeds.

The weakest evidence is the production-scale sufficiency of moment restoration.
The historical covariance/value result and small gradient intervention identify
the leading mechanism, but only a reviewed scalable implementation followed by
the declared multi-seed Kalman-gradient gate can establish that the repair is
complete.
