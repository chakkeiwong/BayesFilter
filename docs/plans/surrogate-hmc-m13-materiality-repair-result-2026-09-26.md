# M13 materiality repair result — 2026-09-26

The bounded fresh-partition repair selected a common control pair under the
original 0.01 acceptance-probability allowance. The selected arm is the
weaker geometric family at multiplier 0.001, using calibration data1706/design
15 and validation data2706/design16. All eight nominal arms and proposal probes
were healthy; stress cases were bounded or explicitly flagged under the non-harm checks. The
selection is scoped to M13, T20, FP32/TF32 GPU/XLA force, the fixed Contract-E
reset, and the recorded endpoint. It is a tuning handoff, not posterior
evidence.

| Decision | Primary criterion | Veto diagnostics | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Select scoped controls | Common fresh calibration/validation pair with maximum acceptance-probability change <= .01 | No nominal/proposal validity failure; no stress non-harm veto | Three weaker arms were probed; trajectory sensitivity beyond this scope is unknown | Run the public artifact-authority tuner and reload its verified result | Posterior correctness, convergence, efficiency, superiority, or all-model validity |

| Inference status | Assessment |
|---|---|
| Hard veto screen | Passed for the selected control handoff |
| Statistically supported ranking | None; this is a control-selection screen |
| Descriptive-only differences | Arm-wise acceptance-probability changes |
| Default readiness | No change; FP32 remains the default force dtype |
| Next evidence needed | Public tuning, retained posterior draws, MCSE-calibrated endpoint comparison, and constructed cheap baselines |

The repair consumed 1,266.589423584 GPU seconds and 1,386.822697 CPU-worker
seconds. The wrapper report is `EXECUTION_FAILED` because two orchestration files were edited during execution to register the
future public-tuning stage; the nested repair
stage itself is `M13_MATERIALITY_CONTROLS_SELECTED` and `qualified=true`.

Artifacts:

* `docs/plans/artifacts/surrogate-hmc-closure-20260922-m13-materiality-repair-01/m13-materiality-repair/report.json`
* `docs/plans/artifacts/surrogate-hmc-closure-20260922-m13-materiality-repair-01/m13-materiality-repair/selection.json`

The repair does not establish that the surrogate force is the endpoint
gradient. The master continues with the separately declared endpoint and
public HMC tuning.

Source reconciliation: every saved source snapshot matches its initial hash,
all 34 selection-bound numerical dependencies remain unchanged, and an AST
comparison confirms the repair dispatch is identical. Both orchestration diffs
were inspected; they only register public tuning, its tests/source list and
output prefix. The separate source-change-review.json preserves that recovery;
the original report remains unchanged. The consumer rechecks the worker result,
review checksums, selection evidence and complete numerical source closure.
