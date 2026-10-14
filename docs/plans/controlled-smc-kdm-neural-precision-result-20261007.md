# Exact-input neural gradient localization — 7 October 2026

The failing float32 case is compatible between author and TensorFlow in float64
when both use exactly the same inputs. All 464 float64 checks pass. Across the
four originally failing gradient tensors, the largest float64 difference is
4.58e-14. The source float32 replay reproduces all captured outputs, loss,
gradient norm, gradients, clipping, parameters and Adam states exactly: 16
source-control checks pass across MDPF and A-MDPF.

The computed target is the author's stopped-sample, truncated-backpropagation
gradient program for this first update. Both implementations agree to the
stated float64 diagnostic precision on these inputs. This is strong evidence
against a derivative-translation error at the failing coordinates. It is not
a proof over every input or a comparison of trained-model tracking accuracy.

The unchanged float32-to-float32 screen still fails 16 checks. Comparing each
implementation with the common same-input author float64 result shows why
author float32 output should not be treated as exact truth here:

| Gradient tensor | Author float32 maximum absolute error | TF float32 maximum absolute error | TF/author float64 maximum difference |
|---|---:|---:|---:|
| Particle encoder layer 4 weight | 5.3000e-5 | 2.4807e-5 | 3.2807e-14 |
| Proposal layer 0 weight | 5.2730e-5 | 2.6326e-5 | 3.2085e-14 |
| Proposal layer 2 weight | 4.0765e-5 | 1.9724e-5 | 2.3981e-14 |
| Proposal layer 4 weight | 7.7731e-5 | 4.1014e-5 | 4.5741e-14 |

Errors are maximum absolute distances over each tensor to the same-input
author float64 calculation. The same first-update result occurs in both
variants. TF also has smaller L2 error on each of these four tensors.
These are deterministic observations on a selected diagnostic fixture, not
a statistically supported ranking of backends, methods or trained models.
No tolerance was relaxed and no runtime algorithm was changed.

## Decision and inference

| Decision | Primary criterion | Veto | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Accept fixed-fixture high-precision localization | 464/464 float64 and 16/16 exact source replay checks | None in this local test | Unseen input regimes and finite float32 sensitivity | Develop independent accuracy/conditioning checks for the remaining numerical comparisons | General derivative proof |
| Preserve float32 diagnostic failure | 16 unchanged raw/clipped-gradient checks fail | Strict source-output comparison | Author float32 also differs from the high-precision result | Assess numerical accuracy relative to reference and actual downstream effects | TF is less accurate than author |
| Preserve full-replication incompleteness | Statistical NLL/RMSE qualification not run | Circular and cSMC fitting gaps remain | Independent trained-unit variability and power | Source-only calibration and untouched qualification after implementation checks | Replication accepted |

| Inference status | Finding |
|---|---|
| Hard veto screen | Preserved float32 discrepancy; no nonfinite value or update failure |
| Statistically supported ranking | None |
| Descriptive-only differences | Fixed-tensor errors and runtime |
| Default-readiness | Not established |
| Next evidence needed | Independent numerical calibration on fresh regimes; downstream and trained-unit noninferiority evidence |

The source runs CPU-only with GPU hidden; TF uses GPU/XLA, TF32 off and
verified memory growth. Both operate on the preserved fresh-A float32 inputs,
parameters and constants, cast exactly for the double comparison. One actual
optimizer update avoids accumulated state differences. Source controls validate
that initial-state replay did not change the tested first-update gradient.

Attempt 02 consumed 4.2733 CPU seconds and 41.8486 GPU seconds. The shared
three-attempt diagnostic allowance has one attempt remaining; no new compute
grant was used. Selected values and coordinates are in
kdm-neural-angle-20261007-02/precision_analysis.json; full inputs, gradients,
source controls, logs, commands and hashes are alongside it.

Post-run red team: a wrong operation shared by both implementations or a
different input regime is not excluded by agreement on this selected fixture.
The independent source consumer and exact replay controls reduce harness risk.
A same-input double mismatch on a fresh regime, omitted source term, or
downstream degradation would overturn a broader compatibility conclusion.
