# AR2 force qualification and controls: existing master continuation

Question: can the full canonical analytical DCTR implementation supply a
numerically qualified FP32 proposal force for the declared AR2 T10 two-noise-
scale scope? The model and normal prior are those of the qualified reference
in `surrogate-hmc-ar2-reference-2026-09-23.md`. Independent Kalman endpoints
are retained. The force need not equal the Kalman likelihood gradient.

Use N=64 and the exact-divisor K=N transport policy. This is an initial small
particle scope for testing an imperfect force, not a posterior approximation
accuracy assertion or a tuned particle-count default. The state is 2D, the
innovation and observation are scalar, and both parameters are unbounded log
noise scales. Fixed scrambled Halton Gaussian paths include x0 and all ten
innovations. A separate centered/whitened fixed Gaussian design is used for
Contract E. Freeze these once, with seeds folding in scope and partition.
The normal prior is applied once directly in log-scale coordinates; there is
no uniform prior, stationary initial-law substitution, or parameter chart.

Calibration and validation consume only their own saved data and references
from structural-03. The claim data remain unused for force/control selection.
Run the existing four-arm geometric protection curve (multipliers 1,.1,.01,
.001), retaining full flow, Contract E and GenUT dual-cap corrections. The
underlying LG1 controls are only a warm-start hypothesis. The existing
repository selection authority chooses the first adjacent pair that passes
non-harm on both partitions. Nominal points are the reference mean, plus one
marginal SD along each coordinate, and minus half an SD in both coordinates.
Stress points are all four corners at plus/minus three prior SDs. They test
opposed process/observation scales and small/large overall noise.

Record values, analytical scores, filtering/reset moments, typed statuses and
wall time. The non-harm criterion is the existing normalized maximum change
<=1e-3 on nominal points, all nominal states healthy, all stress failures
flagged, and no new stress failure relative to the weaker protection. Stress
failures stay visible and block HMC eligibility even when the controls pass
their non-harm comparison. A selected control artifact is
scope/source/partition bound. No online retuning is allowed.

Before issuing force qualification, compare the selected FP32 calculations
with FP64 on the identical once-rounded paths and observations at nominal and
stress points. Require matching value/force validity, no nonfinite healthy
result, and normalized value, score and moment difference <=2e-3 on jointly
healthy rows. A failed precision check triggers a precision/numerical repair;
it does not permit silent FP64 replacement of the candidate. Dedicated CPU
tests check both noise-scale analytical derivatives against the same finite
program, including the prior, and exercise the actual HMC consumer bindings.

Defaults and skeptical audit: T10/N64 and the Halton design are explicitly
scope hypotheses. N64 is justified by the immediate question about a cheap
imperfect force; it cannot establish D/R approximation quality. Normal-prior
tail checks do not certify the entire unbounded parameter space. The
protection controls require measured non-harm curves, not score agreement to
Kalman or selection by endpoint likelihood. A smooth constant force could
pass many mechanical checks while mixing poorly, so only downstream HMC
evidence can establish usefulness. A crash or invalid unflagged value vetoes
this consumer; failed curves trigger localized numerical/control repair.

Cheap adversaries for the later sampler comparison are exact Kalman, direct
reference-grid draws, bootstrap/no-flow filtering and ordinary Kalman-gradient
HMC. This qualification stage makes no method ranking. Posterior and runtime
claims must use the separate B/R, C/D, D/R and C/R comparisons and conditional
cost evidence. Loss/force residuals here are explanatory only.

Execute `--stage closure --closure-stages ar2-controls` in a fresh structural
output directory. Within the remaining structural-pass reservation allow 900
CPU-worker / 1,500 GPU-wall seconds for the numerical stage and 180 CPU seconds
for focused tests. At most three numerical attempts within that reservation.
GPU/XLA FP32/TF32 is the candidate; CPU tests and FP64 comparisons are labeled
reference exceptions. Preserve full logs, typed diagnostics, source snapshot,
seeds, reference/data hashes, protection curves, selection and resource receipt.

Next existing-master tasks: prospective cost/precision design, public
position-only-force HMC tuning, then retained model-endpoint comparison B/R.
Qualify a finite endpoint/reference separately for C/D and D/R. A failed
finite approximation may reject D/R while leaving model-endpoint B viable.
This stage completing is not a stopping reason while budget and prerequisites
permit continued master execution.
