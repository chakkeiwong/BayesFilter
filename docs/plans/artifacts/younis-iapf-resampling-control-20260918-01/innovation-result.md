# Gaussian controls clear the nonlinear screens; affine Kalman and device scope remain limits

Terminal review, 2026-09-19; experiment started on 2026-09-18. The added
Gaussian-innovation controls reduce score MSE relative to matched ancestor-only
controls on all four fixed nonlinear datasets. Observed reductions are
49--73%; all four predeclared paired intervals exclude zero in the favorable
direction. The candidate also passes all four nonlinear observed heuristic
screens. Exact Kalman still dominates on the affine case. No default, HMC
force or LEDH admission is issued.

A launch defect changed device visibility: the previous runs pinned the
RTX5080 UUID, but this command left CUDA_VISIBLE_DEVICES unset. The manifest
records both GPUs; the log maps TensorFlow GPU:0 to RTX4080 SUPER and GPU:1
to RTX5080. Default placement therefore points to the RTX4080 SUPER, but the
driver did not save per-output device ordinals. The paired methods ran in
the same process and retain evidence for this executed scope. Confirmation
on the intended single RTX5080 scope is outstanding. This is an execution
scope deviation, not a hidden claim of hardware parity or a timing comparison.

## What was tested

Same five datasets and frozen iAPF fits as the preceding comparison, scalar
sine-transition/quadratic-observation model, T=2, N=4096. All 96 calibration
and 64 final streams per dataset are fresh. Six Gaussian moment contrasts
are added to the 12 ancestor controls. All coefficients were frozen before
any final replication; saved coefficient files agree with final records.

For each time and coordinate the new control subtracts the first or second
sample moment of 16 independent reference clouds from that of the actual
filter noise. Identical marginal laws make its expectation zero without
assuming exact normal-generator moments. The FP64 correction is subtracted
outside the normalized Fisher statistic. It preserves that statistic's
expectation, including finite-N bias; it is not a finite-likelihood gradient.
Independent reference centering adds 1/16 of the original moment-control
variance. Ordinary floating reduction error remains a numerical qualification.

## Conditional score errors

Entries are mean squared Euclidean errors of six-component scores. Primary
intervals compare the new correction minus ancestor-only correction using
identical 96-run calibration sizes. Four 99.75% paired bootstrap intervals
give the declared approximate Bonferroni 99% primary family. They condition
on the fitted coefficients and fixed observations, not a new dataset population.

| Dataset | Regime | Ancestor-only MSE | New MSE | UKF MSE | No-resampling MSE | Primary difference interval |
|---|---|---:|---:|---:|---:|---|
| 1500 | weak | 0.0041354 | 0.0013284 | 0.0018787 | 0.0068490 | [-0.0038029, -0.0018944] |
| 1501 | weak | 0.0054209 | 0.0014565 | 0.0077080 | 0.0211646 | [-0.0053116, -0.0027561] |
| 1510 | curved | 0.0028124 | 0.0014303 | 0.0259030 | 0.0044258 | [-0.0020245, -0.0007944] |
| 1511 | curved | 0.0098628 | 0.0050386 | 0.3450184 | 0.0116240 | [-0.0074141, -0.0025604] |

Against the previously frozen 192-calibration protected ancestor coefficients,
observed reductions are 47--68%; all four separately exploratory 99% paired
intervals favor the new controls. This supports that the main finding is not
solely an artificially weakened 96-calibration baseline. On weak dataset 1500,
new-minus-UKF has exploratory 99% interval [-0.0008504, -0.0001965]. On curved
dataset 1511, new-minus-no-resampling has interval [-0.0110062, -0.0029229].
All twelve nonlinear comparisons against EKF, UKF and no resampling have
negative observed differences; their individual 99% intervals are exploratory,
not a joint family-wide superiority certificate.

All five mean-bias screens pass, with no proof of exact unbiasedness. New/raw
variance comparisons and percentage reductions are descriptive point estimates.
The affine new MSE is 0.000940884 versus about 7.9e-30 for exact Kalman;
that observed heuristic loss remains a promotion veto. The conditional
regression ranks are 7/13 (ancestor/new) for affine and 8/14 for nonlinear,
consistent with the earlier input-precision rank safeguard. These diagnostics
do not establish that iAPF fitting bounds are well calibrated.

## Verification and accounting

The numerical implementation uses the same iAPF invocation and actual
innovation arrays as the ancestor baseline. 22 focused CPU/XLA tests passed
before execution, including finite-support centering under nonnormal moments,
consumer wiring, Fisher genealogy and regression checks. GPUs were deliberately
hidden for CPU tests. The launch-repair CPU regression is logged separately.
Five refined quadrature references pass mesh, domain and tail checks; the
affine score agrees with Kalman. No nonfinite value or unexpected retracing
occurred. All frozen coefficient and saved correction applications have been
independently replayed using scalar arithmetic; see innovation-analysis.json.

GPU execution was trusted/escalated, FP32/TF32/XLA with verified memory growth
on both visible GPUs. Controls/regression were FP64 diagnostic exceptions;
quadrature was CPU FP64. GPU:0 allocator peak was 18,181,632 bytes. No speed
ranking is made. The driver ran for 36.819829 seconds; total campaign driver
time is 166.131425/1800 seconds. All 4/4 launches and 8000/8000 filter calls
are consumed; adaptive fits stay 5/8. Conservative combined CPU test/probe
accounting is 240/600 seconds. Unused wall time is not extra call authority.

The exact executed driver and plan were preserved before repairing GPU
selection. The current driver now pins the prior physical UUID before
TensorFlow import, checks the resulting device identity, and records output
tensor devices. Its intended GPU behavior still needs a fresh GPU execution;
the repair does not retroactively change this result or consume another launch.

## Decision

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | What is not concluded |
|---|---|---|---|---|---|
| Retain the optional controls as a diagnostic candidate | 4/4 primary comparisons pass in executed scope | No numerical veto; affine heuristic loss and intended device scope unresolved | One calibration and fixed T2 datasets | Pin RTX5080 and confirm on fresh streams under a new bounded plan | Production/default/HMC/LEDH readiness |
| Close this bounded campaign | All planned calls executed | Launch and call budgets exhausted | No remaining same-campaign replication | Preserve results and plan the next discriminating campaign | Completion of the whole master |

## Inference status

| Evidence | Status |
|---|---|
| Hard veto screen | No numerical validity failure; GPU visibility differed from intended prior scope; affine exact comparator wins |
| Statistically supported ranking | Four conditional new-versus-matched-ancestor MSE differences favor new under the declared primary family |
| Descriptive-only differences | Percent reductions, variance ratios, coefficient sizes and runtime; heuristic intervals are exploratory |
| Default readiness | Not established; optional diagnostic only |
| Next evidence needed | Correctly pinned GPU confirmation; fresh observation data and independently repeated calibration; fitting-bound repair and longer horizons |

Engineering checks pass for the control calculation and artifacts. The GPU
selection defect is repaired in code with CPU regression coverage, but the
corrected GPU launch is unevaluated. Numerical references and mean-bias
screens pass. Scientifically, the optional controls remain viable on these
fixed datasets; the uncorrected or ancestor-only estimator's earlier failures
do not establish failure of iAPF, KDM or LEDH as research directions.

Post-run red-team: the strongest alternative explanation is that these
controls work unusually well for the chosen short scalar datasets and the
one calibration realization. Independent datasets/calibration could overturn
the apparent general usefulness. The smallest required repair is a pinned
GPU replication; fitting robustness and genuine fresh-data/longer-horizon
validation then precede iAPF-moment/LEDH integration. Do not retune against
the current final streams. The exact GPU ordinal omission is the weakest
engineering evidence; narrow model coverage is the weakest scientific evidence.

## Evidence

Plan: [innovation-control plan](../../younis-iapf-innovation-control-2026-09-18.md).
Manifest/actual command: [manifest](innovation-confirmation01/manifest.json).
Raw records: [results](innovation-confirmation01/results.json).
Frozen calibration: [calibration](innovation-confirmation01/frozen-calibration.json).
Errors: [CSV](innovation-conditional-errors.csv).
Terminal audit: [analysis](innovation-analysis.json).
Run log: [GPU log](innovation-confirmation01.log).
CPU checks: [tests](innovation-cpu-tests01.log), [launch repair](innovation-launch-repair-tests01.log).
