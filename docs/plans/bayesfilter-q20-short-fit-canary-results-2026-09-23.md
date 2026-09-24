# q20 short fitting test: nonlinear learning, incomplete whitening

**The scalar correction learned useful nonlinear shape and lowered reverse-KL
loss against the original map on an independent 512-point check. Its whitening
remains uneven: the left latent tail had a larger observed score residual.**
The affine-only correction had no detectable loss benefit in its 128-point
canary. These results support continued nonlinear training work; they do not
establish a correctly trained posterior map or HMC readiness.

The [plan](bayesfilter-q20-short-fit-canary-plan-2026-09-23.md) records the
question, numerical hypotheses, skeptical review, pilot and fresh confirmation.
Total supervised GPU-worker time was **645.17 seconds (10.75 minutes)**, within
the 30-minute ceiling. No HMC was run.

## What ran and what was measured

The scientific target was unchanged: the four-parameter q20/T30 **UKF approximate
posterior**, evaluated through its original frozen batch-native TensorFlow
source. This is not an exact state-space posterior calculation. The saved
depth-four map F remained fixed, with 2,048 lifetime updates. Only the correction
C in `F(C(z))` was trained, for 64 updates at batch size 32:

- A free diagonal affine correction, with eight parameters.
- A three-component smooth scalar sigmoid mixture in latent coordinate 2,
  followed by the affine correction, with seventeen parameters.

Both used the standard exact reverse-KL gradient, fresh Adam slots, beta1=.9,
beta2=.999, epsilon=1e-7, and learning rate .01. This rate passed the separate
one-step calibration in both arms; no fallback rate was used. The measured
gradient clipping envelopes were 17.7682 and 25.9342 respectively, ten times
the largest starting calibration-batch norm. **Neither arm clipped any update
or rejected an update.** This does not make these settings optimized defaults.

The computed loss was `-log pi(F(C(z))) - log|J_(F o C)(z)|`. Its expectation
differs from reverse KL by constants common to the compared maps. Therefore
paired expected-loss differences equal reverse-KL differences, even though the
reported absolute losses are not normalized KL divergences. Every changed
physical point received a fresh target value and score.

The geometry check computed `grad_z log pi_z(z)+z`, including the full
log-Jacobian derivative. The RMS figures below are RMS **vector norms**, not
per-coordinate RMS. All required heldout rows had valid, finite target values,
scores and map calculations. Base parameters stayed exactly unchanged.

## Independent 512-point confirmation

The initial 128-point screen was too noisy to resolve the scalar endpoint's
loss difference from the original map. Before another target call, the plan
fixed one fresh 512-point bank and one primary comparison: the same frozen
scalar endpoint against the original map. There were no additional optimizer
updates, checkpoint choices, hyperparameter changes, or pooled pilot points.

| Quantity on the same 512 fresh base points | Original map | Scalar + affine after 64 updates |
| --- | ---: | ---: |
| Valid evaluations | 512/512 | 512/512 |
| Mean RKL objective, up to common constants | 43.57594 | 43.26586 |
| Score-residual vector RMS | 4.32079 | 3.42853 |
| Coordinate-2 residual RMS | 4.00597 | 3.04479 |
| Centered log-density residual RMS | 2.12128 | 1.53102 |
| Log-density residual range | 13.53405 | 12.12408 |

The paired scalar-minus-original loss difference was **-0.31008**, with MC
standard error **0.08103** and a predeclared approximate 95% normal interval
**[-0.46889, -0.15127]**. The primary fixed-checkpoint fit screen passed.
This is statistical evidence about these two specific maps under fresh
base-normal sampling. It does not establish repeatability across training
seeds, architecture superiority, posterior coverage, or a new default.

Overall score-residual RMS fell descriptively by **20.7%**. The error remains
substantial, and the average hides uneven behavior:

| Base-normal coordinate-2 slice | Points | Original score RMS | Corrected score RMS |
| --- | ---: | ---: | ---: |
| z2 < -1 | 82 | 4.94179 | 5.56382 |
| -1 <= z2 <= 1 | 344 | 4.23065 | 3.00420 |
| z2 > 1 | 86 | 4.03336 | 2.08924 |

The correction did not clear the conditional sanity check in the left slice:
its observed squared-residual increase was 6.5349. An exploratory, unadjusted
95% paired interval was [-1.4951, 14.5648], so the direction of that slice's
expected change remains uncertain. Preserve this concern before any quality
promotion. The global squared-residual change was -6.9144, with an exploratory
interval [-8.9520, -4.8769]; geometry remains a diagnostic, not posterior evidence.

The left slice's mean loss contribution also increased, from 42.9623 to
44.7356. These are integrand contributions at **different physical points**
under the maps. A smaller global reverse KL does not require a smaller
integrand at every latent point, so this is not independently proof of worse
physical posterior-tail probabilities. The large and unresolved score error
in that slice is the relevant whitening concern.

## Pilot and gradient-mechanism results

The 128-point pilot preserved both arms' own initial maps. The scalar
initialization was deliberately nonidentity, so learning was evaluated against
it as well as against the original map. Four planned loss contrasts used
simultaneous approximate 95% intervals, with normal multiplier 2.4977.

| Pilot contrast | Mean loss change | Simultaneous interval |
| --- | ---: | --- |
| Trained affine minus original/identity initialization | -0.00733 | [-0.07851, 0.06385] |
| Trained scalar minus its own initialization | -0.38896 | [-0.77270, -0.00522] |
| Trained scalar minus original map | -0.31001 | [-0.73826, 0.11824] |

The scalar arm learned beyond its starting deformation. A saved-state shape
inspection found the nonlinear fraction of the correction coordinate's
variance increasing from 0.0193% to 2.0495% on the pilot bank, relative to its
best affine regression. This describes nonlinear parameter motion; it is not
a percentage of posterior error repaired. The affine-only loss result remained
inconclusive and its residual RMS went from 4.7493 to 4.8379.

At the **unchanged initial scalar checkpoint**, sixteen independent B32
minibatches gave gradient covariance traces 1.96809 for the ordinary estimator
and 1.52994 for the path estimator: an observed **22.3% reduction**. Their mean
gradient norms were 1.0978 and 1.0279. The finite minibatch gradients are not
expected to agree pointwise. This variance comparison is descriptive; no
uncertainty-supported estimator ranking or path-gradient training result is
claimed.

The target was evaluated once per gradient-probe batch. Both gradient
calculations then used the same unchanged physical points and scores, with
an explicit equality check. Mean warm gradient-only times were approximately
1.46 ms and 1.54 ms; the common target/geometry evaluation took about 2.40 s.
These short timings describe the probe, not an end-to-end optimizer benchmark.

## Numerical and execution audit

Twenty-three focused analytic CPU/XLA checks passed before launch, including
the existing sixteen mechanism tests and seven Adam adapter cases. An initial
test failure was resolved by tracing the installed Keras scalar-conversion
semantics in its bias-correction formula; the checked difference was about
6.7e-8 in a first update. Tests cover the actual Adam step, invalid controls,
finite-status rejection, post-update rollback of parameters/slots/iteration,
and restored-state continuation. GPU devices were deliberately hidden for
these analytic tests.

Actual target work ran trusted on GPU1, TensorFlow 2.20.0, float64 and XLA,
with memory growth verified before initialization. GPUs0/2 had other work.
The 32-row kernels were batch-native, with one trace per training and geometry
function, no sample-wise target fallback, no pfor, and no NumPy runtime path.
Warm training updates averaged 2.397 s for affine and 2.393 s for scalar.
Peak TensorFlow allocator usage in the training worker was 268856832 bytes.

Saved endpoint reconstruction reproduced forward coordinates exactly and
log determinants within 1.78e-15. Maximum inverse roundtrip error was
1.48e-10. These diagnostic reconstructions do not issue HMC artifacts.
The original map and source identities stayed fixed. Independent standard-
library audits reproduced paired statistics to 1e-12, checked the complete
row/update counts and validity flags, and verified that the confirmation's
actual latent points and seeds were disjoint from the pilot.

The active harness also corrects a reporting label: the raw confirmation
accounting file called its per-stage remainder `remaining_this_canary_seconds`.
The actual timeout enforced the aggregate cap correctly. Prior artifacts remain
preserved; the aggregate accounting summary states both attempts' actual costs.
Unused dedicated diagnostic allowance was retired at completion, preventing it
from silently authorizing unrelated later diagnostics.

## Decision and remaining uncertainty

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Record successful local nonlinear learning | Fresh 512-point paired loss interval below zero | No numerical veto; conditional left-slice whitening concern remains | One training stream, short tranche, restricted correction family | Continue bounded nonlinear training development and inspect persistent left-slice error with fresh validation | Fully trained map, all-region fit, HMC readiness |
| Do not promote affine-only repair | Pilot loss interval crosses zero | No arithmetic veto | Small effect versus training/MC noise | Keep as a cheap control; no long affine-only campaign justified by this result | Free scale can never help |
| Retain path gradients as a viable mechanism | Observed variance reduction on the paired probe | No derivative or finite-value veto | Sixteen batches at one checkpoint; no trained path arm | Test a matched continuation when training resumes | General variance reduction, faster training, or a superior method |
| Stop this requested test | Fixed training and confirmation banks complete | No budget or infrastructure failure | Full training and downstream sampling are outside this canary | Preserve states/results and use the findings in the next training protocol | A failed research direction or completed posterior estimation |

| Inference status | Finding |
| --- | --- |
| Hard veto screen | Numerical checks pass; left-slice underperformance/uncertainty prevents a general whitening-quality promotion |
| Statistically supported ranking | The fixed scalar endpoint has lower RKL than the original checkpoint on the independent MC check; no ranking across training runs or estimators |
| Descriptive-only differences | Pilot score RMS, conditional summaries, shape fraction, gradient variability and timing; exploratory residual intervals are labeled separately |
| Default-readiness | Not established; optional correction codec and tiny Adam adapter are not the full production/HMC pipeline |
| Next evidence needed | Longer target-specific training with preserved state, fresh conditional geometry checks, standard 1000-point post-training verification, then supported downstream HMC validation |

The strongest alternative explanation is improved fit in the accessible region
while other posterior mass remains missed. The base-normal banks cannot rule
that out. A fresh training stream that fails to reproduce the improvement, poor
conditional geometry, or independent posterior/reference disagreement would
weaken the favorable interpretation. The weakest current evidence is the
single short training stream and unresolved tail behavior. This experiment
demonstrates a repair mechanism that can learn, not completed training.

## Preserved evidence and budget

- [Pilot manifest](artifacts/q20-short-fit-canary-2026-09-23/run-01/worker-01/manifest.json),
  [pilot result](artifacts/q20-short-fit-canary-2026-09-23/run-01/worker-01/result.json),
  and [independent pilot audit](artifacts/q20-short-fit-canary-2026-09-23/run-01/statistics-audit.json).
- [Confirmation manifest](artifacts/q20-short-fit-canary-2026-09-23/run-02/worker-01/manifest.json),
  [confirmation result](artifacts/q20-short-fit-canary-2026-09-23/run-02/worker-01/result.json),
  and [independent confirmation audit](artifacts/q20-short-fit-canary-2026-09-23/run-02/statistics-audit.json).
- [Accounting summary](artifacts/q20-short-fit-canary-2026-09-23/accounting-summary.json).

The manifests preserve exact commands, environment, Git and source identities,
checkpoint/data identities, seeds, hardware and artifact paths. Per-update
history, optimizer state and checkpoints at updates 16/32/48/64 are in the
pilot worker's arm directories. Pilot and confirmation cost 543.14049 and
102.02911 supervised seconds. The unchanged total campaign now has
**143330.11637 seconds (39.81 hours)** remaining. The dedicated test allocation
is closed; the older diagnostic balance remains 7.71141 seconds.
