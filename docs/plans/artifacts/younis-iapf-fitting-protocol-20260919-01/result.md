# iAPF fitting and Monte Carlo precision: completed continuation

2026-09-19. The remaining failure is dataset **1900**: its corrected score
still has larger observed squared error than UKF at 16,384 particles, and its
frozen proposal retains a severe first-step shape mismatch and an earlier
fitting-bound contact. Increasing particles repaired the observed simple-filter
failures on datasets 1901 and 2200. It did not repair the proposal on 1900.

Four stages completed under the [predeclared plan and sequential amendments](../../younis-iapf-fitting-protocol-2026-09-19.md).
Changing fitting-cloud size and stopping alone produced **0/4 statistically
supported validation improvements**. Increasing final particles from 4,096 to
16,384 with each proposal held fixed produced **4/4 primary interval passes**
on new observations, with observed MSE reductions of **72–79%**. A further
replay of three known failures produced **3/3 precision interval passes** but
retained the 1900 promotion veto. These are conditional comparisons on scalar
nonlinear models at horizon two, not evidence of general iAPF superiority.

## Question and executed scope

The lane seeks an accurate score of the observed-data likelihood with respect
to six physical model parameters. It uses a scalar state-space model with a
sine transition term and a quadratic observation term, in weak and more curved
regimes. Parameters, observations and refined grid references are saved per
dataset in each stage's `results.json`. This is an iAPF score investigation;
it neither implements nor admits the canonical LEDH algorithm.

The tested estimator is the normalized complete-data Fisher statistic with
frozen ancestor and Gaussian-innovation control coefficients. Every evaluated
arm has 96 independent control-calibration replicates and 64 separate score
replicates, with 16 independent same-law reference clouds for centering. Each
dataset is compared with EKF, UKF, and matching-particle no-resampling estimates.
Raw Fisher and ancestor-only corrections remain in the saved baseline ladder.

Calibration selected a fitting protocol using four new datasets, before
validation observations were generated. The selected protocol starts with 256
fitting particles, uses a four-estimate likelihood-CV window (`k=3`, `tau=.05`),
allows twelve iterations and caps fitting particles at 2,048. The relative-shape
objective, original standardized fitting box, floor ratio .001 and FP64 fitting
were held fixed. This is a diagnostic candidate, not a new default.

## Results by stage

| Stage | Result | Consequence |
|---|---|---|
| Calibration: 16 fits, four datasets, 2×2 design | `large_stable` selected by calibration MSE; `small_stable` exhausted its iteration cap on 2001 | Selection is descriptive; the failed fit was rejected, without using its last iterate |
| Untouched validation: eight fits, four datasets | All baseline and selected fits valid; 0/4 primary difference intervals exclude zero; 2200 loses to EKF and UKF | Fitting-only accuracy improvement is unsupported; continue to the variance diagnostic |
| New-data precision: four fits, four datasets | Same frozen proposal within each particle-count comparison; 4/4 primary passes, all heuristic and mean-bias screens pass | Supports a final-particle noise mechanism on these datasets |
| Known-failure replay: zero fits, three datasets | Exact prior proposals, fresh score/control randomness; 3/3 precision passes; 1900 still loses to UKF and retains its bound veto | Two failures are repaired at the observed-error screen; proposal diagnosis remains necessary for 1900 |

The validation MSE comparison below is corrected-score MSE against the refined
grid score. Every primary 99.75% paired interval includes zero. The method
comparisons on these four datasets remain statistically unresolved.

| Dataset | Original fitting | Selected fitting | UKF | Selected heuristic screen |
|---|---:|---:|---:|---|
| 2200, weak | .00115255 | .00106633 | .000410762 | Fails EKF and UKF |
| 2201, weak | .00102215 | .00108999 | .00230275 | Pass |
| 2210, curved | .00181526 | .00167388 | .0108005 | Pass |
| 2211, curved | .00189308 | .00161922 | .0244585 | Pass |

The new-data precision experiment gives the following paired intervals for
MSE(16,384) minus MSE(4,096). All four upper endpoints are negative.

| Dataset | MSE, 4,096 | MSE, 16,384 | 99.75% paired interval | Heuristic screen at 16,384 |
|---|---:|---:|---|---|
| 2400, weak | .000957196 | .000266799 | [−.00113701, −.000317137] | Pass |
| 2401, weak | .0256223 | .00549575 | [−.0326209, −.0102361] | Pass |
| 2410, curved | .00257706 | .000601494 | [−.00307340, −.00114339] | Pass |
| 2411, curved | .00151219 | .000385505 | [−.00169374, −.000674488] | Pass |

Both particle counts already cleared the heuristics on those four new datasets.
That is why execution continued to the selected-failure replay rather than
claiming that the earlier failures had disappeared.

| Known failure | MSE, 4,096 | MSE, 16,384 | UKF MSE | Remaining promotion veto |
|---|---:|---:|---:|---|
| 1900 | .0681563 | .0150409 | .0138890 | Observed UKF loss; earlier fitting-bound contact |
| 1901 | .00127234 | .000324232 | .000511308 | None in this conditional replay |
| 2200 | .00105814 | .000313874 | .000410762 | None in this conditional replay |

The replay intervals are [−.0821910, −.0306759], [−.00158525, −.000450453],
and [−.00113522, −.000412490], respectively. The exact old proposals were
reused, including the original small-cloud proposal for 1900. The selected
large-cloud fitting protocol was **not** refitted on 1900 in this campaign.
Thus this replay does not show that the new fitting protocol cannot repair it.
All replay mean-bias screens pass. Passing that screen does not prove zero bias.

The subsequent [mathematical audit](mathematical-explanation.md) recomputed
1900's MSE excess over UKF as .0011519 and its descriptive MSE standard error
as .0015491 from the 64 saved squared errors. The observed excess is .74 such
standard errors. This supports retaining the observed-screen verdict while
leaving the true risk ordering unresolved; it is not a new hypothesis test.
No new filter calls, fits or GPU launches were used for this calculation.

Intervals use 40,000 paired percentile-bootstrap resamples, with separate
predeclared comparison families for each stage. Their uncertainty is conditional
on observations, proposal fits and control fits. They do not cover population
variation, protocol selection, or repeated fitting. Heuristic wins and losses
are observed-error screens; no separate significance claim against UKF or EKF
is made. Full comparisons, including raw Fisher and no resampling, are in
[conditional-errors.csv](conditional-errors.csv) and [analysis.json](analysis.json).

## What the mathematics and code establish

For the current model, the target score is

\[
S(\theta)=\nabla_\theta\log p_\theta(y_{1:T})
=\mathbb E_\theta\!\left[
  \nabla_\theta\log p_\theta(X_{0:T},y_{1:T})\mid y_{1:T}\right].
\]

The second equality follows by differentiating the marginal integral: divide
the integral of the joint density times its log derivative by the marginal
likelihood. The Gaussian model has parameter-independent support and smooth
finite-moment integrands in the tested scope. The code computes a normalized
particle approximation, then subtracts frozen linear control terms. Centered
controls preserve its expectation conditional on the fitted coefficients;
they can reduce variance but do not remove finite-particle normalization bias.
An individual corrected realization is not the exact model score.

For score replicates \(\widehat S_r\), a fixed reference \(S^*\), sample mean
\(\bar S\), and sample covariance \(\widehat\Sigma\) using denominator \(n-1\),

\[
\frac1n\sum_{r=1}^n\|\widehat S_r-S^*\|^2
=\frac{n-1}{n}\operatorname{tr}(\widehat\Sigma)
+\|\bar S-S^*\|^2.
\]

Expanding each error about the sample mean makes the cross terms sum to zero.
The first term accounts for **96.6–99.8%** of selected-arm validation MSE;
see [variance-decomposition.json](variance-decomposition.json). The second term
contains uncertainty in the estimated mean, so this decomposition does not
prove that true bias is negligible. The separate particle-count experiment
provides stronger evidence: fourfold particle work reduced observed error by
roughly the inverse factor on the tested fixed proposals. No general variance
rate or equal-cost advantage is established.

The old likelihood-CV stopping configuration also has a concrete defect. For
two positive likelihood estimates \(a,b\), sample CV is

\[
\mathrm{CV}=\frac{\sqrt2|a-b|}{a+b}\le\sqrt2.
\]

Its threshold 100 cannot reject such a window. The new .05 threshold genuinely
rejects unstable windows, including the calibration cap failure. Nevertheless,
likelihood stability does not certify proposal shape, and the downstream
validation did not establish an accuracy improvement from this change.

Dataset 1900 has a distinct unresolved fitting defect. Its first-step twist has
independently evaluated predictive shape residual about **.9968**, where

\[
R=1-\frac{\langle\psi,h\rangle_w^2}
                 {\langle\psi,\psi\rangle_w\langle h,h\rangle_w}.
\]

Here \(h\) is the observation likelihood times the next-twist normalizer,
\(\psi\) is the fitted Gaussian-plus-floor twist, and the weighted inner
products use the independent predictive reference. Minimizing
\(\|h-a\psi\|_w^2\) over scale \(a\) gives this normalized residual.
The value near one means this twist is a very poor shape approximation under
that predictive weighting. It is not a residual under the small empirical
fitting cloud, so a small optimizer gradient does not contradict it.

The archived fit used 16 particles. Its earlier first-step fit touched a bound;
its final first-step center is 2.79305 and variance .0332908. The final empirical
shape residual is about .4055 despite an almost-zero projected gradient. The
predictive mean of the pointwise floor fraction is about .9978; that quantity
is **not** a sampling mixture frequency. These records establish a poor twist,
but do not separate inadequate cloud coverage, optimization failure, and
insufficient Gaussian-family flexibility. Nor does their coexistence prove the
twist alone caused the remaining UKF loss. The fitted shape and prior
derivation are preserved in the
[earlier fitting diagnosis](../younis-iapf-pinned-continuation-20260919-01/fitting-diagnosis.md).

## Decisions and inference status

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Do not promote fitting-only repair | 0/4 validation passes | 2200 heuristic loss | Small conditional comparison; fits otherwise viable | Preserve it as a diagnostic protocol | iAPF is invalid |
| Retain particle-precision mechanism | 4/4 new-data passes; 3/3 replay passes | 1900 still vetoed | Fixed datasets/fits; fourfold work | Separate proposal error from remaining sampling noise | Equal-cost superiority or a particle default |
| Keep 1900 open | Error decreases but UKF screen still fails | Shape failure and historical bound contact | Coverage versus optimizer versus family | Diagnose the first backward fit on this archived case | The new fitting protocol has already failed on 1900 |
| Preserve cloud-scale diagnostics | Focused numerical/adapter checks | No numerical policy change | Spread alone does not certify coverage | Use scale alongside independent shape residual in future fits | Past results contain newly added fields |

| Inference category | Status |
|---|---|
| Hard veto screen | One calibration arm rejected by its cap; validation 2200 and replay 1900 retain declared promotion vetoes; no experiment-invalidating reference, source, seed, numerical or device failure |
| Statistically supported ranking | Larger final count has lower conditional corrected-score MSE in every declared precision comparison; no fitting-protocol ranking supported on validation |
| Descriptive-only differences | Calibration selection, variance shares, heuristic comparisons, shape/floor/ESS diagnostics and timings |
| Default-readiness | No new default, HMC force or LEDH admission |
| Next evidence needed | Isolate 1900's fit failure; then fresh observation-level replication and an equal-cost comparison before broader claims |

## Reproducibility, accounting and engineering checks

Executed at Git `6fbcf3147660c40d5d5644bbcbcc9fadbcb06aef` with recorded working-tree
source snapshots. Each stage has its exact command, plan snapshot, input hashes,
environment, observations, random seeds, frozen fits/coefficients, raw outputs
and terminal decision. See the manifests for
[calibration](calibration01/manifest.json), [validation](validation01/manifest.json),
[precision](precision01/manifest.json), and [replay](challenge01/manifest.json).
Data are simulated and fully specified by those records; no external dataset
version applies.

All numerical score runs used the pinned RTX 5080 UUID
`GPU-d54fdcfc-c6ed-dbe7-25c7-93f737e0f93a`, escalated device access, FP32/TF32/XLA
and verified memory growth before initialization. Actual output placement is
GPU:0 with exactly that one visible physical GPU. Fitting is explicitly FP64;
grid references and diagnostic regression retain their declared CPU exceptions.
Every recorded numerical kernel has one trace. The environment is
`/home/chakwong/anaconda3/envs/tftwogpu/bin/python`.

| Resource | Used | Ceiling |
|---|---:|---:|
| Scientific launches | 4 | 4 |
| Adaptive fits, including rejected attempts | 28 | 32 |
| Conservatively charged filter/fit calls | 7,988 | 8,000 |
| Cumulative driver seconds | 255.037560 | 1,800 |
| CPU checks, conservative cumulative charge | 160 | 600 |

The remaining twelve calls and zero launches cannot cover another research
stage. This is resource exhaustion of this allocation, not rejection of a
scientific direction. The master retains the next fitting question separately
from completed work. There is no running experiment.

The independent reporter verifies source snapshots, input hashes, frozen
proposal/control identities, raw-score correction arithmetic, MSE arithmetic,
references, device/growth records, traces and seed separation. It checked
14,748 current seed pairs against 17,424 archived pairs with no collision.
Two earlier focused CPU runs passed eleven tests each. CPU tests deliberately
hide all GPUs and do not constitute GPU performance evidence.

After the scientific runs, a reporting-only repair appends minimum and maximum
fitting-cloud SD to the recursive fitter and adapter diagnostics. These
quantities were already computed and discarded. Existing eleven fields retain
their order; new records have thirteen fields. Old artifacts remain immutable,
and the strict scope validator rejects a stale schema instead of upgrading it.
The saved scientific source snapshots therefore intentionally precede this
observability repair. A future campaign must record its new source closure.

All **33** focused fitting, nonlinear adapter, scope and relative-shape tests
pass after that repair, including known cloud spreads and preservation through
the actual mixed-precision consumer. The CPU/XLA check took 76.14 seconds;
the two warnings are existing TensorFlow Probability deprecations. The exact
command was:

```sh
CUDA_VISIBLE_DEVICES=-1 BAYESFILTER_PRELOAD_CUSTOM_OP=0 OMP_NUM_THREADS=4 TF_NUM_INTRAOP_THREADS=4 TF_NUM_INTEROP_THREADS=2 /home/chakwong/anaconda3/envs/tftwogpu/bin/python -m pytest -q tests/highdim/test_younis_score_master_iapf_tf.py tests/highdim/test_younis_score_master_nonlinear_iapf.py tests/highdim/test_younis_score_master_iapf_scope.py tests/highdim/test_younis_iapf_relative_shape_tf.py
```

See [cpu-observability-tests01.log](cpu-observability-tests01.log) and
[repair provenance](observability-repair.json). The final independent artifact
audit passed after the reporting repair; see [final-audit03.log](final-audit03.log).
`git diff --check`, Python syntax checks, and result/plan link checks also pass.

## Post-run skeptical review and next question

The strongest alternative explanation for the apparent general progress is
observation selection: there are only four fresh precision datasets, and the
three replay datasets were deliberately chosen failures. Repeated random
streams within one dataset do not replace independent observation-level
replication. Bootstrap intervals with 64 assessments are approximate, especially
for heavy error tails. Passing the mean-bias screen is weaker than establishing
an acceptably small bias. Timing here includes substantial fixed overhead and
must not be used to claim that four times as many particles cost the same.

The smallest next scientific question is whether 1900's first backward twist
fails because the fitting cloud misses the important predictive region, because
the optimizer finds a poor solution, or because this Gaussian family cannot
represent the target. Preserve the frozen failing fit and observations; inspect
cloud coverage and fit geometry before selecting a repair. Any subsequent
repair needs independent data for an accuracy claim. A good predictive shape
fit that still fails the score comparison would weaken the proposal-error
explanation and redirect attention to the score's remaining variance/bias.
Do not substitute another bound expansion or an unexamined stopping threshold
for that discrimination.
