# q20 NeuTra training gaps: code, cohort and derivative investigation

The strongest supported explanation is incomplete training combined with a
restrictive parameterization and a controller that cannot use downstream failure
to improve the learned map. All twelve direct candidates were still improving
when updates stopped. The selected map has a nearly saturated contraction in
its last IAF stage. Neither result proves how much further training or capacity
will suffice, but both are concrete repair targets.

A new independent finite-difference check substantially weakens the alternative
explanation of incorrect target or transformed scores at the tested points.
The gradient and optimizer machinery has additional executable evidence from
the earlier shadow-update checks. The problem is not established to be a broken
RKL formula, a scalar fallback, or absence of XLA.

This investigation followed the [plan](bayesfilter-q20-training-gap-investigation-2026-09-22.md).
It evaluated no new training candidate and made no production/default change.
The numerical question concerns the specified four-parameter, q20/T30,
float64 UKF approximate posterior; q20 is the latent-state dimension, not the
number of inferred parameters.

## Code path and evidence boundaries

| Consumer or operation | Actual path | What was established |
| --- | --- | --- |
| Master training dispatch | `q20_master_program.py:315` → `Campaign.numerical_stage` → `q20_master_stages.dispatch` → `run_training_cohort` | One supervised worker; `stop_when_trial_ready=True`; budget priced at the cohort floor |
| Initialization | `q20_production_training.py:55` → `WeightedDenseIAFTransport` → prior affine → `TrainingSession` | Two IAF stages, reversal, fixed outer center and scale 4; zero final network outputs initially preserve the prior law |
| Repeated update | `TrainingSession.advance` → `IndependentTemperedReverseKLTrainer._compiled_train_step` | Fresh iid batch of 32; stable seed signature; XLA encloses transport, target, loss, parameter gradient and Adam update |
| Target | `GaussianLikelihoodBridge.value_score_status` → `BatchNativeSSLLSTMComplexityPosteriorTarget._batch_prior_likelihood_value_score_status_impl` → batched sigma-point value/score wrapper | Prior plus beta times the full likelihood; analytical four-coordinate derivatives; leading sample dimension preserved |
| Filter recursion | `experimental_batched_svd_sigma_point_tf.tf_batched_svd_sigma_point_value_and_score_with_rule` | Batched tensor/einsum/linear-algebra operations; time recursion uses `tf.while_loop` in the executed source |
| Training derivative | `_value_with_reviewed_score` attaches the analytic physical score; `GradientTape` differentiates the map/log determinant | Earlier independently written VJP and actual Adam update agree on checked saved states; new finite differences check the supplied physical and transformed scores |
| Assessment | `_evaluate_rung` → `FrozenLossCache` → `assess_training_rung` | Same-base-point loss differences; one 768-row bank; baseline improvement nominates a trial; map parity concerns numerical encoding |
| Export and choice | `export_weighted_transport` → `choose_maps` | Exact frozen map encoding; first eligible candidate in declared order, one map for plain NeuTra |

The Python loop in `TrainingSession.advance` schedules optimizer updates and
records/checkpoints their results. It does not evaluate individual training
samples through a scalar target. Python also unrolls the small fixed number of
network layers at tracing time. Those are different from a Python loop over
the sample batch or filter dates. Per-update host synchronization, copying state
for rollback, finite checks, and JSON logging remain overhead; their individual
costs have not been profiled here and are not established causes of poor fit.

Historical source authority is the preserved r2 execution tree/archive. The
[source comparison](artifacts/q20-training-gap-investigation-2026-09-22/source-comparison.json)
finds 15 of 16 inspected current files identical to it. The current low-level
batched sigma-point module differs: it introduces a compiled recurrence helper
and explicit post-loop branch vetoes, among other changes. This investigation
does not transfer its new numerical evidence to that different implementation.
The stopping, selection, optimizer and training-configuration findings apply to
the identical current files as well.

## The full direct-training cohort

The preserved pre-import cohort has twelve direct beta-one maps and four
historical root-0 continuation maps at beta 0.5. Four direct maps have 1,024
lifetime updates; eight have 512. The additional 512 updates for root 0 preceded
the backend/scope migration. All direct maps have only 512 current-scope updates.
The continuation maps have 512 historical updates and zero updates credited in
the new scope; this is not a completed tempered training bank.

The following are observed diagnostics, not a ranking. The validation banks and
training streams differ across candidate identifiers. The last comparison is
current-scope update 128 versus 512: lifetime steps 640 versus 1,024 for root 0,
and 128 versus 512 for roots 1 and 2. The intervals are descriptive because the
banks were reused adaptively.

| Width / LR / root | Lifetime updates | Last 128 batch-loss mean | Last paired held-out loss change ± half-width | Clipped updates |
| --- | ---: | ---: | ---: | ---: |
| 16 / .0005 / 0, selected | 1,024 | 45.884 | −1.175 ± .352 | 1,021 / 1,024 |
| 16 / .0005 / 1 | 512 | 49.264 | −135.267 ± 12.994 | 512 / 512 |
| 16 / .0005 / 2 | 512 | 49.637 | −159.748 ± 15.220 | 512 / 512 |
| 16 / .001 / 0 | 1,024 | 44.795 | −.720 ± .238 | 1,018 / 1,024 |
| 16 / .001 / 1 | 512 | 46.160 | −38.191 ± 2.540 | 512 / 512 |
| 16 / .001 / 2 | 512 | 46.414 | −38.115 ± 2.400 | 512 / 512 |
| 32 / .0005 / 0 | 1,024 | 46.101 | −1.349 ± .555 | 1,024 / 1,024 |
| 32 / .0005 / 1 | 512 | 47.925 | −76.886 ± 6.194 | 512 / 512 |
| 32 / .0005 / 2 | 512 | 48.364 | −75.529 ± 8.204 | 512 / 512 |
| 32 / .001 / 0 | 1,024 | 42.330 | −1.618 ± .177 | 1,017 / 1,024 |
| 32 / .001 / 1 | 512 | 44.493 | −10.103 ± 1.148 | 510 / 512 |
| 32 / .001 / 2 | 512 | 46.320 | −8.609 ± .695 | 512 / 512 |

Every row's original assessment says `continuing_improvement=true` and
`plateau_observed=false`. None reached the configured 2,048- or 8,192-update
scope rungs. The direct cohort contains 8,192 accepted lifetime updates in
total, roughly 6.12 hours of recorded update work. The four beta-0.5 histories
add 2,048 updates and about 2.01 hours. These are summed per-update records,
including some first-call costs; they exclude much of the campaign's setup,
validation, pricing and tuning time and are not total campaign cost.

All twelve imported assessments subsequently report zero incremental loss.
The later cohort has exactly the same optimization histories; these were
same-map reassessments, not twelve new demonstrations of convergence. The
[cohort summary](artifacts/q20-training-gap-investigation-2026-09-22/saved-cohort-summary.json)
preserves both versions of every assessment.

## Confirmed stopping and allocation gaps

`assess_training_rung` uses improvement relative to initialization as its
learning screen. After the floor, that can produce `hmc_trial_nominee` even
while the incremental comparison is improving. The scheduler skips future
updates for that nominee. The master explicitly requests this behavior and
allocates the training-floor forecast, so merely listing larger rungs does not
provide a useful continuation mechanism for these maps.

The September 20 change had a legitimate purpose: a bounded HMC trial does not
need a converged variational map, and fine plateau validation had become
expensive. The gap is what follows a poor trial. After failure to obtain a
verified member, `run_estimation_attempts` records `tuning_candidate_failed` and
continues to the next method. It has no branch that tries the other eleven
direct exports, diagnoses the map, or returns a nominee to funded training.
The current master therefore does not complete the intended training-to-HMC
repair cycle.

`choose_maps` iterates widths, learning rates, then roots and returns the first
eligible direct map. Consequently the smallest-width/lower-LR/root-0 map is
chosen whenever it passes. Training a width/LR/seed grid is not, by itself,
performing model selection. The table contains descriptively different fits,
but there is no common evaluation bank or uncertainty-supported downstream
selection among them.

The parameter audit also described common-root pairing; `scoped_seed` includes
the full candidate identifier, which contains width and learning rate. The
stored root-0 training and validation seeds differ across all four
configurations. Thus those histories are independent arms, not a matched-noise
experiment. Unequal lifetime training across roots is an additional confounder.

## The learned scale parameterization is a specific concern

Each IAF stage uses

\[
s(a)=2\tanh(a/2),\qquad y_j=e^{s_j}x_j+m_j,
\qquad \frac{ds}{da}=1-(s/2)^2.
\]

In the selected map, the last stage's fourth scale output has median
**−1.966965**, range **−1.974477 to −1.950069**, on the 1,000 saved Gaussian
points. Its derivative with respect to its raw scale logit is therefore about
**0.03276** at the median, roughly 1/30 of the slope at zero. This is a measured
source of attenuation in this parameterization; it is not a claim that the
whole Adam update is 30 times smaller. The contraction approaches the imposed
lower limit of −2 throughout the tested region.

This pattern occurs across the direct cohort: median final-coordinate log
scales range from about −1.901 to −1.990. In the selected map the first stage
has no scale within the descriptive 90%-of-cap display band; the cap pressure
is concentrated in the final stage. Its final hidden layer has 4.36% of
activations above .95 in absolute value, while the other inspected hidden
layers have none. The evidence does not support blaming universal hidden-layer
saturation.

The scale cap is a per-stage conditional coefficient, not a bound on every
singular value of the composed flow. Composition and off-diagonal shears matter.
The selected map's median Jacobian condition number is 13.95, with median
smallest/largest singular values .3075/4.2934. These describe the proposal map,
not posterior whitening. Other layers may still redistribute contraction, so
we have not proved that the cap makes the target unrepresentable. A controlled
scale/capacity contrast is justified; arbitrarily removing bounds is not.

The architecture search cannot currently perform that contrast:
`validate_protocol` rejects anything other than two-stage tanh flows. Only
widths 16 and 32 were tried; depth, ordering, scale range and a learned outer
affine were not explored by this campaign. The outer scale remains the prior
SD 4, so the flow stages carry all learned contraction and dependence.

The stored maps also have markedly different proposal region allocation. On
the same 1,000 Gaussian base points, fractions with positive physical
observation weight range from **27.1% to 99.9%**; the selected map has 50.2%.
For example, the width-32/LR-.001/root-0 map has the lowest displayed batch-loss
mean and 99.9% positive-weight proposals. That does not prove a coverage error
without an independent posterior reference. It does show why choosing the
smallest displayed loss cannot replace a coverage/downstream assessment.

The independent diagnostic flow reconstruction reproduced saved physical
control points to 4.44e-16. Its Jacobians/log determinants also passed internal
central-difference and determinant checks. NumPy was used only for this saved
data/reference analysis; no production or training path uses it here.

## Scores and optimizer: evidence against a simple implementation defect

The new diagnostic checked all four coordinates at twelve fixed points in
both physical and latent coordinates. Eight points were spread through the
saved iid bank; four were the largest-residual stress cases. For each coordinate
it used central value differences at relative steps 1e-3, 1e-4 and 1e-5. All
600 evaluated rows were finite and passed the target-status checks.

| Space | Max absolute error, h scale 1e-3 | At 1e-4 | At 1e-5 | Max scaled error at 1e-5 |
| --- | ---: | ---: | ---: | ---: |
| Physical theta | 1.8271e-4 | 1.8270e-6 | 1.8426e-8 | 9.6163e-10 |
| Transformed z | 6.1625e-5 | 6.1625e-7 | 6.3152e-9 | 1.8572e-9 |

The decrease is consistent with central differences' second-order truncation
error until numerical precision matters. The latent values/scores also
reproduced the saved diagnostic. Errors of this size cannot explain residual
norms of order 7–47 at these points. This supports local score correctness for
the executed UKF target and map; it is not a global derivative theorem or a
check of the UKF likelihood against the exact nonlinear-model likelihood.

The earlier [September 18 diagnostic](bayesfilter-ssl-lstm-q20-training-continuation-2026-09-18.md)
checked eight actual trainer updates against an independently written VJP and
shadow Adam implementation. Stored loss/raw-gradient/optimizer checks passed,
parameter outputs matched exactly, and all three relevant graphs had one XLA
trace and no Python callbacks. Its result checksum was reverified. That is
executable call-chain evidence for those historical states; the new finite
differences address the separate question of supplied-score correctness at
the selected final map.

Clipping remains an uncalibrated hypothesis. Across the twelve direct histories,
**99.32%–100%** of updates were clipped at norm 10. However, Adam does not simply
multiply its parameter step by the clipping ratio. Under constant positive
rescaling of all supplied gradients, `m` scales by c and `v` by c squared, so
`m/(sqrt(v)+epsilon)` becomes `m/(sqrt(v)+epsilon/c)`. The earlier consistently
doubled-gradient/history probe changed the next update by only
0.0019%–0.0093%; abruptly feeding raw gradients into existing clipped moments
changed it by 15.6%–39.7%. Neither is a long-run clipping ablation.

The routine history stores loss, total gradient norm and clipping, but lacks
per-layer update/weight ratios, moment-denominator distributions, gradient
noise across independent batches, or scale-cap occupancy. Those observations
are needed to distinguish unfinished fitting from noisy or poorly conditioned
optimization. A constant learning rate is used; the documented half-rate
continuation experiment is not an active scheduler phase.

## Why clipping is almost continuous

The threshold of 10 was inherited without target-specific calibration. It is
applied to the norm across the transport's network parameters, not the
four-dimensional posterior score. The selected model's last 128 actual updates
had raw-norm median 60.007, range 8.702–159.400, and 126 clipped updates. Across
the twelve direct maps, corresponding medians were 43.307–85.343. Thus routine
clipping is mechanically expected at the chosen cap. The loss already uses a
batch mean; an accidental sum over the 32 samples does not explain the norms.

A new saved-data decomposition locates the large gradients. It restored the
exact selected training map, reproduced its frozen export, and reused the
existing 1,000 Gaussian points and scores. The first 992 points form 31
disjoint batches of 32 at fixed parameters. These are diagnostic batches, not
the original training stream. The following norms are descriptive; norms of
different terms do not add because their gradients can cancel.

| Parameter-gradient term | Median batch norm | Range |
| --- | ---: | ---: |
| Total RKL loss | 55.374 | 23.634–118.798 |
| Negative log likelihood | 55.053 | 24.255–118.536 |
| Negative log prior | 3.091 | 1.453–4.694 |
| Negative log determinant | 4.279 | 4.261–4.312 |

All 31 diagnostic batches would be clipped at 10. The likelihood contribution
accounts for the large norm, while the prior and log-determinant terms are much
smaller. The final IAF stage's shift-output weights and biases account for
**97.862% of summed squared gradient norm** across the batches. Of the whole
network's squared norm, **80.311%** lies in the final shift output for
`observation_bias.0`, **14.496%** in `observation_weight.0.0`, and **3.055%**
in `latent_mean_bias.0`. These are parameter-gradient shares, distinct from the
latent-score residual shares reported earlier.

This concentration has a direct parameterization explanation. In the last
stage, with fixed outer prior scale 4,

\[
\theta_j=c_j+4\{x_j e^{s_j}+m_j\},\qquad
m_j=\sum_k M_{kj}h_k W^{\rm shift}_{kj}+b^{\rm shift}_j.
\]

Writing the physical posterior score as
`a_ij = d log pi(theta_i) / d theta_ij`, the actual batch-mean RKL derivative
with respect to these shift outputs is

\[
\frac{\partial\widehat L}{\partial b^{\rm shift}_j}
=-\frac4B\sum_i a_{ij},\qquad
\frac{\partial\widehat L}{\partial W^{\rm shift}_{kj}}
=-\frac4B\sum_i M_{kj}h_{ik}a_{ij}.
\]

The final shift outputs do not enter this stage's log determinant, and the
outer affine determinant is constant. Their gradients therefore expose the
posterior scores directly, multiplied by the fixed physical scale and hidden
features. Source: `neutra_weighted_training.py:328` and
`tempered_transport_ensemble_tf.py:432` in the archived r2 tree. The inherited
cap does not adjust for these parameter sensitivities or correlated output
weights.

There is substantial minibatch variability even with the parameters fixed.
The norm of the 992-point mean gradient is 24.415, while the estimated RMS
deviation of a 32-point batch gradient from that mean is 63.682. The estimated
RMS sampling error of the bank mean is 11.438, so 24.415 is not an exact
population gradient or a convergence test. Together with the original
training histories, these measurements support large, noisy,
likelihood-dominated shift gradients as the immediate reason for frequent
clipping. They do not establish exploding gradients or a broken derivative.

Even a perfectly fitted Gaussian map need not have zero gradient on each
batch under this estimator. For `theta=m+sigma*z` and target
`N(mu,sigma^2)`, the location derivative is
`(m-mu)/sigma^2 + mean(z)/sigma`. At `m=mu` its expectation is zero but its
variance is `1/(B*sigma^2)`. If `m=c+4*b`, the derivative with respect to `b`
has variance `16/(B*sigma^2)`. Thus sharp physical scales and a small batch can
give large parameter-gradient noise even when latent whitening is exact.
This example explains a mechanism; it does not assert that the current map
has fitted the posterior.

The separate log-scale bound `s=2*tanh(a/2)` limits conditional contraction
and already shows saturation. It is not the global gradient-clipping
operation, and saturation by itself does not explain a large global norm.
Likewise, the full T=30 likelihood must remain a sum: dividing only that term
by 30 would change the posterior. Any optimizer repair must preserve the
declared statistical target.

Cap 10 currently acts as a routine gradient normalization rather than rare
outlier protection. Adam's scaling behavior explains why this need not shrink
updates in proportion to the clipping ratio, but varying clipping factors
change how batches contribute to its moment history. The justified next
experiment is a matched continuation assessing clipping, batch noise and
output scaling with actual learning and downstream map checks. Neither a
larger cap nor removing clipping is established as a repair by this analysis.

| Decision | Primary criterion status | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Attribute routine clipping to the inherited cap acting on large, variable likelihood/shift gradients | Fixed-map gradient decomposition and saved training norms agree descriptively | Exact export, finite gradients, additive identities and saved physical-score parity pass | Long-run optimizer effect; adequacy of batch and parameterization | Matched continuation with gradient/noise and actual-update telemetry | Exploding gradients, optimal cap, harmless clipping or posterior readiness |

No candidate comparison or statistically supported ranking is made. All
twelve maps retain their previously recorded status; the hard checks here
validate this decomposition only. Batch norms, squared-norm shares and noise
estimates are descriptive. Default-readiness is unchanged. Replicated
controlled continuation with common evaluation points and downstream checks
is needed to choose an optimizer repair.

Post-run review: a large gradient in an output head could arise from genuine
posterior curvature, poor fit, its coordinate scaling or estimator noise.
This diagnostic localizes the gradient without separating all those causes.
Its fixed-map evidence cannot reconstruct Adam's counterfactual unclipped
history. A controlled run that improves neither learning nor map usefulness
after reducing routine clipping would weaken clipping as the main bottleneck.

The diagnostic used CPU TensorFlow 2.20 float64/XLA with GPUs intentionally
hidden, one trace per function, zero new target rows and zero optimizer updates.
The recovered physical scores agree with the separately saved twelve-point
evaluations to max absolute error 5.68e-14; positions agree to 1.78e-15.
It consumed **4.507 supervisor seconds** from the existing allocation.
The [manifest](artifacts/q20-recovery-and-affordability-2026-09-22/campaign-05/attempts/00008-saved-clipping-decomposition/worker/manifest.json)
records the command, source, environment and input hashes; the
[result](artifacts/q20-recovery-and-affordability-2026-09-22/campaign-05/attempts/00008-saved-clipping-decomposition/worker/result.json)
and sibling `batch-gradients.json` preserve the decomposition. The
[coordinate shares](artifacts/q20-training-gap-investigation-2026-09-22/clipping-output-coordinate-shares.json)
are diagnostic postprocessing of those preserved gradients. No runtime code
or numerical default was changed.

## Validation and uncalibrated choices

The 768-row held-out screen is suitable for a bounded learning decision with
its stated limitations. Increasing it until an extremely precise plateau is
resolved is not the necessary repair. Assess real checkpoint increments,
evaluate implicated geometry periodically, and use fresh reserved evidence
after development selection. Identical-map reassessment should explicitly say
that no optimization increment occurred and preserve the last genuine change.

| Material choice | Actual value and provenance | Observed gap / smallest discriminating check |
| --- | --- | --- |
| Objective and beta | RKL-form loss `mean(-target-logdet)`; beta 1 for direct maps; derived from change of variables | Formula and tested scores consistent; loss offset is unknown, so raw loss is not absolute KL |
| Prior initialization | Exact center + SD-4 affine with fixed permutation; derived prior map | Correct law does not certify later posterior geometry; all roots start from the same proposal law despite hidden-weight randomness |
| Training effort | 512 current-scope floor; rungs 128/512/2048/8192, inherited hypotheses | All direct maps still improving; fund continuation as a real phase |
| Capacity | Two tanh stages, width 16/32, reverse ordering; inherited hypotheses | Near-cap terminal contraction; depth is hard-coded by validator; evaluate one function-preserving capacity extension or a properly controlled new architecture arm |
| Scale cap | 2; inherited hypothesis | Measured attenuation; test scale sensitivity under an explicit parameterization, recording any changed initial map |
| Optimizer | Constant LR .0005/.001; Adam .9/.999/1e-7; clip 10; inherited hypotheses | Actual updates work on checked states; long-run clipping/noise/LR sensitivity remains unmeasured |
| Batch size | 32; proposed pricing alternatives 8/32/128 | Active `price_training` defaults to only the chosen batch; no automatic selection from `pricing_batches`; record gradient noise and throughput before choosing a larger batch |
| Seeds | Three roots; candidate-specific streams | No common-random-number pairing; root-0 warm starts have more lifetime work; use explicit matched parents/streams for causal contrasts |
| Validation | 768 active rows; old saved 3072/12288 expansion values ignored by first-bank rule | Reused-bank intervals descriptive; self-comparison erases progress; numerical parity is not whitening |
| Selection and repair | Fixed first eligible map, then switch methods on failed tuning | Other direct maps and further training are not exercised as repairs |

## What three GPUs can improve

The trusted inventory confirms three RTX 4080 SUPER devices, each reporting
32,760 MiB. The diagnostic selected idle host GPU 1. Device availability still
varies: GPU 0 had another compute process at that launch, and device 2 reported
load. Three installed GPUs do not guarantee three uncontended workers now.

The current master holds one coordinator lock and waits for one subprocess.
The worker selects a single device and sets `CUDA_VISIBLE_DEVICES` to it; the
cohort loops over candidates sequentially. There is no per-candidate three-GPU
training queue or distributed optimizer. Starting three copies of the current
master is not a supported substitute: it does not partition candidate state or
aggregate budget, and automatic device observations are not reservations.

The simplest useful parallelism is **one independent candidate/continuation per
available GPU**, each retaining its batch-native XLA update. Use a coordinator
with explicit device assignments, separate output/checkpoint directories,
stable candidate seeds, preserved Adam/RNG state and a shared sum-of-worker-time
budget. Existing shared cache reads may be reused by identity; workers should
write separate cache/output directories. No model/API workers or sub-agents
are needed for this numerical parallelism.

Recent saved pricing measured about 2.32–2.34 seconds per batch-32 update for
width 16/32, with the selected trained map's recent history near 2.4 seconds.
As a conditional estimate, **1,024 additional unchanged-map updates take about
40–41 minutes per worker**, excluding setup/validation. Three such independent
jobs could overlap in roughly that wall time if capacity is available, while
consuming roughly 2.0 GPU-hours in aggregate. This is not a measured concurrency
speedup or a price for a deeper/wider map. A new 768-row validation evaluation
at about 2.34 seconds per batch costs roughly 56 seconds before compilation;
only genuinely new map/bank evaluations need be paid.

## Repair order supported by this investigation

1. Repair progress accounting and control flow together: separate historical,
   current-scope and lifetime updates; retain the last distinct comparison;
   allow a trial nominee to resume training; reserve a bounded map-repair phase
   after poor geometry or failed downstream tuning. Preserve the next method
   as an option without treating one failed map as exhausting plain NeuTra.
2. Put independent training jobs on a device-assigned queue. Keep each update
   batch-native and compiled. Charge concurrent work in aggregate rather than
   counting only elapsed coordinator time.
3. First obtain a matched continuation control from the selected map and saved
   Adam/RNG state. Compare it with a single optimizer contrast from the same
   parent, such as the already documented half-LR hypothesis. The third device
   can investigate the observed scale/capacity restriction. This is a proposed
   discriminating design, not a newly promoted parameter set. Changing the
   tanh cap with fixed weights changes the initial map; do not call that an
   identical warm start. A depth expansion must prove function-preserving
   initialization before a matched continuation claim.
4. Record cap sensitivities, per-layer actual updates, gradient noise and
   proposal-region occupancy alongside loss. Evaluate maps on common fresh
   development points when comparing them; use independent reserved evidence
   for selection confirmation and relevant downstream checks. Replicate any
   promising repair before claiming it reliable.
5. Use the existing plain NeuTra/tempered-ensemble downstream machinery to test
   usefulness after the map repair. Small score residuals on proposal points,
   an optimizer plateau, or low variational loss alone cannot qualify posterior
   estimation. No mass-adaptation prerequisite is introduced.

## Decisions, uncertainty and run record

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Treat direct training as unfinished | Twelve genuine incremental screens show ongoing learning; no plateau | No saved invalid direct update | Attainable improvement after more training | Bounded matched continuation with repaired scheduler | A fixed count guarantees adequate learning |
| Investigate cap/capacity | Repeated near-limit contraction with derived small scale-logit slope | No map reconstruction/parity failure | Whether other layers can compensate | Controlled parameterization/capacity contrast | Current family is mathematically incapable |
| Deprioritize a gross score/Adam bug | New FD checks and earlier VJP/Adam checks pass in their scopes | No finite/status/derivative veto at checked points | Untested regions and current-source changes | Preserve evidence; add localized checks if a new failure appears | Global gradient or exact-likelihood validity |
| Repair selection and feedback | First eligible map always selected; failed tuning does not resume direct training | No support for ranking from existing tables | Which map will yield useful HMC | Evaluate viable maps under common downstream criteria and allow repair | Lowest loss is best or posterior-qualified |
| Add concurrent candidate execution | Three devices observed; current scheduler is serial | Per-job capacity and memory policy still required | Concurrent throughput and availability | Explicit candidate/device queue and aggregate budget | A measured threefold speedup |

| Inference status | Finding |
| --- | --- |
| Hard veto evidence | No derivative/finite/status veto in the new test; same-map increments are invalid as convergence evidence |
| Statistically supported ranking | None among the twelve maps or proposed repairs |
| Descriptive-only differences | Training windows, reused-bank intervals, map geometry, sign fractions and timing estimates |
| Default-readiness | No new map, architecture, optimizer, batch size or HMC default is established |
| Next evidence | Controlled continuation/capacity/optimizer work, independent assessment and useful downstream sampling |

Post-run review: insufficient effort and near-cap parameterization can coexist.
The cap observation does not establish that relaxing it will improve inference;
the lower-loss maps' very different sign allocation is a strong reason to keep
coverage distinct from loss. Finite differences cover only twelve deliberately
chosen points. The absence of a training repair loop is directly supported by
code, while the effectiveness of the proposed repair remains unmeasured.

Reproduction scripts and structured records are in
[the investigation directory](artifacts/q20-training-gap-investigation-2026-09-22/).
The derivative command was
`/home/ubuntu/anaconda3/envs/tfgpu/bin/python docs/plans/artifacts/q20-training-gap-investigation-2026-09-22/check_score_derivatives.py`.
It used the archived source, existing map and saved data with no new random
draws, GPU/XLA float64, one trace per coordinate system, and verified memory
growth. The worker manifest records source hashes, target/data identity,
environment, selected rows, device inventory and allocator observations. The
worker took 96.507 seconds; the existing supervisor charged **98.045 seconds**.
There were no optimizer updates. After the additional saved-data clipping
diagnostic, remaining balances are **155,692.207 campaign seconds (43.248
hours)** and **902.358 diagnostic seconds (15.039 minutes)**.
The master remains `ESTIMATION_BUDGET_PAUSED`.

- [Derivative result](artifacts/q20-recovery-and-affordability-2026-09-22/campaign-05/attempts/00007-training-gap-score-derivatives/worker/result.json)
- [Derivative manifest](artifacts/q20-recovery-and-affordability-2026-09-22/campaign-05/attempts/00007-training-gap-score-derivatives/worker/manifest.json)
- [Budget receipt](artifacts/q20-training-gap-investigation-2026-09-22/derivative-budget-receipt.json)
- [Clipping diagnostic budget receipt](artifacts/q20-training-gap-investigation-2026-09-22/clipping-budget-receipt.json)
- [Earlier RKL stopping audit](bayesfilter-q20-rkl-training-audit-2026-09-22.md)

Reset note: continue from these findings, not the imported zero increments.
The twelve final direct maps retain their pre-import weights/history. The
selected map has 1,024 lifetime updates. Local score checks now pass, and prior
optimizer checks exist; further blanket revalidation is not the next useful
step. Repair training continuation, candidate use, measured cap/optimizer
diagnostics, and device scheduling before another long HMC attempt.
The saved-data clipping decomposition now localizes the large gradient norm
to likelihood-driven final-stage shifts, predominantly the observation-bias
output, with substantial batch variability. This replaces an unmeasured
per-layer hypothesis but still does not select a new clipping threshold.

The owner's subsequent calibration-first directive is recorded in the
[complete training hyperparameter audit and calibration design](bayesfilter-q20-training-hyperparameter-calibration-2026-09-22.md).
The existing master's calibration mode is a first-root, 128-update pilot, not
an executed calibration of the training settings. The next implementation work
must repair that mechanism and the documented comparison/continuation defects
before treating another cohort as calibrated. This audit did not launch a new
training search or change numerical settings.

The owner's latest low-cost-canary request has now been executed in the
[sanity report](bayesfilter-q20-training-sanity-results-2026-09-22.md).
Sixteen disposable updates and saved-data screens cost 131.038409 supervisor
seconds. They identify clipping-role failure, scale pressure and batch noise,
without optimizing hyperparameters or promoting a map. The latest balances
are 155,561.168 campaign seconds and 771.319 diagnostic seconds; the preceding
balances above are historical to their earlier diagnostic. The master remains
paused and the original map is preserved.
