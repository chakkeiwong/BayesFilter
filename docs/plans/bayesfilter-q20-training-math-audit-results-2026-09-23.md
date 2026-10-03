# Why the repaired q20 NeuTra maps still have poor Gaussian geometry

The strongest new finding is that the saved flows have learned almost affine
transformations on the tested region, while their largest remaining score error
is nonlinear. Removing routine clipping and adding two stages did not resolve
that mismatch. The executed reverse-KL loss and its first parameter derivative
have the correct mathematical form. Optimization convergence, useful nonlinear
capacity, and posterior sampling performance remain unestablished.

This is a source and saved-data audit under the
[reviewed plan](bayesfilter-q20-training-math-audit-plan-2026-09-23.md). It makes
no new target evaluations or optimizer updates. The scientific target is the
specified **four-parameter, q20/T30 UKF approximate posterior**, not the exact
nonlinear state-space posterior. The three complete
[1,000-point reports](bayesfilter-q20-saved-maps-1000-point-results-2026-09-23.md)
remain the numerical evidence. The batch-128 report is incomplete and is not
substituted into the full-bank comparisons.

## What the executed training actually computes

Execution authority is the frozen tree
`/tmp/BayesFilter-q20-training-repair-20260923-r1`, not the subsequently edited
workspace. Let `F` denote that root in the source table below. SHA-256 identities
for both versions are preserved in
[saved-analysis.json](artifacts/q20-training-math-audit-2026-09-23/saved-analysis.json).

| Operation | Executed source anchor | Finding |
| --- | --- | --- |
| Draw and update | `F/bayesfilter/inference/neutra_training_protocol.py:66`; `tempered_transport_ensemble_tf.py:1130` | A new stateless iid normal batch each update; RNG index advances; accepted state and full optimizer checkpoint are retained |
| RKL scalar and derivative | `F/bayesfilter/inference/tempered_transport_ensemble_tf.py:1138` | Batch mean of minus target minus log determinant; transport gradients, clipping, then Adam |
| Supplied target derivative | Same file, line 134 | Custom gradient attaches the analytic physical score; stopping that score's graph is correct for the first parameter derivative |
| Tempering and prior | `F/bayesfilter/inference/tempered_target_tf.py:467`; `F/bayesfilter/nonlinear/ssl_lstm_complexity_batched_target_tf.py:576` | Prior plus beta times the full UKF likelihood; beta one in these repairs; prior covariance 16 times identity |
| Flow | `F/bayesfilter/inference/neutra_weighted_training.py:190`, `:244`, `:317`, `:456` | Masked autoregressive tanh networks; zero initial output kernels; bounded log scales; full reversal between stages |
| Outer affine | `F/bayesfilter/inference/tempered_transport_ensemble_tf.py:380` | Fixed prior center and scale 4; only the inner flow is trainable |
| Repair state | `F/bayesfilter/inference/q20_training_repair.py:27` | Old parameters and Adam slots retained; additional identity stages receive zero slots at the old global iteration |
| Repair termination | Same file, lines 140, 173, 257, 311 | Predetermined update tranche; parent-to-endpoint assessment; returns `TRAINING_REPAIR_TRANCHE_COMPLETE`, explicitly `production_qualified=False` |

The current and frozen transport, RKL trainer, bridge, and batched target files
are identical. The training protocol, repair controller, production training,
and master files differ; assertions about execution above use the frozen files.
The ordinary master already has `stop_when_trial_ready=False` and next-rung
repair logic (`F/.../q20_master_program.py:357`, `:408`). Repeating the older
claim that this ordinary master still skips all eligible nominees would be
wrong. The recent maps came from the separate bounded repair controller.

Write the invertible map as \(\theta=T_\lambda(z)\), with
\(z\sim\phi=N(0,I_4)\), and let \(\widetilde\pi\) be the unnormalized UKF
posterior. The code computes

\[
L(\lambda)=E_\phi[-\log\widetilde\pi(T_\lambda(z))
                         -\log|\det J_{T_\lambda}(z)|].
\]

The change of variables gives
\(\log q_\lambda(T_\lambda(z))=\log\phi(z)-\log|\det J_T(z)|\).
Consequently, with \(Z=\int\widetilde\pi(\theta)d\theta\),

\[
\operatorname{KL}(q_\lambda\Vert\pi)
=E_\phi\log\phi(z)+L(\lambda)+\log Z.
\]

The recorded loss near 43 is therefore **not a measured KL divergence of 43**,
nor a distance to zero: its additive normalizer is unknown. The parameter
gradient needed by this objective is

\[
\nabla_\lambda L=-E_\phi\left[
 (\partial_\lambda T)^T s_\theta(T(z))
 +\partial_\lambda\log|\det J_T(z)|\right],
\qquad s_\theta=\nabla_\theta\log\widetilde\pi.
\]

That is precisely the first derivative supplied by the custom score bridge.
No target Hessian is required for it. There is no missing entropy sign,
accidental division of the likelihood by the observation count, or omitted
Jacobian in this traced loss. The code uses a batch-native target and a stable
XLA train-step signature. The host loop schedules updates; it does not turn
the batch into scalar target calls.

Previously checked physical and transformed finite differences had maximum
absolute discrepancies about \(1.84\times10^{-8}\) and
\(6.32\times10^{-9}\) at twelve parent-map points, including large-residual
points. Independent trainer VJP/Adam checks also passed on their inspected
states; see the
[earlier investigation](bayesfilter-q20-training-gap-results-2026-09-22.md).
These weaken a simple derivative-bug explanation. They are local evidence,
not a proof for every new parameter state or the exact state-space likelihood.

## The learned maps are almost affine where we tested them

A standard-library diagnostic reconstructed each saved transport from its
weights, exact degree masks, tanh activations, bounded scales, reversal, and
outer affine. Stage mean log scales reproduce the saved TensorFlow reports
to better than \(7\times10^{-16}\), within the inherited \(10^{-12}\)
arithmetic-check tolerance. It then fitted each physical coordinate onto
\(1,z_1,z_2,z_3,z_4\), using all 1,000 saved points.

The table reports the fraction of that coordinate's sample variance left
unexplained by its affine fit. These are descriptive fractions, not significance
tests or newly chosen quality thresholds.

| Map | Latent weight \(\theta_1\) | Latent bias \(\theta_2\) | Observation weight \(\theta_3\) | Observation bias \(\theta_4\) |
| --- | ---: | ---: | ---: | ---: |
| Control | 0.00816% | 0.01909% | 0.00412% | 0.000174% |
| Clipping repair | 0.01357% | 0.01267% | 0.000371% | 0.000916% |
| Depth four | 0.00198% | 0.07464% | 0.00112% | 0.13424% |

Thus even the depth-four map's physical coordinates have **99.866%–99.999%**
of their variation explained by an affine transformation. More stages did not
automatically produce a materially nonlinear proposal on this bank. This is an
empirical mean-square statement, not a global affine identity or a bound on
derivatives: a small-amplitude nonlinear correction can still have an important
derivative. The independent reconstruction is explanatory and cannot issue a
training or posterior admission.

In the opposite direction, the score error is predominantly nonlinear. Define
the pullback density and its residual by

\[
\pi_z(z)=\pi(T(z))|\det J_T(z)|,\qquad
g(z)=\nabla_z\log\pi_z(z)+z.
\]

Exact Gaussianization would give \(g(z)=0\). Partitioning the observed
\(\sum_i\|g(z_i)\|^2\) by coordinate gives:

| Map | Share in \(g_1\) | Share in \(g_2\) | Share in \(g_3\) | Share in \(g_4\) | Affine fit's share of \(g_2\) energy |
| --- | ---: | ---: | ---: | ---: | ---: |
| Control | 22.43% | 72.02% | 5.48% | 0.071% | 0.751% |
| Clipping repair | 15.10% | 78.68% | 6.15% | 0.065% | 0.725% |
| Depth four | 4.86% | **89.02%** | 6.04% | 0.079% | **0.573%** |

For depth four, the dominant coordinate has RMS residual **4.2638**.
Over 99% of that coordinate's squared residual remains after projection onto
the constant and all four linear latent coordinates. This localizes the
remaining problem beyond a simple overall shift or scale error. It does not
prove that optimizing a finite affine correction could never help, since the
target itself is nonlinear and a finite correction changes which points are
visited.

Reversal means latent coordinate 2 is not physical parameter 2. The empirical
depth-four fit is approximately
\(\theta_3=0.2743+0.4958z_2\), with much smaller other coefficients;
\(\theta_3\) is the observation weight in the actual target
(`ssl_lstm_complexity_batched_target_tf.py:303`, `:469`). The residual is thus
concentrated in a direction that principally controls that weight. Coupling and
the log determinant still matter; this is not an exact one-coordinate target
reduction. The saved points alone do not prove multimodality, missing modes, or
that the UKF approximation is accurate.

## Why RKL progress can coexist with this error

The training minimizes an expected log-density discrepancy. The post-training
test measures a squared derivative discrepancy, \(E_\phi\|g\|^2\). They
are different quantities. Even exact stationarity in a restricted map family
only removes the projections represented by its permitted variations; it does
not force the pointwise residual to vanish.

An expressive map that actually reaches zero KL would Gaussianize the target.
The observed loss reduction establishes neither that the needed map has been
learned nor that optimization has reached that solution.

For a simple illustration, consider the one-dimensional density
\(p(x)\propto\exp(-x^4/12)\). A Gaussian proposal
\(x=m+\sigma z\) minimizes its reverse KL at \(m=0,\sigma=1\): its
nonconstant objective is

\[
\frac{m^4+6m^2\sigma^2+3\sigma^4}{12}-\log\sigma.
\]

For fixed \(\sigma\), the minimum is \(m=0\); differentiating there gives
\(\sigma^3-1/\sigma=0\), hence \(\sigma=1\). Nevertheless its pulled-back
score residual is \(g(z)=z-z^3/3\), with
\(E[g]=E[zg]=0\) but \(E[g^2]=2/3\), using the standard-normal moments
\(E[z^2]=1,E[z^4]=3,E[z^6]=15\). A correctly optimized Gaussian approximation
can therefore retain nonlinear score error. This is a mathematical illustration,
not a fitted model for the q20 target.

There is a closely related observation in our actual saved points. For an
infinitesimal precomposition \(T_{a,b}(z)=T(e^a\odot z+b)\), the exact
per-point first derivatives at zero give

\[
\partial_{a_j}L=E[-(g_j-z_j)z_j-1],\qquad
\partial_{b_j}L=E[-(g_j-z_j)].
\]

For depth four the sample log-scale derivatives in coordinates 1 and 2 are
**−0.00170 ± 0.06242** and **0.02450 ± 0.14599**, respectively, where ± denotes
one descriptive standard error. In contrast, the control's coordinate-1
derivative is **3.0453 ± 0.2033**, and the clipping repair's is
**1.9061 ± 0.1460**. Aggregate scale signals can be small while the depth-four
nonlinear residual remains large. These are derivatives for a hypothetical
affine precomposition, not a convergence test of the network parameters; no
formal inference is claimed on this already inspected bank.

Adding \(\|g\|^2\) to the loss is not an automatically correct fix.
Differentiating it with respect to the map generally requires derivatives of
the target score. The existing stopped-score bridge supports the RKL first
derivative; it does not automatically provide those Hessian terms.

## Parameterization and optimizer mechanisms still needing repair

**Initialization can favor affine changes before nonlinear features.** Each
conditioner has two width-16 tanh hidden layers, initialization standard
deviation 0.02, and an exactly zero output kernel. At that initialization,
hidden-layer gradients through the output kernel are exactly zero; output
bias gradients need not be. After the first update the hidden gradients can
become nonzero. Earlier eight-update checks did observe hidden parameters
moving, so it would be wrong to call the hidden layers permanently disconnected.
Small features, zero output initialization, and later contracted inputs offer
a mechanism for learning shifts/scales much faster than nonlinear dependence.
The near-affine endpoint is established; the extent to which initialization
caused it is still a hypothesis.

The masks also make each individual stage affine in its own coordinate:
\(y_j=x_j\exp(s_j(x_{<j}))+m_j(x_{<j})\). Nonlinear marginal deformation
must emerge through conditioning on other coordinates and composing stages;
there is no direct nonlinear function of \(x_j\) in that stage's scale or
shift. Constant scales and affine shifts reduce such stages to affine maps.
The measured near-affine endpoint is consistent with weak learned conditional
dependence. This structure does not prove that the composed family cannot fit
the target, but explains why adding stages alone need not learn the missing
nonlinear shape.

The added third stage is particularly close to the linear regime of tanh:
mean absolute hidden activations are **0.01460** and **0.01427**, with maxima
**0.1053** and **0.1483**. Its log scales have little dependence on the inputs.
Other layers have larger activations; there is no evidence that every hidden
layer is saturated or inactive. Adding depth without checking the feature
scales presented to the new conditioners does not establish useful capacity.

**The inherited output-scale bound is still active.** The saved recipe uses

\[
s(a)=2\tanh(a/2),\qquad y_j=x_je^{s_j}+m_j,
\qquad s'(a)=1-(s/2)^2.
\]

The original second-stage fourth output has mean log scale **−1.98735**,
**−1.99358**, and **−1.97141** for control, clipping repair, and depth four.
The corresponding mean logit derivatives are **0.01261**, **0.00641**, and
**0.02839**. This is a measured attenuation in that parameterization.
For a final-stage local scale logit, holding its input and shift fixed, the
sample loss derivative is

\[
\frac{\partial\widehat L}{\partial a_j}
=-[4s_{\theta,j}x_je^{s_j}+1]\,[1-(s_j/2)^2].
\]

The first factor combines the target force and entropy term; the second is
the attenuation. Network weight derivatives add their input-feature factors.
For the intermediate second stage of the depth-four map, downstream pullbacks
also enter, so this final-stage formula cannot be substituted unchanged there.
The fixed outer scale 4 requires learned contraction inside the flow. Other
stages and shears can compensate; the evidence does not prove the composed
flow lacks sufficient contraction, nor that Adam steps shrink in proportion
to these slopes. A scale reparameterization remains a discriminating experiment,
not an established cure for the dominant nonlinear residual.

**Routine clipping is largely repaired in the intervention arms.** The control
clipped 1,014/1,024 updates (**99.02%**); the higher-envelope arm clipped
7/1,024 (**0.68%**); depth four clipped **0/1,024**; batch128 clipped **0/256**.
Their cap, **159.39995**, was the measured maximum of the declared parent
history/gradient banks, an explicit local-envelope hypothesis rather than a
universal threshold. Persistent poor geometry without frequent clipping rules
out ongoing clipping as a sufficient explanation. It does not isolate the
effect of the earlier clipped training or prove that the new cap is optimal.

**Gradient noise and the optimizer recipe remain incompletely calibrated.**
The saved repair keeps learning rate **0.0005**, Adam coefficients **0.9/0.999**,
epsilon **1e-7**, and batch **32** for the main arms. There is no learning-rate
schedule. Earlier saved-point regrouping measured batch32 gradient variation
far larger than the pooled mean gradient; see the
[sanity results](bayesfilter-q20-training-sanity-results-2026-09-22.md).
Those measurements concern the parent, not a newly measured depth-four endpoint.

Moreover, this correct reparameterization estimator has noise even at a perfect
Gaussian fit. For \(q=N(m,\sigma^2)\), target \(N(\mu,\tau^2)\), its
location gradient estimate is
\(B^{-1}\sum_i(m+\sigma z_i-\mu)/\tau^2\).
At \(m=\mu,\sigma=\tau\), its variance is \(1/(B\tau^2)\), although
the true gradient is zero. A trainable inner shift with outer multiplier 4
has variance \(16/(B\tau^2)\). Thus a large minibatch norm is not itself
proof of nonconvergence, and a noisy constant-rate recipe can obscure small
nonlinear learning signals. Actual per-layer parameter updates and gradient
signal-to-noise are not recorded in these repair histories.

For clarity, variance reduction need not mean changing the objective. Under
the usual differentiability, common-support and integrability conditions,
differentiating \(E_{q_\lambda}\log(q_\lambda/\pi)\) gives

\[
\nabla_\lambda\mathrm{KL}
=E_{q_\lambda}[(\partial_\lambda T)^T(s_q-s_\pi)]
 +E_{q_\lambda}[\partial_\lambda\log q_\lambda(\theta)|_\theta].
\]

The last expectation is \(\partial_\lambda\int q_\lambda=0\). If an exact
proposal score is available, the first expectation is consequently an
alternative unbiased first-gradient estimator that vanishes pointwise at
\(q=\pi\). This derivation identifies a possible variance-control experiment,
not an implemented or validated replacement. It needs exact proposal-score,
gradient-equivalence, cost and downstream checks before use.

Adam migration is another explicit hypothesis. Old second moments retain
coefficients \(0.999^{1024}=0.35897\) after the main repair tranche and
\(0.999^{256}=0.77404\) in the batch128 arm. New stages start with zero
moments but global iteration 1,024. For a new parameter with a nonzero gradient
on iteration 1,025, its update with negligible epsilon is **2.5326 times** a
fresh Adam first step of the same learning rate and gradient. These are derived
properties of the chosen warm start, not proof of a faulty update. In fact,
the inherited part contributes only **0.99%** of the clipping arm's final
second-stage sum of second moments and **5.76%** in depth four; stale moments
are not established as the dominant late-stage problem.

## The repair stopped after an allocation, not demonstrated convergence

Control, clipping repair and depth four received 1,024 added updates, reaching
2,048 lifetime updates. The new stages themselves have only 1,024 updates.
Batch128 received 256 added updates, reaching 1,280 lifetime updates. Each arm
used 32,768 added target rows; equal target rows do not equal optimizer steps
or moment adaptation. This is a bounded repair experiment, not a completed
target-specific training protocol.

The means of every recorded 128-update block are shown, using the existing
checkpoint cadence rather than selecting favorable windows:

| Added updates | Control | Clipping repair | Depth four |
| --- | ---: | ---: | ---: |
| 1–128 | 45.704 | 45.558 | 44.139 |
| 129–256 | 45.333 | 44.876 | 43.524 |
| 257–384 | 45.157 | 44.610 | 43.553 |
| 385–512 | 45.104 | 44.524 | 43.577 |
| 513–640 | 44.993 | 44.421 | 43.610 |
| 641–768 | 44.658 | 44.129 | 43.512 |
| 769–896 | 44.527 | 44.028 | 43.463 |
| 897–1,024 | 44.507 | 44.057 | 43.573 |

Depth four improves early and then has a flat, noisy sequence. Its
`continuing_improvement=true` assessment compares the starting map with the
endpoint on common points. It does **not** compare late checkpoints, and
`training_complete` is explicitly false. Calling that label evidence of
continued late learning would overstate what was checked. Conversely these
unpaired changing-minibatch losses do not prove a stationary optimum.
Paired late-checkpoint evaluation is the missing discriminating measurement.
Simply promising that more identical updates will fix the map is unsupported.

## Decisions and next justified repair

The next repair should distinguish slow/noisy optimization from unused or
restrictive nonlinear capacity. First compare saved late checkpoints on common
points and measure layer-specific gradient variability and actual parameter
updates at the endpoint. Then use that evidence to choose a controlled
optimizer/variance change or a parameterization change that can learn the
remaining observation-weight shape. A trainable affine normalization, properly
scaled conditioner inputs, revised scale parameterization, or an explicitly
nonlinear monotone marginal are hypotheses to test separately; none is
selected as a new default by this audit. More depth with the same unexamined
conditioning and optimizer is not a demonstrated remedy.

Any continuation needs a predeclared allocation and late learning assessment,
the standard 1,000-point report, and eventual downstream sampler evidence.
Perfect Gaussianization is not a prerequisite for useful NeuTra sampling.
These residuals motivate repair; they do not by themselves reject a map for
all HMC use or establish that ordinary mass adaptation is needed.

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Retain the current RKL first-gradient formulation as mathematically supported | Change-of-variables derivation and executed source agree | Earlier local derivative checks passed | Untested states and approximation of the exact state-space likelihood | Preserve score checks when changing training | Globally correct target or converged optimization |
| Treat nonlinear learning as the main unresolved fitting mechanism | Almost-affine proposals coexist with predominantly nonlinear residual | All three full banks valid/finite; no numerical candidate veto | Initialization, conditioning, optimizer and representational causes are confounded | Measure endpoint layer updates/noise; test a targeted nonlinear repair | Unique causal diagnosis or impossibility of the existing flow family |
| Stop blaming ongoing routine clipping alone | Intervention clipping is 0–0.68% with large remaining residual | Control still fails the occasional-guard role | Earlier clipping history and unseen gradient tails | Retain monitoring and controlled contrasts | Optimal clipping or a statistically ranked winner |
| Keep convergence unestablished | No late paired assessment; depth loss flat/noisy | No continuation veto from candidate quality alone | Real plateau versus stochastic noise and slow feature learning | Assess distinct late checkpoints before allocating a long continuation | All maps still improving at their endpoint |

| Inference status | Finding |
| --- | --- |
| Hard veto screen | Three complete banks have finite valid evaluations; numerical failures do not explain their recorded residuals. Batch128 lacks a complete bank. |
| Statistically supported ranking | None: one training stream per repair, no declared ranking analysis. |
| Descriptive-only differences | Residual shares, affine-fit fractions, block losses, activation/scale summaries, and finite-bank derivative means. |
| Default-readiness | No map, optimizer, architecture, or sampler promotion. |
| Next evidence needed | Endpoint optimization/feature diagnostics, controlled repair with uncertainty, and downstream sampling/reference evidence for a posterior claim. |

Terminal skeptical review: a small mean-square departure from an affine map
can conceal important derivative changes, and a fitted proposal may place
little mass in unvisited posterior regions. Both limit this audit. The strongest
alternative explanation is a useful but imperfect transport whose residuals
look poor under this stringent Gaussian diagnostic; only downstream sampling
can resolve usefulness. A successful nonlinear correction under unchanged
optimization would strengthen the capacity/conditioning explanation; successful
continued optimization of the existing family would weaken a capacity claim.
No new training or target run was performed to settle that causal distinction.

The reproducible saved-data scripts and results are
[analyze_saved.py](artifacts/q20-training-math-audit-2026-09-23/analyze_saved.py),
[inspect_saved_transforms.py](artifacts/q20-training-math-audit-2026-09-23/inspect_saved_transforms.py),
[saved-analysis.json](artifacts/q20-training-math-audit-2026-09-23/saved-analysis.json),
and [transform-inspection.json](artifacts/q20-training-math-audit-2026-09-23/transform-inspection.json).
The [audit manifest](artifacts/q20-training-math-audit-2026-09-23/manifest.json)
records source/input identities, commands and execution scope. Routine source
inspection and saved-artifact analysis do not extend or consume the target-run
allocation: campaign balance remains 143975.28597232018 seconds and diagnostic
balance 7.711412891243526 seconds. The pending request for additional diagnostic
time has not been assumed approved.
