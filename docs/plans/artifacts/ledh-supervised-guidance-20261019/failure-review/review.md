# Why the present guidance has not established better LEDH scores

20 October 2026. Reviewed implementation: commit `4abbaf9f1`, branch
`ledh-refactor-with-policy-fix`. Scope: the full shared analytical LEDH consumer
and its T5 simulation-supervised pilot, not invalidated historical LEDH results.
No new GPU experiment, guide fit, OT tuning, or default change was performed.

The population score-risk identity is correct under its stated assumptions.
The experiment does not show that optimizing that objective failed: it selected
among existing message-fitted guides, without optimizing their coefficients
against score risk. There is also a concrete design gap. A future message is
fitted as a function of the physical state, but the current adapter applies it
before the LEDH flow. Even an exact message then generally produces a different
tilt after the flow. The importance correction is still valid. This difference
can make a perfect guide harmful in an exact Gaussian example; its contribution
to the SIR result has not yet been isolated.

The measured result is mixed, not uniformly negative. Unguided LEDH has the
lowest observed validation loss. All guided test averages are lower, but every
guide loses on one test sequence, and four independent test sequences provide
no established statistical ranking. The enormous noise-score discrepancies
also deserve attention beyond merely increasing the number of seeds.

## 1. What the mathematics does and does not prove

Let `H(X,Y)` be the complete-data fixed-path score and
`s(Y)=E[H(X,Y)|Y]` the marginal score. Let `A(Y,U)` be the computed LEDH score,
with particle randomness U independent of the simulated latent label given Y.
For a common frozen positive semidefinite metric W, ordinary square
integrability and Fisher's differentiation-under-the-integral assumptions give

\[
 E\|A-H\|_W^2=E\|A-s\|_W^2+E\|H-s\|_W^2.
\]

Expand `A-H=(A-s)+(s-H)` and condition on `(Y,U)`: the cross term is zero.
The last term is independent of the candidate. Thus differences of population
supervised risks equal differences of score MSE. This remains true for a biased
finite-particle estimator. The SIR label uses derivatives at a fixed latent
path, not derivatives through the simulated path.

The metric here is fixed from separate calibration labels before validation
and test. The identity applies conditionally on that calibration. In-sample
calibration losses using the same labels to choose W need not be unbiased risk
assessments. A selected validation minimum is also not an unbiased estimate of
its population risk. These distinctions are preserved in the manuscript.

Neither the identity nor the correctness of importance weights proves that a
small candidate bank contains a better estimator, that a noisy validation set
will find it, or that a finite-program derivative is the model score. Global
square integrability of every possible finite LEDH estimator remains an
assumption, not something established by 96 finite rows. The previously
checked finite differences verify the derivative of the implemented program.
They do not measure its error against the marginal score.

One elementary calculation explains why likelihood-oriented fitting is
insufficient. If `Zhat(theta)=Z(theta)(1+e(theta))>0`, then

\[
 \nabla\log\widehat Z-\nabla\log Z
 =\frac{\nabla e}{1+e}.
\]

For scalar theta, `e(theta)=delta sin(theta/delta^2)` has uniformly small
amplitude but derivative error `1/delta` at zero. Weight variance, ESS, and
message-value regression do not control this derivative. They are useful
diagnostics or fitting objectives, not certificates of score accuracy.

## 2. The guide changes meaning when it passes through the flow

Condition on one ancestor, the incoming cloud, and its affine LEDH map
`F(z)=Bz+c`. The shared auxiliary trajectory makes B and c independent of that
particle's sampled innovation. Let r be the unguided pre-flow density, and let
`q_F(x)=r(F^{-1}(x))/|det B|` be its post-flow density.

The quadratic arm draws from `r(z)h(z)/C_pre`. Change of variables gives

\[
 q_{\rm implemented}(x)
 =q_F(x)h(F^{-1}(x))/C_{\rm pre}.
\]

Tilting the post-flow proposal by the physical future message would instead
give `q_desired(x) proportional to q_F(x)h(x)`. To implement that construction,
the pre-flow factor must be `h(F(z))`. For
`log h(x)=-x'Lambda x/2+b'x` with symmetric Lambda, the required coefficients are

\[
 \widetilde\Lambda=B^\top\Lambda B,\qquad
 \widetilde b=B^\top(b-\Lambda c).
\]

Expansion leaves the constant `-c'Lambda c/2+b'c`, which cancels when normalizing
the pre-flow proposal. For a Gaussian base with positive definite covariance Q, integrability
requires `Q^{-1}+B'Lambda B` positive definite. B and c vary with theta and the
incoming cloud. Their derivatives must enter the analytical recursion even
though the fitted Lambda and b are frozen. This is not fixed by merely changing
the value coefficients and leaving their tangents zero.

This is a design limitation, not a deviation from the documented pre-flow
algorithm. The old manuscript explicitly says to apply the flow after the
guided draw, and its Jacobian/density correction is correct. What is unsupported
is transferring the optimal future-conditioning argument to that composition
without accounting for the changed coordinates. Even the aligned construction
is an exact smoothing conditional only when both the base post-flow conditional
and the future message are exact. Mixture-product moment closure adds another
approximation; it is not an exact mixture tilt.

**Exact counterexample.** A static latent state has prior N(0,1); conditionally
independent observations with `Y_j | X ~ N(X,1)` equal 2 and 1. After the first observation the state is
N(1,1/2), generated exactly by `F(z)=1+z/sqrt(2)`. The exact future message is
`h(x) proportional to exp(-(x-1)^2/2)`, and the final target is N(1,1/3).
Applying h before F instead gives N(1.353553,1/4). The unguided mean was already
correct. `KL(target || proposal)` increases from 0.036066 to 0.272826. Applying
`h(F(z))` produces the exact target. A second example with future observation 3
also shows the coordinate discrepancy, but its unaligned guide still lowers KL;
the mismatch does not imply that every guide must harm every problem.

The exact examples and KL calculations are in `analysis.json`. MathDevMCP/SymPy
checked the scalar quadratic substitution in `mathdev-pullback.json`. The new
test `test_quadratic_guide_is_applied_in_preflow_coordinates` invokes the full
shared nonlinear-observation executor, checks its actual pre-flow sample and
post-flow child against the shared affine map, reconstructs the inverse, and
checks the pulled-back quadratic identity. Existing actual-flow Jacobian and
total-score derivative checks pass alongside it. Thus this is an executable
call-chain check, not inference from an unused helper. None of these checks
measures the size of this effect in SIR.

## 3. What was actually fitted and selected

The checked endpoint chain is
`supervised_guidance_pilot.main -> SIRGuideFitter.fit/_export ->
build_backward_guided_ledh_kernel -> _value_and_analytical_score_impl ->
guided_gaussian_draw -> _flow_substeps_with_tangent -> corrected weights ->
shared correction/reset recursion`.

| Stage | Actual computation | Consequence |
| --- | --- | --- |
| cSMC fitting | One policy improvement from a zero-policy 512-particle pilot | A bounded message-fit hypothesis, not converged score-risk optimization |
| Message export | Approximate the next fitted policy's integrated future factor on a local Gaussian design | Accuracy outside that design and its score effect remain unproved |
| Projected arm | Project fitted quadratic information to PSD, preserving the gradient at the center | Changes curvature; valid proposal correction does not recover lost proposal quality |
| Kernel fitting | Fit 32 positive RBF components to the already approximate cSMC message; choose bandwidth by message-value MSE | An approximation to an approximation, not independently trained original neural KDM |
| Supervision | Use complete-data scores to compare six fixed guide variants | Coefficients are not trained against the derived score objective |
| Selection | Nominate one cSMC and one kernel variant within their families | Unguided is reported but is not eligible for these family nominations |
| Score | Analytical total derivative with the fitted guide held fixed | Correct finite-program target, distinct from differentiation through refitting |

Source anchors at the committed implementation: `supervised_guidance_pilot.py`
lines 115 (scales), 140 (fit), 193--203 (selection);
`sir_backward_guidance_diagnostic_tf.py` lines 101--187 (export/RBF),
191--235 (one fit); `ledh_backward_guide_tf.py` lines 175--259 (Gaussian draw and
same-density correction), 327 (shared endpoint);
`ledh_canonical_score_tf.py` lines 405--424 (guide placement), 529--553 (actual
flow/density ratio), 851--893 (diagnostic trace), 1312 (shared affine flow).
The ten T5 recorded source hashes match the inspected runtime sources.

This is not evidence that the replicated cSMC algorithm itself failed. In its
own pilot, mean ESS changes from 181.27 to 504.58 out of 512; the smallest
recorded post-initial ESS changes from 19.61 to 456.39. These are descriptive
statistics from the separate fitter, not LEDH ESS. They show that its fitting
objective can look good while the downstream score objective does not.

PSD projection was active at all 32 nonterminal guide fits. Relative to the raw
quadratic fit, its message MSE is 1.83--14.06 times larger. Nevertheless, the raw
arm has slightly worse validation loss and only 1.25% lower descriptive test
loss than the projected arm. Removing projection alone did not resolve the
observed problem. The repaired T5 run accepts an indefinite guide only when the
combined Gaussian precision is positive definite.

All 32 RBF bandwidth selections hit 2.0, the upper grid endpoint. RBF message
MSE is 0.678--0.989 of a constant fit's MSE. This nominates an inadequate design
or bandwidth range as a possible fitting issue; it does not establish either
one, and no new settings were selected from these test diagnostics.

## 4. What the saved SIR data say

The final T5 experiment uses N=64, theta=(0,0,0), FP64 GPU/XLA, two calibration
sequences, two validation sequences, four test sequences and two particle
streams. OT epsilon 1, five Sinkhorn updates and 32 terminal balance updates
are unchanged. All 96 rows are valid; six directional checks passed in the
original run. The loss is the sum of squared score-label residuals divided by
the frozen coordinate scales `(618.405,142.723,5.575)`. These are label losses,
not observed-score-oracle errors.

| Guide | Validation loss | Test loss | Test particle-dispersion fraction |
| --- | ---: | ---: | ---: |
| Unguided | 1653.53 | 65902.87 | 38.25% |
| cSMC half | 4515.01 | 23443.86 | 18.29% |
| cSMC projected | 3136.57 | 9575.05 | 4.72% |
| cSMC raw | 3151.98 | 9454.93 | 5.15% |
| Kernel information | 3328.31 | 15098.74 | 9.15% |
| Kernel product | 3592.32 | 10847.56 | 41.23% |

The last column uses the exact finite-sample identity

\[
 \frac1R\sum_r\|A_r-H\|_W^2
 =\|\bar A-H\|_W^2+\frac1R\sum_r\|A_r-\bar A\|_W^2.
\]

The remainder is the squared error of the two-replicate mean against a noisy
label. It is **not** an estimate of population squared bias. It shows why
reducing the displayed particle spread alone is not enough: for the projected
cSMC arm, 95.28% of test loss is in that remaining term.

Every method's validation and test loss is more than 99.3% attributable to the
scaled observation-noise coordinate. This is a property of the chosen metric,
not evidence that the unscaled infection/removal errors are small. With nine
observed coordinates, the noise label is a sum of `9T` centered squared standard
normal innovations. Its unconditional variance is `18T=90` and SD 9.487 at T5.
This does not give its conditional variance on a particular dataset or an exact
observed-score reference. It nevertheless provides a useful population scale
for the discrepancies below; by conditional Jensen, `E[s_3(Y)^2] <= 90`.

| Test sequence | Noise label | Unguided noise score, stream 1 | Stream 2 | Unguided total loss |
| --- | ---: | ---: | ---: | ---: |
| 261029505 | -1.113 | 3733.846 | 233.801 | 225522.96 |
| 261029506 | 9.941 | 640.564 | 558.432 | 11261.51 |
| 261029507 | 0.105 | 1005.535 | 522.536 | 20663.17 |
| 261029508 | -3.528 | 344.601 | 502.768 | 6163.83 |

The first sequence contributes 85.55% of unguided aggregate test loss. Omitting
it reverses the descriptive mean advantage of half-strength and kernel-
information guidance. Projected cSMC, raw cSMC and kernel-product still have
lower leave-one-out means for every single-sequence deletion. Thus it would
also be wrong to say that all apparent gains disappear without the outlier.
Every guide nevertheless loses on sequence 261029508, and validation favors
unguided. No post-hoc deletion defines a new winner.

The two particle streams are reused across datasets and partitions. They are
independent of the label generator, so this does not invalidate the risk
identity. It does prevent treating all particle rows as independent evidence,
and limits generalization over particle randomness. A future untouched test
should use fresh dataset and particle streams after calibration/selection.
The pilot also covers one theta and a tiny horizon, with scaling estimated
from only two calibration labels. It establishes no cross-parameter or
long-horizon performance.

The analysis verifies all 96 unique keys and recomputes saved losses with
maximum relative arithmetic discrepancy 1.285e-16. That is an integrity check,
not a scientific tolerance or an obstacle to progress. The earlier T2 raw-arm
rejections are preserved as implementation failures; they are not evidence
against raw guidance and are not pooled into this comparison.

## 5. Verdict, unresolved causes, and next discriminating work

| Question | Classification | Evidence or limit |
| --- | --- | --- |
| Does the supervised-risk proof hold? | Correct under stated assumptions | Conditional orthogonality and fixed-path Fisher identity |
| Is the actual guided density corrected through the actual flow? | Correct for tested consumer | Shared endpoint, density/Jacobian and derivative tests |
| Does the current composition equal a future-message tilt of the post-flow cloud? | Wrong relative to that claim | Change-of-variables derivation and executable consumer check |
| Did this pilot optimize guide coefficients for score accuracy? | No | Finite-bank selection; coefficient fitting uses message objectives |
| Did the original neural KDM fail on SIR? | Unsupported | This consumer used a positive-RBF adapter of a cSMC fit |
| Is flow misalignment the main cause of the SIR score errors? | Not checked | No aligned/unaligned matched SIR comparison |
| Is OT or Contract E the cause? | Not checked | Current evidence does not isolate reset, flow, correction, or tangent contributions |
| Does guidance reliably improve the score? | Unsupported as a ranking | Conflicting small validation/test sets, no predeclared uncertainty support |

The most discriminating next repair is an optional **flow-aligned quadratic
guide**, followed by an unchanged-guide/unguided/aligned comparison with the
same physical model, full executor and OT `(1,5)`. Before SIR, verify exact
Gaussian value/density identities and the analytical derivatives of B, c and
the pulled-back coefficients. The new construction must use the shared map,
not reconstruct a second or simplified flow. In SIR, save already-computed
per-step log weights, score/tangent contributions and covariance validity
margins; the present runner discards most of the shared trace. These observables
can distinguish a proposal mismatch from a downstream sensitivity problem.

In parallel with that comparison, the next plan should actually fit or search
guide controls using the supervised downstream objective on a sufficiently
large calibration/validation set, include the zero guide as a candidate, and
freeze selection before fresh test datasets and particle streams. Moment
features should be chosen from the model's score terms and evaluated downstream;
matching arbitrary higher moments alone supplies no score guarantee. Keep
the existing OT values fixed and record target-specific settings as hypotheses.
Do not tune on the four test sequences dissected here. A larger run needs its
own bounded evidence contract, uncertainty analysis and independently held-out
simple heuristic adversaries. This review does not launch that campaign.

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Retain unguided as comparator; promote no guide | Score-risk superiority not established | Numerical validity passes; conditional non-harm is not established | Small dataset bank and unmeasured coordinate-alignment effect | Prove/test the aligned consumer, then plan fresh supervised comparison | Rejection of cSMC/KDM research direction |
| Keep current evidence | Keys, arithmetic and runtime source hashes agree | No corruption or derivative failure found in final T5 evidence | Limited numerical scopes | Preserve artifacts and add the diagnostic trace in the next experiment | Global finite-estimator correctness or HMC readiness |

The conditional loss on sequence 508 is a promotion concern; it does not
invalidate the harness or require stopping a repair phase. It is post-hoc
descriptive evidence, not a newly declared statistical test.

| Inference status | Finding |
| --- | --- |
| Hard veto screen | Final T5 numerical checks pass; earlier T2 raw failure is repaired and separately retained |
| Statistically supported ranking | None |
| Descriptive-only differences | Validation favors unguided; test averages favor guides; all guides lose on test 508 |
| Default-readiness | No new guide or setting is ready for promotion |
| Next evidence needed | Aligned-consumer checks, genuine downstream supervision, fresh independent data/particle streams, predeclared paired uncertainty and heuristic checks |

The strongest alternative explanation is a large finite-N score error in the
shared LEDH approximation that survives either guide placement. Alignment may
fix an interpretation without improving the score. Evidence of no aligned
benefit, alongside unchanged large per-step tangent errors, would redirect the
repair toward that approximation. The weakest evidence is the four-sequence
test bank. Increasing it can resolve statistical ambiguity but cannot by itself
repair a poorly chosen proposal family or incorrect optimization objective.

## Verification and provenance

`analyze.py` is Python-standard-library post-run arithmetic only. It writes
`analysis.json`; no runtime selection consumes its output. Its examples are
independent mathematical references. The new and two existing consumer tests
pass in the original `tftwogpu` environment with GPU devices deliberately
hidden: 3 passed in 26.29 seconds, 16 deselected. An initial attempt in the wrong
`tf-gpu` environment failed at collection because the compiled TensorFlow custom
op had an incompatible symbol; `consumer-tests.log` retains that failure.
No package, custom op, or environment was changed. `consumer-tests-02.log`
retains the successful retry. After tightening the new test's nonconstant-flow
check, that test and the existing normalized-Gaussian-density test also passed:
2 passed in 8.47 seconds, 17 deselected, in `consumer-tests-03.log`. Thus four
distinct focused tests pass; the repeated coordinate test is not a fifth test.

The 88-page PDF builds without warnings in two pdflatex passes. Rendered pages
80--81 contain the new proposition, proof and interpretation; visual inspection
found no clipped equations or text. The adjacent page 82 was also inspected.
The prior 86-page PDF was preserved separately before editing. No prior
derivation was removed.

The original manuscript audit and external-review limitations remain in
`../audit.md`: MathDevMCP did not certify the entire document, and Claude review
could not authenticate. This review adds a successful symbolic scalar
substitution check, direct derivations and executable checks; it does not
claim independent full mathematical certification. The new proposition,
counterexample and measured qualifications have been added to the manuscript.
See `review-run.json` for commands, budgets and artifact paths.
