# iAPF fit and floor choice study: result

**Interpretation update, 26 September:** the [criteria audit](qualification-review-20260926.md)
requires revising the qualification rule before reuse. Zero nominees is correct
under the original screen; it does not establish that every tested choice is
unsuitable for a faithful or useful reconstruction. Numerical findings remain
valid. Floor protection and complete-proposal quality require separate judgments.

The diagnostic study is complete. None of the 14 solver policies qualified for
untouched confirmation. The floor study also found conditional failures against
simple proposals. No fit/floor policy can yet be frozen for the full iAPF
replication. These failures do not reject iAPF as a research direction.

The [plan](../../iapf-eq15-choice-calibration-20260925.md) received a skeptical
self-review before execution. It corrected stale evidence labels, an understated
grid, an arbitrary tolerance, missing floor mathematics and the meaning of a
warm start. This was not an independent review.

## Execution and checks

Attempt 01 evaluated 19 fixtures under 14 policies: 266 fits, plus 324 independent
floor cases. Attempt 02 replayed 26 fits from frozen source and retained their
parameters. All matched the original status, loss, shape and density-scale
change. Total: 292 fits/replays, 63.781892 CPU worker wall seconds, zero GPU time.
Both commands exited zero; 13 source snapshots passed SHA-256 checks.

This is an independent R FP64 reference with GPU deliberately hidden, one CPU
process and one BLAS thread. Python supervision and result assembly use the
standard library. It does not establish TensorFlow/TFP or GPU/XLA conformance.
Source snapshots, inputs, commands, seeds and environment are in the
[main manifest](attempt01/manifest.json) and
[audit manifest](attempt02/manifest.json).

Commands from /home/chakwong/BayesFilter:

    /home/chakwong/anaconda3/envs/tftwogpu/bin/python scripts/run_iapf_choice_calibration.py --timeout 900
    /home/chakwong/anaconda3/envs/tftwogpu/bin/python scripts/run_iapf_choice_calibration.py --audit --timeout 240
    /home/chakwong/anaconda3/envs/tftwogpu/bin/python scripts/review_iapf_choice_calibration.py

Joint/profiled derivative finite-difference relative errors were 9.23e-11 and
2.00e-11. Profiled-value identity, target rescaling and the one-dimensional
proposal normalizer passed. QR exact-Gaussian starts were preserved at
dimensions 1, 5, 20 and 80. This checks preservation of an exact start; it does
not prove recovery from a poor start.

## The objective and its failure mode

The target is accepted-manuscript Section 5.1, Eq. (15):
\[
L(m,V,\lambda)=\sum_i\{\mathcal N(x_i;m,V)-\lambda y_i\}^2,
\qquad V\text{ diagonal},\quad y_i>0.
\]
See the [local paper text](/home/chakwong/BayesFilter/.localresources/papers/guarniero-johansen-lee-2017-iterated-auxiliary-particle-filter.txt:911).
The author optimizer and floor formula have not been recovered.

Writing \(p_i=\mathcal N(x_i;m,V)\), differentiation in \(\lambda\) gives
\[
\lambda^*=(p^\top y)/(y^\top y).
\]
The profiled implementation therefore minimizes the same scalar objective.
Fixed positive loss scaling and a fixed coordinate change preserve minimizers.
Box restrictions change the feasible set and were labeled constrained diagnostics.

For \(V=sI\), \(s\to\infty\), every density at a finite observed point tends to
zero, as does \(\lambda^*\), so \(L\to0\). This does not imply useful target
shape. Profiling cannot remove this escape. The Eq. (16) floor is added after
fitting and also cannot repair it.

## Choice dispositions

| Choice | Checked result | Disposition |
| --- | --- | --- |
| Profile versus joint amplitude | Algebra/derivatives agree; both have fit failures | Profiling is an exact dimension reduction for further diagnostics, not a repair |
| QR versus target-weighted moments | QR preserves all four exact starts; moments do not pass all four | Keep QR as an initializer with explicit failure handling, not a replacement objective |
| nlminb versus damped Gauss–Newton | No arm passes every generated class; all fail correlated Gaussians | Neither is qualified as the reconstruction default |
| More iterations, loose stopping, unscaled coordinates | None of these ablations produces a nominee | More compute alone is unsupported as the repair |
| Boxes of radius 0.25, 1 or 4 | Small boxes suppress extreme escape by hitting constraints; larger boxes still permit escape | Arbitrary bounds do not justify an unconstrained Eq. (15) claim |
| Tail- or peak-scaled floors | Proposal effect varies sharply with dimension and parent | Do not freeze an uncalibrated power of N |
| Minimum mixture-probability floors | Explicit interpretation; distant parents can become almost untwisted | Interpretable protection candidates, not a global default |
| Zero floor | Violates positive-floor requirement; underdispersion can give infinite second moment | Comparator only |

Correlated fixtures have separate held-out target evaluations. Saved failures
do not; their training fit measures are explanatory. The original limited
numerical screen passed all ten saved inputs for QR/LM, but this is not a
useful-fit verdict.

The terminal audit explains misleading convergence. These are gradient infinity
norms in fixed standardized coordinates for profiled QR/LM. The third column
divides by the final squared density norm in the same fixed loss units:

| Correlated dimension | Absolute gradient | Gradient relative to density scale |
| --- | --- | --- |
| 5 | 4.46e-09 | 0.432 |
| 20 | 7.53e-09 | 9.11 |
| 80 | 8.89e-09 | 6.18 |

Gradients below 1e-8 coexist with substantial relative gradients. Those flags do
not establish useful local minima. Relative gradients and finite-difference
Hessians are post-hoc explanations, not a changed selection rule. All three
correlated fits also failed the held-out shape comparison. See
[stationarity.csv](attempt02/stationarity.csv) and retained RDS parameters
under attempt02/fits/.

## Floor effects and heuristic failures

For a Gaussian twist plus constant \(c>0\), the original transition enters with
\[
\pi(\mu)=\frac{c}{Z(\mu)+c},\qquad Z(\mu)=\mathcal N(\mu;m,Q+V).
\]
For this study's \(Q=I\),
\[
Z_{\max}=(2\pi)^{-d/2}|I+V|^{-1/2},\qquad
c=\frac{\delta}{1-\delta}Z_{\max}.
\]
Thus \(\pi(\mu)\ge\delta\), with equality at \(\mu=m\). One constant is used per
twist, independent of the parent. This is a derived local reconstruction, not
a recovered author formula.

For a centered healthy twist, \(V=I\), \(N=1000\), the base probabilities are:

| Floor rule | Dimension 1 | Dimension 80 |
| --- | --- | --- |
| Peak / N² | 1.41421e-06 | 0.999999 |
| Peak / N⁴ | 1.41421e-12 | 0.523699 |
| Peak / N⁸ | 1.41421e-24 | 1.09951e-12 |
| Tail / N⁸ | 1.8366e-23 | 1.7265e-50 |
| Minimum base probability 1/N | 0.001 | 0.001 |
| Minimum base probability 1/N² | 1e-06 | 1e-06 |

Peak/N² becomes almost entirely untwisted at dimension 80, while tail/N⁸ is
negligible. Minimum-probability floors have an explicit meaning, but at a
parent displaced by one unit in each of 80 coordinates, the 1/N and 1/N² rules
give probabilities 0.99999794 and 0.99794309. Any fixed positive c gives
\(\pi(\mu)\to1\) sufficiently far from the fitted mean.

**In the one-dimensional underdispersed case, every tested floor loses to a
plain transition proposal.** The normalized importance-weight second moment
is 1.15470 for the plain transition, 14.41866 for minimum probability 1/N,
456.20926 for 1/N², approximately 2.67e11 for tail/N⁸, and infinite without a
floor. Positive floors help tail protection here, yet finiteness alone cannot
qualify the proposal.

These are deterministic local Gaussian integrals, not full-filter performance
estimates. Zero-floor integrals agree with independent Gaussian formulas.
Of 324 cases, 78 one-dimensional integrals completed, three zero-floor cases
were analytically divergent, and 243 high-dimensional second moments were not
evaluated. High-dimensional mixture probabilities are available.
The conditional heuristic table has 243 comparisons and 100 losses; every
floor policy has at least one loss. See
[comparisons](attempt02/conditional-heuristics.csv) and
[floor cells](attempt01/floors.csv).

## Decision and inference

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Close diagnostic study | All scheduled cells/replays accounted for | No control/artifact failure | Finite fixture coverage | Preserve result; update master | Replication completion |
| Do not freeze solver | Zero of 14 policies qualifies | Shape, escape/boundary and misleading stopping | Useful local branch unspecified | Scale-aware stopping and justified local branch | Rejection of iAPF |
| Do not freeze floor | Protection characterized | Conditional heuristic losses | Downstream cost and high-dimensional moments | Evaluate non-harm with a qualified fit | Universal floor ranking |
| Confirmation/pilot not admitted | Required nominee absent | Scientific prerequisite | Adaptive-filter behavior | Repair fitting first | Incomplete execution of this bounded study |

| Inference status | Finding |
| --- | --- |
| Hard veto screen | Specific numerical, boundary, held-out shape and heuristic failures supported |
| Statistically supported ranking | None; deterministic comparisons apply only to their stated cases |
| Descriptive-only differences | Runtime, finite-fixture counts and observed high-dimensional behavior |
| Default-readiness | No solver/floor default and no TF/GPU/XLA promotion |
| Next evidence needed | Qualified local fit, untouched confirmation, then adaptive pilot with Kalman and simple-filter comparisons |

The next experiment should calibrate scale-aware stopping on the unchanged
objective, preserving exact solutions and exposing escape. It must specify
what makes a finite local solution useful and justify constraints defining that
branch. Repeating arbitrary bounds or silently switching to a normalized or
log-density objective would not resolve the source question. A materially
changed objective remains a separate scientific-direction decision.

The strongest alternative explanation is that diagonal approximation, finite
cloud coverage or initialization explains some failures. Exact-start checks
do not rule this out. A useful interior solution under a justified local
policy, then untouched downstream validation, would overturn the no-promotion
verdict. Generalization beyond the fixed fixtures is the weakest evidence;
saved failures and one-dimensional integrals cannot supply it.

The [machine-readable decision](decision.json) preserves terminal checks,
counts, heuristic vetoes and limitations. Both scientific replications remain
incomplete. KDM/IWSG work was outside this study.
