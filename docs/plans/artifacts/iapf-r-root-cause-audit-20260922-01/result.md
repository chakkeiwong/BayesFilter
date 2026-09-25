# Why the R iAPF reconstruction fails to learn some guides

The failed reconstruction has distinct problems in its fitting step. The
printed absolute-density objective permits the density to disappear from the
training points. Our weighted-log alternative can turn omitted cross-coordinate
terms into an invalid diagonal variance when its weights concentrate. A third
problem confounds the warm-start experiment: a strict preliminary QR fit can
abort before the optimizer receives the previous solution. These conclusions
are supported by the source, algebra and replays below. They do not establish
failure of the authors' unpublished numerical procedure.

The immediate goal remains the paper's first linear-Gaussian likelihood study:
learn proposal guides and estimate `p(y1:T)` accurately with an initial 1,000
particles, through T=100 and dimensions up to 80. This R reference is intended
to help debug the filtering research. This audit makes no new LEDH, KDM,
TensorFlow, GPU, gradient or HMC claim.

## The filter and the fitted function have different jobs

For a latent state x at time t, the ideal guide is the likelihood of the
current and future observations conditional on that state:

\[
 \psi_t^*(x)=p(y_{t:T}\mid X_t=x),\qquad
 \psi_t^*(x)=g_t(x)\int f(x,x')\psi_{t+1}^*(x')\,dx'.
\]

The proposal is proportional to `f(x_previous,x) psi_t(x)`. Its importance
weight includes the inverse guide and the integral of the next guide. GJL's
Propositions 1--2 prove the cancellation preserving the likelihood and the
constant terminal likelihood estimate with the exact future guide. These
identities do not guarantee that a finite-cloud regression learns a useful
guide. The variance proposition and appendix explicitly depend on the quality
of the fitted guide relative to the ideal guide.

The checked consumer path is:

| Paper operation | Current executed code |
|---|---|
| First-study model, N0 and data | `run_iapf_author_choices.R:14,74,84`; `reference_iapf_paper.R:416,426` |
| Algorithm 4, constant initial guides, learning passes, fresh final estimate | `iapf_choice_run:113` → `iapf_iterate:366` |
| Algorithm 5, adaptive resampling and retained weights | `iapf_apf:108` → proposal/draw at 139--142 → weights at 143--147 |
| Algorithm 3, use the newly fitted next-time guide | `iapf_choice_backward:99` → `iapf_backward_log_target:334` |
| Equations 15--16 versus explicitly different log objectives | `iapf_choice_fit:11` → `iapf_fit_objective:166`, or weighted QR at 42--57 |

The `iapf_choice_*` functions are in `reference_iapf_author_choices.R`; the
shared filter, controller, target and objective are in `reference_iapf_paper.R`,
both under `docs/benchmarks/`. `clouds[[t]]` holds particles
after proposal propagation and before any subsequent resampling based on the
time-t observation weights. This is the cloud in Algorithm 3, not a missing
post-resampling correction. The reverse loop uses the freshly fitted
`twists[[time+1]]`, as required by the paper.

Two saved d80 failures were replayed using their original observations and
random seeds. Both t99 clouds and targets matched **exactly**. Independently
computed Gaussian-product target values agreed within 5.7e-14. With the exact
full Gaussian guides, the same shared APF agreed with the Kalman terminal
log likelihood within 7.3e-12. These are executable checks of the examined
paths, not exhaustive certification of every filter branch or model.

## 1. The printed loss can reward disappearing densities

At the fitting points, let `b_i>0` be the backward targets and
`p_i=N(x_i;m,V)` the fitted Gaussian densities. Equation 15 minimizes

\[
 L(m,V,\lambda)=\|p-\lambda b\|^2.
\]

Eliminating the scalar amplitude gives

\[
 \lambda_*={p^Tb\over b^Tb},\qquad
 L_* =\|p\|^2-{(p^Tb)^2\over\|b\|^2}
      =\|p\|^2(1-\cos^2\theta).
\]

Thus the objective mixes shape disagreement, `1-cos²(theta)`, with the size of
the fitted densities. Take `V=s² I` with a fixed mean and let s increase. At
every fixed fitting point `p_i→0`; hence `L_*→0`, even when the Gaussian shape
does not approach the target shape. Unless a finite proportional fit exists,
the zero infimum is not attained at finite parameters. The earlier local
source-reconciliation result gives an explicit cloud with no such finite fit.
This is a defect of interpreting the unrestricted printed argmin as a general
fitting prescription, not a theorem that all initialized local searches fail.

The current implementation profiles lambda correctly and uses fixed scaling
within each fit. The defect is therefore not a reversed residual or missing
normalizing constant in this implementation. Fixed scaling cannot remove
the factor `||p||²` from the objective.

There is direct evidence of this mechanism in a saved F1-loose d20 run,
dataset 92100120, replicate 1, outer iteration 4, backward time 38:

| Diagnostic | Start | Returned optimizer point |
|---|---:|---:|
| Fixed-scaled profiled loss | 7.90275e-5 | 0, by underflow |
| Scale-free shape residual | 0.0366927 | 0.999999999892 |
| Change in maximum log density | 0 | -35614.0478 |

The optimizer returned convergence code 0. The existing guard correctly
rejected this result. Looser convergence rules would accept a meaningless
fit. This does not explain every strict-solver iteration limit or candidate
failure. The tested log-variance bounds ±36.043 are a local configured domain,
not the full floating-point range or recovered author constraints.

The new saved-cloud variance profile independently reproduces the geometry:
in replicate 1, loss falls from 6.016e-5 to 3.432e-50 at 64-fold variance,
while the shape residual rises from .04861 to .99039. The path first rises
substantially before falling; it is **not** evidence of monotone local descent
along that path. The actual saved optimizer failure above is the separate
evidence that disappearance occurred in a real run.

## 2. Concentrated weighting amplifies missing interactions

Our weighted-log alternatives are different objectives from Equation 15.
They regress the log backward target on an intercept, each coordinate, and
each squared coordinate. In dimension 80 this is 161 coefficients. A valid
diagonal Gaussian requires every fitted squared-coordinate coefficient to
be negative.

At t99, after the terminal guide `(m100,S100)` is fitted, the target is

\[
 b(x)=\mathcal N(y_{99};x,I)
       [\mathcal N(Ax;m_{100},I+S_{100})+c_{100}].
\]

Removing the negligible floor for this calculation leaves a full Gaussian
whose precision and linear coefficient are

\[
 P=I+A^T(I+S_{100})^{-1}A,\qquad
 h=y_{99}+A^T(I+S_{100})^{-1}m_{100}.
\]

The precision eigenvalues in both replays lie between 1.01472 and 1.52676.
The target is therefore well behaved. Its nonzero off-diagonal entries create
cross terms that a diagonal regression cannot represent. On the saved points,
the floor changes the log target by at most 1.1e-19; it cannot account for the
observed fitted curvature.

To expose the mechanism, write the standardized design as `D=[1,z,z²]` and
the log target as `ell=D beta0+r`, where r contains the omitted cross terms
and the measured floor contribution. When the weighted design has full rank,

\[
 \widehat\beta=\beta_0+(D^TWD)^{-1}D^TWr.
\]

The second term is how missing interactions alter the diagonal coefficients.
Concentrated weights make that projection poorly conditioned. This is more
specific than saying that a high-dimensional filter is difficult.

In saved replicate 1, the coefficient of squared standardized coordinate 70 is

\[
 \underbrace{-0.6065445434}_{\text{true diagonal curvature}}
 +\underbrace{0.6414596050}_{\text{projected cross terms}}
 =\underbrace{+0.0349150616}_{\text{invalid fitted curvature}}.
\]

The floor contribution to that coefficient is 3.2e-27. QR and an independent
SVD calculation agree within 2.3e-9 across coefficients. With exactly diagonal
targets on the **same points and weights**, SVD recovers negative curvature,
with maximum scaled coefficient error 7.9e-10. This isolates omitted
interactions amplified by the weighting, rather than attributing the sign
change solely to rounding or to a genuinely nonconcave target.

The two saved cases also show why ESS alone is not a rank test:

| Saved case, weight exponent | Weight ESS | QR rank | SVD rank | SVD positive curvatures |
|---|---:|---:|---:|---:|
| Replicate 1, 0 (unweighted) | 1000 | 161 | 161 | 0 |
| Replicate 1, .5 | 4.951 | 161 | 161 | 0 |
| Replicate 1, 1 (actual failure) | 2.371 | 161 | 161 | 1 |
| Replicate 3, 0 (unweighted) | 1000 | 161 | 161 | 0 |
| Replicate 3, .5 | 1.00379 | 161 | 161 | 0 |
| Replicate 3, 1 (actual failure) | 1.00000083 | 136 | 161 | 0 |

In replicate 3, all 1,000 weights are positive; weighting has not changed the
design's exact algebraic rank. It has made its condition number 1.37e10.
`lm.wfit` uses tolerance 1e-7 and declares numerical rank 136. SVD with the
recorded machine-epsilon/dimension threshold retains rank 161 and produces
negative curvatures. Its exact-diagonal control has scaled coefficient error
1.4e-8. This supports a solver/rank-policy explanation for this rejection;
it does not establish that the recovered guide works over a complete run.
The other case shows that a solver replacement alone cannot repair all failures.

At weight exponent 2, even the SVD diagonal controls lose identifiability at
the declared numerical threshold. These exponent checks explain conditioning;
they are not a selection of a new weight exponent.

## 3. Most warm-start rejections occurred before warm starting

`iapf_choice_fit` always calls strict `iapf_fit_initial` at line 20. Only at
line 28 does it inspect the previous guide. The QR initializer also defines
the optimizer's coordinates and fixed loss scale, so it is a compulsory
dependency even when a valid previous solution exists.

Of the 32 rejected F2 runs, 25 stopped because this preliminary QR fit was
nonconcave: 12 strict and 13 loose. Every one had a valid previous Gaussian.
A replay of the d20/replicate1/iteration5/t67 case reproduced the rejection,
recording one QR call and **zero optimizer calls**. Therefore these results
reject the implemented QR-dependent F2 variant. They do not fairly test whether
a previous-guide start can avoid a bad current-cloud QR initializer. The
earlier campaign statement that F2 did not qualify remains true; a broader
claim that warm starting is ineffective would be unsupported.

The earlier 126 successful fixed-cloud fits do not settle this issue either.
`run_iapf_author_choices.R:26--44` generated their training clouds with exact
future guides. The real controller starts with constant guides and bootstrap
particles. Those tests checked fits in a different distribution and did not
exercise the actual initial learning failure. They remain useful controlled
checks, but are insufficient evidence of an operational learning procedure.

## The next hypotheses and tests

| Hypothesis | Smallest discriminating test | Interpretation and boundary |
|---|---|---|
| A previous valid guide can bypass the failed QR initializer | Give F2 independent, predeclared coordinate/loss scales from the previous guide; replay the 25 saved rejected fits before any full run | Confirms or rejects the warm-start mechanism; does not recover author settings |
| Some rank rejections are numerical-policy failures | Compare QR and SVD on all distinct saved clouds, with exact-diagonal controls and perturbation checks | Recovering one fit is insufficient for full-filter success; do not merely loosen tolerances |
| Concentration plus missing interactions causes invalid curvature | Preserve positive precision in a constrained fit and test small, predeclared weighting/regularization choices against the same full-Gaussian controls and fresh predictive/smoothing points | A different fitting method must be labelled as an extension; success is valid, stable guide shape, not a smaller raw density loss |
| Restricting the guide to diagonal Gaussians itself limits achievable accuracy | Compare analytic full Gaussian guides with a predeclared analytic diagonal projection, then learned diagonal guides, on the same data | Separates family error from learning error; analytic model-specific guides are controls, not the general solution |
| A local shape-preserving implementation can avoid the Equation15 escape | Test explicit local constraints or a scale-invariant objective on the actual failed points, with the exact-density objective retained as a separate comparator | Earlier relative-L2/box tests already exist; new tests must answer the identified failure, not repeat them unchanged |

Start with removing the warm-start confound and the saved-cloud controls, then
test an explicitly defined fitting repair. A new full-filter campaign should
follow only after those checks. Repeating SD conventions cannot fix a failure
that occurs in iteration zero, before the stopping rule is eligible. Changing
the tested tiny floor cannot repair the demonstrated t99 projection defect.

## Evidence, source support and review

Primary source: GJL, *The Iterated Auxiliary Particle Filter*, local arXiv v2
`../iapf-r-source-reconciliation-20260920-01/sources/arxiv-extracted/iapf_arxiv.tex`:
lines 230--334 (twisting and exact-guide proofs), 387--420 (variance), 470--547
(backward recursion and iteration), 655--723 (fitting and adaptive resampling),
742--822 (first study), 1148--1173 (acknowledged approximation/fitting choices),
1176--1210 (variance proof). These passages were inspected in this audit.
The mathematical fitting verdicts above are local derivations, not statements
attributed to the paper. The appendix does not establish success of this
particular regression or of its numerical optimizer.

The previous source-support/version ledger is
`../iapf-r-source-reconciliation-20260920-01/source-ledger.json`. It records
arXiv/accepted-manuscript access, unavailable publisher technical text, and
unrecovered author settings/code. The public R comparator at
`.localresources/code/sempreteamo-iapf-a8811439/iapf.R:217--235` was inspected:
it minimizes a parameter-dependently normalized residual and supplies its own
starts, so it cannot establish how the authors minimized the printed loss.

Scope of literature coverage: direct-method diagnosis, not a survey. The
paper's backward references on Feynman--Kac variance, particle-filter proposals
and resampling were considered as background; their independent technical
texts were not audited and supply no new claims here. Existing forward/source
reconciliation is retained; two fresh online queries failed with HTTP 502.
Retraction/correction status was not independently refreshed beyond that local
ledger. Citation counts and venue metrics were not refreshed and play no role
in the verdict. The main omission risk is original-author code or revised
published implementation details, not an unexamined general filtering citation.

`attempt01/results` contains 21 passing identity checks, the curvature
decomposition, conditioning and loss profiles. `attempt03/results` contains the
authoritative saved-optimizer summary and three passing call-chain checks.
Attempt02 also passed its call-chain checks but queried a nonexistent summary
field for the final loss; attempt03 corrected the reporting key from `value`
to `profiled_loss`. Prior outputs are preserved. No numerical runtime changed.
Each attempt has its own preserved source snapshot, input hashes, command log
and manifest. Total worker time is 3.761656 seconds, including the reporting
repair, within the 600-second/three-attempt cap and existing deadline.

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Accept the stated local failure mechanisms | Exact replay, independent algebra and solver agreement | 24 distinct identity/call-chain checks pass | Generality beyond selected clouds | Targeted fitting repair and perturbation controls | Every failure has one cause |
| Reject broad interpretation of F2 failure | 25 failures precede optimizer despite valid previous guide | Experimental confound confirmed | Behavior after removing QR dependency | Replay corrected initialization design | Warm starts are ineffective |
| Preserve the filtering identities as viable | Exact-guide APF/Kalman agreement on the actual model | No identity failure found in inspected paths | Other branches, learned fits and complete runs | Use analytic guide as a control | Full implementation or paper replication certified |

| Inference status | Finding |
|---|---|
| Hard veto screen | Actual rank/nonconcavity/underflow rejections are preserved; checks validate the examined mechanism |
| Statistically supported ranking | None; this is deterministic diagnosis on purposively selected failures |
| Descriptive-only differences | ESS, conditions and exponent responses explain these clouds, not population performance |
| Default readiness | No change or promotion |
| Next evidence needed | Independently valid repaired fits, perturbation checks, then complete learning on preassigned data |

Post-run skeptical review: the strongest alternative is that paper-specific
local constraints, initialization and floor choices avoid these problems.
That remains plausible and would not contradict the checked mathematical
identities or our local failures. The weakest evidence is extrapolation from
two d80 clouds; no such extrapolation is needed for the reported coefficient
decomposition. A verified author implementation, successful corrected warm
start replay, or stable constrained fit would change the next research action.
The evidence rejects these fitting candidates and an overbroad reading of the
warm-start test; it does not reject the iAPF research direction.
