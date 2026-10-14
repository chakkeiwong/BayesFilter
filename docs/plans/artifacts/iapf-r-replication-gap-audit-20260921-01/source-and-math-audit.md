# What the R reference reproduces, and what remains different

The R reference implements the checked Gaussian twisting and particle-weight
identities. Its usable fitting route is a reconstruction with a different
objective from the paper. That distinction, together with incomplete numerical
and statistical information, explains why passing code tests has not completed
the replication.

The immediate purpose is an independent reference for debugging BayesFilter.
Reproducing the first study's behavior is useful evidence for that purpose;
it does not establish the correctness of BayesFilter's LEDH, KDM, gradients,
or GPU execution.

## Sources and actual consumers

Primary source: Guarniero, Johansen and Lee, *The iterated auxiliary particle
filter*, DOI10.1080/01621459.2016.1222291,
[arXiv1511.06286](https://arxiv.org/abs/1511.06286). The local PDF is
`.localresources/papers/guarniero-johansen-lee-2017-iterated-auxiliary-particle-filter.pdf`.
The recovered v2 TeX is
`docs/plans/artifacts/iapf-r-source-reconciliation-20260920-01/sources/arxiv-extracted/iapf_arxiv.tex`;
v1 is in the sibling `arxiv-v1-extracted` directory. Technical sections read
for this audit: v2 lines230--326 (twisting/invariance/exact guide),358--417
(FA correspondence/CLT),475--550 (backward fit/controller),653--723
(numerical fitting/adaptive resampling),742--887 (dimension experiment,
alpha sweep and PMMH), and1174--1193 (variance proof).

The earlier retrieval ledger records author publication pages, Crossref,
Warwick's accepted manuscript, both arXiv versions, and bounded code searches.
It did not recover the final publisher technical text or verified author code.
This turn's attempts to recheck the publisher and author page failed with the
web service's HTTP502 error. Thus this is an audit of the available primary
manuscripts, not certification that the final published text is identical.
No author was contacted. The archives and previous retrieval failures remain
preserved; lack of recovered code is not proof that no author code exists.

The actual current R chain is
`compare_iapf_r_plausible_choices.R` -> `iapf_reconstruction` -> `iapf_iterate`
-> `iapf_apf` and `iapf_fit_backward` -> `iapf_fit_gaussian` ->
`iapf_fit_initial`. The frozen `delayed` arm passes `log_quadratic`, floor8
and `after_k`; it does not execute `paper_eq15`. The constrained diagnostic
uses `iapf_constrained_reconstruction` -> the same controller/APF ->
`iapf_local_eq15_backward` -> `iapf_local_eq15_fit`. The new R identity test
checks these consumer bindings as well as actual numerical identities.

| Claim-bearing part | Paper requirement | Checked code and verdict |
|---|---|---|
| Initial/transition proposals | Eq5 Gaussian times positive twist, normalized | `reference_iapf_paper.R` lines49--93: exact two-component Gaussian mixture; tested density/weight identity |
| Particle weights and likelihood | Eq6, Alg5, ESS<=kappa*N, reset only after resampling | `iapf_apf`, line108: correct on tested path identities, exact oracle and FA equivalence |
| Backward targets | Alg3 uses the newly fitted next-time guide | `iapf_fit_backward`, line340: correct ordering; no stale previous-iteration guide |
| Fit family | Diagonal Gaussian plus positive constant | Correct family; QR objective differs from Eq15 |
| Fitting objective | Eq15 density residual with free scale | `iapf_fit_objective`, line163 computes it; active `log_quadratic` route does not minimize it |
| Stopping and final return | Alg4 l>k, k+1 likelihoods, fresh final APF | `iapf_iterate`, line362: checked, including existing fresh-return mutation test |
| Early doubling | Printed N[l-k] is undefined for l<k | Explicit reconstruction choice remains; delayed convention is not established as authors' convention |
| Data/model | Section5.1 dense A[i,j]=.42^(abs(i-j)+1), all identity covariances, T100 | `iapf_paper_data`, line407 and `iapf_linear_model`: matches the stated generative model; authors' realized data unavailable |
| Baselines | BPF10000, FA5000, exact Kalman | Actual R FA endpoint agrees with observation-twisted APF under matched random numbers; SIS10000 is an added diagnostic |
| Theory | Prop1 invariance, Prop2 exact future guide, Prop3 asymptotic variance | Invariance/oracle used directly. Prop3 is not an exact finite-N variance formula for the adaptive-resampling, learned-guide controller |

## Why Equation15 does not define a unique practical fitting procedure

At a fixed backward time, let b_i be the positive target at training particle
x_i, and p_i(m,V) the candidate normalized Gaussian density. The printed fit is

\[
 L(m,V,\lambda)=\sum_i[p_i(m,V)-\lambda b_i]^2.
\]

For any fixed Gaussian, differentiating in lambda gives
lambda*=(p^T b)/(b^T b). Substituting it back gives

\[
 L_*(m,V)=\|p\|^2-\frac{(p^T b)^2}{\|b\|^2}.
\]

Now take V=s^2 V0 and let s grow while the finite training cloud stays fixed.
Every normalized density value p_i goes to zero like s^(-d), so L* goes to
zero regardless of whether the Gaussian has the right shape. Moving the mean
far from the finite cloud provides another escape. For a generic correlated
Gaussian target on a sufficiently rich cloud, no finite diagonal Gaussian is
proportional to all target values. The infimum is then zero without an attained
finite minimizer. The earlier deterministic correlated-Gaussian counterexample
verifies this case without density underflow.

This is a problem with interpreting the displayed unrestricted global argmin
as a complete numerical specification. It does not prove that the authors'
local optimizer failed. Compact bounds make the objective attain a minimum,
but introduce substantive choices and do not ensure uniqueness or that a local
solver finds the global optimum. A converged constrained solution at a boundary
is valid for that constrained problem. Our previous automatic rejection of such
a solution established bound dependence only; it did not establish an invalid
particle filter.

There is a second difficulty even after excluding escape. Rewriting the loss as
sum_i b_i^2[p_i/b_i-lambda]^2 shows how strongly high target values dominate.
The effective number of observations for these squared target weights is
(sum b_i^2)^2/(sum b_i^4). A fit can be precise on those few particles and
inaccurate where future filtering/smoothing actually needs it. The new
experiment records this concentration and evaluates independent predictive and
smoothing clouds. The smoother is itself checked against independent Gaussian
conditioning using the exact future guide.

The current QR fit instead approximates log b with a diagonal quadratic.
It uses more of the cloud's shape information and avoids the density-amplitude
escape, but changes the objective. The exact Gaussian future guide generally
has off-diagonal precision: even at t=T-1 without the tiny constant,
H=I+(1/2)A^T A for the paper model, which is not diagonal. Diagonal fitting is
therefore an approximation explicitly made in the paper, not a coding error
that can be eliminated without changing its fitting family.

The public unverified R repository also differs from the printed fit. Its
`iapf.R` lines217--235 uses J=||b-p/lambda*||^2. With rho^2=(p^T b)^2 /
(||p||^2||b||^2), J=||b||^2(1/rho^2-1). Our relative-shape objective is
1-rho^2. They have the same ordering of shapes in exact arithmetic, but neither
is the printed L*, and their numerical gradients/conditioning differ. This
repository is useful comparative evidence, not an original-author oracle.

## What positive floors and stopping histories can establish

For a fixed learned Gaussian-plus-constant guide, c_t<=psi_t(x)<=C_t with
0<c_t<C_t<infinity. Gaussian observation densities also have finite maxima G_t.
The transition integral of the next guide is at most C_(t+1), so the incremental
weights are at most G_t*C_(t+1)/c_t, with the finite initial integral included
at t=1. At fixed finite horizon this bounds the ideal-arithmetic likelihood
estimate and gives a finite conditional second moment. The bound can be
enormous and is not uniform over randomly learned guides. It does not establish
small variance, a useful unconditional variance bound, or floating-point safety.

The paper explicitly defines resampling times as t=1,...,T-1 whose weights
have ESS<=kappa*N (v2 lines687--723). R records the corresponding action when
propagating to t+1 and counts each once. This definition is checked and is not
an unresolved algorithmic gap. The numerical table comparison uses the fresh
final filter's count; the unavailable authors' reporting code cannot establish
its implementation identity.

Our Gaussian-limit eigenvalue check tests the zero-floor approximation. It is
a conservative quality screen, not proof that the actual positive-floor filter
has infinite variance if that check fails. The chosen chi-square floor rule and
power8 remain disclosed reconstruction choices; the manuscript specifies only
positivity, not this formula.

The printed CV uses estimates from k+1 different stages of learning. Write each
estimate as Z_l=Z+epsilon_l, with the conditional distribution of epsilon_l
depending on the learned guide and particle count. The empirical CV across
these stages is not a direct estimate of the conditional variance of the final
guide. A poor early guide can keep the CV above .5 even after later guides are
adequate. Conversely, final-guide quality may itself rely on learning after
doubling. This is why the new audit compares independently repeated APFs using
both the guide just before doubling and the final guide. Changing the window
from six estimates to five would change the printed method and remains an
explicit extension, not a claimed correction.

Conditional on all learning history, a fixed guide and N with a fresh final
particle run retain the ideal APF's likelihood unbiasedness. Taking expectation
over learning preserves it. This argument depends on the checked weight and
resampling identities and independent fresh randomness; it does not make a
32-run bootstrap interval a proof of unbiasedness or of adequate tail coverage.

## Remaining gap register and closure tests

| Gap | Current classification | How to close or discriminate it |
|---|---|---|
| R Gaussian proposals, weighting, resampling and fresh return | Checked mechanics in the tested scope | Preserve identity/mutation tests; extend fixtures when a new model or route is introduced |
| Exact Eq15 fitting versus QR | Real method difference | Keep names explicit; compare constrained density fitting and relative/log fitting on heldout states and full filters |
| Unspecified optimizer, starts, scale, bounds and tolerances | Source gap plus numerical sensitivity | Record every choice, test one factor at a time; author implementation/settings needed for identity claim |
| Fit quality away from training particles | Open scientific gap | Predictive/smoothing diagnostics plus fresh end-to-end likelihood and heuristic comparisons; training loss cannot promote |
| Positive floor formula/strength | Source gap, working hypothesis | Preserve floor sensitivity and tail diagnostics; compare fixed guides before changing a floor policy |
| Controller convention and learning transients | Partly specified source rule, causal question | Reproduce histories, capture pre-doubling guide, independent conditional repeats; any changed controller requires new full validation |
| d20 mixed conditional performance | Open empirical question | Two new untouched datasets, retain both successes and losses with paired uncertainty |
| Missing d40 at delayed setting | Coverage gap addressed in this campaign | Fresh32-repeat d40 run under exactly frozen settings |
| Authors' realized data/RNG and table reporting code | Unresolved provenance | Obtain original data/settings if available; report simulated-seed results and the checked paper resampling definition |
| Original tables at1000 repetitions per dimension | Not executed at published scale | Freeze a fully specified reference first, then run all five dimensions with1000 repeats and appropriate uncertainty; do not keep tuning to table entries |
| Equal-cost efficiency | Not established | Count all learning/final work and calibrate BPF/FA cost on the same hardware; paper particle counts alone do not establish matched cost |
| Alpha sensitivity and later studies | Not implemented/tested by this bounded stage | Next first-study extension: fixed data, alpha=.30:.02:.50, paper's BPF50000; PMMH/SV require separate study implementations |
| Published final text and original author code | Not recovered | Preserve available primary-source anchors and explicit source boundary; lack of code is not a computation blocker |
| R-to-TensorFlow full-filter parity | Still absent | Match model, fit family, floor, controller, resampling and supplied randomness before whole-filter comparisons |

The actual TF `execute_iapf` still rejects d!=1 or observation dimension!=1
(`iapf_adapter.py` lines45--60). Its separate log-quadratic routine includes
cross terms (`fitted_twist_tf.py` lines161--186), whereas the R fit is diagonal.
Earlier eight-component FP64 comparisons passed, but this consumer limitation
prevents calling them a multidimensional full-filter replication. The current
R study does not close GPU, differentiation, HMC or nonlinear validation.

## Literature audit scope

Decision: sufficient primary-source coverage for the above narrow implementation
audit; insufficient for an exhaustive literature review or final-version identity.
Metadata date: prior source reconciliation2026-09-20/21, with this turn's
unsuccessful online recheck recorded above. Seed paper: GJL2017 and its two
available manuscript versions. Source support: technical methods, experiment
definitions and appendix inspected directly. Citation/venue counts do not
support any numerical claim. Backward checking covers the argument used in the
paper's own appendix; referenced general SMC books are not claimed freshly read.
The earlier forward search supplied implementation/correction leads only; no
new technical claim rests on an unread citing paper. Quarantined authority:
unverified public R code is not original-author code. Main omission risks:
unrecovered publisher supplement, original optimizer conventions and original
data. Numerical evidence and remaining actions are in result.md.
