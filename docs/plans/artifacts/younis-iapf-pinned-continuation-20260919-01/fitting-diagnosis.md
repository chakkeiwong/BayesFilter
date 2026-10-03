# What the fresh-data failure establishes

The RTX5080 confirmation succeeds on the original fixed datasets. Fresh weak
datasets expose two separate limits: a poor fitted proposal on dataset 1900,
and residual score-estimation error that still exceeds UKF on dataset 1901
even with an interior fit. Widening the fitting box does not remove these
failures. None of these results invalidates Fisher's identity or establishes
that iAPF, KDM or LEDH cannot work.

## The inherited stopping rule cannot certify convergence

The executed adapter uses `k=1` and `tau=100`. Its controller compares the
sample coefficient of variation of the last two positive likelihood estimates
with tau, once at least two fitting updates have occurred. Write these two
estimates as a and b. Log rescaling in the implementation multiplies both by
the same constant and leaves their coefficient of variation unchanged.
Their mean is (a+b)/2 and sample standard deviation is |a-b|/sqrt(2), so

    CV(a,b) = sqrt(2) |a-b| / (a+b) <= sqrt(2).

The bound follows from |a-b| <= a+b. It also covers one estimate underflowing
to zero after log rescaling. Therefore `CV < 100` always passes when the
controller becomes eligible to stop. This configuration effectively performs
two fitting updates; the stopping test cannot distinguish a stable sequence
from a wildly varying one. In dataset 1900, the observed terminal CV is
0.43985 with the original bounds and 0.66128 with wider bounds; both return
`final` at iteration 2 with 16 particles.

This conclusion follows from the actual
[controller](../../../../bayesfilter/score_study/iapf_adapter.py:20) and is
reproduced by six independent analytic checks against that callable in
[stopping-diagnostic.json](stopping-diagnostic.json). The code correctly
executes the specified local configuration. Treating its stop as evidence of
fitting convergence would be wrong. The pre-run audit listed early stopping
as a risk but did not identify this exact range defect; terminal review did.

More generally, let n=k+1 and S=sum_i v_i for nonnegative values v_i.
Then CV^2=n/(n-1) [n sum_i v_i^2/S^2 - 1] <= n, since sum_i v_i^2 <= S^2.
Thus their sample CV has supremum sqrt(k+1).
An informative threshold must lie below that value. This is necessary, not
sufficient: two noisy likelihood evaluations still provide weak evidence of
fit adequacy even with a smaller threshold. Simply replacing 100 by an
arbitrary small number is not a reviewed fitting protocol.

## Local optimizer convergence and proposal accuracy are different

The optimizer fits a Gaussian shape on the small cloud supplied by the
particle pass. It standardizes points by that cloud's mean and SD, constrains
the standardized center and log SD, and stops on a projected-gradient check.
At a constrained optimum, that check can be zero even when the unconstrained
gradient points outside the box. The first fit for dataset 1900 records a
bound contact and projected gradient zero. This is convergence for the bounded
empirical optimization problem, not proof that its Gaussian approximates the
backward information function over the predictive distribution.

The independent reference evaluates shape on a refined predictive grid. Its
scale-invariant residual is

    R = 1 - <p,y>_w^2 / (<p,p>_w <y,y>_w),

where y is the backward target, p is the fitted Gaussian-plus-floor twist,
and w contains predictive reference weights. R=0 means proportional shapes;
R close to one means poor alignment in this weighted norm. The
[reference implementation](../../../../docs/benchmarks/diagnose_younis_iapf_curved.py:127)
computes it independently of the fitting-cloud objective.

| First-step quantity, dataset 1900 | Original box | Wider box |
|---|---:|---:|
| Fitted center | 2.79305 | 4.04559 |
| Fitted variance | 0.0332908 | 0.00317489 |
| Predictive shape residual R | 0.996807 | 0.888282 |
| Predictive average pointwise floor fraction | 0.997756 | 0.999999991 |
| Any offline bound contact | yes | yes |
| Final fitting particle count | 16 | 16 |

The twist has the form psi(x)=Normal(x;m,V)+c with c>0. The last fraction is
the predictive average of c/psi(x): it shows that the constant floor dominates
the fitted twist across much of the predictive distribution. It is **not** a
measured transition-mixture frequency and must not be reported as the fraction
of particles choosing the untwisted transition. A constant twist gives
q(x|previous)=f(x|previous), since the constant cancels between numerator and
normalizer; this explains why a floor-dominated twist can lose useful
observation adaptation, without asserting that these measured averages prove
an exact mixture frequency.

Widening leaves the three previously interior fitted proposals exactly
unchanged, so it passes the predeclared interior non-harm check. It moves
dataset 1900's narrow Gaussian farther out and leaves a bound contact. It
therefore fails as a complete repair for proposal fitting. A tiny cloud,
local optimization, the restricted Gaussian family and the floor remain
possible explanations; the present experiment does not separate them.

## Score accuracy still needs a direct comparison

On fresh dataset 1900, observed corrected-score MSE is 0.085319 versus UKF's
0.013889. The wider-box run gives 0.046264 and still loses to UKF. Those
cross-stage differences are descriptive: they use fresh final streams and
are not the primary paired comparison. Neither stage has a significant
new-versus-ancestor improvement on dataset 1900 under its declared interval.
The original-box correction also has higher observed MSE than raw Fisher
(0.073439); this is a descriptive warning about finite calibration, not a
statistically established ranking.

Dataset 1901 has an interior fit with small reference shape residuals, yet
corrected-score MSE is 0.001079 versus UKF's 0.000511. Thus fixing boundary
contacts is insufficient to clear the heuristic set. On this dataset the
observed squared error is mostly Monte Carlo variation; independently fitted
controls reduce it relative to ancestor controls but do not remove it. Mean
bias screens pass, which is compatible with substantial variance and does not
prove unbiasedness. Both curved datasets clear their observed heuristic
screens, under the same limited conditional interpretation.

The relevant score is s(theta)=grad log p_theta(y). The computed statistic is
a normalized particle estimate of the conditional expectation of the
complete-data score, followed by frozen zero-mean control subtraction. It is
approximately related to s(theta), with finite-N bias and Monte Carlo error;
it is not the derivative of the finite particle likelihood program. Its
controls preserve the raw estimator's expectation under the declared law;
they cannot fix a poor proposal or guarantee a variance reduction for every
independently estimated regression coefficient.

## Decision and next test

| Decision | Primary evidence | Veto status | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| GPU-scope repair complete | One RTX5080 and all score outputs recorded there; original-data 4/4 comparisons pass | No numerical or device veto | Fixed datasets/calibration | Preserve confirmation | Cross-device parity or broad superiority |
| Reject wider bounds as a complete fit repair | Same bound contact and large predictive shape error | Promotion veto remains | Cloud size, optimizer and Gaussian family not separated | Develop a larger-cloud, informative-stopping protocol on separate calibration data | Widening is useless for all purposes |
| Retain optional score controls diagnostically | Fresh and wider stages each pass 3/4 comparisons | Both weak cases still lose to UKF | Calibration and new-data uncertainty | Keep matched controls and UKF in the next downstream test | Default, HMC, KDM or LEDH admission |

The next smallest discriminating campaign should separate inadequate cloud
coverage from optimizer/family failure using a predeclared cloud-size ladder
and informative stopping/validation diagnostics. Freeze its settings on new
calibration observations, then use untouched validation observations and score
streams. Preserve these failed final datasets as failures; do not turn them
into tuning data. The current eight-fit allowance is exhausted, so that
protocol needs a new bounded execution allocation before fitting resumes.
