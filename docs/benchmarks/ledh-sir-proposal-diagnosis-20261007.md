# SIR failure: the flow proposal misses the conditional transition distribution

The existing SIR filter loses hundreds of log-likelihood units while ordinary
bootstrap sampling on the same incoming clouds remains close to exact Gaussian
integration. A controlled covariance substitution identifies a concrete cause:
each ancestor-centred flow uses the accumulated UKF covariance to move samples
whose conditional transition covariance is Q=I. During the early epidemic
growth, that flow moves
particles much farther than the conditional posterior requires. Importance
weights then collapse. The earlier tuning reduced some measures of score
dispersion but did not repair this statistical failure.

## What was executed

The diagnostic replays the actual analytical-score endpoint at N=1008, T=10,
FP64 GPU/XLA, TF32 off, with the saved observations and unchanged guarded
pairwise/marginal controls. The two original design seeds are 261006101 and
261006102. Both likelihoods and first score coordinates reproduce their saved
values exactly. Reconstructing each affine proposal through the shared flow
helper reproduces its children exactly. This verifies the active call chain,
not merely a separate implementation of the intended equations.

The baseline call chain is `sqmc_campaign_tf._kernel` to
`canonical_value_and_analytical_score`. In `ledh_canonical_score_tf.py`, lines
372–398 obtain the UKF prediction but sample `anchors + chol(Q) * noise`;
lines 498–524 supply `predicted_covs` to the flow and evaluate the actual
Gaussian mixture densities. `ledh_marginal_weights_tf.py`, lines 99–130,
uses proposal covariance B Q B', not B P B'. The density correction therefore
does account for the actual affine proposal covariance. The observed failure
is bad proposal overlap, not evidence of a missing P-to-Q density correction.

The SIR adapter has Q=I_18, H selecting the nine infectious coordinates, and
R=100 I_9 at the tested parameter. The observations are linear and Gaussian.
For each actual incoming cloud, the exact predictive integral is therefore
available. It is a reference conditional on that finite cloud, not an oracle
for the complete model likelihood.

## Mathematics of the mismatch

Conditioning on ancestor j gives

\[
 x\mid j\sim N(m_j,Q),\qquad m_j=F_\theta(x_{t-1}^j),\qquad
 y\mid x\sim N(Hx,R).
\]

Completing the square yields the exact conditional posterior

\[
 C=(Q^{-1}+H^TR^{-1}H)^{-1},\qquad
 \mu_j=m_j+QH^T(HQH^T+R)^{-1}(y-Hm_j).
\]

The incoming empirical predictive density integrates exactly to

\[
 Z_t^{\rm cloud}=\sum_j\omega_j
 N(y;Hm_j,HQH^T+R).
\]

For this SIR parameter, infectious-coordinate variance is 100/101 in C and
the observation gain is 1/101. Susceptible coordinates have conditional
variance 1 and no conditional observation shift. Correlation through the
nonlinear dynamics remains represented by the distribution of the ancestor
means; conditioning on an ancestor fixes that mean.

The current flow instead uses each particle's UKF predicted covariance P_j.
The difference is transparent in one dimension. An exact continuous Gaussian
flow designed for prior variance P maps its center to
m + P/(P+R) (y-m) and contracts centered noise by sqrt(R/(P+R)). If the noise
entering that map actually has variance Q, its resulting proposal is

\[
 q_P=N\!\left(m+\frac{P}{P+R}(y-m),\frac{QR}{P+R}\right),
 \qquad
 p(x\mid y,j)=N\!\left(m+\frac{Q}{Q+R}(y-m),\frac{QR}{Q+R}\right).
\]

These distributions agree when P=Q. With Q=1, R=100 and illustrative P=37.8,
the respective gains are approximately 0.274 and 0.00990. The proposal shifts
about 28 times as far toward the observation. This scalar example explains the
mechanism; the measured diagnostic uses the full matrices and the actual eight
discrete flow steps, not this continuous approximation.

Correct importance weighting retains the identity
E_q[g(X) f(X)/q(X)] = integral g(x) f(x) dx for the conditional cloud.
It does not guarantee usable error from 1008 samples. If the flow sends nearly
all samples away from the relevant density, the rare samples carrying that
integral are absent. Summing all ancestors in the numerator and denominator
cannot create those missing samples. The subsequent deterministic reset also
does not preserve unbiasedness of a full particle-filter likelihood estimator;
no full-filter unbiasedness claim is made here.

## Actual local results

For seed 261006101, on the SAME incoming clouds and observations:

| Observation | Exact cloud log increment | Current marginal LEDH | Ordinary bootstrap | Same flow with covariance Q |
|---:|---:|---:|---:|---:|
| 3 | -36.304530 | -44.127002 | -36.343991 | -36.304549 |
| 4 | -36.989083 | -178.233218 | -36.999022 | -36.989597 |
| 5 | -35.976224 | -124.196625 | -35.975260 | -35.976769 |
| 6 | -34.811109 | -67.032637 | -34.807559 | -34.814446 |

At observation 4 the current flow ESS is 1.02175, bootstrap ESS is 855.54924,
and the covariance-Q flow ESS is 1007.70869. Mean infectious UKF variance is
37.82020; mean actual proposal infectious variance is 0.73307, compared with
exact conditional variance 0.99010. RMS displacement of proposal means from
their exact conditional posterior means is 4.65833 over all 18 coordinates.
The minimum child distance to ANY transition-component center is 295.009 in
squared Q-Mahalanobis units. This checks coverage across ancestors, not only
the child's own generating ancestor.

The controlled shadow changes only the covariance passed to the shared flow.
It preserves the incoming cloud, transition samples, observation, eight steps,
and the same marginal-density helper. It is not propagated into the next time
step. Its mean/covariance match to the exact conditional Gaussian is also
recorded, so a favorable likelihood value alone is not the diagnostic.

| Design seed | Sum of current flow log errors | Sum of bootstrap log errors | Sum of covariance-Q shadow log errors |
|---:|---:|---:|---:|
| 261006101 | -279.297867 | -0.064144 | +0.000196 |
| 261006102 | -158.885777 | +0.020976 | +0.003450 |

Each error subtracts the exact integral for its own actual incoming cloud;
these sums are not likelihood estimates from a repaired recursive filter.
The second seed reproduces the same mechanism. These two designs support
localization and a repair hypothesis, not a statistically established ranking
of full filters.

## Why SIR exposes it, and why the score tuning did not repair it

The model's early infectious growth term is (kappa S - nu) I. Initial
S=487,...,495, kappa=0.1 and nu=18 give positive growth coefficients near 31
before migration and susceptible depletion. The measured mean UKF infectious
variance grows from 4.109 to 12.038, 26.151 and 37.820 over the first four
observations. At the same time, each conditional transition retains variance 1.
Nine observed coordinates make displacement and weight concentration compound.
The saved low-dimensional successes do not protect against this regime.
Bootstrap with the same 1008 samples remaining accurate locally demonstrates
that dimensionality alone is not an adequate explanation.

After weights concentrate on one or two particles, the skew/kurtosis repair
fits moments of that damaged weighted cloud. It cannot reconstruct posterior
mass that was never sampled. A cap on reset displacements does not constrain
this earlier flow proposal. Also, the carried UKF covariances are transported
separately from the reset particle locations; they are not automatically made
equal to the covariance of the actual conditional transition samples.

The analytical score differentiates the finite likelihood approximation.
Previously saved small-step finite-difference checks verify that derivative,
not agreement with the derivative of the true likelihood. With
log Z_hat = log sum exp(a_i), its score is sum softmax(a)_i D a_i. When
the normalized weights concentrate, a single particle's sensitivity dominates.
Reducing the observed variance of that derivative can leave its bias intact.
Thus the earlier lowest-variance setting was not an accurate SIR score solution.

## Clarification: this does not establish a UKF covariance error

The user's follow-up correctly distinguishes accuracy of the UKF update from
accuracy of the resulting particle proposal. Sigma points matching a prior's
mean and covariance reproduce affine propagation of those moments exactly.
For a Gaussian prior and linear Gaussian observations, the Kalman conditioning
formulas are exact (apart from numerical regularization). SIR observations are
linear, but its dynamical prediction is nonlinear. A second-order approximation
statement does not establish exact nonlinear filtering, nor does it identify
which distribution a covariance describes.

The checked active UKF update in
`ledh_canonical_score_stages_tf.py::quadrature_update_with_parameter_tangent`
forms the usual gain and P_post = P_pred - K S K'. In the active score endpoint,
this update occurs after the flow and likelihood-weight calculation; its means
are discarded and its covariance is carried to the next step. No experiment
here shows that this UKF covariance update is algebraically wrong. Also,
P_j need not equal the covariance of the whole empirical particle mixture:
it is an auxiliary Gaussian covariance propagated around particle j.

Two different conditionings explain the apparent contradiction. UKF prediction
propagates an uncertain previous state with its carried covariance, giving
P_j approximately Cov[F(X_previous)] + Q under that local Gaussian. The actual
transition draw conditions on the fixed ancestor x_previous^j, giving covariance
Q exactly. A covariance can be appropriate for the first distribution and
inappropriate as a matched Gaussian-flow design for the second. The observed
proposal failure does not prove that the chosen P_j is itself a bad estimate
of the first covariance, or that the implementation contradicts the original
particle-flow paper.

In particular, the full empirical filtering covariance is NOT C=100/101 on
the infectious coordinates. Conditional on the incoming discrete mixture,
the exact updated mixture weights are

\[
 \alpha_j=\frac{\omega_j N(y;Hm_j,HQH^T+R)}{Z_t^{\rm cloud}},\qquad
 \bar\mu=\sum_j\alpha_j\mu_j,
\]

and the law of total covariance gives

\[
 \operatorname{Cov}(X\mid y,\text{incoming cloud})
 =C+\sum_j\alpha_j(\mu_j-\bar\mu)(\mu_j-\bar\mu)^T.
\]

Using Q for each conditional proposal therefore does not discard all uncertainty
from the previous observations; that uncertainty remains in the distribution
and posterior weights of the component means. Conversely, simply substituting
Q for every carried UKF covariance would be a different, unjustified change.

Finally, P_j != Q is not by itself a validity violation: an invertible map with
correct density correction may be used as an importance proposal. What the
two-design diagnostic establishes is a severe proposal-efficiency failure in
this SIR regime, causally sensitive to covariance choice. It does not establish
that UKF must be replaced, that Q is universally optimal, or that the original
published algorithm is mathematically wrong. No additional numerical campaign
was run for this clarification; it follows from the checked call chain and the
Gaussian conditioning and total-covariance identities above.

## AR(1) clarification: preserve the whole cloud's uncertainty

For X_1=phi X_0+eta, Var(eta)=Q, initialize the stationary prior with
P_0=Q/(1-phi^2). Then the full predictive variance is
P_minus=phi^2 P_0+Q=P_0. With phi=0.9 and Q=1, the between-ancestor variance
is 81/19 and the conditional transition variance is 1, for total 100/19.
Replacing the full cloud variance by Q would therefore be wrong. The stationary
prior variance is not generally the steady-state covariance after conditioning
on observations. A measurement with R=100 gives posterior variance 5 in this
example, not 1. Matching the full cloud to the Kalman filtered covariance is
appropriate in the linear Gaussian model; this diagnostic does not reject that
instruction or show an error in the reset moment target.

The distinction missing from the initial explanation is the flow centre.
One common Gaussian flow, centred on the full predictive mean, can legitimately
use P_minus even when each particle was generated by a conditional draw with
noise covariance Q. P_minus != Q is not sufficient to diagnose its failure.
The inspected implementation instead passes each transition anchor as that
particle's prior mean (`ledh_canonical_score_tf.py`, lines 498-505); the mean
enters the affine drift at lines 1208-1218.

The difference can be derived exactly in the scalar, zero-prior-mean example.
Write m_j=phi x_0^j, V=Var(m_j), P_minus=V+Q, K=P_minus/(P_minus+R), and
B=sqrt(1-K). These formulas describe the ideal continuous Gaussian flow,
not a new measurement of the eight-step implementation. A common whole-cloud
map is T(x)=K y+B x. It has covariance B^2(V+Q)=P_plus. An ancestor-centred
map with the same P_minus is T_j(x)=(1-K)m_j+K y+B(x-m_j). Applied to
x=m_j+eta, it produces covariance

\[
 \operatorname{Var}(T_J(X))=(1-K)^2V+(1-K)Q
 =P_+ - K(1-K)V.
\]

The latter contracts the dispersion between ancestor means too strongly.
For the AR(1), Q=1, R=100 example, its unweighted proposal variance is
4.7975 versus the desired posterior variance 5. These are exact algebraic
values, verified with rational arithmetic; no new filtering experiment was
run. Importance weights can correct this proposal in the population limit,
but finite-sample efficiency is a separate question. This illustration does
not claim that per-particle SIR P_j equals its full-cloud covariance.

Two consistent constructions therefore remain distinct: a common flow using
the full predictive distribution's mean and covariance, or conditional flows
using each ancestor mean and Q with the corresponding ancestor/mixture weight
update. The tested Q substitution concerns the latter. It neither licenses
shrinking the whole cloud to Q nor proves that all uses of UKF P are wrong.

## Decision and inference status

| Decision | Primary criterion | Veto diagnostics | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Current proposal fails on inspected SIR steps | Exact conditional integration errors reach -141 log units; bootstrap is close | Finite validity passes, but near-unit ESS and reference discrepancy reject usable accuracy | Two designs; already altered incoming clouds | Repair proposal construction | Wrong TT reference or impossible SIR problem |
| Covariance consistency is a supported repair hypothesis | Controlled substitution removes the large local integration errors in both designs | Replay and affine parity exact; shadow densities finite | No recursive repaired run or repaired total score yet | Derive and implement an explicit proposal policy; validate T=50 and protected models | Default promotion or universal superiority of Q |

| Inference category | Status |
|---|---|
| Hard validity veto screen | No nonfinite result, broken replay, or affine mismatch observed. Large accuracy failures reject the existing SIR results. |
| Statistically supported ranking | None for full filters. |
| Descriptive-only differences | Two-design per-step errors, ESS and covariance/mean differences above. |
| Default readiness | No runtime/default change made. |
| Next evidence needed | Full recursive T=50 likelihood and all three analytical scores against the reference, independent designs, and LGSSM/KSC/PP regressions under separately scoped validation. |

Post-run red-team: another failure in reset or accumulated particle coverage may
remain after repairing the proposal. The exact integral here conditions on the
existing cloud, so a shadow success cannot establish correct recursive filtering.
The weakest evidence is the absence of that recursive repair experiment. The
strongest alternative explanation, a flow-step or mixture-density defect rather
than covariance choice, is weakened by changing only the covariance while
keeping those operations identical. A repaired full run that still fails would
trigger the next localization phase, not invalidate the checked local finding.

## Reproducibility

Plan: `docs/plans/ledh-sir-proposal-diagnosis-20261007.md`.
Program: `docs/benchmarks/diagnose_ledh_sir_proposal.py`.
Evidence: `docs/plans/artifacts/ledh-sir-proposal-diagnosis-20261007-01/`.
Attempt 01 localizes the baseline; attempts 02 and 03 add the controlled shadow.
All three finish successfully, in 35.12, 43.09 and 35.91 seconds respectively
(114.12 seconds total). Manifests record command, Git commit, source hashes,
saved data identity, seeds, TensorFlow environment, GPU memory-growth policy,
XLA/TF32 settings, allocator measurements and artifact paths. Full logs and
unrounded step values are retained. No core implementation or default changed.
