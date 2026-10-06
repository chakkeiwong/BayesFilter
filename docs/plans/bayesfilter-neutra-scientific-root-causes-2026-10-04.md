# Why the corrected NeuTra mixture fits failed

The forward fits learned densities with excessive tails and bridges between
components. The automatic reverse-KL finish then reduced its own objective
partly by reducing the probability of a poorly fitted component. The training
controller switched objectives after a prescribed update count, without evidence
that forward fitting had settled and without protecting its coverage. These
findings explain the current rejected checkpoints. They do not establish that
IAF cannot represent the targets, or that FAB, SMC, AFT or CRAFT failed: the
corrected campaign stopped at an exact-sample student prerequisite before those
teachers ran.

This report traces the r3 execution source, all four forward/final checkpoint
pairs, their training histories and the actual objectives. A new reference
diagnostic evaluated 32,768 common Gaussian draws per target, checked numerical
derivatives and used analytic mixture identities. Its
[plan](bayesfilter-neutra-scientific-root-cause-plan-2026-10-04.md),
[numerical evidence](artifacts/neutra-scientific-2026-10-04/root-cause-r2/result.json),
[manifest](artifacts/neutra-scientific-2026-10-04/root-cause-r2/manifest.json), and
[density figure](artifacts/neutra-scientific-2026-10-04/root-cause-r2/saved-map-densities.png)
preserve the investigation. All uncertainty below conditions on these fixed
maps; it is not uncertainty across training initializations.

## What the executed code computes

Write the normalized target as p(x), the standard Gaussian base as phi(z), and
the learned invertible map as x=T_theta(z). The implemented proposal density is

\[
q_\theta(x)=\frac{\phi(T_\theta^{-1}(x))}
 {|\det J_{T_\theta}(T_\theta^{-1}(x))|}.
\]

Each IAF stage computes y_j=exp(s_j(x_<j))x_j+m_j(x_<j). Its autoregressive
Jacobian is triangular, with diagonal exp(s_j), so its log absolute determinant
is sum_j s_j. Coordinate reversals have unit absolute determinant. The masks,
coordinatewise inverse, log-determinant sign and stage composition implement
these equations. The scale is b_j+2 tanh(h_j/2); the free bias is outside the
cap, as in the inspected author option in
`.localresources/q20-flow-training-literature-20260923/code/neutra-utils.py:739`.
The exclusive first block mask and inclusive subsequent masks correspond to
that source's masked network. This is a check of those operations, not a new
claim to reproduce every setting or result of the paper.

The four exact iid teacher banks contain 4,096 points in total. In
`bayesfilter/testing/neutra_scientific_campaign.py:322`, their normalized
log weights are pooled with equal weight per bank. At line 191, minibatch
indices are sampled from those weights; their minibatch weights are then
uniform. Consequently the expected minibatch gradient is precisely the gradient
of the empirical forward cross entropy

\[
\widehat L_F(\theta)=-\sum_i \bar w_i\log q_\theta(x_i).
\]

There is no double weighting. `neutra_weighted_training.py:498` differentiates
this expression, including the parameter dependence of the inverse. For iid
exact samples, it estimates D_KL(p||q_theta) up to p's entropy. The empirical
teacher has sampling error, but the samples cover the target components. For
the three-component target their responsibility masses were
(0.5247, 0.2475, 0.2278), versus true weights (0.5225, 0.2496, 0.2279).

At `neutra_scientific_campaign.py:351`, the forward endpoint is saved and
assessed. Its pass/fail result does not govern the next training step. A fresh
Adam optimizer is created and 256 reverse-KL updates always follow. The
standard estimator in `neutra_transport_core.py:366` differentiates

\[
L_R(\theta)=E_{z\sim\phi}
 [-\log p(T_\theta(z))-\log|\det J_{T_\theta}(z)|],
\]

which is D_KL(q_theta||p) up to the fixed base entropy. Its derivative is

\[
\nabla_\theta L_R=-E_\phi\left[
 (\partial_\theta T_\theta)^T\nabla_x\log p(T_\theta(z))
 +\nabla_\theta\log|\det J_{T_\theta}(z)|\right].
\]

The stopped-score attachment in the code implements this derivative. It needs
the target score, not a target Hessian. The diagnostic found exact agreement
between that attached gradient and direct differentiation through the target
on the checked batches. This campaign uses the standard estimator; it does not
use the configured optional Roeder/Vaitl path estimator.

## Numerical defects checked and not found

All eight checkpoints were reloaded using their target/hash-checked loader.
Checks included ordinary points and the sixteen largest score-residual points
among the new draws. The key live files match the executed source snapshot.

| Check | Largest observed absolute error across the eight maps |
|---|---:|
| Latent forward/inverse round trip | 2.88e-14 |
| Physical round trip on all 4,096 heldout points | 6.22e-15 |
| Reported log determinant versus full 2D Jacobian | 1.20e-14 |
| Full Jacobian versus finite differences, step 1e-5 | 4.55e-7 |
| Target score versus independent analytic mixture score | 3.55e-15 |
| Manual transformed score versus autodiff | 1.14e-13 |
| Inverse density versus forward change of variables | 1.60e-14 |
| Attached standard RKL gradient versus direct target differentiation | 0 |
| Forward-KL directional parameter derivative, step 1e-6 | 3.46e-9 |
| Reverse-KL directional parameter derivative, step 1e-6 | 1.29e-7 |

The larger finite-difference steps had larger truncation errors; all three
steps, 1e-4, 1e-5 and 1e-6, are retained in the evidence. These are local
FP64 CPU reference checks, not an exhaustive proof about every parameter,
dimension or GPU compiler path. They reproduce the saved heldout forward-KL
values and find no density/gradient error remotely large enough to explain the
observed distribution errors.

The training records also show zero clipped updates. In each saved 1,000-point
probe, zero sampled coordinate/stage scales had conditional-cap slope below
0.1. Thus the former almost-always-clipped optimizer and strongly saturated
scale mechanism are not demonstrated causes of these fits. Saturation at
unsampled points remains unchecked.

## The forward maps put too much probability in the wrong places

For these unwarped targets,

\[
p(x)=\sum_{k=1}^K w_k\varphi_k(x),\qquad
\rho_k(x)=\frac{w_k\varphi_k(x)}{p(x)},\qquad
y_k=\frac{x-\mu_k}{\sqrt{v_k}}.
\]

Multiplying by p cancels the responsibility denominator. Therefore, exactly,

\[
E_p\rho_k=w_k,\quad E_p[\rho_k y_k]=0,\quad
E_p[\rho_k y_k y_k^T]=w_k I.
\]

This identity tests local component shapes as well as their weights; counting
which mode is visited is insufficient. For example, in the width-64
three-component forward fit, the responsibility-conditioned second moment of
the second component's first standardized coordinate is 14.34 instead of 1.
Its mode is represented by a long broad extension, not its intended Gaussian
shape. The density plot shows that extension and the bridges between modes.

An especially simple check avoids estimated reference means altogether. Let
B be the region outside every component ellipse ||y_k||^2<=8. Conditional on
the generating component, being in B requires a chi-square_2 variable to
exceed 8. Thus

\[
p(B)\le \Pr(\chi^2_2>8)=e^{-4}=0.0183156.
\]

The responsibility-weighted radial-tail diagnostic has expectation exactly
e^-4, even when components overlap. The bound on the actual event B is
slightly weaker but easier to interpret.

| Target / width | q(B) after forward fitting | q(B) after RKL | True upper bound |
|---|---:|---:|---:|
| Two / 32 | 19.41% (SE 0.22 pp) | 12.86% (SE 0.18 pp) | 1.83% |
| Two / 64 | 8.39% (SE 0.15 pp) | 6.46% (SE 0.14 pp) | 1.83% |
| Three / 32 | 25.67% (SE 0.24 pp) | 20.84% (SE 0.22 pp) | 1.83% |
| Three / 64 | 21.62% (SE 0.23 pp) | 9.38% (SE 0.16 pp) | 1.83% |

These are 32,768-draw integration estimates for the fixed maps. The discrepancies
are much larger than their Monte Carlo errors. The event includes remote tails
as well as inter-mode regions; the table must not be described as measuring
only bridge probability. This failure persists when the formerly defective
reference-standard-error screen is removed entirely.

Forward KL is mathematically correct but a modest value is a weak promise of
coverage accuracy. For any normalized density r and 0<alpha<1, set
q=(1-alpha)p+alpha r. Since q >= (1-alpha)p pointwise,

\[
D_{\mathrm{KL}}(p\Vert q)\le-\log(1-\alpha).
\]

Thus a map can assign 20% of its probability to an inappropriate broad density
and still have forward KL at most 0.223. This is an exact illustrative bound,
not an assertion that the saved q has that mixture representation. At an
unconstrained global optimum forward KL still selects p. The problem is using
a finite fit's average likelihood as assurance that its local shape is right.

## Why reverse KL sacrifices an already represented mode

The mathematical trade-off can be decomposed exactly even for overlapping
components. Define augmented joint densities

\[
P(x,k)=p(x)\rho_k(x)=w_k\varphi_k(x),\qquad
Q(x,k)=q(x)\rho_k(x).
\]

Both use the same conditional distribution of k given x, so
D_KL(Q||P)=D_KL(q||p). Let a_k=E_q rho_k and
q_k(x)=q(x)rho_k(x)/a_k. Factoring the joint densities gives

\[
\begin{aligned}
D_{\mathrm{KL}}(q\Vert p)
 &=\sum_k\int q(x)\rho_k(x)
     \log\frac{a_k q_k(x)}{w_k\varphi_k(x)}\,dx\\
 &=\sum_k a_k\log\frac{a_k}{w_k}
   +\sum_k a_k D_{\mathrm{KL}}(q_k\Vert\varphi_k).
\end{aligned}
\]

The first term penalizes incorrect component weights. The second penalizes
incorrect conditional shapes, weighted by how much probability q itself puts
in each component. The measured width-64 three-component fit illustrates the
trade-off:

| Quantity | Forward endpoint | RKL endpoint |
|---|---:|---:|
| Probability a_2; true w_2=0.249591 | 0.21572 (SE 0.00224) | 0.02240 (SE 0.00079) |
| Conditional shape KL of that component | 5.641 (SE 0.165) | 3.987 (SE 0.153) |
| Weight KL, sum a log(a/w) | 0.00319 | 0.20984 |
| Weighted conditional-shape KL | 1.69401 | 0.41727 |
| Total reverse KL | 1.69721 (SE 0.03814) | 0.62711 (SE 0.00714) |
| Heldout forward KL | 0.42979 | 1.14216 |

Here component 2 means the second listed center, approximately (5.06,-1.28).
The paired responsibility change is -0.19333, with approximate 95% integration
interval [-0.19753,-0.18913]. The paired reverse-KL change is -1.07010,
SE 0.03867. Thus both changes are resolved for these fixed checkpoints.
The component's weighted shape contribution falls from about 1.217 to 0.0893.
The objective accepts a worse weight allocation because the reduction of shape
cost is much larger. The change-of-variables and gradient checks show that this
is compatible with correctly computed RKL optimization.

For intuition, partition space into disjoint regions A_k and hold each regional
conditional q_k fixed. Write W_k=p(A_k) and
d_k=D_KL(q(.|A_k)||p(.|A_k)). In this idealized unrestricted reweighting problem,

\[
\min_{a_k\ge0,\ \sum a_k=1}
\sum_k a_k\{\log(a_k/W_k)+d_k\}
\quad\Longrightarrow\quad
a_k^*=\frac{W_k e^{-d_k}}{\sum_j W_j e^{-d_j}}.
\]

The result follows by adding a Lagrange multiplier and setting
log(a_k/W_k)+1+d_k+lambda=0. A large shape cost exponentially suppresses the
region's optimal probability. This formula is explanatory: in an IAF the
weights and shapes move together, and the soft-responsibility conditionals in
the preceding exact decomposition cannot be varied independently. It is not
a prediction of the actual Adam trajectory. If the regional shapes are already
correct, d_k=0 and the formula gives a_k=W_k. RKL does not inevitably lose modes.

## Why the learner is still unfinished

`neutra_scientific_design.py:13` specifies two width/update pairs at the same
learning rate. The label `student_deep` increases width from 32 to 64; both
have exactly three IAF stages and two hidden layers per stage. This is not a
depth experiment. Width and update count change together, so their separate
effects are confounded. There is no learning-rate search, plateau-based
continuation, or forward-versus-RKL checkpoint selection in this controller.
The histories are measured, but only finiteness and clipping influence the
forward training loop's continuation.

At width 64 the saved heldout forward KL at 4,096 and 8,192 updates is
0.3011 -> 0.1072 for two modes and 0.5892 -> 0.4298 for three modes. These
points do not establish a converged optimum. At the final forward endpoints,
training/heldout empirical KL values are respectively 0.1005/0.1072 and
0.4111/0.4298. The heldout discrepancy is therefore not explained by a map
that fit the teacher extremely well but only failed on new points. The training
loss remains substantial too. The training/heldout gap is descriptive: the
training set selected the map, so a naive independent-sample significance test
of that gap would be invalid.

The finite optimization recipe failed to deliver accurate maps; why it has not
reached a better forward optimum is not fully resolved. Optimizer/learning-rate
choices, near-identity initialization, finite teacher data and the fixed
three-stage family remain candidates. More updates and wider networks were
changed together, one initialization per target was used, and no matched
stationarity/capacity experiment was run. Claiming either a proven capacity
barrier or guaranteed success from additional training would exceed the evidence.

Separated Gaussian mixtures also demand more deformation than their dimension
suggests. For the one-dimensional symmetric mixture
p(x)=[N(-a,sigma^2)+N(a,sigma^2)]/2, the exact increasing Gaussian-to-mixture
quantile map satisfies F_p(T(z))=Phi(z). Differentiation gives

\[
p(T(z))T'(z)=\phi(z),\qquad
T'(0)=\sigma\exp\{a^2/(2\sigma^2)\}.
\]

At a=5, sigma=1, the slope at the valley is about 268,337. An exact map
exists, but it can require a very sharp transition. This one-dimensional
example is not a lower bound for every two-dimensional transport. It shows why
low dimension alone does not make Gaussianization easy. The actual mixtures
have positive density everywhere; there is no disconnected-support impossibility
theorem here. The conditional cap does not impose a global Lipschitz bound,
because the shift networks and their derivatives remain free.

## Whitening is stronger than a small average KL

In latent coordinates the target is p_z(z)=p(T(z))|det J_T(z)|. The diagnostic
residual obeys the exact identity

\[
\nabla_z\log p_z(z)+z
 =\nabla_z\log\frac{p_z(z)}{\phi(z)}
 =\nabla_z\log\frac{p(T(z))}{q(T(z))}.
\]

Average log-density agreement does not bound this derivative. For example,
p_z(z)=phi(z) exp(epsilon sin(omega z))/Z has residual
epsilon omega cos(omega z). Symmetry makes E_phi sin(omega z)=0 and
Z=E_phi cosh(epsilon sin(omega z))<=cosh(|epsilon|). Since
log cosh(t)=integral_0^t tanh(u)du<=t^2/2 for t>=0,
D_KL(phi||p_z)=log Z<=epsilon^2/2, independently of omega. The score error can
nevertheless grow with omega. This supplies a mathematical counterexample to
the inference that small KL guarantees small force error.

In the new draws from the width-64 two-component final map, the median residual
norm is 0.351 while p95 is 40.72. About 83.6% of the observed squared residual
comes from the 6.46% of draws outside all component ellipses. These tail and
energy-concentration quantities are explanatory estimates, not calibrated HMC
criteria. They explain why a reassuring median misses the remaining geometry.
A fixed invertible map can still be used in a correctly adjusted HMC target;
poor whitening concerns efficiency and convergence evidence, not the
change-of-variables identity itself.

## Consequences for the next repair

The immediate defect to repair is the training protocol. Preserve forward and
RKL endpoints, continue a forward control under an assessed learning schedule,
and evaluate RKL as an optional continuation with explicit coverage/local-shape
regression checks. Rolling back RKL alone will not qualify these forward maps:
all four already fail the analytic shape/tail checks. Matched update-count,
learning-rate and capacity experiments are needed to distinguish unfinished
optimization from an inadequate family. Any architectural departure still
requires the repository's documented source-based justification; this report
does not change the canonical IAF.

A forward-likelihood anchor during later RKL is a plausible mechanism to test:
a positive forward term retains pressure from teacher points in a region even
when q nearly stops visiting it. Its coefficient and schedule need calibration;
no arbitrary weight is installed here. Better teacher sampling alone cannot
repair the demonstrated exact-teacher fitting failure.

| Decision | Primary criterion status | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Keep the eight maps rejected | Exact local-shape/tail reference disagrees | Existing distribution veto corroborated | Full distribution error beyond these features | Repair student training | Six teacher algorithms fail |
| Treat automatic RKL as a causal coverage regression in the wider three-mode run | Paired endpoint change resolved; KL decomposition checked | No density/gradient defect found locally | Variation across fit seeds; optimizer trajectory | Preserve endpoint; test guarded continuation | RKL always loses modes |
| Do not blame clipping or sampled cap saturation | Zero clipped updates, zero sampled slopes below 0.1 | No such mechanism observed | Unsampled tail scales | Examine optimization and expressivity | Global cap inactivity |
| Do not declare representational impossibility | No matched capacity/stationarity test | None establishing impossibility | Optimization versus finite family | Controlled calibration | More time necessarily succeeds |

| Inference status | Conclusion |
|---|---|
| Hard veto screen | Current maps fail their distribution screen; independent analytic tail bounds corroborate the error |
| Statistically supported ranking | None across recipes or algorithms; paired fixed-map coverage/RKL changes have integration uncertainty |
| Descriptive-only differences | Training curves, train/heldout gaps, extreme score quantiles, Jacobian conditions and cross-width comparisons |
| Default-readiness | None; no new training recipe, architecture or HMC admission |
| Next evidence needed | Matched optimization/capacity controls, multiple initializations, protected forward/RKL selection, then fresh generalization |

Post-run skeptical review: the strongest alternative explanation is that a
better optimization schedule within the same IAF family could fix the forward
shape error. The current data do not exclude it. A qualified forward control
that retains its mass and improves geometry through RKL would overturn a claim
that this continuation must fail. The weakest causal evidence is the attribution
of unfinished forward fitting between optimizer, initialization and capacity;
those factors were not separately varied. The exact KL decomposition and
analytic mixture identities do not depend on that unresolved attribution.

The first diagnostic attempt completed its numerical checks but failed while
importing the plotting library. The second used the already installed base
environment to render the same checkpoints and reused the completed numerical
evidence. No package change or new training occurred. Total diagnostic cost was
64.94 seconds of process wall time and 74.758258 CPU-core seconds, with no GPU
work. It is debited from the existing campaign allocation.

The final artifact audit passed after both debits, with no missing jobs,
changed checkpoints or inconsistent shared charges. Its record is
`root-cause-r2/campaign-artifact-audit.json`. Remaining allocation is
3,895.501185 GPU-process seconds and 8,660.105809 CPU-core seconds; the
scientific campaign remains terminal and under-calibrated.
