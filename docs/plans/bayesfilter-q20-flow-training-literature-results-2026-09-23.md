# What the flow literature says about the q20 training failure

Yes: the literature contains closely related failures, including affine
autoregressive flows fitting nonlinear distributions poorly, expressive flows
remaining difficult to optimize, and reverse-KL gradients remaining noisy even
at an exact fit. It also offers concrete remedies. For q20, the most relevant
are **separating global scale from bounded conditional scale, exposing a direct
nonlinear scalar transformation, and testing a flow-correct path-gradient
estimator**. These address different mechanisms and should be diagnosed
separately before combining them.

This survey inspected twelve papers, their relevant mathematical and
experimental sections, and available author implementations under the
[survey plan](bayesfilter-q20-flow-training-literature-plan-2026-09-23.md).
The [source ledger](artifacts/q20-flow-training-literature-2026-09-23/source-ledger.json)
records paper versions, local copies, checksums, code snapshots, and reading
anchors. This is a targeted survey, not an exhaustive systematic review. It
performed no target evaluations, training, or HMC runs.

## The failure we need to explain

The [saved-map mathematical audit](bayesfilter-q20-training-math-audit-results-2026-09-23.md)
found that affine functions explain 99.866–99.999% of the depth-four map's
coordinate variation on its saved bank. Meanwhile, latent coordinate 2 accounts
for 89.02% of its squared Gaussian-score residual; a constant-plus-linear fit
explains only 0.573% of that coordinate's residual energy. That direction
principally controls the physical observation weight. This is evidence of
weak learned nonlinear deformation where we tested, not proof of global
affineness, target multimodality, or an intrinsically impossible architecture.

Routine clipping was already removed in the intervention tranches: 0.68% of
updates were clipped in the clipping repair and none in depth four. A stage's
bounded log scale remains near its lower limit, and the new third-stage tanh
features are small. The inspected reverse-KL scalar and supplied first
parameter derivative have the correct form. Consequently, neither ongoing
clipping nor a missing first-order target derivative is an adequate explanation
by itself. The literature makes parameterization and gradient noise concrete
suspects, without establishing which dominates here.

Our target throughout is the four-parameter q20/T30 **UKF approximate posterior**.
A repair must preserve that density and the full change-of-variables factor.
The scientific aim remains stable estimation using plain NeuTra HMC or the
tempered NeuTra ensemble, with identity latent mass.

## 1. Affine scalar transformations can be difficult to train into nonlinear shapes

Huang et al., *Neural Autoregressive Flows* [1], explicitly identify susceptibility
of conditional affine IAF to poor local minima and failure to capture some
multimodal shapes. Their energy-fitting experiments in §6.1.2 and Figures 1/7
are particularly relevant: they replace the scalar affine transformation with
a directly nonlinear monotone function. Their toy results are evidence that
this mechanism can matter, not evidence that q20 is multimodal or that their
architecture is already tuned for it. The author sine-wave example also uses
annealing, so that example does not isolate architecture alone.

Our stage has the form

\[
y_j=x_j\exp s_j(x_{<j})+m_j(x_{<j}).
\]

At fixed predecessors it is affine in its own coordinate. Nonlinear marginal
shape must emerge through conditional dependence and composition. Reversals
and multiple stages can make the complete map nonlinear, but they do not
ensure that optimization discovers that deformation.

The deep sigmoid flow in [1, §3, Eq. (8)] instead uses

\[
\tau(x)=\operatorname{logit}\!\left[
 \sum_{k=1}^{K}w_k\sigma(a_kx+b_k)\right],
\qquad a_k>0,\quad w_k>0,\quad\sum_k w_k=1.
\]

Here \(\sigma\) is the logistic function. The positive slopes and weights make
the scalar map strictly increasing. With finite parameters its limits are
minus and plus infinity. It can alter scalar shape directly; its parameters
can optionally depend on earlier coordinates. With one component it reduces
exactly to \(a_1x+b_1\), so at least two nonidentical components are needed for
this type of nonlinear deformation. That lower bound is algebraic, not a
recommendation for a sufficient capacity.

There is an important initialization qualification. At the exact redundant
identity, \(a_k=1,b_k=0\) for all components, the scalar derivatives are

\[
\frac{\partial\tau}{\partial a_k}=w_kx,\qquad
\frac{\partial\tau}{\partial b_k}=w_k,\qquad
\frac{\partial\tau}{\partial\eta_k}=0,
\quad w=\operatorname{softmax}(\eta).
\]

This is a local derivation: use
\(\sigma'(x)=\sigma(x)[1-\sigma(x)]\) and differentiate the outer logit.
The initial scalar parameter directions are only affine; changing identical
mixture weights does nothing. Conditional coefficients can still create joint
dependence, but a new unconditional scalar map needs component asymmetry to
expose first-order shape directions. The paper's near-identity initialization
uses small random weights, not an instruction to initialize every component
identically and expect immediate nonlinear learning. We should measure the
resulting shape directions before an expensive run.

The implementation deserves separate scrutiny. The linked author package
[`naf-flows.py:281`](../../.localresources/q20-flow-training-literature-20260923/code/naf-flows.py)
uses positive slopes, softmax weights, and log-domain Jacobian terms, but also
replaces the sigmoid mixture \(S\) by
\((1-\delta)S+\delta/2\), with \(\delta=10^{-6}\) in its
[`nn.py:26`](../../.localresources/q20-flow-training-literature-20260923/code/naf-nn.py).
That scalar output is bounded by finite logit limits. It is therefore different
from the full-range mathematical map and must not be copied literally into a
NeuTra map required to cover the original unconstrained posterior.

A full-range evaluation can instead use the following algebraic identities,
consistent with the paper's log-domain approach in Appendix C:

\[
\begin{aligned}
A&=\operatorname{LSE}_k[\log w_k+\log\sigma(a_kx+b_k)],\\
B&=\operatorname{LSE}_k[\log w_k+\log\sigma(-a_kx-b_k)],\\
C&=\operatorname{LSE}_k[\log w_k+\log a_k+
          \log\sigma(a_kx+b_k)+\log\sigma(-a_kx-b_k)],\\
\tau(x)&=A-B,\qquad \log\tau'(x)=C-A-B.
\end{aligned}
\]

Because \(B=\log(1-S)\), this avoids subtracting a nearly-one mixture from
one or clipping its range. These identities are a proposed local numerical
implementation, not an already checked TensorFlow kernel. Tail arithmetic,
strict monotonicity, inverse accuracy, and score derivatives still need tests.
The ideal scalar function is smooth, but generally has no closed-form inverse.

Durkan et al., *Neural Spline Flows* [2], provide another direct scalar option:
monotone rational-quadratic splines, with positive widths, heights, and knot
derivatives, and a quadratic-equation inverse. This offers convenient explicit
inversion. Their §3.1 also reports failed optimization when boundary derivatives
were not matched to the linear tails. The author implementation matches the
endpoint slope to one
([`nsf-rational_quadratic.py:29`](../../.localresources/q20-flow-training-literature-20260923/code/nsf-rational_quadratic.py)).

For HMC there is an additional distinction: these splines are continuously
differentiable, but generally not twice continuously differentiable. In one
dimension,

\[
\frac{d}{dz}\log\pi_z(z)
=T'(z)s_\pi(T(z))+\frac{T''(z)}{T'(z)}.
\]

Consequently, second-derivative jumps can become force jumps at knots. This
local consequence does not prove that spline NeuTra fails, but prevents calling
ordinary rational-quadratic splines automatically smooth enough for our HMC
requirements. Their knot and tail behavior needs explicit energy-error and
derivative checks. A smooth sigmoid-mixture map is a well-motivated **training
mechanism test**, while a spline is an alternative with different numerical
costs. Neither should be promoted to the production map until its complete
forward/inverse/score contract is implemented and checked. This preference
concerns the proposed scalar component: the smoothness of the complete map
also depends on its conditioners and any other layers.

The computational distinction is important. For an autoregressive map

\[
y_t=\tau_t(x_t;c_t(x_{<t})),
\]

the Jacobian is triangular, so

\[
\log|\det J|=\sum_{t=1}^{D}
  \log\left|\frac{\partial y_t}{\partial x_t}\right|.
\]

The MADE-style conditioner produces all \(c_t\) in one batched pass. For a
DSF with \(K\) sigmoid components, the diagonal terms are computed from the
component slopes and weights with a batched log-sum-exp; there is no full
\(D\times D\) Jacobian. With fixed small \(K\), the scalar work is roughly
\(O(BDK)\). A multi-layer DDSF costs roughly
\(O(BD\sum_\ell K_\ell K_{\ell+1})\), plus the conditioner, and its
conditional pseudo-parameter output can itself be large. The paper's Appendix C
uses stable log-domain matrix products for this reason. At our \(D=4\), this
forward/log-determinant cost is not the main computational objection.

The objection is the inverse. Huang et al. explicitly state that their NAF
architectures have no analytic \(f^{-1}\) and require numerical approximation.
Each scalar inverse requires a monotone root solve, repeated for every
autoregressive coordinate and batch row. That is a bounded XLA `while_loop`
possibility in four dimensions, but it is slower, introduces a tolerance and
iteration budget, and needs a separate inverse-accuracy contract. It is also
incompatible with any training lane that repeatedly evaluates \(q_\lambda(\theta)\)
from physical target samples unless that root solve is accepted and measured.

Our current reverse-KL NeuTra training evaluates \(\theta=T(z)\) and the
forward log determinant, so NAF's lack of a closed-form inverse would not enter
each optimizer update. The latent HMC transition likewise evaluates the
latent-to-physical map and transformed target; it does not need an inverse at
every transition. The inverse is still needed by our density/validation and
some physical-to-latent initialization paths, and a new map is not currently
admitted by the frozen-map HMC payload codec. Therefore the precise verdict is:
**NAF has an efficient Jacobian determinant for reverse-KL training, but it is
not yet an efficient drop-in production replacement for our complete NeuTra
pipeline.**

If production requires an analytic inverse as well as a triangular log
determinant, a monotone rational-quadratic spline is the more direct candidate,
subject to the C1-versus-HMC smoothness checks above. Another low-cost diagnostic
would use a smooth scalar bijection with an analytic inverse, such as a
parameterized sinh/asinh transform; its limited shape family would make it a
mechanism probe rather than a claim of universal expressivity. These alternatives
should be compared by measured forward cost, inverse cost, derivative parity,
and downstream HMC health, not by the paper's generic runtime claims.

## 2. Global contraction should not have to fight a conditional scale cap

Our inherited parameterization is
\(s(a)=2\tanh(a/2)\), followed by \(y=xe^s+m\), with a fixed outer
scale of four. The observed log scale near −1.9936 gives derivative about
0.0064. This attenuates the scale-logit gradient at that stage. It neither
proves the complete map cannot contract nor implies that Adam's update shrinks
by the same factor.

The original NeuTra author code contains a directly relevant option:
[`neutra-utils.py:754`](../../.localresources/q20-flow-training-literature-20260923/code/neutra-utils.py)
bounds the conditional network output **before** adding a free log-scale bias.
In our notation the structure is

\[
s_j(x_{<j})=b_j+c\tanh[h_j(x_{<j})/c].
\]

Then \(\partial s_j/\partial b_j=1\). Global contraction can be learned
through \(b_j\), while the conditional fluctuation remains bounded. Applying
another final cap to the sum would reintroduce the same global saturation.
The source also has an optional learned outer scale, disabled by default in
the inspected constructor. These are inspected implementation capabilities;
the survey does not establish that every published NeuTra experiment used them.

Andrade's *Stable Training of Normalizing Flows for High-Dimensional Variational
Inference* [3, §§3–5] reaches a related design: clamp conditional scales to
limit extreme samples, then include a **trainable final affine** to restore
appropriate global location and scale. The final affine is present in the
author source
[`andrade-core.py:123`](../../.localresources/q20-flow-training-literature-20260923/code/andrade-core.py).

The failure studied there is principally extreme samples from deep RealNVP
flows. We have evidence of contraction near a lower cap, not a demonstrated
exploding-sample mechanism. Its asymmetric clamp constants and tail-compression
layer should therefore not be adopted wholesale. The inspected arXiv version
also contains an inequality following Eq. (7) inconsistent with its stated
clamp constants; the formula and implementation must be distinguished from
that sentence. The useful local lesson is the separation of scale roles.

This repair alone is unlikely to explain away the depth-four nonlinear
residual. It addresses parameter conditioning and allows the nonlinear part
to operate at more useful scales; a diagonal affine transformation does not
directly supply the missing shape flexibility.

## 3. Correct reverse-KL gradients can still obscure nonlinear learning

Roeder, Wu, and Duvenaud's *Sticking the Landing* [4, §2, Eqs. (5)–(8)] shows
that a conventional reparameterized variational gradient contains a
zero-expectation parameter-score term. Removing that term preserves the
expected gradient and makes the estimator vanish pointwise at an exact fit.
The paper explicitly states that variance need not decrease everywhere away
from the optimum. Our gradient-noise measurements motivate testing the
estimator, not assuming that it wins.

Write \(\theta=T_\lambda(z)\), \(z\sim\phi\), and
\(s_q(\theta)=\nabla_\theta\log q_\lambda(\theta)\). Under the usual
common-support, differentiability, and integrability conditions,

\[
\begin{aligned}
\nabla_\lambda\operatorname{KL}(q_\lambda\Vert\pi)
&=E_\phi[(\partial_\lambda T)^T(s_q-s_\pi)]
 +E_q[\partial_\lambda\log q_\lambda(\theta)|_\theta],\\
E_q[\partial_\lambda\log q_\lambda(\theta)|_\theta]
&=\partial_\lambda\int q_\lambda(\theta)d\theta=0.
\end{aligned}
\]

Thus a path-gradient estimator uses
\((\partial_\lambda T)^T(s_q-s_\pi)\). Compute the score difference,
hold it fixed, and apply the transport's parameter vector-Jacobian product.
Only the **first target score** is needed. Derivatives of the transport and
its log determinant are still needed to obtain \(s_q\); this is not an
assertion that all higher flow derivatives disappear.

This keeps reverse KL as the objective. It is **not** training against
\(E\|\nabla\log\pi_z+z\|^2\), which would generally need derivatives of
the target score. Simply stopping gradients through the forward log determinant
would be wrong. Roeder et al.'s §4 identifies the extra dependency problem for
flows; Vaitl et al. [5, §2.3, Algorithm 1] give a flow-correct implementation
using the inverse density and a forward VJP.

The more recent *Fast and Unified Path Gradient Estimators* [6, §3.1,
Proposition 3.2, Appendix B.1 and Algorithm 1] removes the need for inverse
evaluation during reverse-KL training. For a layer \(x_{l+1}=T_l(x_l)\),
let \(J_l=\partial T_l/\partial x_l\) and let \(s_l\) be its input
density's score, as a column vector. The recursion is

\[
s_{l+1}=J_l^{-T}
  \left[s_l-\nabla_{x_l}\log|\det J_l|\right],
\qquad s_0=-z.
\]

It follows by differentiating
\(\log q_{l+1}(T_l(x_l))=\log q_l(x_l)-\log|\det J_l|\).
One can solve the transposed-Jacobian system rather than form an inverse.
Autoregressive Jacobians are triangular; the paper states quadratic rather
than linear dimension scaling for that case. Its optimized implementations
and reported experiments focus on coupling flows, **not our IAF**. Four
dimensions makes the algebra a plausible bounded adaptation, but its runtime
must be measured locally. The paper's speedup factors do not determine the
cost of UKF-dominated training. No identifiable standalone author repository
for [5]/[6] was located in this bounded search; the implementation recommendation
therefore rests on their checked derivations and pseudocode, not an audited
matching fast-path implementation.

An independently inspected author implementation of the path-gradient identity
appears in [3]:
[`andrade-adjusted.py:280`](../../.localresources/q20-flow-training-literature-20260923/code/andrade-adjusted.py)
subtracts `log_q(detached_sample)` from the standard differentiable loss. That
removes the parameter-score gradient. Its modified scalar must not be logged
as the true reverse-KL objective. The same code also filters invalid rows;
copying that behavior would change which samples contribute and is unacceptable
as a silent repair of our objective.

There is a useful exact connection to our existing diagnostic. Define
\(g(z)=\nabla_z\log\pi_z(z)+z\). Change of variables gives

\[
J_T^T(s_q-s_\pi)=-g(z),\qquad
\nabla_\lambda\mathrm{KL}
=-E_\phi[(J_T^{-1}\partial_\lambda T)^Tg(z)].
\]

The last formula explains why a large nonlinear residual can coexist with a
small parameter gradient: the available parameter directions may have little
projection onto the residual. Reducing gradient variance helps resolve an
existing direction; it does not create a missing or nearly redundant direction.
That is why optimizer and parameterization repairs are complementary.

## 4. Initialization and conditioning matter; zero output initialization is not itself a defect

The original IAF paper [7, §3 and Appendices C.6–C.7] uses gated affine
transforms, weight normalization, data-dependent initialization, and ELU
nonlinearities. The NeuTra paper [8, §4.1.1] also uses ELU. These differ from
our two small tanh hidden layers. Salimans and Kingma [9, §§2–3] give the
weight parameterization

\[
w=g\frac{v}{\|v\|},
\]

which separates weight magnitude and direction, and a one-time initialization
that adjusts preactivation means and variances. Glow [10, §3.1] similarly
initializes an affine ActNorm layer from data and then trains its fixed
parameters normally. These are mechanisms for conditioning optimization;
neither paper proves that they will repair q20.

For the small activations in our added stage, a sensible adaptation is to inspect
the **active masked fan-in and preactivation distribution**, then normalize
conditioner inputs or initialize weights accordingly. A normalization fitted
once on a declared representative bank preserves sample independence when
subsequently frozen. For an existing affine first layer, rewriting
\(x=\mu+D\hat x\) gives
\(Wx+b=(WD)\hat x+(W\mu+b)\), so fixed input normalization can preserve
the current map exactly before further training. This algebra does not
preserve Adam's optimizer state automatically; that migration is a separate
choice requiring an explicit check. Weight normalization in a masked network
should account for active weights and constant rows, not divide by the norm
of a zero-degree row.

Changing hidden feature scale, activation, or conditioning should be judged by
whether the map's nonlinear parameter directions respond, not merely whether
hidden weights move. ELU is a source-backed candidate, not a locally validated
replacement; joint smoothness also depends on the chosen activation.

The earlier audit's zero-initialization concern must remain qualified. Glow
[10, §3.3] deliberately zero-initializes the final layer of each coupling
network, and NAF [1, Appendix E] deliberately starts near identity with tiny
weights. Those choices can train successfully. Our hidden gradients vanish
at the exact zero-output initialization but become nonzero after the output
weights move, as earlier local checks observed. The actionable question is
whether feature scaling and the training schedule permit useful nonlinear
directions to develop, not whether zero initialization is categorically wrong.

The original NeuTra unconditional experiments [8] used three stages, two
hidden layers, 5,000 Adam updates, batch 4,096, initial learning rate 0.01,
and tenfold decays at updates 1,000 and 4,000. These are **reported experimental
settings**, not q20 defaults. Our principal repair added 1,024 updates at batch
32 with constant learning rate 0.0005. This establishes that the protocols
differ; it does not establish how many q20 updates or samples are sufficient.
Effective batch and decay should follow measured gradient variation, actual
update size, and fresh checkpoint progress at a priced target-evaluation cost.

## What the literature does not justify

Koehler, Mehta, and Risteski [11, Theorem 1 and Remarks 2–3] show that shallow
affine coupling constructions can approximate bounded-support continuous
distributions in Wasserstein distance while becoming increasingly
ill-conditioned. The statement concerns a particular representational family,
metric, and construction. It does not prove that our bounded tanh IAF is
sufficient, insufficient, trainable, or suitable for HMC. Both “affine flows
are universal, so just train longer” and “four stages failed, so NeuTra cannot
work” go beyond this evidence.

Reverse-KL self-sampling also has a coverage limitation: a region rarely
visited by the proposal contributes little observed training signal. [6]
discusses combining forward and reverse KL when reliable target samples are
available. Our 1,000 base-normal points inspect proposal geometry; they cannot
certify posterior coverage. We have not established missing modes here.
Obtaining additional reference information may become necessary if coverage
checks fail, but speculative mode collapse is not a reason to change the
objective now. A learned mixture or an annealing scheme would need its own
target-specific rationale and validation.

Białas, Korcyl, and Stebel [12, §3, Eqs. (18)–(22), Algorithm 1 and appendix
listing] study expensive target evaluations and a target-gradient-free
score-function estimator. It is useful context if target derivatives dominate
cost. Its own Eq. (21) identifies the factor \((B-1)/B\) induced by subtracting
the same batch's mean baseline. We should not call that exact unbiasedness,
or replace our available analytic target score without a measured cost/variance
advantage. It is not the first repair suggested by our evidence.

## A discriminating repair sequence

The following is a recommendation, not an executed or numerically calibrated
campaign. It preserves the current target. Its controls distinguish mechanisms
within NeuTra and do not introduce an all-method comparison requirement.

| Step | Concrete intervention or check | What it would establish |
| --- | --- | --- |
| Establish the endpoint | At the same saved depth-four checkpoint, measure per-block gradient means/variation, actual updates, scale slopes, and nonlinear parameter-direction response | Whether noise, weak parameter directions, or a particular capped scale remains an active obstruction; parent measurements are not endpoint measurements |
| Check the gradient estimator | Implement the exact proposal score and path VJP; verify analytic fixtures and paired gradient expectations before continuation | Same RKL first gradient in expectation; any variance/cost benefit is then a measured property rather than an assumption |
| Separate scale roles | Use a free global log scale plus a bounded conditional fluctuation, or a learned external affine, with a documented map/optimizer initialization | Whether scale saturation is impairing the fit; affine freedom alone is not a nonlinear repair |
| Expose scalar shape directly | Freeze the current map and precompose a smooth monotone scalar map in latent coordinate 2, with distinct near-identity components; retain identity and affine scalar controls | Whether direct marginal deformation can repair the concentrated residual without requiring indirect cross-coordinate composition |
| Expand only if indicated | If scalar repair leaves dependence on other coordinates, use conditional scalar transforms or repair conditioner scaling and joint training | Separates marginal non-Gaussian shape from coupled posterior structure; coordinate concentration alone does not prove separability |
| Price sustained learning | Choose effective batch, learning-rate schedule, and checkpoint cadence from gradient/update evidence and elapsed target cost | A target-specific continuation protocol, with termination based on budget and declared progress evidence |
| Validate the resulting map | Fresh heldout objective estimates, the standard 1,000-point score diagnostic, numerical/tail checks, and then the existing fixed-map HMC qualification | Geometry can nominate a map; stable posterior estimation remains the downstream success criterion |

The scalar experiment is mathematically specific. With the frozen current map
\(T_0\), define
\(D_\psi(z)=(z_1,\tau_\psi(z_2),z_3,z_4)\) and train
\(T_\psi=T_0\circ D_\psi\) against the original RKL, including
\(\log\tau'_\psi(z_2)\). This changes the proposal, not the posterior.
The dominant residual direction motivates this inexpensive mechanism test;
the same inspected bank must not become final untouched evidence. A failed
scalar candidate does not reject conditional flow repair.

There is a concrete integration gap before a new scalar family can enter HMC.
The current [HMC tuning interface](../reference/hmc-tuning-interface.md) accepts
supported frozen affine or dense-IAF payloads; a sigmoid-mixture or spline
composition is not automatically supported. The inspected
`HMC_TUNING_INTERFACE_CAPABILITIES` entry in
`bayesfilter/inference/tuning_contract.py:1049` requires an exact frozen
transport, identity latent mass, and fresh verification after map changes.
A new family therefore needs checked serialization/reconstruction, forward and
inverse operations, Jacobian and transformed-score parity, and explicit support
in `tune_fixed_transport_hmc_kernel` before sampler qualification. Passing it
as an arbitrary force callback would not close that gap. This is implementation
work for a future repair, not evidence against testing the training mechanism.

The important numerical choices remain hypotheses with explicit ways to set
them:

| Choice | Provenance and how to determine it |
| --- | --- |
| Coordinate 2 for the first scalar diagnostic | Measured 89.02% residual-energy concentration on the depth-four bank; explanatory nomination, to be checked on fresh points |
| Number of sigmoid components | One is exactly affine; two is the algebraic minimum for nonlinear mixtures. Sufficient capacity remains uncalibrated; inspect approximation and gradient directions before increasing it |
| Component asymmetry and initialization scale | Near identity is supported by [1], but exact component equality is tangent-degenerate. Set perturbations using initial map displacement, finite Jacobians, and non-affine parameter directions; the paper's small-weight constants are warm-start hypotheses only |
| Conditional scale cap and free affine scale | Existing cap 2 is inherited. Separate the free scale first; set any revised cap using realized derivative/conditioning and tail evidence, not [3]'s unrelated constants |
| Spline tail extent, bin count, and positive floors if that option is pursued | [2]'s values are implementation settings. Choose the tail extent from declared base coverage and check knot/force behavior; do not silently treat its bin count or floors as q20 defaults |
| Batch, learning rate, stopping, and replication | Existing 32/0.0005 and earlier update allocations are baselines. Determine a priced protocol from gradient variation, checkpoint uncertainty, update stability, and remaining budget before execution |

## Decision and remaining uncertainty

| Decision | Primary criterion status | Veto status | Main uncertainty | Next justified action | What is not established |
| --- | --- | --- | --- | --- | --- |
| Treat nonlinear scalar parameterization and path gradients as concrete repair candidates | Mechanisms supported by inspected equations and source; local benefit untested | Full-support loss, wrong gradient, nonfinite values, or invalid derivative checks would veto an implementation | Relative contribution of noise, weak features, scale saturation, and coverage | Prepare the smallest endpoint/gradient and scalar-deformation checks under a priced repair plan | No candidate is certified trained, converged, or HMC-ready |
| Repair the interpretation of initialization and depth | Literature and local audit reject categorical blame of zero output initialization or a fixed layer count | Unsupported causal attribution is rejected | Whether the current initialization is slow for this specific posterior | Measure nonlinear parameter response; preserve successful-map initialization where possible | No proof that another activation or more layers alone fixes q20 |

No new stochastic comparison was run, so there is no statistically supported
candidate ranking or default promotion. Previously reported local fractions
are descriptive evidence. A later repair comparison needs predeclared paired
uncertainty and fresh validation; passing a geometry screen alone does not
establish superiority or posterior correctness.

The strongest alternative explanation is that the current architecture could
learn the required deformation under an adequately calibrated continuation,
without a new scalar family. Evidence of useful existing nonlinear parameter
directions plus sustained fresh-validation progress would weaken the case for
an architecture change. Conversely, a scalar layer that remains unable to
repair the target with verified gradients would direct attention to joint
dependence, coverage, or the target approximation. The weakest current link
is transfer from other targets: none of the surveyed experiments is this UKF
posterior.

## Inspected references

All links identify the version actually archived, rather than an assumed
latest revision. Author code was read but not executed or imported.

| Ref. | Paper and inspected technical anchors | Author-code inspection |
| --- | --- | --- |
| [1] | Huang et al. (2018), [Neural Autoregressive Flows](https://arxiv.org/abs/1804.00779v1), §§3–4, §6.1.2/§6.2, Appendices B–C/E and relevant universality argument | `CW-Huang/NAF` example and linked `CW-Huang/torchkit` sigmoid flows/initialization; tiny mixture clipping distinguished from ideal full-range map |
| [2] | Durkan et al. (2019), [Neural Spline Flows](https://arxiv.org/abs/1906.04032v2), §§3/5, Appendices A–B | `bayesiains/nsf` scalar forward, inverse, log determinant, positivity, tail matching, autoregressive wrapper, ActNorm |
| [3] | Andrade (2024), [Stable Training of Normalizing Flows for High-Dimensional Variational Inference](https://arxiv.org/abs/2402.16408v1), §§2–6, estimator appendix and ablation table | `andrade-stats/normalizing-flows` clamps, tail map, final affine, estimator, invalid-row filtering |
| [4] | Roeder et al. (2017), [Sticking the Landing](https://arxiv.org/abs/1703.09194v3), §§2–4 and evaluation | `geoffreyroeder/iwae` parameter detachment in Gaussian density; not mistaken for a complete chained-flow implementation |
| [5] | Vaitl et al. (2022), [Gradients Should Stay on Path](https://arxiv.org/abs/2207.08219v1), §1.3, §2.3 Algorithm 1, §§3–4 | No matching standalone repository located; exact inverse-density method checked in paper and related implementation [3] |
| [6] | Vaitl et al. (2024), [Fast and Unified Path Gradient Estimators](https://arxiv.org/abs/2403.15881v1), §§2–3/5, Appendix B.1, Appendix D algorithms, relevant Appendix E evaluation/timing | No matching standalone repository located; recursion/proof/pseudocode inspected; IAF adaptation and runtime untested |
| [7] | Kingma et al. (2016), [Improved Variational Inference with IAF](https://arxiv.org/abs/1606.04934v2), §3 Eqs. (12)–(14), Appendix C.6–C.8 | `openai/iaf` weight-normalized, data-initialized linear/conv helpers and ELU autoregressive conditioner |
| [8] | Hoffman et al. (2019), [NeuTra-lizing Bad Geometry](https://arxiv.org/abs/1903.03704v1), method and §4.1.1/evaluation | `google-research/google-research/neutra` map constructor, scale options, initializer, RKL/Adam/schedule |
| [9] | Salimans and Kingma (2016), [Weight Normalization](https://arxiv.org/abs/1602.07868v3), §§2–3 and evaluation | `TimSalimans/weight_norm` magnitude/direction parameterization and one-time data initialization |
| [10] | Kingma and Dhariwal (2018), [Glow](https://arxiv.org/abs/1807.03039v2), §§3.1–3.3 and evaluation context | `openai/glow/tfops.py` ActNorm and zero-initialized final layers |
| [11] | Koehler et al. (2021), [Representational Aspects of Depth and Conditioning](https://arxiv.org/abs/2010.01155v2), §3.1, Theorem 1, Remarks 2–3 and relevant proof structure | Used for a narrow theoretical limitation, not an implementation prescription |
| [12] | Białas et al. (2023; archived 2024 revision), [Training Normalizing Flows with Computationally Intensive Target Probability Distributions](https://arxiv.org/abs/2308.13294v2), §3 Eqs. (18)–(22), Algorithm 1, evaluation/limitations, appendix code | Paper's implementation listing inspected; target-score-free alternative not selected |

Retrieval used ResearchAssistant metadata discovery and local parsing, with
public arXiv and author-GitHub downloads. Provider coverage was partial,
automatic relevance selection missed important sources, and parser confidence
was low with some incorrect title guesses. Titles, equations, and claims were
therefore checked against the papers' extracted technical text. Some PDFs
reported recoverable cross-reference warnings during extraction; their text
was recovered. A partial spline download is preserved but excluded. Search
challenge pages and absent fast-path code are recorded as coverage limits,
not evidence of source nonexistence.
