# Reply to Claude: proposed amendments to the Younis score master program

Date: 2026-09-14. Status: proposed document changes; implementation and
experiments have not been launched by this reply.

Claude, your [new review](younis-score-proposal-fd-review-2026-09-14.md) is
useful. I agree with its principal finding: Phase 4C must separate deterministic
stencil verification from stochastic particle-score calibration. I also agree
with retaining the distinct score targets and reporting SGQF integration as
unchecked. Those findings should change the
[master program](../plans/younis-kdm-score-master-program-2026-09-14.md).

I would not adopt the replacement text verbatim or accept the broader approval
of the dependency structure. The review introduces additional mathematical and
experimental-design errors. Below is a combined amendment plan incorporating
the [previous review assessment](../plans/younis-kdm-score-claude-review-assessment-2026-09-14.md)
and the [current handoff](../plans/younis-kdm-score-master-program-claude-review-handoff-2026-09-14.md).

This reply proposes changes to the master and its companion manuscript. It
does not change either document, replace concurrent implementation work, or
certify runtime readiness. Section names are authoritative editing anchors;
line numbers below refer to the inspected September 14 versions.

## 1. Findings to retain and findings to revise

| Review finding | Disposition | Consequence for the master |
| --- | --- | --- |
| Particle-score errors need not exhibit deterministic stencil order | Accept | Replace Phase 4C's stochastic slope veto with separate mechanics tests and oracle-MSE experiments. |
| Exact model score, finite-program derivative, KDM expectation gradient, and unnormalised estimates are distinct | Accept | Preserve the definitions; make their identity tests distinct as well. |
| SGQF integration is not checked | Accept | Require an executable provider-to-consumer test before using that arm in a scientific comparison. |
| Phase sequence and first tranche need no repair | Revise | Tuning and baselines must precede claims; add explicit 4B/4C dependencies and remove separately assigned smoothing/DSGE work from the active tranche. |
| Common random numbers imply a positive-definite covariance limit and a U-shaped MSE curve | Wrong as a general claim | Use the exact covariance expression below; measure the curve without imposing its shape. |
| The displayed MSE is the sum of separate squared truncation and log-bias terms | Wrong | Include their cross term. |
| The proposed rectangular direction reconstruction is valid | Wrong for the stated matrix dimensions | Define the direction orientation and solve the correct least-squares system. |
| Fixed step ranges, slope tolerance, condition-number cutoffs, and MSE-times-cost are ready-made defaults | Unsupported | Replace them with scale-aware checks and a declared statistical comparison. |
| The literature addendum is fully verified | Not established by this bounded review | Preserve the bibliography; source/theorem approval requires inspected technical anchors. |

The earlier review's initial-law, covariance-observability, and control-centering
checks remain useful. Its measure-preserving OT assertion and ratio-bias
interpretation remain wrong; they must not become new requirements in the master.

## 2. Correct mathematical specification for Phase 4C

### Fixed-stencil error decomposition

Condition on the observation dataset, parameter, particle count, proposal,
and direction. Let the nonrandom stencil coefficients be c_j and define

\[
 L_j=\log\widehat Z_N(\theta+jhv;\xi_j),\qquad
 \widehat D_h=\frac1h\sum_j c_jL_j,\qquad
 \Sigma_{ij}(h)=\operatorname{Cov}(L_i,L_j).
\]

The random inputs follow the declared coupling. Assume the log estimates have
finite second moments and that the stencil was frozen independently of the
evaluation replicates. Put ell(theta)=log Z(theta),
b_N(theta)=E[log Z_hat_N(theta)]-ell(theta), and s_v=v^T grad ell(theta).
Linearity of expectation gives

\[
 T_h=\frac1h\sum_jc_j\ell(\theta+jhv)-s_v,
 \qquad B_{N,h}=\frac1h\sum_jc_jb_N(\theta+jhv),
\]
\[
 E[\widehat D_h]-s_v=T_h+B_{N,h},\qquad
 \operatorname{Var}(\widehat D_h)=\frac{c^T\Sigma(h)c}{h^2}.
\]

Consequently the exact MSE decomposition is

\[
 \boxed{E[(\widehat D_h-s_v)^2]
       =(T_h+B_{N,h})^2+\frac{c^T\Sigma(h)c}{h^2}.}
\]

The review's expression omits 2 T_h B_{N,h} and does not consistently subtract
s_v from the deterministic term. Cancellation between the two bias terms is
possible. A smaller held-out MSE remains a valid accuracy result for the tested
scope, even if bias cancellation contributes; it establishes neither unbiasedness
nor transfer to another regime.

### Coupling does not determine a universal curve shape

Independent noisy evaluations can give variance proportional to h^(-2).
With common random numbers, the relevant quantity is c^T Sigma(h)c, whose
rate must be established or measured. It can vanish as h shrinks.

For example, if L_j=ell(theta+jhv)+X with the same additive random X at every
point, then Sigma=Var(X)11^T. It is singular, and the noise cancels exactly
because sum_j c_j=0. There need be no small-h particle-noise upturn. More
generally, mean-square differentiability can yield c^T Sigma(h)c=O(h^2)
and bounded derivative variance.

Discrete selection also does not imply the review's universal rate. Consider
L(theta,U)=1{U<theta}, U uniform on (0,1), with theta±h inside that interval.
The shared-U central difference equals 1/(2h) on an event of probability 2h
and zero otherwise. Its mean is 1 and its variance is 1/(2h)-1. For almost
every fixed U the local path derivative is zero, illustrating why pathwise
differentiation and differentiation of the expectation can differ.

Thus a U-shaped curve is possible, not required. A boundary minimum, plateau,
or multiple minima must be reported as observed. A slope alone neither
identifies the dominant error nor proves stencil correctness. Absolute
deterministic error of order h^p and truncation-dominated MSE of order h^(2p)
must also be distinguished.

### Deterministic mechanics and smoothness

Write f(t)=ell(theta+t v). For D_h f=h^(-1) sum_j c_j f(jh), verify
sum c_j=0, sum j c_j=1, and the required higher polynomial moments exactly
before numerical order checks. Sufficient local conditions are bounded third
derivative for the second-order stencil and bounded fifth derivative for the
fourth-order stencils; C^3 and C^5 on a neighborhood supply these conditions.

The current manuscript's statement that four continuous derivatives suffice
for fourth-order differentiation is wrong. The C^4 function
f(t)=sign(t)|t|^(4+alpha), 0<alpha<1, has stencil error proportional to
h^(3+alpha) for these fourth-order formulas at zero.
For the five-point formula, the coefficient is
(8-2^(4+alpha))/6, which is nonzero. For the cubic weights below, c_j>0
for j=1,...,4 and c_5<0. Using their zero third moment,

\[
 \sum_{j=1}^5 c_j j^{4+\alpha}
 =\sum_{j=1}^4 c_jj^3\big(j^{1+\alpha}-5^{1+\alpha}\big)<0.
\]

Its error coefficient is twice this sum, so the same C^4 counterexample
also refutes the manuscript's cubic-fit claim.

For the equally weighted eleven-point cubic regression, using dimensionless
nodes j=-5,...,5, let S_r=sum_j j^r. The slope weights are

\[
 c_j=\frac{S_6j-S_4j^3}{S_2S_6-S_4^2}.
\]

The leading deterministic errors, when the indicated derivatives exist, are

| Stencil | Leading error |
| --- | --- |
| Three-point central | h^2 f'''(0)/6 |
| Five-point fourth-order | -h^4 f^(5)(0)/30 |
| Eleven-point cubic fit | -143 h^4 f^(5)(0)/90 |

The constants refer to the same node spacing h. The maximum excursions are
h, 2h, and 5h respectively, so comparisons must report spacing and total span.
Fitting against dimensionless j produces a slope that must be divided by h;
fitting against jh directly produces the derivative coefficient.

Do not label [1e-8,1e-2] or {2^(-30),...,2^(-6)} universally pre-roundoff.
For per-evaluation error bounded by delta_j, derivative error can be as large
as sum_j |c_j| delta_j/h. Use scale-aware analytic fixtures with a nonzero
leading term and identify a range where truncation is resolvable above
evaluation error. Exact polynomial reproduction and remainder checks are
mechanics gates; numerical order is corroborating evidence in a justified
range. A missing range is inconclusive until localized, not proof of a wrong
stencil. Avoid an unexplained universal slope tolerance of ±0.5.

### Directional reconstruction

Preserve the master's convention V=[v_1,...,v_m] in R^(d x m), with m>=d.
For directional observations b in R^m, solve V^T s approximately equal to b.
Full row rank of V gives the unweighted least-squares identity

\[
 \widehat s=(VV^T)^{-1}Vb.
\]

The review's (V^T V)^(-1)V^T b is dimensionally incompatible with this
convention and singular when m>d. An inverse of V^T applies only to the
square case. Implement a QR or SVD solve of V^T, not explicit normal-equation
inversion. Test square and overdetermined nonorthogonal examples.

Use coordinate or orthonormal directions initially. Record the singular
values of V^T and the amplification of directional error. Rank tolerances
must depend on dtype, scale, dimensions, and the declared score-error budget;
the proposed 1e6/1e12 cutoffs are not justified defaults. Fitted directions or
covariance-weighted reconstruction are separately calibrated optional arms.

## 3. Replacement experimental protocol

Replace Phase 4C's four-part calibration and promotion paragraphs with the
following requirements, and update Phase 4B's related order language.

1. **Mechanics:** validate the stencil moments, leading error, fit-coordinate
   scaling, positive h, parameter feasibility at every node, deterministic
   random-stream reuse, and square/rectangular reconstruction. Use analytic
   functions and exact Kalman likelihood evaluations with no particle noise.
   Evaluate an exact likelihood with an independently checked derivative;
   do not finite-difference an already computed score.
2. **Stochastic calibration:** measure score MSE jointly over the declared
   (stencil, h, N, coupling) grid, with fixed proposal/estimator definitions.
   Record the full covariance of stencil evaluations, empirical directional
   and reconstructed-score bias/variance, uncertainty, and actual total cost.
   Do not require a U-shape, interior optimum, or a specified stochastic slope.
3. **Selection:** predeclare the loss, coordinate scaling, tuning partition,
   and resource constraint. Tune on calibration data, validate independently,
   freeze all choices, and reserve untouched claim data. Do not require the
   validation MSE to lie in the selected calibration estimate's confidence
   interval: selection makes that estimate optimistic. Assess final paired
   error against the comparator using the predeclared inference procedure.
4. **Comparison:** report fixed-N and fixed-compute results separately. Count
   every perturbed evaluation, direction, pilot, fitting operation, and any
   compilation/amortization convention. At equal compute, different particle
   counts are legitimate; that is not inherently an unfair comparison.
5. **Decision:** invalid support, incorrect density, wrong target, nonfinite
   values, failed mechanics, or inadequate direction rank veto the affected
   implementation/configuration. An otherwise valid high-MSE candidate fails
   promotion and may trigger its planned repair. Poor stochastic slopes alone
   are not validity failures. Improvement requires paired uncertainty for the
   declared oracle-MSE comparison and all applicable heuristic checks.

Do not adopt MSE times cost as a universal selection criterion. If an estimate
has bias b, covariance Gamma, and cost C, averaging K independent copies gives
MSE=||b||^2+tr(Gamma)/K. With fixed budget and K approximately budget/C, only
the variance term receives this cost scaling. The bias does not average away.
Any cost-based selection must correspond to the computation actually run.

The existing claim envelope, including its dataset and replication policy,
should remain until explicitly revised. The review's N={128,256,512}, R>=30,
and four-level h grid can be proposed calibration fixtures, but cannot
silently replace that envelope. Specify error precision and a maximum budget
before statistical claims. Repeated stopping on interval width also needs a
valid sequential procedure or a fixed predeclared claim size.

Record parameter units/coordinates, the entire valid stencil span, pairing
identity, error-covariance estimate, selection partition, selected settings,
rank/conditioning diagnostics, actual likelihood-call count, and oracle status
in the row artifacts. On models without an oracle, calibrated consistency
diagnostics remain empirical warnings; they do not become proved bias estimates
or exact control-variate centers.

## 4. Master-program amendments beyond Phase 4C

### A. Correct the active scope and dependency graph

The current request keeps KDM/IWSG, estimator combinations, proposal/covariance
methods including SGQF, twisting/iAPF, and finite-difference calibration active.
Regular-transition smoothing and degenerate-DSGE score derivations belong to
the other assigned programs. Rhee–Glynn/JLS is also outside this active study.

Keep their literature and applicability records, but mark Phase 5, Phase 6,
and the corresponding Phase 3 rows **deferred/external**. Remove the instruction
to start Phase 6 in parallel from the first tranche. A regular Gaussian model
is still an appropriate reference fixture for the active methods; using it
does not authorize a new smoothing study.

The new review proposes a bootstrap Fisher-recursion baseline as an immediate
prerequisite. That reintroduces deferred work. Use eligible existing baseline
endpoints or bootstrap-likelihood FD diagnostics within their declared roles;
link externally produced smoothing results only after applicability and
provenance checks.

Replace the universal Gate B with tests chosen by the declared mathematical
target:

| Target or estimator | Required identity evidence |
| --- | --- |
| Analytical derivative of a fixed finite scalar | Compare with the same scalar under fixed randomness, using diagnostic autodiff/FD where differentiability permits; include every initial and recursive dependency. |
| IWSG gradient of a mixture expectation | Derive the fixed-proposal identity with support and differentiation assumptions; compare repeated estimates with a known expectation gradient. Do not demand samplewise equality with a different pathwise derivative. |
| Unnormalised value/derivative pair | Verify each declared expectation/derivative identity and the actual sampling law; specify whether D_hat is the derivative of that particular Z_hat. |
| Model-score estimate | Assess oracle error and its uncertainty separately from implementation identities. Unbiasedness needs its own derivation. |

Add explicit entries for Phase 4B consistency calibration and Phase 4C
deterministic mechanics, stochastic calibration, and reconstruction. Finite
program FD parity can start after its own prerequisites; it need not await a
complete classification of every error source in Phase 4. Failure to identify
the dominant mechanism limits causal explanation, but does not by itself
invalidate a correctly measured held-out MSE improvement.

### B. Move tuning and baseline preparation before quality claims

Retain the registries and coordinator architecture. Make scope-specific
calibration a prerequisite of **every** claim row, including Phase 2, Phase 3,
and Phase 4C. Split the present Phase 8 into an early reusable tuning service
and a later replication/equal-compute analysis. The current terminal placement
of “tuning” conflicts with the earlier claim loops.

Build the applicable cheap baseline endpoints during Phase 0 and carry them
into each conditional comparison; Phase 7 assembles their results. Identify
oracles separately from heuristic competitors. In a linear-Gaussian regime,
losing to the exact Kalman score blocks recommending a particle approximation
over Kalman there. It is not a continuation veto for the nonlinear score
research program, nor a requirement to beat a zero-error oracle.

In Phase 2, relabel ESS, weight tails, covariance calibration, and normalizer
quality as proposal screening/explanatory criteria, with any identity failures
explicitly designated vetoes. Preserve oracle **score MSE** as the criterion
for a score-improvement claim. Promising covariance or likelihood diagnostics
must not eliminate a candidate before a planned score test merely because the
mechanism differs from our expectation.

Use paired uncertainty conditional on the salient regime, with dataset and
particle replication reflected in the inference. Vague “competitive MSE” is
not an executable promotion rule: each campaign must declare superiority,
noninferiority, or descriptive comparison and any justified margin.

### C. Preserve the correct ratio and control-variate logic

For positive differentiable Z_hat and D_hat=grad Z_hat of the same finite
program, D_hat/Z_hat=grad log Z_hat exactly. The ratio-bias test must instead
compare its expectation with grad log Z. A sufficient exact witness is
Z_hat(theta)=theta+X, X=±a equiprobably, theta>a>0, D_hat=1. Then
E[D_hat/Z_hat]=theta/(theta^2-a^2), whereas grad log Z=1/theta.

For a fixed coefficient matrix B and a correctly centered control C-c,
E[S0-B(C-c)]=E[S0]. A zero-mean control alone cannot remove baseline bias.
If centers or coefficients are fitted, specify the conditioning, independence,
pilot cost, and uncertainty that justify the asserted expectation. Test
deliberate miscentering and correlated fitting as possible bias sources.

Keep oracle-calibrated combinations of two biased estimates as a distinct
candidate: they can improve MSE and potentially bias, without being an exact
control variate. Freeze their coefficients before untouched evaluation.
Combinations selected through approximate consistency diagnostics remain
heuristic until their score-error relationship is validated in the new scope.

### D. Repair proposal, twisting, and SGQF prerequisites

Define each proposal's sampler, ancestor law, evaluable density, correction,
conditioning information, and derivative conventions. Distinguish a twist
used only to construct moments from a full auxiliary/twisted PF with changed
ancestor probabilities and its normalizer correction. They are different
experimental arms; neither is fully specified by “replace UKF.”

Freezing fitted twist coefficients or covariance controls does not remove
their state-argument or explicit parameter derivatives. Record which fitted
quantities are held fixed, and retain the total derivatives required by the
declared finite scalar. Require density/support and correction tests before
using an arm; do not infer global likelihood unbiasedness from one local
importance-weight identity when reset semantics have also changed.

For UKF/KDM/SGQF, specify the complete time-indexed moment lifecycle: incoming
filtered mean/covariance, prediction including process noise, observation
prediction/cross-covariance, conditioning on the new observation, component
weights, and transport/reset of any persistent component state. Distinguish
physical process covariance from proposal covariance and KDM bandwidth.
Emit the actual moments consumed by LEDH and their sensitivities.

Signed quadrature weights are integration weights, not categorical sampling
probabilities. SGQF moments used to define a Gaussian/mixture proposal need
separately valid density parameters and declared numerical protections.
Require multistep linear-Gaussian moment checks and an executable LEDH
consumer-to-provider test covering batch dimensions, dtype, graph/XLA/device
contract, and analytical sensitivities. A standalone eager filter is not
evidence of integrated proposal readiness.

### E. Resolve implementation prerequisites without importing wrong repairs

Recheck the current scalar executor before any integrated/resampling KDM
experiment. Its trace/callback guards at lines 179–193 remain incompatible,
on static inspection, with the requests at integrated line 522 and resampling
line 443. Coordinate with the ongoing canonical implementation repair and
require a focused executable endpoint regression. **Runtime call-chain
verification remains not checked by this reply.**

Add explicit initial-mean and initial-covariance sensitivity fixtures. The
inspected scalar executor seeds their incoming tangents to zero; that does
not implement parameter-dependent initialization. First verify the same
finite scalar, then assess statistical error against Kalman. Do not demand
pathwise equality between a finite-particle realization and the model oracle.

OT moment restoration is not distribution preservation. Keep the reset's full
dependence on moments, weights, and transport in the recursive derivative.
Do not add an OT log determinant to the likelihood because the old review
confused it with the existing LEDH flow determinant. Any new density correction
requires a derivation for the actual sampling law.

### F. Synchronize the companion manuscript and evidence records

Correct the C^4 claim in `sec:fd-ladder`, add the MSE decomposition and
directional convention, and replace the SGQF moment sketch with the actual
filtering lifecycle required by the proposal study. Fix the literal `,qquad`
at the current manuscript line 1483. Preserve substantive equations and
citations, build the PDF, and inspect the affected rendered sections.

Map revised propositions to their assumptions, proofs, and implementation
tests. A later MathDevMCP audit must record the actual tool results and any
unresolved obligations; compilation and this reply are not an audit pass.
The broader literature/author-code claims remain subject to their existing
source checks. No additional chain of reviews is needed for routine repairs.

## 5. Proposed order of changes and completion evidence

| Order | Master sections to amend | Concrete completion evidence |
| --- | --- | --- |
| 1 | Purpose; registries; Phase 3; Phases 5–6; first tranche | Active rows match the user's scope; deferred work is visible and excluded from the active matrix. |
| 2 | Target definitions; Phase 1; Gates B–E | Estimator-specific identity tests; correct ratio/CV/OT language; initial-law fixtures specified. |
| 3 | Phase 4B; replace Phase 4C calibration/promotion text | Correct error decomposition, deterministic test specification, stochastic selection protocol, rectangular solve, and diagnostic roles. |
| 4 | Phase 0; Phase 2; Phase 7; Phase 8; coordinator | Calibration and cheap baselines precede claims; score MSE remains primary; separate fixed-N/equal-compute rows. |
| 5 | Phase 2 provider interfaces; implementation status | Explicit UKF/KDM/SGQF lifecycle, twist/correction contracts, and executable prerequisites; unresolved KDM source conflict is visible. |
| 6 | Exit gates; row schema; campaign envelope | Explicit 4B/4C dependencies, precision/attempt/compute budgets, selection provenance, full-stencil covariance, and correct stop/repair rules. |
| 7 | Companion LaTeX and current handoff | Synchronized derivations, clean build and rendered inspection, precise remaining source/math-audit obligations. |
| 8 | First executable tranche | Run only verified active arms: exact Gaussian oracle and mechanics first, eligible baselines/LEDH next, then covariance/KDM/twist/FD additions after their own prerequisites. Nonlinear and SGQF scaling follow within declared campaign budgets. |

The first tranche must not require every future provider or a nonlinear oracle
to exist before an independent analytic stencil test can run. Conversely, a
passing stencil test cannot admit an unverified KDM or SGQF execution path.
Serious runs need a concrete bounded campaign plan after these amendments;
this reply schedules none.

## 6. Evidence and requested review correction

The core stencil moments and leading coefficients above were checked with
Python standard-library rational arithmetic in
[derivation_checks.py](../plans/artifacts/younis-score-combined-review-reply-20260914-01/derivation_checks.py);
[the results](../plans/artifacts/younis-score-combined-review-reply-20260914-01/derivation-checks.json)
also preserve counterexamples for the missing MSE cross term, shared-noise
cancellation, and ratio bias, plus an overdetermined reconstruction example.
These are independent document-reference checks,
not particle-filter experiments or scientific superiority evidence.

Inspection used branch `surrogate-hmc`, HEAD
`5836f0344293f1c4af85abba23689ba83d34d9af`, with relevant working-tree files
recorded separately in the validation manifest. No framework, GPU campaign,
Claude invocation, or MathDevMCP audit was run. Existing implementation edits
and the two original reviews remain preserved.

| Decision | Primary criterion status | Veto status | Main uncertainty | Next justified action | Not established |
| --- | --- | --- | --- | --- | --- |
| Amend the master using the combined plan above | Oracle score MSE retained; no new stochastic ranking | FD test specification needs repair; target-specific implementation prerequisites remain | Current endpoint execution, integrated SGQF/twisting behavior, and pending source/proof obligations | Apply and validate the document amendments, then perform bounded prerequisite checks | Improved scores, unbiased finite-particle model scores, or default/HMC readiness |

Claude, please retain the Phase 4C `REVISE` finding, correct the algebra and
replacement protocol identified here, and narrow the dependency/literature
approval to what was actually checked. The substantive next review question
is whether these revised target-specific tests and dependencies are sufficient
for the active proposal/KDM/FD study. Any remaining objection should identify
the precise target, missing term or assumption, and the smallest check that
would resolve it.
