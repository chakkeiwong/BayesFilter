# Generic NeuTra recovery and transfer plan

Status: proposed and skeptically reviewed design, October 3, 2026, revised
to require systematic tests of FAB, Gabrié, annealed AIS, annealed SMC, AFT
and CRAFT. This document answers the request for a generic,
literature-grounded repair plan and a complete six-method assessment.
It does not report a new implementation or authorize an unpriced experiment.
The next implementation should extend the shared training and campaign
components; it should not create another competing NeuTra implementation.

The intended outcome is a reproducible procedure that obtains useful,
mode-covering posterior estimates with plain NeuTra HMC or the already
requested tempered NeuTra ensemble. Either can satisfy the estimation goal.
Perfect Gaussianization is a useful mathematical reference, not a prerequisite
for correctness or a substitute for downstream evidence.

The six-method assessment is mandatory under the owner's latest request.
One method succeeding does not cancel the other five. The older q20 directive
against requiring an all-method comparison before producing an estimate does
not remove this newly requested study. A qualified q20 estimate may be reported
when available; the systematic study has its own completion conditions.

## 1. What the plan must repair

The current problem has several distinct causes. A single change of sampler,
or another fixed training-count increase, does not address all of them.

| Checked evidence | Consequence for this plan |
|---|---|
| FAB's tested q20 alpha-two route failed calibration; the ideal ordinary-UKF auxiliary density was non-integrable for the audited initial map | Preserve that rejection. Do not resume FAB merely with more particles or a different step size. Executable numerical-domain semantics remain a separate question. |
| The September 30 mixture study failed even with exact target samples; many warm maps were nearly Gaussian, and RKL often lost a component | Teacher quality alone cannot explain or repair the fitting problem. Use an exact-teacher control and inspect function shape. |
| October 1 continuations and consumer repairs produced some passing maps under their declared profiles, while some Gabrié fits still stalled | Reuse verified state restoration and consumer fixes. Do not claim all prior attempts failed, or transfer a pass across changed criteria. |
| October 2 exact-transform and teacher controls passed; none of eight learned cases reached fresh posterior confirmation; all 48 fit jobs stopped as budget-limited | Combined FKL/RKL and longer training have already been tried. They remain baselines, not an untested cure. A work cap is not proof of convergence or plateau. |
| Directed geometry remained poor in some maps with favorable ordinary losses or median residuals | Separate probability fitting, derivative geometry and finite HMC information. Sampling only from a collapsed map cannot check missing regions. |
| Existing fixtures and decision code use known centers, signs, valley intervals and reference banks | Parameterize the target and remove evaluation truth from the practical training, initialization and selection path before claiming transfer. |
| Prior experiments used different targets, budgets, initialization and levels of source correspondence | The quoted historical method table is an inventory, not a controlled comparison. Run the explicit matrix in Section 3.1; retain scoped failures without excluding whole method families. |

Evidence: [mixture diagnosis](bayesfilter-unwarped-mixture-failure-results-2026-09-30.md),
[causal repair](bayesfilter-neutra-causal-repair-results-2026-10-01.md),
[controlled repair](bayesfilter-neutra-controlled-repair-results-2026-10-02.md),
[FAB assessment](bayesfilter-fab-monograph-coverage-result-2026-09-28.md).
The earlier September 30 pass counts alone are an incomplete current history.
Likewise, the earlier statement that `gabrie_discovered` was never executed is
stale: the [September 30 pipeline audit](bayesfilter-neutra-warm-start-pipeline-audit-2026-09-30.md)
records executions on Gaussian and mixture fixtures, with truth-assisted
downstream starts. That establishes neither autonomous discovery nor full
upstream-controller equivalence. FAB's without-replacement replay bias is
also acknowledged in its paper; it is not by itself proof of an incorrect
TensorFlow port. Source correspondence and correctness of a claimed gradient
are separate questions.

There is also a dimension distinction. In the inspected complexity adapter,
q=20 sets latent and hidden state dimensions to 20. The estimated parameter
vector has four entries, and the current horizon is 30:
ssl_lstm_complexity_target_tf.py, make_complexity_config and FREE_NAMES;
ssl_lstm_complexity_batched_target_tf.py, parameter_dim. Record state dimension,
hidden dimension, parameter dimension and horizon separately. A 20-parameter
experiment is an additional scope, not something established by the name q20.

## 2. Research intent and evidence contract

| Role | Definition |
|---|---|
| Main question | For each of FAB, Gabrié, annealed AIS, annealed SMC, AFT and CRAFT, does a frozen procedure provide valid sampling/training, useful canonical IAF maps and accurate downstream estimates on simple and then unseen targets, within its declared total budget? |
| Mechanisms | Each method's published sampling/training mechanism; a controlled posterior-teacher-to-IAF experiment; protected refinement; separately labeled optional repairs; exact corrected final sampling |
| Baseline | Direct canonical RKL, common-IAF forward-only and FKL-then-RKL training, existing combined FKL/RKL, independently calibrated AIS/SMC and the cheap controls below; original fixed mixture remains a regression case |
| Exact controls | Analytic Gaussian/transport where available, independent exact target draws, exact mixture identities; exact-teacher training only as a diagnostic control |
| Primary success | Fresh downstream posterior estimates meet frozen numerical, convergence, precision and reference requirements within the complete procedure's budget |
| Promotion veto | Invalid target/score/weight, truth leakage, missing required uncertainty, supported material component/shape error, or failed final numerical/posterior checks |
| Repair trigger | Missing discovery, poor overlap, inaccurate teacher, noisy/stalled fitting, refinement damage, difficult transformed geometry, bounded tuning failure or insufficient information |
| Continuation veto | Invalid shared mathematics/harness/reference, corrupted evidence, unavailable required computation, or exhausted total allocation; blocks the affected dependent work |
| Explanatory diagnostics | Loss, finite score residuals, acceptance, ESS/CESS, ancestry, gradients, clipping and time. These locate failure; they are not global coverage proofs. |
| Not concluded | Exhaustive discovery for arbitrary targets, optimizer convergence to a global solution, uniform whitening, method superiority, or true state-space posterior correctness from a deterministic approximate likelihood |

A candidate rejection must name the failed stage and the next applicable
repair. Do not turn a failed fit into rejection of SMC, or a failed teacher
into rejection of the IAF. A later repair designed for that failure remains
eligible under the same total budget.

The cheap sanity controls are constructed explicitly: a single Gaussian fitted
to the practical bank tests whether the neural map learned more than moments;
a discovered-center Laplace/defensive mixture tests whether expensive learning
has lost coverage already present in a cheap proposal; corrected weighted
teacher estimates test whether fitting destroys useful information; and exact
iid/analytic controls test whether the demanded accuracy is attainable.
Evaluate them separately for balanced/unequal weights, comparable/unequal
widths and moderate/large separation. Statistically supported underperformance
blocks a usefulness or efficiency claim. These controls are not a new
reason to withhold an otherwise qualified q20 estimate. They also do not
replace the mandatory six-method study or its untouched generalization tests.

## 3. Published methods, adaptations and new features

The technical sections and the relevant local author-source operations below
were inspected for this proposal. Operation checks are narrower than full
source equivalence. References and exact local anchors are in Section 12.

| Mechanism | Literature basis | Status and restriction |
|---|---|---|
| Canonical IAF and exact latent HMC target | Hoffman et al. (2019), Section 4.1.1 and author code | Retain masks, ELU, reversal, free scale bias outside the cap and the shared authority. Source architecture does not guarantee a trainable mixture map. |
| FAB | Midgley et al. (2023), Sections 3.1–3.2, Algorithm 1 and Appendix A | Mandatory alpha-two arm on eligible targets, with fresh-AIS and replay checks. Preserve the rejected q20 configuration; normalized finite weights do not prove auxiliary-target integrability. |
| Posterior annealed AIS | Neal (2001), Section 2, equations (3)–(11) | Mandatory posterior-target arm without resampling, with properly calibrated bridges and invariant mutation. It is distinct from AIS targeting FAB's auxiliary density. |
| Posterior annealed SMC | Del Moral et al. (2006); Dai et al. (2022), Sections 2.1–2.4 | Mandatory posterior-target arm with corrected weighting, resampling and mutation. Earlier two-known-region results are not general qualification. |
| Corrected global/local exploration | Gabrié et al. (2022), Sections III–IV and Appendix G | Mandatory adaptive-sampling/training arm and frozen-sampler assessment. Preserve MH correction and explicit initialization limits; the method does not promise exhaustive mode discovery. |
| Stagewise learned transport | Arbel et al. (2021), AFT Algorithm 1, Section 3.3 and Appendix F | Mandatory AFT arm with disjoint training, validation and evaluation populations, exact flow weights and fresh final sampling. |
| Repeated transport learning | Matthews et al. (2022), CRAFT Section 2.3, equations (5)–(8), Algorithms 2–3 | Mandatory CRAFT arm with repeated fresh passes, checked stage-update order and frozen-map deployment. No q20 transfer is presumed. |
| Concurrent likelihood and energy losses | Noé et al. (2019), Methods equation (9), supplementary Figure S2 | Retain as a source-based IAF adaptation and matched baseline. Their energy clipping is not imported; it would change our target objective. |
| Overlapping biased windows | Thiede et al. (2016), EMUS equations (14)–(15), Theorem VII.4 | Optional deliberate rare-region teacher with corrected weights, overlap and window-mixing checks. Its asymptotic assumptions are not finite-run certificates. |
| Path-gradient estimator | Roeder et al. (2017); Vaitl et al. (2024), Proposition 3.2, equation (16), Appendix B.1 | Retain the checked shared estimator or its checked reparameterized comparator. The IAF recursion is quadratic in dimension; no coupling-flow linear-cost claim. |
| Data-derived bridge probes, protected checkpoint decisions, and targeted latent-score loss | Local additions | Define the measure/objective below, check total derivatives, preserve no-change controls, and require independent downstream evidence. No literature guarantee is asserted for this combination. |

Complete the selected author-controller comparisons before claiming faithful
procedural replication. For Gabrié this includes loss-jump retries, persistent
walkers, scalar-loss constants relevant to retries, effective MALA time-step
conventions and RNG consumption. For AFT/CRAFT it includes independent
populations, gradient detachment, pre-update selection, stage state and update
ordering. A TensorFlow adaptation using canonical IAF differs from an author's
RealNVP or JAX experiment even when selected numerical operations agree.

FAB's old q20 rejection does not exclude FAB on eligible simpler targets.
A defensive variant is a separately labeled extension requiring an
integrability argument and objective/gradient audit; it cannot silently
replace the native arm or the final canonical IAF. SMC² is not substituted:
it requires a valid extended target and an unbiased nonnegative likelihood
estimator, while the current deterministic UKF likelihood defines a different
inference problem.

### 3.1 Mandatory six-method test matrix

Every row below is required. These are six distinct method families, not six
names for a common SMC implementation. A shared primitive is welcome, but the
actual consumer must exercise the declared objective, transition sequence,
resampling policy and training controller. "Native" below means the method's
own objective/controller; using the canonical IAF in place of an author's
flow family remains an explicit adaptation, not a full reproduction of the
published experiment.

| Method | Required implementation and mathematics checks | Native scientific assessment | Posterior bank and canonical IAF handoff |
|---|---|---|---|
| FAB | Auxiliary-tail eligibility; alpha-two gradient with detached AIS samples/weights; fresh and replay loss scaling; stale-weight correction, clipping and replacement law; complete state/RNG resume | Fresh-AIS training without replay, then replay variants on an eligible Gaussian and mixtures. Compare finite-bank gradients with their exact finite sums; assess the learned density and its missing-region behavior | Test the directly trained canonical IAF. Separately create a fresh weighted auxiliary population at a frozen map and apply the posterior correction in Section 4.0 before the common-IAF experiment. A new posterior AIS pass is a labeled FAB-plus-AIS variant, not free FAB output |
| Gabrié | Global-MH and local-MALA ratios; time-step convention; persistent walkers; retry/loss-jump controller; constants affecting that controller; checkpoint and RNG restoration | Assess both coupled training and independent runs of the frozen sampler, with correlated-chain uncertainty and independent starts | Evaluate its directly trained canonical IAF and, separately, train the common IAF from qualified frozen-sampler draws. Adaptive training walkers are not silently relabeled as stationary posterior draws |
| Annealed AIS | Support, stage ordering and telescoping weights; invariant mutation; no resampling; pilot/frozen distinction; known normalizer/expectation checks | Fresh independent weighted paths, posterior quantities and weight-concentration sensitivity under an independently calibrated schedule | Correctly weighted terminal posterior bank feeds the common IAF; the bank is not iid posterior data |
| Annealed SMC | Same posterior bridge as the paired AIS control; weight/reset/normalizer bookkeeping; unbiased resampling offspring; mutation and ancestry; pilot/frozen distinction | Independent complete populations, posterior quantities, ancestry and population/mutation sensitivity | Correctly weighted final populations feed the common IAF; uncertainty uses population replication, not resampled copies as independent observations |
| AFT | Stage Jacobian weights and mutation; training/validation/evaluation separation; pre-update selection and full stage state, checked against author code | Native stage training followed by fresh corrected evaluation populations; identity-map reduction to ordinary SMC is a required control | Qualified final posterior populations feed the common IAF. The stagewise transport plus resampling/mutation is not a single canonical NeuTra map |
| CRAFT | Detached particles; stage-gradient objective; pass-before-update order; fresh particle replenishment; optimizer/stage state; frozen deployment | Repeated-pass training followed by independent corrected deployment populations; frozen identity-map reduction and comparison with AFT at matched total cost | Qualified final posterior populations feed the common IAF. An AFT run renamed CRAFT, or an unchecked composition of stage maps, does not satisfy this row |

For each method, maintain separate results for (i) implementation/mathematical
validity, (ii) its native sampler or learned density, (iii) the controlled
posterior-bank-to-IAF experiment, and (iv) fresh downstream NeuTra estimation.
Passing one column does not pass the others. A valid native posterior estimator
can still fail as a teacher at the available sample budget; a qualified
teacher can still produce a poor IAF; a poor IAF may still allow accurate but
expensive corrected HMC. Preserve these outcomes instead of issuing one
undifferentiated "method failed" verdict.

The common fitting experiment uses the same canonical architecture,
initialization distribution, target-specific calibration rule, student
optimization budget and heldout assessment for all qualified posterior banks.
Check the exact normalized bank weights actually passed to training. Matched
branches retain the forward-only map and continue a copy by RKL; the existing
combined FKL/RKL recipe is a separate development baseline. For directly
trained FAB/Gabrié maps, retain the native map before any optional RKL as well.
Evaluate before and after refinement so loss of coverage cannot be hidden.

The native and common-fitting results answer different questions. FAB/Gabrié
can help by training a map directly; AIS/SMC/AFT/CRAFT can help by producing a
better weighted teacher. Distillation into a new IAF is not evidence that the
native direct map had the same quality. Conversely, native training budgets
must not be omitted from the common-fitting cost. Choose one complete
pipeline per method on development/validation targets before confirmation;
the choice and any automatic repair rules are part of that method's frozen
procedure. Claims about additional pipelines require their own replication
and multiplicity allowance.

### 3.2 Shared targets, controls and calibration

The compulsory target ladder is:

| Target set | Required methods and purpose | Advancement rule |
|---|---|---|
| Analytic Gaussian and method-specific finite controls | All six: exact weights, moments, gradients where available, mutation identities and restored-state behavior | Invalid shared target/reference blocks dependent work; an invalid method implementation blocks that method until repaired |
| Fixed unwarped mixture and fixed warped mixture | All six: native assessment, common-IAF fitting, pre/post-refinement checks and fresh downstream inference | Each method gets its own calibrated bounded attempt and prescribed repairs; success of another row never cancels it |
| Development and validation random two-/three-center mixtures, including boundary cases | All six eligible methods: calibrate generic rules without exposing final targets | Freeze the entire per-method procedure and budget before final target generation |
| Untouched random two-/three-center families | Every method passing the applicable simple-case contract | Per-method, per-family outcomes and uncertainty; use the same drawn targets across methods and fresh algorithmic streams |
| Higher dimension, anisotropy, nonlinear deformation, reduced state-space models, then q20 | Every method still viable at the preceding relevant scope | Preserve a row for each method at each rung, including explicit prerequisite failures; do not generalize only the first successful method |

Development uses three independent algorithmic streams per fixed method-target
cell as a proposed inexpensive restart screen, not an estimate of reliability
or a basis for declaring superiority. Six methods times the Gaussian and two
fixed mixtures makes 18 core cells, with 54 such initial replicates if all are
eligible. Native/replay/refinement ablations are additional priced jobs, not
free replicates. P6 adds untouched confirmation streams; it does not reuse
these development trials. Randomized confirmation sizes are in Section 6.

Use the same density/score, computational precision, practical information,
reference estimands and accuracy tolerances in a matched cell. Give each method
the same declared development-resource ceiling and the same end-to-end
delivery ceiling within a target family, with method-specific tuning inside
those ceilings. Do not impose a sparse AIS ladder or an SMC setting merely
because a previous campaign used it. If an unequal resource allocation is
scientifically needed, report it and abandon an equal-cost comparison for
that contrast. Source-default and independently tuned profiles are separate.

Common initialization means the same allowed information, not incorrectly
claiming identical initial densities. Posterior AIS/SMC/AFT/CRAFT can use the
same normalized practical proposal r. FAB starts from its current q, and
Gabrié has both a walker distribution and a map initialization. Record both;
charge any fitted initialization to the respective method. The practical mode
search is a separately identified shared heuristic, not attributed to Gabrié
as an exhaustive mode-finding algorithm. Supplied-mode and intentionally
missing-region controls are labeled fault-isolation tests and cannot support
autonomous discovery claims.

The controlled AIS/SMC ablation uses the same frozen ladder, initial proposal,
particle count and mutation kernel, changing resampling only. AFT/CRAFT use
an identity-map control on a compatible bridge. These isolate mechanisms;
each method also receives its own calibration for the complete delivery test.
Method-specific calibration must cover at least:

- FAB: tail eligibility, intermediate distributions, mutation effort, fresh
  batch size, replay capacity/refresh, replacement law, stale-weight clipping,
  gradient scale and optimizer sensitivity.
- Gabrié: local step size, local/global move cadence, walker count, training
  cadence, retry rule and independent frozen-sampler equilibration.
- AIS/SMC: bridge spacing, particles/paths, corrected mutation step size and
  trajectory effort; SMC also resampling trigger and offspring scheme.
- AFT: bridge/particle/mutation controls plus per-stage learning budget,
  validation selection and map capacity.
- CRAFT: bridge/particle/mutation controls plus number of fresh passes,
  per-stage learning rates, update ordering and heldout deployment.

The smallest Gaussian value/gradient/invariance check precedes costlier
calibration. Use finite parameter brackets with provenance and measured costs;
freeze brackets, selections and work/repair caps in P2. An improving fit at
its cap remains budget-limited. Local geometry loss, EMUS or a defensive FAB
extension is a separately labeled repair, with an unchanged-method control;
none can silently replace a required native-method row.

### 3.3 Assessment, uncertainty and completion

Teacher accuracy uses independent evaluations of component responsibilities,
within-component moments, physical-region probabilities, tails and projected
shape against the exact evaluator where available. Use confidence intervals
on discrepancies inside predeclared scientific tolerances, not merely a test
that fails to reject equality. Validate both fresh independent populations and
the particular weighted bank delivered to training. Weight ESS, cESS, ancestry
and acceptance locate failure and guide development; they cannot certify
coverage. Known normalizers are additional implementation checks where the
method actually estimates them; do not invent a normalizer estimate for
Gabrié's sampler.

IAF assessment includes the owner-required 1,000-point Gaussian score probe,
independent target-based checks, inverse/Jacobian validity and pre/post-RKL
coverage. The probe alone cannot promote a map. Downstream assessment uses
fresh map-specific public tuning and the shared sequential controller, with
the same posterior accuracy, precision, region/shape and numerical checks.
Plain NeuTra and the tempered ensemble are recorded separately. If both are
allowed in a delivery policy, its switch rule and combined cost are frozen
before confirmation; choosing whichever happens to pass afterward is invalid.

The primary endpoint is delivery of the declared posterior quantities within
the full procedure's budget. Report accuracy and cost also for methods that
fail delivery. Comparison is systematic without requiring a winner: paired
targets support contrasts, but descriptive ESS, residual or runtime tables do
not establish superiority. If a ranking is requested later, predeclare the
contrast, meaningful effect, target-level uncertainty and correction across
the 15 pairwise method contrasts before looking at confirmation data. Do not
rank only the successful subset; treat capped failures explicitly. The
per-method reliability analysis in Section 6 does not itself test differences
between methods.

The master must emit a row for every planned
method × target/scope × stage × replicate × variant, including unexecuted
dependent stages. Allowed outcomes distinguish `passed`, `implementation_invalid`,
`mathematically_inapplicable`, `sampling_failed`, `fitting_failed`,
`refinement_failed`, `downstream_failed`, `budget_limited`,
`prerequisite_failed` and `not_run`. A missing implementation or absent tail
argument is unresolved work, not a mathematical rejection of a method.
`mathematically_inapplicable` requires the stated assumptions and a checked
argument; it does not justify evaluating a different target silently.

An affected downstream stage may be blocked without forcing an invalid or
unqualified run. Independent method rows continue under the remaining budget.
The report is incomplete if required feasible cells are `not_run`, or if a
known localized implementation failure has neither been repaired within its
allowance nor reported as an unresolved engineering blocker. A documented
valid candidate failure is a completed test, not a passing method. The study
can close with negative findings; it does not require all six methods to work.
Passing
one row, an implementation parity suite, or the original fixed mixture never
closes the six-method study.

On confirmation targets, the practical controller sees only density/score
access and its frozen internal diagnostics. The external evaluator may reject
a result, but cannot supply true component weights, select a checkpoint,
trigger a target-specific repair or replace a teacher within a supposedly
autonomous trial. Diagnostic experiments using exact teachers remain separate.
An exposed confirmation failure can motivate a new development revision and
fresh targets, not a hidden retry until the reference agrees.

## 4. Generic procedure and its mathematical justification

Let gamma(x) be the fixed unnormalized density in the sampled coordinates,
0<Z=integral gamma<infinity, p=gamma/Z, and phi the standard normal base.
Any physical-parameter chart contributes its Jacobian to gamma once. It must
not be omitted or counted again inside the learned map.

### 4.0 FAB's objective and the posterior handoff

For a normalized q_theta with target support, write

\[
C_\theta=\int\frac{\gamma(x)^2}{q_\theta(x)}\,dx,\qquad
g_\theta(x)=\frac{\gamma(x)^2}{C_\theta q_\theta(x)}.
\]

This defines a sampling density only when 0<C_theta<infinity. If
differentiation under the integral is justified, then

\[
\nabla_\theta\log C_\theta
=-E_{g_\theta}[\nabla_\theta\log q_\theta(X)].
\]

Thus the expectation uses the auxiliary density g_theta, not p. Fresh AIS
samples and weights are detached when estimating this gradient. Unnormalized
AIS gradients and finite self-normalized estimates have different bias and
scaling properties; the latter are not unbiased simply because AIS has the
correct extended-target identity. The author code additionally averages its
softmax-weighted loss, adding 1/B relative to the paper's sum. Check both the
literal source scale and the intended objective scale before matching learning
rates or changing batch size.

A cheap analytic applicability test uses p=N(mu,Sigma), q=N(m,C). The
quadratic precision of p^2/q is 2*Sigma^{-1}-C^{-1}, so its integral is finite
exactly when that matrix is positive definite. For an isotropic Gaussian
mixture and q=N(m,tau^2 I), expansion into positive component-pair terms gives
the condition tau^2>max_k(v_k)/2: the diagonal terms require it, and the cross
precisions are averages of the diagonal precisions. Mean separation changes
the finite normalizer and sampling difficulty, not this condition. These
are exact fixtures, not certificates for a trained nonlinear IAF.

Include both eligible and ineligible Gaussian fixtures to test rejection of
an undefined auxiliary target. For the requested mixtures, obtain a tail
argument for the actual IAF using declared target-class bounds or a justified
density bound, and recheck its scope after map updates. No hidden centers,
weights or exact target samples may enter practical calibration. A Gaussian
initialization argument alone cannot certify later nonlinear maps. If a
bound is unavailable, label applicability unresolved and investigate it;
finite weights do not settle integrability. This difficulty belongs in the
method's recorded outcome, not in a silent omission of FAB from the study.

For a frozen qualified map, let (X_i,A_i) be a fresh correctly weighted AIS
population targeting the unnormalized density gamma^2/q_theta. Multiplying
by q_theta/gamma converts its weighted target to gamma:

\[
\widehat E_p[f]=
\frac{\sum_i A_i\,q_\theta(X_i)/\gamma(X_i)\,f(X_i)}
     {\sum_i A_i\,q_\theta(X_i)/\gamma(X_i)}.
\]

This is the common-teacher handoff for FAB, subject to support, weight accuracy
and finite-sample uncertainty. It is not a variance guarantee. Using A_i alone
would train forward KL against g_theta, which is wrong relative to a posterior
FKL claim. Do not apply this formula to stale replay without its generating
density/corrections, or count a fresh correction pass as costless. An
alternative fresh posterior-AIS pass targets gamma directly, as suggested
in the FAB paper, and is explicitly recorded as a composed procedure.

Fresh training without replay is the first native FAB control. The replay
study then compares the exact frozen-bank gradient, categorical draws with
replacement and the author's without-replacement profile. The paper itself
acknowledges the latter's bias. With-replacement sampling removes that specific
finite-bank sampling mismatch; it does not remove finite AIS, normalization,
stale-state or clipping errors. Changing clipping or the replay law is
recorded explicitly. Source-faithful reproduction of a heuristic remains a
heuristic, and the rejected q20 auxiliary target remains rejected until its
own mathematical problem is repaired.

### 4.1 Discovery, initialization and posterior annealing

Use dispersed starts from a declared broad distribution, local optimization
with score/curvature checks, and duplication checks in a documented metric.
Mode heights and optimizer hit counts are not mode probabilities. Retain
independent explorations at lower temperatures and outside discovered regions.
Fit a normalized initial proposal from discovered locations/local shapes,
with an explicitly evaluable broad defensive component. Locations and shapes
come from the pilot, not from benchmark truth.

This discovery is a heuristic. If one iid start reaches basin j with
probability s_j, n independent starts miss it with probability (1-s_j)^n.
If there were K basins and a known lower bound s_min, the union bound would
give P(any missed)<=K exp(-n*s_min). Neither K nor s_min is known for q20.
More starts and independent agreement therefore provide evidence, not an
exhaustive-discovery theorem.

With normalized proposal r of full target support, define

\[
\gamma_\beta(x)=r(x)^{1-\beta}\gamma(x)^\beta,\qquad 0\leq\beta\leq1.
\]

Hölder's inequality gives

\[
\int r^{1-\beta}\gamma^\beta
\leq\left(\int r\right)^{1-\beta}\left(\int\gamma\right)^\beta=Z^\beta.
\]

Thus the bridges are normalizable under the stated target/support assumptions.
This avoids the particular inverse-proposal tail factor in FAB's alpha-two
target; it does not imply small importance-weight variance or mode discovery.
Likelihood-only annealing from the actual prior is a separate valid path,
with its definition recorded explicitly.

For particles at beta, a temperature increment Delta has potential
G_i=exp(Delta*(log gamma(x_i)-log r(x_i))). Update W_i proportionally to
W_i*G_i. For normalized current weights, the relative conditional ESS is

\[
cESS=\frac{(\sum_iW_iG_i)^2}{\sum_iW_iG_i^2}.
\]

It lies in (0,1] and differs from cumulative weight ESS. Use it to nominate
temperature spacing, resampling to address weight concentration, and corrected
mutation to address movement. High cESS in a population missing a mode cannot
certify coverage.

An adaptive pilot may choose the schedule, proposal and mutation settings.
Freeze these for new independent teacher populations. Conditioned on that
frozen pilot, the subsequent algorithm has a specified target and transition
sequence. Claims about normalizer unbiasedness require the actual SMC
assumptions; self-normalized expectation estimates are generally biased at
finite particle counts. Independent replications and particle/mutation
sensitivity remain necessary.

### 4.2 Corrected exploration and flow-assisted bridges

For a frozen normalized global proposal q_g, the independence acceptance is

\[
\alpha(x,y)=\min\{1,\gamma(y)q_g(x)/[\gamma(x)q_g(y)]\}.
\]

The accepted off-diagonal flux equals
min{p(x)q_g(y),p(y)q_g(x)}, so detailed balance holds. A corrected local
kernel followed by this kernel preserves p; their composition need not be
reversible. During adaptation, this fixed-kernel argument alone does not
justify treating training walkers as stationary draws. Freeze and assess
fresh runs, retaining within-chain and between-run uncertainty.

For example q_g=epsilon*r+(1-epsilon)*sum_j a_j*q_j with epsilon>0
protects proposal support. The weights a_j are proposal allocations, not
asserted posterior masses. If p<=M*r everywhere, then q_g>=epsilon*p/M
and the independence kernel minorizes p by epsilon/M. This follows because
both terms in min{q_g(y),p(y)q_g(x)/p(x)} are at least epsilon*p(y)/M.
Full support or a Student tail alone does not establish the finite M bound.
No uniform mixing claim for q20 follows without it.

For a frozen AFT/CRAFT stage map S_k, y=S_k(x), the incremental weight is

\[
G_k(x)=
\frac{\gamma_{\beta_k}(S_k(x))|\det J_{S_k}(x)|}
     {\gamma_{\beta_{k-1}}(x)}.
\]

Changing variables proves
E_{p_{k-1}}[G_k(X)f(S_k(X))]
=(Z_k/Z_{k-1})E_{p_k}[f]. This explains the correction; it does not make
a finite learned particle cloud exact. AFT's practical independent populations
and CRAFT's fresh passes/frozen evaluation address different reuse problems.
Do not differentiate blindly through accept/reject or resampling decisions
and call the result the source gradient.

Stage-map composition is a teacher mechanism here. The final NeuTra map still
uses the canonical IAF. Distilling a good teacher can fail, as the oracle
experiments already demonstrate; AFT/CRAFT are not substitutes for diagnosing
that failure.

### 4.3 Deliberate rare-region sampling with the original measure preserved

Use discovered regions, pilot energy levels and pilot transition failures to
identify neglected strata. No fixed x_1 sign, known mixture label or analytic
valley boundary may enter the practical procedure. Freeze the stratum
definition for a training block and retain a broad exploration component.

For a partition A_j, write m_j=p(A_j), p_j=p(.|A_j). Then

\[
E_p[f]=\sum_jm_jE_{p_j}[f].
\]

If a minibatch selects j with probability a_j>0 and X~p_j, the contribution
(m_j/a_j)f(X) has expectation E_p[f]. Equal stratum allocation without this
correction trains a different distribution. Estimated m_j and approximate
conditional samples instead define an approximate teacher; preserve their
uncertainty and test the actual pooled bank used by training.

With independent conditional draws, the stratified estimator has variance
sum_j m_j^2*sigma_j^2/n_j, where sigma_j^2=Var_{p_j}(f). Minimizing that
variance at fixed cost sum_j c_j*n_j=B gives n_j proportional to
m_j*sigma_j/sqrt(c_j): differentiate the Lagrangian with respect to n_j.
This is a principled allocation pilot, not a universal training batch rule.
For gradient vectors, a declared trace-of-covariance criterion gives the
corresponding version. Autocorrelated windows require long-run covariance
and cost estimates; uncertain pilot estimates and representation floors
must remain explicit. Oversampling a rare stratum retains the original
expectation only with the correction above.

Pool R independent normalized banks using weights W_ri/R when equal
replication weighting is intended. Save the replication labels and inspect
the pooled bank itself. This pooling identity is not a finite-sample
unbiasedness theorem for self-normalized teacher estimates.

For a finite normalized weighted bank (x_i,W_i), sample index i with
replacement using known probabilities s_i>0. The forward contribution is
-W_i*log q_theta(x_i)/s_i, averaged over the batch. Its conditional
expectation is exactly -sum_i W_i log q_theta(x_i). Do not repeat the
without-replacement weighted replay shortcut: that requires appropriate
inclusion-probability corrections.

Where simple normalized proposals cannot populate an important region,
overlapping windows provide a source-based alternative. With psi_j>=0,
S=sum_j psi_j>0 on target support, p_j proportional to psi_j*p and relative
normalizers z_j, set F_ij=E_{p_i}[psi_j/S]. The EMUS identities are

\[
z^\mathsf{T}F=z^\mathsf{T},\qquad
E_p[f]=\frac{\sum_jz_jE_{p_j}[f/S]}
             {\sum_jz_jE_{p_j}[1/S]}.
\]

They follow by substituting the window densities and summing psi_j/S=1.
Connected overlap and adequately sampled windows are necessary. The paper's
CLT assumes within-window CLTs, independent windows and positive limiting
allocations. A disconnected overlap graph or trapped windows cannot be fixed
by merely solving the eigenvector equation more accurately.

The generic construction of windows from pilot geometry is a local heuristic,
not something the EMUS theorem supplies. Freeze it before independent
evaluation. Multilevel splitting remains an optional event-estimation tool,
with its own conditional-sampling and reaction-coordinate assumptions; it is
not another mandatory full training arm.

### 4.4 Fitting and refinement

Let T_theta be the canonical invertible map and q_theta=T_theta#phi.
Use the exact target in

\[
L_F=-E_p\log q_\theta(X),\qquad
L_R=E_\phi[-\log\gamma(T_\theta(Z))-\log|\det J_{T_\theta}(Z)|].
\]

These equal their respective KL divergences up to theta-independent
constants. With an approximate teacher, L_F is replaced by its explicitly
weighted empirical cross-entropy, not an exact posterior expectation.

The source-based combined baseline is L_R+lambda*L_F, lambda>0. For any
fixed partition with target masses m and proposal masses v,

\[
D_{\rm KL}(p\Vert q)
=D_{\rm KL}(m\Vert v)+
 \sum_jm_jD_{\rm KL}(p(.|A_j)\Vert q(.|A_j)).
\]

Substitute p=m_j*p_j and q=v_j*q_j on A_j to derive the decomposition.
Losing a material region sends the forward cost to infinity as v_j tends
to zero. By contrast, for p=w_j*p_j+other nonnegative components,
p>=w_j*p_j gives D_KL(p_j||p)<=-log w_j. These facts explain the reason
to preserve forward coverage, but do not prove successful nonconvex training
or accurate relative probabilities in extremely rare regions.

The previous combined-objective campaign is the baseline. A new attempt
must change one diagnosed cause: teacher quality, conditioning/noise,
continuation budget, initialization, canonical capacity, or geometry.
Restore map, Adam state, lifetime steps, sampler state and RNG position for
true continuation. Distinguish a reset intervention explicitly. Match
optimizer resets when isolating objective changes.

Preserve eligible earlier maps. RKL is optional: a forward-only map that
passes downstream checks is sufficient. The practical checkpoint rule uses
independent estimated-teacher diagnostics and sampler evidence available on
unknown targets; benchmark truth is reserved for external assessment.
It must not use the old known-true forward KL or known component masses
inside the practical selection path.

### 4.5 Conditional new feature: training the geometry that ordinary losses miss

This is the principal new objective to investigate when a qualified teacher,
checked derivatives and development evidence show density fitting progressing
without resolving the transformed-geometry failures. It has not been executed
in the controlled campaign. A proved optimizer plateau is not a prerequisite
for a bounded matched test: if both fits remain work-limited, the experiment
answers a finite-budget repair question, not whether density training could
ever solve the problem alone. Preserve that alternative explanation.

The exact pullback density and its Gaussian score residual are

\[
p_{\theta,z}(z)=p(T_\theta(z))|\det J_{T_\theta}(z)|,\quad
r_\theta(z)=J_{T_\theta}(z)^\mathsf{T}\nabla\log\gamma(T_\theta(z))
 +\nabla_z\log|\det J_{T_\theta}(z)|+z.
\]

Construct a fixed physical probe distribution nu from independent pilot
information: diffuse proposal draws, discovered local neighborhoods, and
Gaussian tubes around several pilot paths between regions. Tubes must have
positive width in every dimension. A straight line alone cannot identify
off-axis defects. A normalized finite mixture of such Gaussian/Student
components supplies samples and an evaluable density without knowing any
posterior normalizer. Choose endpoints from discovery, not true centers.
Check that the proposed squared-residual expectation is finite under the
chosen tails; full support alone does not establish integrability. Begin
with finite Gaussian-component proposals on the analytic controls. A
finite-bank objective can be studied explicitly, but its finite evaluations
cannot establish the population zero-loss premise. At a target with an
unresolved numerical status domain, invalid directed probes trigger a
target/derivative investigation; do not drop them and silently redefine nu.

With nu frozen independently of the current theta, define

\[
G(\theta)=E_{X\sim\nu}
  [\|r_\theta(T_\theta^{-1}(X))\|^2]/D,\qquad
J=L_R+\lambda L_F+\eta G,\quad\eta\geq0.
\]

This deliberately emphasizes the chosen physical probe measure. It is not
posterior-weighted score matching unless nu=p. The eta=0 matched control is
essential. Underestimated regions in the fitted proposal need not receive
negligible training allocation under nu.

There is a precise zero-loss statement. Assume p is positive on all of R^D,
T_theta is a diffeomorphism from R^D onto R^D, nu has positive density
everywhere, the residual is continuous, G is well-defined and G(theta)=0.
Then r_theta is zero everywhere: a nonzero continuous residual
would have positive integral in an open neighborhood. Thus
grad[log p_theta,z+||z||^2/2]=0. Constancy and normalization imply
p_theta,z=phi. If the map family contains such a map, all nonnegative KL
and G terms have a common zero. This is not an existence theorem for the
finite canonical IAF, a finite-probe certificate, or a guarantee that Adam
will find the map. The full-support premise matters: a truncated or disconnected
numerical status domain does not satisfy this Gaussianization argument.
In a restricted family, eta changes the best approximation and may trade
density fit for geometry. It changes training, not the final sampler's gamma.

The required derivative is total. Writing z_theta=T_theta^{-1}(X) and
R(theta,X)=r_theta(z_theta),

\[
\partial_\theta z_\theta
=-J_{T_\theta}(z_\theta)^{-1}
  \partial_\theta T_\theta(z_\theta),\qquad
\partial_\theta G=\frac2D E_\nu[R^\mathsf{T}\partial_\theta R].
\]

Interchanging differentiation and expectation requires the relevant local
integrability/dominating conditions; an empirical fixed-bank loss has its
ordinary finite-sum derivative. The chain rule in partial_theta R includes
the inverse derivative. Detaching
the current inverse changes the gradient. Because T_theta(z_theta)=X,
the physical target score at fixed X can be cached; this avoids a target
Hessian only when the implementation preserves that exact substitution and
the remaining map derivatives. Check it against total directional differences.
ELU is piecewise smooth: derivative checks must identify kinks and not
misinterpret a finite difference crossing one as a smooth-Hessian comparison.

For a convex latent region K where ||r||<=e, the fundamental theorem of
calculus along segments gives

\[
|h(z)-h(z')|\leq e\|z-z'\|,\qquad
h=\log(p_{\theta,z}/\phi).
\]

Consequently osc_K(h)<=e*diam(K). This is a conditional local relation between
score error and density-ratio variation. A finite grid does not establish the
uniform premise; controlling h also does not by itself bound all Hessians or
leapfrog errors. The downstream HMC test remains necessary.

Before any serious geometry training: verify value/inverse/Jacobian identities,
the total parameter gradient, mixed derivative cost, actual GPU/XLA support,
memory growth and single-batch execution. Calibrate eta on development
targets using gradient/update scales and fresh downstream checks. If the
pilot reveals a prohibitive cost or unstable derivative, record that branch
as ineligible and continue to the planned ensemble route.

### 4.6 Exact final sampling and the ensemble alternative

Freeze the map and run the public fixed-transport tuner with the complete
transformed value/score and identity latent mass. Change of variables makes
the target p_theta,z exact relative to the declared gamma even for an
imperfect map. The usual reversible volume-preserving leapfrog proposal,
momentum reversal and MH correction then preserve that target in exact
arithmetic under their assumptions. Numerical implementation and finite-time
convergence need their separate checks.

Consult the actual capability registry and public API, not a historical helper.
The current authority is tune_fixed_transport_hmc_kernel; a bare chain runner
cannot issue its tuning evidence. Starts are latent inverse images of the
recorded physical starts. Distinguish map-generated and independently
discovered dispersed starts, and check their roundtrips.

If a single map remains inadequate, use the authorized tempered NeuTra
ensemble with the same canonical maps. At physical coordinates for the
bridge in Section 4.1, a swap of x_i,x_j has acceptance

\[
\min\{1,\exp[(\beta_i-\beta_j)
 ((\log\gamma-\log r)(x_j)-(\log\gamma-\log r)(x_i))]\}.
\]

Implementing swaps in different learned coordinates must preserve the
corresponding physical states and Jacobians through the repository route.
Assess cold-chain quantities at beta=1, travel across temperatures, region
movement and independent populations. Replica round trips alone do not
establish cold posterior correctness. A mixture of local map densities is
an evaluable proposal, not a single globally invertible map.

## 5. Phased work with explicit completion conditions

| Phase | Work and smallest discriminating check | Completion and failure response |
|---|---|---|
| P0 — recover and bind scope | Inventory all six methods, actual target/chart, dimensions, precision, previous controls and live resource ledger. Create the complete method/target/stage matrix before running it. | Every method has source, implementation and prior-evidence status. Do not rerun successful evidence whose exact scope is unchanged. |
| P1 — generic interfaces and mechanics | Parameterize targets and separate evaluator truth. Check each method's actual consumer, objective, weights, transition order, derivatives, resume and failure behavior against Section 3.1. | Shared invalidity blocks dependent work; a method-local failure blocks that row. Missing implementation is repaired or explicitly unresolved, never a scientific rejection. |
| P2 — controls, calibration and pricing | Gaussian and exact controls for all six; eligible/ineligible FAB fixtures; matched AIS/SMC and identity-flow controls; measured end-to-end costs and bounded method-specific tuning. | Freeze brackets, budgets, replication, estimands, tolerances and repairs. Reserve the required matrix and confirmation costs before long development. |
| P3 — mandatory native-method study | Execute FAB, Gabrié, AIS, SMC, AFT and CRAFT on the fixed unwarped and warped mixtures. Preserve each native objective/controller and independently assess its sampler or density. | Record all six outcomes with uncertainty. Each gets its declared calibration/repair allowance; another method's success is not a stop condition. |
| P3a — posterior teacher assessment | For all applicable methods, generate fresh independent posterior banks using the correct handoff, including FAB's auxiliary-to-posterior correction and frozen Gabrié sampling. | Assess both replicated populations and the bank actually used. A failed bank blocks that fitting branch and triggers a diagnosed sampling repair. |
| P4 — common IAF and native-map study | Qualified banks versus exact-teacher control, with matched canonical forward-only and FKL-then-RKL branches; combined-loss development control; native FAB/Gabrié maps retained separately. | Localize teacher, fitting and refinement failures. Use 1,000-point plus target-based checks and downstream evidence; improving-at-cap is budget-limited. |
| P5 — diagnosed optional repairs | Use geometry loss, EMUS, longer continuation or a justified defensive FAB variant only with their own validity checks, unchanged-method controls and bounded allowances. | Label the changed procedure; keep original failures. Optional repairs cannot displace untested mandatory methods or silently redefine a native result. |
| P6 — per-method simple-case confirmation | Freeze one complete procedure per method; fresh fixed-unwarped and warped-mixture trials with fresh final references and public tuning. | Each advancing method passes both simple targets under its frozen contract. Failed candidates return to development; unrelated rows continue. |
| P7 — required randomized generalization | Every qualifying method runs both owner-requested two-/three-center families on the same fresh target specifications and independent method streams. | Report per-method, per-family reliability, uncertainty, failures and total cost. No winner-only confirmation or replacement of difficult targets. |
| P8 — dimensional and geometric transfer | For each still-viable method, increase dimension and then covariance/curvature difficulty on new targets, one factor at a time. | Record every method at every rung, including prerequisite failures. Isotropic 20D success does not establish recurrence/likelihood transfer. |
| P9 — state-space transfer and q20 | Each still-viable procedure progresses through exact-likelihood small models, nonlinear low-complexity LSTM and the existing q20 target. | Separate sampler-target validity and likelihood approximation. A qualified estimate can be reported while other systematic-study rows remain open. |

P7 is required for every method that passes P6; it cannot be dropped after a
favorable plot or the first successful method. P8/P9 follow that method's
successful relevant generalization. A diagnostic exploratory attempt may be
recorded before its predecessor passes, but is not progression or promotion.
Native-method results remain reportable even when a later IAF or HMC stage
fails. Preserve the complete roster and explicit reasons for dependent work
that could not validly run.

For P6, propose three new independent procedure trials per method on each of
the two fixed mixtures, all meeting the frozen delivery criteria: 36 complete
trials if all six methods are eligible. This count is a convenience screen
for restart variability, not a reliability estimate; P7 supplies the larger
target-level assessment. Draw these streams only after each procedure is
frozen. A failed P6 trial is not replaced by a favorable rerun under the same
claimed confirmation.

## 6. The generalization experiment

The [random-mixture design](bayesfilter-neutra-random-mixture-generalization-design-2026-10-03.md)
defines the two added families. Begin in D=2 with

\[
p_\psi(x)=\sum_{k=1}^{K}w_k N(x;\mu_k,v_kI),\quad K=2,3.
\]

Proposed distances are U[6,10], component variances U[0.5,2], translations
with coordinates U[-2,2], and random orientations. For K=3, draw all three
side lengths independently; 10<2*6 guarantees a nondegenerate triangle.
Weights may be generated by w_k=0.1+(1-0.1*K)V_k,
V~Dirichlet(1,...,1). These are transparent bounded stress choices around the
old unit-scale example, not literature defaults. Their limitations include
the 0.1 minimum component mass and isotropic within-component geometry.

Keep the original distance-ten fixture as a regression. Include separate
development corner cases at the declared parameter bounds because random
draws rarely land exactly at boundaries. Do not reject and redraw targets
after seeing a difficult fit. Unequal widths/weights can shift density modes
away from component means or merge modes; the evaluator must distinguish
components from local maxima.

Use disjoint development targets, procedure-validation targets and final
confirmation targets, with separate numerical randomness. The practical
callable supplies dimension and log-density/score access, not true K,
centers, weights, exact samples or a target name exposing the answer.
True values belong to the evaluator. This is an ordinary software interface
and falsifying test boundary, not a new access-control system.

Train a new map for each method and target. Freeze how capacity, learning
rates, batch, temperature steps, repairs and budgets are selected automatically.
Use the same heldout target list across methods, but independent method-specific
algorithmic streams. Freezing a procedure does not require using an identical
step size on every target. Manual case-specific intervention is development,
not successful transfer. A repaired procedure needs new confirmation targets;
never tune it using another method's results on the exposed holdout targets.

For responsibilities r_k=w_k*phi_k/p, evaluate

\[
E_p r_k=w_k,\quad E_p[r_kX]=w_k\mu_k,\quad
E_p[r_k(X-\mu_k)(X-\mu_k)^\mathsf{T}]=w_kv_kI.
\]

These follow directly from p*r_k=w_k*phi_k. Combine them with physical-region,
tail and projected-shape checks. Neither a component label nor a Voronoi cell
has probability w_k in general. Responsibility masses alone can be matched
by a wrong broad density.

The unit of generalization for one method is a newly drawn target together
with a fresh algorithmic random stream and its whole frozen bounded-repair policy.
Multiple chains on one target are not additional independent targets.
Count every failed/incomplete procedure trial within its cap as a delivery
failure, while preserving whether the cause was scientific or infrastructure.

A proposed reliability objective is at least 90% successful trials for each
of six methods in each of two specified families, with simultaneous one-sided
95% confidence. These are explicit planning choices, not claims already
earned. With zero failures in n independent fixed-procedure trials, the
exact binomial upper confidence bound on failure probability is
1-alpha^(1/n). For the twelve method-family claims, allocate
alpha=0.05/12 by Bonferroni; requiring this bound below 0.1 gives

\[
n\geq\left\lceil\frac{\log(0.05/12)}{\log(0.9)}\right\rceil=53
\quad\hbox{per method and family}.
\]

Thus use 53 new targets in each family, shared across methods: 106 unique
targets and 636 complete method-target trials if all six are eligible. The
all-passing design supports those twelve reliability claims. Pairing targets
across methods introduces dependence between methods but does not invalidate
the Bonferroni bound; targets and streams must remain independent within each
method-family sequence. Keep the twelve-claim allocation when fewer methods
qualify rather than choosing a more favorable allocation after seeing results.

For a 95% success objective at the same confidence, the analogous design needs
107 targets per method-family pair, or 1,284 complete trials over 214 unique
targets. The previous 36-per-family/72-total calculation covered a single
frozen procedure across two families; it was insufficient for this six-method
claim and is superseded here. The additional native/common-fitting development
branches and P6 replications are outside the 636-trial confirmation count.

A smaller balanced study of all six methods can answer useful exploratory
questions, but cannot claim the stated reliability. If failures occur, report
the exact binomial interval and the frozen criterion; do not keep drawing
targets until an unadjusted bound passes. If applicability depends on a target,
an ineligible draw counts against unconditional delivery for that family;
silently filtering to the eligible subset changes the claim. Report conditional
applicability separately. These formulas describe delivery under the declared
benchmark distribution, not universal correctness or statistical superiority.

Inspect results conditionally on width ratio, separation measured in component
standard deviations, weight imbalance and orientation. These subgroup summaries
are descriptive unless sized and adjusted for their own inferential claims.
Maintain exact accuracy checks as well as pass counts; a reliably permissive
screen is not evidence of useful inference.

## 7. Numerical choices and how to determine them

No old constant becomes a production value merely because it appears in a
master. Every effective configuration records origin, failure mode and pilot.

| Choice | Proposed starting status | Adequacy test or derivation |
|---|---|---|
| Canonical family | Owner-designated source architecture: three reversed IAF stages, ELU, author masks, free bias outside conditional cap | Core conformance and actual consumer path; no fallback to legacy masks |
| Width | Source-profile baseline and previously studied 16/32; 64 only as labeled capacity hypothesis | Crossed target-specific pilots and measured continuation; oracle fit distinguishes data from capacity/optimization |
| Initialization | Author variance scale 0.02 baseline; prior 0.2 is a separate local intervention | Matched initialization controls only if plateau evidence makes them relevant |
| Conditional cap | Recorded c=2 local choice | Check conditioner saturation and actual gradients; do not cap total Jacobian magnitude based on the old valley example |
| Learning rate | Existing 0.0003/0.001 hypotheses, not universal settings | Actual per-layer Adam updates, noise and independent development progress; bracket before nomination |
| Batch | Existing 256 baseline | Independent frozen-map minibatch gradients; enlarge when noise dominates useful update directions, pricing memory and cost |
| Clipping | Finite gradients plus calibrated emergency clipping, not a fixed frequent-clipping target | Record unclipped norms and realized Adam updates; clipping frequency alone neither proves nor rules out harm |
| Forward/reverse balance | lambda=1 source-based control | Mean scaling and sum-of-gradients parity; teacher error/term dominance diagnosed before changing lambda |
| Geometry coefficient | eta=0 control; nonzero values a new local hypothesis | Pilot gradient scales, dimensional normalization, derivative/cost checks and independent downstream viability |
| Probe distribution | Frozen pilot-derived diffuse/local/path mixture | All-dimensional noise, heldout alternative paths, sensitivity to allocations; no true centers in practical construction |
| Temperature spacing | cESS 0.8 inherited SMC pilot; 0.7 sensitivity hypothesis | Independent schedule/population sensitivity plus movement; other methods get their own calibration, and AIS does not acquire resampling implicitly |
| Particles and mutation | Measured ladder, changing one failure-related axis at a time | Repeated populations; weight concentration, ancestry and scale-adjusted movement; no acceptance-only nomination |
| Teacher repetitions | Allocate for declared uncertainty of the actual pooled bank | Whole independent populations are replication units; resampled copies are not iid observations |
| Work rungs | Geometric continuation as an engineering allocation | Paired distinct-checkpoint progress; failed-to-detect gain is not a plateau; resource caps remain explicit |
| Validation cadence | Measured cost share, e.g. 0.1 as an engineering hypothesis | If update cost C and validation V, block size at least ceil[V*(1-rho)/(rho*C)] achieves share <=rho |
| HMC settings | Public tuner, identity latent mass, independently checked epsilon/L candidates | Actual per-L measurements and fresh verification, not a Gaussian starting formula treated as qualification |
| HMC stopping | Shared sequential policy: minimum 2,000 warm-up, recent 1,000 window, max split/folded rank R-hat <=1.05; retained <=1.01; caps 10,000 per chain | Inherited operational criteria, plus declared observable ESS/MCSE and reference checks; not a convergence theorem |
| 1,000-point probe | Owner standard | Retain it and supplement physical/directed measures; a small median is not a promotion criterion |
| Method-specific controls | Source-profile hypotheses and the controls listed in Section 3.2 | Explicit bounded calibration per method; exact objective/scale checks precede optimization, and shared settings are not treated as independently tuned |
| Simple-case confirmation | Three fresh procedure trials per method and fixed target, a convenience screen | Up to 36 complete trials; all pass for that method to advance, with no broad reliability inference |
| Randomized confirmation | Proposed 90% delivery, simultaneous one-sided 95% confidence over twelve method-family claims | Derived 53 zero-failure trials per pair, 636 total; smaller studies remain exploratory and do not inherit the reliability claim |
| Arithmetic | GPU/XLA/batch-native route; FP32/TF32 intended training path, FP64 numerical reference | Reconcile the documented precision-interface mismatch first; compare actual values, inverse, scores, updates and downstream behavior in the executed dtype |
| Randomness | Independent target, teacher, initialization, tuning and confirmation streams | Save stream identities and actual controlled draws for parity; a source-hashed seed change is not an algorithmic effect |

To set posterior accuracy, specify each observable f_j, reference theta_j,
and scientific tolerance delta_j before confirmation. Use uncertainty on
the discrepancy, including reference error, and require the interval to lie
inside [-delta_j,delta_j]. A confidence interval merely containing zero does
not establish accuracy. For means with variance sigma_j^2, the iid planning
requirement for half-width delta_j is approximately
N>=(z*sigma_j/delta_j)^2. Adjust for simultaneous quantities and MCMC
dependence; calibrate the interval procedure on exact controls.

For an event of probability p, relative MCSE is approximately
sqrt((1-p)/(N_eff*p)). A required relative MCSE r therefore calls for
N_eff>=(1-p)/(p*r^2). A confidence-interval half-width requirement includes
the additional squared quantile factor; it is not the same as an MCSE target.
Use independent weighted/biased estimators for deliberately rare probabilities
when ordinary posterior draws cannot supply adequate information. Do not make
all chains observe an arbitrarily rare event by decree.

Retain pre-existing qualified profile requirements where their scope is reused.
For new random targets, specify meaningful regions without copying the old
sign/valley thresholds. The old broad-event 20% relative-MCSE requirement is
scope-specific; it is not automatically a sensible common criterion across
all new separation/variance draws. If a requested precision is unaffordable,
label the study underfunded for that quantity before claiming success.

Operational MCMC and EMUS intervals depend on assumptions and finite
calibration. State those limits separately from exact binomial target-level
reliability calculations. Gaussian looking chains or high ESS cannot exclude
an unseen region that every initialization missed.

## 8. Transfer to state-space inference

Use the same practical target interface and each method's frozen adaptation
rules throughout. Every method that passes the preceding relevant scope
receives the next scope's assessment; preserve all six rows with explicit
prerequisite failures. Do not transfer only SMC or a retrospectively chosen
winner while describing the study as six-method generalization.

1. Increase synthetic parameter dimension through D=2,5,10,20. These are
   convenience rungs with 20 chosen to address the requested dimension; they
   are not theoretical thresholds. Two/three centers have a low-dimensional
   span even when embedded in 20 dimensions.
2. Add anisotropic component covariance and known invertible nonlinear
   deformation as separate axes, with exact reference generation reserved for
   evaluation. Record covariance eigenvalue/condition ranges prospectively.
   Repeating easy isotropic components is not adequate high-dimensional evidence.
3. Use a small linear-Gaussian state-space target with Kalman likelihood and
   one/two estimated parameters for independently checked posterior quadrature.
   Its parameter posterior need not be Gaussian. Check prior, chart, likelihood
   and gradient through the full consumer.
4. Use the same nonlinear LSTM recurrence and current likelihood evaluator at
   lower q, initially the existing q=1 scope, with short/full horizons as
   distinct experiments. Increase q through already supported intermediate
   configurations toward q=20; inspect feasibility before choosing exact rungs.
5. Run the full current q20 four-parameter target only after the relevant
   precursor checks. New datasets and starts are separate confirmation scopes.
   Estimate physically meaningful and predictive quantities, not just raw
   parameter signs, especially where model symmetries exist.

Keep two error questions separate: whether the sampler targets the chosen
deterministic filter-based density, and whether that density adequately
approximates the intended state-space posterior. Compare filter likelihoods
and scores with independently controlled references on feasible reduced scopes.
Increasing sampler accuracy cannot repair an inaccurate likelihood. An
unbiased particle likelihood/pseudo-marginal or SMC² method would require a
new reviewed method scope, not a silent likelihood substitution.

For full q20, independent populations, alternative initializations and agreement
of posterior quantities are necessary evidence but not an exhaustive truth
certificate. Carry the remaining coverage and approximation limitations into
the result. Do not supply the two known sign regions to training and then
describe the result as unknown-mode discovery.

## 9. Implementation map, master behavior and resource discipline

| Existing component | Required work |
|---|---|
| neutra_warm_start_targets_tf.py and target adapters | Frozen parameterized specifications, batched density/score, separate evaluator truth and exact references; no duplicated target formula |
| neutra_warm_start_closure.py and qualification consumers | Replace fixed signs/centers/valleys and truth-based selection with the generic practical protocol; external reference assessment remains separate |
| neutra_transport.py / neutra_transport_core.py | Retain the single canonical map authority; any new total-derivative primitive lives here or calls it |
| Shared weighted/joint training and controlled-repair training consumer | Preserve stratified weights and restorable state; add the optional geometry loss only after its bounded audit |
| Existing FAB port and shared warm-start/flow-SMC components | Map all six actual endpoints to their source operations; retain distinct alpha-two/posterior targets, AIS/SMC resampling policies and AFT/CRAFT controllers; repair complete state and posterior-bank handoffs |
| Shared public HMC tuner / neutra_hmc controller | Fresh map-specific tuning, exact targets and sequential evidence; no private tuner or silent mass adaptation |
| Existing master components | Explicit six-method matrix, per-method phase dependencies, bounded repairs, full native/common-fitting cost and required P7–P9 progression for every viable method |

A focused call-chain test must show the actual master/worker reaches the
intended target, trainer and tuner. Tests only showing an unused generic
function exists are insufficient. Include collapsed-map false-pass cases,
three-component labels, shifted/rotated targets, missing-mode discovery,
teacher weight mistakes, inverse-gradient detachment, state restoration and
changed-parent resume. Add FAB auxiliary samples misused as posterior samples,
an AIS route that accidentally resamples, AFT/CRAFT ordering confusion, and a
master that stops after SMC succeeds while five rows remain untested. These
are falsifying numerical/integration tests,
not tests mirroring implementation details.

The controller must record, after each phase: evidence paths, classification,
retained candidate/optimizer, actual resource use, remaining budget, next
phase and its reason. A failed infrastructure attempt can receive one
localized retry after a focused repair, plus a bounded resource-completion
attempt if the total budget permits; this is a convenience proposal to freeze
at launch, not a reason to consume unlimited retries. Scientific repair axes
and per-axis attempt counts must likewise be finite and declared. Automatic
confirmation retries are part of the frozen procedure or new exposed trials,
never erased failed evidence.

Use Git provenance, ordinary hashes, saved configurations and unique versioned
outputs. Do not regenerate retired approval tokens or require a separate human
approval for each unchanged local retry. Sandbox permissions still apply.
A plan review and terminal review are normally sufficient; focused repairs use
focused checks.

Proposed future output root:
docs/plans/artifacts/neutra-generic-transfer-2026-10-03/campaign-r1, followed
by fresh versioned roots for later procedure revisions. Each serious run
records git/source version, exact command, environment, target/data identity,
seeds, CPU/GPU allocation, dtype/TF32/XLA, memory growth, wall time, complete
cost, plan and result paths. No directory or campaign is launched by this plan.

No numerical campaign is launched while writing this plan. Historical
allowances and the October 2 unspent suballocation are not a priced forecast
for six-method development, 36 simple-case confirmation trials and up to 636
randomized confirmation trials. Before launching, read the current shared ledger and set
explicit total GPU-process and CPU-core ceilings no greater than its available
authorization. Never add repeated historical hour grants together without
their accounting.

Price each actual full procedure, including discovery, method training,
discarded pilots, failed candidates, startup/XLA compilation, teacher generation,
replay, common-IAF fitting, validation, tuning and inference. For method m and
target-family h with N_mh required trials and conservative measured cost c_mh,
require separately for GPU-process hours and CPU-core hours

    B_confirm >= sum_m sum_h N_mh*c_mh

and reserve that amount before spending the remaining envelope on development.
Add separately priced engineering, all native/replay/refinement branches,
simple-case and state-space phases, shared reference generation and the bounded
repair allowance. Count shared work once in the campaign total and state how
it is attributed in per-method cost; report cold and amortized compilation
costs without hiding either. If the full design does not fit, retain a balanced
exploratory matrix across all six methods and state which reliability/transfer
claims remain underfunded. Do not drop five methods to fund one, reduce n while
retaining the same claim, or silently enlarge compute. A priced change of
scope is a direction decision, not a repair retry.

This is therefore a complete scientific and engineering proposal, not yet a
launch-ready numeric budget. Pricing is a required phase, not an unspecified
long experiment. The later execution manifest must freeze concrete commands,
counts, total/phase/attempt ceilings and stop rules before serious work.

Use at most the actually available GPUs, one assigned worker per device
initially, for independent teachers/targets/seeds. Sum worker time, not calendar
time. External reference/sample generation uses bounded CPU workers;
in-graph transition/training noise stays in the appropriate compiled kernel.
Configure and verify GPU memory growth before framework initialization.
Use fixed-signature batched TensorFlow kernels and evaluated XLA, without
sample-wise scalar loops, NumPy runtime computation or unapproved pfor.

## 10. Required reports and decisions

Keep separate engineering, numerical and scientific results. Every serious
result contains target/configuration status, declared scope and tuning status
before metric tables; all failures remain visible. Report hard vetoes first.

The decision table must contain decision, primary criterion, veto status,
main uncertainty, next justified action and what is not established. The
inference table must contain hard-veto screen, statistically supported ranking,
descriptive differences, default readiness and next evidence needed.
Systematic assessment requires all six method rows, not a forced ranking.
Observed cost/loss/ESS differences are descriptive without a declared
uncertainty analysis. Each row reports mathematical/source validity, native
sampling or density quality, common-teacher fitting, pre/post-RKL coverage,
fresh HMC/ensemble outcome, full cost and remaining limitations.

At P6 report whether each frozen simple-case procedure passed, rather than
whether some map eventually passed after unlimited interventions. At P7 report
per-method/per-family target counts and uncertainty, not pooled chains or
winner-only outcomes. The required coverage table lists every planned cell,
including unexecuted work, with the status vocabulary from Section 3.3.
At P9 report
both sampler-target validity and likelihood-approximation uncertainty.
The terminal review asks whether a failure invalidated the harness/target,
only the current candidate, or the claimed scope of generalization.

## 11. Skeptical review of this proposal

The review passes this as a research proposal with the following corrections
to earlier designs incorporated:

- The original proposal's material omission is repaired: FAB, Gabrié, AIS,
  SMC, AFT and CRAFT are all mandatory. Successful SMC cannot close the study.
- Native objectives and teacher handoffs are distinct. In particular, FAB's
  auxiliary density is not a posterior teacher without correction, and
  AFT/CRAFT stage maps are not silently substituted for the canonical IAF.
- Full source equivalence is not inferred from primitive parity. The FAB
  paper's admitted replay bias and the JAX loss's extra 1/B are explicitly
  separated from port defects; ineligible targets are not forcibly sampled.
- All viable methods receive unseen-target assessment. The prior 36-trial
  calculation is replaced by 53 per method-family pair for the twelve-claim
  90%-success design, and budgets count all methods and development branches.
- Matching teacher sample counts alone is not a fair cost comparison; each
  method has explicit calibration and complete resource accounting.
- It does not present already-tested FKL/RKL training as a new remedy.
- An exact teacher separates sampling failure from map fitting; a generic
  practical path prevents an oracle-assisted toy fit from being called transfer.
- Probe/stratum allocation is corrected where the objective is E_p; deliberate
  geometry reweighting is declared as a different regularizer with a total
  derivative and a qualified zero-loss argument.
- No implication from small KL to small score, from finite score probes to
  uniform geometry, or from acceptance to convergence is used.
- AFT's regularity/bounded-weight assumptions and EMUS's mixing/overlap
  assumptions are recorded rather than claimed for q20.
- Progress at a work cap remains inconclusive; neither old short limits nor
  endless continuation is a solution.
- Exact and iid controls check feasibility before demanding rare-event
  information the retained cap cannot supply.
- State dimension, sampled parameter dimension and horizon are distinct.
- Unseen two- and three-center targets are required after the simple cases;
  their confirmation trials cannot be used to manually tune the procedure.
- Local-map proposals and the tempered NeuTra ensemble remain available when
  a global map is inefficient. This does not change the canonical architecture.
- Total cost includes discovery, discarded pilots, repairs and verification.
  The plan does not pretend that historical budget balances fund an unpriced
  full generalization study.

The strongest alternative explanation for failure remains inadequate
optimization or a finite-information/tuning limitation rather than a need for
the new geometry loss. Matched eta=0 and exact controls can falsify that proposed
repair. Conversely, a geometry term improving residuals while hurting posterior
delivery would reject that intervention. A randomized procedure failing despite
simple-case success would reject the claimed transfer, not justify another
target-specific patch hidden inside confirmation.

This is skeptical self-review. No independent reviewer or new numerical result
is claimed. The plan guarantees explicit targets, corrections, falsifying
checks and phase accounting; it does not mathematically guarantee success on
every multimodal density.

## 12. Literature and inspected source anchors

All listed material sources have local copies. Source-only reads were used;
no foreign framework was imported and no package/environment was changed.

1. Hoffman et al. (2019), [NeuTra](https://arxiv.org/abs/1903.03704),
   Section 4.1.1 and Section 5. Local
   .localresources/q20-flow-training-literature-20260923/papers/hoffman-2019-neutra.txt,
   lines 337–360 and 553–566; author code/neutra-utils.py, 695–770, especially
   739–763. [Implementation correspondence](../reference/neutra-implementation.md).
2. Dai, Heng, Jacob and Whiteley (2022),
   [An Invitation to Sequential Monte Carlo Samplers](https://arxiv.org/abs/2007.11936),
   Sections 2.1–2.4. Local
   .localresources/fab-coverage-followup-20260928/dai-smc.layout.txt,
   400–486 and 655–736: path-space correction, mutation and adapt-then-freeze
   caveat. Foundational source: Del Moral, Doucet and Jasra (2006),
   Sequential Monte Carlo samplers, JRSS B 68, 411–436; local
   .localresources/smc2-parameter-state-review-20260929/delmoral-smc-samplers.pdf.
   No full author-SMC-program equivalence is asserted by this plan.
3. Gabrié, Rotskoff and Vanden-Eijnden (2022),
   [Adaptive Monte Carlo augmented with normalizing flows](https://arxiv.org/abs/2105.12603),
   III.B, IV.A–E, Appendix G.1. Local
   .localresources/fab-coverage-followup-20260928/gabrie-adaptive-flows.layout.txt,
   216–244, 275–355 and 1701–1728; author
   .localresources/flonaco-author-20260929/upstream/flonaco/sampling.py,
   151–167 and 207–249; training.py, 221–234.
   [Local adaptation record](bayesfilter-gabrie-flonaco-source-adoption-2026-09-29.md).
4. Arbel, Matthews and Doucet (2021),
   [Annealed Flow Transport Monte Carlo](https://proceedings.mlr.press/v139/arbel21a.html),
   equations (8)–(9), Section 3.3, Appendix C.1 and practical Algorithm 2.
   Local .localresources/smc-modern-improvements-20260929/aft.txt, 217–246
   and 284–327; aft-linear.txt, 1580–1627, contains the moment, compactness,
   bounded-weight and smoothness assumptions, and 7435–7488 introduces the
   practical three-population algorithm. Local aft-author.py, 118–171,
   checks validation selection before updating parameters. Source family:
   google-deepmind/annealed_flow_transport.
5. Matthews, Arbel, Rezende and Doucet (2022),
   [CRAFT](https://proceedings.mlr.press/v162/matthews22a.html),
   Section 2.3, equations (5)–(8), training/deployment algorithms and
   supplement. Local same directory, craft.txt, 295–355; craft-linear.txt,
   1610–1638 distinguishes finite-particle and ideal objectives; craft-author.py,
   120–170 runs the temperature pass before applying stage updates, and
   craft_evaluation_loop freezes maps.
6. Noé et al. (2019),
   [Boltzmann Generators](https://arxiv.org/abs/1812.01729),
   Methods equation (9), reweighting equations (15)–(16), supplementary
   Figure S2. Local
   .localresources/neutra-mode-initialization-literature-20260929/noe-boltzmann-generators.txt,
   1326–1360, 1470–1499, 2320–2335; noe-archive-invertible.py,
   462–474 and 581–607. Source energy clipping is an explicit difference.
7. Thiede et al. (2016),
   [Eigenvector method for umbrella sampling enables error analysis](https://arxiv.org/abs/1603.04505),
   Sections II–III and VII, equations (14)–(15), assumptions VII.1/VII.3
   and Theorem VII.4. Local .localresources/neutra-rare-events-20261002/
   thiede-emus-2016.layout.txt, 129–229 and 697–779; emus-author-reference.py,
   calculate_zs, emus_iter and calculate_Fi. No NumPy author implementation
   becomes an admitted TensorFlow runtime by citation alone.
8. Vaitl et al. (2024), Fast Path Gradients for Normalizing Flows,
   Proposition 3.2, equation (16), Appendix B.1. Local
   .localresources/q20-flow-training-literature-20260923/papers/vaitl-2024-fast-path.txt,
   326–337 and 886–903. Roeder et al. (2017), Sticking the Landing,
   is retained as the upstream estimator reference in the shared implementation
   record; no new variance dominance claim is made.
9. Midgley et al. (2023), Flow Annealed Importance Sampling Bootstrap,
   Sections 3.1–3.2, Algorithm 1, Section 4.1 and Appendix A. Local
   .localresources/neutra-highdim-literature-20260925/fab-2023.txt,
   163–325 for objectives/replay and the admitted without-replacement bias,
   346–381 for the original mixture comparison, and 791–899 for normalizers,
   gradient derivation, finite-normalization bias and tail limitations.
   Appendix C.1, 1231–1246, supplies a Gaussian gradient diagnostic;
   it is not a guarantee for general multimodal targets. Pinned author source:
   .localresources/fab-jax-c9f9913/fabjax/train/fab_without_buffer.py,
   34–41 and 66–86; train/fab_with_buffer.py, 20–36;
   buffer/prioritised_buffer.py, 82–101 and 166–190; sampling/smc.py,
   78–110 and 123–145. Check paper and code separately: extra loss scaling,
   clipped corrections, nonfinite handling and replay law affect the claim.
10. Neal (2001), Annealed Importance Sampling, Statistics and Computing
    11, 125–139; local 1998 technical-report version
    .localresources/neutra-highdim-literature-20260925/neal-ais-2001.txt,
    Section 2, 161–264, equations (3)–(11): support, invariant kernels,
    weighted paths and the reverse-kernel derivation. The inspected FAB-JAX
    sampling/smc.py supports alpha=1 and use_resampling=False; reusing that
    primitive does not make posterior AIS the same training method as FAB.

The Hölder bridge bound, Gaussian FAB integrability condition, auxiliary-to-
posterior weight correction, finite-discovery bound, minibatch expectation,
partition KL decomposition, local score-residual argument, and confirmation
sample-size formulas are explicitly derived in this document. They should be
reviewed on their assumptions rather than attributed to an unrelated paper.

Document verification: local cross-links and the cited source ranges were
checked; binomial sample sizes, including all twelve method-family claims,
were independently recomputed with Python's standard library. Matrix coverage,
phase dependencies and linked checkpoint/design consistency were checked.
No numerical framework, training test, scientific experiment or external
reviewer was invoked for this proposal.
