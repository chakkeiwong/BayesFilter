# Audit: Discontinuous ZLB HMC Survey

Audit date: 2026-08-19

Audited source: `docs/surveys/zlb_discontinuous_hmc/zlb_discontinuous_hmc_survey.md`

## Executive Verdict

**Major revision required before this survey is an implementation authority.**
The revision made important corrections: it distinguishes a continuous `max`
kink from a density jump and mixed support; identifies the current MacroFinance
implementation as softplus; adds the Holden, Boehl, Boehl--Strobel,
Opschoor--van der Wel, and Pakman--Paninski lines; and treats OBC multiplicity
and nonexistence as target-definition problems. Keep those changes.

The remaining high-risk issues are:

1. `dsge_hmc` is not absent. The standalone path named in the manuscript is
   absent, but the package exists under `/home/ubuntu/workspace/python/src/dsge_hmc`.
   Its BGS surface is currently a restricted linear no-binding object, not a
   genuine OBC/ZLB likelihood. The manuscript needs this distinction.
2. The MacroFinance "exactly weighted fully adapted" filter is an ideal
   decomposition, not a presently implementable closed-form authority. General
   Gaussian polytope masses and truncated-normal draws are numerical problems;
   pruning cells or taking one Pakman--Paninski HMC trajectory changes the
   proposal/weight claim.
3. MacroFinance has separate targets: continuous-maturity hard max, finite
   `K=40` hard-max quadrature, and the current finite `K=40` softplus. They need
   separate identifiers and evidence contracts.
4. Measure conventions, proposal coordinates, nonidentity mass matrices, and
   pseudo-marginal HMC assumptions are not yet precise enough for implementation.

This is an audit and change proposal. It does not claim posterior correctness,
HMC convergence, production readiness, or scientific superiority.

## Source Identity And Caveat

The user named a `.tex` file, but no
`docs/surveys/zlb_discontinuous_hmc/zlb_discontinuous_hmc_survey.tex` exists in
the current workspace. The Markdown file is therefore audited as the canonical
source; the rendered HTML is not treated as an independent source.

| Item | Value |
|---|---|
| Source | `zlb_discontinuous_hmc_survey.md` |
| Size | 2,326 lines; 102,308 bytes |
| SHA-256 | `d8fdb7bf07e43c1eb6385e42119ab6aa96afa67740a4300589ba396466f32b29` |
| Requested `.tex` | absent |
| Task-authored change | this audit file only |

Line references below refer to this source identity. Existing dirty and
untracked user work was preserved.

## Findings

### P0-1: Correct The `dsge_hmc` Identity

**Affected manuscript lines:** 10, 1351--1353, 1798--1808, 2032--2035,
2078--2081.

The statement that `dsge_hmc` does not exist is false as a workspace statement.
The relevant package tree is present at:

```text
/home/ubuntu/workspace/python/src/dsge_hmc
```

The useful diagnosis is that the package exists but the currently validated BGS
surface does not implement a true lower-bound regime. The local source anchors
are:

- `models/bgs_restricted_surface_generated.py:225`: notional policy rule for
  `rn`;
- `models/bgs_restricted_surface_generated.py:228`: constrained-rate residual
  is the placeholder `r = rn`;
- `models/bgs_restricted_surface_tf_coefficients.py:17`: rows tagged
  `OBC_ZLB_NO_RUN_GUARD`;
- `/home/ubuntu/workspace/python/docs/plans/actual-bgs-restricted-surface-port-master-program-v2-2026-07-09.md:98-100`:
  no OBC/ZLB logic is active in the witness likelihoods;
- `/home/ubuntu/workspace/python/docs/plans/actual-bgs-restricted-surface-port-master-program-v2-2026-07-09.md:145-155`:
  the evidence contract explicitly excludes OBC/ZLB estimation;
- `/home/ubuntu/workspace/python/docs/plans/actual-bgs-restricted-surface-port-phase10-qe-shadow-zlb-extension-subplan-2026-07-09.md:9-19,68-77`:
  true OBC/ZLB estimation requires a new architecture.

This is a material error because Section 14 presents a generic three-equation NK
toy as if it were a contract for the actual BGS application. Replace the
wording with:

> The `dsge_hmc` package exists under `/home/ubuntu/workspace/python/src/dsge_hmc`.
> Its validated BGS restricted surface has the linear placeholder `r=rn` and
> is not a true OBC/ZLB model. A genuine BGS ZLB target requires a new,
> source-anchored solution, selection, filtering, and inference architecture.

Retitle Section 14 to say that the NK example is pedagogical and add a separate
BGS subsection with these anchors. Do not transfer the toy's solution geometry
to the 46-parameter BGS surface without deriving the active equations.

### P0-2: Downgrade The MacroFinance Fully Adapted Claim

**Affected manuscript lines:** 1572--1654, 1788--1795, 2049--2053.

Equations (84)--(86) are a valid one-step cell decomposition. The claim that
(87) is an available closed-form, exactly weighted fully adapted filter is not
established. Each parent particle requires multivariate Gaussian polytope
probabilities and an exact draw from the selected truncated Gaussian. Those are
numerical computations, not closed-form operations in general. With two country
factor blocks and FX information, the joint posterior cell probability can be
six-dimensional unless a factorization is proved; "three-dimensional per
country" is not enough.

Two specific statements are wrong as implementation guarantees:

- Lines 1624--1626 retain cells with "non-negligible" mass. Dropping a cell
  changes `sum_B Z_B`, cell-selection probabilities, and the weight. The result
  is approximate unless omitted masses are exactly zero and certified.
- Lines 1645--1647 treat one Pakman--Paninski HMC trajectory as an exact
  truncated-Gaussian draw. Their method supplies an invariant HMC kernel for a
  truncated Gaussian, not an independent exact draw after an arbitrary finite
  trajectory. Rejection sampling is exact in principle but may be unusable.

**Required change.** Rename (84)--(87) an *ideal fully adapted proposal
identity*. Make the primary hard-bound marginal likelihood authority a bootstrap
PF with exact Gaussian transition draws and exact observation-density weights:

```text
x_t^j ~ N((I-Phi) theta_bar + Phi x_{t-1}^{A_t^j}, Q)
v_t^j = N(y_t; h_hard(x_t^j), R)
```

This estimator is nonnegative and unbiased under standard support/integrability
conditions, so PMMH has the ordinary particle-MCMC extended-target guarantee.
Keep the split/UKF construction as a diagnostic or proposal experiment. If it
uses pruning, its actual proposal density and importance ratio must be used; it
cannot retain the constant fully adapted weight from (87).

### P0-3: Name Three Distinct MacroFinance Targets

**Affected manuscript lines:** 1403--1416, 1506--1533, 1779--1795.

The current text conflates these objects:

| Identifier | Definition | Geometry / claim |
|---|---|---|
| `mf_c0_root_integral_hardmax` | `tau^-1 integral max(ell,f(u;x)) du` | continuous-maturity target; root/tangency regularity required |
| `mf_c1_k40_hardmax` | current nodes/weights with softplus replaced by `max` | finite hyperplane-arrangement kink target |
| `mf_s1_k40_softplus` | current coded `K=40` softplus | smooth declared model |

The manuscript derives the second object in (81), while (78) begins from the
continuous integral and then introduces quadrature. C0 and C1 are not the same
posterior. Under a simple transverse root, moving-boundary terms cancel because
the two integrand branches agree at the crossing, so C0 can be smoother than the
finite pointwise-max sum C1. Tangencies, endpoint roots, and an identically
bound curve need separate cases.

**Required change.** Add a target table before Section 13.3 and reserve
"exact" for inference relative to a named target and numerical program. Make
C1 the first hard-bound counterfactual because it changes one operation in the
current source. Add root-aware C0 integration as a discretization sensitivity
experiment before choosing event/kink HMC as the final hard-ZLB design. Keep S1
as a valid smooth model, never as silent hard-ZLB inference.

### P0-4: Make Measures, Initialization, And Proposal Coordinates Coherent

**Affected manuscript lines:** 90--107, 125--149, 263--303, 774--823,
1278--1349.

Equation (4) writes a Lebesgue density for a deterministic transition. A map
driven by shocks can be singular or require a change-of-variables Jacobian. The
clean baseline is a Markov kernel or a non-centered shock chart. Add the kernel

```text
K_theta(x, A, r)
  = integral 1{R(x,epsilon,theta)=r}
             1{F_r(x,epsilon,theta) in A} phi_theta(d epsilon)
```

and then state whether a density exists relative to a declared dominating
measure. In (5a), `pi_r(q)` must be a density in a named branch coordinate; if
branches use different coordinates, their Jacobian factors belong there.

Also repair two concrete inconsistencies:

- (4) starts at `x_0`, but (12a) starts from `p(X_1|theta)`. Define the
  induced initial law for `x_1` or carry `x_0` in every particle.
- In (53), the numerator is in `epsilon_t` while the proposal is written in
  `z_t`. Either propose shocks directly or include the transformation Jacobian
  and use the matching density.

### P0-5: Derive Events For The Actual Mass Matrix

**Affected manuscript lines:** 936--974.

The event rule is only for unit mass, but project HMC uses a mass matrix `M`.
For `K(p)=p^T M^-1 p/2` and boundary normal `n`, define

```text
a   = n^T M^-1 n
s_- = n^T M^-1 p_- / sqrt(a)
```

For a potential jump `Delta U`, cross when `s_-^2 > 2 Delta U`, with

```text
s_+ = sign(s_-) sqrt(s_-^2 - 2 Delta U)
p_+ = p_- + n (s_+ - s_-) / sqrt(a)
```

and reflect otherwise with

```text
p_+ = p_- - 2 n (n^T M^-1 p_-) / (n^T M^-1 n).
```

Add boundary orientation, equality/grazing conventions, and tests with dense
and diagonal nonidentity metrics. The unit-mass formulas cannot be presented as
the project event kernel.

### P0-6: State The Pseudo-Marginal HMC Assumptions Precisely

**Affected manuscript lines:** 1147--1160 and 1210--1229.

Alenlov, Doucet, and Lindsten require: (i) a nonnegative unbiased
`Lhat(theta,U)`; (ii) `U ~ N(0,I)`; (iii) continuous differentiability of
`(theta,U) -> Lhat(theta,U)`; and (iv) pointwise evaluability of
`grad log Lhat(theta,U)`. They explicitly note that ordinary particle-filter
likelihoods are typically discontinuous in `(theta,U)` because resampling
indices change at cumulative-weight thresholds.

Therefore an unbiased bootstrap PF supports PMMH, not pseudo-marginal HMC by
itself. A differentiable resampling or transport gradient is an approximation
or a different extended target until its full law and Metropolis correction are
derived. Add these assumptions and make PM-HMC conditional on a separately
constructed differentiable extended estimator.

## Additional Mathematical Corrections

### P1-1: Stable Tail Evaluation

Lines 440--510 derive the correct affine-truncation moments, but direct
evaluation of `phi(gamma)/Phi(gamma)` and
`phi(gamma)/(1-Phi(gamma))` will underflow in the tails. Require `logcdf`,
`logsf`, and log-density arithmetic (or a tested inverse-Mills-ratio routine).
Any variance clamp should be documented as a roundoff safeguard, not as a
replacement for stable probability evaluation.

### P1-2: Include The Identically-Zero Root Case

Lines 1560--1564 say that the Nelson--Siegel gap has at most two zeros. Add the
exception `L-ell=0, S=0, C=0`, for which the gap is identically zero. Also
separate the per-country three-dimensional factor geometry from the full
domestic/foreign probability in (86), which is generally six-dimensional.

### P1-3: Correct The Softplus Sign Statement

Lines 1695--1710 correctly give an absolute Gaussian log-density bound, but the
wording that the log-likelihood perturbation accumulates one-signedly is false.
For `Delta=h_alpha-h_max`,

```text
log N(y;h_alpha,R)-log N(y;h_max,R)
  = (y-h_max)^T R^-1 Delta - Delta^T R^-1 Delta / 2.
```

The individual yield-map shift is nonnegative, but this likelihood difference
can have either sign. The FX row is a domestic-minus-foreign combination and
can have a signed mean shift too. Say that the mean-map discrepancy is
one-sided and may induce systematic fitted-factor adjustment; report signed
likelihood changes rather than asserting their sign.

### P1-4: Qualify The `O(sqrt(alpha))` Step-Size Claim

Lines 1712--1730 correctly bound scalar softplus curvature by `1/(4 alpha)`.
The resulting `epsilon=O(sqrt(alpha))` statement is a local stability heuristic
under a fixed metric and comparable posterior curvature, not a universal
necessity theorem. State the dependence on chart, mass matrix, trajectory, and
near-bound occupancy.

### P1-5: Add Parametric-LCP Regularity Conditions

Lines 1827--1844 use the P-matrix criterion appropriately as a uniqueness gate,
but pointwise uniqueness does not by itself prove continuity of the solution
map. State a fixed finite horizon, continuous `q(theta)` and `M(theta)`, a
P-matrix condition over the declared parameter domain, and the applicable
parametric-LCP regularity result. A changing horizon, terminal condition, or
loss of the P-matrix property can create discontinuity.

### P1-6: Narrow The Kinked-HMC Proposition

Lines 876--930 should state a proposition for a finite (or locally finite)
piecewise-regular partition with deterministic branch gradients and a boundary
convention. The null-preimage argument is not valid for an arbitrary
nonsmooth composition merely because individual shears preserve measure. Keep
the conclusion for the finite hyperplane arrangement in C1 and explicitly
exclude an implicit multi-root equilibrium solver until a separate theorem is
proved.

### P1-7: Avoid Tobit Terminology For The MacroFinance Ordering

The manuscript correctly explains at lines 1656--1665 that MacroFinance adds
Gaussian noise after the hard map, whereas a type-I Tobit model censors after
noise and can have an atom. Apply that distinction consistently: rename the
route table's "censored-measurement particle filter" to
`hard-bound observation-map particle filter`. Reserve `censored measurement`
for the noise-ordering actually used.

## Recommended MacroFinance Design

Freeze a `target_id` in every fixture, likelihood, sampler, and result:

| `target_id` | Definition | Allowed claim |
|---|---|---|
| `mf_s1_k40_softplus` | current finite softplus program | smooth-model inference |
| `mf_c1_k40_hardmax` | same nodes/weights, hard max | exact relative to the finite hard-quadrature model |
| `mf_c0_root_integral_hardmax` | root-aware continuous-maturity hard integral | exact relative to the declared integral/tolerance contract |

The recommended sequence is:

1. Implement S1, C1, and C0 as independent evaluators. Test no crossing, one
   crossing, two crossings, tangency, endpoint root, and identically-bound
   cases.
2. Build C1 joint non-centered HMC and a bootstrap-PF PMMH cross-check with
   exactly matched priors, initial-state treatment, and measurement rows.
3. Implement root-bracketed C0 integration and compare C0 to C1 before
   promoting an event-aware hard-bound design.
4. Keep the split/mixture UKF as a diagnostic/proposal arm. Do not use its
   moment-matched likelihood as an exact authority.
5. Only revisit a cell-adapted proposal after inspecting Genz/Bretz Gaussian
   probability algorithms and Botev truncated-normal methods, with numerical
   error and support included in the proposal contract.

The bootstrap PF is less elegant than the ideal cell decomposition but is
implementable, unbiased for the named finite model, and honest about its Monte
Carlo variance.

## Recommended Actual BGS `dsge_hmc` Design

Split the present Section 14 into two layers.

**Pedagogical layer.** Retain the three-equation NK/LCP example to explain
support, multiplicity, and selection-law requirements. Label it as a toy and do
not call it the BGS model.

**BGS layer.** Before proposing a sampler, document:

1. the actual BGS source/version, lower-bound level, policy equation,
   expectations convention, terminal condition, and measurement system;
2. every active-constraint alteration, including QE feedback and lagged-state
   carriers;
3. a source-anchored solution operator and verification loop;
4. the uniqueness/nonexistence domain and a selection law on multiplicity
   regions;
5. the likelihood value when no verified path exists; and
6. source/Dynare path parity before particle or HMC promotion.

The first BGS deliverable is therefore a transition/solution kernel, not a
particle filter. Once the kernel is defined, compare inversion filtering,
piecewise Kalman, EnKF, bootstrap PF, and any gradient method on the same
target. The current restricted-surface parity program remains a valuable
no-binding baseline and must not be silently relabelled as ZLB evidence.

## Missing And Underpromoted Literature

Several works appear only in `omitted_papers.md` or indirectly through a
secondary source, even though the proposed implementation depends on them.
Promote the following before the corresponding design decision.

| Priority | Work/family | Why needed | Current status | Change |
|---|---|---|---|---|
| P0 | Cuba-Borda, Guerrieri, Iacoviello, and Zhong (2019), *Likelihood Evaluation of Models with Occasionally Binding Constraints* | Direct inversion-filter likelihood comparator | secondary mention only | inspect primary source; add to OBC baseline section |
| P0 | Holden and Paetz (2012), *Efficient Simulation of DSGE Models with Inequality Constraints* | Original anticipated-shock route | absent from main references | add citation and route comparison |
| P0 | Pericoli and Taboga (2018), *Nearly Exact Bayesian Estimation of Non-linear No-Arbitrage Term-Structure Models* | Direct Bayesian nonlinear shadow-rate/MCMC comparator | local PDF recovered by ResearchAssistant; absent from survey | inspect and add to Section 13; narrow the "frontier" claim |
| P0 | Genz and Bretz (2009), multivariate normal/t probabilities | Required for Gaussian polytope masses | omission register only | inspect before a fully adapted-cell proposal |
| P0 | Botev (2017), normal law under linear restrictions/minimax tilting | Required truncated-normal probability/draw benchmark | absent | inspect and benchmark before claiming exact cell draws |
| P0 | Deligiannidis, Doucet, and Pitt (2018), correlated pseudo-marginal method | PMMH variance/mixing comparator | absent | add, explicitly not as a PM-HMC differentiability fix |
| P0 | Nemeth, Fearnhead, and Mihaylova, linear-cost particle score | Particle-score cost/variance alternative | local source exists; omission register only | inspect before selecting a score estimator |
| P0 | Dahlin, Lindsten, and Schon, gradient/Hessian particle MH | Gradient-informed particle-MCMC comparator | absent | inspect before defaulting to random-walk PMMH |
| P1 | Particle Gibbs with backward simulation | Distinct path update under genealogy collapse | omission register only | add as a comparison arm to PGAS |
| P1 | Kim--Singleton, Ichiue--Ueno, Gorovoi--Linetsky | Shadow-rate predecessor lineage | prose-only/underreferenced | add accurate bibliography and scope |
| P1 | Published Krippner and post-2015 shadow-rate refinements | Prevents an early literature cutoff | omission register only | inspect before empirical benchmark freeze |
| P1 | Fernandez-Villaverde et al., *Nonlinear Adventures at the ZLB* | Nonlinear economic solution benchmark | absent | add context section |
| P1 | Gust et al. (2017), *The Empirical Implications of the Interest-Rate Lower Bound* | Major estimated-ZLB comparator | secondary-source mention | add before BGS empirical claims |
| P1 | Atkinson, Richter, and Throckmorton (2020), *The Zero Lower Bound and Estimation Accuracy* | Direct estimation-accuracy evidence | secondary-source mention | add to baseline limitations |
| P1 | Keen, Richter, and Throckmorton (2017), *Forward Guidance and the State of the Economy* | Anticipated-shock/forward-guidance context | secondary-source mention | add if that route is used |
| P1 | Adjemian--Juillard extended-path and stochastic extended-path work | Nonlinear OBC solver alternative | indirect mention | add with convergence caveats |
| P2 | Reversible-jump/involutive MCMC; Zig-Zag/Bouncy Particle; RATTLE/manifold HMC | Alternatives if support or dimension changes | omission register only | keep as conditional fallback, not current authority |
| P2 | 2022--2026 differentiable PF and SMC2 work | Recent optimization comparisons | metadata/extraction only | classify as approximate/extended-target methods |

The Pericoli--Taboga recovery is especially important. The current text at
1498--1504 should not call MacroFinance's UKF-plus-HMC route "at the current
frontier" based on the inspected set alone. A safer statement is that the
inspected shadow-rate papers predominantly use Gaussian closure, while the
broader Bayesian nonlinear term-structure literature still requires comparison.

## Suggested Revised Outline

1. Geometry taxonomy plus base measures and non-centered coordinates.
2. Linear OBC baselines: OccBin, PKF, inversion filter, anticipated shocks,
   Boehl, and extended path.
3. Particle/state-space targets: bootstrap PF authority, COPF special case,
   PMMH, correlated PMMH, PGAS/backward simulation, and score methods.
4. HMC by geometry: kink, jump, mixed support, full-metric events, and PM-HMC
   assumptions.
5. MacroFinance target ladder C0/C1/S1 and root-aware integration.
6. Actual BGS package status and source-anchored true-OBC architecture gap.
7. Evidence gates, nonclaims, and implementation roadmap.

This ordering puts implementable authorities before ambitious cell-adapted and
event-aware methods and prevents the generic word "discontinuous" from choosing
an algorithm prematurely.

## Acceptance Checklist

### MacroFinance

- [ ] C0, C1, and S1 have separate equations and `target_id`s.
- [ ] Every "exact" claim names its target, integration rule, and coordinate
      chart.
- [ ] C1 bootstrap PF is the first unbiased marginal-likelihood authority;
      PMMH is its exact wrapper.
- [ ] (84)--(87) are labelled ideal/advanced until polytope probabilities,
      truncated draws, and pruning error are controlled.
- [ ] Joint two-country probability dimension is corrected or factorization is
      proved.
- [ ] Root-aware C0 integration is compared with K=40 C1.
- [ ] Softplus reports distinguish one-sided map shifts from signed likelihood
      differences and handle the FX row.

### Actual BGS

- [ ] Package path and `r=rn` no-binding placeholder are corrected.
- [ ] The generic NK example is explicitly pedagogical.
- [ ] Active BGS equations, bound, terminal condition, solver, selection law,
      and no-solution treatment are source-anchored.
- [ ] Source/Dynare path and likelihood parity precede HMC.
- [ ] Inversion-filter and current OBC estimation literature are included.

### General methods

- [ ] Deterministic transitions use kernels or a declared non-centered chart.
- [ ] Proposal coordinates and Jacobians are coherent.
- [ ] Event rules use the actual positive-definite mass matrix and include
      grazing/equality tests.
- [ ] PM-HMC assumptions and ordinary-PF exclusion are explicit.
- [ ] Tail-stable truncated moments and degenerate root cases are tested.
- [ ] The kink proof is scoped to a piecewise-regular partition.

## Tool Record And Limits

The requested workspace tools were used as bounded research aids. They inform
discovery and structural checking; none certifies the manuscript.

| Tool | Evidence used | Limitation |
|---|---|---|
| ResearchAssistant | `/tmp/ra-zlb-audit-20260819`; eight local PDFs, including Holden, Boehl, Boehl--Strobel, Opschoor--van der Wel, Pericoli--Taboga, and differentiable-PF/SMC2 leads | offline providers; summaries are marked `needs_review`; primary claims above are not based on a summary alone |
| MathDevMCP | `/tmp/mathdevmcp-zlb-reread-20260819/audit-f6e15233a5f825bc5ab4b1dd48e927703fe0eed97081374dc50f50ecd1900610.json` | completed with limits; all 12 selected obligations were `not_checkable`, with zero semantic findings. This is structural coverage, not proof certification. |
| DynareMCP | `/tmp/dynaremcp-zlb-audit-20260819`; audited `dynare_occbin_example.mod` | the fixture is an irreversible-investment OBC example, not the BGS monetary ZLB model; Dynare JSON was unavailable, so output is diagnostic/source-heuristic only |

Additional local inspection established the actual BGS package, its source row
tags, the `r=rn` placeholder, and the project plans that explicitly block true
OBC/ZLB estimation in the current lane. No sampler was run, no package or
environment was mutated, and no manuscript/source/code other than this audit
was edited.

## Bottom Line

The revision now contains the right conceptual split: the current MacroFinance
code is a smooth softplus model, its finite hard-max counterfactual is a kink
target rather than a density jump, and a genuine DSGE OBC becomes discontinuous
only after a solution correspondence and selection law are specified. The next
revision should turn that insight into this implementable hierarchy:

```text
MacroFinance C1: joint hard-kink HMC + bootstrap-PF PMMH authority
MacroFinance C0: root-aware continuous-maturity sensitivity target
MacroFinance S1: current smooth model, honestly labelled
BGS dsge_hmc: source-anchored true-OBC solver/filter master program before HMC
```

That is a defensible route to solving the requested discontinuous problem
without promoting an unimplemented polytope-probability oracle or a hypothetical
repository into a solution claim.
