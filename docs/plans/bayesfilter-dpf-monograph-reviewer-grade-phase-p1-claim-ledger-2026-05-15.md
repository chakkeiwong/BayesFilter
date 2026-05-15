# Phase P1 result: skeptical-reader argument map and claim ledger

## Date

2026-05-15

## Governing plan

- `docs/plans/bayesfilter-dpf-monograph-reviewer-grade-revision-master-program-2026-05-14.md`
- `docs/plans/bayesfilter-dpf-monograph-reviewer-grade-phase-p0-preflight-result-2026-05-15.md`
- `docs/plans/bayesfilter-dpf-monograph-reviewer-grade-phase-p1-claim-ledger-plan-2026-05-14.md`

## Scope and safety state

Branch at phase start:

- `main`

Dirty files classified before execution:

- Unrelated student-baseline files remain dirty and were not edited:
  - `docs/plans/bayesfilter-student-dpf-baseline-master-program-2026-05-10.md`
  - `docs/plans/bayesfilter-student-dpf-baseline-reset-memo-2026-05-09.md`
- This phase created only this DPF reviewer-grade result artifact.

ResearchAssistant status:

- ResearchAssistant was available in read-only/offline mode.
- Local searches returned no stored paper summaries for SMC/bootstrap filters,
  EDH/LEDH/PF-PF, differentiable resampling/OT/Sinkhorn, or HMC/pseudo-marginal
  methods.
- Therefore P1 treats source support as "chapter/bibliography-spine pending P2",
  not as reviewed ResearchAssistant support.

## Argument map

The revised DPF block must read as the following dependency ladder.

1. Exact nonlinear filtering object:
   - state transition density;
   - observation density;
   - prediction law;
   - filtering law;
   - marginal-likelihood factorization.
2. Classical particle approximation:
   - empirical measure approximation;
   - sequential importance sampling;
   - bootstrap/SIR specialization;
   - bootstrap likelihood estimator and its precise unbiasedness status.
3. Degeneracy and differentiability problem:
   - ESS collapse and resampling need;
   - categorical ancestor discontinuity;
   - distinction between unbiased likelihood estimation and pathwise
     differentiability.
4. Particle-flow bridge:
   - homotopy density between predictive and filtering laws;
   - continuity equation;
   - velocity-field non-uniqueness;
   - EDH as Gaussian-closure affine flow;
   - LEDH as particle-local linearization;
   - exactness only under stated special cases.
5. Proposal-corrected particle flow:
   - flow map as proposal transformation;
   - change-of-variables density;
   - corrected PF-PF weight;
   - Jacobian/log-determinant evolution;
   - finite-particle and numerical limits.
6. Differentiable resampling:
   - categorical resampling as exact but discontinuous;
   - soft resampling as a differentiable surrogate;
   - OT equal-weighting as a transport projection;
   - EOT/Sinkhorn as regularized and numerical layers.
7. Learned transport:
   - teacher map from EOT/Sinkhorn;
   - learned student map;
   - equivariance constraints;
   - residual, training-distribution, and target-shift diagnostics.
8. HMC target analysis:
   - scalar value and gradient must correspond to the same target;
   - exact special-case, corrected-particle, relaxed, and learned-surrogate
     rungs must not be conflated;
   - banking/structural-model suitability requires validation evidence beyond
     algebraic smoothness.
9. Debugging and implementation contract:
   - each failure mode must be tied to equations and diagnostics;
   - diagnostics do not validate a method unless they test the relevant target,
     density-ratio, or approximation claim.

## Claim ledger

Status vocabulary:

- exact model identity;
- unbiased particle estimator;
- consistent approximation;
- approximate closure;
- relaxed target;
- learned surrogate;
- engineering hypothesis;
- unsupported claim.

| ID | Current claim locus | Claim to control | Status | Source/support state | Assumptions and approximation layer | Implementation implication | Reviewer risk |
|---|---|---|---|---|---|---|---|
| P1-C01 | `ch19_particle_filters.tex`, `eq:bf-pf-predictive-law`--`eq:bf-pf-marginal-factorization` | The nonlinear filtering recursion and marginal-likelihood factorization define the exact target object. | exact model identity | Chapter equations; cited SMC sources pending P2 role notes. | Dominated transition and observation densities; fixed parameter; well-defined integrals. | Code must expose likelihood factors and filtering/predictive laws separately. | Low if symbols are fully defined; medium if non-specialists cannot see Bayes-rule derivation. |
| P1-C02 | `ch19_particle_filters.tex`, `eq:bf-pf-empirical-measure` | Weighted particles approximate the filtering law. | consistent approximation | Chapter text; SMC literature pending P2. | Finite particle empirical measure; convergence requires standard regularity not yet spelled out. | Tests should compare expectations of bounded functions against analytic or high-N references. | Medium: "approximation improves with particle count" needs assumption qualifiers. |
| P1-C03 | `ch19_particle_filters.tex`, `eq:bf-pf-importance-weight`--`eq:bf-pf-sis-recursion` | Sequential importance weights follow from trajectory target/proposal ratios. | exact model identity for the algebra; particle approximation for finite N | Chapter equations; SMC sources pending P2. | Proposal support must cover target support; weights finite. | Implementation must compute proposal log-density and target log-density consistently. | Medium: current derivation is compressed and should be expanded in P3. |
| P1-C04 | `ch19_particle_filters.tex`, `prop:bf-pf-bootstrap-likelihood-status` | The bootstrap PF likelihood estimator is unbiased for the marginal likelihood. | unbiased particle estimator | Cited Doucet and Andrieu/PMCMC sources pending P2; no RA record. | Standard bootstrap filter construction; resampling scheme preserving empirical expectation; finite expected weights. | Value-side estimator can support pseudo-marginal logic; this does not imply pathwise differentiability or unbiased score. | High: proof sketch is too short for skeptical reviewers; P3 must expand conditional-expectation filtration. |
| P1-C05 | `ch19_particle_filters.tex`, chapter boundary | A differentiable surrogate may not preserve the exact estimator status. | engineering warning grounded in estimator distinction | Chapter logic; source support pending P2. | Surrogate changes random map or resampling law unless proven otherwise. | DPF code must label scalar as original likelihood estimator, relaxed scalar, or surrogate scalar. | Low to medium; needs consistent terminology across later chapters. |
| P1-C06 | `ch19_particle_filters.tex`, `eq:bf-pf-ess` and degeneracy section | ESS collapse motivates resampling and later DPF modifications. | engineering hypothesis with SMC diagnostic support | Chapter citation to high-dimensional degeneracy sources pending P2. | ESS is diagnostic, not theorem of failure; model dimension and observation informativeness matter. | Tests should track ESS, log-weight variance, unique ancestor count, and likelihood variance. | Medium: avoid saying ESS alone proves invalidity. |
| P1-C07 | `ch19b_dpf_literature_survey.tex`, `eq:bf-pff-log-homotopy`--`eq:bf-pff-homotopy-normalizer` | The homotopy endpoints recover predictive and filtering densities. | exact model identity | Chapter derivation; source role pending P2. | Predictive density and likelihood positive where needed; normalizer finite. | Code must evaluate log predictive density, log likelihood, and log normalizer/proxy consistently. | Medium: endpoint exactness must not be conflated with exactness of the numerical flow. |
| P1-C08 | `ch19b_dpf_literature_survey.tex`, `eq:bf-pff-continuity-equation`--`eq:bf-pff-flow-pde` | The continuity equation is exact as a conservation statement, but does not determine a unique velocity field. | exact model identity plus non-uniqueness warning | Chapter PDE; derivation obligation for P4. | Smooth density path and regular enough velocity field; boundary flux conditions. | Flow solvers must record chosen velocity-field parameterization and boundary/numerical assumptions. | Medium: P4 must expand regularity and non-uniqueness. |
| P1-C09 | `ch19b_dpf_literature_survey.tex`, `eq:bf-pff-homotopy-covariance`--`eq:bf-pff-edh-ode` | EDH follows from Gaussian closure and affine velocity matching. | approximate closure except in exact linear-Gaussian case | Chapter equations; Daum-Huang/source role pending P2; MathDev labels available from P0. | Gaussian predictive law and Gaussian/linear observation closure; ODE discretization error. | Implementation must separate closure inputs, affine coefficients, ODE integration, and log-det/Jacobian where used. | High: current text still formula-led; P4 must derive from precision evolution and velocity matching. |
| P1-C10 | `ch19b_dpf_literature_survey.tex`, `eq:bf-pff-kalman-endpoint` | EDH recovers the Kalman update in the linear-Gaussian case. | exact model identity in special case | Chapter equation; source support pending P2. | Truly linear-Gaussian model; exact Gaussian predictive law; exact integration. | Linear-Gaussian recovery should be a required implementation regression test. | Medium: mean recovery is asserted more briefly than covariance; P4 should expand. |
| P1-C11 | `ch19b_dpf_literature_survey.tex`, `eq:bf-pff-local-jacobian`--`eq:bf-pff-ledh-ode` | LEDH improves local adaptation by using particle-local linearizations. | approximate closure | Chapter equations; source role pending P2. | Observation function locally linearized at each particle; Jacobian stable; local covariance/information matrices well-conditioned. | Code must cache per-particle Jacobians, local affine coefficients, and failure diagnostics. | High: "improves" must be phrased as intended adaptation, not guaranteed accuracy. |
| P1-C12 | `ch19b_dpf_literature_survey.tex`, stiffness section | Pseudo-time stiffness and discretization can break nominal flow behavior. | engineering warning | Chapter text; source role pending P2. | Numerical solver tolerance and pseudo-time grid influence final cloud. | Flow tests must compare pseudo-time refinements and log-weight sensitivity. | Low to medium; should be equation-indexed in P9/P11. |
| P1-C13 | `ch19c_dpf_implementation_literature.tex`, `eq:bf-pfpf-postflow-density` | The post-flow proposal density follows from change of variables. | exact model identity | Chapter equation; derivation audit in P5/P11. | Invertible differentiable flow map; support preservation; nonzero Jacobian determinant. | Implementation must track forward map, inverse/pre-image, and log absolute determinant with a fixed sign convention. | High: reviewers will check sign/inverse convention. |
| P1-C14 | `ch19c_dpf_implementation_literature.tex`, `eq:bf-pfpf-weight` | PF-PF weights restore proposal-to-target density-ratio accounting for transported proposals. | exact model identity for the importance ratio; finite-N particle approximation for estimator | Chapter equation; PF-PF source role pending P2; MathDev labels available from P0. | Correct target/proposal definitions; invertible flow; finite densities. | Unit tests should compare corrected weights against brute-force transformed densities on affine toy flows. | High: P5 must derive one-step and trajectory-level ratios, not only state formula. |
| P1-C15 | `ch19c_dpf_implementation_literature.tex`, `eq:bf-pfpf-jacobian-ode`--`eq:bf-pfpf-logdet-affine` | Log-determinant evolution follows from the Jacobian ODE and trace identity. | exact model identity | Chapter equations; MathDev derivation obligation for P5/P11. | Sufficient differentiability of velocity field; nonsingular Jacobian along path. | Code can accumulate trace rather than materializing full Jacobians in affine cases. | High: P5 must show Jacobi formula and sign convention explicitly. |
| P1-C16 | `ch19c_dpf_implementation_literature.tex`, "what correction restores" | PF-PF correction does not automatically produce an exact nonlinear filter or HMC-ready target. | engineering warning / conservative downgrade | Chapter text; source support pending P2. | Flow construction, discretization, and finite particles remain. | HMC promotion requires value-gradient and variance diagnostics. | Low if retained; high if later prose overstates PF-PF. |
| P1-C17 | `ch32_diff_resampling_neural_ot.tex`, `eq:bf-dr-categorical-selection` | Categorical resampling is exact for the classical resampling law but not pathwise differentiable. | exact model identity plus differentiability limitation | Chapter equation; DPF/resampling source role pending P2. | Ancestor draw from categorical weights; discontinuity except measure-zero tie/boundary caveats. | Differentiating through realized ancestors is not a valid pathwise gradient route. | Medium: exact law must not be described as exact posterior inference by itself. |
| P1-C18 | `ch32_diff_resampling_neural_ot.tex`, `eq:bf-dr-soft-resampling`--`eq:bf-dr-soft-test-bias` | Soft resampling is a differentiable surrogate with nonlinear test-function bias. | relaxed target / learned-free surrogate | Chapter equation; source role pending P2. | Convex shrinkage/interpolation; bias depends on test function nonlinearity. | Tests should evaluate linear and nonlinear test functions separately. | Medium: current bias argument should be expanded before recommendations. |
| P1-C19 | `ch32_diff_resampling_neural_ot.tex`, `eq:bf-dr-unregularized-ot` | Unregularized OT is exact for the chosen transport projection, not for categorical resampling. | relaxed target | Chapter equation; OT sources pending P2. | Cost function, source weights, target weights, support choice define the projection. | Code must report cost, marginals, and support convention. | High: reviewers may object if "OT resampling" seems to replace categorical law without target change. |
| P1-C20 | `ch32_diff_resampling_neural_ot.tex`, `eq:bf-dr-eot-primal`--`eq:bf-dr-sinkhorn-form` | EOT is exact for the entropy-regularized problem; finite Sinkhorn is a numerical approximation. | relaxed target plus numerical approximation | Chapter equations; Cuturi/Peyre/Schmitzer/Corenflos roles pending P2. | Regularization parameter fixed; Sinkhorn iterations finite; stabilization required. | Diagnostics must include row/column residuals, epsilon sensitivity, iteration budget, and log-domain stabilization. | High: P6 must derive primal/dual/scaling enough for non-specialists. |
| P1-C21 | `ch19d_dpf_hmc_dsge_macrofinance_assessment.tex`, teacher/student equations | Learned OT is a learned approximation to an already relaxed transport teacher. | learned surrogate | Chapter equations; DeepSets/Set Transformer/Corenflos roles pending P2. | Training distribution, architecture equivariance, teacher quality, and generalization domain. | Validation must compare teacher residual, student residual, posterior sensitivity, and out-of-distribution failures. | High: no theorem currently supports posterior preservation for banking models. |
| P1-C22 | `ch19d_dpf_hmc_dsge_macrofinance_assessment.tex`, residual and target-shift tables | Residuals in map space, distribution space, posterior summaries, and HMC target status are not interchangeable. | engineering warning | Chapter tables; source support pending P2. | Discrepancy metric must match the claim being made. | Report residual hierarchy; never use training loss alone as posterior validity evidence. | Medium: table should become a stronger audit contract in P7/P9. |
| P1-C23 | `ch19e_dpf_hmc_target_suitability.tex`, `eq:bf-dpf-hmc-contract` | HMC requires value-gradient consistency for the same scalar target. | exact model identity for HMC implementation contract | HMC sources pending P2; equation present. | HMC transition uses a scalar potential and gradient of that potential; Metropolis correction tied to same target value. | Finite-difference and autodiff checks must verify same scalar path. | High: P8 must separate HMC correctness, pseudo-marginal validity, and surrogate-target coherence. |
| P1-C24 | `ch19e_dpf_hmc_target_suitability.tex`, EDH rung | EDH is exact-target HMC only in linear-Gaussian recovery; otherwise approximate benchmark. | exact special case / approximate closure | Cross-ref to particle-flow chapter; source pending P2. | Same EDH scalar and gradient; closure and discretization errors outside special case. | EDH HMC tests must include exact linear-Gaussian recovery and nonlinear sensitivity checks. | Medium. |
| P1-C25 | `ch19e_dpf_hmc_target_suitability.tex`, PF-PF rung and table | PF-PF is the first rung in the local DPF-HMC ladder with an explicit proposal-corrected value-side interpretation, but only as an experimental hypothesis. | engineering hypothesis | Chapter logic; source support pending P2. | Corrected value and gradient path; finite-particle variance controlled; log-det audited. | Before any recommendation, require repeated-evaluation stability, variance diagnostics, and value-gradient consistency. | Medium to high: the phrase "first serious" should remain absent from live recommendation text; any priority statement must stay evidence-gated. |
| P1-C26 | `ch19e_dpf_hmc_target_suitability.tex`, soft/OT/learned rungs | Soft, OT/EOT, and learned OT rungs may be coherent HMC targets only for their chosen relaxed or learned scalar. | relaxed target / learned surrogate | Chapter logic; source support pending P2. | The scalar target is explicitly changed; gradient matches same scalar; posterior drift accepted and measured. | HMC output must be labelled by target status and compared to trusted references. | High: avoid any implication that differentiability recovers original posterior. |
| P1-C27 | `ch19e_dpf_hmc_target_suitability.tex`, nonlinear DSGE/MacroFinance sections | Structural/banking models amplify target drift, invalid regions, and validation burden. | engineering hypothesis / banking hypothesis | Chapter reasoning; source support pending P2; no local validation evidence. | Model-specific constraints, determinacy/pruning/timing, high-dimensional latent states. | Requires model-specific stress tests and posterior comparisons before deployment language. | High: must remain conservative; no banking validation claim. |
| P1-C28 | `ch19e_dpf_hmc_target_suitability.tex`, recommendation section | BayesFilter should prioritize corrected PF-PF over raw flow or relaxed resampling for first DPF-HMC experiments. | engineering recommendation | Derived from ledger; source and implementation evidence pending. | Recommendation is for experimental priority, not production selection. | P8 must state required promotion evidence and stop short of validated HMC claim. | High: recommendation must not outrun evidence. |
| P1-C29 | `ch19f_dpf_debugging_crosswalk.tex`, failure-mode crosswalk | Debugging should proceed from target definition outward to lower-level numerical diagnostics. | engineering recommendation | Cross-refs to prior chapters; source support pending P2. | The target status must be known before interpreting numerical stability. | Crosswalk should become an equation-indexed test plan in P9. | Medium: table alone is not a validation proof. |
| P1-C30 | `ch19f_dpf_debugging_crosswalk.tex`, boundary section | The debugging crosswalk does not validate any DPF implementation. | conservative limitation | Chapter text. | Diagnostics identify failure modes; validation requires passing target-specific tests. | Result artifacts should distinguish diagnostics run from claims validated. | Low if preserved. |

## Claims to strengthen

These claims may be strengthened only by adding derivations, assumptions, and
source-role notes.

1. Bootstrap likelihood unbiasedness:
   - strengthen from proof sketch to conditional-expectation derivation with
     filtration and resampling assumptions in P3.
2. Homotopy and continuity-equation foundations:
   - strengthen by deriving endpoint identities, the log-homotopy derivative,
     and the conservation equation with regularity assumptions in P4.
3. EDH linear-Gaussian recovery:
   - strengthen by deriving both covariance and mean endpoints, not only
     stating the covariance endpoint in P4.
4. PF-PF proposal correction:
   - strengthen by deriving the target/proposal ratio at one-step and
     trajectory levels, with an explicit forward/inverse map convention in P5.
5. Log-determinant evolution:
   - strengthen by showing the Jacobian ODE, Jacobi formula, trace identity,
     affine simplification, and sign convention in P5/P11.
6. EOT/Sinkhorn:
   - strengthen by deriving the regularized primal problem, scaling form,
     marginal residuals, and finite-iteration approximation status in P6.
7. HMC contract:
   - strengthen by separating ordinary HMC target invariance,
     pseudo-marginal value-side validity, noisy-gradient failure modes, and
     surrogate-target coherence in P8.
8. Debugging crosswalk:
   - strengthen by making every diagnostic equation-indexed and by separating
     algebraic, numerical, statistical, and target-level tests in P9.

## Claims to weaken or qualify

These phrases are not necessarily wrong, but they are too strong unless
locally defined and evidenced.

1. `HMC-ready particle backend` in `ch19_particle_filters.tex`:
   - replace with "candidate value/gradient backend subject to target-status,
     variance, and implementation audits."
2. `first serious DPF-based HMC candidate` in
   `ch19e_dpf_hmc_target_suitability.tex`:
   - replace or define as "first rung in this ladder with an explicit
     proposal-corrected value-side interpretation."
3. `strong target story` for PF-PF:
   - qualify as "stronger than uncorrected flow transport because it includes an
     explicit proposal-density correction."
4. Any use of `production-ready`, `validated`, or `suitable`:
   - restrict to negative or conditional statements unless P8/P13 records
     concrete validation evidence.
5. `improves adaptation` for LEDH:
   - phrase as an intended/local-adaptation mechanism, not a guaranteed
     accuracy improvement.
6. `coherent HMC target` for relaxed or learned rungs:
   - qualify as coherent only for the explicitly chosen relaxed or learned
     scalar when value and gradient match.
7. Banking/MacroFinance suitability:
   - maintain as a validation burden and governance problem, not as an achieved
     capability.

## Missing prerequisite definitions

Later phases must add or consolidate definitions before making central claims.

1. Dominating measure and density convention for transition, observation,
   proposal, target, and empirical laws.
2. Support condition for importance sampling and change-of-variables ratios.
3. Filtration for the bootstrap likelihood unbiasedness proof.
4. Meaning of "unbiased likelihood estimator" versus "unbiased score" versus
   "pathwise differentiable scalar."
5. Homotopy normalizer and endpoint densities.
6. Regularity assumptions for the continuity equation and flow-map
   differentiability.
7. Velocity-field non-uniqueness and chosen EDH/LEDH closure.
8. Forward map, inverse/pre-image, Jacobian determinant, and log-det sign
   convention for PF-PF.
9. Categorical resampling law versus deterministic transport projection.
10. EOT regularization parameter, finite Sinkhorn iteration budget, and
    stabilization convention.
11. Teacher map, student map, training distribution, equivariance, and residual
    metrics for learned OT.
12. HMC scalar target, gradient object, value-gradient consistency, and
    target-status vocabulary.
13. Banking/structural-model validation evidence categories: analytic recovery,
    synthetic stress, posterior comparison, repeated compiled evaluation, and
    failure-region diagnostics.

## Chapter-order risks

1. `ch19_particle_filters.tex` uses "HMC-ready particle backend" only as a
   non-claim boundary item.  P10/P13 should keep it out of method labels and
   recommendation text unless a later implementation artifact proves the
   required target-status, variance, and value-gradient gates.
2. `ch19b_dpf_literature_survey.tex` states EDH/LEDH formulas before enough
   derivation for a skeptical non-specialist.  P4 must be derivation-led.
3. `ch19c_dpf_implementation_literature.tex` supplies the PF-PF correction
   before all target/proposal objects are defined at trajectory level.  P5 must
   close that gap.
4. `ch32_diff_resampling_neural_ot.tex` has several tables carrying
   central distinctions.  P6 must ensure tables summarize derivations rather
   than replace them.
5. `ch19d_dpf_hmc_dsge_macrofinance_assessment.tex` appears before the HMC
   suitability chapter in file naming and content flow; P7/P10 must keep learned
   OT clearly subordinate to the relaxed-transport and HMC target definitions.
6. `ch19e_dpf_hmc_target_suitability.tex` contains the recommendation before
   final debugging and hostile-reader audits.  P8 can make a provisional
   recommendation, but P13 must decide final readiness.
7. `ch19f_dpf_debugging_crosswalk.tex` is currently too short to govern
   implementation.  P9 must convert it from a crosswalk into an audit contract.

## Derivation obligations opened for later phases

| Phase | Obligation |
|---|---|
| P3 | Derive filtering recursion from Bayes' rule and prove the bootstrap likelihood estimator status with conditional expectations. |
| P4 | Derive homotopy endpoint identities, log-homotopy derivative, continuity equation, EDH covariance/mean evolution, and LEDH local precision/information vectors. |
| P5 | Derive PF-PF target/proposal density ratio, post-flow density, corrected weight, Jacobian ODE, and log-det trace identity. |
| P6 | Derive soft-resampling bias, unregularized OT projection, EOT primal/scaling form, and finite Sinkhorn residual status. |
| P7 | Derive teacher/student residual hierarchy and equivariance constraints; do not claim posterior preservation from architecture alone. |
| P8 | Derive the HMC target contract and pseudo-marginal/surrogate distinctions. |
| P9 | Convert diagnostics into equation-indexed tests with pass/fail interpretation limits. |
| P11 | Use MathDevMCP on load-bearing equations where possible and record unresolved symbolic obligations. |

## P1 audit result

Audit rules checked:

- Claims using `exact`, `unbiased`, `correct`, `valid`, `target`, `HMC`,
  `production`, `validated`, or `suitable` have local status classifications in
  the claim ledger above.
- HMC and banking claims are included and classified conservatively.
- Source support and derivation obligations are separated: most source support
  is marked pending P2 because ResearchAssistant has no local reviewed records.
- Tables that currently carry central claims are flagged for derivation-backed
  rewrite in later phases.

Veto diagnostics:

- No P1 veto fired.
- Caution: P2 is mandatory before chapter rewrites because ResearchAssistant
  did not provide local reviewed source support, and because several major
  claims currently rely on chapter citations whose precise source roles have
  not yet been audited.

Exit decision:

- P1 passes with mandatory P2 source-grounding follow-up.
- Completed handoff: P2 source-grounding was later completed in
  `docs/plans/bayesfilter-dpf-monograph-reviewer-grade-phase-p2-source-grounding-2026-05-15.md`.
  The current governing next gate is P3 baseline expansion, with provisional
  P4/P5 reconciliation before P6.
