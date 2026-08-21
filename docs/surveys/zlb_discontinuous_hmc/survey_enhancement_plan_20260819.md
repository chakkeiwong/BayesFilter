# Survey Enhancement Plan Before Implementation

Date: 2026-08-19. Status: **executed on 2026-08-19 (second pass)** — see the
execution record at the end of this file. The original plan text is preserved
below unchanged.

This plan records the gap analysis between the current survey package and the
two application targets, based on direct inspection of
`/home/ubuntu/workspace/MacroFinance` on this date. It supersedes nothing; it
sequences the expansion work already flagged in `hostile_review.md` and
`omitted_papers.md` and adds application-driven items.

## Inspected application facts (2026-08-19)

- `/home/ubuntu/workspace/dsge_hmc` still does not exist. Its "genuine
  discontinuous regime" is intent, not code.
- MacroFinance two-currency double-ZLB model
  (`two_currency_double_zlb_{math,contract,target}.py`): 8-dim linear-Gaussian
  OU state (two DNS factor triples + two basis factors); measurement = yields
  as 40-node Gauss-Legendre averages of lower-bounded instantaneous forwards
  that are affine in the factors at fixed decay, plus one raw log FX forward;
  the bound is always the smooth map `lb + alpha*softplus((f-lb)/alpha)` with
  production `alpha = 2.0e-3`; active inference = BayesFilter direct-factor
  SR-UKF marginal likelihood with custom-gradient score, HMC/NeuTra over a
  9-parameter chart (`one_country_zlb_ns_estimation.py`, `dz5_*`).
- No regime-dependent transition exists in that model (no `tf.where`, regime,
  or switching term in the transition path).
- A separate, solver-induced discontinuity family exists in the CCMA lineage:
  the Padé(13)/scaling matrix-exponential kernel is branch-discontinuous in its
  derivative at norm thresholds (`ccma_g_v7_pade_frechet_diagnostic.py`).

## Consequences the survey does not yet draw

1. Under the exact `max`, the MacroFinance measurement is continuous
   piecewise-affine in the state, so the exact-model posterior is geometry
   case 1 (kink), not case 2 (density jump), of survey Sec. 2.1. Event
   (reflection/refraction) HMC and discontinuous HMC are not the binding tools
   for this model as coded. A case-2 target arises only if the owners adopt
   regime-dependent dynamics at the bound; that is a modeling decision the
   contract must pin down.
2. Linear-Gaussian transition + continuous piecewise-affine measurement is a
   PLC-class model: conditional on a binding pattern the model is
   linear-Gaussian, the exact filtering density is a truncated-Gaussian
   mixture over polyhedral cells, and the Aruoba-style COPF and the Sec. 4.3
   truncated-moment machinery apply exactly. An exact (or exactly weighted)
   likelihood authority for MacroFinance is derivable; the survey currently
   presents Sec. 4.3 only as an approximation-plus-correction layer.
3. The production softplus route is a declared approximate target with no
   error analysis in the survey; the UKF marginal likelihood is likewise a
   declared target with no bias diagnostic defined near binding.
4. The DSGE-side discontinuity that practitioners actually observe is
   parameter-space: the set of OccBin regime-path fixed points can change
   discretely with theta even when policy functions are continuous
   piecewise-linear in states. The survey's multi-root section covers this
   only abstractly.

## Phase A: application target contracts (write first; unblocks everything)

- **A1 MacroFinance contract chapter.** Derive from code: state ordering,
  parameters, the quadrature measurement map, FX identity, noise; the softplus
  model as a declared target; the exact-max variant; a code-anchored geometry
  classification of both (kink for the current model; jump only under a
  regime-dependent-dynamics extension, to be stated explicitly if adopted).
  Exit: the finite log-density program of both variants in survey notation.
- **A2 dsge_hmc benchmark contract chapter.** Choose the smallest credible
  NK-ZLB model; derive its OccBin solution matrices in survey notation; prove
  continuity/piecewise linearity of the solved policy function in states and
  shocks; derive the theta-space likelihood-jump mechanism via discrete
  changes of the regime-path fixed-point set; state the
  multiplicity/sunspot variant and a candidate selection law (from the
  Aruoba--Cuba-Borda--Schorfheide line) as the genuinely stochastic-regime
  target where mixed HMC/PGAS applies. Exit: targets (71)/(73) instantiated.

## Phase B: new theory sections

- **B1 Exact censored-measurement filter.** Specialize Sec. 4.3 from
  transition branching to measurement censoring: binding-pattern cells from
  forward-curve crossings on the quadrature grid, per-cell linear-Gaussian
  updates, truncated-Gaussian mixture recursion, COPF-style exactly weighted
  particle version, and pruning as a declared-target modification.
- **B2 Softplus error analysis.** Pointwise `|softplus-max| <= alpha*log 2`
  propagated through quadrature and measurement noise to likelihood and
  posterior perturbation bounds; alpha-sensitivity protocol anchored to the
  existing `dz5_alpha_counterfactual` fixtures; the alpha->0 gradient-stiffness
  trade-off for HMC step size, mass, and NeuTra.
- **B3 Kinked-potential HMC validity.** Formal statement and proof sketch for
  Metropolized leapfrog HMC on continuous, a.e.-differentiable, piecewise-smooth
  potentials with measure-zero kink sets; why event mechanics degenerate when
  Delta U = 0; energy-error behavior at kinks. Reclassify Pakman--Paninski
  exact/truncated-Gaussian HMC from omitted to inspected: it is directly
  relevant to truncated-Gaussian state blocks.
- **B4 Route comparison for MacroFinance.** Joint (theta, x_{0:T}) HMC on the
  exact-max model (non-centered shock chart; no filtering approximation),
  versus UKF-marginal declared-target HMC (current route; define its
  near-binding bias diagnostic against the B1 authority), versus PF-marginal
  PMMH. State which claims each route can carry.
- **B5 Identification near the bound.** Information degeneracy of shadow
  parameters during long binding spells; consequences for priors, HMC
  conditioning, and claim design.
- **B6 Solver-branch discontinuities.** Model-level versus implementation-level
  discontinuity (Pade/expm scaling branches, QR/SVD sign and ordering
  branches, factor downdates near rank loss); branch-stability requirements
  and a test checklist feeding BayesFilter.

## Phase C: literature audit expansion (six ledgers, per the scholarly policy)

- **C1 Shadow-rate term structure** (domain baselines for MacroFinance,
  currently absent from the survey). Verified locally available in full text
  under `google-drive-papers/finance/yield curve/ZLB/` and
  `.../discrete time/` on 2026-08-19: Krippner (2012); Priebsch (2013); Kim
  (2013) multi-factor shadow-rate estimation; Wu--Xia (2015/2016);
  Bauer--Rudebusch (2015); Christensen--Rudebusch (2015); Lemke--Vladu (2017);
  Roussellet (2021); Han (2023); Kim (2024); and Opschoor et al. (2024),
  *A Smooth Shadow-Rate Dynamic Nelson-Siegel Model for Yields at the Zero
  Lower Bound*, which is the closest published counterpart of the MacroFinance
  softplus-DNS design and must be audited for both C1 and B2. Black (1995)
  remains a metadata-level historical anchor if no local copy exists.
- **C2 Censored/Tobit Kalman filtering** (Allik et al.; the censored-
  measurement fusion survey already in the local corpus).
- **C3 Execute existing expansion-required rows**: Holden 2016/2022; Boehl;
  Cuba-Borda et al. likelihood line; Aruoba--Cuba-Borda--Schorfheide sunspot
  equilibria; Fernandez-Villaverde et al. nonlinear ZLB; Gust et al.
- **C4** Gaussian-splitting primary source (Sec. 4.3.2 currently rests on
  project reasoning); differentiable-particle-filter refresh 2022--2026.

## Phase D: verification hardening

- **D1** Re-run MathDevMCP on a labeled-equation LaTeX export (its earlier
  zero-labelled-equation scope covers neither Sec. 4.3 nor the new chapters);
  retry the bounded Claude review (`claude_review_bundle.md`).
- **D2** Extend `derivation_check_ukf_section_20260819.py` to the B1 cell
  moments, B2 bias bounds, and finite-difference checks of the B4 joint
  gradient on a toy.
- **D3** Rewrite roadmap WP0--WP3 exit criteria per contract: MacroFinance
  authority = B1 exact mixture/COPF filter; dsge_hmc authority = particle
  filter plus enumeration toys.

## Explicit non-actions before Phase A/B complete

- No discontinuous-HMC or mixed-HMC implementation for the current
  MacroFinance model; its coded target has the wrong geometry for those
  kernels.
- No promotion of the softplus or UKF-marginal posteriors to exact-model
  claims.
- No dsge_hmc implementation before its contract chapter (A2) and the
  Holden/sunspot audit (C3).

## Non-claims

Repository inspection was bounded to the files named above; this plan asserts
nothing about uninspected MacroFinance lineages. The geometry classification
applies to the model as coded on 2026-08-19 and must be re-derived if the
model adopts regime-dependent dynamics. Nothing here establishes posterior
correctness, HMC readiness, or production readiness.

## Execution record (2026-08-19, second pass)

- **A1 done.** Survey Sec. 13.1--13.3: the code-derived contract, geometry
  classification (kink for the model as coded, with the three jump-restoring
  extensions named), and the crossing lemma, eqs. (75)--(82).
- **A2 done at contract level.** Survey Sec. 14: LCP representation (93), the
  fully derived lagged-rule multiplicity example (94)--(96), the
  selection-jump mechanism (97), the sunspot completion, estimation
  baselines, and the Sec. 14.5 checklist. The repository itself still does
  not exist.
- **B1 done.** Sec. 13.4, eqs. (83)--(87): exact per-step cell filter and
  exactly weighted conditionally optimal particle authority.
- **B2 done.** Sec. 13.5, eqs. (88)--(91), including the two readings of
  alpha and the alpha-halving protocol.
- **B3 done.** Sec. 5.1 with (56a); Pakman--Paninski reclassified.
- **B4 done.** Sec. 13.7 route table; roadmap Sec. 16 updated.
- **B5 done.** Sec. 13.6 with eq. (92) and the inspected sensitivity
  findings.
- **B6 done.** Sec. 15 with (98) and the Pade-branch implementation evidence.
- **C1 done.** Nine shadow-rate sources audited (eight local full text, Black
  metadata); Opschoor--van der Wel audited first as planned.
- **C2 done at survey scope.** Geng et al. survey inspected; Allik et al.
  metadata-cited; noise-ordering distinction derived.
- **C3 done for Holden (2016, 2017, 2023), Boehl (2022), Boehl--Strobel
  (2023), Lubik--Schorfheide (2003); ACS (2018) metadata-cited.** The
  Cuba-Borda/inversion-filter line, Gust et al., and Fernandez-Villaverde et
  al. remain open in `omitted_papers.md`.
- **C4 partially done.** van der Merwe et al. audited; Gaussian-splitting
  primary source and the 2022--2026 differentiable-PF refresh remain open.
- **D1 done except the Claude retry.** Rendering, citation, and
  cross-reference checks rerun; MathDevMCP `audit-math-document-rigor` re-run
  on a labeled-equation export focused on the 25 new equations (one
  actionable proposal, applied; report archived in the package). The bounded
  Claude review was not retried.
- **D2 done.** `derivation_check_contract_sections_20260819.py`, all checks
  pass, log preserved.
- **D3 done.** Roadmap Phases 0, 2, 3, and 5 updated to name the
  contract-specific authorities and gates.
- **Correction to the facts block above:** deeper fixture inspection found the
  coded sharpness constants are alpha_d = 1.5e-3 and alpha_f = 1.0e-3
  (`DOMESTIC_CONFIG`/`FOREIGN_CONFIG` in `two_currency_double_zlb_fixtures.py`);
  the 2.0e-3 value above is the dz5 foreign-alpha revalidation counterfactual,
  not the production constant. Survey Sec. 13.1 records the correct values.

## Correction appendix (19 August 2026, release revision)

The inspected-facts bullet "`/home/ubuntu/workspace/dsge_hmc` still does not exist" is
false as a workspace statement (preserved above as written). The `dsge_hmc`
package exists at `/home/ubuntu/workspace/python/src/dsge_hmc`; its validated BGS
restricted surface is a no-binding `r = rn` placeholder (rows tagged
`OBC_ZLB_NO_RUN_GUARD`) whose evidence contract excludes OBC/ZLB estimation.
See the audit `zlb_discontinuous_hmc_survey_audit_20260819.md` (P0-1) and the
release rewrite plan
`docs/plans/zlb-discontinuous-hmc-survey-release-rewrite-plan-2026-08-19.md`.
