# Nonlinear ZLB Filtering and Discontinuous HMC Survey Plan

Date: 2026-08-18

## Research question

Can a practically implementable Bayesian inference method preserve a genuine
zero-lower-bound regime discontinuity in a general nonlinear state-space model,
rather than smoothing the bound or restricting the model to a piecewise-linear
Kalman representation? Which results from discontinuous, reflective/refractive,
mixed discrete-continuous, and particle HMC are actually transferable to that
problem?

## Scope and baselines

The survey will derive the posterior for a nonlinear state-space model with an
occasionally binding lower bound and compare four method families:

1. OccBin and piecewise-linear Kalman filtering as the linear benchmark;
2. event-driven reflective/refractive HMC for piecewise-smooth continuous
   targets;
3. auxiliary-variable discontinuous and mixed HMC for discrete regime paths;
4. particle and pseudo-marginal state-space methods that can preserve nonlinear
   filtering uncertainty.

The primary application anchor is the ZLB work in
`/home/ubuntu/workspace/MacroFinance`. The user-named
`/home/ubuntu/workspace/dsge_hmc` path was absent at the start of the study.
(Corrected 19 August 2026, release revision: the `dsge_hmc` package exists at
`/home/ubuntu/workspace/python/src/dsge_hmc`; its validated BGS restricted surface is
a no-binding `r = rn` placeholder, so the true-OBC model remains an
unresolved future integration target.) The work belongs in BayesFilter
because BayesFilter is the active filtering and HMC authority for MacroFinance.

## Evidence contract

- **Primary criterion:** a self-contained linear exposition derives each
  candidate transition kernel far enough that a reader can implement it,
  identifies the invariant distribution and detailed-balance or
  volume-preservation argument, and states exactly what must change for a
  nonlinear ZLB state-space posterior.
- **Required comparisons:** ordinary smooth HMC, OccBin/piecewise Kalman,
  discontinuous HMC, mixed HMC, and particle-MCMC/state-space alternatives.
- **Vetoes:** reliance on abstracts for technical claims; treating a kink as a
  jump; claiming that a discontinuous-HMC paper already solves nonlinear ZLB
  inference; omitting the likelihood-estimation problem; silently smoothing the
  bound; or citing a source without an inspected technical anchor.
- **Explanatory diagnostics:** citation counts, venue indicators, repository
  code examples, and local project history. These guide coverage but cannot
  prove mathematical validity.
- **Nonconclusions:** the survey will not claim posterior correctness,
  production readiness, HMC convergence, source-faithful implementation, or
  superiority for any new BayesFilter route. Those require later implementation
  and model-specific evidence.

## Derivation obligations

The manuscript must derive:

- the ZLB complementarity/regime representation;
- the piecewise-linear conditional Kalman likelihood;
- refraction and reflection from Hamiltonian energy conservation;
- the coordinatewise discontinuous-HMC update with Laplace momentum;
- the mixed HMC composition and its invariance logic;
- the nonlinear state-space marginal likelihood and particle estimate;
- a valid extended-space posterior for particle MCMC; and
- a proposed event-regime particle HMC architecture, clearly labeled as a
  project synthesis rather than a theorem from the literature.

## Source and search procedure

Inspect the four local seed papers under
`google-drive-papers/bayesian and SSM/hamiltonian MC/discrete`, the local
OccBin/OBC and nonlinear-state-space sources, their backward references, and
bounded public metadata/forward-citation results from ResearchAssistant. Record
full-text anchors, citation metadata with access date, retraction/version
checks, snowball decisions, claim mappings, and omission risks in six separate
ledgers.

## Bounded work budget and stop conditions

- At most three bounded public discovery queries and 40 nominated records.
- No package installation, paid API, credential use, model execution, GPU run,
  or posterior campaign.
- Stop public discovery if the configured provider is unavailable; record the
  gap and continue with the strong local corpus.
- Do not overwrite prior evidence: ResearchAssistant output goes under the
  versioned `research_assistant_20260818` directory.

## Skeptical preflight

The plan distinguishes a discontinuous target from a merely nondifferentiable
one, makes nonlinear likelihood estimation part of the central problem, and
keeps linear OccBin/Kalman methods as a fair baseline rather than a straw man.
The primary deliverable is mathematical synthesis, not a numerical ranking, so
no experiment or proxy metric is being promoted. The main residual risk is
literature incompleteness; the ledgers and hostile-review pass make that risk
visible instead of hiding it.

## Result status (2026-08-18)

The bounded survey task is complete. The manuscript, rendered PDF/HTML,
16-entry bibliography, six scholarly ledgers, project roadmap, MathDevMCP
reports, and hostile-review record are present in this directory. The result is
`bounded_local_survey_with_explicit_gaps`: it is suitable for internal project
scoping and implementation planning, but it does not establish literature
completeness, posterior correctness, HMC convergence, or production readiness.

The configured ResearchAssistant public-provider mission stopped at
`terminal_blocked_bootstrap_unavailable`; that limitation and the resulting
metadata/search coverage gap are preserved in the ledgers. No implementation,
package/environment mutation, GPU run, or posterior campaign was performed.

## Result status update (2026-08-19)

On owner request, Section 4.3 (piecewise, truncated, and mixture UKF
construction) was added with full derivations, growing the bibliography to 25
entries. Four new sources were inspected as local full text; five pay-walled
sources are cited from verified metadata as orientation only. The section's
closed forms were spot-checked numerically
(`derivation_check_ukf_section_20260819.py`); equations were renumbered into
one sequence; and the ledgers, rendered PDF/HTML, and hostile review were
updated. The overall result classification is unchanged.

## Result status update (2026-08-19, second pass)

The pre-implementation enhancement plan
(`survey_enhancement_plan_20260819.md`) was executed on owner request.
Additions: Section 5.1 (validity of Metropolized HMC on kinked targets),
Section 13 (the MacroFinance shadow-rate contract, derived from five
inspected code modules, with the crossing lemma, the hard-bound
observation-map cell decomposition, softplus bias bounds, identification
analysis, and a
route claim table), Section 14 (the dsge_hmc contract: LCP
representation, the fully derived lagged-rule multiplicity example, the
selection-jump mechanism, the sunspot completion, and estimation baselines),
Section 15 (solver-induced discontinuities), and twenty audited sources
(sixteen local full-text inspections, four metadata-only), for 45 references
at that pass. New closed forms are spot-checked in
`derivation_check_contract_sections_20260819.py` (all pass). `dsge_hmc`
remained absent at the originally named path on 2026-08-19 (see the
correction above for the actual package location). The overall result
classification is
unchanged: a bounded local survey and design artifact with explicit gaps,
now including two application target contracts.

A release revision (2026-08-19/20) executed the audit
`zlb_discontinuous_hmc_survey_audit_20260819.md` under
`docs/plans/zlb-discontinuous-hmc-survey-release-rewrite-plan-2026-08-19.md`:
dsge_hmc identity corrected package-wide; the Sec. 13 target ladder
(S1/C1/C0) introduced with the bootstrap PF as C1 authority and the cell
filter reclassified as an ideal proposal identity; the four-route table
extended to five routes; general-mass-matrix event maps, kernel-based measure
foundations, explicit PM-HMC assumptions, LCP parametric-regularity
conditions, tail-stable Mills-ratio requirements, and the signed
softplus-likelihood identity added; Pericoli--Taboga (2018, inspected) plus
five metadata-cited sources added, for 51 references in total.
