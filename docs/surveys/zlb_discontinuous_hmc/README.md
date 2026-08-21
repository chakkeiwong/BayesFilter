# Nonlinear ZLB and Discontinuous HMC Survey Package

This directory contains the 18--19 August 2026 literature survey and project
design for Bayesian inference in nonlinear state-space models with a genuine
zero or effective lower bound. Section 4.3 (piecewise, truncated, and mixture
UKF construction) was added on 19 August 2026; a second same-day pass executed
the enhancement plan, adding Section 5.1 (kinked-HMC validity), Section 13
(the MacroFinance shadow-rate contract, derived from inspected code), Section
14 (the dsge_hmc contract with the worked multiplicity example), and Section 15
(solver-induced discontinuities). A release revision (19--20 August 2026)
executed the audit `zlb_discontinuous_hmc_survey_audit_20260819.md` under
`docs/plans/zlb-discontinuous-hmc-survey-release-rewrite-plan-2026-08-19.md`:
it corrected the dsge_hmc package identity across the package, introduced the
MacroFinance target ladder (S1/C1/C0) with the bootstrap particle filter as
the hard-bound likelihood authority and the cell decomposition demoted to an
ideal proposal identity, added the general-mass-matrix event kernel, the
Markov-kernel measure foundations, the explicit pseudo-marginal-HMC
assumptions, and the Pericoli--Taboga closure-free comparator, and split
Section 14 into pedagogical and source-anchored BGS layers.

## Reader path

1. [Internal survey record (Markdown)](zlb_discontinuous_hmc_survey.md)
   --- the technical record the audit ledgers anchor to; equation numbers
   in the ledgers refer to this file's tags
2. [Implementation roadmap](project_roadmap.md)
3. [Pre-implementation enhancement plan](survey_enhancement_plan_20260819.md)
4. [2026-08-19 audit](zlb_discontinuous_hmc_survey_audit_20260819.md) and the
   [release rewrite plan](../../plans/zlb-discontinuous-hmc-survey-release-rewrite-plan-2026-08-19.md)
5. [LaTeX manuscript](zlb_discontinuous_hmc_survey.tex) --- **the
   human-facing primary artifact.** This is a full prose rewrite of the
   survey for a reader with no knowledge of any codebase: every section
   opens with motivation, the derivations carry their connecting steps, the
   two applications are presented as self-contained case studies (a
   two-country shadow-rate term-structure model; a New Keynesian model with
   an occasionally binding policy rate), and all internal file paths,
   project names, and governance vocabulary have been removed. The three
   model variants are named $\mathcal{S}_1/\mathcal{C}_1/\mathcal{C}_0$
   in the manuscript; they correspond one-to-one to the `mf_s1_k40_softplus`
   / `mf_c1_k40_hardmax` / `mf_c0_root_integral_hardmax` target identifiers
   used by the internal ledgers and the Markdown. Compile with pdflatex
   (three passes for the TOC).
6. [Rendered PDF](zlb_discontinuous_hmc_survey.pdf) (46 pp., abstract +
   hyperlinked TOC)
7. [Rendered HTML](zlb_discontinuous_hmc_survey.html) (rendered from the
   internal Markdown, not from the LaTeX manuscript)

The central result is a target classification, not a universal sampler:

- a continuous hard `max` is a kink;
- a deterministic branch with unequal one-sided densities is a continuous-space
  jump target;
- a probabilistic regime is a mixed discrete-continuous target; and
- multiple equilibria require a statistical selection law before inference.

## For documentation tooling

- [Humanvoice case study](humanvoice_case_study.md) --- what went wrong (and
  how it was fixed) turning this package's internal survey into the
  human-facing LaTeX manuscript; written for the humanvoice agent building
  doc tooling. Companion: [style contract](latex_manuscript_style_contract.md).

## Scholarly audit

- [Source support](source_support.md)
- [Citation and venue metadata](citation_venue_metadata.md)
- [Backward snowball](backward_snowball.md)
- [Forward snowball](forward_snowball.md)
- [Claim support](claim_support.md)
- [Omitted papers and reviewer risks](omitted_papers.md)
- [Bibliography](references.bib)
- [Hostile review](hostile_review.md)
- [Bounded Claude review bundle](claude_review_bundle.md)
- [MathDevMCP audit, 18 Aug scope](mathdevmcp_audit.md) and
  [19 Aug focused re-run on the new equations](mathdevmcp_audit_20260819.md)
- [Section 4.3 numerical derivation check](derivation_check_ukf_section_20260819.py)
  and its [log](derivation_check_ukf_section_20260819.log)
- [Sections 13/14/5.1 numerical derivation check](derivation_check_contract_sections_20260819.py)
  and its [log](derivation_check_contract_sections_20260819.log)

`research_assistant_20260818/` preserves the ResearchAssistant mission state.
Its provider bootstrap stopped honestly at
`terminal_blocked_bootstrap_unavailable`; direct read-only metadata checks and
the strong local paper corpus were used to continue.

## Scope

The package is a literature and design artifact. It does not implement a
sampler, run a posterior campaign, choose an equilibrium-selection law, or
establish posterior correctness, HMC convergence, default readiness, or
production readiness. **Correction (19 August 2026, release revision):** the
originally requested path `/home/ubuntu/workspace/dsge_hmc` is absent, but the
`dsge_hmc` package exists at `/home/ubuntu/workspace/python/src/dsge_hmc`; its
validated BGS restricted surface is a no-binding linear placeholder
(`r = rn`, rows tagged `OBC_ZLB_NO_RUN_GUARD`), so a true OBC/ZLB model
remains an unresolved integration target and survey Sections 14.5--14.6 are
the contract it must satisfy. The central application
findings are that the MacroFinance model as coded is a hard-bound
observation-map kink target whose claim-bearing likelihood authority is a
bootstrap particle filter for the finite hard-quadrature target
(`mf_c1_k40_hardmax`), with the per-step cell decomposition available as an
ideal proposal identity, and that the genuine DSGE-side discontinuity lives
in the solution correspondence and its selection law rather than in the max
operator.
