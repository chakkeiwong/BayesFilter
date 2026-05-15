# Phase P3 result: classical filtering and SMC baseline expansion

## Date

2026-05-15

## Branch and worktree classification

- Branch: `main`.
- Divergence: `main...origin/main [ahead 2]`.
- In-lane DPF chapter changes present before or during this phase:
  - `docs/chapters/ch19_particle_filters.tex`;
  - `docs/chapters/ch19b_dpf_literature_survey.tex`;
  - `docs/chapters/ch19c_dpf_implementation_literature.tex`;
  - `docs/chapters/ch19d_dpf_hmc_dsge_macrofinance_assessment.tex`;
  - `docs/chapters/ch19e_dpf_hmc_target_suitability.tex`;
  - `docs/chapters/ch19f_dpf_debugging_crosswalk.tex`;
  - `docs/chapters/ch32_diff_resampling_neural_ot.tex`.
- In-lane reviewer-grade planning/result artifacts are present as untracked
  files.
- User-owned in-lane governance edit remains present:
  - `docs/plans/bayesfilter-dpf-monograph-reviewer-grade-revision-master-program-2026-05-14.md`.
- Out-of-lane student-baseline and controlled-baseline experiment files remain
  present and were not touched.

## Conditional execution note

The user's execution order placed P3 after P7 and only "if needed."  During
the P10 consolidation scan, P3 was found to be needed: the later DPF chapters
had reviewer-grade object inventories and claim-status boundaries, while the
classical particle-filter baseline was still thinner.  P3 was therefore run as
a required backfill before continuing P10.

## Allowed write set used

- `docs/chapters/ch19_particle_filters.tex`.
- This result artifact.

No student-baseline file, controlled-baseline experiment file, git history
operation, deletion, push, merge, rebase, reset, or staging operation was
performed.

## Chapter changes completed

P3 expanded the classical particle-filter baseline so later DPF layers can be
judged against a clear reference object.

Completed items:

1. Added an object/notation inventory for latent states, observations,
   parameters, transition and observation densities, predictive and filtering
   laws, likelihood factors, proposal densities, weights, ancestors, empirical
   measures, average bootstrap weights, and the bootstrap likelihood estimator.
2. Added standing assumptions: dominated kernels, proposal support, finite
   integrals, and conditionally unbiased resampling.
3. Added claim-status vocabulary distinguishing exact model identities,
   finite-particle approximations, unbiased likelihood estimation, and
   diagnostics.
4. Expanded the filtering recursion derivation from Chapman--Kolmogorov,
   Bayes' rule, conditional independence, and likelihood-factor chain rule.
5. Expanded SIS conditioning and support requirements, and clarified that the
   trajectory importance ratio is a target/proposal correction rather than a
   differentiable likelihood map.
6. Clarified the bootstrap/SIR algorithm around ancestor selection, propagation,
   weight recording, and optional resampling.
7. Strengthened the bootstrap likelihood-estimator proposition with assumptions,
   filtration notation, conditional-expectation steps, tower-property induction,
   and explicit separation from log-likelihood and score unbiasedness.
8. Added a differentiability-boundary section separating the true likelihood,
   the unbiased randomized likelihood estimator, and a pathwise differentiable
   relaxed/transformed/learned scalar.
9. Added a baseline audit table separating exact identities, Monte Carlo
   estimator claims, diagnostics, and pathwise-gradient non-claims.

## ResearchAssistant evidence

ResearchAssistant remains available as a read-only/offline local workspace.

Query run:

- `bootstrap particle filter sequential Monte Carlo likelihood estimator
  unbiased particle marginal likelihood`.

Result: no local paper summaries.  Therefore P3 does not describe SMC source
support as ResearchAssistant-reviewed.  Source support remains
bibliography-spine plus local derivation and claim-boundary text.

## Source and citation checks

The required citation keys exist in `docs/references.bib`:

- `gordon1993novel`;
- `doucet2001sequential`;
- `chopin2020introduction`;
- `bengtsson2008curse`;
- `snyder2008obstacles`;
- `andrieu2009pseudo`;
- `andrieu2010particle`.

No bibliography file edit was needed.

## Derivation-obligation audit

| Obligation | Chapter location | Method | Result |
|---|---|---|---|
| Filtering prediction recursion | `eq:bf-pf-predictive-law` and surrounding derivation | Manual Chapman--Kolmogorov audit | Passed. The text now derives prediction from conditional transition and prior filtering law. |
| Filtering update recursion | `eq:bf-pf-filtering-law` and surrounding derivation | Manual Bayes-rule audit | Passed. The text now identifies the predictive law as prior, the observation density as likelihood, and the predictive observation density as normalizer. |
| Marginal likelihood factorization | `eq:bf-pf-marginal-factorization` | Manual chain-rule audit | Passed. The text states the chain-rule product and empty-history convention. |
| SIS trajectory ratio | `eq:bf-pf-trajectory-proposal`--`eq:bf-pf-sis-recursion` | Manual target/proposal audit | Passed. The text states conditioning variables, proposal support, and trajectory-level ratio status. |
| Bootstrap specialization | `eq:bf-pf-bootstrap-proposal`--`eq:bf-pf-bootstrap-weight` | Manual specialization audit | Passed. The bootstrap proposal cancels the transition density in the incremental ratio. |
| Bootstrap likelihood-estimator status | `prop:bf-pf-bootstrap-likelihood-status` | MathDevMCP label lookup plus manual measure-theoretic review | Passed manually with explicit caveat. MathDevMCP found the label and context but did not certify the proof; its derivation audit returned `unverified/inconclusive`, which is appropriate for a Feynman--Kac conditional-expectation argument rather than a scalar symbolic identity. |
| Likelihood/log-likelihood/score boundary | proposition text and `sec:bf-pf-differentiability-boundary` | Manual claim-status audit | Passed. The chapter explicitly denies unbiased log-likelihood and score claims. |
| Resampling differentiability boundary | `sec:bf-pf-differentiability-boundary` and `tab:bf-pf-baseline-audit` | Manual pathwise-differentiability audit | Passed. The chapter states that categorical resampling is not a smooth fixed-seed map in general. |

## Local checks run

- `git status --short --branch`.
- Citation-key search in `docs/references.bib`.
- Label uniqueness spot checks for:
  - `sec:bf-pf-objects`;
  - `sec:bf-pf-differentiability-boundary`;
  - `tab:bf-pf-baseline-audit`.
- Text audit `rg` for:
  - `unbiased`;
  - `log likelihood`;
  - `log-likelihood`;
  - `score`;
  - `exact`;
  - `differentiab`;
  - `HMC-ready`;
  - `ResearchAssistant`;
  - `reviewed`.
- ResearchAssistant query listed above.
- MathDevMCP:
  - `latex_label_lookup` for `prop:bf-pf-bootstrap-likelihood-status`;
  - `audit_derivation_v2_label` for `prop:bf-pf-bootstrap-likelihood-status`,
    which returned `unverified/inconclusive` and was recorded as manual-review
    evidence rather than certification.
- LaTeX build:
  - working directory: `docs`;
  - command: `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`;
  - result: passed after required reruns;
  - final PDF size: 217 pages.
- Post-build log scan:
  - no undefined citations;
  - no undefined references;
  - no rerun warnings;
  - no multiply defined label warnings found by targeted `rg`.

## Text-audit findings

Flagged words remain in controlled contexts:

- `exact` is used for model identities, exact distributions, and the point at
  which later methods depart from the classical construction.
- `unbiased` is tied to likelihood estimation and conditionally unbiased
  resampling; the chapter explicitly denies unbiased log-likelihood and score
  claims.
- `differentiable` appears in boundary language explaining what the classical
  filter does not provide.
- `HMC-ready` appears only in the final non-claim list.
- No text describes source support as ResearchAssistant-reviewed.

## Layout inspection

The build passes, but P3 adds table pressure.  P3-local warnings include:

- a small overfull paragraph around lines 124--129;
- a small overfull paragraph around line 241;
- an overfull page-header warning of about 22.6pt after the expanded chapter;
- a small overfull paragraph around lines 421--429;
- underfull warnings in the baseline audit table around lines 462--489.

These are not mathematical vetoes, but P10/P12 should consider table layout and
header pressure across the full DPF block.

## P4/P5 impact note

P3 clarifies baseline objects and claim-status vocabulary without changing the
meaning of the predictive law, filtering law, bootstrap likelihood estimator,
trajectory proposal, normalized weights, or empirical measure used by P4 and
P5.  No P4/P5 repair is required before P10.  P10 should still harmonize the
cross-chapter notation index and claim-status vocabulary.

## Veto diagnostic review

- Likelihood-estimator proof remains only a citation-backed sketch: no.
- Conditioning is ambiguous in the SIS derivation: no.
- Unbiased likelihood and log-likelihood are blurred: no.
- Resampling differentiability is treated informally: no.
- Later DPF chapters still rely on undefined baseline objects: no immediate
  blocker; P10 will consolidate the cross-chapter index.
- Student-baseline or controlled-baseline files edited or staged: no.
- ResearchAssistant-empty source status converted into source-reviewed support:
  no.
- MathDevMCP/manual derivation-obligation evidence omitted: no.
- Build status omitted: no.
- P4/P5 impact note omitted: no.

## Exit gate

Passed with layout caution.

A mathematically mature non-specialist can now identify the exact filtering
target, the particle approximation, the bootstrap likelihood estimator, and the
differentiability gap without leaving the document.

## Next recommended phase

Resume P10:

- `docs/plans/bayesfilter-dpf-monograph-reviewer-grade-phase-p10-notation-claim-consolidation-plan-2026-05-14.md`.
