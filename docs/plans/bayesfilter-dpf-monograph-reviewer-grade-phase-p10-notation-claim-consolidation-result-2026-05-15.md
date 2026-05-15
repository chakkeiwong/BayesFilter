# Phase P10 result: cross-chapter notation, claim-status, and literature consolidation

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

## Allowed write set used

- `docs/chapters/ch19_particle_filters.tex`.
- This result artifact.

No student-baseline file, controlled-baseline experiment file, git history
operation, deletion, push, merge, rebase, reset, or staging operation was
performed.

## Prerequisite reconciliation

P10 initially found that conditional P3 was needed because the classical
particle-filter baseline lacked the same object-inventory and claim-status
machinery as P4--P9.  P3 was completed before this P10 gate and its result
records that the new baseline definitions do not invalidate P4/P5.

## Consolidation changes completed

P10 made the DPF block read as one governed argument by adding a block-level
index near the start of the DPF sequence.

Completed items:

1. Added `sec:bf-dpf-block-notation-index` to
   `docs/chapters/ch19_particle_filters.tex`.
2. Added `tab:bf-dpf-notation-index`, covering state and observations,
   parameters, filtering laws, particles and ancestors, weights and empirical
   measures, likelihood estimates, proposals, flow maps/Jacobians, transport
   couplings, Sinkhorn scalings, learned maps, and HMC scalar/gradient objects.
3. Added `tab:bf-dpf-claim-status-index`, covering exact model identity,
   unbiased particle estimator, consistent approximation, approximate closure,
   relaxed target, learned surrogate, engineering hypothesis, and unsupported
   claim.
4. Harmonized the baseline chapter with the claim-status vocabulary required by
   later chapters.
5. Confirmed P4--P9 chapters already contain object inventories, assumptions or
   source-role sections, diagnostic sections, and boundary/non-claim sections.
6. Confirmed cross-chapter source support still follows P2's
   bibliography-spine fallback.  No DPF source support is described as
   ResearchAssistant-reviewed.

## ResearchAssistant evidence

ResearchAssistant remains available as a read-only/offline local workspace.

Query run:

- `differentiable particle filter particle flow optimal transport Sinkhorn HMC
  target correctness`.

Result: no local paper summaries.  P10 therefore did not upgrade any source
claim to ResearchAssistant-reviewed support.  Source language remains
bibliography-spine plus local derivation and local claim-boundary text.

## Source and citation checks

The expected DPF bibliography-spine keys exist in `docs/references.bib`,
including:

- SMC and PF: `gordon1993novel`, `doucet2001sequential`,
  `chopin2020introduction`, `bengtsson2008curse`, `snyder2008obstacles`;
- particle flow/PF-PF: `daumhuang2008`, `li2017particle`, `hu2021particle`;
- differentiable resampling/OT/Sinkhorn: `zhumurphyjonschkowski2020`,
  `corenflos2021differentiable`, `villani2003topics`, `cuturi2013sinkhorn`,
  `peyre2019computational`, `schmitzer2019stabilized`;
- learned maps: `zaheer2017deep`, `lee2019set`;
- HMC/pseudo-marginal/surrogate: `neal2011mcmc`,
  `betancourt2017conceptual`, `andrieu2009pseudo`, `andrieu2010particle`,
  `greydanus2019hamiltonian`.

No bibliography edit was needed.

## Derivation and consolidation-obligation audit

| Obligation | Location | Method | Result |
|---|---|---|---|
| Block notation index covers all required P10 categories | `tab:bf-dpf-notation-index` | Manual table audit | Passed. The table covers all required categories in the P10 plan. |
| Claim-status vocabulary is explicit and bounded | `tab:bf-dpf-claim-status-index` | Manual claim-ledger audit | Passed. Each status phrase has a use condition and disallowed shortcut. |
| New labels are unique | `sec:bf-dpf-block-notation-index`, `tab:bf-dpf-notation-index`, `tab:bf-dpf-claim-status-index` | Direct `rg` label checks | Passed. Each new label appears once. |
| MathDevMCP label lookup for new claim-status table | `tab:bf-dpf-claim-status-index` | MathDevMCP label lookup attempt | Tool-limited. The lookup failed with a tool execution error, while direct `rg` and the LaTeX build confirmed the label exists and compiles. |
| Claim-status word usage | all DPF chapters | Manual `rg` audit | Passed. Uses of `exact`, `unbiased`, `relaxed`, `surrogate`, `valid`, and `validated` remain bounded by local assumptions, target status, or non-claim language. |
| Source-support status | all DPF chapters and P2 source register | Manual source-language audit plus RA query | Passed. No source is promoted from bibliography-spine support to ResearchAssistant-reviewed support. |
| Chapter transitions | P3 through P9 chapters | Manual dependency audit | Passed. Later chapters now have a visible block-level notation/status index and do not assume unproved exactness from earlier approximation layers. |

## Local checks run

- `git status --short --branch`.
- `rg` checks for object/assumption/status/diagnostic/boundary language across
  all DPF chapters.
- `rg` checks for claim-status terms:
  - `exact`;
  - `unbiased`;
  - `consistent`;
  - `approximate`;
  - `approximation`;
  - `closure`;
  - `relaxed`;
  - `surrogate`;
  - `engineering`;
  - `unsupported`;
  - `validated`;
  - `valid`;
  - `ResearchAssistant`;
  - `reviewed`.
- Direct label checks for the new P10 labels.
- Bibliography-key checks in `docs/references.bib`.
- ResearchAssistant query listed above.
- MathDevMCP:
  - `search_latex` for claim-status/target-status content;
  - `latex_label_lookup` attempt for `tab:bf-dpf-claim-status-index`, which
    failed inside the tool and was recorded as tool-limited.
- LaTeX build:
  - working directory: `docs`;
  - command: `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`;
  - result: passed after required reruns;
  - final PDF size: 218 pages.
- Post-build log scan:
  - no undefined citations;
  - no undefined references;
  - no rerun warnings;
  - no multiply defined label warnings found by targeted `rg`.

## Layout inspection

The build passes, but the DPF block remains table-heavy.  P10-local warnings in
`ch19_particle_filters.tex` include:

- underfull warnings in the DPF notation index around lines 98--118;
- underfull warnings in the DPF claim-status index around lines 157--182;
- small overfull paragraph warnings around lines 243--248 and 360;
- a small overfull paragraph around lines 540--548;
- underfull warnings in the baseline audit table around lines 581--608.

Existing P7--P9 longtable pressure remains visible, including overfull
alignment warnings in the P9 equation-to-test and diagnostic-contract tables.
These are not mathematical vetoes, but P12 should inspect the rendered PDF
pages rather than relying only on log status.

## Veto diagnostic review

- Notation conflicts remain unresolved: no.
- Chapter transitions smuggle in claims: no.
- Claim-status vocabulary is inconsistent: no.
- Literature synthesis contradicts itself: no.
- Local limitations are deferred to other chapters: no.
- Student-baseline or controlled-baseline files edited or staged: no.
- Source-support status conflicts with P2: no.
- Build or table-readability status omitted: no.

## Exit gate

Passed with layout caution.

The DPF block now has a single notation and claim-status reference point for
the derivation audit.

## Next recommended phase

Proceed to P11:

- `docs/plans/bayesfilter-dpf-monograph-reviewer-grade-phase-p11-derivation-audit-plan-2026-05-14.md`.
