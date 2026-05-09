# Phase M2 result: DPF monograph mathematical architecture and chapter-map design

## Date

2026-05-09

## Purpose

This note designs the target mathematical chapter architecture for the DPF
monograph rebuild based on the Phase M1 literature map.

## Core architectural conclusion

The literature now clearly supports a larger DPF block than the rejected
three-chapter draft.  A mathematically serious treatment should separate at
least six reader-facing tasks:

1. particle-filter foundations and likelihood estimators;
2. particle-flow mathematics;
3. PF-PF and importance-corrected flow proposals;
4. differentiable resampling and OT relaxations;
5. learned / amortized OT and implementation-facing mathematical constraints;
6. HMC target interpretation and DSGE/MacroFinance suitability.

This can be organized as five or six chapters depending on how tightly PF-PF is
folded into the flow chapter.  The current recommendation is **six chapters**,
because PF-PF and differentiable-resampling relaxations answer different
mathematical questions and should not be compressed together.

## Proposed chapter list

| Proposed chapter | Purpose | Main sources | Main equations / objects |
| --- | --- | --- | --- |
| DPF-1 Particle-filter foundations | establish filtering recursion, empirical measures, importance weighting, bootstrap PF likelihood estimator, degeneracy | classical PF / SMC sources, CIP ch26 | empirical filtering measure, likelihood estimator, ESS |
| DPF-2 Particle-flow foundations | derive homotopy, continuity equation, EDH and LEDH constructions, stiffness discussion | CIP ch26/ch27, Daum--Huang lineage, Li/Coates lineage | homotopy density, flow PDE/ODE, EDH / LEDH coefficients |
| DPF-3 PF-PF and proposal correction | treat flow as proposal, derive Jacobian/log-det and corrected importance weights, compare EDH/PF vs LEDH/PF | CIP ch27, PF-PF sources, student coverage | change-of-variables weights, Jacobian/log-det identities |
| DPF-4 Differentiable resampling and OT | develop soft resampling, OT/EOT resampling, Sinkhorn formulation, bias-versus-differentiability analysis | CIP ch26/ch32, Corenflos line, Zhu/Murphy/Jonschkowski line | soft-resampling formulas, OT primal/dual/barycentric forms, Sinkhorn updates |
| DPF-5 Learned transport / implementation mathematics | treat learned/amortized OT as approximation layer, state implementation constraints from mathematics, not governance | advanced repo architecture note plus supporting literature | learned approximation to transport map, approximation residual interpretation |
| DPF-6 HMC target correctness and structural-model suitability | analyze rung-by-rung HMC interpretation and suitability for nonlinear DSGE and MacroFinance | CIP HMC chapters, PMCMC literature, BayesFilter HMC doctrine | exact/unbiased/approximate/surrogate target map, structural-model stress points |

## Dependency graph

| Chapter | Depends on | Enables |
| --- | --- | --- |
| DPF-1 | nonlinear SSM / filtering preliminaries | DPF-2, DPF-3, DPF-4, DPF-6 |
| DPF-2 | DPF-1 | DPF-3, DPF-6 |
| DPF-3 | DPF-1, DPF-2 | DPF-6 |
| DPF-4 | DPF-1 | DPF-5, DPF-6 |
| DPF-5 | DPF-4 | DPF-6 |
| DPF-6 | DPF-1 through DPF-5, HMC doctrine chapters | final BayesFilter DPF interpretation |

## Proposed location in the monograph

Insert the rebuilt DPF block after the current particle-filter gateway chapter
and before filter-choice / HMC chapters.  The most natural structure is:

- keep `ch19_particle_filters.tex` as a short gateway or rewrite it into the
  opening of DPF-1 if that proves cleaner;
- then add the larger DPF block;
- then continue with `ch20_filter_choice.tex` and the HMC chapters.

This preserves the logic that DPFs are part of the nonlinear filtering story,
while allowing later HMC chapters to refer back to a detailed DPF-target
analysis.

## Replacement map

| Existing draft artifact | Keep / split / rewrite / retire | Reason |
| --- | --- | --- |
| `docs/chapters/ch19b_dpf_literature_survey.tex` | retire as final chapter; reuse only for provenance | too commentary-heavy and too broad for final exposition |
| `docs/chapters/ch19c_dpf_implementation_literature.tex` | retire as final chapter; mine for implementation topics | too implementation-memo-like and insufficiently mathematical |
| `docs/chapters/ch19d_dpf_hmc_dsge_macrofinance_assessment.tex` | retire as final chapter; reuse structure ideas | useful question framing, but still too prose-driven |
| `docs/chapters/ch19_particle_filters.tex` | keep as gateway or merge into DPF-1 | already doctrinally useful, but may need expansion |
| CIP DPF chapters | rewrite heavily into BayesFilter notation | theory-rich but not BayesFilter-specific and not always at the right level of distinction |
| student report / advanced repo | do not treat as monograph voice; use for gap detection and comparison | useful critique/coverage sources, not exposition authorities |

## Architectural interpretation

The earlier three-chapter structure failed because it asked one chapter set to do
three different jobs at once:

- survey the literature,
- explain implementation implications,
- and assess HMC suitability.

The literature map shows that these jobs split naturally into a larger structure.
The rebuilt architecture should therefore be accepted as intentionally longer.
That is a feature, not a failure, for a mathematically difficult topic.

## Audit

The six-chapter recommendation is justified by the Phase M1 evidence.
Compressing the treatment into fewer chapters would likely reintroduce one of
the same failures as before: either mathematical derivations get shortened too
aggressively, or implementation/HMC caveats are mixed into survey text in a way
that weakens rigor.

## Next phase justified?

Yes.

Phase M3 is justified because the architecture is now explicit enough to design
the particle-filter / particle-flow theory chapters in detail.
