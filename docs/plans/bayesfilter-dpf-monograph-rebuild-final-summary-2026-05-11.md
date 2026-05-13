# Summary: DPF monograph rebuild execution results and next hypotheses

## Date

2026-05-11

## Scope

This summary covers the completed DPF monograph rebuild program through the
reader-facing rewrite sequence now present on branch `dpf-monograph-rebuild`.
It records the planning results, the chapter rewrites, the architectural
corrections, and the main remaining work implied by the rebuild.

## 1. High-level result

The work succeeded in replacing the earlier weak DPF chapter pass with a much
stronger mathematical architecture and a substantially improved reader-facing
chapter sequence.

The completed sequence now has a coherent internal ladder:

1. `ch19_particle_filters.tex` — particle-filter foundations;
2. `ch19b_dpf_literature_survey.tex` — particle-flow foundations;
3. `ch19c_dpf_implementation_literature.tex` — PF-PF and proposal correction;
4. `ch32_diff_resampling_neural_ot.tex` — differentiable resampling and OT;
5. `ch19d_dpf_hmc_dsge_macrofinance_assessment.tex` — learned/amortized OT and
   implementation mathematics;
6. `ch19e_dpf_hmc_target_suitability.tex` — DPF-specific HMC target correctness
   and structural-model suitability.

The main improvement is not cosmetic.  The DPF material is now separated by its
mathematical questions rather than by informal topic buckets.

## 2. What was completed in the planning program

The rebuild began by creating a dedicated planning program and then executing it
phase by phase.

Completed planning phases:
- M0 preflight and supersession audit;
- M1 literature survey and source-grounding audit;
- M2 mathematical architecture and chapter-map design;
- M3 particle-filter and particle-flow theory planning;
- M4 differentiable-resampling and OT planning;
- M5 HMC-target and structural-model suitability planning;
- M6 drafting and equation-audit protocol;
- M7 integration and final audit protocol.

The key planning result was that the topic required more than the earlier
three-chapter treatment.  The chapter count expanded because the literature and
the mathematical distinctions demanded it, not because of stylistic preference.

## 3. What was completed in the reader-facing rewrite sequence

### R1: particle-filter foundations

Completed rewrite of `docs/chapters/ch19_particle_filters.tex`.

Main gains:
- exact nonlinear filtering recursion stated clearly;
- empirical filtering measure defined explicitly;
- SIS and SIR logic made explicit;
- bootstrap PF likelihood estimator stated at proposition level;
- unbiasedness status stated under standard assumptions;
- ESS and degeneracy placed at the mathematical foundation of the later DPF
  discussion.

Interpretation:
- this chapter now provides the reference object against which every later DPF
  construction can be interpreted.

### R2: particle-flow foundations

Completed rewrite of `docs/chapters/ch19b_dpf_literature_survey.tex`.

Main gains:
- transport motivation separated from the generic survey voice;
- homotopy density stated explicitly with a normalizing constant;
- continuity equation and flow PDE written explicitly;
- EDH under Gaussian closure derived at the right level;
- linear-Gaussian recovery isolated as the exact special case;
- LEDH and local linearization separated cleanly;
- stiffness recognized as a mathematical and numerical concern.

Interpretation:
- the flow chapter now gives a clean transport foundation without prematurely
  claiming corrected-target status.

### R3: PF-PF and proposal correction

Completed rewrite of `docs/chapters/ch19c_dpf_implementation_literature.tex`.

Main gains:
- raw flow versus flow-as-proposal distinguished clearly;
- proposal density under the flow map derived by change of variables;
- corrected PF-PF importance weights derived explicitly;
- Jacobian/log-determinant evolution made explicit;
- EDH/PF versus LEDH/PF separated as proposal constructions;
- the chapter now states exactly what proposal correction restores and what
  remains approximate.

Interpretation:
- this is the first rung where a strong particle-based HMC-relevant target story
  emerges.

### R4: differentiable resampling and OT

Completed rewrite of `docs/chapters/ch32_diff_resampling_neural_ot.tex`.

Main gains:
- the resampling bottleneck is now formalized at the level of weighted and
  equal-weight empirical measures;
- standard resampling is described as a discontinuous map;
- soft resampling is described as a smooth but biased relaxation;
- equal-weight resampling is cast as a transport problem;
- entropic OT, Sinkhorn form, and barycentric projection are all stated at the
  correct mathematical level;
- the bias-versus-differentiability trade-off is explicit.

Interpretation:
- this chapter now explains exactly why differentiable resampling changes the
  mathematical object being differentiated.

### R5: learned/amortized OT and implementation mathematics

Completed rewrite in substance of
`docs/chapters/ch19d_dpf_hmc_dsge_macrofinance_assessment.tex`.

Main gains:
- teacher OT map versus learned OT map made explicit;
- learned OT treated as a second approximation layer on top of OT;
- map-level residuals and target shift interpreted mathematically;
- training-distribution dependence made explicit;
- implementation-facing mathematics tied to approximation status rather than to
  casual speed claims.

Interpretation:
- learned OT is now clearly represented as a surrogate layer, not as a magical
  exact acceleration.

### R6: DPF-specific HMC target correctness and structural-model suitability

Completed new dedicated chapter:
- `docs/chapters/ch19e_dpf_hmc_target_suitability.tex`

Also added a new slot in `docs/main.tex` for this chapter.

Main gains:
- generic BayesFilter HMC doctrine remains in `ch21_hmc_for_state_space.tex`;
- DPF-specific rung-by-rung target analysis now has its own chapter;
- exact / unbiased / relaxed / learned-surrogate interpretations are separated;
- nonlinear DSGE and MacroFinance difficulties are analyzed as structural-model
  stresses rather than as casual caveats;
- the relation to the dsge_hmc surrogate-HMC / HNN line is made explicit.

Interpretation:
- the architectural blocker was resolved correctly by creating a dedicated DPF
  HMC chapter rather than overloading the generic HMC chapter.

## 4. Architectural corrections that mattered

The most important architectural corrections were:

### A. separating raw flow from corrected PF-PF

Without this split, the text would continue to blur a geometrically attractive
transport approximation with a mathematically corrected proposal mechanism.

### B. separating differentiable resampling from learned OT

Without this split, the text would treat OT relaxation and learned OT
approximation as if they were a single layer.  That would erase one full level
of target drift.

### C. introducing a dedicated DPF HMC chapter

Without the new `ch19e` chapter, the final HMC-target analysis would have been
forced into the generic BayesFilter HMC chapter or into the learned-OT chapter,
which would again have mixed conceptually distinct jobs.

## 5. What remains incomplete or still weak

The rebuild is much better, but it is not the end of the story.

### A. Some chapter filenames no longer match their mathematical roles

For example:
- `ch19b_dpf_literature_survey.tex` now functions as the particle-flow
  foundations chapter;
- `ch19c_dpf_implementation_literature.tex` now functions as the PF-PF /
  proposal-correction chapter;
- `ch19d_dpf_hmc_dsge_macrofinance_assessment.tex` now functions as the
  learned/amortized OT chapter.

This is acceptable for the current pass, but a later cleanup should probably
rename or renumber files so the source tree matches the mathematical content.

### B. The surrounding shared monograph still has broader citation/reference debt

The DPF local rewrites now compile, but the broader monograph still carries
pre-existing unresolved or unstable reference/citation issues outside the DPF
core.  Those were not all introduced by this pass and should be handled in a
separate consolidation audit.

### C. The current DPF HMC chapter is doctrinally strong, but not yet backed by
fresh BayesFilter-specific experimental evidence

That is acceptable at the monograph level so long as the claims remain properly
qualified, but the next engineering phase should test the recommended first DPF
HMC rung explicitly.

## 6. Concrete next phases recommended

### Phase N1: filename / chapter-graph cleanup

Goal:
- rename or rationalize the DPF chapter files so the filenames match the new
  mathematical roles.

Why:
- current reader-facing structure is correct, but source filenames are legacy
  and mildly misleading.

### Phase N2: DPF bibliography and cross-reference consolidation audit

Goal:
- do a bounded cross-reference and bibliography cleanup pass focused on the DPF
  block and its adjacent chapters.

Why:
- the DPF chapters are now structurally much stronger, and they deserve a clean,
  stable bibliography surface.

### Phase N3: experimental validation of the first HMC-relevant DPF rung

Goal:
- validate proposal-corrected PF-PF as the first justified DPF HMC candidate in
  controlled settings.

Why:
- the monograph now recommends PF-PF as the first justified rung; that should be
  backed by experiments aligned with the same target interpretation.

## 7. Explicit hypotheses to test next

The next work should be hypothesis-driven.  The clearest hypotheses now are:

### H1. PF-PF is the first justified HMC-relevant DPF rung

Claim:
- among the current ladder, PF-PF is the earliest rung with a sufficiently clear
  target interpretation to justify serious HMC development.

Test:
- compare EDH-only, PF-PF, soft-resampling, OT, and learned-OT value/gradient
  behavior on controlled models where references are available.

### H2. Relaxed resampling methods should be interpreted as target-defining
surrogates, not hidden exact replacements

Claim:
- soft resampling and entropic OT should be treated as explicit relaxed targets
  in HMC analysis.

Test:
- compare posterior shifts under categorical, OT-relaxed, and learned-OT paths
  in settings with a trusted baseline.

### H3. Learned OT introduces a second approximation layer that is not negligible
by default

Claim:
- a learned OT operator should be expected to contribute its own approximation
  residual on top of the OT relaxation.

Test:
- quantify teacher-versus-learned residuals and posterior sensitivity across
  state dimension, particle count, and regularization range.

### H4. Nonlinear DSGE models amplify target-drift risks more than toy systems

Claim:
- in nonlinear DSGE settings, stacked approximations in flow, resampling, and
  learned transport are more likely to produce meaningful posterior drift than
  in small toy SSMs.

Test:
- compare controlled toy-model target drift to DSGE-like structural cases using
  the same DPF rung.

### H5. MacroFinance models amplify long-run compiled-path and reproducibility
requirements

Claim:
- in latent-factor MacroFinance settings, even small target inconsistencies or
  unstable compiled-path behavior become materially important under long HMC
  runs.

Test:
- run repeatability and compiled/eager parity experiments on a representative
  MacroFinance-style controlled benchmark.

### H6. Surrogate-HMC acceleration and DPF target construction are complementary,
not interchangeable

Claim:
- the dsge_hmc HNN/surrogate-HMC line addresses how to sample a target more
  efficiently, whereas DPF concerns how the likelihood target itself is defined.

Test:
- compare a fixed target under surrogate-HMC acceleration to target-changing DPF
  ladders, rather than comparing them as if they solved the same problem.

## 8. Final assessment of this pass

This pass did not finish the entire BayesFilter monograph, but it did complete a
coherent and valuable unit of work: the DPF section is now built around the
right mathematical distinctions.

The single biggest improvement is that the monograph now has an explicit ladder
from exact baseline to approximate transport, corrected proposal, relaxed
resampling, learned transport, and finally HMC target interpretation.  That was
the structural gap in the earlier draft, and it is now closed.
