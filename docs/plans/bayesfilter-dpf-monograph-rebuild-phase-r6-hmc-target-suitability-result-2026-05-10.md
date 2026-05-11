# Phase R6 result: HMC-target correctness and structural-model suitability rewrite

## Date

2026-05-10

## Purpose

This note records the sixth reader-facing rewrite phase of the DPF monograph
rebuild.

## Target file

- `docs/chapters/ch21_hmc_for_state_space.tex`

## Plan

Rewrite the HMC chapter so that the DPF-target question is integrated explicitly
into the reader-facing exposition: what scalar each rung evaluates, what it
 differentiates, whether the two correspond to the same target, and why this is
especially delicate for nonlinear DSGE and MacroFinance models.

## Execution

This phase was **not executed**.

## Why it was not executed

The current DPF block still uses `docs/chapters/ch19d_dpf_hmc_dsge_macrofinance_assessment.tex`
as the learned/amortized OT and implementation-mathematics chapter.  That means
the final reader-facing HMC-target and structural-model suitability chapter is
not yet assigned a clean dedicated file in the active chapter graph.

Rewriting `docs/chapters/ch21_hmc_for_state_space.tex` immediately would force a
premature merge between:

- the generic BayesFilter HMC doctrine that already lives in Chapter 21, and
- the DPF-specific rung-by-rung target analysis that should first be given its
own explicit chapter slot in the rebuilt DPF block.

That would risk collapsing two different expository jobs into one chapter:

1. generic HMC target-contract doctrine for BayesFilter as a whole;
2. DPF-specific target-status analysis for the rebuilt DPF ladder.

The planning program did not yet add a dedicated final DPF HMC chapter file or
update `docs/main.tex` to reserve such a slot.  Because of that, executing the
rewrite now would violate the monograph architecture discipline already
established in earlier phases.

## Tests

- Verified the current chapter-role situation in the active DPF block.
- Confirmed that `ch19d_dpf_hmc_dsge_macrofinance_assessment.tex` is now being
  used in substance for learned/amortized OT and implementation mathematics.
- Confirmed that `ch21_hmc_for_state_space.tex` still carries broader BayesFilter
  HMC doctrine rather than a DPF-specific final assessment role.

## Audit

### Primary criterion

Not satisfied, because the chapter slot needed for the final DPF-specific HMC
assessment is not yet cleanly available in the active chapter architecture.

### Veto diagnostics

Triggered.

- The DPF-specific HMC-target rewrite would currently blur generic BayesFilter
  HMC doctrine with the final DPF rung-by-rung assessment.
- The active chapter graph does not yet reserve a dedicated reader-facing slot
  for the final DPF HMC-suitability chapter.

## Interpretation

This is a real architectural blocker, not a cosmetic one.  The earlier phases
succeeded because each rewrite phase had a clear mathematical role and a clear
file target.  That condition now fails for the final HMC-target chapter.

The right next step is not to force the rewrite into the wrong file.  The right
next step is to decide how the final DPF HMC-suitability chapter should appear
in the chapter graph.  The cleanest options are:

1. create a new dedicated DPF HMC chapter file inside the Part IV DPF block,
   leaving `ch21_hmc_for_state_space.tex` as the generic HMC doctrine chapter;
2. or deliberately refactor the chapter graph so the DPF-specific HMC chapter
   lives elsewhere with explicit user approval.

## Next phase justified?

No, not yet.

The final DPF HMC-suitability rewrite is not justified until the chapter-slot
question is resolved.  Continuing without resolving that would violate the same
architectural discipline that the rebuild program was created to enforce.
