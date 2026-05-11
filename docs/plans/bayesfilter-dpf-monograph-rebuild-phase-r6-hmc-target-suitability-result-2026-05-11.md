# Phase R6 result: DPF HMC-target correctness and structural-model suitability rewrite

## Date

2026-05-11

## Purpose

This note records the final reader-facing DPF rewrite phase of the current
monograph rebuild sequence.

## Target files

- `docs/main.tex`
- `docs/chapters/ch19e_dpf_hmc_target_suitability.tex`

## Plan

Create a dedicated DPF-specific HMC chapter slot inside the Part IV DPF block,
then write the final rung-by-rung DPF target-analysis chapter there so that the
generic BayesFilter HMC doctrine chapter remains separate.

## Execution

1. Added a new dedicated DPF HMC chapter slot to `docs/main.tex`:
   - `\input{chapters/ch19e_dpf_hmc_target_suitability}`
2. Created the new chapter file:
   - `docs/chapters/ch19e_dpf_hmc_target_suitability.tex`
3. Wrote the new chapter as a DPF-specific HMC-target and structural-model
   suitability chapter covering:
   - the DPF target contract for HMC;
   - rung-by-rung target analysis across EDH, PF-PF, soft resampling,
     OT/EOT, and learned OT;
   - nonlinear DSGE stress points;
   - MacroFinance stress points;
   - relation to surrogate-HMC / HNN acceleration ideas from `dsge_hmc`;
   - BayesFilter's recommendation for the first justified HMC-relevant DPF rung;
   - and a final chapter-boundary section stating what remains unclaimed.
4. Fixed the LaTeX syntax issue caused by TeX-breaking inline markup around the
   `dsge_hmc` project reference.

## Tests

- `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex` completed.
- A second rerun confirmed the monograph build was stable and up to date.
- `git diff --check` passed.
- The temporary LaTeX syntax blocker from the new chapter was resolved.

## Audit

### Primary criterion

Satisfied.

The final DPF-specific HMC chapter now has its own dedicated slot and makes the
value/gradient/target correspondence explicit rung by rung.

### Veto diagnostics

- **Differentiability-implies-validity veto**: cleared.  The chapter now states
  explicitly that smoothness alone is insufficient.
- **Rung-table vagueness veto**: cleared.  The chapter includes an explicit
  rung-by-rung target-status table.
- **Structural-model-informality veto**: cleared.  The DSGE and MacroFinance
  difficulties are treated as structural model stresses, not casual remarks.
- **Architecture-slot veto**: cleared by creating the dedicated `ch19e` chapter
  slot.

## Interpretation

This phase resolves the main architectural blocker that remained after the first
five rewrite phases.  The DPF-specific HMC chapter can now exist without being
forced into either:

- the learned-OT chapter, or
- the generic BayesFilter HMC doctrine chapter.

That separation is the correct mathematical outcome.  It preserves the logic of
the rebuilt block:

1. classical PF baseline;
2. particle-flow transport;
3. PF-PF correction;
4. differentiable resampling and OT;
5. learned/amortized OT;
6. DPF-specific HMC target analysis.

## Tidy-up

No phase-local compile blocker remains.

## Next phase justified?

Yes.

The current reader-facing DPF rewrite sequence is now complete enough for an
end-of-plan consolidation step: final summary, reset-memo closure, and the final
commit bundle for the accumulated uncommitted phases.
