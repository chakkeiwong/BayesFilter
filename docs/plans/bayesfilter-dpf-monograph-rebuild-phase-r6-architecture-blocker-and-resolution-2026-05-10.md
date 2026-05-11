# Phase R6 architecture blocker note: dedicated DPF HMC chapter insertion

## Date

2026-05-10

## Purpose

Record the architectural decision that a dedicated DPF-specific HMC chapter is
needed, rather than forcing the final rung-by-rung DPF target analysis into the
generic BayesFilter HMC doctrine chapter.

## Blocker summary

The final DPF HMC-suitability rewrite could not proceed cleanly because the file
roles had shifted during the earlier rewrite phases:

- `ch19d_dpf_hmc_dsge_macrofinance_assessment.tex` was repurposed in substance
  for learned/amortized OT and implementation mathematics;
- `ch21_hmc_for_state_space.tex` still carries generic BayesFilter HMC doctrine.

Thus the final DPF-specific HMC target chapter lacked a clean dedicated slot in
the active chapter graph.

## Resolution

The correct architectural resolution is to introduce a new dedicated reader-
facing DPF HMC chapter inside the DPF block, leaving
`ch21_hmc_for_state_space.tex` as the broader BayesFilter HMC doctrine chapter.

## Recommended new chapter slot

Insert a new chapter immediately after the learned/amortized OT chapter and
before `ch20_filter_choice.tex`.

Recommended file name:
- `docs/chapters/ch19e_dpf_hmc_target_suitability.tex`

Recommended role:
- rung-by-rung DPF target analysis;
- exact/unbiased/relaxed/surrogate distinction;
- nonlinear DSGE structural stress points;
- MacroFinance latent-factor stress points;
- BayesFilter recommendation for the first justified HMC-relevant DPF rung.

## Consequence

With this new slot, the chapter graph becomes cleaner:

- `ch19_particle_filters.tex` — particle-filter foundations
- `ch19b_dpf_literature_survey.tex` — particle-flow foundations
- `ch19c_dpf_implementation_literature.tex` — PF-PF / proposal correction
- `ch32_diff_resampling_neural_ot.tex` — differentiable resampling and OT
- `ch19d_dpf_hmc_dsge_macrofinance_assessment.tex` — learned/amortized OT and implementation mathematics
- `ch19e_dpf_hmc_target_suitability.tex` — final DPF-specific HMC target and structural-model suitability chapter

This preserves a clean separation between:
- generic HMC doctrine in Chapter 21,
- and DPF-specific HMC target interpretation inside the nonlinear filtering block.

## Next phase justified?

Yes.

Once the new chapter slot is created, the final DPF HMC-target rewrite becomes
justified again.
