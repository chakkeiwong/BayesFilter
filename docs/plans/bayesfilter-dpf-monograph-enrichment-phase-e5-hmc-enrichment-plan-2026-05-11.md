# Phase E5 plan: DPF-specific HMC enrichment

## Date

2026-05-11

## Purpose

Deepen the DPF-specific HMC chapter so that it becomes more than a strong
architectural endpoint: it should be a detailed target-interpretation and model-
suitability chapter that can guide future experiments and implementation design.

## Target chapter

- `docs/chapters/ch19e_dpf_hmc_target_suitability.tex`

## Main questions

1. For each DPF rung, what target is HMC actually sampling?
2. How does that target differ from the exact classical particle-filter target?
3. Why do nonlinear DSGE and MacroFinance models magnify target drift?
4. How does this compare to dsge_hmc HNN / surrogate-HMC acceleration ideas?

## Required enrichment topics

- deeper rung-by-rung target analysis;
- stronger exact/unbiased/relaxed/surrogate distinctions;
- deeper discussion of value/gradient consistency;
- fuller nonlinear DSGE structural argument;
- fuller MacroFinance latent-factor argument;
- more explicit comparison between target-construction methods and
  geometry-acceleration methods.

## Required source lanes

- the rebuilt DPF chapters;
- BayesFilter generic HMC doctrine;
- dsge_hmc HNN / surrogate-HMC material;
- relevant approximate-target and pseudo-marginal references.

## Required outputs

1. strengthened rung-by-rung target table;
2. stronger DSGE and MacroFinance stress analysis;
3. explicit comparison note against surrogate-HMC / HNN acceleration;
4. clearer recommendation for the first justified HMC-relevant DPF rung.
5. explicit structured comparison to the `dsge_hmc` surrogate-HMC / HNN line at
   the level of what is being approximated or changed.

## Required tables

### Table E5-1: enriched rung-by-rung HMC target map

| Rung | Value object | Gradient object | Same target? | Approximation layers | HMC interpretation |
| --- | --- | --- | --- | --- | --- |

### Table E5-2: model-class sensitivity map

| Model class | Why target drift matters | Why geometry matters | DPF implication | Surrogate-HMC implication |
| --- | --- | --- | --- | --- |

### Table E5-3: DPF versus surrogate-HMC comparison

| Method family | What is approximated? | What target changes? | What geometry changes? | What remains exact? |
| --- | --- | --- | --- | --- |

### Table E5-4: rung promotion evidence ladder

| Rung | Current HMC interpretation | What would have to be shown next to promote it? |
| --- | --- | --- |

## Primary criterion

The chapter must become detailed enough that a future experimenter can use it to
classify DPF-HMC experiments by target type rather than only by algorithm name.

## Veto diagnostics

Do not proceed if:
- the chapter still treats differentiability as nearly equivalent to HMC
  validity;
- DSGE and MacroFinance are still discussed only at a slogan level;
- the comparison to surrogate-HMC / HNN remains only superficial.

## Exit gate

Proceed to E6 only once the HMC chapter is rich enough that implementation or
experimental decisions can be grounded directly in its target analysis.
