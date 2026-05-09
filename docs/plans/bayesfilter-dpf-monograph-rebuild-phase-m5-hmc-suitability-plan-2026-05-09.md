# Phase M5 plan: HMC target correctness and DSGE/MacroFinance suitability

## Date

2026-05-09

## Purpose

Plan the chapters or sections that assess whether the DPF path is suitable for
HMC in nonlinear DSGE and MacroFinance models.

## Main question

Given the literature and the mathematical construction of the DPF ladder, what
can be said rigorously about the target that HMC would sample at each rung, and
what does that imply for nonlinear structural models?

## Scope

This phase covers:

- exact-target versus surrogate-target interpretations;
- pseudo-marginal or corrected-target reasoning where applicable;
- value/gradient consistency requirements;
- structural state-space complications for nonlinear DSGE models;
- latent-factor and mixed-frequency complications for MacroFinance models;
- implications of compiled differentiable implementations for HMC.

## Required output

Produce a chapter/section plan that answers the HMC question rung by rung.

## Required structure

### Table 1: rung-by-rung HMC assessment

| Rung | What quantity is evaluated? | What quantity is differentiated? | Same target? | Bias source | Viable HMC interpretation |
| --- | --- | --- | --- | --- | --- |

### Table 2: structural-model stress points

| Model class | Main mathematical difficulty | Filtering implication | HMC implication |
| --- | --- | --- | --- |

## Mandatory topics

- why differentiability alone is insufficient for HMC correctness;
- when proposal-correction logic helps and when it does not;
- how structural state partitions affect the DPF path in DSGE models;
- how latent-factor dimension, missing-data structure, and stiffness affect the
  MacroFinance case;
- why learned resamplers or surrogates require stronger caution.

## Exit gate

Proceed only when the HMC-assessment chapter plan can state clearly what the
monograph will and will not claim for each rung of the DPF ladder.
