# Phase M5 result: HMC target correctness and DSGE/MacroFinance suitability plan

## Date

2026-05-09

## Purpose

This note records the chapter-plan result for the HMC-target and
DSGE/MacroFinance suitability portion of the rebuilt DPF monograph.

## Core conclusion

The HMC question should be written as a dedicated chapter or chapter-equivalent
unit that answers, rung by rung, what target would be sampled if HMC were run on
a given DPF construction.  This chapter must be mathematically sharper than the
existing broad DPF assessment draft, because the central issue is not whether a
method is differentiable, but whether the value and gradient correspond to a
well-defined target appropriate for structural models.

## Recommended chapter structure

### Section 1. Framing the HMC question

- restate BayesFilter's HMC contract in DPF-specific language;
- define the difference between:
  - exact target,
  - unbiased-estimator target used within a corrected MCMC scheme,
  - approximate or relaxed target,
  - learned surrogate target.

### Section 2. Rung-by-rung target analysis

For each rung in the agreed DPF ladder, specify:

- what scalar is evaluated;
- what quantity is differentiated;
- whether those two correspond to the same target;
- where any bias or approximation enters;
- what kind of HMC interpretation, if any, is justified.

### Section 3. Structural-model difficulties for nonlinear DSGE

Analyze:

- structural state partitions and deterministic-completion implications;
- nonlinear observation maps and local linearization risk;
- invalid or barely valid parameter regions;
- whether flow- or resampling-based approximations are likely to change the
  effective target in economically important ways.

### Section 4. Structural-model difficulties for MacroFinance

Analyze:

- latent-factor dimension and degeneracy concerns;
- missing-data and mixed-frequency structure;
- sharp curvature in volatility and measurement-noise directions;
- compiled-path and reproducibility requirements for long HMC runs.

### Section 5. Practical interpretation for BayesFilter

Conclude with a mathematically grounded recommendation for which rung is the
first justified HMC-relevant development target and which rungs remain strictly
research or surrogate territory.

## Table 1: rung-by-rung HMC assessment

| Rung | What quantity is evaluated? | What quantity is differentiated? | Same target? | Bias source | Viable HMC interpretation |
| --- | --- | --- | --- | --- | --- |
| EDH | flow-based approximate likelihood object | derivative of same approximate flow object if implemented directly | yes locally, but approximate object | Gaussian closure / flow discretization | value-side benchmark or approximate-target only |
| EDH/PF with weights | proposal-corrected particle likelihood estimator | derivative of corrected estimator if differentiation is well defined | in principle yes, subject to implementation | Monte Carlo variability, flow discretization, Jacobian numerical error | first serious HMC-related candidate, but still variance-limited |
| soft/differentiable resampling | relaxed differentiable surrogate | gradient of same surrogate | yes as surrogate, no as original target | resampling relaxation | surrogate-target HMC only unless explicitly accepted as approximation |
| OT/EOT resampling | relaxed transport-based surrogate / approximate target | gradient through OT/Sinkhorn path | yes as relaxed target, not exact resampling target | entropy regularization and finite solver accuracy | approximate-target HMC candidate |
| learned/amortized OT | learned approximation to OT-based target | gradient of learned surrogate | yes as learned surrogate, not exact OT nor exact resampling target | learning approximation plus OT relaxation | surrogate-target HMC only, with extra caution |

## Table 2: structural-model stress points

| Model class | Main mathematical difficulty | Filtering implication | HMC implication |
| --- | --- | --- | --- |
| nonlinear DSGE | structural partitions, determinacy/solution validity, nonlinear observation maps | approximation may alter economically meaningful state law | target and gradient can drift in sensitive regions |
| MacroFinance latent-factor models | high dimension, degeneracy, mixed frequency, sharp curvature | particle count and transport approximation matter materially | long-run chain stability and compiled reproducibility are critical |
| generic small nonlinear SSM | benchmark scale, reference filters available | useful for staged testing | useful only as controlled evidence, not final structural proof |

## Main interpretation

The key mathematical lesson is that HMC suitability does not attach to a method
family name.  It attaches to a particular value-gradient construction.  Two
algorithms both called "differentiable particle filter" may sit in different
places on the target-validity spectrum if one uses proposal correction and the
other uses a relaxed resampling surrogate.

This is especially important for BayesFilter because the intended model classes
are not generic low-dimensional toy systems.  Nonlinear DSGE and MacroFinance
models amplify target-definition mistakes: structural-state semantics, local
Jacobians, near-boundary parameter regions, and long-chain compiled execution
requirements make a small surrogate drift potentially important.

## Audit

The HMC question is now isolated cleanly enough that the final monograph can
avoid one of the biggest failure modes of the earlier DPF draft: speaking as if
smooth gradients and HMC-readiness were almost the same question.

No blocker prevents moving to the drafting-execution phase.  The main burden for
later drafting is to keep every HMC statement tethered to the rung-by-rung table
rather than broad prose.

## Next phase justified?

Yes.

Phase M6 is justified because the chapter architecture and mathematical burdens
are now explicit enough to govern the actual reader-facing rewrite and equation
audit execution.
