# Phase M3 result: particle-filter and particle-flow theory chapter plan

## Date

2026-05-09

## Purpose

This note defines the mathematical section map for the particle-filter and
particle-flow theory portion of the rebuilt DPF monograph.

## Core conclusion

The particle-filter / particle-flow block should not be a single chapter.  It
should be split at least into:

1. a particle-filter foundations chapter; and
2. a particle-flow and PF-PF chapter,

with PF-PF either as a major second half of the particle-flow chapter or as a
standalone follow-on chapter if the derivations become too long.

The recommended default is to let PF-PF stand as a distinct chapter-equivalent
unit, because proposal correction is the mathematical bridge between flow-based
approximation and a clearer likelihood interpretation.

## Proposed chapter A: particle-filter foundations

### Mathematical role

Provide the measure-theoretic and likelihood-estimation baseline from which all
DPF variants depart.

### Required sections

1. **Nonlinear state-space model and filtering recursion**
   - define the latent state, observation process, predictive law, filtering law,
     and marginal-likelihood factorization;
   - keep notation compatible with the existing BayesFilter monograph.

2. **Empirical filtering measures**
   - define the atomic filtering approximation
     `\hat\pi_t^N = \sum_i w_t^{(i)} \delta_{x_t^{(i)}}`;
   - explain approximation of posterior expectations by weighted sums.

3. **Sequential importance sampling and resampling**
   - derive the SIS and SIR recursions carefully;
   - distinguish proposal density, target density, and incremental weights.

4. **Bootstrap PF likelihood estimator**
   - present the standard marginal-likelihood estimator;
   - state precisely the unbiasedness claim and its scope.

5. **Degeneracy and curse-of-dimensionality discussion**
   - give the mathematical reason weight collapse matters;
   - keep this connected to the later motivation for particle flow.

### Main equations / objects

- filtering recursion;
- empirical filtering measure;
- SIS weight recursion;
- bootstrap PF likelihood estimator;
- ESS and degeneracy indicators.

### Exact / approximate distinction

- empirical approximation to filtering law;
- unbiased likelihood-estimator claim for the bootstrap PF under standard
  assumptions;
- no pathwise differentiability claim.

### Implementation implications to record

- what must be sampled;
- what must be weighted;
- what diagnostics are needed (ESS, variance, resampling counts);
- why this gives a value-side baseline but not yet a differentiable HMC path.

## Proposed chapter B: particle-flow foundations

### Mathematical role

Develop the homotopy and continuity-equation framework underlying EDH, LEDH,
and related flow-based particle transports.

### Required sections

1. **Motivation: transport versus resampling**
   - why discrete reweighting/resampling motivates flow formulations.

2. **Homotopy density and pseudo-time path**
   - define the path from predictive law to filtering law.

3. **Continuity equation / flow PDE**
   - derive the basic conservation equation;
   - state exactly what assumptions are required.

4. **EDH under Gaussian closure**
   - derive or carefully reconstruct the affine flow under Gaussian prior and
     Gaussian likelihood assumptions;
   - state the exact special-case linear-Gaussian recovery result.

5. **LEDH and local linearization**
   - define the local Jacobian / local precision construction;
   - state where the approximation enters.

6. **Stiffness and discretization**
   - explain why low-noise or sharp-likelihood regimes create numerical
     stiffness;
   - relate this to implementation and later stress testing.

### Main equations / objects

- homotopy density;
- continuity equation / PDE form;
- affine EDH ODE coefficients;
- local LEDH coefficients;
- linear-Gaussian recovery statement.

### Exact / approximate distinction

- exact special case: linear-Gaussian recovery;
- approximate flow under Gaussian closure or local linearization;
- no generic corrected-target claim yet.

### Implementation implications to record

- need for Jacobians / local Jacobians;
- ODE integration choices;
- stiffness diagnostics;
- why a smooth flow is still not enough to define the final HMC target.

## Proposed chapter C: PF-PF and proposal correction

### Mathematical role

Explain how a particle flow becomes part of an importance-corrected particle
method rather than only an uncontrolled transport approximation.

### Required sections

1. **Flow as proposal rather than final target**
   - formal change-of-variables interpretation.

2. **Proposal density under the flow map**
   - derive the transformed density using the Jacobian determinant.

3. **Importance weights after flow**
   - derive corrected incremental weights for PF-PF;
   - distinguish EDH/PF and LEDH/PF where needed.

4. **Log-determinant or Jacobian evolution**
   - derive the auxiliary ODE or equivalent mechanism for practical evaluation.

5. **Comparison to uncorrected flow filtering**
   - explain mathematically what the correction restores and what it does not.

### Main equations / objects

- proposal density under flow;
- corrected importance-weight formula;
- Jacobian determinant / log-det identity.

### Exact / approximate distinction

- clearer target relationship than raw flow;
- still subject to discretization and finite-particle approximation;
- not the same as exact analytic likelihood in general.

### Implementation implications to record

- need to track Jacobian/log-det quantities;
- variance and numerical burden;
- why this rung is the first serious HMC-related value-side candidate.

## Chapter-level comparison requirements

The final exposition must compare mathematically:

- bootstrap PF versus PF-PF;
- EDH versus LEDH;
- flow-only transport versus proposal-corrected flow;
- global Gaussian closure versus local linearization;
- proposal correctness versus numerical tractability.

## Load-bearing derivation obligations from this phase

The following must be carried forward as explicit derivation tasks:

1. bootstrap PF likelihood estimator and exact scope of unbiasedness;
2. homotopy density and continuity-equation derivation;
3. EDH affine flow derivation under Gaussian closure;
4. linear-Gaussian recovery proposition;
5. PF-PF change-of-variables weight formula;
6. Jacobian/log-determinant evolution identity.

## Audit

This phase gives enough structure for the particle-filter and particle-flow
portion of the monograph to be drafted later without collapsing distinct
mathematical questions together.

The key improvement is the explicit separation between:
- particle-filter likelihood estimation,
- flow approximation mathematics,
- and proposal-corrected flow filtering.

That separation was missing from the earlier DPF draft at the needed level of
rigor.

## Next phase justified?

Yes.

Phase M4 is justified because the resampling and OT material is now clearly
separable from the particle-flow and PF-PF theory block.
