# Phase M3 plan: particle-filter and particle-flow theory chapters

## Date

2026-05-09

## Purpose

Plan the chapters that treat the mathematical foundations of particle filters,
particle-flow filters, and proposal-correction logic in a rigorous,
self-contained way.

## Scope

This phase covers the mathematics of:

- classical particle-filter likelihood estimators;
- degeneracy and the role of resampling;
- EDH and related homotopy formulations;
- LEDH and local-linearization variants;
- PF-PF importance correction;
- Jacobian and log-determinant identities;
- stochastic and kernel/related particle-flow variants as needed for
  comparison.

## Required output

Produce a detailed writing plan for the theory chapters in this bucket,
including section-by-section mathematical goals.

## Required section-level plan items

For each planned chapter/section, specify:

- the exact mathematical object being defined or derived;
- which claims are exact and which are approximate;
- which equations must be derived in BayesFilter notation;
- which equations may be cited from source literature with adaptation;
- which comparisons to other methods are required;
- which implementation implications must be recorded.

## Mandatory mathematical topics

- empirical filtering measures;
- marginal-likelihood estimators;
- unbiasedness versus proposal-corrected approximations;
- homotopy and continuity-equation framing;
- global versus local flow assumptions;
- change-of-variables and proposal-density correction;
- stiffness and discretization implications.

## Exit gate

Proceed only when the particle-filter / particle-flow theory chapters have a
complete mathematical section map and explicit source support for the central
formulas.
