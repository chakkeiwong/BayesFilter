# Phase M4 plan: differentiable resampling and optimal transport chapters

## Date

2026-05-09

## Purpose

Plan the chapters that treat differentiable resampling rigorously, including
soft resampling, OT resampling, and learned/amortized OT operators.

## Scope

This phase covers:

- why standard resampling is not pathwise differentiable;
- soft-resampling constructions and bias analysis;
- OT and entropic OT resampling as geometric relaxations;
- Sinkhorn-type formulations and their implementation consequences;
- learned or amortized OT operators as approximations to the OT map;
- what each resampling family changes in the target being differentiated.

## Required output

Produce a detailed chapter/section writing plan with explicit mathematical
questions and comparison goals.

## Mandatory topics

- categorical ancestor selection as a discontinuous map;
- what differentiability means for the resampling step;
- bias-versus-differentiability trade-offs;
- Wasserstein / barycentric interpretations of OT resampling;
- regularization and solver-budget dependence;
- learned transport operators and approximation error;
- which objects can still be interpreted as estimators of the original target.

## Required comparison structure

The plan must include a rigorous comparison template for each resampling family:

| Method | Mathematical object | Source of differentiability | Source of bias / approximation | Implementation burden | HMC interpretation |
| --- | --- | --- | --- | --- | --- |

## Exit gate

Proceed only once the resampling block has an explicit map of what is exact,
what is relaxed, and what is learned or surrogate.
