# Phase E6 plan: literature-to-debugging crosswalk

## Date

2026-05-11

## Purpose

Turn the enriched DPF monograph into a practical debugging reference by mapping
likely implementation failures back to chapter sections, equations, and source
papers.

## Main question

If a future implementation fails, how does a reader locate the right chapter,
formula, and source paper quickly enough to diagnose the problem correctly?

## Scope

This phase spans the full rebuilt DPF block and creates a crosswalk rather than a
single chapter rewrite.

## Required outputs

1. a debugging-oriented literature crosswalk note;
2. chapter-local pointers to failure modes where appropriate;
3. a short issue taxonomy usable by future coding agents.

## Required issue categories

At minimum include:
- particle degeneracy / ESS collapse;
- flow stiffness;
- EDH/LEDH approximation mismatch;
- PF-PF Jacobian/log-det mismatch;
- soft-resampling bias problems;
- OT regularization and Sinkhorn convergence issues;
- learned-OT extrapolation / residual issues;
- DPF-HMC target mismatch issues;
- mismatch between value object and gradient object.

## Required table

| Failure mode | Likely mathematical layer | Relevant chapter section | Relevant literature | Suggested next diagnostic |
| --- | --- | --- | --- | --- |

## Primary criterion

A future agent should be able to use this crosswalk to move from an observed
implementation failure to a short list of mathematically relevant sources and
chapter sections.

## Veto diagnostics

Do not proceed if:
- the crosswalk stays too abstract to guide debugging;
- the mapping from issue to chapter/literature is still vague;
- the resulting artifact would not materially help implementation diagnosis.

## Exit gate

Proceed to E7 only once the monograph is not merely rich, but also usable as a
technical troubleshooting reference.
