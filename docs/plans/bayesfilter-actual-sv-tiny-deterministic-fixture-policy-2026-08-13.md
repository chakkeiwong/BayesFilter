# Tiny deterministic actual-SV fixture policy

Date: 2026-08-13
Status: `DIAGNOSTIC_ONLY`

## Purpose

Record that the two-row hard-coded actual-SV fixture used in early comparison scripts is a **debugging fixture only** and must not be used as evidence for statistical approximation quality, convergence, or root-cause diagnosis.

## What the fixture is

In `docs/benchmarks/benchmark_actual_sv_two_lane_comparison.py`, the helper `_observations(dim)` returns a fixed array

\[
\begin{bmatrix}
0.12 & -0.08 & 0.05 \\
-0.07 & 0.11 & -0.04
\end{bmatrix}
\]

truncated to the requested dimension.

So the sample size is only **2 time points**.

## Policy

This fixture may be used for:
- shape checks,
- branch-consistency checks,
- finite-difference smoke tests,
- regression tests for known code paths.

It must **not** be used for:
- conclusions about approximation quality,
- claims of convergence,
- claims of equivalence between approximation families,
- statistical root-cause diagnosis.

## Why it is not admissible evidence

A two-observation hard-coded fixture is not generated from the DGP and is far too small to support inference about approximation behavior for stochastic volatility filtering. It is a deterministic benchmark input, not a stochastic experiment.

## Required future usage rule

Whenever a result note or benchmark artifact mentions this fixture, it must be labeled:
- `diagnostic_only`
- `smoke_test_only`
- or `not evidence for approximation-quality conclusions`

## Suggested next step

Use simulation from the exact SV DGP for any approximation-comparison or root-cause claim, and reserve the tiny deterministic fixture only for local debugging.
