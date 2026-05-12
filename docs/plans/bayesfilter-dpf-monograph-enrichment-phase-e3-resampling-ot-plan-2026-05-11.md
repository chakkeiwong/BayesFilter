# Phase E3 plan: differentiable resampling and OT enrichment

## Date

2026-05-11

## Purpose

Deepen the differentiable-resampling and OT chapter so that soft resampling,
transport/projection formulations, entropic OT, Sinkhorn structure, and the
bias-versus-differentiability trade-off are treated at monograph depth.

## Target chapter

- `docs/chapters/ch32_diff_resampling_neural_ot.tex`

## Main questions

1. Why exactly is standard resampling not pathwise differentiable?
2. What does soft resampling preserve, and where precisely does bias enter?
3. What is the exact mathematical object defined by entropic OT resampling?
4. How should regularization and solver-budget choices be interpreted in target
   terms?

## Required enrichment topics

- more formal discontinuity analysis of standard resampling;
- deeper moment/bias treatment for soft resampling;
- stronger transport/projection interpretation;
- fuller entropic OT setup, including primal/dual or scaling interpretations as
  needed;
- fuller Sinkhorn discussion with numerical interpretation;
- stronger explanation of barycentric projection;
- explicit hierarchy among categorical, soft, OT-relaxed, and later learned
  resampling.

## Required source lanes

- OT / differentiable-resampling primary literature;
- CIP resampling/OT chapter;
- student notes only as implementation-issue or coverage checks.

## Required outputs

1. source ledger for soft resampling and OT claims;
2. bias-source map by method family;
3. implementation issue map for Sinkhorn and OT regularization;
4. chapter-local comparison of exactness, differentiability, and target drift.
5. explicit subsection or artifact on how solver-budget choices (for example,
   finite Sinkhorn iterations) change the practical target.

## Required tables

### Table E3-1: resampling family comparison

| Method | Exactness status | Differentiability source | Bias source | Solver / runtime issue |
| --- | --- | --- | --- | --- |

### Table E3-2: implementation debug map

| Problem | Likely mathematical cause | Relevant literature |
| --- | --- | --- |

### Table E3-3: source claim / local derivation / debugging use

| Item | Source claim | Local derivation task | Debugging relevance |
| --- | --- | --- | --- |

### Table E3-4: OT object versus solver approximation

| Layer | Mathematical object | Exact or approximate? | Source of approximation |
| --- | --- | --- | --- |

## Primary criterion

The chapter must make the bias-versus-differentiability trade-off mathematically
and operationally explicit enough that later HMC discussion can rely on it.

## Veto diagnostics

Do not proceed if:
- OT is still described too much as an intuitive idea rather than a precise
  relaxed transport construction;
- soft-resampling bias is still only verbally described;
- regularization and finite-iteration issues remain disconnected from target
  interpretation.

## Exit gate

Proceed to E4 only once the chapter gives a strong enough OT/resampling account
that learned/amortized OT can be treated explicitly as a further approximation
layer, not as a replacement summary.
