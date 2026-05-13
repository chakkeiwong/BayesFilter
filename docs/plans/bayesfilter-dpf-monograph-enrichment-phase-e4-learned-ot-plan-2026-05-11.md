# Phase E4 plan: learned/amortized OT enrichment

## Date

2026-05-11

## Purpose

Deepen the learned/amortized OT chapter so that it no longer reads like a short
interpretive note, but like a mathematically grounded treatment of a
teacher-student approximation layer.

## Target chapter

- `docs/chapters/ch19d_dpf_hmc_dsge_macrofinance_assessment.tex`

## Main questions

1. What exact map is being approximated by the learned operator?
2. How should the approximation residual be interpreted mathematically?
3. What kinds of training-distribution assumptions are structurally relevant?
4. What implementation consequences arise from this approximation hierarchy?

## Required enrichment topics

- fuller teacher-map / student-map formulation;
- stronger approximation-residual interpretation;
- stronger distinction between map-level residual and posterior-level shift;
- deeper treatment of state-dimension, weight-geometry, and regularization-range
  dependence;
- more explicit implementation consequences for deterministic evaluation,
  runtime, and debugging.

## Required source lanes

- advanced student architecture note and related notebooks;
- any primary learned-transport literature used to support or contrast that
  treatment;
- OT baseline from the preceding chapter.

## Required outputs

1. chapter-local source ledger for learned/amortized OT claims;
2. approximation hierarchy table;
3. implementation/debugging issue map for learned OT;
4. explicit list of what the chapter will not claim.
5. a multi-level residual/target-shift separation covering:
   - teacher-map residual;
   - transport-cost residual;
   - distribution-level discrepancy;
   - posterior-level shift;
   - HMC-level target change.

## Required tables

### Table E4-1: approximation hierarchy

| Layer | Mathematical object | New approximation introduced | Main risk |
| --- | --- | --- | --- |

### Table E4-2: learned-OT implementation issue map

| Issue | Mathematical source | Why it matters |
| --- | --- | --- |

### Table E4-3: residual and target-shift hierarchy

| Residual layer | Mathematical object | Why it matters | Relevant diagnostic |
| --- | --- | --- | --- |

### Table E4-4: residual interpretation guardrail

| Residual quantity | What it measures directly | What it does not by itself prove | What evidence would be needed next |
| --- | --- | --- | --- |

## Primary criterion

The chapter must make it impossible to read learned OT as a free exact
acceleration of OT rather than as a second approximation layer.

## Veto diagnostics

Do not proceed if:
- the teacher-versus-learned distinction is still too compressed;
- residuals are treated only as empirical error bars without target meaning;
- the training-distribution dependence remains only a casual remark.

## Exit gate

Proceed to E5 only once the chapter clearly positions learned OT as a layered
surrogate construction that the final HMC chapter can reason about explicitly.
