# Phase E1 plan: particle-flow theory enrichment

## Date

2026-05-11

## Purpose

Deepen the particle-flow foundations chapter into a more monograph-grade
mathematical treatment, especially around EDH, LEDH, homotopy construction,
continuity-equation reasoning, and the exact/approximate boundary.

## Target chapter

- `docs/chapters/ch19b_dpf_literature_survey.tex`

## Main questions

1. What exact mathematical problem is the flow solving?
2. Under what assumptions is EDH exact or only approximate?
3. How does LEDH alter the derivation and the approximation status?
4. What numerical and implementation issues emerge directly from the flow
   mathematics?

## Required enrichment topics

- fuller derivation of the homotopy path and normalized density;
- fuller continuity-equation argument and velocity-field interpretation;
- clearer statement of what is exact in the linear-Gaussian recovery case;
- deeper EDH derivation and assumptions;
- deeper LEDH derivation with explicit local linearization steps;
- stronger discussion of stiffness, pseudo-time discretization, and what these
  imply for implementation.

## Required source lanes

- CIP particle-flow chapters;
- Daum--Huang source papers;
- Li/Coates / LEDH sources;
- student report material only as comparison on derivation presentation or topic
  coverage.

## Required outputs

1. expanded section-by-section chapter outline;
2. EDH and LEDH theorem/derivation support ledger;
3. explicit exact-vs-approximate comparison table;
4. implementation issue notes tied to the literature.
5. per-section triage statements of:
   - exact source claim or theorem;
   - BayesFilter local derivation or adaptation;
   - implementation/debugging implication.

## Required tables

### Table E1-1: EDH/LEDH derivation support

| Formula / claim | Source | Derive locally? | Approximation point | Audit tool |
| --- | --- | --- | --- | --- |

### Table E1-2: exact / approximate boundary

| Construction | Exact in what regime? | Approximate where? | Why |
| --- | --- | --- | --- |

### Table E1-3: debugging relevance map

| Mathematical issue | Likely implementation symptom | Relevant source / section | Why |
| --- | --- | --- | --- |

## Primary criterion

The chapter must become substantially richer in derivation content and make the
exact/approximate distinction impossible to miss.

## Veto diagnostics

Do not proceed if:
- EDH remains only a short sketch rather than a developed derivation;
- LEDH remains only a brief verbal variation on EDH;
- stiffness remains only a casual note rather than a mathematically motivated
  implementation concern.

## Exit gate

Proceed to E2 only after the flow chapter is deep enough that a reader could
implement or debug EDH/LEDH from the exposition with the cited literature trail.
