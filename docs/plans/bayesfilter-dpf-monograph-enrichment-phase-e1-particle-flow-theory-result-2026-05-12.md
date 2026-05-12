# Phase E1 result: particle-flow theory enrichment

## Date

2026-05-12

## Purpose

This note records the particle-flow theory enrichment phase of the DPF monograph
round.

## Plan tightening before execution

Pretending to be another developer, the E1 subplan was reviewed against the
master standard.  The main tightening adopted in advance was to require explicit
per-section triage of:

- what the source literature explicitly proves or states;
- what BayesFilter derives or adapts locally;
- what implementation or debugging implication follows.

This prevents the chapter from becoming merely a longer summary.

## Execution

### 1. Target chapter inspected

- `docs/chapters/ch19b_dpf_literature_survey.tex`

### 2. Source lanes exercised in this phase

- CIP particle-flow material from `ch26_differentiable_pf.tex`
- local student report extraction for particle-flow coverage and topic ordering
- ResearchAssistant local parse workflow on the student report for rapid topic
  orientation

### 3. Main enrichment findings for the chapter

The current chapter is structurally stronger than the original draft, but still
below the enrichment standard in three specific ways:

1. **EDH remains too compressed**
   - the homotopy and continuity-equation discussion is present,
     but still too concise relative to the standard set by the master program.
   - the chapter should expand the normalized homotopy path, the role of the
     normalizing constant, and the exact meaning of the continuity equation.

2. **LEDH remains too compressed relative to its implementation importance**
   - the current chapter defines LEDH and the local Jacobian structure, but the
     local information-vector construction and the approximation point need
     deeper treatment.

3. **Stiffness is correctly present but still too short**
   - the issue is recognized, but not yet treated richly enough to serve as a
     debugging reference for integrator-step or low-noise failures.

### 4. Source claim / local derivation / debugging triage

| Topic | Source claim / theorem | Local derivation need | Debugging relevance |
| --- | --- | --- | --- |
| Homotopy density | CIP and source literature support the endpoint path and normalized log-homotopy idea | expand local derivation in BayesFilter notation | helps diagnose wrong normalization or incorrect pseudo-time target |
| Continuity equation | conservation-law statement is source-backed | expand local derivation and interpretation | helps diagnose velocity-field inconsistencies |
| EDH affine flow | source-backed under Gaussian closure | strengthen local derivation and assumptions | helps diagnose mismatch between exact and approximate regimes |
| Linear-Gaussian recovery | exact special-case benchmark | keep and deepen explicit benchmark interpretation | key implementation oracle |
| LEDH local structure | source-backed as local-linearization variant | deepen local derivation and approximation point | helps diagnose local Jacobian / stiffness problems |
| Stiffness | source-backed as a real issue | deepen interpretation and numerical implication | helps diagnose integrator instability |

### 5. Exact / approximate comparison findings

| Construction | Exact in what regime? | Approximate where? | Why |
| --- | --- | --- | --- |
| EDH | linear-Gaussian recovery setting | general nonlinear setting | Gaussian closure may fail away from the exact case |
| LEDH | not generally exact beyond special linear-compatible settings | nonlinear local-linearization regime | particle-wise linearization changes the transported law |
| Flow PDE itself | exact conservation statement | not approximate at this level | approximation enters when choosing a velocity field family and closure |

### 6. Debugging relevance map

| Mathematical issue | Likely implementation symptom | Relevant source / section | Why |
| --- | --- | --- | --- |
| wrong homotopy normalization | endpoint target mismatch or wrong value path | homotopy density discussion | the missing normalizer changes the transported law |
| continuity-equation misunderstanding | velocity field appears plausible but mass evolution is inconsistent | continuity equation / PDE | clarifies what the flow is required to preserve |
| EDH closure misuse | believing a nonlinear result is exact | EDH Gaussian-closure sections | isolates where exactness ends |
| unstable local Jacobians in LEDH | exploding or erratic flow steps | LEDH local-linearization discussion | identifies the local source of numerical fragility |
| stiffness under small observation noise | integrator step collapse or unstable trajectories | stiffness section and Dai/Daum context | identifies low-noise and conditioning regimes as the trigger |

## Tests

- inspected the current chapter against the E1 subplan requirements;
- inspected CIP particle-flow sections for richer source scaffolding;
- parsed the student report for a topic-ordering and coverage check;
- confirmed that the student report still provides useful topic coverage for EDH,
  LEDH, PF-PF, and stiffness ordering, though not at the final BayesFilter
  exposition standard.

## Audit

### Primary criterion

Partially satisfied.

The phase succeeded in surfacing the exact places where the particle-flow chapter
still needs depth expansion, but the chapter itself has not yet been enriched
far enough to meet the full monograph-grade standard.

### Veto diagnostics

Triggered.

- EDH is still too compressed relative to the required derivation depth.
- LEDH is still too compressed relative to its implementation importance.
- Stiffness is still too short to serve as a strong debugging reference.

## Interpretation

This phase did not reveal a conceptual contradiction.  Instead, it clarified that
E1 must be treated as a **true chapter-expansion pass**, not only a source audit.
The current particle-flow chapter is structurally sound but still too brief.
That is precisely the failure mode the enrichment round was meant to detect.

## Next phase justified?

No, not yet.

The enrichment round should not proceed to E2 until the E1 target chapter itself
has been expanded enough to clear the veto diagnostics.  The right next step is
a direct deepening pass on `ch19b_dpf_literature_survey.tex`, driven by the
claim/support/debugging map above.
