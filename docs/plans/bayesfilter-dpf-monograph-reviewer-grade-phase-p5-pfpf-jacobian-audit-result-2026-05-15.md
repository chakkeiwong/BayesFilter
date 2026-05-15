# Phase P5 result: PF-PF proposal correction and Jacobian audit

## Date

2026-05-15

## Branch and worktree classification

- Branch: `main`.
- Divergence: `main...origin/main [ahead 2]`.
- In-lane DPF chapter changes present:
  - `docs/chapters/ch19b_dpf_literature_survey.tex` from P4;
  - `docs/chapters/ch19c_dpf_implementation_literature.tex` from P5.
- In-lane reviewer-grade planning/result artifacts present as untracked files.
- User-owned in-lane governance edit present:
  - `docs/plans/bayesfilter-dpf-monograph-reviewer-grade-revision-master-program-2026-05-14.md`.
  - This result does not revert or overwrite it.  The live user instruction for
    this execution still governs the phase order.
- Out-of-lane student-baseline files present and not touched:
  - `docs/plans/bayesfilter-student-dpf-baseline-master-program-2026-05-10.md`;
  - `docs/plans/bayesfilter-student-dpf-baseline-reset-memo-2026-05-09.md`;
  - untracked `docs/plans/bayesfilter-student-dpf-baseline-*` artifacts.

## Governance reconciliation note

This result remains an in-lane provisional P5 artifact.  A later supervisor
audit tightened the governing execution order so that P3 is the next baseline
gate before P6.  Do not revert this P5 work solely because it was produced
before P3; after P3 completes, record whether P3 changes any baseline
definition, estimator-status claim, proposal notation, or differentiability
boundary that requires repairing this chapter before P6.

## Allowed write set used

- `docs/chapters/ch19c_dpf_implementation_literature.tex`.
- This result artifact.

No student-baseline file, git history operation, deletion, push, merge, rebase,
or reset was performed.

## Chapter changes completed

P5 turned `ch19c` from a conceptual PF-PF bridge into an explicit
proposal-density correction chapter.

Completed items:

1. Added an object inventory for ancestor state, pre-flow particle, post-flow
   particle, pre-flow proposal, flow map, inverse map, Jacobian, post-flow
   proposal, one-step target density, and corrected weight.
2. Added source-role language limiting the cited sources to proposal/SMC and
   invertible-flow support, not HMC validation.
3. Derived the transported proposal density using a differentiable bijective
   flow map and stated the forward versus inverse determinant convention.
4. Added the generic proposal-corrected weight
   `eq:bf-pfpf-generic-weight`.
5. Specialized the generic formula to the transition-prior proposal and
   clarified the numerator target and denominator proposal in the density ratio.
6. Expanded EDH/PF versus LEDH/PF status, including the particle-specific
   Jacobian burden for LEDH.
7. Derived the Jacobian ODE and log-determinant trace identity through Jacobi's
   formula, including the affine-flow simplification.
8. Added a section on what proposal correction restores and what it does not
   restore: it restores the proposal-to-target density ratio for the stated
   transported proposal, but it does not remove flow-closure, discretization,
   finite-particle, or log-determinant numerical approximation.
9. Added equation-indexed implementation audit obligations for affine density,
   forward/inverse sign, Jacobian ODE, corrected-versus-uncorrected weights,
   and path consistency.
10. Weakened HMC-facing language to avoid saying PF-PF is HMC-ready.  The
    chapter now says only that proposal-corrected EDH/PF or LEDH/PF are the
    first ladder rungs here with an explicit proposal-corrected value-side
    interpretation.

## Derivation-obligation audit

| Obligation | Chapter location | Method | Result |
|---|---|---|---|
| Change of variables for transported proposal density | `eq:bf-pfpf-postflow-density` | MathDevMCP label lookup plus manual density check | Passed; determinant is stated as the forward map and inverse convention is explicit. |
| Generic proposal-corrected weight | `eq:bf-pfpf-generic-weight` | MathDevMCP label lookup plus manual target/proposal audit | Passed; numerator and denominator objects are named and the bootstrap formula is a special case. |
| Jacobian ODE and log-det trace identity | `eq:bf-pfpf-jacobian-ode`, `eq:bf-pfpf-logdet-ode` | MathDevMCP label lookup plus manual Jacobi-formula audit | Passed; trace cyclicity and affine simplification are stated. |
| HMC-facing claim boundary | `sec:bf-pfpf-hmc-bridge` | Manual overclaim audit | Passed; no claim that PF-PF is HMC-ready. |

MathDevMCP evidence recorded during the phase:

- `latex_label_lookup` succeeded for `eq:bf-pfpf-generic-weight`;
- `latex_label_lookup` succeeded for `eq:bf-pfpf-logdet-ode`;
- `latex_label_lookup` succeeded for `eq:bf-pfpf-postflow-density`.

ResearchAssistant was not used as supporting claim evidence in P5 because P2
established that the local ResearchAssistant workspace has no DPF paper
summaries in read-only/offline mode.  Source support remains bibliography-spine
plus local derivation, not ResearchAssistant-reviewed support.

## Local checks run

- `git branch --show-current`.
- `git status --short --branch`.
- `git log --oneline --decorate -5`.
- `rg` checks in `ch19c` for:
  - `sec:bf-pfpf-objects`;
  - `eq:bf-pfpf-generic-weight`;
  - `eq:bf-pfpf-logdet-ode`;
  - `eq:bf-pfpf-postflow-density`;
  - `correct`;
  - `Jacobi`;
  - `trace`;
  - `HMC-ready`;
  - `first serious`.
- `rg` check in `docs/main.log` for undefined citations, undefined references,
  multiply defined labels, and rerun warnings.
- LaTeX build command:
  - working directory: `docs`;
  - command: `latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex`;
  - result: passed; `latexmk` reported all targets up to date.

## Remaining cautions

- P5-local layout caution: an overfull line remains around
  `ch19c_dpf_implementation_literature.tex` lines 338--341 in the audit
  obligation list.  This is not a mathematical veto, but it should be revisited
  in P10/P12 layout consolidation.
- Existing DPF/local table and header warnings remain from earlier chapters.
- The master-program governance file currently has a user-owned diff that
  changes the executable order to put P3 before P4.  This result follows the
  user's latest explicit execution order instead of rewriting that governance
  file.

## Veto diagnostic review

- PF-PF remains only a conceptual bridge: no.
- Determinant sign convention ambiguous: no.
- Trajectory-level status unclear: no; one-step conditional and path-level
  limits are stated.
- HMC relevance asserted before caveats: no.
- Diagnostics not tied to equations: no.

## Exit gate

Passed.

P5 is ready because a skeptical reviewer can audit PF-PF as a proposal-density
correction and identify what remains approximate: flow construction,
ODE/discretization, finite-particle error, and numerical determinant/log-det
computation.

## Superseded next recommendation at the time this result was written

At the time this result was first written, the provisional next recommendation
named P6:

- `docs/plans/bayesfilter-dpf-monograph-reviewer-grade-phase-p6-resampling-ot-sinkhorn-plan-2026-05-14.md`;
- target chapter: `docs/chapters/ch32_diff_resampling_neural_ot.tex`.

That recommendation is superseded by the current governing gate below.

Current governing next gate:

- P3 baseline expansion, followed by P4/P5 impact reconciliation before P6.
