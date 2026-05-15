# Phase P10 plan: cross-chapter notation, claim-status, and literature consolidation

## Date

2026-05-14

## Target scope

All reviewer-grade DPF chapters after P3-P9 revisions.

## Governing prerequisites and lane guard

- Required prior results: P0, P1, P2, and passed P3-P9 results.
- P2 remains the source-grounding register unless superseded by a later
  reviewed source-intake artifact.
- Allowed write set: DPF reviewer-grade chapters, this phase result artifact,
  and reviewer-grade DPF planning notes needed for consolidation.  Do not touch
  student-baseline files.
- Before editing, record branch, `git status --short`, out-of-lane dirty files,
  and this write set.

## Purpose

Make the revised DPF block read as one coherent monograph argument.  This phase
exists because rigorous local chapters can still fail if notation, claim-status
language, or literature interpretation shifts across chapter boundaries.

## Required implementation instructions

1. Create a DPF notation index covering:
   - state and observation variables;
   - parameter variables;
   - particles and ancestors;
   - weights and empirical measures;
   - proposals;
   - flow maps and Jacobians;
   - transport couplings;
   - Sinkhorn scalings;
   - learned maps;
   - likelihood estimates;
   - HMC scalar targets and gradients.
2. Search all DPF chapters for inconsistent notation.
3. Normalize claim-status vocabulary:
   - exact model identity;
   - unbiased particle estimator;
   - consistent approximation;
   - approximate closure;
   - relaxed target;
   - learned surrogate;
   - engineering hypothesis;
   - unsupported claim.
4. Ensure each chapter has:
   - notation/object inventory;
   - assumptions;
   - derivation before verdict;
   - claim-status table or equivalent;
   - literature synthesis;
   - implementation diagnostics;
   - skeptical-reader limitations.
5. Check chapter transitions:
   - no later chapter assumes an unproved earlier claim;
   - no approximation layer disappears;
   - no target-status downgrade is delayed until a later chapter.
6. Harmonize literature descriptions across chapters.

## Required result artifact

Create:

- `docs/plans/bayesfilter-dpf-monograph-reviewer-grade-phase-p10-notation-claim-consolidation-result-{YYYY-MM-DD}.md`

## Audit rules

- If a symbol changes meaning, either rename it or explicitly state the change.
- If the same source is described in multiple chapters, the descriptions must
  not conflict.
- Every use of `exact`, `unbiased`, `relaxed`, `surrogate`, and `validated` must
  match the claim ledger.
- Tables must summarize arguments, not replace them.

## Required local tests/checks

- Run `rg` searches for core symbols and status words.
- Run label/citation checks.
- Use MathDevMCP search tools if useful for label neighborhoods.
- Compile or static-check table readability if full build is unavailable.
- Record whether source language still follows P2's bibliography-spine fallback
  and whether any claim incorrectly implies ResearchAssistant-reviewed support.
- Run the established LaTeX build and explicitly inspect table/layout risks
  because P0 identified DPF-local table readability warnings.

## Veto diagnostics

The phase fails if:

- notation conflicts remain unresolved;
- chapter transitions smuggle in claims;
- claim-status vocabulary is inconsistent;
- literature synthesis contradicts itself;
- local limitations are deferred to other chapters.
- the phase edits or stages student-baseline files;
- source-support status conflicts with P2;
- build or table-readability status is omitted.

## Exit gate

The DPF block is ready for derivation audit only when notation and claim-status
language are consistent across chapters.
