# Phase P1 plan: skeptical-reader argument map and claim ledger

## Date

2026-05-14

## Governing program

- `docs/plans/bayesfilter-dpf-monograph-reviewer-grade-revision-master-program-2026-05-14.md`
- P0 preflight result, once created.

## Purpose

Create a claim ledger and argument map before rewriting any mathematical prose.
The DPF block must read as a cumulative argument for skeptical academics, not as
a sequence of method summaries.

## Target scope

All DPF chapters and their cross-references:

- classical filtering and SMC;
- particle-flow homotopy, EDH, LEDH;
- PF-PF proposal correction;
- differentiable resampling and OT/Sinkhorn;
- learned/amortized OT;
- DPF-HMC target correctness;
- debugging and implementation verification.

## Implementation instructions

1. Build a chapter dependency map:
   - exact nonlinear filtering recursion;
   - particle likelihood estimator;
   - weight degeneracy and resampling bottleneck;
   - homotopy density and flow transport;
   - EDH/LEDH closure;
   - proposal correction and Jacobian determinant;
   - categorical versus relaxed resampling;
   - OT/EOT/Sinkhorn solver path;
   - learned teacher/student transport;
   - HMC scalar target and gradient path;
   - banking-model suitability and validation evidence.
2. Extract every central claim from the current DPF chapters.
3. Classify each claim using the master-program vocabulary:
   - exact model identity;
   - unbiased particle estimator;
   - consistent approximation;
   - approximate closure;
   - relaxed target;
   - learned surrogate;
   - engineering hypothesis;
   - unsupported claim.
4. For each claim, record:
   - chapter and section;
   - local equation/table/proposition if any;
   - source support;
   - assumptions;
   - approximation layer;
   - implementation implication;
   - reviewer risk.
5. Identify claims that should be weakened before any prose expansion.
6. Identify claims that require derivation expansion before they can remain in
   the reader-facing text.

## Required result artifact

Create:

- `docs/plans/bayesfilter-dpf-monograph-reviewer-grade-phase-p1-claim-ledger-2026-05-15.md`

This artifact already exists and is the governing P1 result unless superseded
by a later explicitly dated claim-ledger result.

The result artifact must include:

1. argument map;
2. claim ledger table;
3. claims to strengthen;
4. claims to weaken or qualify;
5. missing prerequisite definitions;
6. chapter-order risks.

## Audit rules

- Every claim using words like `exact`, `unbiased`, `correct`, `valid`,
  `target`, `HMC`, `production`, `validated`, or `suitable` must have a local
  status classification.
- Tables cannot substitute for missing derivations; flag any table row that
  carries a central claim without surrounding derivation.
- Banking and HMC claims must be classified more conservatively than algorithmic
  identities.
- If a claim cannot be classified, mark it unsupported rather than leaving it
  ambiguous.

## Veto diagnostics

The phase fails if:

- the claim ledger omits HMC or banking claims;
- exact/unbiased/validated language is left unclassified;
- later phases can begin without knowing which claims are hypotheses;
- source support and derivation obligations are blurred.

## Exit gate

Proceed to P2 only when every central DPF claim has a status, source-support
state, and reviewer-risk classification.
