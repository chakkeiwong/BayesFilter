# Phase P2 plan: source-grounding and literature synthesis

## Date

2026-05-14

## Governing program

- `docs/plans/bayesfilter-dpf-monograph-reviewer-grade-revision-master-program-2026-05-14.md`
- P0 preflight result.
- P1 claim ledger.

## Purpose

Turn the DPF bibliography into a source-grounded literature synthesis.  The
review panel should not need to search papers to understand what each source
contributes or what it does not prove for BayesFilter.

## Target source families

- SMC and bootstrap particle filters.
- High-dimensional particle degeneracy.
- Particle flow, EDH, LEDH, and related flow filters.
- PF-PF and proposal correction.
- Differentiable resampling.
- Optimal transport, entropy regularization, Sinkhorn, stabilization.
- Learned set maps, DeepSets, attention/set transformers, amortized transport.
- HMC, pseudo-marginal MCMC, noisy-gradient and surrogate MCMC.
- DSGE and MacroFinance structural filtering constraints.

## Implementation instructions

1. Extract all citation keys from the DPF chapters.
2. For each key, inspect `docs/references.bib` and record citation metadata.
3. Query ResearchAssistant for local summaries or source records.  If none are
   available, record the source as bibliography-only and do not overstate it.
4. For every source used in a load-bearing claim, write a source-role note:
   - what question the source addresses;
   - assumptions;
   - theorem, algorithm, or warning supported;
   - translation into BayesFilter notation;
   - limitation for DPF-HMC or banking use.
5. Identify source gaps:
   - missing primary source;
   - source present but not reviewed;
   - citation metadata uncertain;
   - source supports algorithm but not BayesFilter interpretation.
6. Draft chapter-local literature synthesis paragraphs for later insertion.
7. Create a "do not claim" list for claims that source support does not justify.

## Required result artifact

Create:

- `docs/plans/bayesfilter-dpf-monograph-reviewer-grade-phase-p2-source-grounding-2026-05-15.md`

This artifact already exists and is the governing P2 result unless superseded
by a later explicitly dated source-grounding result.

The result artifact must include:

1. source-role table;
2. source-gap register;
3. source-to-claim map;
4. literature synthesis insertion plan by chapter;
5. claims blocked by source gaps.

## Audit rules

- Prefer primary sources over secondary summaries when a claim is mathematical.
- Do not add citations for decoration.
- Do not cite student reports as authority; use them only for coverage,
  implementation warnings, or comparison.
- If ResearchAssistant has no local record, say so explicitly.
- If a source supports a method but not an HMC or banking conclusion, record the
  limit locally.

## Veto diagnostics

The phase fails if:

- bare citation lists remain the main source treatment;
- a major source has no role note;
- any banking or HMC claim is supported only by analogy;
- source identity is uncertain for a citation used in a central claim;
- missing sources are not recorded.

## Exit gate

Proceed to mathematical rewriting only when each central source has a clear
role and each central claim has either source support, a derivation obligation,
or a conservative downgrade.
