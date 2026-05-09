# Phase M6 plan: DPF chapter drafting, derivation audit, and rewrite execution

## Date

2026-05-09

## Purpose

This phase plans the actual writing and rewriting execution once the literature
survey and chapter architecture are complete.

## Scope

- drafting the rebuilt reader-facing DPF chapters;
- replacing or heavily rewriting the current DPF draft chapters;
- auditing load-bearing equations with MathDevMCP;
- tying claims to ResearchAssistant-backed sources;
- maintaining a disciplined distinction between derivation, sourced formula,
  approximation, and hypothesis.

## Required output

Produce an execution ladder for the drafting phase.

## Required execution cycle

Each chapter should follow:

```text
source map -> derivation map -> draft -> tool audit -> revise -> compile -> claim audit
```

## Required per-chapter checklist

- section outline completed;
- claim/source ledger completed;
- equations derived or source-tagged;
- MathDevMCP audit run on bounded formulas;
- ResearchAssistant support recorded for literature-heavy claims;
- prose checked for repo/governance drift;
- implementation consequences stated mathematically rather than casually.

## Required audit artifacts

- one phase note recording equations audited and tool limitations;
- one claim audit note recording unsupported or deferred claims;
- one replacement map showing which old DPF draft text was retired or rewritten.

## Exit gate

Proceed only when the draft chapters can be compiled and every load-bearing claim
has a known evidence status.
