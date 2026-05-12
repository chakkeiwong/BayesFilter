# Phase E0 plan: source and claim-audit setup for DPF monograph enrichment

## Date

2026-05-11

## Purpose

This phase establishes the source-grounding and claim-audit scaffolding for the
DPF monograph enrichment round.  Its purpose is to prevent the next chapter
expansion passes from drifting into shallow summary prose or unsupported
mathematical paraphrase.

## Main question

For each rebuilt DPF chapter, what exact claims, derivations, and implementation
issues need explicit source support, and which literature should be attached to
each of them?

## Scope

This phase covers the current DPF chapter sequence:
- `docs/chapters/ch19_particle_filters.tex`
- `docs/chapters/ch19b_dpf_literature_survey.tex`
- `docs/chapters/ch19c_dpf_implementation_literature.tex`
- `docs/chapters/ch32_diff_resampling_neural_ot.tex`
- `docs/chapters/ch19d_dpf_hmc_dsge_macrofinance_assessment.tex`
- `docs/chapters/ch19e_dpf_hmc_target_suitability.tex`

## Required MCP usage

### ResearchAssistant

Use `~/python/ResearchAssistant` whenever possible to:
- verify paper identity and metadata;
- inspect source sections, equations, theorems, and citation neighborhoods;
- produce claim-support notes for chapter-level mathematical claims;
- identify missing or stronger primary sources.

### MathDevMCP

Use `~/python/MathDevMCP` whenever possible to:
- audit bounded derivations and source-equation translations;
- extract source LaTeX contexts for load-bearing formulas;
- compare local notation to source expressions;
- identify where human review remains necessary.

## Required outputs

Create at least these artifacts:

1. **Per-chapter claim inventory**
   - list theorems, propositions, derivations, approximation-status claims,
     and implementation-facing claims for each chapter.

2. **Per-chapter source ledger**
   - for each load-bearing claim, record:
     - source paper;
     - source section/equation/theorem if available;
     - whether the chapter should derive it locally or cite/adapt it.

3. **Implementation-issue literature map**
   - record likely implementation/debugging issues and which papers/sections are
     relevant to diagnosing them.

4. **Gap register**
   - identify where CIP, student sources, or the current BayesFilter chapters are
     too thin or potentially misleading.

5. **Per-chapter triage field**
   - for each load-bearing item, classify:
     - what the literature/source explicitly proves or states;
     - what BayesFilter will derive locally;
     - what implementation/debugging use the item serves.

6. **Per-chapter priority register**
   - rank claims/formulas into:
     - critical for implementation;
     - critical for target interpretation;
     - useful supporting context;
     - optional enrichment.

## Required tables

### Table E0-1: chapter claim ledger

| Chapter | Claim / derivation | Status now | Needed source support | Needed action |
| --- | --- | --- | --- | --- |

### Table E0-2: implementation-issue map

| Issue | Affected chapter(s) | Relevant literature | Why relevant |
| --- | --- | --- | --- |

### Table E0-3: source gap register

| Topic | Current coverage | Missing or weak source support | Next source action |
| --- | --- | --- | --- |

## Primary criterion

This phase succeeds only if every chapter has a clear claim ledger and there is
no load-bearing formula whose support status is still implicit.

## Veto diagnostics

Do not proceed if:
- chapter claims are still described only at a topic level rather than at a
  theorem/derivation level;
- source identity is still ambiguous for important references;
- implementation-useful literature links are still missing.

## Exit gate

Proceed to E1 only when the enrichment work can be driven chapter by chapter by
explicit claim and source obligations rather than by memory or intuition.
