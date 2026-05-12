# Program: BayesFilter DPF monograph enrichment and source-audit round

## Date

2026-05-11

## Status

This program governs the **second major round** of the DPF monograph work.  The
first major round repaired the architecture of the DPF block and produced a much
stronger reader-facing rewrite sequence.  That round is not enough by itself.
The current round is a deeper enrichment and source-audit round whose purpose is
not to summarize existing material but to expand it into a more self-contained,
literature-connected, implementation-useful monograph treatment.

This new program should therefore be read as a successor to the earlier rebuild
program in `docs/plans/bayesfilter-dpf-monograph-rebuild-master-program-2026-05-09.md`.
The earlier program established the chapter architecture and first substantial
rewrite.  The present program governs the next standard: a monograph that is
rich enough, rigorous enough, and well-sourced enough to act as a technical
reference for future implementation and debugging.

## Purpose

Deepen and enrich the rebuilt DPF monograph block so that it:

1. is genuinely self-contained at the level of a mathematical R\&D monograph;
2. gives substantially more derivation detail than the first rewrite round;
3. contains a real literature survey rather than a minimal source-supported
   summary;
4. verifies, audits, and cross-checks claims from the CIP monograph, student
   reports, and source papers;
5. explicitly connects mathematical claims to implementation questions,
   numerical pathologies, and debugging routes;
6. is detailed enough that a strong coding agent could use the text as a serious
   implementation and troubleshooting guide.

## Governing correction to the current state

The rebuilt DPF chapters are structurally much better than the earlier draft,
but they are still too brief and too compressed relative to the intended
standard.  The main deficiency is no longer architecture; it is **depth**.

In particular, the current chapters are still too close to:
- concise mathematical overviews,
- compressed chapter summaries,
- and broad interpretive statements.

They are not yet close enough to:
- full monograph-grade derivations,
- detailed literature positioning,
- and implementation-useful mathematical exposition.

This program exists to prevent the work from drifting toward the easier option:
writing short clean summaries instead of deep, technically rich chapters.

## Required standard

The standard for this round is intentionally high.

A chapter is not good enough merely because:
- it is correct at a high level;
- it has the right architecture;
- it compiles;
- or it sounds mathematically serious.

A chapter is good enough only if all of the following are true:

### 1. Mathematical self-containment

A reader should be able to follow the chapter without needing to consult the CIP
monograph, student reports, or internal plans to fill in essential steps.
Important equations, definitions, and approximation points must appear in the
chapter itself or be tied to explicit source equations in a way that still makes
the local exposition complete.

### 2. Literature depth

The chapter should not merely mention a few references.  It should summarize the
relevant literature enough that the reader understands:
- what the literature proves or claims;
- where different papers make different assumptions;
- what mathematical route each method takes;
- where our exposition agrees or departs;
- and which papers are relevant to which implementation or debugging issue.

### 3. Derivation depth

If a formula is load-bearing for implementation or interpretation, the chapter
should either:
- derive it in BayesFilter notation, or
- point to the exact source equation / theorem and explain its translation and
  assumptions.

Handwaving phrases like “this follows similarly” or “one can show” are not
acceptable for central formulas unless the omitted detail is genuinely routine
and non-load-bearing.

### 4. Approximation transparency

Every approximation layer must be explicit:
- Gaussian closure;
- local linearization;
- finite-particle approximation;
- transport relaxation;
- entropic regularization;
- learned surrogate approximation;
- compiled-path or numerical approximation where mathematically relevant.

The reader should always be able to answer:
- what is exact;
- what is unbiased;
- what is approximate;
- what is relaxed;
- what is learned;
- and what target HMC would actually sample.

### 5. Implementation usefulness

The chapter should help solve implementation problems.  If an implementation
issue arises, the chapter should help the reader identify:
- which quantity must be computed;
- which approximation layer may be responsible;
- which numerical pathology is relevant;
- and which literature should be consulted next.

### 6. Agent-readability standard

A good test is:

> if this chapter is given to an intelligent coding agent, the agent should be
> able to produce substantially better implementation code and debugging plans
> than it could from the current shorter summary-style draft.

If that test is not plausibly met, the chapter is still too thin.

## Required source lanes

This round must make deliberate use of the local MCP-capable tools whenever
possible.

### Lane A: ResearchAssistant source and claim lane

Use `~/python/ResearchAssistant` whenever possible to:
- verify paper identity and citation metadata;
- inspect source sections, equations, and labels when available;
- build source-support notes for claims;
- organize citation neighborhoods for method families;
- record what each paper is actually useful for.

ResearchAssistant should be used proactively, not only when something is broken.
The goal is to enrich the monograph with literature that is usable when future
implementation issues arise.

### Lane B: MathDevMCP derivation and source-audit lane

Use `~/python/MathDevMCP` whenever possible to:
- audit bounded derivations;
- locate and extract source LaTeX contexts;
- compare local formulas to source expressions;
- track exact load-bearing formulas and approximation points.

MathDevMCP is not an oracle.  But it should be used routinely enough that the
chapter does not drift into unsupported mathematical paraphrase.

### Lane C: CIP monograph source lane

Use the relevant CIP chapters not as a text to summarize, but as a source spine
to expand, verify, and improve upon where needed.  The local BayesFilter chapter
should become richer and more implementation-useful than a simple local summary
of CIP.

### Lane D: student-report comparison lane

Use the student sources as:
- coverage checks,
- alternative derivation presentations,
- implementation issue flags,
- and possible literature pointers.

They are not to be treated as monograph authority, but they are useful for
spotting omitted topics, overclaims, and implementation concerns that should be
connected back to the literature.

## Explicit anti-drift rules

Because the agent tends to drift toward easier options, this round must enforce
explicit anti-drift rules.

### Drift rule 1: no short summary substitution

Do not replace a needed derivation or literature discussion by a short summary
just because the summary is easier to write.

### Drift rule 2: no architecture complacency

Do not assume that a structurally correct chapter is already good enough.  The
new task is depth expansion, not architecture maintenance.

### Drift rule 3: no shallow citation padding

Do not add references merely to decorate the text.  Every important citation
should answer a question:
- what does this source prove or claim?
- why does it matter for this method?
- what implementation/debugging issue does it illuminate?

### Drift rule 4: no false self-containment

Do not call a chapter self-contained if it still requires the reader to infer
load-bearing steps from CIP or from a student report.

### Drift rule 5: no implementation-free mathematics

Do not write mathematics that is elegant but disconnected from what must be
computed in code.  The monograph should remain mathematically rich and
implementation-useful.

## Main outputs required from this round

This round should produce:

1. enriched DPF chapters with substantially deeper derivations and literature
   placement;
2. chapter-local claim/source support notes where needed;
3. a literature-to-implementation issue map;
4. a debugging-oriented literature guide embedded in the exposition or adjacent
   notes;
5. updated reset-memo records and final summary after each major enrichment pass.

## Program phases

This round should proceed in explicit phases.

1. **E0: source and claim-audit setup**
   - build per-chapter source obligations and literature gaps.

2. **E1: particle-flow theory enrichment**
   - deepen EDH / LEDH / flow PDE / linear-Gaussian recovery treatment.

3. **E2: PF-PF and Jacobian/log-det enrichment**
   - deepen proposal-correction derivations and interpretation.

4. **E3: differentiable resampling and OT enrichment**
   - deepen soft resampling, OT, Sinkhorn, barycentric map, and bias discussion.

5. **E4: learned/amortized OT enrichment**
   - deepen the teacher-student map treatment, residual interpretation, and
     implementation consequences.

6. **E5: DPF-specific HMC enrichment**
   - deepen rung-by-rung target analysis and DSGE/MacroFinance suitability.

7. **E6: literature-to-debugging crosswalk**
   - create explicit mappings from likely implementation issues to literature and
     chapter sections.

8. **E7: final enrichment audit and consolidation**
   - final audit of self-containment, literature depth, and implementation
     usefulness.

## Global execution cycle

Every phase must follow:

```text
plan -> execute -> test -> audit -> tidy -> update reset memo
```

And continue automatically only when:
- the primary criterion is met, and
- no veto diagnostic has fired.

## Phase-level primary criterion

For every enrichment phase, the primary criterion is:
- the chapter or artifact is materially more useful than before for both
  mathematical understanding and implementation/debugging.

## Phase-level veto diagnostics

A phase should not be considered complete if any of the following remain true:
- the chapter is still mostly summary rather than derivation;
- important claims lack source support or exact assumptions;
- the literature is mentioned but not synthesized;
- implementation implications remain vague;
- the result would not materially improve a strong coding agent's ability to
  produce or debug code.

## Success criteria for this round

This round succeeds only if:

1. the DPF chapters are no longer merely structurally correct, but genuinely
   rich and monograph-grade;
2. EDH, LEDH, PF-PF, resampling, OT, learned OT, and DPF-HMC each receive a
   treatment substantial enough to guide implementation;
3. the literature survey becomes a real technical reference rather than a thin
   reference list;
4. the text gives a reader pathways from implementation issues back to relevant
   literature and equations;
5. the final result is detailed enough that an intelligent coding agent could use
   it as a high-quality implementation and debugging guide.

## Immediate next step

Begin with E0: source and claim-audit setup for the enriched round, organized by
chapter and by method family, before attempting further deep chapter expansion.
