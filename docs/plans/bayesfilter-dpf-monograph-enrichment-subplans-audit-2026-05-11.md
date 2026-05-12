# Audit: DPF monograph enrichment subplans against the master standard

## Date

2026-05-11

## Scope

This audit reviews the enrichment subplans:

- `bayesfilter-dpf-monograph-enrichment-phase-e0-source-claim-audit-plan-2026-05-11.md`
- `bayesfilter-dpf-monograph-enrichment-phase-e1-particle-flow-theory-plan-2026-05-11.md`
- `bayesfilter-dpf-monograph-enrichment-phase-e2-pfpf-jacobian-plan-2026-05-11.md`
- `bayesfilter-dpf-monograph-enrichment-phase-e3-resampling-ot-plan-2026-05-11.md`
- `bayesfilter-dpf-monograph-enrichment-phase-e4-learned-ot-plan-2026-05-11.md`
- `bayesfilter-dpf-monograph-enrichment-phase-e5-hmc-enrichment-plan-2026-05-11.md`
- `bayesfilter-dpf-monograph-enrichment-phase-e6-debug-crosswalk-plan-2026-05-11.md`
- `bayesfilter-dpf-monograph-enrichment-phase-e7-final-audit-plan-2026-05-11.md`

The goal is to determine whether they actually enforce the stricter standard
laid down in the master enrichment program, especially the anti-drift rules, the
self-containment requirement, the literature-depth requirement, and the
implementation-usefulness requirement.

## Overall assessment

Assessment: **strong enough to proceed, but should be tightened in four ways**.

The subplans are directionally correct and materially better than a loose set of
chapter TODOs.  They do the most important thing right: they translate the high
standard from the master program into chapter-specific purposes, explicit tables,
primary criteria, and veto diagnostics.  That means they are already much less
likely than the earlier planning round to drift into the easy summary-first
mode.

However, they are still vulnerable to a quieter failure mode: they can be
executed competently and still yield chapters that are better than before but
not yet at the very high monograph-grade standard the master program calls for.
The issue is not missing chapter buckets.  The issue is that some subplans still
need stronger enforcement of depth, source specificity, and implementation
backlinking.

## What the subplans already do well

### 1. They preserve chapter separation

The subplans correctly keep distinct the mathematical layers:
- particle-flow foundations,
- PF-PF correction,
- differentiable resampling / OT,
- learned OT,
- HMC target suitability,
- debugging crosswalk,
- final audit.

This is essential, because merging them back together would recreate the earlier
structural failure.

### 2. They use primary criteria and veto diagnostics

This is the strongest feature of the subplans.  They do not simply list tasks;
they define failure conditions.  That is exactly what is needed to resist drift
into short summary-style prose.

### 3. They explicitly mention MCP usage

E0 in particular correctly requires proactive use of ResearchAssistant and
MathDevMCP.  This is essential if the resulting text is supposed to be usable as
a technical reference rather than just a polished overview.

### 4. They stay implementation-aware

The subplans do not lose sight of the implementation-usefulness standard.  That
is important, because the master program explicitly says the chapters should help
a future coding agent implement and debug the methods.

## Four needed tightenings

### Tightening 1: every enrichment phase should require a “what the literature proves vs what we infer” split

The master program strongly implies this distinction, but the subplans do not
always enforce it explicitly enough.

For chapter enrichment phases E1--E5, each should require a per-section split of:
- what the cited source explicitly proves or states,
- what BayesFilter derives from it in local notation,
- what remains an approximation or engineering interpretation.

Without this, a chapter can still sound rigorous while quietly upgrading source
claims into stronger local claims.

### Tightening 2: every enrichment phase should require a “debugging relevance” subsection, not just implementation relevance

The master program asks for literature that helps when implementation issues
arise.  The subplans often mention implementation implications, but they should
be stricter and require a specific debugging linkage.

For E1--E5, require a short artifact or section answering:
- if this part of the implementation fails, what literature should we consult,
  and why?

This matters because implementation usefulness is not the same thing as saying
what quantity a method computes.

### Tightening 3: E4 needs a stronger requirement to separate map-level approximation from posterior-level consequences

E4 is already good, but because learned/amortized OT is the place where agents
most easily drift toward casual ML language, it needs an even stricter rule.

E4 should explicitly require a subsection or table that distinguishes:
- teacher-map residual,
- transport-cost residual,
- distribution-level discrepancy,
- posterior-level shift,
- and HMC-level target change.

Otherwise the chapter may still stay too close to an architecture note rather
than a monograph treatment.

### Tightening 4: E5 should require explicit comparison to the dsge_hmc surrogate-HMC / HNN line at the level of *problem solved*, not just method family

E5 already mentions the comparison, but it should demand a sharper analytical
table such as:

| Method family | What is approximated? | What target changes? | What geometry changes? | What remains exact? |
| --- | --- | --- | --- | --- |

Without that, the comparison may remain too verbal and not sufficiently useful
for future design decisions.

## Phase-by-phase audit

### E0

Strong and necessary.  It is the right foundation.

Recommendation:
- add one required output: a per-chapter
  “source claim / local derivation / implementation-debug use” triage field.

### E1

Good, but still too easy to satisfy with a merely longer EDH/LEDH exposition.

Recommendation:
- require explicit per-section statements of:
  1. exact source claim,
  2. local derivation,
  3. implementation/debugging implication.

### E2

Strong and well targeted.

Recommendation:
- add one explicit requirement that the chapter explain how Jacobian/log-det
  mistakes would manifest in implementation outputs or diagnostics.

### E3

Strong.  This is one of the best subplans because it already frames the chapter
around the bias-versus-differentiability trade-off.

Recommendation:
- require a short subsection on how solver-budget choices (e.g. finite Sinkhorn
  iterations) change the practical target.

### E4

Good, but most in danger of drifting toward a polished ML summary.

Recommendation:
- strengthen residual / target-shift separation as described above.

### E5

Good and essential.

Recommendation:
- strengthen the dsge_hmc comparison into a structured analytical table.

### E6

Strong and very useful.  This directly supports the “feed it to a coding agent”
standard.

Recommendation:
- add one more issue class: mismatch between value object and gradient object.
  That is likely to be a major debugging category for DPF-HMC work.

### E7

Strong.  This is the right final test.

Recommendation:
- require that the final audit explicitly answer whether each chapter is now
  good enough to serve as a technical reference for future coding agents.

## Audit disposition

Disposition: **approve with tightenings**.

The subplans are already strong enough that execution would be worthwhile.
However, if the goal is the very high standard in the master program, the four
strengthenings above should be adopted before or during execution, especially in
E1, E4, and E5 where there is still the greatest risk of drifting toward easier
summary-style writing.

## Summary of required tightenings

1. Add explicit “source claim vs local derivation vs approximation/engineering
   interpretation” splits to E1--E5.
2. Add a required debugging-relevance subsection or artifact to E1--E5.
3. Strengthen E4 with a multi-level residual/target-shift separation.
4. Strengthen E5 with a structured comparison to dsge_hmc surrogate-HMC / HNN at
   the level of what is being approximated or changed.
5. Add “value object vs gradient object mismatch” to E6 issue categories.
6. Require E7 to assess whether each chapter is now genuinely useful to a strong
   coding agent.

With those tightenings, the subplans will match the stated master standard much
more closely.
