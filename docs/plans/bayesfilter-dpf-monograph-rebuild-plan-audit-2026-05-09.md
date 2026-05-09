# Audit: DPF monograph rebuild program and phase plans

## Date

2026-05-09

## Scope

This note audits the following plan set as if written by another developer:

- `docs/plans/bayesfilter-dpf-monograph-rebuild-master-program-2026-05-09.md`
- `docs/plans/bayesfilter-dpf-monograph-rebuild-phase-m0-preflight-plan-2026-05-09.md`
- `docs/plans/bayesfilter-dpf-monograph-rebuild-phase-m1-literature-survey-plan-2026-05-09.md`
- `docs/plans/bayesfilter-dpf-monograph-rebuild-phase-m2-architecture-plan-2026-05-09.md`
- `docs/plans/bayesfilter-dpf-monograph-rebuild-phase-m3-particle-flow-theory-plan-2026-05-09.md`
- `docs/plans/bayesfilter-dpf-monograph-rebuild-phase-m4-resampling-ot-plan-2026-05-09.md`
- `docs/plans/bayesfilter-dpf-monograph-rebuild-phase-m5-hmc-suitability-plan-2026-05-09.md`
- `docs/plans/bayesfilter-dpf-monograph-rebuild-phase-m6-drafting-execution-plan-2026-05-09.md`
- `docs/plans/bayesfilter-dpf-monograph-rebuild-phase-m7-integration-audit-plan-2026-05-09.md`

The purpose of the audit is to determine whether the program is complete enough,
properly ordered, mathematically disciplined enough for the stated monograph
standard, and free of obvious omissions before execution begins.

## Overall assessment

Assessment: **proceed, with two additions recorded below**.

The new plan set is materially better aligned with the user's corrected goal
than the earlier DPF program notes.  It correctly shifts the work from a
program-commentary style toward a mathematically serious monograph rebuild.  The
strongest improvements are:

- it explicitly treats the earlier DPF chapter pass as inadequate and subject to
  replacement rather than patching;
- it makes the literature survey primary and architecture secondary;
- it restores the distinction between reader-facing mathematical exposition and
  internal planning/governance;
- it uses the student work as critique and coverage input rather than as the
  voice of the monograph;
- it states target-status discipline as a first-class requirement.

The program is therefore justified.  However, two additions are needed so later
phases do not silently lose mathematical rigor.

## What is already strong

### 1. Correct ordering

The phase order is sound:

- preflight/supersession,
- literature grounding,
- chapter architecture,
- particle-flow theory planning,
- differentiable-resampling planning,
- HMC suitability planning,
- drafting/audit execution,
- integration/final audit.

This sequencing ensures that the final chapter structure is derived from the
literature rather than from a preconceived chapter count.

### 2. Correct standard for the final product

The master program now states the right standard:

- mathematically self-contained exposition,
- explicit target-status discipline,
- rigorous comparison to the literature,
- mathematically motivated implementation relevance,
- and a stronger HMC suitability analysis.

That is the correct repair to the earlier draft failure.

### 3. Proper use of source lanes

The four-lane structure is good:

- core literature,
- CIP extraction and rewrite,
- student critique and coverage,
- MathDevMCP / ResearchAssistant audit lane.

This should reduce the risk of overrelying on any one source family.

### 4. Separation of planning buckets

Breaking the later work into particle-flow theory, resampling/OT, and HMC
suitability is especially important.  Those are different mathematical layers,
and combining them too early is exactly how the earlier prose became vague.

## Missing point 1: bibliography / citation-key audit should be explicit earlier

The current plan set mentions bibliography discipline mainly in the final
integration phase.  That is necessary, but not sufficient.  The DPF literature
cluster is dense and likely to involve:

- duplicate or variant bibliographic keys,
- multiple versions of the same arXiv/published work,
- inconsistent naming conventions across CIP, BayesFilter, and student sources,
- possible citation drift where the student report cites a summary source rather
  than the primary theorem source.

### Required addition

Phase M1 should explicitly require a **citation-key and source-identity audit**
with a table such as:

| Claim family | Primary paper | Alternate versions | Preferred citation key | Notes |
| --- | --- | --- | --- | --- |

Without this, the later chapter drafting phase may become clogged with
bibliography corrections that should have been settled during the literature
survey.

## Missing point 2: theorem/derivation obligation register should be explicit

The plans rightly say that equations should be derived or carefully sourced, but
there is not yet an explicit per-equation obligation register.  For a topic this
subtle, that can lead to prose drifting ahead of derivation.

### Required addition

Phase M1 or M2 should generate a **load-bearing theorem/equation register**
listing at least:

- bootstrap PF likelihood estimator and status;
- EDH homotopy and flow equations;
- linear-Gaussian recovery statement for EDH;
- PF-PF change-of-variables weight formula;
- Jacobian/log-determinant evolution formula;
- soft-resampling bias statement;
- OT resampling primal/dual/barycentric map formulas;
- any claim relating a differentiable DPF objective to an HMC target.

For each item, record one of:

- derive fully in BayesFilter notation;
- cite and adapt carefully;
- compare source variants;
- human review required.

Without this register, the drafting phase could still become prose-first rather
than derivation-first.

## Phase-by-phase audit

### Phase M0

Good and necessary.  It should explicitly classify the current DPF chapter files
as reader-facing drafts to be replaced or heavily rewritten, not incrementally
polished.

### Phase M1

This is the most important phase and is correctly prioritized.  It should add
the two missing explicit outputs above:

1. citation-key/source-identity audit;
2. theorem/equation obligation register.

### Phase M2

Good.  The architecture must be allowed to expand beyond three chapters if the
literature demands it.  This phase is appropriately framed.

### Phase M3

Strong.  The scope is correct for the particle-filter and particle-flow
mathematics.  It would benefit from an explicit requirement to distinguish:

- exact flow in special cases,
- approximate flow under closure assumptions,
- and proposal-corrected usage under PF-PF.

That distinction is already implied, but it should remain explicit in execution.

### Phase M4

Strong.  It correctly centers the differentiability-versus-bias question.

### Phase M5

Essential and well scoped.  It correctly reframes the HMC question as a
rung-by-rung target question rather than a vague sampler question.

### Phase M6

Good execution discipline.  The phase should not begin until M1 and M2 have
produced the theorem/equation register and final chapter map.

### Phase M7

Good final integration plan.  This phase is correctly separated from the earlier
mathematical phases.

## Execution recommendation

Proceed with the plan set, but fold in the following two additions during the
first literature-survey phase and record them in the reset memo:

1. **citation-key/source-identity audit**;
2. **load-bearing theorem/equation obligation register**.

With those additions, the plan set is sufficiently complete to execute without
further architectural redesign.

## Audit disposition

Disposition: **approved with additions**.

No blocker prevents execution.  The next phase remains justified provided the
first literature phase explicitly records the citation-identity and derivation
obligation audits above.
