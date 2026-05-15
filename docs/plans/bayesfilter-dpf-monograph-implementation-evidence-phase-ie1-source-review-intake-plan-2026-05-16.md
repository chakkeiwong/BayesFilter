# Phase IE1 plan: source-review intake

## Date

2026-05-16

## Purpose

Decide whether the implementation-and-evidence program can upgrade source
support from bibliography-spine provenance to reviewed primary-source evidence.
IE1 does not block numerical diagnostic work if source intake is not authorized;
it records the caveat explicitly.

## Allowed Write Set

- `docs/plans/bayesfilter-dpf-monograph-implementation-evidence-phase-ie1-source-review-result-{YYYY-MM-DD}.md`;
- optional `experiments/dpf_monograph_evidence/reports/outputs/source_status_register.json`
  only if `experiments/dpf_monograph_evidence/` already exists.

No ResearchAssistant artifacts may be written during IE1 execution.  Any source
intake, network discovery, fetch, ingestion, or ResearchAssistant mutation
requires a separate grant-bound workflow and is not part of this phase.

## Inputs

- reviewer-grade P2 source-grounding result:
  `docs/plans/bayesfilter-dpf-monograph-reviewer-grade-phase-p2-source-grounding-2026-05-15.md`;
- `docs/references.bib`;
- local ResearchAssistant workspace status and paper summaries, read-only.

## Tasks

1. Query ResearchAssistant for stored summaries covering SMC, EDH/LEDH, PF-PF,
   differentiable resampling, Sinkhorn/EOT, learned OT, HMC, DSGE, and
   MacroFinance.
2. Record whether each source family has reviewed local summaries,
   bibliography-spine-only support, or no local support.
3. If intake is desired, produce a grant-bound intake plan rather than fetching.
4. Create a source-status register for use by IE2--IE8.
5. Define the source-support vocabulary that every later Markdown and JSON
   result must carry:
   - `reviewed_local_summary`;
   - `bibliography_spine_only`;
   - `local_derivation_only`;
   - `source_gap`;
   - `not_source_dependent`.
6. If local read-only summaries are absent or insufficient and no separate
   intake authorization exists, record source families as
   `bibliography_spine_only`, `source_gap`, or `not_source_dependent` as
   appropriate, exit `ie1_source_review_deferred`, and explicitly state that the
   deferral does not block IE2.

## Source-Support Vocabulary

- `reviewed_local_summary`: a local read-only reviewed summary exists and was
  consulted during this program.
- `bibliography_spine_only`: citation/provenance exists in
  `docs/references.bib` or the prior reviewer-grade source map, but no reviewed
  local summary was available.
- `local_derivation_only`: support comes from current derivation, controlled
  fixture logic, or schema/governance reasoning, not source review.
- `source_gap`: a later phase would need source-backed grounding beyond the
  current bibliography spine before making a stronger claim.
- `not_source_dependent`: the item is purely engineering, schema, governance,
  or artifact-control work.

## Primary Criterion

Every source family has a recorded support class and no later phase can confuse
bibliography-spine support with reviewed-source support.

## Veto Diagnostics

- network or arXiv intake is attempted without explicit grant;
- source support is overstated;
- missing sources are silently converted into positive evidence.
- absence of local reviewed summaries is treated as a blocker for IE2 rather
  than a deferred source-review caveat.

## Expected Artifacts

- `docs/plans/bayesfilter-dpf-monograph-implementation-evidence-phase-ie1-source-review-result-{YYYY-MM-DD}.md`;
- optional `experiments/dpf_monograph_evidence/reports/outputs/source_status_register.json`
  if the evidence root already exists.

## Exit Labels

- `ie1_source_review_registered`;
- `ie1_source_review_deferred`;
- `ie1_source_review_blocked`.
