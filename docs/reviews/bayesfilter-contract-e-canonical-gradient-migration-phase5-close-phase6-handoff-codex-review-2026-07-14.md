# Phase 5 Close / Phase 6 Handoff Review

Date: 2026-07-14

Program ID: `contract-e-canonical-gradient-migration-20260713`

Reviewer: fresh bounded Codex substitute reviewer after Claude was platform-blocked

## First Verdict

`VERDICT: REVISE`

The first review found three material documentation/evidence defects:

1. the Phase 6 inventory contract did not cover enough value-route and
   executable-boundary roots;
2. the proposed historical-mathematics preservation check was too weak; and
3. the Phase 5 wording overstated construction identity by implying one
   mechanically shared trace rather than separate primal and manual-JVP
   traversals checked pointwise against forward autodiff.

## Repairs

- Expanded the inventory to public APIs, CLIs, dispatchers, artifact builders,
  factories, aggregates/consumers, exception/fallback boundaries, value and
  score routes, with zero unclassified hits required.
- Added exact protected numerical-kernel AST hashes and representative
  pre/post six-route bitwise diagnostic records.
- Corrected the Phase 5 result to state that the concrete graph packages a
  primal traversal and a separate manual-JVP traversal; the checked manual JVP
  matches forward autodiff of the private primal at `0 ULP`, but is not
  mechanically generated from that primal and is not immune to future drift.

## Follow-Up Verdict

The repaired Phase 5 result and Phase 6 handoff are consistent, feasible, and
preserve the scientific boundary.

`VERDICT: AGREE`

