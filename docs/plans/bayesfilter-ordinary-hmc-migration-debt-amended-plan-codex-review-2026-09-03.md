# Fresh Codex Review: Amended Ordinary HMC Migration Plan

Date: 2026-09-03

Reviewer: fresh Codex bounded red-team review

Reviewed paths: the amended migration plan, the amended Claude handoff,
`bayesfilter/inference/hmc_tuning_dispatch.py`,
`bayesfilter/hmc_route_contract.py`, and the named ordinary orchestration
blocks in `bayesfilter/inference/hmc_kernel_tuning.py`.

## Review Scope And Verdict

The review was read-only. It did not run HMC, tuning, benchmarks, GPU work, or
the focused pytest commands. It checked the amended branch trace, authority
semantics, static-phase boundaries, and the evidence contract. Source paths
outside the bounded list remain `not checked`.

Initial verdict: `VERDICT: REVISE`.

The following findings were material and were incorporated into the plan and
handoff before execution:

| Severity | Classification | Finding | Disposition |
|---|---|---|---|
| P1 | wrong relative to the stated target | The handoff route-contract path was written as `bayesfilter/inference/hmc_route_contract.py`; the implementation is `bayesfilter/hmc_route_contract.py`. | Corrected in the handoff source-review item. |
| P1 | unsupported relative to the stated target | `operational_authority=True` is paired with `evidence_role=engineering_only` and `promotion_role=stage_handoff_only` in `hmc_route_contract.py:157-200`; it cannot stand for public artifact or scientific authority. | Plan now requires separate `artifact_authority` and scientific/promotion roles and independent invariants. |
| P1 | wrong relative to the stated NumPy target | `hmc_kernel_tuning.py:35` imports NumPy and the ordinary windowed path uses `np.zeros`/`np.eye` near `:26794-26829`; the repository policy forbids this in admitted runtime paths. | Claim-bearing artifact authority remains disabled until bounded NumPy repair or a reviewed exception; engineering diagnostics remain explicitly non-admitting. |
| P2 | unsupported | Constant-string dynamic-import handling did not cover computed `importlib`, `getattr`, or entry-point indirection. | New `unknown_dynamic_import`/`unresolved` rows block claim-adjacent admission until manual classification. |
| P2 | unsupported | Static pytest/renderer commands were asserted to be non-HMC without source confirmation, and the external scan had no explicit scope/output bound. | Commands are conditional on source confirmation; Phase 0 now records exact paths, exclusions, provenance, and a versioned output root. |
| P2 | unsupported | Future method-comparison ledger omitted the required baseline ladder. | Phase 8 now requires naive, best tuned classical, plain proposed, and enhanced proposed rungs when applicable, or a justification for omission. |

## Source-Confirmed Positive Findings

The ordinary dispatcher branches by typed configuration in
`hmc_tuning_dispatch.py:29-84`. The ordinary default is constructed by
`_run_canonical_hmc_tuning` and uses the operational algorithm ID; the route
resolver accepts legacy IDs as supported but marks them non-promoting. The
public ordinary call checks support but, before repair, did not require the
returned authority state (`hmc_kernel_tuning.py:13280-13328`). The Phase 7
direct final payload accepts `direct_phase5_candidate` and
`operational_selection_v2` without an authority check (`:26654-26757`). Thus
the plan's legacy authority-boundary finding is correct as a static finding;
it does not assert a successful runtime or scientific result.

The default branch correction is also correct: an operational warm-up result
selects the operational fixed-trajectory screen; the explicit legacy mapping
is what reaches the alternate joint-grid branch. Direct stage tests are not
evidence that the public default takes that branch.

## Deferred Items

The TensorFlow-native payload details, selector replication implementation,
capability registry semantics, verification metric roles, downstream consumer
roles, stale-document contents, and dynamic-import call sites outside the
listed paths were `not checked`. They remain required sequential review items
and cannot be promoted from plan assertions to source facts until inspected.

## Execution Boundary

The static phase may proceed only for bounded construction, serialization,
AST/import classification, documentation wiring, and authority-guard work.
Numerical policy selection, XLA-default changes, NumPy migration beyond the
bounded guard, HMC/tuning runs, and promotion claims require a separate
reviewed evidence contract and owner decision. A passing construction or
serialization test is an engineering prerequisite, not numerical or
scientific evidence.

Final bounded-review verdict: `VERDICT: REVISE` (resolved by the amendments
recorded in the plan and handoff; this memo does not certify numerical work).

## Post-Review Adjudication

After this review, the plan received the remaining specification refinements
identified by the earlier Claude response and by the execution dry run:

* Phase 0 now gives an evidence-based consumer-role decision procedure and
  records exact NumPy import anchors for the seven ordinary-family modules.
* Phase 2 now requires an owner decision about whether the legacy joint-grid
  implementation is promoted, assigned a new policy identity, or retained as a
  diagnostic comparison before numerical work.
* Phase 3 now says that old payloads lacking resolved policy/authority fields
  load only as historical diagnostics and fail closed for authority replay.
* Phase 5 now states the dependency order for construction, replay, and docs
  tests, and the compatibility-delegate forwarding invariant has a regression
  test.
* The bounded scanner reports `unknown_dynamic_import` and
  `unresolved_dynamic_attribute` separately and fails closed on either class.
* The ordinary module header and Phase 5 docstring were corrected to stop
  calling the legacy joint grid promoted; a regression test now protects the
  operational-default versus legacy-diagnostic distinction.

These additions do not widen execution authority. The static phase remains
complete only for the bounded checks recorded in the execution note; consumer
manual classification, NumPy cleanup, XLA policy selection, and HMC runs remain
open or blocked.
