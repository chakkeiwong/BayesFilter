# Review of the MacroFinance typed retained bridge request

Date: 2026-09-14

The request is reasonable and its central diagnosis is correct. The proposed
validation rules need revision, and delivering three builders alone would not
make the current MacroFinance evaluator eligible for retained handoff.

## Scope and baseline

Reviewed the complete MacroFinance memo at
`/home/ubuntu/python/MacroFinance/docs/plans/bayesfilter_typed_candidate_retained_bridge_handoff_memo_2026_09_14.md`,
its cited consumer code and attempt07 result, and the relevant BayesFilter
controller, adapter, replay, and artifact boundaries. The memo pins
`b603363df95ce5eb18b28202544dafabbc58be56`. The current integration also contains
upstream `29c6d2f4` and the ordinary R-hat repair `b027982a`; distinguish these
baselines when discussing the NumPy blocker. This review does not implement
the new bridge or run MacroFinance's target or its reported 59-test suite.

## Findings

1. **Correct: the typed result has no executable retained handoff.**
   `HMCTypedCandidateSetAdapter` supplies observations and identity metadata,
   but no bound retained kernel, frozen numerical geometry, or verified final
   state. Its payload and `HMCTuningCandidateSetResult.payload()` explicitly
   deny numerical handoff authority. `replay_candidate()` returns a candidate
   record. Existing retained builders type-check the older ordinary or
   TensorFlow result types. A checked member ID plus a qualified executable
   binding is the appropriate missing public boundary. This is part of the
   unification plan's unfinished P2/P3/P5 work, not a completed implementation
   with only an omitted export.

   Sources: `hmc_candidate_set_adapters.py:46,102,150,189`;
   `hmc_candidate_set_tuning.py:559`; `hmc_tuning_dispatch.py:52`;
   `hmc_tensorflow_tuning.py:1904`; `hmc_kernel_tuning.py:8222,8518`;
   unification plan Sections 14–16. Inference module paths are under
   `bayesfilter/inference/`.

2. **Wrong as written: every repair ancestor must be verified.**
   A fixed-L family can have A fail directionally, B fail directionally, and C
   pass fresh verification. C is eligible even though A-to-B did not produce a
   verified member. Every ancestral link and directional receipt must be
   valid; the selected child's own incoming repair and fresh verification
   must be completed and passing. Do not require unsuccessful intermediate
   candidates to pass retroactively.

   A pure-controller debugging fixture using the existing unit-test scope
   settings (`L=3`, epsilon `0.25`, repair factor `2`, maximum two repairs)
   produced epsilon `1.0` with statuses `not_executed_with_reason` followed by
   `executed_and_verified`. Both live replay and `require_verified_member`
   accepted the final child. These are synthetic control-flow values, not
   calibrated numerical defaults. Sources: controller `_request_repair`,
   `_apply_observation`, `replay_candidate`; artifact validation at
   `hmc_candidate_set_artifacts.py:257`.

3. **Additional BayesFilter defect: inconclusive repaired children can be
   marked verified in memory.** The repair-specific branch of
   `_apply_observation` handles hard failures and directional results, then
   treats all remaining decisions as passing. That includes
   `inconclusive_evidence` and `inconclusive_conflict`. A debugging fixture
   reproduced a verified ID and successful live `replay_candidate`, followed
   by durable validation rejecting it with `verified candidate lacks a
   passing verification receipt`. Durable replay prevents this invalid
   handoff, but a new in-memory builder must not trust the verified ID alone.
   Repair this state transition and give live/durable bridge entry points the
   same full validation. Source: `hmc_candidate_set_tuning.py:1292` and
   `hmc_candidate_set_artifacts.py:249`.

4. **Missing from the memo: the current MacroFinance observation decision is
   insufficient.** `OperationalCandidateAdapter.observe` reports `passed`
   whenever its listed health checks pass and aggregate acceptance is finite.
   It does not compare acceptance with the declared band or construct the
   dependence-aware acceptance decision. The pure controller trusts this
   supplied decision; a synthetic `passed` observation with acceptance `0.99`
   is accepted even by the durable record validator. That is expected for a
   scheduling boundary that delegates numerical judgment, but it cannot
   qualify this consumer. The evaluator must derive acceptance, directional
   repair, inconclusive, and health outcomes from actual TF/TFP execution
   under the declared policy. Neither hashes nor a caller-supplied
   `qualification_status="qualified"` establish that property.

   The same adapter uses `use_xla=False`, returns no persisted verified final
   state, and uses NumPy in its decision path. The bridge therefore also
   needs numerical state capture and explicit backend/XLA qualification for
   the actual scope. Current `HMCCandidateSetScope` has no explicit XLA
   field. Sources: MacroFinance
   `daily_asset_midas_phase14_candidate_set_adapter.py:301,330,377`;
   BayesFilter `hmc_candidate_set_tuning.py:75,1175` and
   `hmc_candidate_set_adapters.py:150,189`.

5. **Correct for the pinned snapshot, stale for current main: the ordinary
   NumPy blocker.** Upstream migrated the ordinary numerical path and added
   `ordinary_tf_tfp_runtime_v1` to both configuration and resolved policy.
   Historical or mismatched backend identities still fail the claim-bearing
   replay guard. Preserve that distinction rather than reinstating a global
   historical blocker or silently upgrading old results. This independent
   ordinary-backend repair does not qualify the typed candidate-set route or
   MacroFinance's evaluator. Source: `hmc_kernel_tuning.py:8383` and
   `tests/test_hmc_backend_tf_repair.py`.

6. **Correct stop, overstated opening description.** Attempt07 records M0/M1
   complete, M2 blocked, zero new numerical seconds, and no candidate-set
   result or retained samples. The broad-grid route is wired but was not
   numerically exercised in this attempt. MacroFinance's capability scan
   deliberately returns `supported=False` and `exact_api_qualified=False`
   even if a matching function name appears; implementation and downstream
   qualification must therefore be delivered together. Sources: attempt07
   `result.json` and `scripts/daily_asset_midas_phase14_active_support.py:429`.

## Recommended implementation scope

Finish the numerical adapter and retained handoff portions of the existing
unification plan together. First fix repaired-child inconclusive handling and
use a qualified numerical observation evaluator that preserves the actual
geometry, execution settings, and candidate-specific final state. Then expose
one shared member/binding validation and replay implementation behind the
necessary public convenience builders. Keep mechanics execution, eligibility
for a declared retained analysis, and scientific conclusions separate.

Durable archives need validated export, reload, and continuation from the
predecessor's final active state. An in-memory adapter need not acquire an
arbitrary disk-file requirement if the same complete member/binding checks
can be made in memory. Frozen kernel parameters stay fixed; retained chunk
lengths, fresh random streams, and continuation states follow their own
declared execution contract. Ordinary tuning R-hat remains reporting-only;
posterior convergence requirements belong to the later retained analysis.

The decisive checks are a real deterministic-target TF/TFP tune-to-retained
round trip, exact selected-member/geometry/state replay, durable continuation,
and negative controls for acceptance failures, inconclusive repair children,
valid multi-repair descendants, changed scope or binding, and unsupported
backend/XLA settings. Controller callback fixtures establish scheduling
mechanics only. Update the guide and capability registry from those results,
then update and qualify MacroFinance's pinned integration.
