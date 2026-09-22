# Ordinary HMC Corrective Repair Plan

Date: 2026-09-03

Repository: `/home/ubuntu/python/BayesFilter`

Status: executed; bounded static repair complete; numerical and external
migration work remain open

## Purpose

This plan responds to the ordinary-HMC findings in
`/home/ubuntu/python/MacroFinance/docs/plans/bayesfilter_ordinary_hmc_amended_tree_review_findings_handoff_memo_2026_09_03.md`.
The objective is to make the executable policy and replay roles unambiguous
for downstream agents, while preserving the distinction between engineering
mechanics and scientific or posterior authority.

The plan deliberately does not use a short acceptance screen, a construction
test, or a replay test as evidence of sampler correctness. Those checks answer
only the engineering questions they are designed to answer.

## Research intent ledger

| Field | Decision |
| --- | --- |
| Main question | Can a downstream reader identify the ordinary epsilon/L policy and the exact authority role of every replay API from source and serialized artifacts? |
| Mechanism under review | Ordinary `tune_hmc_kernel` with the operational fixed-trajectory selector, plus its legacy joint-grid branch and durable handoff helpers. |
| Expected failure mode | A reader treats shared-anchor-epsilon screening as joint epsilon/L tuning, or treats a mechanics-only result as claim-bearing retained-sampling authority. |
| Static promotion criterion | Policy identity, selection semantics, authority role, and replay status are present and internally consistent in code, payloads, guide, chapter, generated table, and tests. |
| Hard vetoes | Schema/hash mismatch, missing policy role, an unclassified replay consumer, a stale generated document, or a claim-bearing replay accepted while a blocker is present. |
| Explanatory diagnostics | Existing acceptance, R-hat, ESS, phase, movement, and harmonic-oscillator fixtures. They do not rank samplers or certify convergence. |
| Repair trigger | Any source or test showing that the shared-epsilon route can be serialized as claim-bearing, or that a mechanics payload is silently treated as posterior admission. |
| Continuation veto | None for this static pass. A numerical run requires a new target-specific evidence contract and explicit authorization. |
| Nonclaims | No conclusion about the MIDAS target, covariance quality, HMC validity, posterior convergence, superiority, default readiness, or scientific direction. |

## Assumption and default audit

| Choice | Provenance | Justification | Failure mode | Earliest check | Status |
| --- | --- | --- | --- | --- | --- |
| Operational ordinary route remains the default | `HMCKernelTuningConfig` and route resolver | Matches the current public dispatch and avoids an unreviewed default change | Agents may infer that operational means joint tuning | Resolve the route and inspect the selected stage | inherited, explicitly disclosed |
| Shared-anchor-epsilon screen is diagnostic-only | `hmc_kernel_selection.run_operational_fixed_trajectory_selection` | The selector receives one frozen epsilon for all candidate L values | A candidate can be rejected for an uncalibrated pair | Policy payload and negative handoff test | reviewed static disposition |
| Legacy joint grid is not promoted by this change | route contract and prior plan | Its numerical evidence and ownership decision remain open | A caller may use the legacy ID as an authority shortcut | Route guard and legacy construction test | diagnostic comparison only |
| Mechanics replay remains available | existing MacroFinance post-check workflow | It is needed to verify mass/step lineage without launching sampling | A vague API name can be mistaken for retained authority | Explicit role field and consumer-role test | bounded compatibility |
| Claim-bearing replay requires no policy blocker | this plan | A retained consumer must not bypass backend/policy gates | A future blocker could be ignored at replay | Guard test with the exact NumPy blocker | hard invariant |
| No numerical defaults are selected here | AGENTS.md evidence policy | Numeric grids and budgets need target-specific evidence | Convenience values could harden into a default | Plan scope check and no-HMC execution note | non-scope |

## Scope

### Included

1. Add repository-owned ordinary policy identifiers and a serialized policy
   description that states whether epsilon is shared across L or measured per
   `(epsilon, L)`.
2. Mark the current shared-epsilon route and the legacy joint-grid route as
   non-claim-bearing until a separately reviewed numerical policy is adopted.
3. Add an explicit mechanics-only replay boundary and an explicit
   claim-bearing replay guard. The guard must reject missing policy fields and
   `ordinary_runtime_numpy_policy_pending` (as well as any future blocker).
4. Include the resolved policy and role in durable mechanics payloads and bind
   them through the existing mechanics hash.
5. Add regression tests for policy serialization, blocked claim replay,
   mechanics-only replay, compatibility delegates, and non-identity geometry.
6. Update the Markdown guide, maintained LaTeX chapter, generated route table,
   examples, and route-contract tests so they describe the same executable
   policy.
7. Inventory the six historical fixed-transport callers named in the review
   and the additional bare fixed-transport configurations found during the
   source trace. Every BayesFilter production or benchmark caller must either
   declare the measured policy with an explicit pair grid or explicitly select
   `legacy_directional_diagnostic_v1` and emit diagnostic-only metadata. Unit
   tests, examples, and explicitly diagnostic fixtures may retain dataclass
   defaults when their role is documented. Write exact migration instructions
   for external maintainers. Because this checkout is the BayesFilter writable
   root, no MacroFinance or `dsge_hmc` files are edited or declared migrated
   here.
8. Record the exact dirty-tree/source-state limitation and the commands run in
   an execution note and reply memo.

### Excluded

* HMC transitions, tuning ladders, target evaluations, GPU initialization, or
  numerical policy selection.
* Changing `use_xla=False` to `True` or completing the NumPy migration. Those
  are separate source and evidence projects; the blocker remains visible.
* Promoting the legacy joint grid or changing the ordinary default.
* Deleting historical callers, plans, or artifacts.
* Git commit, merge, or push in this turn.

## Implementation phases

### Phase 0: Freeze and classify source

1. Re-read the current route resolver, public dispatcher, ordinary selector,
   replay helpers, fixed-transport config, and downstream callers.
2. Record whether each anchor comes from `HEAD` or an uncommitted overlay.
3. Run the bounded AST/import inventory over BayesFilter and the two named
   downstream repositories. Distinguish direct public tuners, mechanics
   replay, retained/posterior consumers, dynamic imports, and unresolved
   attributes. A scan result with unknown dynamic imports is not an admission.
4. Preserve the existing C4R result as evidence for its pinned snapshot only.

5. Classify each consumer with an explicit decision rule: a call that emits or
   loads a tuning artifact for retained/posterior use is claim-adjacent; a
   method-comparison artifact is a candidate route; a raw mechanics call with
   explicit nonclaims is mechanics-only; a fixture, smoke, reference, or dated
   archive is diagnostic/historical. If a file has more than one role or the
   role cannot be decided statically, classify it as claim-adjacent pending
   manual review. Record the exact role and the evidence for the decision.
6. Record the ordinary NumPy import inventory with module/line anchors and
   distinguish public-tuner reachability from diagnostic-only imports. Record
   the scanner's dynamic-import and unresolved-attribute rows separately; an
   unresolved row is never treated as a clean admission result.

Deliverable: a source-state paragraph in the execution note and a consumer
role table in this plan's reply memo.

### Phase 1: Make policy identity executable

1. Add stable IDs for the operational shared-epsilon screen and the legacy
   per-L joint-grid diagnostic branch. IDs must describe behavior, not imply
   promotion.
2. Add a pure, side-effect-free resolver used by payload construction and
   tests. It must report:
   * epsilon/L treatment;
   * candidate construction rule;
   * mass-signature freeze requirement;
   * seed-separation requirement; and
   * `claim_bearing_artifact_authority` plus a list of blockers.
3. Include the resolved policy in `HMCKernelTuningConfig.payload`, public
   result payloads, Phase 7 summaries, and final-kernel handoffs.
4. Ensure the old shared-epsilon route cannot claim to have selected a tuned
   pair. If the legacy branch remains callable for fixtures, its payload must
   say diagnostic/non-promoting.

### Phase 2: Repair replay authority boundaries

1. Keep the existing in-memory reconstruction path for mechanics validation;
   do not reinitialize non-identity geometry from missing caller hints.
2. Keep `build_retained_frozen_kernel_hmc_adapter_from_tuning_result` and
   `build_retained_frozen_kernel_hmc_adapter_from_tuning_payload` as backward-
   compatible mechanics-only delegates. Their returned contract and durable
   payload must carry `replay_role="mechanics_only"` and
   `claim_bearing_artifact_authority=False`.
3. Add
   `build_claim_bearing_retained_frozen_kernel_hmc_adapter_from_tuning_result`
   and its serialized-payload counterpart. Each requires a resolved policy,
   `claim_bearing_artifact_authority=True`, and an empty blocker list before
   any adapter reconstruction. The guard recomputes the ordinary policy from
   the repository-owned serialized configuration and compares the embedded
   policy and blockers; caller-edited authority fields cannot clear a blocker.
   Missing fields fail closed as historical diagnostics. The current ordinary
   route is expected to fail this guard until the separate NumPy/backend policy
   repair is completed.
4. Add
   `build_claim_bearing_retained_frozen_kernel_hmc_adapter_from_mechanics_payload`
   for a future claim-bearing mechanics artifact. The existing
   `build_retained_frozen_kernel_hmc_adapter_from_mechanics_payload` remains
   explicitly mechanics-only and rejects a payload whose role is not exactly
   `mechanics_only`.
5. Ensure the durable mechanics extractor uses the mechanics-only path and
   binds the role, resolved policy, and source `tuning_config` into
   `mechanics_sha256`.
6. Update only BayesFilter claim-adjacent internal consumers to call the new
   claim-bearing path; leave external consumers as handoff work items.
7. Treat payloads that predate `resolved_policy`, replay role, or the current
   schema as historical/diagnostic-only on load. They remain readable for
   provenance, but any authority-producing replay must reject them and require
   fresh tuning under the current schema.

### Phase 3: Migrate and classify callers

1. Inspect every `build_retained_*`, `admitted_kernel_mechanics_*`, direct
   `run_full_chain_tfp_hmc`, and low-level tuner import in BayesFilter, and
   inventory (without editing) the corresponding MacroFinance and `dsge_hmc`
   callers.
2. For each fixed-transport caller named by the review, and every additional
   bare `FixedTransportHMCKernelTuningConfig` call found by the trace, record
   whether it is historical/diagnostic or needs a declared measured
   `(epsilon, L)` grid. Historical BayesFilter callers are patched to select
   `legacy_directional_diagnostic_v1`, set the historical acceptance-distance
   selector explicitly, and carry diagnostic-only status fields. Active
   callers are not given invented numeric grids; they fail or remain blocked
   until their owner supplies a reviewed measured grid. External migration is
   incomplete until its own repository records the change.
3. Resolve what can be resolved in the bounded scanner and emit separate
   `unknown_dynamic_import` and `unresolved_dynamic_attribute` rows for the
   rest. Unknown rows remain hard blockers for claim-adjacent admission; this
   pass does not claim a clean external scan.
4. Run construction/import checks before replay or documentation tests. A
   static caller test must prove that no bare measured-policy configuration
   remains in the traced historical set and that every legacy caller exposes a
   diagnostic role. This ordering prevents a stale constructor contract from
   being hidden by a later artifact test.

5. Recheck every claim-bearing authority field against repository-owned policy
   inputs at the replay boundary. A serialized payload that merely changes
   `claim_bearing_blockers` or `claim_bearing_artifact_authority` is a failed
   authority check, even when its mechanics hash is internally consistent.

### Phase 4: Synchronize documentation and tests

1. Rewrite the ordinary guide section to say “shared/frozen epsilon screen,
   then exact-L retune,” and reserve “joint tuning” for an actually measured
   pair policy.
2. Add a geometry-precedence and covariance-first handoff section. State that
   the tuner validates caller hints; it does not manufacture a scientifically
   justified covariance.
3. Explain the two replay roles and show the exact claim-bearing guard.
4. Update the maintained chapter, generated route table, examples, and tests
   from the same policy payload.
5. Keep fixed-transport wording as engineering-artifact authority, not
   posterior or scientific authority; retain its exploratory evidence limits.
6. Add a stale-policy scan for active guides and generated tables: phrases that
   describe non-XLA or legacy directional behavior as a default must either be
   removed or carry an explicit historical/non-default classification.

### Phase 5: Static verification and handoff

First inspect the named test files and verify that their chain runners are
fixtures. Then run only bounded checks:

```text
python -m pytest -q \
  tests/test_hmc_tuning_policy_replay_authority.py \
  tests/test_fixed_transport_historical_caller_policy.py \
  tests/test_hmc_tuning_documentation_contract.py \
  tests/test_hmc_kernel_tuning_outer_loop.py \
  tests/test_hmc_kernel_tuning_public_api.py \
  tests/test_fixed_transport_hmc_tuning.py \
  tests/test_hmc_tuning_policy_repair.py
python scripts/render_hmc_tuning_interface_docs.py --check
python scripts/inventory_hmc_tuning_routes.py --check
python -m compileall -q bayesfilter/inference bayesfilter/hmc_route_contract.py
git diff --check
```

If a command imports a target or starts a chain, stop and classify it as an
invalid command for this plan. Record the actual command and environment in
the execution note. A full test failure caused by unrelated dirty-tree work is
reported with the exact failing path; unrelated files are not reverted.

## Acceptance matrix

| Check | Required result | Evidence role |
| --- | --- | --- |
| Policy resolver unit tests | Shared route says shared epsilon and non-claiming; legacy route says per-L diagnostic | Engineering contract |
| Claim-bearing replay with NumPy blocker | Fails before adapter reconstruction | Hard authority veto |
| Mechanics replay with blocker | Succeeds only with explicit mechanics-only status and all nonclaims | Engineering mechanics |
| Non-identity durable replay | Preserves geometry and mass signatures without target/HMC calls | Engineering regression |
| Serialized payload tamper | Hash, role, or repository-policy mismatch fails closed | Artifact integrity and authority |
| Caller inventory | Every bare BayesFilter fixed-transport config has an explicit measured or legacy policy; external unknown rows explicitly reported as residual blockers | Migration prerequisite |
| Guide/chapter/generated table | Renderer and semantic tests pass | Documentation consistency |
| Numerical run | Must not occur in this plan | Scope boundary |
| Artifact/construction smoke | May establish only schema and mechanics invariants, never sampler promotion | Engineering prerequisite |

## Future numerical plan requirements

Before any ordinary policy is promoted, write a new target-specific experiment
plan with a baseline ladder (naive, best tuned classical, plain proposed, and
enhanced proposed where applicable), disjoint calibration/verification data,
multiple seeds or a justified uncertainty design, explicit health vetoes,
MCSE/precision criteria, compute budget, source snapshot, and a pre-mortem.
The future plan must choose between a measured joint grid, independent per-L
adaptation, or a reviewed dynamic trajectory method. Reusing the current
shared-epsilon result as a tuned-pair baseline is prohibited.

## Planned artifacts

* This plan: `docs/plans/bayesfilter-ordinary-hmc-corrective-repair-plan-2026-09-03.md`
* Fresh skeptical review:
  `docs/plans/bayesfilter-ordinary-hmc-corrective-repair-plan-codex-review-2026-09-03.md`
* Static execution note:
  `docs/plans/bayesfilter-ordinary-hmc-corrective-repair-execution-2026-09-03.md`
* Reply to the reviewing agent:
  `docs/plans/bayesfilter-ordinary-hmc-corrective-repair-reply-to-other-agent-2026-09-03.md`
