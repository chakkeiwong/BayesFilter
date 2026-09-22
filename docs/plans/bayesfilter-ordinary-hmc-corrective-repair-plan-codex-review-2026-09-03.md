# Skeptical Review: Ordinary HMC Corrective Repair Plan

Date: 2026-09-03

Reviewer: fresh Codex bounded review

Reviewed path: `docs/plans/bayesfilter-ordinary-hmc-corrective-repair-plan-2026-09-03.md`,
with source cross-checks against the existing ordinary tuning and replay
helpers.

## Verdict

Initial verdict: `VERDICT: REVISE`.

The plan has the right scientific boundary and correctly treats the shared
epsilon route as a policy mismatch rather than evidence against HMC. It was
not yet executable without the following amendments.

| Severity | Classification | Finding | Required amendment |
| --- | --- | --- | --- |
| P1 | underspecified | The plan offered “a clearly named mechanics-only wrapper” and “a claim-bearing entry point” without naming the compatibility behavior. A partial rename could break existing downstream imports or leave the old ambiguous function in use. | Name the APIs and preserve a compatibility delegate whose returned role is explicit. Add forwarding and failure tests before changing consumers. |
| P1 | scope mismatch | The plan said to migrate callers in MacroFinance and `dsge_hmc`, but this checkout is the BayesFilter writable root. Editing those repositories is outside this bounded change. | Limit edits to BayesFilter. Produce a path-and-role inventory and handoff instructions for the other repositories; do not claim their migration is complete. |
| P1 | authority ambiguity | A mechanics payload can be valid for adapter construction while still being forbidden for posterior admission. A boolean `passed` or function name cannot express that distinction. | Require an explicit `replay_role`/`authority_status` in the payload and return contract. Provide a separate claim-bearing guard that fails on missing fields or blockers. |
| P2 | numerical-policy overreach | Adding an ordinary joint-grid ID could be read as selecting a numerical default even though no target-specific evidence or owner choice exists. | Add only descriptive IDs for the observed shared route and legacy diagnostic route in this pass; keep any future measured policy ID reserved and non-runnable. |
| P2 | scanner false closure | “No unknown claim-adjacent consumer” cannot be established by the existing scanner while computed imports and unresolved attributes remain. | Treat unknown rows as an explicit unresolved result and report them; do not make a clean scan an acceptance claim for this pass. |
| P2 | test-scope risk | The proposed pytest list contains TensorFlow tests whose source must be checked to ensure they use only fake runners. A test command that launches a real chain would violate the plan. | Add a preflight source check and run the smallest policy/replay tests first; record any skipped broader test and why. |
| P2 | source-state wording | The plan status said “proposed, then executed” before execution, and did not distinguish current dirty overlays from `HEAD`. | Set status to proposed until the execution note is written, then record exact `HEAD`, dirty paths, and touched paths. |

## Source-confirmed positives

* The current non-identity durable-replay regression already exercises a
  non-identity covariance, JSON round trip, mass-signature preservation, and
  no-runtime-invocation invariant. The plan correctly schedules preservation
  and documentation of that result instead of inventing a duplicate geometry
  repair.
* `_public_resolved_policy_payload` already exposes the NumPy blocker. The
  missing check is at the consumer boundary, not in the route-level resolver.
* The ordinary public facade already rejects the legacy algorithm for artifact
  authority. The corrective work should not weaken that guard while adding
  descriptive policy identity.
* The fixed-transport measured-grid changes are separate from ordinary HMC and
  should remain exploratory; the plan's separation is appropriate.

## Amendments applied before execution

The plan is amended with these concrete decisions:

1. Keep the existing low-level replay function as a compatibility
   **mechanics-only** delegate and stamp its result with
   `replay_role="mechanics_only"` and
   `claim_bearing_artifact_authority=False`.
2. Add a new repository-owned
   `build_claim_bearing_retained_frozen_kernel_hmc_adapter_from_tuning_result`
   entry point. It calls the compatibility reconstruction only after checking
   a resolved policy, an explicit true claim-bearing flag, and an empty blocker
   list. The current ordinary result must fail this guard because its NumPy
   blocker is still real.
3. Add the corresponding serialized-payload guard and an explicit
   `build_claim_bearing_retained_frozen_kernel_hmc_adapter_from_mechanics_payload`
   entry point. The existing mechanics builder remains mechanics-only.
4. Add descriptive policy IDs only; do not add a runnable ordinary joint-grid
   default or change `use_xla`/NumPy behavior.
5. Restrict source edits to BayesFilter. Classify external callers in the reply
   memo and provide exact migration actions for their maintainers.
6. Make the scanner's unresolved rows a reported residual blocker, not a
   falsely passing acceptance criterion.
7. Run the policy/replay unit tests first, then the broader static suite only
   after confirming its runners are fixtures; no command may initialize a GPU
   or execute a target/HMC transition.
8. Change the plan status to `proposed` and let the execution note establish
   completion status.

## Final review disposition

After these amendments, the plan is suitable for bounded static execution.
The remaining numerical, backend, external-consumer, and promotion questions
are material open items and must remain visible in the execution note and the
reply memo.  The review does not authorize numerical work.

Final bounded-review verdict: `PASS_FOR_BOUNDED_STATIC_EXECUTION`.
The initial `VERDICT: REVISE` findings were applied before execution; no
claim-bearing numerical route was introduced.

## Post-review authority hardening

During the final source audit, Codex found that a caller could copy a valid
resolved-policy shape and remove its serialized blockers before reaching the
new claim-bearing boundary.  The plan was tightened before closeout: the
boundary now recomputes the ordinary policy from the repository-owned
`config`/`tuning_config`, compares the embedded policy and blocker list, and
rejects a mismatch before adapter reconstruction.  Durable mechanics payloads
carry the source `tuning_config` under the existing mechanics hash.  The
regression test
`test_claim_bearing_replay_cannot_clear_repository_policy_in_serialized_fields`
records this invariant.  This hardening preserves the bounded verdict; it does
not clear the current NumPy or shared-epsilon blockers.

## Final amended-plan audit

The amended plan was re-read after implementation and checked against the
actual source changes and recorded commands. The audit passes for the bounded
static scope:

* **Baseline:** the current shared-epsilon ordinary route remains explicitly
  diagnostic; no weak acceptance screen or legacy grid was promoted.
* **Criteria:** the acceptance checks test policy identity, replay role,
  schema/hash consistency, caller classification, documentation freshness, and
  the existing fixed-mass mass-signature hard veto; they do not use acceptance,
  ESS, R-hat, or short-chain output as a promotion criterion.
* **Stop conditions:** no target evaluation, HMC transition, GPU initialization,
  numerical grid selection, or backend-policy promotion was attempted. The
  unresolved external dynamic-import/attribute rows remain visible blockers.
* **Source and environment:** the dirty `HEAD` and the separately pinned C4R
  snapshot are recorded; no historical result was relabeled for the overlay.
* **Evidence:** the focused suite passed 25 tests and the broader bounded suite
  passed 277 tests; renderer, route inventory, compilation, and diff checks
  passed. The migration scan is retained as an inventory, not an admission.

Final verdict: `PASS_FOR_BOUNDED_STATIC_EXECUTION`. Numerical promotion and
external-repository migration require new owner-scoped plans.
