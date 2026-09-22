# Ordinary HMC Corrective Repair Execution Note

Date: 2026-09-03

Repository: `/home/ubuntu/python/BayesFilter`

Plan: `docs/plans/bayesfilter-ordinary-hmc-corrective-repair-plan-2026-09-03.md`

Review: `docs/plans/bayesfilter-ordinary-hmc-corrective-repair-plan-codex-review-2026-09-03.md`

Status: bounded static repair and in-repo historical-caller quarantine complete;
numeric and external-consumer migration work remain open

## Scope and source state

The skeptical review initially returned `VERDICT: REVISE`. Its amendments were
applied before execution, and the final disposition is
`PASS_FOR_BOUNDED_STATIC_EXECUTION`. This execution used no HMC campaign, no
target evaluation, no GPU initialization, no package changes, and no external
repository edits.

The source checkout was at `HEAD`
`2323065da5348fbf3aaabbd712afc2a028ca81a4` with a dirty worktree. The dirty
tree contains unrelated Q20 and transport work in addition to the files listed
below. The older C4R result is bound to
`dc22cbfc1b2e5d1f112bead424542898b03b5911`; it was not relabeled as evidence
for the current tree. Any future numerical campaign must use a clean commit or
an explicitly copied source snapshot and a fresh output root.

## Changes executed

### Ordinary epsilon/L policy identity

`bayesfilter/inference/hmc_kernel_tuning.py` now exposes a pure policy resolver
and three descriptive identifiers:

* `ordinary_shared_epsilon_screen_v3`: the observed operational path, which
  screens candidate `L` values with one shared/frozen epsilon and retunes at
  the nominated `L` afterward;
* `ordinary_legacy_joint_l_epsilon_grid_v1`: the legacy per-`L` grid branch;
  and
* `ordinary_engineering_joint_l_epsilon_grid_v1`: the explicit engineering
  probe branch.

The latter two remain diagnostic or engineering-only and non-promoting. The
ordinary shared route is also blocked from claim-bearing replay because it did
not measure a joint epsilon/L pair. The policy is serialized in configuration,
loop, result, and final-kernel summaries. No numerical grid or default was
selected by this change.

### Replay authority roles

The old replay names remain available as compatibility mechanics-only paths and
now return explicit `replay_role="mechanics_only"` and
`claim_bearing_artifact_authority=False`.

New claim-bearing names are exported from both `bayesfilter.inference` and the
top-level package:

* `build_claim_bearing_retained_frozen_kernel_hmc_adapter_from_tuning_payload`;
* `build_claim_bearing_retained_frozen_kernel_hmc_adapter_from_tuning_result`;
* `build_claim_bearing_retained_frozen_kernel_hmc_adapter_from_mechanics_payload`.

They fail closed before reconstruction unless the repository-issued resolved
policy has an explicit empty blocker list and
`claim_bearing_artifact_authority=True`. Once those fields appear clear, the
guard recomputes the ordinary policy from the repository-owned `config` (or
durable `tuning_config`) and compares the embedded policy and blockers; a
caller-edited authority field cannot clear a repository blocker. The durable
mechanics builder requires the persisted role to be exactly `mechanics_only`;
the claim-bearing wrapper owns its private expected-role argument. Durable
mechanics extraction binds the role, authority status, blocker list, resolved
policy, and source `tuning_config` into the hashed mechanics mapping.

The two claim-adjacent retained replay calls in
`bayesfilter/inference/neutra_end_to_end.py` now use the claim-bearing APIs.
They therefore stop at the current policy blocker instead of silently turning a
mechanics result into retained authority. The separate typed TensorFlow archive
runner remains a mechanics screen with `artifact_authority=False`; it was not
reclassified by this pass.

### Documentation and tests

The maintained Markdown guide and LaTeX chapter now describe:

* shared/frozen epsilon screening versus genuinely measured joint selection;
* the distinction between route artifact authority and claim-bearing authority;
* mechanics-only and claim-bearing replay roles and exact imports;
* the exploratory status of fixed-transport measured-grid evidence; and
* the explicit backend and policy blockers that remain.

The Neutra synthetic durable-mechanics fixture was upgraded with the explicit
mechanics-only role fields. All 12 historical fixed-transport benchmark
callers found by the source trace now select
`legacy_directional_diagnostic_v1` with the historical
`acceptance_target_distance` selector; two other callers already declare the
measured policy and explicit step-size grid. The static caller contract checks
that no bare policy remains in `docs/benchmarks` and that these historical
callers cannot emit a verified handoff. No historical artifact was deleted.

### Post-review hardening

The final source audit identified a consistency gap in the first guard: a
serialized caller could remove the blocker list while retaining the surrounding
policy shape. The guard and durable mechanics schema were amended as described
above, and a regression now exercises the forged-clear case. This is an
authority-boundary repair only; it does not make the current ordinary route
claim-bearing.

## Verification evidence

All commands below were run with `CUDA_VISIBLE_DEVICES=-1` and
`TF_CPP_MIN_LOG_LEVEL=3` where TensorFlow was imported. The test doubles and
source-contract tests were inspected before the broader runs.

| Check | Result | Evidence role |
| --- | --- | --- |
| `tests/test_hmc_tuning_policy_replay_authority.py` plus `tests/test_hmc_tuning_documentation_contract.py` plus `tests/test_fixed_transport_historical_caller_policy.py` | `25 passed, 2 warnings` | policy, role, documentation, and in-repo caller contract |
| `tests/test_hmc_kernel_tuning_outer_loop.py` plus `tests/test_hmc_kernel_tuning_public_api.py` | `183 passed` | ordinary orchestration and durable replay regression |
| `tests/test_fixed_transport_hmc_tuning.py` plus `tests/test_hmc_tuning_policy_repair.py` | `47 passed` | fixed-transport and policy-repair mechanics |
| `tests/test_neutra_all_models_end_to_end_contract.py` | `47 passed` | claim-adjacent source contract and mechanics fixture |
| `python scripts/render_hmc_tuning_interface_docs.py --check` | passed | generated route-document freshness |
| `python scripts/inventory_hmc_tuning_routes.py --check` | passed; no unclassified or stale registry rows | executable route inventory |
| `python -m compileall -q bayesfilter/inference bayesfilter/hmc_route_contract.py` plus patched benchmark modules | passed | syntax/import compilation |
| `git diff --check` on touched paths | passed | patch integrity |
| `python scripts/audit_ordinary_hmc_migration_surface.py --downstream-root /home/ubuntu/python/MacroFinance --downstream-root /home/ubuntu/python/dsge_hmc` | completed; report written under ignored plan artifacts | bounded source inventory |

The current inventory reports 350 Python files scanned, 202 relevant consumer
rows, 32 unknown dynamic-import rows, 19 unresolved dynamic-attribute rows,
seven ordinary runtime candidate modules with NumPy imports, and zero
unqualified non-XLA findings. These are source findings, not numerical
evidence. The unknown and unresolved rows remain blockers for claim-adjacent
external admission. The combined broader in-repository suite reported
`277 passed, 6377 warnings`; the focused suite reported `25 passed, 2
warnings` as shown above.

The source-level review also confirmed that the fixed-mass Phase 5 joint-grid
path and frozen-trajectory Phase 6 stage bind the mass artifact signature before
candidate screening and emit a hard veto when the signature changes. The
operational Phase 5 path binds the same signature into its adapter and lineage,
and the Phase 7 verification boundary applies the corresponding mutation veto.
This is an engineering invariant, not evidence that the ordinary
shared-epsilon policy is a tuned pair.

## Decision and inference status

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Policy identity repaired | Executable resolver and serialized IDs agree with the observed branches | shared-epsilon and NumPy blockers remain active | Which target-specific ordinary policy should be promoted | Write a separate numerical policy plan and obtain its authorization | no claim that the shared route is a tuned pair |
| Replay boundary repaired | Mechanics and claim APIs have distinct roles; repository policy is recomputed before reconstruction | forged/missing policy fields are rejected; current NumPy/shared-epsilon blockers remain | External callers have not yet migrated | Owners classify and update downstream repositories | no posterior or scientific admission |
| Nonidentity durable replay retained | Existing JSON round-trip regression passes without runtime invocation | schema/hash/role mismatches veto replay | Broader consumers still need role review | Keep the regression and migrate callers | no covariance-quality claim |
| Fixed-transport wording repaired | Guide/chapter call the measured grid artifact-authoritative and exploratory | candidate health failures remain hard in that route | Short-run ranking precision and target-specific health | Define a target-specific evidence contract if stronger use is needed | no superiority, convergence, or default claim |

The hard veto evidence supports only the authority and schema boundaries above.
Observed test counts and inventory counts are descriptive engineering evidence;
they do not rank samplers or establish statistical performance. No stochastic
comparison was performed, so there is no statistically supported ranking.

## Remaining blockers and handoff

1. The ordinary runtime still imports NumPy in candidate modules, and the
   ordinary default still records the existing XLA-policy mismatch. Claim-
   bearing ordinary replay must remain blocked until those issues have their
   own reviewed repair and evidence.
2. The ordinary epsilon/L policy is classified but not promoted. A future plan
   must choose measured joint pairs, independent per-L adaptation, or a reviewed
   dynamic method, with provenance for every numeric choice and disjoint
   calibration/verification data.
3. MacroFinance and `dsge_hmc` callers were inventoried but not edited from
   this BayesFilter checkout. Their unknown dynamic imports and claim-adjacent
   raw-runner/public-tuner rows require owner-side classification and migration.
4. The 12 historical fixed-transport callers inside this BayesFilter checkout
   are now explicitly quarantined as legacy diagnostics. Any external copy of
   those callers must receive the same policy fields, or be migrated to a
   declared measured grid in its owning repository.
5. The dirty worktree must not be used as a scientific source snapshot. No
   commit, merge, push, HMC run, or promotion change was performed by this
   execution.

The companion reply memo gives the finding-by-finding response for the other
agent and lists the downstream paths that still need owner action.
