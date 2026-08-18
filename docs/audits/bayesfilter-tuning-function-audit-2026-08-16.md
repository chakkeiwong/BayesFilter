# BayesFilter Tuning Function Audit

Date: 2026-08-16
Repository: `/home/ubuntu/python/BayesFilter`
Scope: checked-out Python source, including the uncommitted
`hmc_robust_broad_grid.py` and related new files. Existing worktree changes were
preserved.

## Executive Verdict

The tuning surface is not one coherent implementation. It is a family of
overlapping orchestration layers around a shared HMC runner.

| Layer | Current evidence | Verdict |
|---|---|---|
| HMC execution | `hmc.py`, `native_tfp_hmc.py`, `neutra_hmc.py` | Reusable execution authorities; keep separate from tuning selection. |
| Geometry/mass/verification | `hmc_kernel_tuning.py` | Valuable but overgrown: 1.14 MB, 493 functions, mixing geometry, bootstrap, mass, selection, verification, artifacts, timeout, and compatibility. |
| Generic fixed grids | `generic_hmc_tuning.py`, `hmc_fixed_metric_grid_search.py`, `hmc_budget_ladder.py` | Duplicated candidate, repair, evidence, and artifact semantics. External consumers depend on them. |
| Candidate/selection evidence | `hmc_kernel_selection.py` | 118-definition sanitized candidate, handoff, repair, and selection contract layer. It is an active dependency of ordinary-HMC selection, not an optional historical helper. |
| Operational broad grid | `hmc_operational_broad_grid.py` | A policy/state machine and process-parallel route, not the same contract as the robust runner. |
| New robust broad grid | `hmc_robust_broad_grid.py` | Best current ordinary-HMC orchestrator candidate, but not canonical yet. It delegates into the monolith, defaults `use_xla=False`, fixes `L=(3,5,9,13,18,25)`, and uses short tuning/qualification budgets. |
| Fixed transport | `fixed_transport_hmc_tuning_tf.py` and grid-policy modules | Distinct transformed-coordinate tuner required by dsge_hmc; retain as the second public interface. |
| LEDH/capacity | `highdim/capacity_tuning.py`, `ledh_tuning_scope.py`, `ledh_tuning_registry.py` | Separate high-dimensional scope/selection domain; do not collapse into HMC epsilon/L tuning. |
| NeuTra training/search | `neutra_*` search/training modules | Training controls, not HMC kernel tuning; keep a separate training interface. |

The safest target is two public interfaces:

1. `tune_hmc_kernel(...)` for ordinary/fixed-mass HMC, with one typed request,
   one repository-issued artifact, and one fail-closed selection contract.
2. `tune_fixed_transport_hmc_kernel(...)` for a fixed transport coordinate,
   sharing evidence/artifact contracts but retaining transport-specific mechanics.

The robust broad grid, operational broad grid, fixed-metric grid, budget ladder,
generic orchestration, and stage functions should become internal
implementation or explicit diagnostic/history entry points. They cannot be
deleted until MacroFinance and dsge_hmc consumers and tests migrate.

## Audit Method and Counts

The core source inventory was generated with Python `ast` over 26 files selected by
name/content as tuning, grid, kernel, budget, capacity, broad-grid, scope,
registry, frozen, trajectory, or staged routes. It found **1,417 class/function
definitions**. This is a broad inventory, not a claim that every definition is
an active tuner or that it covers every adjacent execution authority.

The adjacent authority inventory is in
`docs/audits/bayesfilter-tuning-adjacent-authority-inventory-2026-08-16.md`.
It covers `hmc.py`, verification/convergence/warmup, mass construction,
fixed-transport mechanics/candidate discovery, and the value/score adapter
contract. Those modules add 482 definitions and must be reviewed before code
extraction, but are deliberately not folded into the core tuning count.

The direct dependency closure is larger than that first adjacent inventory.
The following 10 modules are also imported by core tuning routes and must be
included in extraction review: `bayesfilter/hmc_budget_contract.py`,
`bayesfilter/hmc_route_contract.py`,
`bayesfilter/inference/batched_value_score.py`,
`bayesfilter/inference/hmc_coordinates.py`,
`bayesfilter/inference/hmc_diagnostics.py`,
`bayesfilter/inference/hmc_posterior_diagnostics.py`,
`bayesfilter/inference/neutra_shared_procedure.py`,
`bayesfilter/runtime/__init__.py`,
`bayesfilter/inference/native_tfp_hmc.py`, and
`bayesfilter/inference/neutra_hmc.py`. Their fresh AST total is 271
definitions. The first adjacent inventory's 482 count remains a named
execution/evidence subset; it is not the complete dependency-closure count.

External usage was searched in all Python files outside `.git`, `.claude`, and
`__pycache__`:

| Consumer | Tuning-related files | Important direct contracts |
|---|---:|---|
| MacroFinance | 113 | `tune_hmc_kernel` (49 files), `tune_hmc_kernel_robust_broad_grid`, `run_fixed_metric_grid_search`, `run_operational_broad_grid`, `run_fixed_mass_hmc_tuning_budget_ladder`, and source/private imports of `hmc_kernel_tuning`. |
| dsge_hmc | 16 | `run_fixed_metric_grid_search`, `tune_fixed_transport_hmc_kernel`, `run_fixed_mass_hmc_tuning_budget_ladder`, private `hmc_kernel_tuning` helpers, and contract tests for BayesFilter entrypoints. |

Counts are search counts, not execution-frequency claims. The 49 MacroFinance
ordinary-HMC files use an explicit `--no-ignore` on-disk Python-file criterion;
the 113/16 broad "tuning-related file" totals were inherited from an earlier
union whose exact pattern was not recorded and are therefore unsupported as
precise counts. Use the reproducible symbol-specific commands in the Fable
handoff memo and report active files separately from archival snapshots under
`results/**/functional_source_snapshot`.

The repo-wide text-reference inventory is in
`docs/audits/bayesfilter-tuning-repo-wide-reference-inventory-2026-08-16.md`.
Its original scan reported 3,553 matching files, including 3,314 under `docs`,
but that exact pattern was not recorded and Fable could not reproduce it. Those
numbers are provisional and not evidence. Most matching files are historical
plans/results rather than APIs; route them as provenance/history unless an
active script reads them. Recompute the inventory with the explicit pattern
recorded in the Fable handoff before using any count for a migration decision.

## Function-by-Function Audit

### `hmc_robust_broad_grid.py`

| Definition | What it does | Classification |
|---|---|---|
| `_seed` | SHA-256 domain/L/index deterministic two-int seed derivation. | Internal deterministic helper. |
| `_finite_positive` | Positive finite scalar validation. | Internal validation helper. |
| `RobustBroadGridConfig` | Freezes seeds, fixed L grid, acceptance/repair bands, DA/repair/qualification budgets, target scope, execution mode, and XLA flag. | Candidate policy; numeric defaults need target-scope evidence. |
| `__post_init__` | Enforces exact grid, budget bounds, symmetric acceptance band, and mode/XLA relation. | Good fail-closed validation; not proof defaults are universal. |
| `payload` | Serializes policy and nonclaims. | Artifact helper; should use common artifact schema. |
| `_jsonable` | Converts mappings, tensors, arrays, scalar objects, and nonfinite values to JSON-safe values. | Duplicate helper; centralize. |
| `_trace_tensors` | Extracts/validates samples and standard traces, including finiteness and optional divergence. | Candidate evidence helper; share trace contract. |
| `_evidence` | Calls `evaluate_hmc_acceptance_evidence` with divergence provenance. | Evidence helper; canonicalize. |
| `_fixed_config` | Builds one fixed-kernel `FullChainHMCConfig`. | Internal config builder. |
| `_tune_one_l` | Runs independent TFP dual averaging for one L and returns final epsilon/trace. | Nomination stage only. |
| `_repair_one_l` | Repeats fixed-kernel screens and scales epsilon from typed acceptance evidence. | Repair stage; preserve fresh-seed lineage. |
| `_qualification` | Runs 500 draws, maps latent samples through adapters, computes acceptance and rank-normalized diagnostics. | Qualification only; not convergence evidence. |
| `_suitable` | Requires valid acceptance, no veto/cost stop, zero exposed divergences, finite diagnostics, max R-hat <= 1.05. | Selection gate; ESS minima of 1 are weak. |
| `select_robust_candidate` | Selects suitable row by minimum bulk ESS, then lower L, then signature. | Deterministic descriptive selection. |
| `tune_hmc_kernel_robust_broad_grid` | Runs geometry, bootstrap, windowed mass, per-L DA/repair/qualification, and emits a JSON-safe result. | Best candidate ordinary tuner; not canonical until repaired. |

### `hmc_kernel_tuning.py`

This file contains 32 classes and 493 functions. Its definitions group into:

| Family | Representative definitions | Function |
|---|---|---|
| Validation/policy | `_validate_*`, `_public_*`, `_trajectory_window_*`, `_phase7_*` | Normalize enums, seeds, bands, budgets, timeout, trajectory, and public/private boundaries. |
| Geometry | `HMCGeometryInitializationConfig/Result`, `initialize_hmc_kernel_geometry`, `_select_geometry_hint`, `_build_mass_artifact`, `_curvature_report` | Select/validate geometry and construct mass artifacts. |
| Bootstrap | `HMCBootstrapScreenConfig/Result`, `run_hmc_bootstrap_screen`, `_BootstrapFixedMassLatentValueScoreAdapter` | Initial fixed-mass health/acceptance screen and handoff. |
| Windowed mass | `HMCWindowedMassStageConfig/Result`, `run_hmc_windowed_mass_stage`, `_operational_windowed_mass_capture` | Warmup windows, covariance/mass updates, budget/telemetry. |
| Fixed-mass step | `HMCFixedMassStepStageConfig/Result`, `run_hmc_fixed_mass_step_stage` | Epsilon tuning at fixed mass. |
| Frozen trajectory | `HMCFrozenStepTrajectoryStageConfig/Result`, `run_hmc_frozen_step_trajectory_stage` | L/trajectory selection and repair. |
| Main loop | `HMCTuneVerifyRepairLoopConfig/Result`, `run_hmc_tune_verify_repair_loop`, `tune_hmc_kernel` | Orchestrate stages and emit artifacts. This is the ordinary-HMC public boundary. |
| Phase records | `_HMCPhase5*`, `_HMCPhase7*`, `_HMCPhaseAttemptState`, `_phase5_*`, `_phase7_*` | Private candidate/resume/verification bookkeeping. |
| Artifact projection | `_public_tuning_*`, `_write_private_*`, `_phase7_public_*` | Sanitize and write public/private progress/results. |
| Replay/admission | `build_retained_frozen_kernel_hmc_adapter_from_*`, `admitted_kernel_mechanics_payload_from_*` | Reconstruct and validate frozen kernels. |
| Budget/timeout | `_HMCAttemptBudgetPolicy`, `_staged_timeout_*`, `_phase7_*budget*` | Progress-aware budgets and closeout. |

The mechanics are reusable, but the file should be split. MacroFinance uses the
public `tune_hmc_kernel`; dsge_hmc also imports private `_mass_artifact_signature`
and bootstrap helpers, so extraction requires compatibility shims.

### `hmc_kernel_selection.py`

This 164 KB module contains 13 classes and 105 functions (118 AST
definitions). It defines the sanitized candidate/selection evidence layer:
`VerifiedFixedKernelHandoff` validates a repository-owned handoff;
`FixedTrajectoryCandidate`, replication, result, failure, and selection records
carry candidate lineage and evidence; `deterministic_candidate_order`,
`fixed_trajectory_candidate_values`, and paired seed helpers enforce stable
candidate identity; the bounded operational selection functions run candidate
replications, exact-L retuning, evidence extensions, and repair attempts; and
the payload/signature methods prevent fabricated or stale handoffs. It imports
NumPy and is currently migration debt for admitted runtime paths. Phase 2 must
assign this family explicitly, preferably alongside kernel stages/selection,
with a single canonical mass-artifact signature test.

### `hmc_fixed_metric_grid_search.py`

`FixedMetricSearchLineage`, `FixedMetricGridSearchConfig`, and
`FixedMetricGridExecutionConfig` bind target/adapter/seed/grid/execution
lineage. `FixedMetricTuneRequest/Outcome` and `FixedMetricScreenRequest/Outcome`
define injected callback contracts. Typed exceptions distinguish candidate
failure, shared invalidity, resource closeout, and target veto.
`FixedMetricScreenRecord`, `FixedMetricCandidateRecord`, evidence-policy,
aggregate, and confirmation records store per-candidate and replication
evidence. `refinement_l_values` creates local L refinements. `_run_tune`,
`_run_screen`, `_run_candidate`, and `run_fixed_metric_candidate` execute
callbacks; `aggregate_fixed_metric_candidate_evidence` and
`confirm_fixed_metric_candidate` classify evidence; `run_fixed_metric_grid_search`
coordinates serial/process execution. This is a diagnostic/compatibility route,
not a second canonical tuner.

### `hmc_operational_broad_grid.py`

`OperationalBroadGridPolicy` and execution/result classes define primary L
requests and same-epsilon neighbor guards. Pair/statistical epsilon evidence
classifiers distinguish viable, nominated, repair, and veto states.
`primary_requests` and `expand_same_epsilon_neighbor_guards` build coverage;
`assemble_operational_broad_grid_result` and
`select_operational_candidate_union` form deterministic results;
`run_operational_broad_grid` and its process-parallel variant execute callbacks.
This policy engine is not equivalent to the robust broad grid and remains a
diagnostic/compatibility route until MacroFinance migrates.

### `hmc_budget_ladder.py`

`run_fixed_mass_hmc_tuning_budget_ladder` validates a mass artifact, builds a
fixed-mass adapter, runs bounded tune/screen rounds with optional reusable
runners and progress monitors, classifies hard vetoes and directional repair,
and emits a ladder artifact. Its callback, timeout, telemetry, and repair
helpers duplicate other candidate loops. Keep the artifact reader and a
diagnostic wrapper; migrate active callers.

### `generic_hmc_tuning.py`

`GenericHMCTuningConfig` and `GenericHMCFixedGridScaleConfig` validate grids and
scale policies. `run_generic_hmc_tuning_orchestration` executes client-provided
candidate callbacks; `orchestrate_generic_hmc_tuning` adapts a target and writes
an artifact; scale-selection helpers classify pilot acceptance. This is a
generic callback layer, not a target-specific authority. Mark historical or
diagnostic and retain only as a migration shim.

### `fixed_transport_hmc_tuning_tf.py`

`tune_fixed_transport_hmc_kernel` builds a fixed-transport value/score adapter,
binds identity mass in transport coordinates, runs dual averaging/fixed-grid
candidates, verifies them, and emits a frozen kernel payload. Its helpers
`_candidate_attempts`, `_dual_averaging_candidate`, `_fixed_grid_attempts`,
`_run_verification`, `_select_candidate`, and `_chain_config` implement the
candidate route. This is the second legitimate public interface because its
coordinate and transport identity differ from ordinary HMC and dsge_hmc calls
it directly.

`fixed_transport_hmc_tuning.py` is a 553-byte compatibility placeholder with no
definitions; the TensorFlow implementation is authoritative.

### Other tuning-named modules

| Module | Function summary | Classification |
|---|---|---|
| `fixed_trajectory_hmc_tuning_v2.py` | Tiny Gaussian fixed-trajectory config/result, grid validation, candidate selection. | Reference/mechanics diagnostic. |
| `fixed_kernel_arm.py` | Fixed-kernel run, finite sample validation, minimum ESS/checkpoints. | Diagnostic primitive. |
| `fixed_transport_hmc_grid_policy.py` | Prepared transport grids, exact identities, adaptive scale/refinement, launch eligibility. | Transport policy/diagnostic. |
| `frozen_kernel_validation.py` | Frozen candidate/artifact/scope/observation records and replay validation. | Replay/admission validation. |
| `staged_fixed_kernel_hmc.py` | Staged fixed-kernel estimation and result payload. | Runtime estimation, not tuning API. |
| `hmc_tuning.py` | Policy/config/result classes, dual averaging, trajectory rules, acceptance helpers. | Shared mechanics; split policy/execution. |
| `hmc_tuning_artifacts.py` | Canonical JSON/hash/atomic-write helpers and kernel/start-bank/timeout summaries. | Shared artifact authority. |
| `hmc_tuning_state.py` | Typed state machine and repair aggregation. | Shared state/evidence authority. |
| `hmc_uncertainty_retuning.py` | Descriptive independent-chain spread and fresh-retuning nomination. | Diagnostic only; not convergence/promotion evidence. |
| `highdim/capacity_tuning.py` | Significant-place value comparisons, frozen-scope checks, capacity nomination. | Highdim diagnostic. |
| `highdim/ledh_tuning_scope.py` | Exact LEDH scope/hash/match contract, including chunk geometry. | Active LEDH contract. |
| `highdim/ledh_tuning_registry.py` | Route-specific tunable/fixed controls and active tuner status. | Active LEDH registry. |
| `highdim/rank_budget.py` | Rank/capacity budgets, ladders, nomination records. | Highdim policy/diagnostic. |

NeuTra broad-grid/training files (`neutra_broad_grid.py`,
`neutra_state_continuing_broad_grid.py`, curriculum/staged/training modules)
control learned transport training, not HMC epsilon/L tuning. They must remain a
separate batch-native GPU/XLA training interface and historical diagnostics.

## Cross-Repository Findings

### MacroFinance

MacroFinance's 113 tuning-related files reduce to three compatibility patterns:

1. Ordinary HMC calls to `tune_hmc_kernel`, including geometry repair, map
   tuning, retained-validation preparation, and repeated calls in
   `one_country_zlb_ns_estimation.py`.
2. The MIDAS robust broad-grid driver imports `RobustBroadGridConfig`,
   `stable_adapter_signature`, and `tune_hmc_kernel_robust_broad_grid`.
3. CCMA and two-currency scripts use `run_fixed_metric_grid_search` and
   `run_operational_broad_grid`; diagnostic runners inspect `hmc_kernel_tuning`
   source or import private helpers.

MacroFinance tests assert source-level call counts and forbidden imports. A
migration must update those tests while preserving their intent.

### dsge_hmc

The BGS stage-C grid imports fixed-metric request/lineage types and
`run_fixed_metric_grid_search`. The Rotemberg NeuTra smoke imports
`tune_fixed_transport_hmc_kernel`. Public-explicit-state scripts import
`_mass_artifact_signature` and private bootstrap helpers. Contract tests inspect
BayesFilter entrypoints, artifact schema, XLA flags, and no-local-sampler
conditions. These are hard compatibility constraints.

## Concrete Problems and Nonclaims

1. At least five ordinary-HMC orchestration routes duplicate candidate,
   repair, and evidence semantics.
2. The largest module mixes numerical mechanics, policy, timeout, progress,
   private serialization, and replay admission.
3. `inference/__init__.py` exports historical routes alongside public ones,
   making accidental use easy.
4. The robust route has useful staging but no centralized tuning scope,
   artifact schema, XLA/default policy, or uncertainty-aware ranking yet.
5. Private helper imports in dsge_hmc block direct deletion.
6. Legacy NumPy in runtime/reporting paths needs isolation under the repository
   TensorFlow/TFP policy.
7. Local BayesFilter tests do not cover all MacroFinance/dsge_hmc contracts.

Passing a tuning screen, including the robust broad grid, does not establish
posterior convergence, posterior correctness, sampler superiority, default or
production readiness, HMC scientific validity, or empirical model validity.

## Mechanical Inventory Totals

| Module | Bytes | Classes | Functions | Definitions |
|---|---:|---:|---:|---:|
| `fixed_kernel_arm.py` | 15,340 | 2 | 7 | 9 |
| `fixed_trajectory_hmc_tuning_v2.py` | 13,318 | 2 | 11 | 13 |
| `fixed_transport_hmc_grid_policy.py` | 83,442 | 8 | 58 | 66 |
| `fixed_transport_hmc_tuning.py` | 553 | 0 | 0 | 0 |
| `fixed_transport_hmc_tuning_tf.py` | 46,993 | 3 | 40 | 43 |
| `frozen_kernel_validation.py` | 16,554 | 6 | 22 | 28 |
| `generic_hmc_tuning.py` | 52,470 | 8 | 35 | 43 |
| `hmc_budget_ladder.py` | 124,458 | 5 | 73 | 78 |
| `hmc_fixed_metric_grid_search.py` | 71,710 | 22 | 56 | 78 |
| `staged_fixed_kernel_hmc.py` | 20,483 | 2 | 10 | 12 |
| `hmc_kernel_selection.py` | 164,060 | 13 | 105 | 118 |
| `hmc_kernel_tuning.py` | 1,144,371 | 32 | 493 | 525 |
| `hmc_operational_broad_grid.py` | 80,162 | 14 | 73 | 87 |
| `hmc_robust_broad_grid.py` | 27,253 | 1 | 15 | 16 |
| `hmc_tuning.py` | 89,859 | 13 | 73 | 86 |
| `hmc_tuning_artifacts.py` | 53,688 | 1 | 20 | 21 |
| `hmc_tuning_state.py` | 24,399 | 2 | 14 | 16 |
| `hmc_uncertainty_retuning.py` | 22,459 | 3 | 19 | 22 |
| `neutra_broad_grid.py` | 23,194 | 2 | 13 | 15 |
| `neutra_staged_training.py` | 31,410 | 9 | 20 | 29 |
| `neutra_state_continuing_broad_grid.py` | 26,755 | 2 | 12 | 14 |
| `highdim/capacity_tuning.py` | 14,141 | 1 | 8 | 9 |
| `highdim/ledh_tuning_registry.py` | 5,566 | 1 | 2 | 3 |
| `highdim/ledh_tuning_scope.py` | 2,871 | 1 | 5 | 6 |
| `highdim/rank_budget.py` | 31,382 | 6 | 25 | 31 |
| `highdim/zhao_cui_frozen_proposal_apf_tf.py` | 40,553 | 5 | 44 | 49 |
| **Total AST definitions** |  |  |  | **1,417** |

The total is the AST count from the checked-out files and must be regenerated
if files change.

The complete line-level generated list for the core surface is in
`docs/audits/bayesfilter-tuning-definition-inventory-2026-08-16.md`. It records
path, line, definition kind/name, and either the first docstring line or an AST
behavioral fingerprint for all 1,417 definitions; entries without docstrings
are explicitly marked for body-level review.

The selection rule is intentionally name-based and includes two known false
positive classes: `highdim/zhao_cui_frozen_proposal_apf_tf.py` matches the
`frozen` token but is an auxiliary particle-filter proposal module, not HMC
tuning; `bayesfilter/ssm/filter_registry.py` and the zero-definition
`bayesfilter/testing/ksc_gaussian_sum_ukf_scope.py` were excluded because they
do not define tuning controls or selection behavior. New NeuTra curriculum
files are classified under the separate learned-transport training domain,
not silently omitted: `neutra_curriculum_search.py`,
`neutra_curriculum_training.py`, `neutra_end_to_end.py`,
`neutra_shared_procedure.py`, and related testing fixtures require a separate
NeuTra inventory. `hmc_phase5_evidence_resume.py`, staged center exports,
`neutra_artifacts.py`, and tuning fixtures are adjacent historical/diagnostic
routes and are recorded in the dependency/reference inventories.

### Baseline Export Defects

Phase 0 found three missing inference-package exports that broke committed tests
or an active MacroFinance import: `HMCStagedTimeoutPolicy`,
`prepare_fixed_transport_hmc_adaptive_joint_grid_policy`, and
`prepare_fixed_transport_hmc_joint_grid_policy`. They were already present in
the top-level `bayesfilter` lazy map but absent from
`bayesfilter.inference.__all__`. The minimal export repair is now applied in
`bayesfilter/inference/__init__.py`; focused verification is separate from
broader refactor evidence.

## Limitations

This is a source and consumer audit, not a scientific validation of each
target. GPU HMC runs were not launched. MacroFinance and dsge_hmc tests were
inspected but not run in this documentation pass; the refactor plan requires
running them.
