# NeuTra Shared Procedure Infrastructure Result

Date: 2026-08-03
Plan: `docs/plans/bayesfilter-neutra-shared-procedure-infrastructure-plan-2026-08-03.md`

## Decision table

| Field | Status |
| --- | --- |
| Decision | **Engineering objective passed, with one remaining governance/provenance defect.** The shared top-level NeuTra broad-grid infrastructure executed all three targets, emitted explicit procedure variants, preserved the reviewed generic and repaired route semantics, and exposed target/procedure mismatches that were previously implicit. |
| Primary criterion | **Passed.** New shared modules executed `operational_broad_grid_v1` and `state_continuing_epsilon_repair_v1`; focused contract tests all passed; KSC and LGSSM generic runs completed with explicit generic-variant artifacts; PP repaired run completed with explicit repaired-variant artifact and reproduced the historical next-round union. |
| Veto diagnostics | **No engineering veto fired.** No target/signature mismatches, no GPU memory-growth failures, no missing status-key failures, and no broad-grid barrier invalidity. |
| Main uncertainty | Whether LGSSM should eventually become a third explicit reviewed tuning variant (fixed-`L` step-size probe nomination) instead of being forced through the generic broad-grid route. |
| Next justified action | Record this infrastructure result, then fix the remaining manifest/plan-path provenance issue for PP-style runs and decide whether to add an explicit LGSSM tuning variant or keep LGSSM on a reviewed generic route study path. |
| Not concluded | No posterior, convergence, superiority, or default-readiness claim for any target. KSC/LGSSM `no_viable_pair` outcomes are evidence **about those targets under the generic route only**, not global evidence against NeuTra on those models. |

## What was built

### New shared infrastructure

- `bayesfilter/inference/neutra_shared_procedure.py`
  - single top-level frozen-transport NeuTra broad-grid procedure
  - explicit reviewed variants:
    - `operational_broad_grid_v1`
    - `state_continuing_epsilon_repair_v1`
  - shared variant metadata and centralized sequential-handoff parser
- `bayesfilter/inference/neutra_state_continuing_broad_grid.py`
  - PP repaired route extracted into shared inference code
  - preserves per-`L` dual averaging, bounded epsilon repair, state continuation, fresh final screens, and calibrated-parent guard probes
- `docs/benchmarks/run_neutra_shared_procedure_20260803.py`
  - single driver with explicit `--variant`
- `bayesfilter/inference/neutra_end_to_end.py`
  - frozen broad-grid wrapper now delegates to the shared procedure
  - sequential broad-grid handoff now uses the shared parser
- `bayesfilter/inference/neutra_broad_grid.py`
  - shared target-health helper for fixed screens
  - explicit required-status-key contract
  - variant metadata passthrough into private/public artifacts

### Tests added/updated

- new: `tests/test_neutra_shared_procedure.py`
- updated: `tests/test_neutra_all_models_end_to_end_contract.py`

## Verification summary

Focused verification all passed before execution:

- `tests/test_neutra_shared_procedure.py` — **12 passed**
- `tests/test_hmc_operational_broad_grid.py` — **21 passed**
- `tests/test_pp_ukf_state_continuing_epsilon_repair_driver.py` — **6 passed**
- `tests/test_neutra_all_models_end_to_end_contract.py` + `tests/test_ksc_gaussian_sum_ukf_neutra_target.py` — **50 passed**

Total focused contract coverage executed in this phase: **89 tests passed**.

## Execution summary by target

### 1. PP-UKF repaired shared variant

Run root:
`docs/plans/artifacts/bayesfilter-neutra-shared-procedure-20260803/pp-state-continuing-attempt01/PP-UKF/`

- Procedure variant: `state_continuing_epsilon_repair_v1`
- Required status keys:
  `status_code`, `valid_pre_regularized_score`, `floor_count_value`,
  `min_innovation_eigenvalue`, `innovation_condition_estimate`
- Wall time: **9874 s**
- Result: `BROAD_GRID_TUNING_VIABLE_PAIR_SET`
- Viable primaries: `L = 5, 9, 13, 18, 25`
- Viable guards: `L = 12, 14, 17, 19, 24`
- Next-round union:
  `(5, 9, 12, 13, 14, 17, 18, 19, 24, 25)`

This exactly matches the historically reviewed PP repaired route semantics encoded
in `tests/test_pp_ukf_state_continuing_epsilon_repair_driver.py` and the prior
PP follow-on campaign. The event log confirms the extracted mechanics executed:

- `state_continuing_adaptation_calibration`
- `state_continuing_epsilon_repair_calibration`
- `state_continuing_epsilon_tune`
- `state_continuing_primary_fresh_screen`
- `state_continuing_same_epsilon_neighbor_guard`

Interpretation:
- the extracted repaired route is faithful enough to reproduce the historical
  PP next-round set under the shared infrastructure;
- this is strong evidence that the PP repaired route has been successfully
  codified as shared inference code.

### 2. KSC gaussian-sum SV generic shared variant

Run root:
`docs/plans/artifacts/bayesfilter-neutra-shared-procedure-20260803/ksc-operational-attempt01/KSC-UKF-GAUSSIAN-SUM-T20/`

- Procedure variant: `operational_broad_grid_v1`
- Required status keys:
  `status_code`, `valid_pre_regularized_score`, `floor_count_value`,
  `min_innovation_eigenvalue`, `innovation_condition_estimate`
- Wall time: **295 s**
- Result: `BROAD_GRID_TUNING_NO_HANDOFF`
- Viable primaries: none
- Viable guards: none

Every primary `L` was classified `needs_higher_epsilon`, including `L=25`. No
repair loop was attempted because that is not part of the generic route.

Interpretation:
- the shared infrastructure worked;
- the explicit variant metadata makes the real issue unambiguous:
  **KSC was tested under the generic operational route, not the repaired PP
  route**;
- the result is evidence against `KSC + operational_broad_grid_v1`, not against
  KSC NeuTra/HMC in general.

### 3. LGSSM-EXACT generic shared variant

Run root:
`docs/plans/artifacts/bayesfilter-neutra-shared-procedure-20260803/lgssm-operational-attempt01/LGSSM-EXACT/`

- Procedure variant: `operational_broad_grid_v1`
- Required status keys: `status_code`, `valid_pre_regularized_score`
- Wall time: **2353 s**
- Result: `BROAD_GRID_TUNING_NO_HANDOFF`
- Viable primaries: none
- Viable guards: none

Every primary `L` was classified `needs_higher_epsilon`.

Interpretation:
- the shared infrastructure correctly supported a different telemetry contract;
- LGSSM historically succeeded under a different procedure (`run_f2_tuning_and_admission`
  in `bayesfilter/testing/lgssm_new_fixture_neutra_hmc_f2_tf.py`), namely a
  fixed-`L` step-size probe nomination path followed by shared sequential HMC;
- therefore the LGSSM `no_viable_pair` result is evidence against
  `LGSSM + operational_broad_grid_v1`, not evidence that LGSSM NeuTra/HMC is
  globally invalid.

## Audit findings

### Finding 1 — main engineering objective passed

The repository now has one shared top-level Python NeuTra broad-grid procedure
that prevents the exact regression seen in KSC:

- the selected procedure is explicit at execution time,
- the artifact names the procedure variant used,
- the sequential handoff parser is centralized,
- the PP repaired route is no longer benchmark-script-only.

That is the primary requested infrastructure outcome.

### Finding 2 — the infrastructure exposed two genuine procedure mismatches

The new explicit variants show:

- KSC was implicitly using the wrong route before;
- LGSSM also does not naturally belong to the generic operational broad-grid
  route if the historical successful procedure is the fixed-`L` probe route.

This is a feature, not a bug: the shared infrastructure is now forcing the
procedure choice to be explicit instead of hidden in benchmark scripts.

### Finding 3 — remaining provenance defect: PP plan path in the manifest

The PP shared run manifest records the generic registry `plan_path`
`docs/plans/bayesfilter-neutra-all-executable-models-end-to-end-python-plan-2026-07-18.md`
rather than a PP repaired-route-specific plan path. This is inherited from the
registry `CellSpec.plan_path`, not from the new shared procedure itself.

Why this matters:
- the result artifact correctly records the **procedure variant**, which is the
  most important new infrastructure control;
- but the plan-path field is still too generic for PP-style specialized runs.

Verdict:
- this is a **real governance/provenance defect** that should be repaired next,
  but it does not invalidate the engineering conclusion that the shared
  procedure variants executed correctly.

### Finding 4 — LGSSM likely needs a third explicit reviewed variant

The current shared variants are:

- generic operational broad-grid
- state-continuing epsilon repair

But the historical LGSSM F2 route is neither. It is better described as:

- fixed `L`
- step-size probe nomination grid
- shared sequential HMC admission/confirmation

Recommendation: either
1. add a third explicit reviewed tuning variant for that route, or
2. review and migrate LGSSM onto one of the first two shared variants under a
   dedicated comparison plan.

## Inference-status table

| Row | Status |
| --- | --- |
| Hard veto screen | none of the three runs failed for engineering invalidity; all fail-closed contracts held |
| Statistically supported ranking | none; broad-grid outputs are unranked viability evidence only |
| Descriptive-only differences | per-`L` acceptance means/intervals, tuned epsilons, wall times, and which pairs remained viable |
| Default readiness | not assessed |
| Next evidence needed | repair PP plan-path provenance; decide whether to add an LGSSM-specific reviewed variant; then rerun the target-appropriate shared variant for any claim-bearing campaign |

## Post-run red-team note

Strongest alternative explanation against the positive infrastructure verdict:
- the PP match could be superficial while subtle state-continuation semantics
  drifted. Counterevidence: the event-role sequence, next-round union, and
  repair-vs-no-repair contrast across targets all match the expected reviewed
  distinctions.

Strongest alternative explanation against the negative KSC/LGSSM generic results:
- the generic route may simply be the wrong procedure family for those targets.
  Current evidence supports that explanation more strongly than a claim that the
  shared infrastructure is wrong.
