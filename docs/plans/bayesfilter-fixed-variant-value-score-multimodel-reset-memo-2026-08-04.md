# Reset memo: fixed-variant multimodel value/score campaign reboot

## Date
2026-08-04

## Context
The current task is a cross-model fixed-variant campaign comparing two methods
for **likelihood value** and **score** across five model families:

- LGSSM
- KSC SV
- actual SV
- predator-prey
- Austria SIR

The intended scientific order was clarified during this session and should not be
re-litigated on reboot:

1. **Method A** = original single-fitted frozen version (fit once, freeze, no
   interpolation).
2. **Method B** = tangent/interpolation extension, only after Method A is
   admitted for that model.

The main conceptual resolution of this pass was that the campaign should not
start from a source-native Zhao-Cui reimplementation, nor from a global tangent
extension. The baseline object to test first is the once-fitted frozen parent.

## Decision / policy
Future sessions should assume the following.

Anti-drift authority for the current execution block:
`docs/plans/bayesfilter-zhao-cui-fixed-variant-all-model-campaign-control-master-program-2026-08-06.md`

- The campaign is about **declared fixed-variant scalars and their scores**, not posterior correctness, exact physical likelihood, source faithfulness, HMC readiness, or production readiness.
- Method A is the primary cross-model baseline and must be implemented/reported for all five models before Method B is interpreted as anything more than a per-model empirical extension.
- Method A admission now requires a manual or analytical score route for the declared scalar; same-scalar autodiff agreement is diagnostic evidence only.
- Method B must be admitted **model by model**; missing Method B routes are to be reported explicitly as `blocked` or `missing`, never silently omitted.
- Austria SIR Method B may be runtime-analytical for the child scalar while still
  relying on empirical/offline tangent issuance. Do not promote that into an
  end-to-end analytical-derivative claim without new evidence.
- The Austria SIR Method B loader paths are currently the main execution blocker;
  do not let them block Method A completion across the full model set.

## What changed
- File: `docs/plans/bayesfilter-fixed-variant-value-score-multimodel-plan-2026-08-04.md`
  - Created the main campaign plan.
  - Froze the method order, scope, verification strategy, and skeptical audit.
- File: `docs/plans/bayesfilter-fixed-variant-value-score-multimodel-target-registry-2026-08-04.json`
  - Created the initial registry of model × method rows.
  - Recorded current Method A / Method B readiness and derivative backend labels.
- File: `docs/plans/bayesfilter-fixed-variant-value-score-multimodel-adapter-schema-2026-08-04.json`
  - Created the uniform result payload schema for the campaign.
- File: `scripts/run_fixed_variant_value_score_multimodel_20260804.py`
  - Created an initial cross-model runner that binds Method A adapters for all
    five models and attempts Austria SIR Method B via persisted artifacts.
  - Fixed two early script bugs:
    - wrong import location for `load_selected_t2_parameter_parent_compat`
    - tensor telemetry casting for int32 status tensors

## Bugs / blockers resolved
- Symptom: the initial runner imported
  `load_selected_t2_parameter_parent_compat` from the wrong module.
- Root cause: the function lives in
  `bayesfilter.highdim.zhao_cui_austria_sir_parameter_child_tf`, not in
  `...lane_b_t2_tf`.
- Resolution: updated the import in
  `scripts/run_fixed_variant_value_score_multimodel_20260804.py`.

- Symptom: the runner crashed while serializing LGSSM telemetry because int32
  status tensors were forced into float64 conversion.
- Root cause: `_tensor_list()` used `tf.convert_to_tensor(..., dtype=float64)` on
  int tensors.
- Resolution: changed `_tensor_list()` to cast after conversion.

## Verification already run

Anti-drift / Austria Method B recovery work completed after the original reboot:

- Created the campaign-control master program:
  `docs/plans/bayesfilter-zhao-cui-fixed-variant-all-model-campaign-control-master-program-2026-08-06.md`
- Reissued a fresh Austria T1 score/tangent artifact under current source closure:
  `docs/plans/artifacts/zhao-cui-austria-sir-lane-b-t1-score-20260806/pilot-01-selected-current-closure/`
  - result status: `PASS_T1_SCORE_PILOT_ARM`
  - score artifact identity:
    `e1a30968fa84fb46f53ddd31cdfb1fc9790ea156dd7d719f0100d48561a13e41`
  - child identity:
    `9b1188671eb1b0c0833d7bd5f714a7326521913ed6b94552414f00fb3a587bf1`
- Reissued a fresh Austria T1 training-JVP artifact under current source closure:
  `docs/plans/artifacts/zhao-cui-austria-sir-lane-b-t1-training-jvp-20260806/attempt-01-current-closure/`
  - result status: `PASS_T1_MATERIAL_TRAINING_REPLAY_AND_FD_TANGENT`
  - issuer identity:
    `cc8460bffd737bcf682434c8ff49c9c52ceb8af45ec81fba92a5afcb4d1556d0`
  - child identity:
    `5a006e8f55423cb08e6b3b1b08443c6ac8fb3af1c637ff48c20ed7941cae0603`
- Reissued a fresh Austria T2 training-JVP artifact under current source closure:
  `docs/plans/artifacts/zhao-cui-austria-sir-lane-b-t2-training-jvp-20260806/attempt-01-current-closure/`
  - result status: `PASS_T1_T2_MATERIAL_TRAINING_REPLAY_AND_FD_TANGENT`
  - issuer identity:
    `9b6dfaecdd311741facca0b31fb1e69c0accf82a79fd66ad76cc0481ca377313`
  - T2 child identity:
    `17e33778c558e62972eb5bfe342e297520ab1475b3722602aeab7827c60cf263`
- Verified fresh strict loads under current source for:
  - `load_t1_score_artifact(...)`
  - `load_t1_training_jvp_child(...)`
  - `load_selected_t2_parameter_parent_compat(...)`
  - `load_t2_training_jvp_child(...)`
- Updated the multimodel runner to point at the fresh Austria Method B paths.
- While rerunning the CPU-only multimodel campaign, discovered and repaired two
  unrelated stale actual-SV import breakages in:
  - `bayesfilter/testing/zhao_cui_actual_sv_neutra_target_tf.py`
  - `bayesfilter/testing/zhao_cui_actual_sv_neutra_target_analytic_tf.py`
  These were module-surface repairs needed just to import the existing actual-SV
  Method A adapter under the current tree; they do not alter the campaign target.

```bash
CUDA_VISIBLE_DEVICES=-1 python scripts/run_fixed_variant_value_score_multimodel_20260804.py
```

Observed:
- The script now reaches Method A execution logic and binds all five baseline
  adapters.
- Austria SIR Method B T1 is no longer stale-blocked and now loads/executed from
  the fresh current-source score artifact path.
- Austria SIR Method B T2 is no longer blocked by a missing historical JVP path;
  the fresh current-source T2 training-JVP artifact now loads and executes.
- The current machine-readable result therefore contains live Austria Method B T1
  and T2 rows, while LGSSM / KSC SV / actual SV / predator-prey remain honestly
  blocked for Method B because no persisted route exists yet.

Additional diagnosis already run:
```bash
CUDA_VISIBLE_DEVICES=-1 python - <<'PY'
import json, sys
from pathlib import Path
sys.path.insert(0, '.')
from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_t1_score_tf import t1_score_source_closure
manifest = json.loads(Path('docs/plans/artifacts/zhao-cui-austria-sir-lane-b-t1-score-20260731/attempt-02-gpu-xla-smoke-shuffle-repair/artifact/manifest.json').read_text())
stored = manifest['source_closure']
current = dict(t1_score_source_closure())
for key in sorted(set(stored) | set(current)):
    if stored.get(key) != current.get(key):
        print('STALE:', key)
PY
```

Observed:
- The stored T1 score artifact closure is stale relative to current source.
- The mismatched paths are:
  - `bayesfilter/highdim/zhao_cui_austria_sir_lane_b_t1_score_tf.py`
  - `docs/plans/bayesfilter-zhao-cui-austria-sir-lane-b-t1-score-plan-2026-07-31.md`
  - `scripts/run_zhao_cui_austria_sir_lane_b_t1_score.py`
- Therefore the historical Austria T1 Method B artifact cannot be loaded under
  strict closure checking in the current tree.

Further diagnosis already run:
```bash
python - <<'PY'
import json, glob
for path in sorted(glob.glob('docs/plans/artifacts/zhao-cui-austria-sir-lane-b-t1-score-20260731/**/result.json', recursive=True)):
    p=json.load(open(path))
    print(path, '::', p.get('status'))
PY
```

Observed:
- There is a T1 score artifact with status
  `PASS_LANE_B_T1_VALUE_AND_TOTAL_SCORE` at:
  `docs/plans/artifacts/zhao-cui-austria-sir-lane-b-t1-score-20260731/attempt-05-calibration/result.json`
- There are multiple pilot artifacts, but the loader currently points at the
  smaller `attempt-02-gpu-xla-smoke-shuffle-repair/artifact` path.

Further diagnosis already run:
```bash
python - <<'PY'
import json, glob
for path in sorted(glob.glob('docs/plans/artifacts/zhao-cui-austria-sir-lane-b-t2-20260731/**/result.json', recursive=True)):
    p=json.load(open(path))
    print(path, '::', p.get('status'))
PY
```

Observed:
- The T2 value baseline artifact exists and passed as
  `PASS_NEW_FIXED_VARIANT_T1_T2_VALUE_BASELINE` at:
  `docs/plans/artifacts/zhao-cui-austria-sir-lane-b-t2-20260731/attempt-13-selected-untouched-value-claim-xla-repair/result.json`
- However, the expected T2 training-JVP artifact directory for Method B was not
  found in the current worktree, so a strict T2 child replay route cannot yet be
  treated as available.

## Current policy
- Preserve the multimodel plan, registry, adapter schema, and runner as the campaign-control anchor, but interpret them under the stricter Method A manual/analytical-score rule.
- Method A admission now requires manual or analytical score provenance for the declared scalar.
- KSC SV should be rebound to the manual KSC UKF route rather than the autodiff-backed Gaussian-sum route.
- actual SV should be rebound to the analytic target adapter rather than the autodiff-backed batch TT adapter.
- Predator-prey, Austria SIR, and LGSSM remain Method A candidates because their active routes are manual/analytical under the current evidence.
- Austria Method B work remains historical context only for this reset.

## Known limitations / cautions
- The machine-readable multimodel result now has live Austria Method B rows, but
  their `same_scalar_check` fields are still replay-loader statuses (`NOT_RUN` in
  the campaign summary), not newly rerun held-out local-comparison diagnostics.
- The registry still likely underspecifies current Austria Method B freshness and
  should be refreshed if it is going to remain the long-lived authority.
- LGSSM Method B remains only a wrapper candidate, not an implemented persisted
  route.
- KSC SV, actual SV, and predator-prey still have no real Method B path in this
  campaign.
- The fresh Austria reissue artifacts were created under trusted GPU execution;
  preserve them and do not silently fall back to the stale 20260731/20260801
  paths in future runner edits.

## Suggested next steps
1. Decide whether the campaign should stop with Austria-only live Method B
   evidence and honest blocked rows elsewhere, or whether a follow-on phase
   should build a truly same-scalar persisted LGSSM Method B wrapper.
2. If LGSSM Method B is pursued, do **not** reuse the existing persisted
   frozen-transport / NeuTra artifacts as Method B rows, because they change the
   scalar via transport pullback plus log-Jacobian relative to Method A.
3. If stronger Austria Method B interpretation is needed, add the promised
   held-out local-comparison diagnostics rather than inferring them from issuer
   replay passes alone.

## Current policy