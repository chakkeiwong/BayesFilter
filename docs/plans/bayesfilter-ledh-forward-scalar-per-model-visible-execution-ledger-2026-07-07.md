# LEDH Forward Scalar Per-Model Visible Execution Ledger

Date: 2026-07-07

Status: `DRAFT_LAUNCH_PACKAGE_CREATED`

Master program:

- `docs/plans/bayesfilter-ledh-forward-scalar-per-model-master-program-2026-07-07.md`

Runbook:

- `docs/plans/bayesfilter-ledh-forward-scalar-per-model-visible-gated-execution-runbook-2026-07-07.md`

## Ledger

### 2026-07-07 - Launch Package - PRECHECK

Evidence contract:

- Question: Can each intended high-dimensional LEDH row produce an executable
  same-target observed-data log likelihood estimator artifact?
- Baseline/comparator: July 6 Phase 3 admitted/blocked result, July 6
  per-model amendment plan, current forward contract metadata, row datasets,
  and row-specific reference checks where available.
- Primary criterion: a row is value-admitted only when a validated executable
  artifact reports finite `log_likelihood` values from the row target
  correction at the required row scale.
- Veto diagnostics: metadata-only admission, callback-only admission,
  proposal/flow objective used as likelihood, wrong row target, actual-SV/KSC
  artifact borrowing, score implementation before scalar admission, or
  runtime/memory/finite output promoted as correctness.
- Nonclaims: no score correctness, score admission, HMC readiness, posterior
  correctness, scientific superiority, or fair runtime ranking.

Actions:

- Created draft master program, visible runbook, stop handoff, Phase 0 subplan,
  and launch review bundle.

Artifacts:

- `docs/plans/bayesfilter-ledh-forward-scalar-per-model-master-program-2026-07-07.md`
- `docs/plans/bayesfilter-ledh-forward-scalar-per-model-visible-gated-execution-runbook-2026-07-07.md`
- `docs/plans/bayesfilter-ledh-forward-scalar-per-model-visible-stop-handoff-2026-07-07.md`
- `docs/plans/bayesfilter-ledh-forward-scalar-per-model-phase0-baseline-guard-subplan-2026-07-07.md`
- `docs/reviews/bayesfilter-ledh-forward-scalar-per-model-launch-review-bundle-2026-07-07.md`

Gate status:

- `IN_PROGRESS_PENDING_LOCAL_CHECKS_AND_REVIEW`

Next action:

- Run launch package local checks, then bounded read-only review.

### 2026-07-07 - Launch Package - REVIEW_REPAIR_LOOP_1

Evidence contract:

- Question: Does the launch package prevent another bundled Phase 3 failure by
  making same-target scalar admission executable, row-specific, and
  forward-scalar-only?
- Primary criterion: review must agree that stop conditions cover the known
  failure modes before Phase 0 launches.
- Veto diagnostics: missing stop condition for callback-only evidence or
  actual-SV/KSC artifact borrowing.
- Nonclaims: no Phase 0 execution, model implementation, value admission, or
  score admission yet.

Actions:

- Ran bounded Claude read-only launch review.
- Review returned `VERDICT: REVISE`.
- Finding: high-level evidence contracts listed callback-only admission and
  actual-SV/KSC artifact borrowing as vetoes, but the concrete Phase 0 and stop
  handoff stop conditions omitted them.
- Patched Phase 0 stop conditions and visible stop handoff to make those
  blockers explicit.

Artifacts:

- Review run dir:
  `/home/chakwong/BayesFilter/.claude_reviews/20260707-022825-ledh-forward-scalar-per-model-launch`
- Patched:
  `docs/plans/bayesfilter-ledh-forward-scalar-per-model-phase0-baseline-guard-subplan-2026-07-07.md`
- Patched:
  `docs/plans/bayesfilter-ledh-forward-scalar-per-model-visible-stop-handoff-2026-07-07.md`

Gate status:

- `IN_PROGRESS_PENDING_REPAIR_CHECKS_AND_REVIEW`

Next action:

- Rerun launch package local checks and a focused read-only repair review.

### 2026-07-07 - Launch Package - REVIEW_REPAIR_ACCEPTED

Evidence contract:

- Question: Did the focused repair close the launch blocker?
- Primary criterion: review agrees the concrete stop surfaces now include
  callback-only evidence and actual-SV/KSC cross-use blockers.
- Nonclaims: no Phase 0 execution, model implementation, value admission, or
  score admission yet.

Actions:

- Ran focused local repair checks.
- Ran bounded Claude read-only repair review.

Artifacts:

- Review run dir:
  `/home/chakwong/BayesFilter/.claude_reviews/20260707-023901-ledh-forward-scalar-per-model-launch-repair1`
- Review status: `REVIEW_STATUS=agreed`, `VERDICT=AGREE`.

Gate status:

- `PASSED_LAUNCH_GATE_PHASE0_MAY_START`

Next action:

- Launch Phase 0 only.

### 2026-07-07 - Phase 0 - BASELINE_GUARD_PASSED

Evidence contract:

- Question: Does the current repo state distinguish metadata-only forward
  contracts from executable same-target scalar admission, and what is the exact
  admitted/blocked baseline?
- Target scalar: `observed_data_log_likelihood_estimator`, reported as
  `log_likelihood`.
- Primary criterion: local focused checks pass and Phase 0 records exactly two
  value-admitted rows and four value-blocked rows.
- Veto diagnostics: metadata-only row promotion, blocked-row promotion, score
  work, or target redefinition.
- Nonclaims: no new value admission, score admission, score correctness, GPU
  evidence, leaderboard rebuild, HMC readiness, posterior correctness,
  scientific superiority, or runtime ranking.

Actions:

- Ran focused local Phase 0 checks.
- Wrote Phase 0 result.
- Drafted Phase 1 shared runner schema subplan.
- Patched the Phase 0/Phase 1 review bundle so the bounded packet states the
  target scalar explicitly.
- Patched Phase 1 nonclaims to include score correctness explicitly.
- Ran bounded Claude read-only review of the Phase 0 result and Phase 1
  subplan handoff.

Artifacts:

- `docs/plans/bayesfilter-ledh-forward-scalar-per-model-phase0-baseline-guard-result-2026-07-07.md`
- `docs/plans/bayesfilter-ledh-forward-scalar-per-model-phase1-runner-schema-subplan-2026-07-07.md`
- `docs/reviews/bayesfilter-ledh-forward-scalar-per-model-phase0-result-phase1-subplan-review-bundle-2026-07-07.md`

Local checks:

```text
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest \
  tests/highdim/test_ledh_phase3_forward_admission.py \
  tests/highdim/test_ledh_forward_contract_phase2.py -q
```

Result:

```text
12 passed, 2 warnings in 2.74s
```

Review:

- Run dir:
  `/home/chakwong/BayesFilter/.claude_reviews/20260707-025112-ledh-forward-scalar-per-model-phase0-phase1-handoff`
- `REVIEW_STATUS=agreed`, `VERDICT=AGREE`.

Gate status:

- `PASSED_PHASE0_PHASE1_HANDOFF_PHASE1_MAY_START`

Next action:

- Execute Phase 1 shared executable artifact schema guard.

### 2026-07-07 - Phase 1 - SCHEMA_GUARD_PASSED

Evidence contract:

- Question: Is there a shared executable artifact schema/validator that
  prevents metadata-only, callback-only, wrong-target, and actual-SV/KSC
  cross-use admission?
- Target scalar: `observed_data_log_likelihood_estimator`, reported as
  `log_likelihood`.
- Primary criterion: local validator/tests reject artifacts without executable
  `log_likelihood` evidence and reject target/flow ambiguity before row
  admission.
- Veto diagnostics: metadata-only contract passes; callback-only evidence
  passes; proposal scalar passes; target density correction missing; tiny
  artifact is admitted; actual-SV/KSC cross-use passes; theta mismatch passes.
- Nonclaims: no model row admission, score admission, score correctness, GPU
  evidence, leaderboard rebuild, HMC readiness, posterior correctness,
  scientific superiority, or runtime ranking.

Actions:

- Added `validate_ledh_forward_scalar_artifact(...)`.
- Added canonical executable schema version:
  `bayesfilter.highdim.ledh_forward_scalar_artifact.v1`.
- Added focused schema/admission guard tests.
- Ran Phase 1 required local checks.
- Wrote Phase 1 result.
- Drafted Phase 2 LGSSM subplan.
- Ran bounded Claude read-only Phase 1 handoff review.
- Review returned `VERDICT=REVISE` because theta values were not enforced
  against forward-contract `truth_theta`.
- Patched validator, tests, Phase 1 result, and Phase 2 stop conditions.
- Reran focused and required Phase 1 checks.
- Ran bounded Claude read-only repair review.

Artifacts:

- `bayesfilter/highdim/ledh_forward_contract.py`
- `tests/highdim/test_ledh_forward_scalar_admission_guard.py`
- `docs/plans/bayesfilter-ledh-forward-scalar-per-model-phase1-runner-schema-result-2026-07-07.md`
- `docs/plans/bayesfilter-ledh-forward-scalar-per-model-phase2-lgssm-subplan-2026-07-07.md`
- `docs/reviews/bayesfilter-ledh-forward-scalar-per-model-phase1-review-bundle-2026-07-07.md`
- `docs/reviews/bayesfilter-ledh-forward-scalar-per-model-phase1-repair1-review-bundle-2026-07-07.md`

Local checks:

```text
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest \
  tests/highdim/test_ledh_forward_scalar_admission_guard.py -q
```

Result:

```text
12 passed, 2 warnings in 3.23s
```

```text
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest \
  tests/highdim/test_ledh_phase3_forward_admission.py \
  tests/highdim/test_ledh_forward_contract_phase2.py \
  tests/highdim/test_ledh_forward_scalar_admission_guard.py -q
```

Result:

```text
24 passed, 2 warnings in 2.72s
```

Review:

- Initial review run dir:
  `/home/chakwong/BayesFilter/.claude_reviews/20260707-032055-ledh-forward-scalar-per-model-phase1-phase2-handoff`
- Initial status: `REVIEW_STATUS=revise`, `VERDICT=REVISE`.
- Repair review run dir:
  `/home/chakwong/BayesFilter/.claude_reviews/20260707-033721-ledh-forward-scalar-per-model-phase1-repair1`
- Repair status: `REVIEW_STATUS=agreed`, `VERDICT=AGREE`.

Gate status:

- `PASSED_PHASE1_PHASE2_HANDOFF_PHASE2_MAY_START`

Next action:

- Execute Phase 2 LGSSM forward scalar reconfirmation.

### 2026-07-07 - Phase 2 - LGSSM_FORWARD_SCALAR_LOCAL_PASS

Evidence contract:

- Question: Can the LGSSM row produce or preserve an executable same-target
  observed-data log likelihood artifact under the shared schema?
- Target scalar: `observed_data_log_likelihood_estimator`, reported as
  `log_likelihood`.
- Primary criterion: canonical LGSSM artifact validates with
  `require_admitted=True`, finite `log_likelihood_by_seed`, full-row scale, and
  exact Kalman target identity.
- Veto diagnostics: missing likelihood vector, wrong target scalar, wrong
  exact comparator, theta mismatch, failed schema validation, score evidence
  used for value admission, or runtime-only admission.
- Nonclaims: not nonlinear-row evidence, no score admission, no score
  correctness, no leaderboard rebuild, no new GPU model evidence, no HMC
  readiness, no posterior correctness, no scientific superiority, and no
  runtime ranking.

Actions:

- Normalized the existing LGSSM N=10000 value artifact into the Phase 1
  canonical executable artifact schema.
- Validated the canonical artifact with
  `validate_ledh_forward_scalar_artifact(..., require_admitted=True)`.
- Added Phase 2 LGSSM artifact replay tests.
- Ran focused and combined local checks.
- Wrote Phase 2 result.
- Drafted Phase 3 fixed SIR subplan.

Artifacts:

- `docs/plans/ledh-phase2-lgssm-forward-scalar-artifact-2026-07-07.json`
- `docs/plans/ledh-phase2-lgssm-forward-scalar-artifact-2026-07-07.md`
- `tests/highdim/test_ledh_phase2_lgssm_forward_scalar_artifact.py`
- `docs/plans/bayesfilter-ledh-forward-scalar-per-model-phase2-lgssm-result-2026-07-07.md`
- `docs/plans/bayesfilter-ledh-forward-scalar-per-model-phase3-fixed-sir-subplan-2026-07-07.md`

Local checks:

```text
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest \
  tests/highdim/test_ledh_phase2_lgssm_forward_scalar_artifact.py -q
```

Result:

```text
2 passed, 2 warnings in 2.55s
```

```text
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest \
  tests/highdim/test_ledh_phase3_forward_admission.py \
  tests/highdim/test_ledh_forward_contract_phase2.py \
  tests/highdim/test_ledh_forward_scalar_admission_guard.py \
  tests/highdim/test_ledh_phase2_lgssm_forward_scalar_artifact.py -q
```

Result:

```text
26 passed, 2 warnings in 2.77s
```

Gate status:

- `IN_PROGRESS_PENDING_PHASE2_PHASE3_REVIEW`

Next action:

- Run bounded read-only review of Phase 2 result and Phase 3 fixed SIR
  subplan.

### 2026-07-07 - Phase 2 - REVIEW_REPAIR_ACCEPTED

Evidence contract:

- Question: Did the Phase 2/3 handoff repair make fixed SIR replay mandatory?
- Primary criterion: read-only review agrees the Phase 3 fixed SIR subplan
  requires a replay test that reads the actual Phase 3 canonical JSON artifact
  and validates it with `require_admitted=True`.
- Nonclaims: no Phase 3 execution yet, no new model row beyond LGSSM
  reconfirmation, no score admission, no score correctness, no leaderboard
  rebuild, no new GPU model evidence, and no scientific conclusion.

Actions:

- Ran bounded Claude read-only review of Phase 2 result and Phase 3 subplan.
- Review returned `VERDICT=REVISE` because fixed SIR artifact replay was
  optional.
- Patched Phase 3 subplan so
  `tests/highdim/test_ledh_phase3_fixed_sir_forward_scalar_artifact.py` is
  mandatory and appears in required checks and handoff conditions.
- Patched Phase 2 result to record the repair.
- Ran bounded Claude read-only repair review.

Artifacts:

- `docs/plans/bayesfilter-ledh-forward-scalar-per-model-phase2-lgssm-result-2026-07-07.md`
- `docs/plans/bayesfilter-ledh-forward-scalar-per-model-phase3-fixed-sir-subplan-2026-07-07.md`
- `docs/reviews/bayesfilter-ledh-forward-scalar-per-model-phase2-review-bundle-2026-07-07.md`
- `docs/reviews/bayesfilter-ledh-forward-scalar-per-model-phase2-repair1-review-bundle-2026-07-07.md`

Review:

- Initial review run dir:
  `/home/chakwong/BayesFilter/.claude_reviews/20260707-040054-ledh-forward-scalar-per-model-phase2-phase3-handoff`
- Initial status: `REVIEW_STATUS=revise`, `VERDICT=REVISE`.
- Repair review run dir:
  `/home/chakwong/BayesFilter/.claude_reviews/20260707-040534-ledh-forward-scalar-per-model-phase2-repair1`
- Repair status: `REVIEW_STATUS=agreed`, `VERDICT=AGREE`.

Gate status:

- `PASSED_PHASE2_PHASE3_HANDOFF_PHASE3_MAY_START`

Next action:

- Execute Phase 3 fixed SIR forward scalar reconfirmation.

### 2026-07-07 - Phase 3 - FIXED_SIR_FORWARD_SCALAR_LOCAL_PASS

Evidence contract:

- Question: Can the fixed SIR row produce or preserve an executable same-target
  observed-data log likelihood artifact under the shared schema and amended
  free-parameter target?
- Target scalar: `observed_data_log_likelihood_estimator`, reported as
  `log_likelihood`.
- Primary criterion: canonical fixed SIR artifact validates with
  `require_admitted=True`, finite `log_likelihood_by_seed`, full-row scale,
  `sir_log_scale_theta`, theta `[0,0,0]`, and target-density correction.
- Veto diagnostics: old `no_free_theta` admission, missing likelihood vector,
  theta mismatch, missing target-density correction, score evidence used for
  value admission, or runtime-only admission.
- Nonclaims: not old `no_free_theta` admission, not predator-prey/SV evidence,
  no score admission, no score correctness, no leaderboard rebuild, no new GPU
  model evidence, no exact nonlinear likelihood correctness, no Zhao-Cui
  TT/SIRT source-faithfulness, no HMC readiness, no posterior correctness, no
  scientific superiority, and no runtime ranking.

Actions:

- Normalized the existing fixed SIR N=10000 value artifact into the Phase 1
  canonical executable artifact schema.
- Validated the canonical artifact with
  `validate_ledh_forward_scalar_artifact(..., require_admitted=True)`.
- Added mandatory fixed SIR artifact replay tests.
- Ran focused and combined local checks.
- Wrote Phase 3 result.
- Drafted Phase 4 predator-prey subplan.

Artifacts:

- `docs/plans/ledh-phase3-fixed-sir-forward-scalar-artifact-2026-07-07.json`
- `docs/plans/ledh-phase3-fixed-sir-forward-scalar-artifact-2026-07-07.md`
- `tests/highdim/test_ledh_phase3_fixed_sir_forward_scalar_artifact.py`
- `docs/plans/bayesfilter-ledh-forward-scalar-per-model-phase3-fixed-sir-result-2026-07-07.md`
- `docs/plans/bayesfilter-ledh-forward-scalar-per-model-phase4-predator-prey-subplan-2026-07-07.md`

Local checks:

```text
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest \
  tests/highdim/test_ledh_phase3_fixed_sir_forward_scalar_artifact.py -q
```

Result:

```text
2 passed, 2 warnings in 2.50s
```

```text
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest \
  tests/highdim/test_ledh_phase3_forward_admission.py \
  tests/highdim/test_ledh_forward_contract_phase2.py \
  tests/highdim/test_ledh_forward_scalar_admission_guard.py \
  tests/highdim/test_ledh_phase2_lgssm_forward_scalar_artifact.py \
  tests/highdim/test_ledh_phase3_fixed_sir_forward_scalar_artifact.py -q
```

Result:

```text
28 passed, 2 warnings in 2.57s
```

Gate status:

- `IN_PROGRESS_PENDING_PHASE3_PHASE4_REVIEW`

Next action:

- Run bounded read-only review of Phase 3 result and Phase 4 predator-prey
  subplan.

### 2026-07-07 - Phase 3 - REVIEW_ACCEPTED

Evidence contract:

- Question: Did Phase 3 correctly close fixed SIR scalar reconfirmation and
  safely hand off to predator-prey as the first previously blocked model?
- Primary criterion: read-only review agrees Phase 3 stays within fixed SIR
  scalar scope and Phase 4 has safe predator-prey stop/handoff conditions.
- Nonclaims: no predator-prey execution yet, no score admission, no score
  correctness, no leaderboard rebuild, no new GPU model evidence, and no
  scientific conclusion.

Actions:

- Ran bounded Claude read-only review of Phase 3 result and Phase 4 subplan.

Artifacts:

- `docs/plans/bayesfilter-ledh-forward-scalar-per-model-phase3-fixed-sir-result-2026-07-07.md`
- `docs/plans/bayesfilter-ledh-forward-scalar-per-model-phase4-predator-prey-subplan-2026-07-07.md`
- `docs/reviews/bayesfilter-ledh-forward-scalar-per-model-phase3-review-bundle-2026-07-07.md`

Review:

- Run dir:
  `/home/chakwong/BayesFilter/.claude_reviews/20260707-041754-ledh-forward-scalar-per-model-phase3-phase4-handoff`
- `REVIEW_STATUS=agreed`, `VERDICT=AGREE`.

Gate status:

- `PASSED_PHASE3_PHASE4_HANDOFF_PHASE4_MAY_START`

Next action:

- Execute Phase 4 predator-prey inventory and forward scalar build/blocker
  path.

### 2026-07-07 - Phase 4 - PREDATOR_PREY_FORWARD_SCALAR_LOCAL_PASS

Evidence contract:

- Question: Can the predator-prey row produce an executable same-target
  observed-data log likelihood artifact under the shared schema?
- Target scalar: `observed_data_log_likelihood_estimator`, reported as
  `log_likelihood`.
- Primary criterion: canonical predator-prey artifact validates with
  `require_admitted=True`, finite `log_likelihood_by_seed`, full-row scale,
  physical theta `(0.6,114,25,0.3,0.5,0.5)`, and target-density correction.
- Veto diagnostics: LGSSM/SIR evidence borrowing, metadata-only evidence,
  callback-only evidence, missing likelihood vector, wrong theta, missing
  target correction, score evidence used for value admission, or runtime-only
  admission.
- Nonclaims: no score admission, no score correctness, no exact nonlinear
  likelihood correctness, no Zhao-Cui TT/SIRT source-faithfulness, no HMC
  readiness, no posterior correctness, no scientific superiority, and no
  runtime ranking.

Actions:

- Inventoried predator-prey row callbacks and prior blocker records.
- Added a current-route streaming LEDH-PFPF-OT predator-prey forward scalar
  runner.
- Ran a tiny CPU-hidden smoke artifact.
- Ran the trusted GPU/XLA full-row N=10000 artifact.
- Added mandatory predator-prey artifact replay tests.
- Ran the required local check set.
- Wrote Phase 4 result.
- Drafted Phase 5 actual-SV subplan.

Artifacts:

- `docs/benchmarks/benchmark_ledh_same_target_predator_prey_value.py`
- `docs/plans/ledh-phase4-predator-prey-forward-scalar-artifact-2026-07-07.json`
- `docs/plans/ledh-phase4-predator-prey-forward-scalar-artifact-2026-07-07.md`
- `tests/highdim/test_ledh_phase4_predator_prey_forward_scalar_artifact.py`
- `docs/plans/bayesfilter-ledh-forward-scalar-per-model-phase4-predator-prey-result-2026-07-07.md`
- `docs/plans/bayesfilter-ledh-forward-scalar-per-model-phase5-actual-sv-subplan-2026-07-07.md`

Full-row trusted GPU command:

```text
MPLCONFIGDIR=/tmp python \
  docs/benchmarks/benchmark_ledh_same_target_predator_prey_value.py \
  --device-scope visible --device /GPU:0 --expect-device-kind gpu \
  --batch-seeds 81120,81121,81122,81123,81124 --time-steps 20 \
  --num-particles 10000 --transport-policy active-all \
  --sinkhorn-iterations 10 --sinkhorn-epsilon 1.0 \
  --row-chunk-size 512 --col-chunk-size 512 --particle-chunk-size 512 \
  --history-mode value-only --warmups 0 --repeats 1 \
  --output docs/plans/ledh-phase4-predator-prey-forward-scalar-artifact-2026-07-07.json \
  --markdown-output docs/plans/ledh-phase4-predator-prey-forward-scalar-artifact-2026-07-07.md
```

Full-row result:

- `admission_status = n10000_same_target_value_admitted`;
- `log_likelihood_by_seed =
  [-169.6912841796875, -169.636962890625, -169.46498107910156,
  -171.49961853027344, -169.044677734375]`;
- output tensor device `/job:localhost/replica:0/task:0/device:GPU:0`;
- compile plus first call `34.134711731923744` seconds;
- warm call `19.095908388961107` seconds;
- finite output and schema validation passed.

Local checks:

```text
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest \
  tests/highdim/test_ledh_phase3_forward_admission.py \
  tests/highdim/test_ledh_forward_contract_phase2.py \
  tests/highdim/test_ledh_forward_scalar_admission_guard.py \
  tests/highdim/test_ledh_phase2_lgssm_forward_scalar_artifact.py \
  tests/highdim/test_ledh_phase3_fixed_sir_forward_scalar_artifact.py \
  tests/highdim/test_ledh_phase4_predator_prey_forward_scalar_artifact.py -q
```

Result:

```text
30 passed, 2 warnings in 2.79s
```

Gate status:

- `IN_PROGRESS_PENDING_PHASE4_PHASE5_REVIEW`

Next action:

- Run bounded read-only review of Phase 4 result and Phase 5 actual-SV
  subplan.

### 2026-07-07 - Phase 4 - REVIEW_REPAIR_ACCEPTED

Evidence contract:

- Question: Did the Phase 4/5 handoff repair close the read-only review
  findings?
- Primary criterion: review agrees the predator-prey artifact/replay now
  consistently preserve the scientific-superiority nonclaim and the Phase 5
  fallback-review clause is narrowed to documented Claude-unavailability or
  policy-blocked contingency only.
- Nonclaims: no Phase 5 execution yet, no actual-SV admission, no score
  admission, no score correctness, no leaderboard rebuild, and no scientific
  conclusion.

Actions:

- Ran bounded Claude read-only review of Phase 4 result and Phase 5 subplan.
- Review returned `VERDICT=REVISE`.
- Finding 1: predator-prey runner/artifact/replay did not consistently include
  or assert the scientific-superiority nonclaim.
- Finding 2: Phase 5 handoff had an overly broad fallback Codex review clause.
- Patched the runner, artifact JSON, artifact markdown, replay test, and Phase
  5 subplan.
- Ran focused repair checks.
- Ran bounded Claude read-only repair review.

Artifacts:

- Initial review run dir:
  `/home/chakwong/BayesFilter/.claude_reviews/20260707-044709-ledh-forward-scalar-per-model-phase4-phase5-handoff`
- Repair review run dir:
  `/home/chakwong/BayesFilter/.claude_reviews/20260707-045118-ledh-forward-scalar-per-model-phase4-repair1`
- `docs/reviews/bayesfilter-ledh-forward-scalar-per-model-phase4-review-bundle-2026-07-07.md`
- `docs/reviews/bayesfilter-ledh-forward-scalar-per-model-phase4-repair1-review-bundle-2026-07-07.md`

Focused repair checks:

```text
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest \
  tests/highdim/test_ledh_phase4_predator_prey_forward_scalar_artifact.py -q
```

Result:

```text
2 passed, 2 warnings in 3.04s
```

```text
python -m py_compile \
  docs/benchmarks/benchmark_ledh_same_target_predator_prey_value.py \
  tests/highdim/test_ledh_phase4_predator_prey_forward_scalar_artifact.py
```

Result: passed.

```text
git diff --check -- \
  docs/benchmarks/benchmark_ledh_same_target_predator_prey_value.py \
  tests/highdim/test_ledh_phase4_predator_prey_forward_scalar_artifact.py \
  docs/plans/bayesfilter-ledh-forward-scalar-per-model-phase5-actual-sv-subplan-2026-07-07.md \
  docs/plans/ledh-phase4-predator-prey-forward-scalar-artifact-2026-07-07.json \
  docs/plans/ledh-phase4-predator-prey-forward-scalar-artifact-2026-07-07.md
```

Result: passed.

Review:

- Initial status: `REVIEW_STATUS=revise`, `VERDICT=REVISE`.
- Repair status: `REVIEW_STATUS=agreed`, `VERDICT=AGREE`.

Gate status:

- `PASSED_PHASE4_PHASE5_HANDOFF_PHASE5_MAY_START`

### 2026-07-07 - Phase 5 Repair - ACTUAL_SV_ADAPTER_SMOKE_PASSED

Evidence contract:

- Question: Can a tiny actual-SV LEDH adapter execute the exact transformed
  observed-data target correction before a full-row run?
- Target scalar: `observed_data_log_likelihood_estimator`, reported as
  `log_likelihood`.
- Primary criterion: tiny JSON artifact validates under the Phase 1 schema with
  `admission_status=tiny_executed_not_full_row`, finite log likelihood,
  `target_observation_policy=transformed_actual_sv_log_y_square`, exact
  `log(y^2)` transform with offset `0.0`, and exact log-chi-square target
  correction.
- Veto diagnostics: full-row run/admission, raw Gaussian target correction, KSC
  substitution, augmented-noise Gaussian closure, target density omitted, score
  work, or scientific/runtime claims.
- Nonclaims: no full actual-SV row admission, score admission, score
  correctness, HMC readiness, posterior correctness, scientific superiority, or
  runtime ranking.

Actions:

- Created a dedicated adapter-smoke subplan and review bundle.
- Claude review gate probe timed out twice, but a direct tiny Claude probe
  returned `CLAUDE_PROBE_OK`.
- Ran a narrowed direct Claude read-only plan review; result `VERDICT: AGREE`.
- Implemented `docs/benchmarks/benchmark_ledh_same_target_actual_sv_value.py`.
- Implemented
  `tests/highdim/test_ledh_phase5_actual_sv_forward_scalar_tiny_artifact.py`.
- Ran trusted GPU tiny smoke at `T=4,N=128,seed=81120`.
- Wrote tiny JSON and markdown artifacts.

Artifacts:

- `docs/plans/bayesfilter-ledh-forward-scalar-per-model-phase5-actual-sv-adapter-smoke-subplan-2026-07-07.md`
- `docs/reviews/bayesfilter-ledh-forward-scalar-per-model-phase5-actual-sv-adapter-smoke-plan-review-bundle-2026-07-07.md`
- `docs/benchmarks/benchmark_ledh_same_target_actual_sv_value.py`
- `docs/plans/ledh-phase5-actual-sv-forward-scalar-tiny-smoke-artifact-2026-07-07.json`
- `docs/plans/ledh-phase5-actual-sv-forward-scalar-tiny-smoke-artifact-2026-07-07.md`
- `tests/highdim/test_ledh_phase5_actual_sv_forward_scalar_tiny_artifact.py`
- `docs/plans/bayesfilter-ledh-forward-scalar-per-model-phase5-actual-sv-adapter-smoke-result-2026-07-07.md`

Local checks:

```text
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m py_compile \
  docs/benchmarks/benchmark_ledh_same_target_actual_sv_value.py
```

Result: passed.

```text
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest \
  tests/highdim/test_ledh_forward_scalar_admission_guard.py \
  tests/highdim/test_ledh_phase5_actual_sv_forward_scalar_tiny_artifact.py -q
```

Result:

```text
15 passed, 2 warnings in 2.83s
```

Tiny smoke result:

- output device: `/job:localhost/replica:0/task:0/device:GPU:0`;
- admission status: `tiny_executed_not_full_row`;
- `log_likelihood_by_seed`: `[-7.566841125488281]`;
- `average_log_likelihood_by_seed`: `[-1.8917102813720703]`;
- finite output: `true`;
- full-row admission: `false`.

Gate status:

- `PASSED_TINY_ADAPTER_SMOKE_NO_FULL_ROW_ADMISSION`

Next action:

- Run a bounded read-only implementation/result review, then draft the full-row
  actual-SV Phase 5 subplan if review agrees.

### 2026-07-07 - Phase 5 - ACTUAL_SV_FORWARD_SCALAR_LOCAL_PASS

Evidence contract:

- Question: Can the exact transformed actual-SV LEDH adapter produce a
  full-row executable same-target observed-data likelihood artifact?
- Target scalar: `observed_data_log_likelihood_estimator`, reported as
  `log_likelihood`.
- Primary criterion: canonical actual-SV artifact validates with
  `require_admitted=True`, finite `log_likelihood_by_seed`, row id
  `zhao_cui_sv_actual_nongaussian_T1000`, theta coordinate
  `synthetic_unconstrained`, theta values
  `[0.2533471031357997,-0.916290731874155]`, `T=1000`, `N=10000`, seeds
  `[81120,81121,81122,81123,81124]`, exact transformed target policy, target
  density correction, and GPU output device.
- Veto diagnostics: no tiny artifact admission, no raw Gaussian target
  correction, no KSC finite mixture, no augmented-noise Gaussian closure, no
  positive transform offset, no wrong theta/seeds/T/N, no nonfinite output, no
  score evidence, and no runtime-only admission.
- Nonclaims: no score admission, score correctness, generalized-SV admission,
  KSC admission, HMC readiness, posterior correctness, scientific superiority,
  or runtime ranking.

Actions:

- Full-row Phase 5 subplan passed read-only review before execution.
- Patched the actual-SV runner with explicit `--run-scope` guard.
- Ran trusted GPU/XLA full row at `T=1000,N=10000` with five seeds.
- Added mandatory full-row replay test.
- Repaired a metadata-only nonclaim mismatch: admitted full-row artifact must
  not carry tiny-only `not full actual-SV row admission`.
- Claude read-only review returned `VERDICT=REVISE` because the cached
  `validator_normalized_core.nonclaims` block still carried the tiny-only
  nonclaim even though the top-level artifact nonclaims were clean.
- Patched the cached normalized block and strengthened the full-row replay
  test to require cached normalized nonclaims to equal top-level nonclaims.
- Reran compile, through-Phase-5 replay, and diff hygiene checks.
- Wrote Phase 5 result.
- Drafted Phase 6 generalized-SV subplan.

Artifacts:

- `docs/benchmarks/benchmark_ledh_same_target_actual_sv_value.py`
- `docs/plans/ledh-phase5-actual-sv-forward-scalar-artifact-2026-07-07.json`
- `docs/plans/ledh-phase5-actual-sv-forward-scalar-artifact-2026-07-07.md`
- `tests/highdim/test_ledh_phase5_actual_sv_forward_scalar_artifact.py`
- `docs/plans/bayesfilter-ledh-forward-scalar-per-model-phase5-actual-sv-full-row-subplan-2026-07-07.md`
- `docs/plans/bayesfilter-ledh-forward-scalar-per-model-phase5-actual-sv-result-2026-07-07.md`
- `docs/plans/bayesfilter-ledh-forward-scalar-per-model-phase6-generalized-sv-subplan-2026-07-07.md`

Full-row trusted GPU command:

```text
MPLCONFIGDIR=/tmp python docs/benchmarks/benchmark_ledh_same_target_actual_sv_value.py \
  --run-scope full-row-admission --time-steps 1000 --num-particles 10000 \
  --batch-seeds 81120,81121,81122,81123,81124 \
  --transport-policy active-all --sinkhorn-iterations 10 \
  --row-chunk-size 512 --col-chunk-size 512 --particle-chunk-size 512 \
  --history-mode value-only --warmups 0 --repeats 1 \
  --device /GPU:0 --expect-device-kind gpu \
  --output docs/plans/ledh-phase5-actual-sv-forward-scalar-artifact-2026-07-07.json \
  --markdown-output docs/plans/ledh-phase5-actual-sv-forward-scalar-artifact-2026-07-07.md
```

Full-row result:

- `admission_status = n10000_same_target_value_admitted`;
- `log_likelihood_by_seed =
  [-2290.10205078125, -2289.888916015625, -2289.83154296875,
  -2289.517333984375, -2290.427490234375]`;
- `average_log_likelihood_by_seed =
  [-2.29010205078125, -2.289888916015625, -2.28983154296875,
  -2.289517333984375, -2.290427490234375]`;
- output tensor device `/job:localhost/replica:0/task:0/device:GPU:0`;
- compile plus first call `1105.1444279109128` seconds;
- warm call `1066.3097243178636` seconds;
- finite output and schema validation passed.

Local checks:

```text
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m py_compile \
  docs/benchmarks/benchmark_ledh_same_target_actual_sv_value.py \
  tests/highdim/test_ledh_phase5_actual_sv_forward_scalar_tiny_artifact.py \
  tests/highdim/test_ledh_phase5_actual_sv_forward_scalar_artifact.py
```

Result: passed.

```text
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest \
  tests/highdim/test_ledh_phase3_forward_admission.py \
  tests/highdim/test_ledh_forward_contract_phase2.py \
  tests/highdim/test_ledh_forward_scalar_admission_guard.py \
  tests/highdim/test_ledh_phase2_lgssm_forward_scalar_artifact.py \
  tests/highdim/test_ledh_phase3_fixed_sir_forward_scalar_artifact.py \
  tests/highdim/test_ledh_phase4_predator_prey_forward_scalar_artifact.py \
  tests/highdim/test_ledh_phase5_actual_sv_forward_scalar_tiny_artifact.py \
  tests/highdim/test_ledh_phase5_actual_sv_forward_scalar_artifact.py -q
```

Result:

```text
35 passed, 2 warnings in 2.69s
```

```text
git diff --check -- \
  docs/benchmarks/benchmark_ledh_same_target_actual_sv_value.py \
  tests/highdim/test_ledh_phase5_actual_sv_forward_scalar_tiny_artifact.py \
  tests/highdim/test_ledh_phase5_actual_sv_forward_scalar_artifact.py \
  docs/plans/bayesfilter-ledh-forward-scalar-per-model-phase5-actual-sv-full-row-subplan-2026-07-07.md \
  docs/plans/ledh-phase5-actual-sv-forward-scalar-artifact-2026-07-07.md \
  docs/plans/bayesfilter-ledh-forward-scalar-per-model-visible-execution-ledger-2026-07-07.md
```

Result: passed.

Gate status:

- `IN_PROGRESS_PENDING_PHASE5_PHASE6_REVIEW`

Next action:

- Run bounded read-only review of Phase 5 result and Phase 6 generalized-SV
  subplan.

### 2026-07-07 - Phase 5 - REVIEW_REPAIR_ACCEPTED

Evidence contract:

- Question: Did the Phase 5 metadata repair fully close the actual-SV full-row
  artifact inconsistency, and is the Phase 6 generalized-SV handoff safe?
- Primary criterion: read-only review agrees the cached
  `validator_normalized_core.nonclaims` block no longer carries the tiny-only
  nonclaim, the replay test guards cached/top-level equality, and Phase 6
  remains boundary-safe.
- Nonclaims: no Phase 6 execution yet, no generalized-SV admission, no score
  admission, no score correctness, no leaderboard rebuild, and no scientific
  conclusion.

Actions:

- Standard Claude review gate timed out at probe:
  `REVIEW_STATUS=probe_timeout`, `VERDICT=NONE`.
- Direct tiny Claude probe returned `CLAUDE_PROBE_OK`.
- Narrowed direct read-only review returned `VERDICT: REVISE`.
- Review finding: top-level actual-SV nonclaims were repaired, but cached
  `validator_normalized_core.nonclaims` still carried
  `not full actual-SV row admission`.
- Patched the full JSON artifact cached normalized block to match top-level
  nonclaims.
- Strengthened the full-row replay test to assert cached normalized nonclaims
  equal top-level nonclaims and exclude the tiny-only phrase.
- Updated Phase 5 result to record the two-stage metadata repair.
- Reran focused and through-Phase-5 checks.
- Ran focused direct Claude repair review; result `VERDICT: AGREE`.

Artifacts:

- Standard gate run dir:
  `/home/chakwong/BayesFilter/.claude_reviews/20260707-174947-ledh-forward-scalar-per-model-phase5-phase6-handoff`
- Review bundle:
  `docs/reviews/bayesfilter-ledh-forward-scalar-per-model-phase5-actual-sv-full-row-result-review-bundle-2026-07-07.md`
- Patched:
  `docs/plans/ledh-phase5-actual-sv-forward-scalar-artifact-2026-07-07.json`
- Patched:
  `tests/highdim/test_ledh_phase5_actual_sv_forward_scalar_artifact.py`
- Patched:
  `docs/plans/bayesfilter-ledh-forward-scalar-per-model-phase5-actual-sv-result-2026-07-07.md`

Focused repair checks:

```text
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest \
  tests/highdim/test_ledh_phase5_actual_sv_forward_scalar_artifact.py \
  tests/highdim/test_ledh_phase5_actual_sv_forward_scalar_tiny_artifact.py -q
```

Result:

```text
5 passed, 2 warnings in 5.34s
```

Through-Phase-5 replay:

```text
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest \
  tests/highdim/test_ledh_phase3_forward_admission.py \
  tests/highdim/test_ledh_forward_contract_phase2.py \
  tests/highdim/test_ledh_forward_scalar_admission_guard.py \
  tests/highdim/test_ledh_phase2_lgssm_forward_scalar_artifact.py \
  tests/highdim/test_ledh_phase3_fixed_sir_forward_scalar_artifact.py \
  tests/highdim/test_ledh_phase4_predator_prey_forward_scalar_artifact.py \
  tests/highdim/test_ledh_phase5_actual_sv_forward_scalar_tiny_artifact.py \
  tests/highdim/test_ledh_phase5_actual_sv_forward_scalar_artifact.py -q
```

Result:

```text
35 passed, 2 warnings in 5.36s
```

Review:

- Initial narrowed review: `VERDICT=REVISE`.
- Focused repair review: `VERDICT=AGREE`.

Gate status:

- `PASSED_PHASE5_PHASE6_HANDOFF_PHASE6_MAY_START`

Next action:

- Execute Phase 6 generalized-SV inventory and tiny-smoke implementation path.

### 2026-07-07 - Phase 6 - GENERALIZED_SV_FORWARD_SCALAR_LOCAL_PASS

Evidence contract:

- Question: Can the source-route prior-mean generalized-SV LEDH adapter produce
  a full-row executable same-target observed-data likelihood artifact?
- Target scalar: `observed_data_log_likelihood_estimator`, reported as
  `log_likelihood`.
- Primary criterion: canonical generalized-SV artifact validates with
  `require_admitted=True`, finite `log_likelihood_by_seed`, row id
  `zhao_cui_generalized_sv_synthetic_from_estimated_values`, theta coordinate
  `source_route_active_transformed_prior_mean`, theta values
  `[1.0824113944610982,-2.076793740349318,0.0]`, `T=1008`, `N=10000`, seeds
  `[81120,81121,81122,81123,81124]`, source-route prior-mean target policy,
  target density correction, and GPU output device.
- Veto diagnostics: no tiny artifact admission, no actual-SV target borrowing,
  no KSC mixture borrowing, no native generalized-SV dense fixture
  substitution, no SP500 benchmark observations, no author-default truth
  substitution, no log-square proposal promoted to target likelihood, no
  wrong theta/seeds/T/N, no nonfinite output, no score evidence, and no
  runtime-only admission.
- Nonclaims: no score admission, score correctness, KSC admission, actual-SV
  admission, HMC readiness, posterior correctness, scientific superiority, or
  runtime ranking.

Actions:

- Patched the generalized-SV runner with explicit `--run-scope` guard.
- Ran trusted GPU/XLA tiny smoke at `T=4,N=128` with one seed.
- Added mandatory tiny replay test.
- Ran trusted GPU/XLA full row at `T=1008,N=10000` with five seeds.
- Added mandatory full-row replay test.
- Ran compile, focused Phase 6 replay, through-Phase-6 replay, and diff hygiene
  checks.
- Wrote Phase 6 result.
- Drafted Phase 7 KSC-SV subplan.

Artifacts:

- `docs/benchmarks/benchmark_ledh_same_target_generalized_sv_value.py`
- `docs/plans/ledh-phase6-generalized-sv-forward-scalar-tiny-smoke-artifact-2026-07-07.json`
- `docs/plans/ledh-phase6-generalized-sv-forward-scalar-tiny-smoke-artifact-2026-07-07.md`
- `docs/plans/ledh-phase6-generalized-sv-forward-scalar-artifact-2026-07-07.json`
- `docs/plans/ledh-phase6-generalized-sv-forward-scalar-artifact-2026-07-07.md`
- `tests/highdim/test_ledh_phase6_generalized_sv_forward_scalar_tiny_artifact.py`
- `tests/highdim/test_ledh_phase6_generalized_sv_forward_scalar_artifact.py`
- `docs/plans/bayesfilter-ledh-forward-scalar-per-model-phase6-generalized-sv-result-2026-07-07.md`
- `docs/plans/bayesfilter-ledh-forward-scalar-per-model-phase7-ksc-sv-subplan-2026-07-07.md`
- `docs/reviews/bayesfilter-ledh-forward-scalar-per-model-phase6-review-bundle-2026-07-07.md`

Full-row trusted GPU command:

```text
MPLCONFIGDIR=/tmp python docs/benchmarks/benchmark_ledh_same_target_generalized_sv_value.py \
  --run-scope full-row-admission --time-steps 1008 --num-particles 10000 \
  --batch-seeds 81120,81121,81122,81123,81124 \
  --transport-policy active-all --sinkhorn-iterations 10 \
  --row-chunk-size 512 --col-chunk-size 512 --particle-chunk-size 512 \
  --history-mode value-only --warmups 0 --repeats 1 \
  --device /GPU:0 --expect-device-kind gpu \
  --output docs/plans/ledh-phase6-generalized-sv-forward-scalar-artifact-2026-07-07.json \
  --markdown-output docs/plans/ledh-phase6-generalized-sv-forward-scalar-artifact-2026-07-07.md
```

Full-row result:

- `admission_status = n10000_same_target_value_admitted`;
- `log_likelihood_by_seed =
  [-1438.90966796875, -1438.8360595703125, -1438.908935546875,
  -1439.0206298828125, -1438.9456787109375]`;
- `average_log_likelihood_by_seed =
  [-1.427489749968998, -1.427416725764199, -1.427489023359995,
  -1.427599831232949, -1.4275254749116444]`;
- output tensor device `/job:localhost/replica:0/task:0/device:GPU:0`;
- compile plus first call `1187.9543538549915` seconds;
- warm call `1048.247488206951` seconds;
- finite output and schema validation passed.

Local checks:

```text
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m py_compile \
  docs/benchmarks/benchmark_ledh_same_target_generalized_sv_value.py \
  tests/highdim/test_ledh_phase6_generalized_sv_forward_scalar_tiny_artifact.py \
  tests/highdim/test_ledh_phase6_generalized_sv_forward_scalar_artifact.py
```

Result: passed.

```text
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest \
  tests/highdim/test_ledh_phase6_generalized_sv_forward_scalar_tiny_artifact.py \
  tests/highdim/test_ledh_phase6_generalized_sv_forward_scalar_artifact.py -q
```

Result:

```text
5 passed, 2 warnings in 5.02s
```

```text
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest \
  tests/highdim/test_ledh_phase3_forward_admission.py \
  tests/highdim/test_ledh_forward_contract_phase2.py \
  tests/highdim/test_ledh_forward_scalar_admission_guard.py \
  tests/highdim/test_ledh_phase2_lgssm_forward_scalar_artifact.py \
  tests/highdim/test_ledh_phase3_fixed_sir_forward_scalar_artifact.py \
  tests/highdim/test_ledh_phase4_predator_prey_forward_scalar_artifact.py \
  tests/highdim/test_ledh_phase5_actual_sv_forward_scalar_tiny_artifact.py \
  tests/highdim/test_ledh_phase5_actual_sv_forward_scalar_artifact.py \
  tests/highdim/test_ledh_phase6_generalized_sv_forward_scalar_tiny_artifact.py \
  tests/highdim/test_ledh_phase6_generalized_sv_forward_scalar_artifact.py -q
```

Result:

```text
40 passed, 2 warnings in 2.64s
```

Gate status:

- `IN_PROGRESS_PENDING_PHASE6_PHASE7_REVIEW`

Next action:

- Run bounded read-only review of Phase 6 result and Phase 7 KSC-SV subplan.

### 2026-07-07 - Phase 6 - REVIEW_ACCEPTED

Evidence contract:

- Question: Is the Phase 6 generalized-SV local result internally consistent,
  and is the Phase 7 KSC-SV handoff safe?
- Primary criterion: read-only review agrees Phase 6 admits only the
  full-row generalized-SV value artifact and Phase 7 keeps KSC as a distinct
  finite Gaussian-mixture surrogate target with tiny-before-full discipline.
- Nonclaims: no Phase 7 execution yet, no KSC admission, no score admission, no
  score correctness, no leaderboard rebuild, and no scientific conclusion.

Actions:

- Standard Claude review gate timed out at probe:
  `REVIEW_STATUS=probe_timeout`, `VERDICT=NONE`.
- Direct tiny Claude probe returned `CLAUDE_PROBE_OK`.
- Narrowed fixed-path read-only review returned `VERDICT: AGREE`.

Artifacts:

- Standard gate run dir:
  `/home/chakwong/BayesFilter/.claude_reviews/20260707-200604-ledh-forward-scalar-per-model-phase6-phase7-handoff`
- Review bundle:
  `docs/reviews/bayesfilter-ledh-forward-scalar-per-model-phase6-review-bundle-2026-07-07.md`
- Phase 6 result:
  `docs/plans/bayesfilter-ledh-forward-scalar-per-model-phase6-generalized-sv-result-2026-07-07.md`
- Phase 7 subplan:
  `docs/plans/bayesfilter-ledh-forward-scalar-per-model-phase7-ksc-sv-subplan-2026-07-07.md`

Review:

- Direct narrowed review: `VERDICT=AGREE`.

Gate status:

- `PASSED_PHASE6_PHASE7_HANDOFF_PHASE7_MAY_START`

Next action:

- Execute Phase 7 KSC-SV inventory and tiny-smoke implementation path.

### 2026-07-07 - Phase 7 - KSC_SV_FORWARD_SCALAR_LOCAL_PASS

Evidence contract:

- Question: Can the declared KSC finite Gaussian-mixture surrogate SV row
  produce an executable same-target observed-data likelihood artifact under
  LEDH?
- Target scalar: `observed_data_log_likelihood_estimator`, reported as
  `log_likelihood`.
- Primary criterion: canonical KSC-SV artifact validates with
  `require_admitted=True`, finite `log_likelihood_by_seed`, row id
  `zhao_cui_sv_ksc_gaussian_mixture_surrogate_T1000`, theta coordinate
  `synthetic_unconstrained`, theta values
  `[0.2533471031357997,-0.916290731874155]`, `T=1000`, `N=10000`, seeds
  `[81120,81121,81122,81123,81124]`, target policy
  `ksc_log_chi_square_gaussian_mixture_surrogate`, transform offset `1e-8`,
  finite KSC mixture target likelihood, target-density correction, and GPU
  output device.
- Veto diagnostics: no tiny artifact admission, no exact actual-SV target
  density, no generalized-SV target density, no raw Gaussian callback target,
  no KSC mixture as proposal-only evidence, no wrong theta/seeds/T/N, no
  nonfinite output, no score evidence, and no runtime-only admission.
- Nonclaims: no score admission, score correctness, exact native actual-SV
  likelihood, actual-SV admission, generalized-SV admission, HMC readiness,
  posterior correctness, scientific superiority, or runtime ranking.

Actions:

- Patched the KSC-SV runner with explicit `--run-scope` guard.
- Added approved wrapper modes for trusted GPU tiny and full KSC runs.
- Ran trusted GPU/XLA tiny smoke at `T=4,N=128` with one seed.
- Added mandatory tiny replay test.
- Ran trusted GPU/XLA full row at `T=1000,N=10000` with five seeds.
- Added mandatory full-row replay test.
- Ran compile, focused Phase 7 replay, through-Phase-7 replay, and diff hygiene
  checks.
- Wrote Phase 7 result.
- Drafted Phase 8 value-only integration subplan.

Artifacts:

- `docs/benchmarks/benchmark_ledh_same_target_ksc_sv_value.py`
- `scripts/run_gpu_benchmark.sh`
- `docs/plans/ledh-phase7-ksc-sv-forward-scalar-tiny-smoke-artifact-2026-07-07.json`
- `docs/plans/ledh-phase7-ksc-sv-forward-scalar-tiny-smoke-artifact-2026-07-07.md`
- `docs/plans/ledh-phase7-ksc-sv-forward-scalar-artifact-2026-07-07.json`
- `docs/plans/ledh-phase7-ksc-sv-forward-scalar-artifact-2026-07-07.md`
- `tests/highdim/test_ledh_phase7_ksc_sv_forward_scalar_tiny_artifact.py`
- `tests/highdim/test_ledh_phase7_ksc_sv_forward_scalar_artifact.py`
- `docs/plans/bayesfilter-ledh-forward-scalar-per-model-phase7-ksc-sv-result-2026-07-07.md`
- `docs/plans/bayesfilter-ledh-forward-scalar-per-model-phase8-integration-subplan-2026-07-07.md`
- `docs/reviews/bayesfilter-ledh-forward-scalar-per-model-phase7-review-bundle-2026-07-07.md`

Full-row trusted GPU command:

```text
bash scripts/run_gpu_benchmark.sh ledh_phase7_ksc_full_row
```

Full-row result:

- `admission_status = n10000_same_target_value_admitted`;
- `log_likelihood_by_seed =
  [-2288.165771484375, -2287.877685546875, -2287.852294921875,
  -2287.529296875, -2288.34375]`;
- `average_log_likelihood_by_seed =
  [-2.288165771484375, -2.287877685546875, -2.287852294921875,
  -2.287529296875, -2.28834375]`;
- output tensor device `/job:localhost/replica:0/task:0/device:GPU:0`;
- compile plus first call `1054.9561004990246` seconds;
- warm call `1054.2882441349793` seconds;
- finite output and schema validation passed.

Local checks:

```text
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m py_compile \
  docs/benchmarks/benchmark_ledh_same_target_ksc_sv_value.py \
  tests/highdim/test_ledh_phase7_ksc_sv_forward_scalar_tiny_artifact.py \
  tests/highdim/test_ledh_phase7_ksc_sv_forward_scalar_artifact.py
```

Result: passed.

```text
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest \
  tests/highdim/test_ledh_phase7_ksc_sv_forward_scalar_tiny_artifact.py \
  tests/highdim/test_ledh_phase7_ksc_sv_forward_scalar_artifact.py -q
```

Result:

```text
5 passed, 2 warnings in 5.11s
```

```text
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest \
  tests/highdim/test_ledh_phase3_forward_admission.py \
  tests/highdim/test_ledh_forward_contract_phase2.py \
  tests/highdim/test_ledh_forward_scalar_admission_guard.py \
  tests/highdim/test_ledh_phase2_lgssm_forward_scalar_artifact.py \
  tests/highdim/test_ledh_phase3_fixed_sir_forward_scalar_artifact.py \
  tests/highdim/test_ledh_phase4_predator_prey_forward_scalar_artifact.py \
  tests/highdim/test_ledh_phase5_actual_sv_forward_scalar_tiny_artifact.py \
  tests/highdim/test_ledh_phase5_actual_sv_forward_scalar_artifact.py \
  tests/highdim/test_ledh_phase6_generalized_sv_forward_scalar_tiny_artifact.py \
  tests/highdim/test_ledh_phase6_generalized_sv_forward_scalar_artifact.py \
  tests/highdim/test_ledh_phase7_ksc_sv_forward_scalar_tiny_artifact.py \
  tests/highdim/test_ledh_phase7_ksc_sv_forward_scalar_artifact.py -q
```

Result:

```text
45 passed, 2 warnings in 5.24s
```

Gate status:

- `IN_PROGRESS_PENDING_PHASE7_PHASE8_REVIEW`

Next action:

- Run bounded read-only review of Phase 7 result and Phase 8 value-only
  integration subplan.

### 2026-07-07 - Phase 7 - REVIEW_ACCEPTED

Evidence contract:

- Question: Is the Phase 7 KSC-SV local result internally consistent, and is
  the Phase 8 value-only integration handoff safe?
- Primary criterion: read-only review agrees Phase 7 admits only the full-row
  KSC finite-mixture surrogate value artifact and Phase 8 consumes only
  admitted forward-scalar artifacts, with score integration, runtime
  cross-ranking, old score artifacts, and parameterized SIR diagnostic
  promotion forbidden.
- Nonclaims: no Phase 8 execution yet, no score admission, no score
  correctness, no score-inclusive leaderboard rebuild, no all-algorithm
  comparison, and no scientific conclusion.

Actions:

- Direct tiny Claude probe returned `CLAUDE_PROBE_OK`.
- A broader fixed-path review prompt produced no output within repeated polls,
  so it was stopped.
- A narrowed packet-only review returned `VERDICT: AGREE`.
- Review reminder: Phase 8 should preserve each row's target-policy label,
  especially that the KSC row is a finite-mixture surrogate target and not
  exact native actual-SV likelihood.

Artifacts:

- Review bundle:
  `docs/reviews/bayesfilter-ledh-forward-scalar-per-model-phase7-review-bundle-2026-07-07.md`
- Phase 7 result:
  `docs/plans/bayesfilter-ledh-forward-scalar-per-model-phase7-ksc-sv-result-2026-07-07.md`
- Phase 8 subplan:
  `docs/plans/bayesfilter-ledh-forward-scalar-per-model-phase8-integration-subplan-2026-07-07.md`

Review:

- Direct narrowed packet review: `VERDICT=AGREE`.

Gate status:

- `PASSED_PHASE7_PHASE8_HANDOFF_PHASE8_MAY_START`

Next action:

- Execute Phase 8 value-only integration from admitted forward-scalar
  artifacts.

### 2026-07-07 - Phase 8 - VALUE_ONLY_INTEGRATION_LOCAL_PASS

Evidence contract:

- Question: Can the six main LEDH high-dimensional model rows be assembled
  into one value-only integration artifact using only admitted same-target
  forward-scalar artifacts?
- Target scalar: `observed_data_log_likelihood_estimator`, reported as
  `log_likelihood`.
- Primary criterion: integration JSON has exactly six main LEDH value rows,
  each replay-validated from an admitted Phase 2-7 forward-scalar artifact;
  score fields are absent and score integration is blocked; runtime
  cross-ranking is disabled; all-algorithm comparison is disabled; the
  parameterized SIR diagnostic row is excluded from main rows.
- Veto diagnostics: no missing source artifact, stale row id, target-policy
  mismatch, tiny artifact admission, score field merge, old score artifact use,
  runtime ranking, diagnostic SIR promotion, KSC exact native actual-SV claim,
  or metadata-only admission.
- Nonclaims: no score admission, score correctness, all-algorithm comparison,
  HMC readiness, posterior correctness, scientific superiority, or runtime
  ranking.

Actions:

- Added a value-only integration builder.
- Added a replay test for the integration artifact.
- Built JSON and markdown integration artifacts from six admitted Phase 2-7
  forward-scalar artifacts.
- Ran compile, focused Phase 8 replay, through-Phase-8 replay, and diff hygiene
  checks.
- Wrote Phase 8 result and review bundle.

Artifacts:

- `docs/benchmarks/benchmark_ledh_forward_scalar_value_integration.py`
- `tests/highdim/test_ledh_phase8_value_integration_artifact.py`
- `docs/plans/bayesfilter-ledh-forward-scalar-value-integration-results-2026-07-07.json`
- `docs/plans/bayesfilter-ledh-forward-scalar-value-integration-results-2026-07-07.md`
- `docs/plans/bayesfilter-ledh-forward-scalar-per-model-phase8-integration-result-2026-07-07.md`
- `docs/reviews/bayesfilter-ledh-forward-scalar-per-model-phase8-review-bundle-2026-07-07.md`

Integration row means:

- `benchmark_lgssm_exact_oracle_m3_T50`: `-135.96007385253907`;
- `zhao_cui_spatial_sir_austria_j9_T20`: `-902.8301513671875`;
- `zhao_cui_predator_prey_T20`: `-169.8675048828125`;
- `zhao_cui_sv_actual_nongaussian_T1000`: `-2289.953466796875`;
- `zhao_cui_generalized_sv_synthetic_from_estimated_values`:
  `-1438.9241943359375`;
- `zhao_cui_sv_ksc_gaussian_mixture_surrogate_T1000`:
  `-2287.953759765625`.

Local checks:

```text
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m py_compile \
  docs/benchmarks/benchmark_ledh_forward_scalar_value_integration.py \
  tests/highdim/test_ledh_phase8_value_integration_artifact.py
```

Result: passed.

```text
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest \
  tests/highdim/test_ledh_phase8_value_integration_artifact.py -q
```

Result:

```text
3 passed, 2 warnings in 2.62s
```

```text
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest \
  tests/highdim/test_ledh_phase3_forward_admission.py \
  tests/highdim/test_ledh_forward_contract_phase2.py \
  tests/highdim/test_ledh_forward_scalar_admission_guard.py \
  tests/highdim/test_ledh_phase2_lgssm_forward_scalar_artifact.py \
  tests/highdim/test_ledh_phase3_fixed_sir_forward_scalar_artifact.py \
  tests/highdim/test_ledh_phase4_predator_prey_forward_scalar_artifact.py \
  tests/highdim/test_ledh_phase5_actual_sv_forward_scalar_tiny_artifact.py \
  tests/highdim/test_ledh_phase5_actual_sv_forward_scalar_artifact.py \
  tests/highdim/test_ledh_phase6_generalized_sv_forward_scalar_tiny_artifact.py \
  tests/highdim/test_ledh_phase6_generalized_sv_forward_scalar_artifact.py \
  tests/highdim/test_ledh_phase7_ksc_sv_forward_scalar_tiny_artifact.py \
  tests/highdim/test_ledh_phase7_ksc_sv_forward_scalar_artifact.py \
  tests/highdim/test_ledh_phase8_value_integration_artifact.py -q
```

Result:

```text
48 passed, 2 warnings in 2.92s
```

Gate status:

- `IN_PROGRESS_PENDING_PHASE8_REVIEW`

Next action:

- Run bounded read-only review of Phase 8 result and value-only integration
  artifact.

### 2026-07-07 - Phase 8 - REVIEW_ACCEPTED_RUNBOOK_VALUE_LAYER_CLOSED

Evidence contract:

- Question: Does Phase 8 safely close the forward-scalar value-only runbook?
- Primary criterion: read-only review agrees the packeted Phase 8 result is
  internally consistent and boundary-safe: value-only, no score creep, no
  runtime-ranking/all-algorithm claim, no diagnostic SIR promotion, and no KSC
  exact-SV overclaim.
- Nonclaims: no score admission, score correctness, all-algorithm comparison,
  HMC readiness, posterior correctness, scientific superiority, or runtime
  ranking.

Actions:

- Ran bounded packet-only Claude read-only review of Phase 8 result.
- Review returned `VERDICT: AGREE`.
- Review limitation: packet-only; Claude did not independently inspect the
  JSON/MD/builder/test contents. Local replay and diff hygiene provide the
  implementation-text evidence.

Artifacts:

- Review bundle:
  `docs/reviews/bayesfilter-ledh-forward-scalar-per-model-phase8-review-bundle-2026-07-07.md`
- Phase 8 result:
  `docs/plans/bayesfilter-ledh-forward-scalar-per-model-phase8-integration-result-2026-07-07.md`
- Integration JSON:
  `docs/plans/bayesfilter-ledh-forward-scalar-value-integration-results-2026-07-07.json`
- Integration markdown:
  `docs/plans/bayesfilter-ledh-forward-scalar-value-integration-results-2026-07-07.md`

Review:

- Packet-only review: `VERDICT=AGREE`.

Gate status:

- `PASSED_PHASE8_VALUE_ONLY_RUNBOOK_CLOSED`

Next action:

- Any future score or all-algorithm leaderboard work requires a separate
  reviewed plan and must not infer score admission from this value-only
  runbook.
