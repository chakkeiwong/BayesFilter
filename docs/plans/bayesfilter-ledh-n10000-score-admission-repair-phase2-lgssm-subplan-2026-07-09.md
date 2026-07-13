# Phase 2 Subplan: LGSSM N10000 Score Admission

Date: 2026-07-09

Status: `DRAFT_READY_FOR_REVIEW`

## Phase Objective

Produce a schema-valid, validator-admitted compact `N=10000,T=50` LGSSM score
artifact for row `benchmark_lgssm_exact_oracle_m3_T50`, bound to the admitted
value artifact:

```text
docs/plans/ledh-phase2-lgssm-forward-scalar-artifact-2026-07-07.json
```

The preferred route is a trusted rerun through
`docs/benchmarks/benchmark_ledh_same_target_lgssm_m3_t50_value.py` with
`--score-mode compact-sensitivity`, because the legacy July 6 raw score-memory
JSON lacks the full Phase 1 score artifact schema.

## Entry Conditions Inherited From Previous Phase

- Phase 1 added `build_ledh_score_artifact` and tests proving raw, tiny,
  missing-memory, mismatch, and historical-route cases do not full-admit.
- LGSSM value artifact is admitted with:
  - `num_particles = 10000`;
  - `time_steps = 50`;
  - `batch_seeds = [81120, 81121, 81122, 81123, 81124]`;
  - target scalar `observed_data_log_likelihood_estimator`;
  - output field `log_likelihood`;
  - theta coordinate system `physical_benchmark_exact_oracle`.
- Legacy raw LGSSM score evidence:
  `docs/plans/ledh-phase5-lgssm-score-memory-n10000-2026-07-06.json`
  is compact but not admitted because it lacks Phase 1 schema.

## Required Artifacts

- Full LGSSM score artifact:
  `docs/plans/bayesfilter-ledh-n10000-score-admission-repair-phase2-lgssm-score-artifact-2026-07-09.json`
- Markdown/log result if produced by the runner:
  `docs/plans/bayesfilter-ledh-n10000-score-admission-repair-phase2-lgssm-score-artifact-2026-07-09.md`
- Phase 2 result:
  `docs/plans/bayesfilter-ledh-n10000-score-admission-repair-phase2-lgssm-result-2026-07-09.md`
- Phase 3 subplan:
  `docs/plans/bayesfilter-ledh-n10000-score-admission-repair-phase3-fixed-sir-subplan-2026-07-09.md`
- Review bundle:
  `docs/reviews/bayesfilter-ledh-n10000-score-admission-repair-phase1-result-phase2-subplan-review-bundle-2026-07-09.md`

## Required Checks, Tests, And Reviews

Precheck:

```bash
python - <<'PY'
import json
from pathlib import Path
from bayesfilter.highdim.ledh_forward_contract import validate_ledh_forward_scalar_artifact, LGSSM_M3_T50_ROW_ID
value = json.loads(Path("docs/plans/ledh-phase2-lgssm-forward-scalar-artifact-2026-07-07.json").read_text())
core = validate_ledh_forward_scalar_artifact(value, expected_row_id=LGSSM_M3_T50_ROW_ID, require_admitted=True)
print(core["row_id"], core["num_particles"], core["time_steps"], core["batch_seeds"])
PY
```

Trusted GPU/full run command shape, with full logs redirected:

```bash
MPLCONFIGDIR=/tmp python docs/benchmarks/benchmark_ledh_same_target_lgssm_m3_t50_value.py \
  --num-particles 10000 \
  --time-steps 50 \
  --batch-seeds 81120,81121,81122,81123,81124 \
  --transport-policy active-all \
  --sinkhorn-iterations 10 \
  --sinkhorn-epsilon 0.5 \
  --transport-ad-mode full \
  --transport-gradient-mode manual_streaming_finite_sinkhorn_stopped_scale_keys \
  --row-chunk-size 512 \
  --col-chunk-size 512 \
  --particle-chunk-size 256 \
  --score-mode compact-sensitivity \
  --history-mode value-only \
  --dtype float32 \
  --tf32-mode enabled \
  --device /GPU:0 \
  --device-scope visible \
  --expect-device-kind gpu \
  --output docs/plans/bayesfilter-ledh-n10000-score-admission-repair-phase2-lgssm-score-artifact-2026-07-09.json \
  --markdown-output docs/plans/bayesfilter-ledh-n10000-score-admission-repair-phase2-lgssm-score-artifact-2026-07-09.md
```

Post-run validation:

```bash
python - <<'PY'
import json
from pathlib import Path
from bayesfilter.highdim.ledh_forward_contract import LGSSM_M3_T50_ROW_ID
from bayesfilter.highdim.ledh_score_contract import validate_ledh_score_artifact
value = json.loads(Path("docs/plans/ledh-phase2-lgssm-forward-scalar-artifact-2026-07-07.json").read_text())
raw = json.loads(Path("docs/plans/bayesfilter-ledh-n10000-score-admission-repair-phase2-lgssm-score-artifact-2026-07-09.json").read_text())
artifact = raw.get("score_artifact", raw)
core = validate_ledh_score_artifact(artifact, source_value_artifact=value, expected_row_id=LGSSM_M3_T50_ROW_ID, require_admitted=True)
print(core["row_id"], core["score_admission_status"], core["memory_diagnostics"].get("peak_mib"))
PY
```

Focused tests:

```bash
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest \
  tests/highdim/test_ledh_score_artifact_emitter_phase1.py \
  tests/highdim/test_ledh_lgssm_score_phase2_contract.py \
  tests/highdim/test_ledh_score_contract_phase1.py \
  tests/test_two_lane_highdim_ledh_leaderboard.py -q
```

Review:

- Read-only review of Phase 1 result and this Phase 2 subplan before running
  the trusted full command.
- Read-only review of Phase 2 result and Phase 3 subplan after the run.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Can LGSSM produce a schema-valid compact `N=10000,T=50` score artifact admitted by the shared score validator? |
| Baseline/comparator | Admitted LGSSM value artifact, July 6 raw compact score-memory JSON as non-admitted legacy evidence, and Phase 1 shared emitter/validator tests. |
| Primary criterion | The output contains a score artifact admitted by `validate_ledh_score_artifact(..., require_admitted=True)` with compact LGSSM provenance and `memory_diagnostics.n10000_memory_pass = true`. |
| Veto diagnostics | Wrong scalar; row/value mismatch; raw legacy JSON reused without schema; manual/historical route; missing memory pass; nonfinite score; FD correctness failure; CPU/sandbox result misreported as trusted GPU full run. |
| Explanatory diagnostics | Runtime, peak memory, FD errors, chunk sizes, TF32/device status, and comparison to old raw score-memory values. |
| Not concluded | Fixed-SIR or nonlinear score admission, HMC readiness, posterior correctness, scientific superiority, runtime ranking, or public benchmark readiness. |

## Forbidden Claims And Actions

- Do not use the July 6 raw score-memory JSON as admitted evidence unless it is
  first proven to contain complete schema-valid score-artifact payload, which
  Phase 1 inventory suggests it does not.
- Do not run a GPU command without trusted/escalated execution.
- Do not change LGSSM value target, theta coordinate system, batch seeds,
  particle count, or time steps.
- Do not claim non-LGSSM score readiness from this phase.
- Do not claim HMC readiness, posterior correctness, runtime ranking, or
  scientific superiority.

## Exact Next-Phase Handoff Conditions

Phase 3 may begin only if:

- LGSSM has an admitted score artifact or a blocker result explains the exact
  missing schema/runtime/memory condition;
- focused tests pass;
- the Phase 3 fixed-SIR subplan forbids reuse of historical
  `manual_total_vjp*` full evidence;
- read-only review agrees or a packet-only fallback review is recorded.

## Stop Conditions

Stop and write a blocker result if:

- trusted GPU execution is rejected or unavailable;
- LGSSM runner emits raw result without a score artifact and normalization
  cannot be done without guessing missing fields;
- validation fails for target, row, parameter, memory, or correctness reasons;
- the run OOMs or exceeds the memory budget;
- continuing requires target changes, pass/fail criterion changes, package
  installation, network/data fetches, credentials, destructive actions, or
  unrelated dirty-worktree modification.

## Skeptical Audit Before Execution

- Wrong baseline: baseline is the admitted value artifact plus July 8 blocker,
  not raw `primary_pass`.
- Proxy metric: runtime and raw memory pass cannot admit without validator.
- Missing stop condition: stop on untrusted GPU or validator failure.
- Hidden assumption: runner output must include or expose a score artifact.
- Environment mismatch: full run must be trusted GPU; CPU-only tests must hide
  GPU.
- Useless artifact risk: output JSON must contain an admitted score artifact
  or result is a blocker, not a score admission.

Audit status: ready for read-only review.
