# Phase 2S Subplan: LGSSM Score Procedure And Memory Repair

Date: 2026-07-09

Status: `DRAFT_READY_FOR_REVIEW`

## Phase Objective

Repair the LGSSM `N=10000,T=50` score procedure so the compact no-tape score
can be measured, serialized, correctness-checked, and admitted without changing
the target scalar or relying on historical/manual-total-VJP routes.

This phase targets mechanical/procedural blockers first. It does not change the
mathematics of the compact score.

## Entry Conditions Inherited From Previous Phase

- Phase 1 shared score artifact emitter and validator tests passed.
- Phase 2 trusted full LGSSM run emitted no score artifact and exceeded the
  reviewed memory window.
- Phase 2R smaller-chunk trusted retry emitted no score artifact.
- The admitted source value artifact remains:
  `docs/plans/ledh-phase2-lgssm-forward-scalar-artifact-2026-07-07.json`.
- The target score remains the derivative of the same realized finite-`N`
  `observed_data_log_likelihood_estimator`, output field `log_likelihood`, for
  row `benchmark_lgssm_exact_oracle_m3_T50`.

## Required Artifacts

- Phase 2S result:
  `docs/plans/bayesfilter-ledh-n10000-score-admission-repair-phase2s-lgssm-score-procedure-repair-result-2026-07-09.md`
- Repaired full score artifact if successful:
  `docs/plans/bayesfilter-ledh-n10000-score-admission-repair-phase2s-lgssm-score-artifact-2026-07-09.json`
- Markdown companion if successful:
  `docs/plans/bayesfilter-ledh-n10000-score-admission-repair-phase2s-lgssm-score-artifact-2026-07-09.md`
- Logs:
  `docs/plans/logs/bayesfilter-ledh-n10000-score-admission-repair-phase2s-*.log`
- Tests or focused diagnostics covering:
  - value-only FD scalar parity against the existing compact score objective on
    tiny/prefix cases;
  - score-specific memory diagnostics are used for score admission;
  - blocked score-only artifacts cannot be admitted without correctness;
  - full artifacts still require validator admission.
- Phase 3 fixed-SIR subplan if LGSSM is admitted, or a precise blocker with the
  next smallest implementation repair if it is not.

## Required Checks, Tests, And Reviews

Pre-edit inspection:

```bash
rg -n "score-mode|_manual_score_diagnostic|_compact_value_and_score_from_components|gpu_memory_info_after|score_gpu_memory" \
  docs/benchmarks/benchmark_ledh_same_target_lgssm_m3_t50_value.py \
  tests/highdim/test_ledh_lgssm_score_phase2_contract.py \
  tests/test_ledh_compact_transport_jvp.py
```

Required code repair scope:

1. In `docs/benchmarks/benchmark_ledh_same_target_lgssm_m3_t50_value.py`,
   split the compact score diagnostic so coordinate-wise finite differences use
   a value-only forward scalar route, not `_compact_value_and_score_from_components`.
2. Add score-specific GPU memory measurement around the score diagnostic. Reset
   TensorFlow GPU memory stats before the score run when available, record
   `score_gpu_memory_info_before/after`, and make score artifact memory
   diagnostics read the score-specific peak when score mode is active.
3. Add an explicit blocked score-only serialization path only if needed for
   debugging. It must use
   `score_admission_status = tiny_score_diagnostic_not_admitted` or
   `blocked_score_not_run`, never full admission.
4. Do not change the transport target, row id, seeds, `N`, `T`, theta
   coordinate system, or admitted value artifact.

Focused local tests after edits:

```bash
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest \
  tests/highdim/test_ledh_lgssm_score_phase2_contract.py \
  tests/test_ledh_compact_transport_jvp.py \
  tests/highdim/test_ledh_score_artifact_emitter_phase1.py \
  tests/highdim/test_ledh_score_contract_phase1.py -q
```

Prefix smoke before full run:

```bash
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python docs/benchmarks/benchmark_ledh_same_target_lgssm_m3_t50_value.py \
  --num-particles 64 \
  --time-steps 3 \
  --batch-seeds 81120,81121 \
  --transport-policy active-all \
  --sinkhorn-iterations 2 \
  --sinkhorn-epsilon 0.5 \
  --transport-ad-mode full \
  --transport-gradient-mode manual_streaming_finite_sinkhorn_stopped_scale_keys \
  --row-chunk-size 16 \
  --col-chunk-size 16 \
  --particle-chunk-size 16 \
  --score-mode compact-sensitivity \
  --history-mode value-only \
  --dtype float32 \
  --tf32-mode enabled \
  --device /CPU:0 \
  --device-scope cpu \
  --expect-device-kind cpu \
  --output /tmp/bayesfilter-phase2s-lgssm-prefix-score.json
```

Trusted full retry only after tests and prefix smoke pass:

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
  --row-chunk-size 128 \
  --col-chunk-size 128 \
  --particle-chunk-size 128 \
  --score-mode compact-sensitivity \
  --history-mode value-only \
  --dtype float32 \
  --tf32-mode enabled \
  --device /GPU:0 \
  --device-scope visible \
  --expect-device-kind gpu \
  --output docs/plans/bayesfilter-ledh-n10000-score-admission-repair-phase2s-lgssm-score-artifact-2026-07-09.json \
  --markdown-output docs/plans/bayesfilter-ledh-n10000-score-admission-repair-phase2s-lgssm-score-artifact-2026-07-09.md
```

Post-run validation:

```bash
python - <<'PY'
import json
from pathlib import Path
from bayesfilter.highdim.ledh_forward_contract import LGSSM_M3_T50_ROW_ID
from bayesfilter.highdim.ledh_score_contract import validate_ledh_score_artifact
value = json.loads(Path("docs/plans/ledh-phase2-lgssm-forward-scalar-artifact-2026-07-07.json").read_text())
raw = json.loads(Path("docs/plans/bayesfilter-ledh-n10000-score-admission-repair-phase2s-lgssm-score-artifact-2026-07-09.json").read_text())
artifact = raw.get("score_artifact", raw)
core = validate_ledh_score_artifact(artifact, source_value_artifact=value, expected_row_id=LGSSM_M3_T50_ROW_ID, require_admitted=True)
print(core["row_id"], core["score_admission_status"], core["memory_diagnostics"].get("peak_mib"))
PY
```

Review:

- Claude read-only review may be used only if the local policy permits the
  bounded review bundle. If Claude is blocked by the external-disclosure gate,
  use a fresh Codex packet-only review and record the limitation.
- Review must occur before code edits and again before any full trusted run if
  the plan changes materially.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Can LGSSM score admission be unblocked by repairing score procedure, score memory measurement, and value-only same-scalar FD checks before another full `N=10000,T=50` run? |
| Baseline/comparator | Phase 2 and 2R no-artifact runs; admitted LGSSM value artifact; Phase 1 validator; existing compact transport JVP tiny tests. |
| Primary criterion | Full Phase 2S artifact validates with `validate_ledh_score_artifact(..., require_admitted=True)` and reports score-specific `memory_diagnostics.n10000_memory_pass = true`. |
| Veto diagnostics | Wrong target scalar; historical route admitted; FD uses score/JVP route instead of value-only scalar route; score memory measured only around value route; raw legacy JSON promoted; no artifact; nonfinite score; correctness failure; trusted GPU evidence absent. |
| Explanatory diagnostics | Runtime, score GPU peak memory, value GPU peak memory, chunk sizes, FD residuals, and prefix smoke performance. |
| Not concluded | HMC readiness, posterior correctness, exact Kalman score equality, runtime ranking, scientific superiority, or non-LGSSM row admission. |

## Forbidden Claims And Actions

- Do not change the target scalar away from
  `observed_data_log_likelihood_estimator`.
- Do not change `row_id`, seeds, `N=10000`, `T=50`, theta coordinate system, or
  parameter order for full admission.
- Do not admit score-only artifacts without correctness and score-memory pass.
- Do not use `GradientTape`, `ForwardAccumulator`, or stopped partial
  derivatives in the production score route.
- Do not revive `manual_total_vjp*` or historical manual-reverse evidence as
  admission evidence.
- Do not raise the 14 GiB score memory budget after seeing results.
- Do not claim a TensorArray/stack transport rewrite is necessary until the
  cheaper procedural fixes and score-specific memory diagnostics have been
  tested.

## Exact Next-Phase Handoff Conditions

Phase 3 fixed-SIR may begin only if:

- LGSSM score is validator-admitted, or Phase 2S writes a blocker result with a
  precise next implementation step;
- focused local tests pass;
- any full GPU claim is backed by trusted GPU execution;
- the ledger is updated;
- the Phase 3 subplan is refreshed and reviewed.

## Stop Conditions

Stop if:

- focused local tests fail and the fix is not obvious;
- prefix smoke fails target/scalar/FD correctness;
- full trusted run exceeds the reviewed time or memory stop window;
- score artifact validation fails;
- repair requires target/admission-contract changes;
- trusted GPU access is unavailable;
- review does not converge after five rounds.

## Skeptical Audit Before Execution

- Wrong baseline risk: Phase 2S uses Phase 2/2R failed runs and admitted value
  artifact, not raw legacy JSON, as baseline.
- Proxy metric risk: lower runtime or lower memory alone cannot admit a score;
  validator admission is required.
- Hidden assumption risk: value-only FD may reduce runtime but not peak score
  memory; if so, stop for a deeper transport tensor-lifetime plan.
- Environment risk: CPU prefix smoke can check mechanics only; full evidence
  still requires trusted GPU.
- Artifact risk: blocked/debug artifacts must remain non-admitted.

Audit status: ready for read-only review.
