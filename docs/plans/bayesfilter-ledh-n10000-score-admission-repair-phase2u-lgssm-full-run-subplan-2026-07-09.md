# Phase 2U Subplan: LGSSM Full N10000 Score Admission Run

Date: 2026-07-09

Status: `DRAFT_READY_FOR_REVIEW`

## Phase Objective

Run the full LGSSM `N=10000,T=50` compact score admission attempt using the
Phase 2S score-procedure repair and Phase 2T disclosed no-TF32 correctness arm.

## Entry Conditions Inherited From Previous Phase

- Phase 1 shared score artifact validator passed.
- Phase 2S repaired value-only FD and score-specific memory measurement.
- Phase 2T selected and smoke-tested the correctness policy:
  production TF32 enabled, FD correctness arm TF32 disabled and disclosed.
- Trusted GPU smoke at `N=256,T=3` passed under the selected policy.
- No full LGSSM score artifact is admitted yet.

## Required Artifacts

- Full score artifact:
  `docs/plans/bayesfilter-ledh-n10000-score-admission-repair-phase2u-lgssm-score-artifact-2026-07-09.json`
- Markdown companion:
  `docs/plans/bayesfilter-ledh-n10000-score-admission-repair-phase2u-lgssm-score-artifact-2026-07-09.md`
- Log:
  `docs/plans/logs/bayesfilter-ledh-n10000-score-admission-repair-phase2u-lgssm-full-run-2026-07-09.log`
- Phase 2U result:
  `docs/plans/bayesfilter-ledh-n10000-score-admission-repair-phase2u-lgssm-full-run-result-2026-07-09.md`
- Phase 3 fixed-SIR subplan if admitted, or blocker result if not.

## Required Checks, Tests, And Reviews

Pre-run focused tests:

```bash
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest \
  tests/highdim/test_ledh_lgssm_score_phase2_contract.py \
  tests/test_ledh_lgssm_manual_score_phase4.py \
  tests/highdim/test_ledh_score_contract_phase1.py -q
```

Trusted full run:

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
  --score-fd-tf32-mode disabled \
  --history-mode value-only \
  --dtype float32 \
  --tf32-mode enabled \
  --device /GPU:0 \
  --device-scope visible \
  --expect-device-kind gpu \
  --output docs/plans/bayesfilter-ledh-n10000-score-admission-repair-phase2u-lgssm-score-artifact-2026-07-09.json \
  --markdown-output docs/plans/bayesfilter-ledh-n10000-score-admission-repair-phase2u-lgssm-score-artifact-2026-07-09.md
```

Post-run validator:

```bash
python - <<'PY'
import json
from pathlib import Path
from bayesfilter.highdim.ledh_forward_contract import LGSSM_M3_T50_ROW_ID
from bayesfilter.highdim.ledh_score_contract import validate_ledh_score_artifact
value = json.loads(Path("docs/plans/ledh-phase2-lgssm-forward-scalar-artifact-2026-07-07.json").read_text())
raw = json.loads(Path("docs/plans/bayesfilter-ledh-n10000-score-admission-repair-phase2u-lgssm-score-artifact-2026-07-09.json").read_text())
artifact = raw.get("score_artifact", raw)
core = validate_ledh_score_artifact(artifact, source_value_artifact=value, expected_row_id=LGSSM_M3_T50_ROW_ID, require_admitted=True)
print(core["row_id"], core["score_admission_status"], core["memory_diagnostics"].get("peak_mib"), core["score_correctness"].get("tf32_mode"))
PY
```

Review:

- Read-only review of this subplan before full run.
- Claude may be used only if local policy permits bounded review. Otherwise
  use fresh Codex packet review and record the limitation.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Can LGSSM emit a validator-admitted compact `N=10000,T=50` score artifact under the repaired score procedure and reviewed correctness policy? |
| Baseline/comparator | Phase 2S/2T smoke gates and admitted LGSSM value artifact. |
| Primary criterion | Output nested `score_artifact` validates with `validate_ledh_score_artifact(..., require_admitted=True)` and reports score-specific `n10000_memory_pass = true`. |
| Veto diagnostics | No artifact; validator failure; FD correctness fail; score-memory peak above 14 GiB; raw top-level status admitted while nested artifact is not; hidden no-TF32 substitution; target/row/parameter mismatch; historical route. |
| Explanatory diagnostics | Runtime, value memory, score memory, FD residuals, precision-arm metadata, chunk sizes. |
| Not concluded | Non-LGSSM admission, HMC readiness, posterior correctness, exact Kalman score equality, runtime ranking, or scientific superiority. |

## Forbidden Claims And Actions

- Do not admit if nested `score_artifact` is absent or validator-blocked.
- Do not hide that FD correctness uses `--score-fd-tf32-mode disabled`.
- Do not change target scalar, row id, source value artifact, seeds, `N`, `T`,
  theta coordinate system, or parameter order.
- Do not raise memory budget after seeing results.
- Do not continue arbitrary full reruns if this run fails.

## Exact Next-Phase Handoff Conditions

Phase 3 fixed-SIR may begin only if:

- LGSSM score is admitted, or Phase 2U writes a precise blocker result;
- result and ledger are updated;
- Phase 3 subplan is refreshed and reviewed.

## Stop Conditions

Stop if:

- full command exceeds reviewed runtime window without artifact;
- GPU memory exceeds budget;
- FD correctness fails;
- score artifact validation fails;
- trusted GPU access is unavailable;
- hidden precision-arm or target mismatch is detected;
- review does not converge after five rounds.

## Skeptical Audit Before Execution

- Wrong baseline: this compares to Phase 2S/2T smoke and admitted value
  artifact, not raw legacy score-memory JSON.
- Proxy metric: full run completion is insufficient; validator admission is
  required.
- Hidden assumption: no-TF32 FD arm certifies a stable same-scalar check, not
  direct TF32 FD parity; metadata must disclose this.
- Environment: GPU evidence must be trusted.
- Artifact: top-level raw status cannot override nested Phase 1 validation.

Audit status: ready for read-only review.
