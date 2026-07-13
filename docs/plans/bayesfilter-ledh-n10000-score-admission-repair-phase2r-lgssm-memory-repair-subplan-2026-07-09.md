# Phase 2R Subplan: LGSSM Full Score Memory Repair

Date: 2026-07-09

Status: `DRAFT_READY_FOR_REVIEW`

## Phase Objective

Repair the fixable Phase 2 LGSSM runtime/memory blocker and produce an
admitted compact `N=10000,T=50` LGSSM score artifact without changing the
target scalar or source value artifact.

## Entry Conditions Inherited From Previous Phase

- Phase 1 shared emitter gate passed.
- Phase 2 full LGSSM command was reviewed and launched in trusted GPU context.
- The command was interrupted after prolonged execution with no output
  artifact.
- Trusted GPU memory polling reached about `15.7 GiB`, above the previous
  `14 GiB` memory budget.
- The stack trace after interruption points into streaming transport
  value+JVP TensorArray row-block write/stack code.

## Required Artifacts

- Phase 2 blocker result:
  `docs/plans/bayesfilter-ledh-n10000-score-admission-repair-phase2-lgssm-blocker-result-2026-07-09.md`
- Phase 2R result:
  `docs/plans/bayesfilter-ledh-n10000-score-admission-repair-phase2r-lgssm-memory-repair-result-2026-07-09.md`
- Retried score artifact if successful:
  `docs/plans/bayesfilter-ledh-n10000-score-admission-repair-phase2r-lgssm-score-artifact-2026-07-09.json`
- Logs:
  `docs/plans/logs/bayesfilter-ledh-n10000-score-admission-repair-phase2r-*.log`
- Phase 3 fixed-SIR subplan if Phase 2R succeeds or a precise blocker if it
  does not.

## Required Checks, Tests, And Reviews

Precheck:

```bash
rg -n "streaming_softmin|value_ta|tangent_ta|TensorArray|row_chunk_size|col_chunk_size|particle_chunk_size" \
  experiments/dpf_implementation/tf_tfp/resampling/annealed_transport_tf.py \
  docs/benchmarks/benchmark_ledh_same_target_lgssm_m3_t50_value.py
```

First repair attempt, trusted GPU smaller chunks:

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
  --output docs/plans/bayesfilter-ledh-n10000-score-admission-repair-phase2r-lgssm-score-artifact-2026-07-09.json \
  --markdown-output docs/plans/bayesfilter-ledh-n10000-score-admission-repair-phase2r-lgssm-score-artifact-2026-07-09.md
```

Post-run validation:

```bash
python - <<'PY'
import json
from pathlib import Path
from bayesfilter.highdim.ledh_forward_contract import LGSSM_M3_T50_ROW_ID
from bayesfilter.highdim.ledh_score_contract import validate_ledh_score_artifact
value = json.loads(Path("docs/plans/ledh-phase2-lgssm-forward-scalar-artifact-2026-07-07.json").read_text())
raw = json.loads(Path("docs/plans/bayesfilter-ledh-n10000-score-admission-repair-phase2r-lgssm-score-artifact-2026-07-09.json").read_text())
artifact = raw.get("score_artifact", raw)
core = validate_ledh_score_artifact(artifact, source_value_artifact=value, expected_row_id=LGSSM_M3_T50_ROW_ID, require_admitted=True)
print(core["row_id"], core["score_admission_status"], core["memory_diagnostics"].get("peak_mib"))
PY
```

If smaller chunks still fail memory/runtime:

- stop and implement a reviewed reduce-only streaming score plan instead of
  trying arbitrary chunk ladders;
- do not change `N`, `T`, seeds, target scalar, or admission criteria.

Focused tests after any code edit:

```bash
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest \
  tests/highdim/test_ledh_score_artifact_emitter_phase1.py \
  tests/highdim/test_ledh_lgssm_score_phase2_contract.py \
  tests/highdim/test_ledh_score_contract_phase1.py -q
```

Review:

- Read-only review before the smaller-chunk retry.
- If a code repair is needed, review the code repair plan before editing.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Can the LGSSM compact score run be made to emit an admitted `N=10000,T=50` score artifact within memory budget? |
| Baseline/comparator | Phase 2 interrupted trusted run, admitted LGSSM value artifact, Phase 1 shared score contract, and old raw LGSSM memory JSON as non-admitted reference only. |
| Primary criterion | Output artifact validates with `validate_ledh_score_artifact(..., require_admitted=True)` and records `memory_diagnostics.n10000_memory_pass = true`. |
| Veto diagnostics | Wrong scalar, value mismatch, raw JSON promotion, memory over budget, no artifact, historical route, CPU/sandbox run represented as trusted GPU evidence, or arbitrary criterion changes. |
| Explanatory diagnostics | Runtime, GPU peak memory, chunk sizes, TensorArray stack behavior, FD errors, and old raw score comparison. |
| Not concluded | Non-LGSSM score admission, HMC readiness, posterior correctness, runtime ranking, or scientific superiority. |

## Forbidden Claims And Actions

- Do not change the target scalar, row id, theta coordinate system, parameter
  order, seeds, `N=10000`, or `T=50`.
- Do not raise the memory budget after seeing the over-budget observation.
- Do not admit a score without validator success.
- Do not continue arbitrary chunk sweeps without writing a new reviewed
  subplan.
- Do not claim the compact method is mathematically wrong from this runtime
  blocker.

## Exact Next-Phase Handoff Conditions

Phase 3 may begin only if:

- LGSSM score artifact is admitted by the validator, or Phase 2R writes a
  blocker with a precise reduce-only streaming implementation plan;
- local tests pass;
- the Phase 3 subplan is refreshed and explicitly forbids historical fixed-SIR
  full evidence reuse;
- read-only review agrees.

## Stop Conditions

Stop if:

- smaller chunks still exceed memory budget or no artifact is emitted in the
  reviewed time window;
- validation fails;
- repair requires changing target/admission criteria;
- trusted GPU access is unavailable;
- code repair would be broad or affect unrelated dirty worktree changes;
- review does not converge after five rounds.

## Skeptical Audit Before Execution

- Wrong baseline: Phase 2R uses the interrupted trusted run, not raw
  `primary_pass`, as baseline.
- Proxy metric: lower memory is not enough; validator admission is required.
- Hidden assumption: chunking may not fix TensorArray stacking; if so, stop for
  reduce-only implementation.
- Environment mismatch: full retry must be trusted GPU.
- Useless artifact risk: no artifact means blocker, not partial admission.

Audit status: ready for read-only review.
