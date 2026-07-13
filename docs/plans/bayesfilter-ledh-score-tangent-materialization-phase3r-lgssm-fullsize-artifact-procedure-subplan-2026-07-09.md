# Phase 3R Subplan: LGSSM Full-Size Artifact Procedure Repair

Date: 2026-07-09

Status: `READY_FOR_REVIEW_BEFORE_EXECUTION`

## Phase Objective

Fix the full-size LGSSM score diagnostic procedure so every `N=10000,T=50,
Sinkhorn=10` score-only run emits either a valid terminal JSON score artifact
or a structured terminal failure/blocker artifact with enough runtime, process,
and GPU memory evidence to identify the next blocker. Intermediate progress
records are useful observability evidence, but they are not terminal artifacts
and do not satisfy this phase.

This phase must not change the finite-`N` LEDH target scalar, parameter order,
transport settings, seeds, score route, score admission criteria, or memory
budget.

## Entry Conditions Inherited From Previous Phase

- The Phase 3 same-points softmin VJP repair passed focused CPU-hidden tests.
- Fresh Codex read-only review returned `VERDICT: AGREE`.
- Trusted GPU `N=1000,T=10,Sinkhorn=10` score-only emitted under budget:
  `score_peak_mib = 262.915283203125`, budget `14000.0`.
- Trusted GPU `N=10000,T=50,Sinkhorn=10` was attempted only after the blocker
  rung passed, but no output artifact was written.
- Full score admission remains blocked because all current large-rung evidence
  is score-only and missing same-scalar finite-difference correctness.

## Required Artifacts

- Implementation diff, expected primary file:
  `docs/benchmarks/benchmark_ledh_same_target_lgssm_m3_t50_value.py`
- Tests, if the runner has testable helper extraction:
  `tests/test_ledh_lgssm_manual_score_phase4.py` or a new focused runner
  artifact-procedure test.
- Tiny diagnostic artifact or failure artifact under:
  `docs/plans/artifacts/`
- Phase 3R result:
  `docs/plans/bayesfilter-ledh-score-tangent-materialization-phase3r-lgssm-fullsize-artifact-procedure-result-2026-07-09.md`

## Required Checks, Tests, And Reviews

Review:

- Claude is not retried for this program unless the human explicitly requests a
  new approval path; the previous Claude artifact-disclosure approval was
  rejected by the approval reviewer.
- Use fresh Codex read-only review for the implementation and test plan before
  rerunning the full-size GPU command.

Pre-implementation trace:

1. Trace `benchmark_ledh_same_target_lgssm_m3_t50_value.py` from argument
   parse to final artifact write.
2. Identify where value execution, manual score execution, memory telemetry,
   JSON serialization, and file writing occur.
3. Determine whether current artifact writing happens only after the entire
   score path completes.
4. Locate any broad exception handling, `sys.exit`, TensorFlow async behavior,
   or cleanup paths that can bypass artifact writing.

Implementation requirements:

1. Add or reuse a single artifact-emission helper that writes atomically:
   temporary file in the output directory, then rename to the requested output.
2. Emit progress/failure records at minimum:
   - `started`;
   - `value_completed`;
   - `score_started`;
   - `score_completed`;
   - `failed_exception`;
   - `completed`.
3. Every progress or terminal record must include:
   - `artifact_status`;
   - `terminal_artifact` boolean;
   - process PID;
   - stage start timestamp;
   - current stage timestamp;
   - last completed stage;
   - elapsed seconds since runner start;
   - command-shape metadata (`N`, `T`, seed list, Sinkhorn iterations, chunks).
4. Failure records must include:
   - command-shape metadata (`N`, `T`, seed list, Sinkhorn iterations, chunks);
   - target identity metadata available before score execution;
   - Python exception type/message/traceback when available;
   - TensorFlow GPU memory info when available;
   - process PID and elapsed seconds;
   - whether score execution began;
   - whether score result was finite if available;
   - nonclaims and score-admission status.
5. The final success artifact must remain backward-compatible with current
   downstream readers.
6. Do not turn progress records into admitted score artifacts.
7. If the implementation cannot convert process death below Python exception
   handling into a terminal artifact from inside the runner, the result phase
   must classify a leftover progress-only artifact as
   `BLOCK_INCOMPLETE_PROGRESS_ARTIFACT` and hand off to a wrapper-side
   finalizer/supervisor.

CPU-hidden checks after implementation:

```bash
python -m py_compile docs/benchmarks/benchmark_ledh_same_target_lgssm_m3_t50_value.py
```

```bash
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest \
  tests/test_ledh_lgssm_manual_score_phase4.py \
  tests/highdim/test_ledh_lgssm_score_phase2_contract.py \
  tests/highdim/test_ledh_score_contract_phase1.py -q
```

If helper extraction adds a new artifact-procedure unit test, include it in the
test command.

Trusted GPU smoke after review:

Run a tiny score-only command that exercises the same artifact emission helper,
using `N=256,T=3,Sinkhorn=2`, and verify the artifact contains a completed
status and memory fields.

Trusted GPU full-size rerun only after smoke passes:

```bash
MPLCONFIGDIR=/tmp python docs/benchmarks/benchmark_ledh_same_target_lgssm_m3_t50_value.py \
  --num-particles 10000 \
  --time-steps 50 \
  --batch-seeds 81120 \
  --transport-policy active-all \
  --sinkhorn-iterations 10 \
  --sinkhorn-epsilon 0.5 \
  --annealed-scaling 0.9 \
  --annealed-convergence-threshold 0.001 \
  --transport-ad-mode full \
  --transport-gradient-mode manual_streaming_finite_sinkhorn_stopped_scale_keys \
  --row-chunk-size 128 \
  --col-chunk-size 128 \
  --particle-chunk-size 128 \
  --score-mode manual-reverse \
  --score-diagnostic-stage score-only \
  --history-mode value-only \
  --warmups 0 \
  --repeats 1 \
  --dtype float32 \
  --tf32-mode enabled \
  --device /GPU:0 \
  --device-scope visible \
  --expect-device-kind gpu \
  --output docs/plans/artifacts/ledh-score-tangent-materialization-phase3r-lgssm-score-only-n10000-t50-seed81120-2026-07-09.json
```

## Evidence Contract

| Field | Contract |
| --- | --- |
| Engineering question | Can the LGSSM full-size score diagnostic reliably emit a success or failure artifact so the next blocker is observable? |
| Baseline/comparator | Phase 3 full-size attempt exited anomalously with no requested artifact. |
| Primary criterion | Tiny GPU smoke and full-size single-seed run each leave a terminal structured artifact at the requested output path: either `terminal_artifact=true` with `artifact_status=completed`, or `terminal_artifact=true` with `artifact_status` beginning `failed_` or `blocked_`. |
| Correctness criterion | Runner emits the same final success schema as before when successful; CPU-hidden tests still pass. |
| Veto diagnostics | Target scalar drift; parameter-order drift; exact Kalman score substitution; score admission claim from score-only run; failure artifact missing exception/runtime/memory/process/stage fields; progress-only nonterminal artifact at process exit; no artifact. |
| Explanatory diagnostics | Runtime stage, PID, start/end timestamps, elapsed seconds, process/session state when externally observable, TensorFlow GPU current/peak bytes, score-started flag, score-completed flag, traceback if any. |
| Not concluded | Full score admission, leaderboard completion, HMC readiness, posterior correctness, or scientific superiority. |

## Forbidden Claims And Actions

- Do not claim score admission from a progress, failure, smoke, or score-only
  artifact.
- Do not change `N`, `T`, chunks, seeds, transport policy, Sinkhorn settings, or
  memory budget for the full-size rerun after seeing results.
- Do not use exact Kalman score or any surrogate target as the LEDH score.
- Do not run multi-seed full leaderboard rows in this phase.
- Do not classify no-artifact failure as a mathematical score failure without a
  failure artifact proving that path.

## Exact Next-Phase Handoff Conditions

If the full-size single-seed run emits a terminal success artifact under
budget, hand off to the score+FD aggregation/admission re-entry subplan. If it
emits a terminal failure/blocker artifact, hand off to the smallest repair named
by that artifact. If it leaves only a nonterminal progress artifact, stop with
`BLOCK_INCOMPLETE_PROGRESS_ARTIFACT` unless a wrapper-side finalizer has already
converted it into a terminal blocker artifact. If it still emits no artifact,
stop with `BLOCK_ARTIFACT_PROCEDURE_UNREPAIRED`.

## Stop Conditions

Stop if review finds an unpatched material flaw, CPU-hidden tests fail, tiny GPU
smoke emits no artifact, tiny GPU smoke emits only nonterminal progress, the
full-size rerun emits no artifact, the full-size rerun leaves only nonterminal
progress, GPU memory exceeds the reviewed budget, or continuing would require
package installation, network/data fetches, credentials, destructive git
actions, or changing pass/fail criteria after seeing results.

## Skeptical Audit Before Execution

- Wrong baseline checked: this phase compares against the no-artifact full-size
  attempt, not against exact Kalman likelihood.
- Proxy metric checked: artifact emission is procedural observability, not score
  admission.
- Hidden assumption checked: Phase 3 repaired the original Sinkhorn reverse
  lifetime blocker at `N=1000,T=10,Sinkhorn=10`; this phase targets the new
  full-size no-artifact behavior.
- Environment checked: GPU commands require trusted/escalated execution.
- Artifact sufficiency checked: both success and failure paths must write
  terminal JSON; progress-only JSON is a blocker, not a pass.

Audit status: `PASS_FOR_CODEX_REVIEW_BEFORE_IMPLEMENTATION`.
