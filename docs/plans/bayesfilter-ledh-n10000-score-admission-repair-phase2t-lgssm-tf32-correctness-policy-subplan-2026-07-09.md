# Phase 2T Subplan: LGSSM TF32 Correctness Policy Repair

Date: 2026-07-09

Status: `DRAFT_READY_FOR_REVIEW`

## Phase Objective

Resolve the Phase 2S default-TF32 correctness blocker before any full
`N=10000,T=50` LGSSM score admission run.

The phase must decide and test a correctness policy that is valid for the
compact no-tape score of the same realized finite-`N` LEDH log-likelihood
scalar.

## Entry Conditions Inherited From Previous Phase

- Score-specific memory instrumentation is implemented.
- Same-scalar FD now uses the value-only route, not the score/JVP route.
- Focused CPU tests passed.
- CPU prefix smoke passed.
- Trusted GPU TF32 smoke failed same-scalar FD at `N=256,T=3`.
- Trusted GPU no-TF32 smoke passed same-scalar FD at the same size.
- No full `N=10000,T=50` LGSSM score artifact is admitted.

## Required Artifacts

- Phase 2T result:
  `docs/plans/bayesfilter-ledh-n10000-score-admission-repair-phase2t-lgssm-tf32-correctness-policy-result-2026-07-09.md`
- Any calibration artifact:
  `docs/plans/bayesfilter-ledh-n10000-score-admission-repair-phase2t-lgssm-tf32-correctness-calibration-2026-07-09.json`
- Logs:
  `docs/plans/logs/bayesfilter-ledh-n10000-score-admission-repair-phase2t-*.log`
- Phase 2U full-run subplan if a correctness policy is selected, or blocker
  result if no policy is acceptable.

## Required Checks, Tests, And Reviews

Precheck:

```bash
rg -n "tf32-mode|score_fd_step|same_scalar_fd|score_correctness|exact_reference" \
  docs/benchmarks/benchmark_ledh_same_target_lgssm_m3_t50_value.py \
  bayesfilter/highdim/ledh_score_contract.py \
  tests/highdim/test_ledh_lgssm_score_phase2_contract.py
```

Candidate policies to evaluate:

1. **TF32 execution with no-TF32 correctness arm**:
   run the production score/value route under TF32, but compute
   same-scalar FD correctness with TF32 disabled on the same fixed finite-`N`
   random tensors and scalar. The artifact must disclose this as a correctness
   arm and must not claim TF32 FD parity.
2. **TF32 FD calibration**:
   choose a bounded FD step/tolerance based on a predeclared smoke ladder. This
   can pass only if the criterion remains sensitive enough to catch route
   mismatch; loose tolerances that would admit wrong scores are forbidden.
3. **LGSSM exact/reference correctness**:
   use a reviewed reference correctness check only if it compares the compact
   finite LEDH score to an allowed reference for the same finite target or
   clearly states the relation being certified. Exact Kalman score alone cannot
   certify the finite LEDH estimator derivative unless a derivation/review
   justifies it.

Required review:

- Read-only review before implementing or selecting a policy.
- Claude may be used only if local policy permits bounded review. Otherwise use
  fresh Codex packet review and record the limitation.

Required tests after policy implementation:

```bash
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest \
  tests/highdim/test_ledh_lgssm_score_phase2_contract.py \
  tests/test_ledh_lgssm_manual_score_phase4.py \
  tests/highdim/test_ledh_score_contract_phase1.py -q
```

Trusted GPU smoke after policy implementation:

```bash
MPLCONFIGDIR=/tmp python docs/benchmarks/benchmark_ledh_same_target_lgssm_m3_t50_value.py \
  --num-particles 256 \
  --time-steps 3 \
  --batch-seeds 81120,81121 \
  --transport-policy active-all \
  --sinkhorn-iterations 2 \
  --sinkhorn-epsilon 0.5 \
  --transport-ad-mode full \
  --transport-gradient-mode manual_streaming_finite_sinkhorn_stopped_scale_keys \
  --row-chunk-size 64 \
  --col-chunk-size 64 \
  --particle-chunk-size 64 \
  --score-mode compact-sensitivity \
  --history-mode value-only \
  --dtype float32 \
  --tf32-mode enabled \
  --device /GPU:0 \
  --device-scope visible \
  --expect-device-kind gpu \
  --output /tmp/bayesfilter-phase2t-lgssm-gpu-smoke-score.json
```

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | What correctness policy can validly certify the compact no-tape LGSSM score when default TF32 execution fails direct same-scalar FD at smoke scale? |
| Baseline/comparator | Phase 2S TF32-fail/no-TF32-pass smoke pair and Phase 1 score validator. |
| Primary criterion | A reviewed policy with tests and trusted GPU smoke that preserves the same finite target and catches wrong score routes. |
| Veto diagnostics | Loose FD tolerance that would admit wrong scores; exact Kalman overclaim; target scalar change; no-TF32 route silently replacing production TF32 without disclosure; historical route admission; no trusted GPU smoke. |
| Explanatory diagnostics | FD step ladder, TF32/no-TF32 residuals, runtime, memory, and score values. |
| Not concluded | Full LGSSM admission, non-LGSSM admission, HMC readiness, posterior correctness, runtime ranking, or scientific superiority. |

## Forbidden Claims And Actions

- Do not launch full `N=10000,T=50` score admission before the correctness
  policy passes review and smoke.
- Do not change target scalar, row id, source value artifact, parameter order,
  seeds, `N`, or `T` for full admission.
- Do not claim exact Kalman score validates finite LEDH score without reviewed
  derivation.
- Do not loosen tolerances merely to pass TF32.
- Do not admit historical/manual-total-VJP routes.

## Exact Next-Phase Handoff Conditions

Phase 2U full LGSSM run may begin only if:

- Phase 2T selects a reviewed correctness policy;
- focused tests pass;
- trusted GPU smoke passes under that policy;
- score-memory instrumentation remains active;
- the full-run subplan is written and reviewed.

## Stop Conditions

Stop if:

- no correctness policy can be justified without changing the admission
  contract;
- TF32 calibration cannot distinguish known wrong route/mismatch cases;
- review returns `REVISE` after five rounds for the same blocker;
- trusted GPU smoke fails under the selected policy;
- continuing requires product/scientific boundary changes.

## Skeptical Audit Before Execution

- Wrong baseline: do not use no-TF32 pass as proof that TF32 FD passes.
- Proxy metric: FD residual alone is not sufficient unless tied to same finite
  target and route-mismatch sensitivity.
- Hidden assumption: TF32 numerical behavior may require a separate correctness
  arm; this must be disclosed.
- Environment: GPU evidence must be trusted/escalated.
- Artifact: calibration JSON cannot be admitted as a full score artifact.

Audit status: ready for read-only review.
