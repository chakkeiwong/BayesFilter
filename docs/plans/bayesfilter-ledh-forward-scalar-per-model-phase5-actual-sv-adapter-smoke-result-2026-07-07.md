# Phase 5 Repair Result: Exact Transformed Actual-SV Adapter Smoke

metadata_date: 2026-07-07
status: `PASSED_TINY_ADAPTER_SMOKE_NO_FULL_ROW_ADMISSION`
subplan: `docs/plans/bayesfilter-ledh-forward-scalar-per-model-phase5-actual-sv-adapter-smoke-subplan-2026-07-07.md`

## Decision Table

| Field | Result |
| --- | --- |
| Decision | Tiny exact transformed actual-SV LEDH adapter smoke passed. |
| Primary criterion status | Passed for tiny executable artifact only. |
| Veto diagnostic status | No full-row run, no full-row admission, no raw Gaussian target correction, no KSC substitution, no augmented-noise Gaussian closure, no score work. |
| Main uncertainty | This is not a full `N=10000,T=1000` actual-SV admission artifact. Full-row runtime/memory and Monte Carlo behavior remain unchecked. |
| Next justified action | Review this repair result, then plan the full actual-SV row run as a separate gated Phase 5 step. |
| What is not concluded | No actual-SV full-row admission, score admission, score correctness, generalized-SV admission, KSC admission, HMC readiness, posterior correctness, scientific superiority, or runtime ranking. |

## Evidence Contract Result

| Field | Result |
| --- | --- |
| Question | Can a tiny actual-SV LEDH adapter execute the exact transformed observed-data target correction before a full-row run? |
| Answer | Yes, for a tiny `T=4,N=128,seed=81120` smoke artifact. |
| Target scalar | `observed_data_log_likelihood_estimator`, reported as `log_likelihood`. |
| Target correction | Initial stationary SV prior at `t=0`; SV transition density for `t>0`; exact `exact_log_chi_square_log_density(z_t - 2log(beta) - x_t)` observation density for all `t`; minus LEDH proposal `pre_flow_log_density`; plus `forward_log_det`. |
| Artifact | `docs/plans/ledh-phase5-actual-sv-forward-scalar-tiny-smoke-artifact-2026-07-07.json` |

## Artifact Summary

Tiny artifact:

- row id: `zhao_cui_sv_actual_nongaussian_T1000`;
- admission status: `tiny_executed_not_full_row`;
- theta coordinate: `synthetic_unconstrained`;
- theta values: `[0.2533471031357997, -0.916290731874155]`;
- time steps: `4`;
- particles: `128`;
- batch seeds: `[81120]`;
- target observation policy: `transformed_actual_sv_log_y_square`;
- flow observation policy: `gaussianized_exact_log_square_actual_sv_flow_observation`;
- target observation density: `exact_log_chi_square_log_density`;
- transform offset: `0.0`;
- output device: `/job:localhost/replica:0/task:0/device:GPU:0`;
- `log_likelihood_by_seed`: `[-7.566841125488281]`;
- `average_log_likelihood_by_seed`: `[-1.8917102813720703]`;
- finite output: `true`.

## Commands Run

Plan review:

```text
bash /home/chakwong/python/claudecodex/scripts/claude_review_gate.sh ...
```

Result:

```text
REVIEW_STATUS=probe_timeout
VERDICT=NONE
```

Tiny direct Claude probe:

```text
claude -p "Return exactly CLAUDE_PROBE_OK."
```

Result:

```text
CLAUDE_PROBE_OK
```

Narrow direct Claude read-only plan review:

```text
claude -p "READ-ONLY REVIEW ONLY. ... End with exactly VERDICT: AGREE or VERDICT: REVISE."
```

Result:

```text
VERDICT: AGREE
```

Compile check:

```text
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m py_compile \
  docs/benchmarks/benchmark_ledh_same_target_actual_sv_value.py
```

Result: passed.

Trusted GPU tiny smoke:

```text
MPLCONFIGDIR=/tmp python docs/benchmarks/benchmark_ledh_same_target_actual_sv_value.py \
  --time-steps 4 \
  --num-particles 128 \
  --batch-seeds 81120 \
  --transport-policy active-all \
  --sinkhorn-iterations 2 \
  --row-chunk-size 64 \
  --col-chunk-size 64 \
  --particle-chunk-size 64 \
  --history-mode full \
  --warmups 0 \
  --repeats 1 \
  --device /GPU:0 \
  --expect-device-kind gpu \
  --output docs/plans/ledh-phase5-actual-sv-forward-scalar-tiny-smoke-artifact-2026-07-07.json \
  --markdown-output docs/plans/ledh-phase5-actual-sv-forward-scalar-tiny-smoke-artifact-2026-07-07.md
```

Result: passed; XLA compiled on GPU.

Focused replay checks:

```text
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest \
  tests/highdim/test_ledh_forward_scalar_admission_guard.py \
  tests/highdim/test_ledh_phase5_actual_sv_forward_scalar_tiny_artifact.py -q
```

Result:

```text
15 passed, 2 warnings in 2.83s
```

## Implementation Notes

- Added `docs/benchmarks/benchmark_ledh_same_target_actual_sv_value.py`.
- Added `tests/highdim/test_ledh_phase5_actual_sv_forward_scalar_tiny_artifact.py`.
- The runner rejects accidental full-row settings in this subphase.
- The replay test validates the tiny artifact without admission and confirms
  that `require_admitted=True` rejects it.
- Proposal noise is fixed per batch seed and time index.
- LEDH flow uses a Gaussianized observation proposal surface; this surface is
  not the target likelihood.
- Target correction uses the exact transformed actual-SV log-chi-square
  observation density.

## Nonclaims

- This result does not admit the full actual-SV row.
- This result does not implement or admit scores.
- This result does not establish score correctness.
- This result does not admit generalized SV or KSC SV.
- This result does not establish HMC readiness, posterior correctness,
  scientific superiority, or runtime ranking.

## Next Handoff

The next Phase 5 step may plan the full actual-SV `N=10000,T=1000` run using
the exact transformed adapter route implemented here, but only after a
dedicated full-row subplan/review gate. Full-row execution remains forbidden by
this repair subphase.
