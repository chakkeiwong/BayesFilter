# Phase 5 Result: Actual-SV Score

metadata_date: 2026-07-07
status: `TINY_DIAGNOSTIC_PASSED_FULL_SCORE_NOT_ADMITTED`
master_program: `docs/plans/bayesfilter-ledh-score-per-model-master-program-2026-07-07.md`
phase: 5

## Phase Objective

Build and admit, or explicitly block, the LEDH score for:

```text
zhao_cui_sv_actual_nongaussian_T1000
```

The score target is the no-tape total derivative of the same realized
finite-`N` LEDH scalar:

```text
observed_data_log_likelihood_estimator
```

reported as:

```text
log_likelihood
```

with target observation policy:

```text
transformed_actual_sv_log_y_square
```

and score parameter order:

```text
gamma_unconstrained, log_beta
```

## Decision Table

| Field | Result |
| --- | --- |
| Decision | Actual-SV tiny no-tape same-target score diagnostic passed; full actual-SV score is not admitted. |
| Primary criterion status | Partially met only for tiny diagnostic: all two coordinate FD checks pass at `T=2,N=64`, but no `N=10000,T=1000` score/memory artifact exists. |
| Veto diagnostic status | Full admission is vetoed by absent full-row score/memory evidence. |
| Main uncertainty | Need a reviewed full-row score/memory plan that preserves the no-tape total route without storing an unsafe full history. |
| Next justified action | Draft and review a full-row actual-SV score/memory subplan before any full `N=10000,T=1000` score run. |
| What is not concluded | No full actual-SV score admission; no HMC readiness, posterior correctness, exact raw-observation likelihood, KSC score admission, generalized-SV score admission, runtime ranking, or scientific superiority claim. |

## What Was Implemented

Added:

- `docs/benchmarks/benchmark_ledh_same_target_actual_sv_score.py`
- `tests/highdim/test_ledh_actual_sv_score_phase5_contract.py`

The score route is:

```text
manual_total_vjp_no_autodiff_same_scalar_actual_sv_ledh_pfpf_ot
```

It uses the exact transformed actual-SV correction:

```text
transition_log_density + observation_log_density
- pre_flow_log_density + forward_log_det
```

where the observation density is the exact transformed log-chi-square density
for:

```text
z_t = log(y_t^2),    z_t - 2 log(beta) - x_t ~ log(chi_square_1)
```

The route computes a bounded finite-`N` manual VJP through:

- stationary initial proposal variance;
- transition proposal mean;
- exact transformed target transition and observation density terms;
- affine LEDH flow with manual aux/VJP;
- streaming finite Sinkhorn transport manual VJP;
- the same normalized log-weight scalar used by the value route.

## Tiny Diagnostic Artifact

Command:

```text
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python \
  docs/benchmarks/benchmark_ledh_same_target_actual_sv_score.py \
  --source-value-artifact docs/plans/ledh-phase5-actual-sv-forward-scalar-artifact-2026-07-07.json \
  --output docs/plans/bayesfilter-ledh-score-per-model-phase5-actual-sv-tiny-score-diagnostic-2026-07-07.json \
  --markdown-output docs/plans/bayesfilter-ledh-score-per-model-phase5-actual-sv-tiny-score-diagnostic-2026-07-07.md \
  --batch-seeds 81120 \
  --time-steps 2 \
  --num-particles 64 \
  --sinkhorn-iterations 2 \
  --row-chunk-size 64 \
  --col-chunk-size 64 \
  --particle-chunk-size 64 \
  --dtype float64 \
  --tf32-mode disabled \
  --fd-step 1.0e-4
```

Result:

- artifact:
  `docs/plans/bayesfilter-ledh-score-per-model-phase5-actual-sv-tiny-score-diagnostic-2026-07-07.json`;
- summary:
  `docs/plans/bayesfilter-ledh-score-per-model-phase5-actual-sv-tiny-score-diagnostic-2026-07-07.md`;
- `score_admission_status = tiny_score_diagnostic_not_admitted`;
- `score_correctness.status = pass`;
- `score = [-0.13676940977770666, 0.38478205551642064]`;
- finite-difference score:
  `[-0.13676941023277323, 0.3847820563884774]`;
- `max_abs_error = 8.720567601372409e-10`;
- `max_rel_error = 3.3272540035606193e-09`;
- `memory_diagnostics.n10000_memory_pass = false`.

## Local Checks

Compile:

```text
python -m py_compile \
  docs/benchmarks/benchmark_ledh_same_target_actual_sv_score.py \
  tests/highdim/test_ledh_actual_sv_score_phase5_contract.py
```

Result:

```text
passed
```

Focused Phase 5 score tests:

```text
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest \
  tests/highdim/test_ledh_actual_sv_score_phase5_contract.py -q
```

Result:

```text
7 passed, 2 warnings
```

Combined Phase 5 replay/schema checks:

```text
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest \
  tests/highdim/test_ledh_actual_sv_score_phase5_contract.py \
  tests/highdim/test_ledh_phase5_actual_sv_forward_scalar_artifact.py \
  tests/highdim/test_ledh_score_contract_phase1.py -q
```

Result:

```text
28 passed, 2 warnings
```

Static no-autodiff token check:

```text
rg -n "GradientTape|ForwardAccumulator|stop_gradient|stopped_partial" \
  docs/benchmarks/benchmark_ledh_same_target_actual_sv_score.py \
  tests/highdim/test_ledh_actual_sv_score_phase5_contract.py
```

Result:

- no positive autodiff route use found;
- only negative artifact fields such as
  `uses_stopped_partial_derivative: False` were found.

## Evidence Contract Result

| Field | Result |
| --- | --- |
| Question | Can actual-SV produce a no-tape total derivative of the same finite-`N` exact transformed LEDH `log_likelihood` scalar admitted by the value artifact? |
| Answer | Yes at bounded tiny diagnostic scale. Not yet at full `N=10000,T=1000` admission scale. |
| Baseline/comparator | Admitted actual-SV value artifact, exact transformed log-square target density, same-scalar coordinate FD with fixed randomness, and Phase 1 score artifact validator. |
| Primary criterion | Tiny diagnostic passed; full admission failed/not met because no full score/memory artifact exists. |
| Veto diagnostics | Full admission vetoed by absent full-row memory/correctness evidence. |
| Explanatory diagnostics | FD errors, dtype, CPU-hidden execution, runtime, and score values. |
| Artifact | This result, the tiny JSON/Markdown artifact, score adapter, and tests. |

## Next Handoff

The next step is a dedicated reviewed subplan for full-row actual-SV
score/memory execution:

- `docs/plans/bayesfilter-ledh-score-per-model-phase5-actual-sv-full-row-score-subplan-2026-07-07.md`

That subplan must decide how to attempt full `N=10000,T=1000` score evidence
without unsafe full-history storage and without changing the target scalar,
parameter order, or no-tape total derivative route.

## Nonclaims

- Full actual-SV score is not admitted.
- Tiny FD correctness is not `N=10000,T=1000` score admission.
- This result does not claim KSC surrogate likelihood evidence.
- This result does not claim raw Gaussian observation likelihood evidence.
- This result does not claim augmented-noise Gaussian-closure evidence.
- This result does not claim HMC readiness, posterior correctness, runtime
  ranking, scientific superiority, or all-algorithm comparison.
