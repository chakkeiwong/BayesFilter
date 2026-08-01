# One-Seed UKF / SGQF / Zhao-Cui / GenUT Feasibility Plan

Date: 2026-07-22  
Status: `completed_attempt03`

Terminal result: `docs/plans/bayesfilter-one-seed-four-filter-feasibility-result-2026-07-22.md`  
Terminal artifact: `docs/benchmarks/artifacts/one_seed_four_filter_feasibility_20260722/attempt03/`

## Research intent ledger

| Field | Declaration |
|---|---|
| Main question | On which current canonical model rows can UKF, fixed SGQF, fixed-variant Zhao-Cui, and repaired GenUT emit finite values and analytical or recursive scores for the same data, parameter chart, target density, and time order in a one-seed feasibility run? |
| Candidate | Repaired row-quotient GenUT route `cubature_genut_nonfused_positive_ot_row_quotient_candidate_v2`, with `N=1002`, FP32, TF32, GPU, and XLA. |
| Baselines | Same-target UKF, fixed SGQF, and fixed-variant Zhao-Cui routes already implemented in the repository. None is called an oracle for a nonlinear model. |
| Expected failure mode | A method is unavailable for the target; target/time-order/chart mismatch; non-finite value or score; invalid GenUT reset diagnostics; or a historical Zhao-Cui retained-grid route being mistaken for the active fixed variant. |
| Primary feasibility criterion | Every executed cell emits a finite value and finite runtime analytical/manual/recursive score from one scalar route; GenUT additionally passes its program-valid and residual checks. |
| Promotion veto | Any target/hash/time-order/chart mismatch, runtime FD or autodiff score, non-finite output, GenUT program invalidity, or use of the demoted Zhao-Cui multistate retained-grid route. |
| Continuation veto | Corrupt canonical data, invalid shared GenUT kernel, unavailable GPU/XLA/memory-growth policy, or artifacts unable to distinguish comparable and non-comparable cells. |
| Repair trigger | Local harness/serialization error or a route call that is wired to the wrong prefix or coordinate system. |
| Explanatory diagnostics | Values, score coordinates, pairwise differences, runtimes, route IDs, score provenance, GenUT reset diagnostics, and GPU allocator peak. |
| Must not be concluded | One seed cannot rank methods, estimate uncertainty, establish full-horizon accuracy, promote a default, certify HMC, or make an unavailable method look like a failed numerical method. |

## Scope

The run uses canonical data seeds but short prefixes where full horizons would turn a simple feasibility check into a serious campaign:

| Model | Seed | Horizon | Comparable methods | Important limitation |
|---|---:|---:|---|---|
| KSC Gaussian-mixture transformed SV | 81101 | 10 | UKF, SGQF, Zhao-Cui, GenUT | Complete four-way same-target diagnostic on the explicit amended initial-observation-first fixture; KSC is a surrogate target, not exact native SV or the source-order leaderboard row. |
| Exact transformed SV | 81101 | 10 | SGQF, Zhao-Cui, GenUT | Uses the explicit amended initial-observation-first fixture. Raw-observation augmented-noise UKF uses a different value measure and is not inserted into this transformed-value comparison. |
| Generalized SV prior-mean source row | 81105 | 10 | SGQF, Zhao-Cui, GenUT | No reviewed same-target UKF route is implemented. The source row transitions before every observation. |
| Predator-prey | 81104 | 20 | SGQF, GenUT | The fixed-variant Zhao-Cui source-route evaluator is missing; the historical retained-grid result is forbidden. The existing UKF uses a different first-observation convention. |

All GenUT cells use `N=1002`, which is greater than 1000 and divisible by `2d` for state dimensions one, two, and three. One particle seed is used per model. Comparators are deterministic fixed-design routes at the frozen data and parameter point.

## Evidence contract

- Exact comparator: no exact comparator is claimed for the nonlinear rows.
- Primary pass/fail result: route coverage and finite internally valid value/score execution.
- Hard vetoes: mismatched target semantics, non-finite outputs, forbidden score backend, invalid GenUT diagnostics, wrong device/XLA policy, or demoted Zhao-Cui route use.
- Explanatory only: all numerical differences between methods and all runtimes.
- Nonclaim: a missing cell is `not_implemented_or_not_comparable`, not evidence that the method numerically failed.
- Artifact root: `docs/benchmarks/artifacts/one_seed_four_filter_feasibility_20260722/`; attempts are append-only and the terminal result is `attempt03`.

## Default and assumption audit

| Choice | Provenance | Justification | Failure mode | Early diagnostic | Status |
|---|---|---|---|---|---|
| `N=1002` | User requires `N>1000`; GenUT designs require divisibility by `2d` | Small common count for scalar and two-state rows | Monte Carlo noise remains large | One-seed label and no ranking claim | Convenience feasibility scope |
| `T=10` for SV rows | Prior short-prefix feasibility tests | Keeps the four-route check bounded | Does not predict full-horizon accumulation | Preserve full horizon as an explicit nonclaim | Convenience scope |
| Predator-prey `T=20` | Canonical row definition | Short canonical horizon is already affordable | Score variance remains unestimated | Report one seed only | Canonical scope |
| GenUT controls `epsilon=2`, Sinkhorn 8, balance 8, ridge `1e-5` | Previous repaired-route warm start | Known finite starting configuration | Cross-model controls may be suboptimal | Record residuals; no accuracy/default claim | Warm-start hypothesis, not tuned default |
| Zhao-Cui degree 16 / order 41 on SV prefixes | Prior bounded fixed-variant diagnostic | Cheap analytical-score smoke | Can be too coarse to represent full route quality | Label as fixed-variant feasibility configuration | Convenience diagnostic |
| FP64 comparator / FP32 GenUT | Existing route implementations and requested GenUT lane | Tests each method's current intended numerical path | Differences mix method and precision effects | Record dtype and forbid pure-method attribution | Explicit limitation |

## Skeptical plan audit

The plan was checked for wrong baselines, proxy promotion, missing stop conditions, unfair comparison, stale context, environment mismatch, and uninformative artifacts.

- Wrong baseline: repaired. The predator-prey historical Zhao-Cui retained-grid number is excluded, not reused. UKF rows with different likelihood measure or first-observation timing are excluded from same-target differences.
- Proxy promotion: passed. Finiteness and reset residuals establish only engineering feasibility; numerical differences are descriptive.
- Hidden assumptions: passed after recording short prefixes, warm-start GenUT controls, bounded Zhao-Cui design, mixed precision, and one seed as limitations.
- Stop conditions: passed. A method-specific unavailable or non-finite cell does not stop later models unless it exposes corrupt data or a shared-kernel failure.
- Environment: passed conditionally on verified TensorFlow memory growth, visible GPU, TF32, and XLA output-device evidence.
- Artifact sufficiency: passed. The JSON must contain row/method route identity, target/timing/chart, value, score, provenance, dtype, device, runtime, status, and explicit unavailable-cell reason.

## Execution

Initial attempt budget: two launches, at most 20 GPU-minutes total. A localized harness defect may be repaired and retried without changing the scientific contract. Both initial attempts stopped on localized harness defects; one explicitly approved final retry completed while the total GPU budget remained below 20 minutes.

```bash
CUDA_VISIBLE_DEVICES=-1 python -m py_compile docs/benchmarks/run_one_seed_four_filter_feasibility.py
nvidia-smi
TF_FORCE_GPU_ALLOW_GROWTH=true /home/chakwong/anaconda3/envs/tf-gpu/bin/python \
  docs/benchmarks/run_one_seed_four_filter_feasibility.py \
  --output-root docs/benchmarks/artifacts/one_seed_four_filter_feasibility_20260722/attempt01
```
