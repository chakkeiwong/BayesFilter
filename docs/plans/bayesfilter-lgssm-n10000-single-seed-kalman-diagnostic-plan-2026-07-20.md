# LGSSM N=10000 Single-Seed Kalman Diagnostic Plan

Date: 2026-07-20
Status: `CLOSED_ENGINEERING_PASS_MIXED_ONE_SEED`
Parent result:
`docs/plans/bayesfilter-lgssm-particle2000-particle5000-kalman-bias-ladder-result-2026-07-20.md`

## Research Intent And Evidence Contract

| Field | Declaration |
| --- | --- |
| Question | For the same estimator seed at `T=50`, is the canonical `N=10000` value and total score descriptively closer to the exact Kalman target than the preserved `N=5000` singleton result? |
| Candidate | One canonical Contract E--Chol value-and-total-score run at `N=10000,K=2500`, a `4 x 4` block grid. |
| Exact comparator | Preserved singleton seed `82220` at `T=50,N=5000,K=2500`, selected controls `(20,5)`, from `n5000_repair_scope_attempt01`. |
| Controls | `(sinkhorn_steps=20,balance_steps=5)` transferred from `N=5000` as a warm-start diagnostic only; not tuned or selected for `N=10000`. |
| Pass/fail | Engineering feasibility requires GPU/TF32/XLA, exact chunks, finite value/total score, bitwise replay, chart/reset/marginal/work validity, and allocator peak below 8192 MiB. Closeness is reported per output as a paired descriptive observation, not a promotion criterion. |
| Veto | OOM/resource failure under the fixed limit, wrong route/chunk/scope, non-finite result, replay failure, or failed canonical hard gate. |
| Explanatory diagnostics | Value and five score errors relative to Kalman, relative errors where the oracle is nonzero, change in absolute error from paired `N=5000`, runtime, and peak allocator bytes. |
| Artifact | `docs/benchmarks/artifacts/lgssm_n10000_single_seed_kalman_20260720/attempt01/`. |
| Nonclaims | One seed does not estimate bias, establish monotonic convergence, validate `(20,5)` for `N=10000`, support a statistical ranking, or certify HMC/default readiness. |

## Default And Assumption Audit

| Choice | Provenance/status | Failure mode | Early diagnostic |
| --- | --- | --- | --- |
| Seed `82220` | First seed of final `N=5000` untouched claim | A fresh unrelated seed would confound particle count with realization | Pair exact same prepared root seed across `N` |
| Controls `(20,5)` | Selected for `N=5000`; cross-scope warm start only | Marginal failure or misleading particle comparison because `N=10000` needs different tuning | Preserve direct gates and label all Kalman differences diagnostic-only |
| `K=2500`, `4 x 4` | Required active chunk policy | Wrong chunks answer an ineligible route | Repository selector plus preparation identity |
| Singleton execution | Correctness fallback after TF32 batch-shape parity failure | Low throughput, but no batch-shape semantic drift relative to comparator | Exact batch size one in both paired arms |
| GPU/float32/TF32/XLA | Repository production target | TF32 recursion is numerically sensitive | Replay and same singleton geometry; no cross-batch comparison |

## Skeptical Plan Audit

Verdict: `PASS` for a one-seed diagnostic.

- Wrong baseline is prevented by the same seed, target, horizon, route, dtype,
  TF32/XLA mode, and singleton geometry. Particle count and policy block grid
  are the intended changes.
- One-seed closeness is explicitly explanatory and cannot become a bias or
  promotion claim.
- Controls are visibly cross-scope and may veto feasibility; they cannot be
  called tuned `N=10000` controls.
- The command writes exact Kalman value/score, per-seed output, replay,
  allocator, device, graph, chunk, and source evidence sufficient to answer the
  limited question.
- One attempt is bounded by a 45-minute wall-time cap and the 8192 MiB logical
  GPU limit. Existing external GPU contention may affect timing but not the
  paired numerical interpretation or TensorFlow allocator peak.
