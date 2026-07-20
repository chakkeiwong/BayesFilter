# SSL-LSTM Precision Accuracy And Speed Plan

Date: 2026-07-20  
Status: `COMPLETE`  
Tier: Tier 2 material research engineering

## Question

For the same selected four-parameter SSL-LSTM principal-square-root UKF
value/score target, can either of these precision policies preserve adequate
accuracy while reducing warm GPU/XLA evaluation time relative to the current
all-float64 implementation?

1. `mixed_lstm32_filter64`: LSTM transition, observation, and their local
   derivatives use float32/TF32-eligible matrix products; sigma-point moments,
   covariance derivatives, square roots, solves, likelihood, and score
   accumulation use float64.
2. `all_float32_tf32`: every floating tensor uses float32 and TF32 execution is
   enabled for eligible GPU matrix products. TF32 is not a storage dtype and
   does not apply to eigendecompositions or arbitrary elementwise operations.

The production implementation remains unchanged during this experiment.

## Research Intent And Evidence Contract

| Field | Contract |
| --- | --- |
| Main question | Does mixed or all-float32 execution retain an accurate selected value/score and materially reduce warm GPU/XLA time? |
| Exact baseline | Current all-float64, direct-JVP, principal-square-root UKF target at identical q, observations, parameter points, equations, floors, XLA setting, and GPU. |
| Candidate mechanism | Precision changes only: mixed LSTM float32/filter float64, then all-float32 with TF32 eligible matmuls. |
| Primary promotion criterion | Against all-float64 at every measured point: mixed value absolute error <= `2e-5`, score max absolute error <= `2e-4`, score max relative-to-scale error <= `2e-4`; all-float32 value error <= `2e-3`, score max absolute error <= `2e-2`, score max relative-to-scale error <= `2e-3`; identical floor-count branches and finite outputs. |
| Promotion veto | Non-finite output, XLA/device failure, covariance/square-root assertion, changed floor-count branch, tolerance failure, source drift, timeout, or malformed artifact. |
| Explanatory only | Warm wall time, compile time, allocator memory, host RSS, and per-q speed ratios. They cannot override an accuracy veto. |
| Continuation veto | Invalid FP64 reproduction, equations differ between arms beyond declared casts, GPU unavailable in trusted context, or another process changes in-scope source. |
| Repair trigger | Local mixed-precision failure triggers separation of transition/JVP error from filter recurrence error; all-float32 failure does not invalidate mixed precision. |
| Nonclaims | No posterior correctness, HMC convergence, NeuTra quality, scientific superiority, or default-policy promotion follows from this benchmark. |
| Result artifact | `docs/plans/bayesfilter-ssl-lstm-precision-accuracy-speed-result-2026-07-20.md` plus structured JSON under `docs/plans/artifacts/ssl-lstm-precision-accuracy-speed-2026-07-20/`. |

## Baseline Ladder

- Naive/current reference: all-float64 direct-JVP target.
- Plain proposed method: mixed LSTM float32 and filter/score float64.
- Enhanced/aggressive method: all-float32 with TF32-enabled eligible matmuls.

No weak surrogate metric may replace end-to-end value/score parity.

## Execution

Implement an isolated experimental precision target so production dtype
contracts are not silently changed. Verify it reproduces the current FP64
target before measuring candidates. Run CPU-hidden focused correctness tests,
then trusted GPU/XLA comparisons at `q in {5, 10, 20}` using identical fixed
points, alternating arm order, synchronized calls, and fresh processes.

The benchmark records actual tensor dtypes and the TF32 policy separately. It
must not label float32 tensors as a distinct `tf32` dtype.

## Resource Stop

- One physical GPU only; do not use or interrupt the GPU occupied by the other
  lane.
- Maximum 24 fresh worker cells and 45 minutes total GPU wall time.
- Memory growth required; stop above 28 GiB TensorFlow allocator use or 64 GiB
  process high-water RSS.
- Stop an individual worker after 600 seconds.
- Two paired repetitions are descriptive engineering evidence, not a
  statistically supported runtime ranking.

## Pre-Mortem

- The benchmark could appear fast because a candidate omits derivative work.
  Guard with FP64 reproduction, output shapes, score parity, and source tests.
- Mixed precision could pass at the center but fail nearby or at larger q.
  Use five fixed parameter points at every q.
- All-float32 could change covariance-floor branches while values look close.
  Floor-count identity is a hard veto.
- GPU contention could dominate timing. Record prelaunch utilization and mark
  contaminated pairs; accuracy remains usable while timing stays descriptive.
- TF32 could be reported as active even though tensors remain float64. Record
  storage dtype and TF32 policy independently.

## Skeptical Pre-Execution Audit

Passed after revision. The plan uses the actual all-float64 selected-score
target rather than a value-only or neural-network-loss proxy; accuracy vetoes
precede timing; mixed and aggressive candidates share the same equations and
fixed points; TF32 terminology is corrected; floor-branch changes, GPU
contamination, environment, resource stops, and nonclaims are explicit. The
artifacts answer the stated accuracy-and-speed question without authorizing
HMC, NeuTra, or a production dtype change.
