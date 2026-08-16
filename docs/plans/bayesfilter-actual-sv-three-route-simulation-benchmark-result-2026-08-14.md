# Result: actual-SV three-route simulation benchmark (2026-08-14)

Plan: `docs/plans/bayesfilter-actual-sv-three-route-simulation-benchmark-plan-2026-08-13.md`
(with the 2026-08-14 skeptical pre-execution audit appended there).

## Recovery context

This execution continues a stalled session ("load delegated dancing plan").
Recovered state: the plan file already existed; the two-lane harness had been
partially migrated from the two-row deterministic fixture to simulated paths
(a dims>1 slicing defect in that migration was found and fixed — the frozen
generator emits a `[horizon, 1]` path, so each panel coordinate now gets its
own independently simulated path); no consolidated three-route script,
fitted-mixture builder, or artifact existed yet. All were built in this
session.

## Run manifest

- Git commit (dirty working tree, uncommitted benchmark work): `18cfe609`
- Command:
  `python docs/benchmarks/benchmark_actual_sv_three_route_simulation.py --dims 1,2,3 --horizon 20 --output docs/benchmarks/artifacts/actual_sv_three_route_simulation_20260814/attempt01/result.json --markdown-output .../result.md`
- Environment: conda `tf-gpu`, Python 3.11.14, TensorFlow 2.19.1, float64
- Device: CPU-only, GPUs deliberately hidden (`CUDA_VISIBLE_DEVICES=-1`);
  this is the plan's declared CPU-only diagnostic scope, not a sandbox failure
- Data: simulated exact actual-SV paths, horizon 20, seed base 83120,
  independent per-coordinate paths; deterministic (no runtime randomness)
- Wall time: 162 s
- Artifacts:
  `docs/benchmarks/artifacts/actual_sv_three_route_simulation_20260814/attempt01/{result.json,result.md}`
- Schema test: `tests/test_actual_sv_three_route_benchmark_script.py` (passed)
- Internal validity checks (both passed inside the run): mixture-Kalman
  coordinate-factorization vs joint enumeration (gap 0.0); fitted-mixture L1
  error monotone decreasing 7 -> 14 -> 28

## Numbers

Same-target (route-internal) gaps, log-likelihood units:

| Route | dim 1 | dim 2 | dim 3 |
|---|---|---|---|
| Fixed-variant actual-SV batch TT vs exact dense | 5.5e-3 | scalar-only (N/A) | scalar-only (N/A) |
| Exact-transformed Zhao-Cui TT vs own dense | 3.5e-8 | 3.4e-7 | 1.3e-7 |
| KSC-surrogate Zhao-Cui TT vs own dense KSC | 5.2e-6 | 3.4e-5 | 6.1e-5 |
| Collapse mixture-Kalman vs dense KSC (same mixture) | 2.5e-3 | 3.0e-3 | 5.3e-3 |

Score vs centered finite difference (dim 1 only): exact-transformed TT
relative error 2.1e-10; KSC-surrogate TT 1.2e-10; both pass.

Dense Gaussian-mixture / Kalman refinement ladder (fitted 7/14/28-component
mixtures to the exact log-chi-square density; quadrature-EM, weighted L1
density errors 9.6e-3 / 1.1e-3 / 4.2e-4):

| dim | rel. change 7->14 | rel. change 14->28 | stabilized (<1%) |
|---|---|---|---|
| 1 | 5.1e-6 | 4.8e-7 | yes |
| 2 | 3.2e-4 | 2.6e-5 | yes |
| 3 | 7.7e-5 | 4.9e-7 | yes |

Cross-family raw-`y` gaps to the exact dense reference (descriptive only,
after exact Jacobian correction; KSC rows use the offset-aware correction):

| dim | KSC-7 dense - exact dense | fitted-28 Kalman - exact dense |
|---|---|---|
| 1 | -0.066 | +0.0020 |
| 2 | -0.074 | -0.0144 |
| 3 | +0.104 | +0.0038 |

## Decision table

| Item | Status |
|---|---|
| Decision | All three routes pass their own same-target references on simulated data; the consolidated benchmark is admitted as the standing comparison artifact |
| Primary criterion (per-route same-target gap) | Exact-transformed ZC and KSC-surrogate ZC essentially exact (<=6e-5); batch TT gap 5.5e-3 (finite frozen-core TT truncation, small but nonzero) |
| Veto diagnostics | None fired: all values finite, factorization check exact, TT status codes clean, fitted-mixture quality monotone |
| Main uncertainty | Single simulated path per dimension; all continuous gaps are descriptive with unquantified Monte Carlo variation |
| Next justified action | If a ranking or default change is ever wanted: multi-seed replication with paired uncertainty analysis; otherwise none required |
| Not concluded | No statistical superiority of any family; no HMC readiness; no posterior-correctness claim; no production-performance claim |

## Interpretation by elimination (per plan)

- Zhao-Cui does **not** fail its own dense same-target references: the
  exact-transformed route is exact to ~1e-7 and the KSC-surrogate route to
  ~6e-5. No Zhao-Cui-side issue on these fixtures.
- Dense Kalman does **not** change materially under 7->14->28 refinement
  (all changes <=3.2e-4 relative, far below the 1% screen): no budget issue
  in the mixture-resolution sense at these sizes, and no erratic behavior
  (no bug signal).
- The families **do** remain apart after both stabilize, but the remaining
  raw-`y` cross-family gap shrinks by an order of magnitude when the KSC-7
  mixture is replaced by mixtures fitted to the exact log-chi-square density
  (|gap| ~0.07-0.10 -> ~0.002-0.014). Classification: the dominant
  cross-family difference is **approximation-family bias of the 7-component
  KSC mixture**, not an implementation defect in either route.
- The batch TT route's 5.5e-3 same-target gap is `heuristic only` evidence of
  frozen-core TT truncation at (degree 10, rank 2); it is not a correctness
  defect claim and was not score-inconsistent (its manual score admission
  evidence lives in the SVX campaign artifacts).

## Inference-status table (stochastic-comparison discipline)

| Row | Status |
|---|---|
| Hard veto screen | Passed for all routes (finite values, clean status, internal checks passed) |
| Statistically supported ranking | None — single path, no replication; no ranking is claimed |
| Descriptive-only differences | All cross-family gaps and the batch-TT truncation gap |
| Default-readiness | Unchanged — this artifact changes no defaults |
| Next evidence needed | Multi-seed paired replication if any family ranking is ever to be claimed |

## Post-run red team

- Strongest alternative explanation: the small remaining fitted-28 vs exact
  gaps (~0.002-0.014) could be collapse (GPB1-style) error of the mixture
  Kalman rather than residual mixture bias — the measured collapse-vs-dense
  gap (2.5e-3..5.3e-3) is the same order, so these are not separable here and
  no claim distinguishes them.
- What would overturn the conclusion: a replication showing erratic
  refinement behavior on other simulated paths, or a dense-KSC-reference
  defect; both are bounded by the internal checks run here.
- Weakest evidence: horizon-4 smoke showed a much larger batch-TT gap (0.37),
  so the batch-TT truncation gap is horizon/dataset-dependent; the 5.5e-3
  figure is specific to this path and horizon.
