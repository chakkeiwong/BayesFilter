# Final public-owner resource result

Runs05249--05304 complete42 fresh GPU cost workers,12 CPU/GPU reuse workers
and161 terminal checks. All declared numerical records agree at the unchanged
bounds. Run05255 remains failed: a diagnostic NumPy boolean could not be JSON
serialized after its numerical/timing checks. The retry changes only diagnostic
serialization. Terminal readback verifies AST identity of every measured/helper
function with the archived original harness; none of the numerical sources,
inputs or tolerances changed.

The plan is filter_gradient_final_owner_resources_20260930.md. Manifests retain
exact commands, frozen sources, environment, fixtures/seeds, device and elapsed
time. GPU costs use the same uncontended GPU3, verified memory growth, FP64 and
TF32 explicitly off. CPU workers are lifetime references. Three counterbalanced
process pairs each contain30 synchronized warm calls; calls within a process
are not independent replications. Means below summarize process medians.

| Complete callable | GPU warm before/current, ms | Paired geometric ratio [approximate95% interval] | Cold before/current, s | Mean RSS change, MiB |
|---|---|---|---|---|
| D3 geometry, original/current | 5.797 /6.016 | 1.0375 [0.9499,1.1332] | 3.681 /4.229 | +9.10 |
| D23 geometry, original/current | 17.335 /25.831 | 1.4897 [1.3012,1.7056] | 9.352 /9.916 | +7.90 |
| SQMC IID live preparation/value/score | 49.080 /47.267 | 0.9631 [0.9585,0.9677] | 5.344 /6.643 | +10.46 |
| SQMC Halton live preparation/value/score | 87.967 /58.964 | 0.6703 [0.6345,0.7080] | 9.747 /8.873 | -33.15 |
| Static trace summary | 4.522 /4.424 | 0.9783 [0.9730,0.9838] | 6.263 /6.216 | -17.82 |
| Dynamic trace summary | 7.863 /7.860 | 0.9997 [0.9873,1.0122] | 6.500 /6.586 | -18.01 |

The D23 warm regression exceeds the master's20% investigation trigger. Its
three-arm attribution isolates the stable-angle increment:17.335 to24.998ms,
ratio1.4419 [1.2525,1.6600]. Adding the subspace-resolution guard gives24.998
to25.831ms, ratio1.0332 [0.9281,1.1502]. D3 corresponding ratios are1.0518
[0.9947,1.1121] and0.9865 [0.9342,1.0417]. These compare analytically resolved
10/25/40-degree rotations and changed inputs, with exact rank decisions and
the existing1e-10 geometry bound. They do not make the inaccurate saved
original D23 tiny-angle result an eligible speed baseline.

Accept the D23 increase as a scoped correctness cost of the stable residual-SVD
angle calculation already qualified through04922 and the resolution guard
qualified through04935. Reverting the calculation would restore the preserved
accuracy defect. This is an explicit engineering tradeoff under the master,
not a claim of optimal implementation or equal speed. Maximum current GPU
allocator peak is114432 bytes for D23 versus89856 before; no2x allocator,
256MiB host or2x cold trigger occurs in this cohort. The remaining cold/RSS
differences are descriptive; the IID call pays preparation compilation before
benefiting in warm execution.

Public geometry and SQMC timings include their live public preparation, trace
and first execution. Static/dynamic trace-summary timings include the complete
summary kernel and public report conversion, with identical prepared input
tensors outside that boundary. Input construction for those traces is measured
separately by the public IID/Halton scopes. Comparison compilation and graph
inspection follow primary memory/timing samples. Output lifetimes match.

Each current callable passes256 alternating-input calls on CPU and GPU with
exact replay, a fixed cache size, one trace per compiled owner and no host
callbacks. Late128-call RSS growth is0--126976 bytes across all12 workers,
well below16MiB. GPU live allocator bytes are stable; peaks are25856/114432
for D3/D23,62720/64256 for IID/Halton and71168/74496 for static/dynamic traces.
GPU process reservations of461373440--467664896 bytes include context/runtime
overhead and are not live tensor memory. Module LRU owners are intentionally
retained. Parent-observed process exit releases worker residency. These checks
qualify finite fixed configurations, not arbitrary-shape capacity or native
cache eviction.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action / nonclaim |
|---|---|---|---|---|
| Close these owner resource obligations | Complete records, repeatability, bounded reuse and attribution pass | D23 speed trigger explained and explicitly accepted;05255 preserved | Three process pairs, finite fixtures | Reuse these exact source scopes; no universal speed/capacity claim |
| Preserve current accurate geometry | Independent resolved rotations and earlier tiny-angle/error tests support the repair | Saved inaccurate original remains ineligible | Further optimization not evaluated | Keep stable calculation and refusal guard |
| Continue core-cost reconciliation | Core execution qualification already exists | Four TT fixtures were accidentally deleted by merge9d8202b77 | Several early cost records are smoke/descriptive only | Restore only TT fixtures and renew genuinely missing matched costs |

| Inference status | Result |
|---|---|
| Hard veto screen | Numerical/status/source/device/lifetime checks pass; failed serialization retained |
| Statistically supported ranking | Approximate paired intervals exclude1 for D23 slowdown and IID/Halton/static-summary reductions within these fixtures |
| Descriptive-only differences | Cold time, RSS, allocator peaks, and all generalization beyond this cohort |
| Default readiness | This closes scoped resources only; remaining current-caller/core-cost/terminal gates still block main merge |
| Next evidence needed | Missing core-filter costs and final affected-use/source dispositions |

Primary-agent skeptical review: the strongest alternative explanation for the
small warm differences is process/system variation; only three independent
pairs limit inference despite30 inner calls. The D23 attribution is materially
larger and consistent, but does not prove that its cost is minimal. Fixed-shape
stability does not prove unbounded cache behavior. No tolerance relaxation,
canonical LEDH admission, HMC/training run, or deferred iAPF/KDM work occurred.

Evidence: run05304/final-resource-readback.json and the verified archive
`artifacts/filter-gradient-repair-20260917/final-owner-resource-05249-05304-evidence.tar.gz`.
