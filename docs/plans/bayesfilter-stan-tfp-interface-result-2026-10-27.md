# TFP Stan interface: engineering result

Historical implementation record: the October 28 owner-requested rewrite
replaced this direct-tree wrapper with native TFP NUTS. This report retains
its original evidence; see the [current result](bayesfilter-stan-tfp-native-result-2026-10-28.md)
for the new implementation and timings.

Status: interface implemented and bounded engineering validation completed under the
[reviewed plan](bayesfilter-stan-tfp-interface-2026-10-27.md).

The wrapper implements TFP's TransitionKernel protocol while delegating
numerical bootstrap and transitions to the existing direct Stan authority.
`from_warmup` carries the existing adaptation's final state into fixed-kernel
TFP sampling. No shared sampler numerical code was changed.

Attempt 01: 18 passed, one failed, in 71.770 subprocess seconds. Transition
parity, XLA sampling, continuation and adaptation handoff passed. The sole
failure was the invalid scalar-seed test expecting ValueError/TypeError while
TensorFlow's shape check emitted InvalidArgumentError. Repair: explicitly
reject a non-pair shape with ValueError at the adapter boundary. This was an
API error-class mismatch, not a stochastic or numerical mismatch.

Attempt 02: all 12 interface tests and six scope checks passed; the remaining
scope-discovery check failed because the validation launcher archived the
existing stock-TFP adapter as a `.py` file beneath docs/benchmarks. The guard
correctly treated that copied file as a second executable route. Repair:
rename source snapshots to `.py.txt`, preserving every byte and recording
the rename/hash map. The numerical code and tests remain unchanged. Rerun
the scope checks only; passing interface checks do not need repetition.

Attempt 03 passed all seven scope regressions in 32.441 subprocess seconds.
The final implementation has 12 passing interface tests from attempt 02 and
seven passing scope tests from attempt 03: **19 tests passed across the final
two focused invocations**. Their code hashes agree. Total validation time
was 176.587 seconds across three attempts, below the 600-second/four-attempt
engineering allowance. No benchmark or posterior campaign was run.

The interface tests cover unit, diagonal and dense inverse metrics; jitter;
target and gradient caches; acceptance, work, depth, divergence and energy;
exact random-state progression; TFP burn-in/thinning and checkpointed
continuation; explicit seed reproduction and change; actual dense Stan
adaptation handoff; copy behavior; invalid dtype, epsilon, metric, position,
missing seed and duplicated stream keys. Optimized HLO retrieval succeeds for
the enclosing seeded and continuation sampling graphs. The graph has no
Python callback operations, and tested functions retain one trace.

Observed environment: Python 3.13.13, TensorFlow 2.20.0, TFP 0.25.0. GPU
devices were intentionally hidden; the environment report records none.
The CUDA_NO_DEVICE log in that run is expected under this CPU reference
configuration and is not GPU readiness evidence. No GPU workload was run.

Evidence root: `docs/benchmarks/artifacts/stan-tfp-interface-20261027/`.
Each attempt retains its command, source hashes, environment and wall time.
The three pre-existing pytest warnings are TFP deprecations and an unrelated
source scan escape warning. They do not change the pass/fail evidence.

The installed TFP source was inspected directly: python/mcmc/kernel.py,
python/mcmc/sample.py and python/internal/samplers.py under the tfgpu
environment. Official API web requests returned no content, so the
implementation contract is grounded in that installed source, not a claimed
successful web retrieval.

## Final review and decision

The claimed target is TFP driving the audited Stan transition, not reproducing
stock TFP's own NUTS tree or native Stan's random-number generator. On the
tested common-input fixtures, states and floating telemetry agree within
2e-13 relative/absolute tolerance; RNG counters and discrete results agree
exactly. The checked warm-up bridge carries frozen parameters and keeps
adaptation draws out of retained sampling. The core and adaptation files are
byte-for-byte unchanged from the committed baseline.

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Unsupported conclusion |
| --- | --- | --- | --- | --- | --- |
| Accept standalone TFP interface | Shared-transition, sampling and continuation checks pass | No unresolved API/XLA/scope failure in tested cases | General targets, GPU wrapper execution, untested TFP wrappers | Use documented API; add target-specific validation when applying it | Faster transition or posterior convergence |
| Retain existing performance interpretation | No new timing comparison | Prior centered-funnel validity failure remains | Three-seed descriptive timings and unmatched native boundary | Profile fixed states/metrics/work before optimizing | Interface closes the native-Stan speed gap |

| Inference status | Evidence |
| --- | --- |
| Hard veto screen | Engineering fixture checks pass after two localized repairs; no new posterior screen |
| Statistically supported ranking | None; no new comparison was run |
| Descriptive-only differences | Original seven-target timing report remains the only timing evidence |
| Default-readiness | Not assessed; public NUTS routes remain retired |
| Next evidence needed | Controlled profiling and replicated, validity-qualified posterior benchmarks for any speed claim |

Post-run review: a shared numerical implementation reduces duplication but
also means parity cannot independently prove the core's correctness. The
existing native-oracle tests remain its separate evidence; they were not
rerun for an unchanged core. The weakest new evidence is coverage beyond
the tiny banana target, especially wrapper GPU execution and uncommon TFP
compositions. A mismatch under common inputs or loss of enclosing XLA
compilation would overturn interface acceptance for that path.

The failures invalidated the first error-class expectation and the second
launcher's source-snapshot naming, respectively. They did not invalidate the
target, direct algorithm or research direction. Both triggered their bounded
repairs and were resolved.

Continuation/reset note: new code is `docs/benchmarks/stan_nuts_tfp.py`; usage
is in `docs/reference/nuts-stan-tfp.md`. No numerical speed optimization was
made, no prior raw evidence was overwritten, and no broad environment or
public sampler default changed. The next performance task is the controlled
profile already described in the earlier full-port result.
