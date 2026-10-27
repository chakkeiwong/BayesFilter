# Stan adaptation transplant: code audit and repairs

The Python/TensorFlow transplant implements the supported diagonal/dense
Euclidean warm-up correctly under the checks below. The audit found and
repaired concrete edge-case and validation defects. It is reusable for
arbitrary fixed differentiable targets satisfying its batched FP64 contract,
without calling CmdStan at runtime. It remains a standalone controller around
TFP NUTS, and is **not a complete implementation of every Stan sampler or
adaptation option**.

The [guide](../reference/nuts-tuning.md#reusing-the-python-code) now contains
an importable arbitrary-target example, named results, checked warm-up,
freezing and continued sampling, and the supported/unsupported interface.
The [pre-run audit plan](bayesfilter-stan-adaptation-code-audit-2026-10-27.md)
states the evidence contract. No production default or retired public route
was changed.

## Defects found and repaired

| Finding | Why it matters | Repair and evidence |
| --- | --- | --- |
| Singleton metric windows divided by n-1 | An accepted configuration could produce NaNs; Stan retains the preceding metric estimate and then shrinks it | Previous metric is used for n<=1; streamed singleton and expanding-window comparisons use actual Stan estimators |
| Dense Welford outer product was transposed relative to Stan | Algebraically symmetric in exact arithmetic, but a different floating-point update order | Use (q-new_mean) times (q-old_mean) transpose, matching the installed source |
| Finite epsilon above 1e7 was probed and rejected | Stan skips the reasonable-step search for such inputs | Match the skip; still invalidate a search that crosses the cap or underflows; actual base_hmc search oracle covers both |
| Configuration accepted NaN/infinity and non-integer counts | Invalid parameters could survive validation and fail deep in tracing/adaptation | Validate finite positive reals, positive integer sizes, metric/target domain and int32 depth safety |
| Final smoothed epsilon and newly installed metric/cache were not included in final validity | A successful status could precede a bad handoff | Validate bootstrap, metric boundaries and final smoothing; injected bad smoothing is rejected |
| Sampling did not report validity for supplied epsilon/factor | A raw handoff could silently use an invalid factor or negative epsilon | Per-chain cumulative validity, factor structure/diagonal checks and host helpers; benchmark checks the new status |
| Target shape, dtype and connectivity were implicit | A scalar reduction can accidentally couple all chains or give unclear trace errors | Require float64 [chains] output and a connected gradient; caller still owns row independence |
| Native oracle could be stale or silently skipped | Passing tests did not necessarily compare the current transplant with Stan | Rebuild current driver against pinned actual headers; audit requires the native installation and records every oracle call |
| Existing arithmetic test used batch-centered covariance | It did not exercise the implemented streaming recurrence or resets | Streaming native comparisons plus replay of complete actual local warm-up through Stan adapters |

No claim is made that the transposed covariance update caused a scientific
failure in the old comparison. The ordinary n>1 diagonal arithmetic was
already equivalent. Previous benchmark artifacts and their source snapshots
remain unchanged; their timings refer to that version. This audit does not
renew timing or posterior claims for the repaired code.

## Source coverage

The comparator is CmdStan **2.37.0**, using installed unmodified headers.
The [native build manifest](../benchmarks/artifacts/stan-adaptation-audit-20261027/cpu-01/oracle-build.json)
records all inspected source hashes, compiler command and oracle hash.
The actual TFP 0.25.0 NUTS source is preserved alongside the audit evidence.

| Component | Checked source | Verdict and scope |
| --- | --- | --- |
| Window schedule | windowed_adaptation.hpp | Every active/end flag agrees for 12 schedules spanning 1–10,000 transitions, including tiny budgets and custom/singleton windows |
| Dual averaging | stepsize_adaptation.hpp | Count, acceptance cap, hbar, log epsilon, smoothing, restart and final estimate agree; default and nondefault settings tested |
| Streaming metric | welford_var_estimator.hpp, welford_covar_estimator.hpp, var_adaptation.hpp, covar_adaptation.hpp | Window-local samples, sample denominator, regularization and reset agree for diag/dense; singleton carry now handled |
| Initial step search | base_hmc.hpp::init_stepsize | Actual Stan search with scripted probes verifies repeated initial epsilon, fresh-probe cadence, direction, crossing, cap, NaNs and underflow; positive finite inputs only |
| Probe mechanics | expl_leapfrog.hpp and base_leapfrog.hpp | Actual native dense-metric integrator matches transformed TF position, momentum and energy ratio for forward/backward steps |
| Initial shrinkage center | hmc_nuts_diag_e_adapt.hpp, run_adaptive_sampler.hpp | mu uses the supplied epsilon before the initial search |
| Metric-boundary order | adapt_diag_e_nuts.hpp, adapt_dense_e_nuts.hpp | Transition, DA update, covariance update, search, new mu, restart; every step replayed through native adapters for two chains and both metrics |
| Boundary coordinates/caches | Local affine implementation and TFP kernel results | Physical q is preserved within FP64 tolerance; instrumented tests recompute incoming/outgoing target and gradient at every transition |
| Acceptance statistic | base_nuts.hpp and installed TFP nuts.py | Mean Metropolis probability across trajectory leapfrogs, not selected-state frequency; rejected nonfinite leaves contribute zero without invalidating a valid retained state |
| Final freeze and resume | Stan complete_adaptation; local sample controller | Final smoothed epsilon checked; no adaptation in sampling; two chunks match one longer chunk within 2e-12 in the fixture |
| Execution | Enclosing tf.function signatures and optimized HLO | CPU and GPU XLA pass for warm-up and sampling, no PyFunc/pfor or retracing; GPU growth configured and verified |

The search oracle controls Hamiltonian differences to isolate Stan's actual
search loop. The warm-up replay supplies the locally searched epsilon at a
boundary and compares subsequent adapter state; it is not an independent
end-to-end Stan NUTS trajectory. The native leapfrog test and search tests
independently check those supplied-search mechanics. Differences in RNG,
tree construction and floating-point evaluation prevent an assertion of
identical Stan/TFP trajectories.

In TFP, a divergent leaf's Metropolis contribution is omitted while its
leapfrog remains in the denominator; Stan adds that contribution. With the
fixed energy threshold of 1000, its finite-energy probability is at most
exp(-1000), which rounds to zero in FP64. This is why the omission does not
change the adaptation statistic under the present setting. Changing that
threshold would require reassessment.

## Mathematical target and limits of completeness

The learned matrix C is Stan's inverse metric. For C=A Aᵀ, the implementation
uses q=A z and p_q=A⁻ᵀp_z. Thus

    p_zᵀp_z / 2 = p_qᵀ C p_q / 2,
    Δq = A Δz = epsilon C p_q,
    grad_z log p(Az) = Aᵀ grad_q log p(q).

These equalities establish the constant-metric Hamiltonian and leapfrog
mapping in exact arithmetic. The omitted log determinant is constant while
A is frozen and cancels in the Hamiltonian differences. At each change of A,
the code recomputes z and all transformed target/gradient caches. Actual
floating-point agreement is checked by the native leapfrog and cache tests.
This argument does not equate the complete tree-building algorithms.

The API supports positive warm-up, positive fitting windows, identity initial
metric, one shared supplied starting epsilon, FP64 and diagonal/dense learned
metrics, with separate state per chain. Tiny warm-up below 20 has no metric
updates. There is no unit-metric warm-up option, supplied initial metric,
zero-warm-up/adaptation-disabled controller entry, zero-length buffer,
automatic short-budget rescaling, step-size jitter, Stan parameter transforms
or Stan initialization. Frozen sampling can be called separately with a valid
factor/step-size state.

The short-budget rescaling difference is intentional and explicit:
Config rejects insufficient fitting windows. The installed Stan fallback
changes lengths and returns without restarting the window counter; silently
implementing an inferred schedule would not establish parity with this
release. No hidden fallback was added.

Both metrics currently use dense matrix storage and Cholesky/triangular
operations. The diagonal results are mathematically diagonal, but the
implementation does not reproduce Stan's O(d) diagonal memory/arithmetic.
This is a concrete remaining optimization opportunity. It was not optimized
inside a correctness audit or used to make a new speed claim.

The arithmetic helpers can be reused inside TensorFlow graphs. The supplied
controller is specific to TFP NUTS and is not a public generic TFP kernel
adapter. Arbitrary target correctness, properness, cross-chain independence,
constraint Jacobians and target-specific warm-up adequacy remain the caller's
responsibility. Input checks cannot prove those mathematical properties.
Batch masking can consume RNG/work for completed chains; separate adaptation
state does not imply invariance to chain layout or batching.

## Executed validation

The [CPU run](../benchmarks/artifacts/stan-adaptation-audit-20261027/cpu-01/manifest.json)
ran the expanded adaptation tests, six-target native density/gradient checks
and retired-route scope guards: **64 passed, zero skipped**, in 66.42 worker
seconds, including 9.89 seconds to compile the current native oracle. The
remaining warnings are dependency deprecation and an existing escape-sequence
warning. GPU devices were intentionally hidden.

The [GPU run](../benchmarks/artifacts/stan-adaptation-audit-20261027/gpu-01/manifest.json)
used TF 2.20.0 / TFP 0.25.0 on NVIDIA GeForce RTX 4080 SUPER with FP64,
TF32 off, trusted execution and verified memory growth. Both metrics ran
two-chain 160-transition warm-up with three metric boundaries and 16 retained
transitions. All validity flags passed; all outputs were on GPU; each
enclosing function traced once and produced saved optimized HLO. These are
mechanics checks, not posterior diagnostics.

On identical deterministic inputs, the largest CPU/GPU relative discrepancy
across Welford, regularization, dual averaging and leapfrog was
7.01e-17, below the predeclared 2e-12 rounding allowance. GPU allocator peak
was 110,848 bytes for this small fixture. The GPU run took 78.05 seconds,
including compilation. It had finite loop counts and was interactively
monitored; unlike the CPU subprocess, it did not install an external wall-time
watchdog. It completed within the declared budget.

Two validation launches consumed 144.47 worker seconds, including the oracle
build, leaving four launches and more than 17 minutes of the conservative
20-minute worker budget unused. No failed launch or repair retry occurred.
The work stopped after the required tests passed; no speed campaign was
repeated.

## Decisions and inference status

| Decision | Primary criterion | Veto diagnostics | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Accept repairs for the documented standalone scope | Actual-source arithmetic, state replay and XLA checks pass | No unresolved failed invariant in the audited scope | Finite test coverage and target-specific behavior | Reuse checked interface; validate each new target | All possible inputs or targets are correct |
| Retain explicit implementation limits | Supported/unsupported options are documented and rejected when represented in Config | No unsupported feature silently promoted | Broader Stan option coverage is absent | Add an option only with its source/behavior tests if needed | Full Stan replacement or public production admission |
| Preserve prior benchmark as versioned evidence | Original source/results remain intact | New timing would require fresh run | Repairs can change paths and cost | Use a separate reviewed speed experiment for optimization | Updated speed or posterior ranking |

| Inference status | Finding |
| --- | --- |
| Hard veto screen | Invalid inputs, invalid final smoothing and invalid handoffs are rejected; rejected nonfinite proposals preserve valid states and zero acceptance in the test |
| Statistically supported ranking | None sought or established |
| Descriptive-only differences | CPU/GPU rounding residuals and engineering run durations; no stochastic efficiency ranking |
| Default-readiness | No production default change; standalone exception remains the scope |
| Next evidence needed | Target-specific multi-chain posterior checks for a new target; fresh paired timing/ESS evidence for any speed claim |

Engineering correctness has deterministic source-oracle evidence. Sampler
validity has mechanics/cache/compiled-handoff evidence, plus the preserved
prior target comparison under its original source version. Scientific
interpretation has no new convergence or comparative-performance evidence.
These three evidence classes have not been substituted for one another.

## Terminal self-review and continuation note

The strongest alternative explanation is that tests share an unnoticed
assumption with the transplant. Actual Stan headers, a separate native
integrator, scripted search cases, negative-input tests and cache recomputation
reduce that risk. The weakest remaining evidence is coverage of arbitrary
target callbacks and pathological numerical scales. A reproducible mismatch
under the supported contract would overturn the corresponding acceptance and
trigger a focused repair; failure of a new target to mix would instead call
for geometry/tuning investigation.

This was a source audit and skeptical self-review, not independent reviewer
certification. The initial result exposed implementation and interface defects,
not evidence against Stan adaptation or NUTS. The planned repair phase was
completed. No continuation veto fired and no remaining required validation is
being deferred. Future work should start from the checked interface and this
coverage matrix, rather than infer all-option completeness from the earlier
four arithmetic/integration tests.
