# Remaining execution gaps: source trace and discriminating diagnostics

Continue from `54bc96fc9` to answer which remaining gaps have an evidenced repair
and which still need an experiment. The existing 32 CPU / 52 GPU process-hour
caps, single numerical worker, source freeze during each worker, verified GPU
growth, automatic non-desktop GPU selection and stable campaign prefix apply.
This investigation changes tests and records only. External consumer files,
target semantics, numerical tolerances, packages and main remain unchanged.

Source inspection already identifies unwired public posterior/sequential
controllers, Python iterative/block decisions, static-shape geometry fitting
after dynamic row compaction, a host-owned permutation extent and callback-based
external deadlines. Inspect the current external source rather than reusing the
September 16 adapter description: its credit target now selects rectangular
SR-UKF. The original baseline and versioned TensorFlow random stream are
separate authorities; frozen comparison clouds preserve the numerical question.

The next diagnostics are deliberately narrower than repair qualification:

1. Test a diagnostic-only fixed-capacity Fisher-Yates permutation with a tensor
   active count against the existing `geometry_tf_philox_cpu_xla_v1` kernel.
   Require exact active permutations for empty, singleton and changing extents,
   untouched inactive suffix, a single trace, unchanged HLO and retained count
   and seed operands. Repeat on CPU and GPU; CPU stream identity is the
   authority on either device. Failure means an enclosing RNG repair is not
   yet demonstrated, not permission to change seeded draws.
2. Feed an exact quadratic's active rows and zero-padded rows to the existing
   fixed-shape fit. Record intercept, residuals and precision to determine
   whether padding is semantically harmless. This is an explanatory counterexample,
   not a new tolerance or accepted numerical implementation.
3. Probe a dynamic batch slice in XLA with changing counts and inspect its HLO
   and outputs. A compile failure or count-dependent executable documents why
   a polymorphic callback alone cannot close this boundary. Expected blocker
   observations are never reported as repaired functionality.
4. Attribute fresh-process geometry-fit host memory at build, trace, first
   execution, warm reuse and Python release for graph and XLA. Retain RSS,
   smaps residency, peak observations, graph sizes and weak references. Compare
   identical inputs and complete outputs, without calling a retained allocator
   leak-free or treating Python garbage collection as native code eviction.

Use registered 120/300-second test groups and versioned campaign directories,
at most three localized retries for each job. Plan at most six CPU and four GPU
workers, bounded by 3,000 seconds in total and the remaining cumulative caps.
Inspect the generated artifacts before interpreting them. Unsupported callbacks,
source changes, unexpected numerical disagreement, invalid artifacts and budget
exhaustion stop the affected diagnostic. A recorded unresolved result triggers
the next discriminating test; it does not relax a completion requirement.

Review before execution: component parity cannot establish public wiring;
padding changes reductions and possibly target calls; a constant-folded count
can make one trace look reusable; historical GPU-index assertions cannot
validate UUID-selected devices; callback heartbeats cannot enforce a compiled
call's wall time. These risks are explicit in the tests and source review.
This work will produce a gap-to-source-to-repair table and refresh the master
program. It will not establish whole-repository policy compliance, actual DZ5
transition qualification, timing superiority, HMC readiness or merge readiness.

After the first six workers, add one CPU follow-up group under the same limits.
Trace one actual sequential lifecycle and drop caller references, then clear
the lifecycle cache alone and finally its loaded inference dependency caches.
Record weak references and cache sizes at each stage. This tests Python callback
and graph ownership only; it does not attribute native compiler allocations.
Separately execute an exact AST excerpt of the current external pure progress
supervisor with its current constant values. Record source hashes and test both
an over-budget run with advancing progress and a quiet phase within its total
guard before a stage report arrives. These are source-excerpt diagnostics, not
an external worker integration test. Do not import or edit external modules.
A final CPU policy group completes the planned six CPU and three GPU workers
(including one formatting-only repeat); an audit is a separate short static
check. No extra numerical target or campaign budget is introduced.

## Findings and repair order

The known public execution gaps have concrete repair paths. They are not yet
installed or qualified. The main unresolved engineering choices are how much
static branching is affordable for exact active-row semantics, and how much
native XLA memory remains after Python-owned graphs have been released. Neither
question requires a change to the filtering algorithm, accepted-result
tolerances, or the owner's rejected-geometry decision.

| Gap and source | Confirmed cause | Repair and next acceptance test |
| --- | --- | --- |
| Geometry preparation to fitting: `quadratic_geometry.py:347–396`, `_quadratic_fit_kernel:656–736`; `quadratic_geometry_prepare_tf.py:118`, `quadratic_geometry_fit_tf.py:56` | Preparation carries fixed storage and active counts, but the public route compacts rows on the host and the native fit factory requires static training/holdout extents. Zero padding is numerically wrong for its current reductions. | Thread active counts through the complete fit, including intercept/loss/score means and the rank cutoff. Preserve compact QR+SVD semantics, callback order and exact replay. Compare complete original records for changing finite-row counts, zero/singleton/threshold cases, and corrupted inactive rows on CPU/GPU. Require unchanged HLO with runtime count operands. |
| Pilot callback extents: `quadratic_geometry.py:767–843`, `quadratic_geometry_pilot_tf.py:66` | Dropping zero directions changes the target batch extent. A dynamically sliced input specializes XLA even when TensorFlow reports one trace. | Use bounded static-shape dispatch where the callback requires the original extent, or an explicitly compatible batch-native masked callback. Do not add padded target evaluations or a scalar fallback. Preserve scalar/batch ordering. Measure graph growth across legal counts before selecting the implementation. |
| Permutation: `geometry_random_tf.py:153–158` | The public stream converts the retained count to a Python integer and caches by shape. | Move only the permutation extent/control into a fixed-capacity tensor program. Precompute the same stream/call-index seeds at configuration time and preserve draw order. The diagnostic proves exact CPU-stream permutations on CPU/GPU; production wiring and end-to-end seeded-cloud comparisons remain. |
| Iterative geometry: `quadratic_map_covariance.py:538–839`, loop at 622 | A Python loop makes recentering, incumbent and stopping decisions after each geometry result. | After the full geometry step exists, carry center/value/score, incumbent, attempt count and status through an ordered native loop; buffer completed records for reporting. Test every stopping branch and exact evaluation counts against original 3582b4ac. |
| Posterior public endpoint: `posterior_curvature_refinement.py:152–349`, evaluator 352–385 | The public endpoint explicitly requires eager mode and loops over partitions/replicates. `posterior_curvature_tf.py:181` and its report adapter exist but are not selected by this endpoint. | Connect host validation to the cached native controller and completed report. Test through the public endpoint, including malformed callbacks, eligibility failures, accepted full records and rejected/no-use cases. Retain the September 22 rejection criterion; do not reopen discarded-matrix equivalence. |
| Sequential public endpoint: `sequential_map_covariance.py:270–918`, loop at 459 | The locator is compiled, but the public outer refinement/terminal lifecycle is still Python. `sequential_lifecycle_tf.py:16` is qualified internally only. | Wire the native lifecycle and implement complete tensor report metrics and final mass preparation. The test-only materializer in `test_filter_repair_lifecycle_actual.py` still computes report metrics on the host and is not a production report adapter. Compare public payloads, progress records, budgets and fallback behavior; then qualify full costs. |
| Ordered blocks: `block_coordinate_center.py:259–560`, loop 325, host value 417 | Block closures capture the current center and call the public sequential initializer; transaction and reversal decisions are host-side. | Carry the current full state through ordered native block control, with static block topology and dynamic centers. Preserve heterogeneous block sizes, exact full-target replay and rollback. Test a coupled target where an earlier block changes a later optimum; unchanged closure-captured centers and parallel block updates must fail. |
| Python cache ownership: `sequential_lifecycle_tf.py:15`, refinement/terminal factory LRUs, `test_filter_repair_lifecycle_runtime.py:81–90` | Evicting only the outer lifecycle leaves target callbacks retained by nested factory caches; the existing ownership test globally clears dependencies before claiming release. | Introduce coordinated, bounded ownership of callback-dependent programs. Preserve active compiled handles and useful warm reuse. Test repeated target identities, outer/dependency eviction and weak references separately from native allocation measurements. A global cache sweep after every call is not an established runtime design. |
| Host memory and compilation | Most geometry-fit RSS growth occurs at first execution. The Python graph can be collected while resident memory stays high; the cache diagnostic separately proves Python callback retention. | First fix callback lifetime and measure graph/branch counts. Then compare same-signature reuse with distinct-signature churn in fresh processes, including post-eviction residency. Process lifetime is the reliable isolation boundary if native retention cannot be controlled. Do not call either plateauing RSS or Python collection proof of leak freedom. |
| GPU cost reporting | Six archived cost analyzers require device index3 although the runner now records UUID-selected devices. | Add versioned analyzers that validate identical physical UUID, TensorFlow provenance, source/input identities and clean preflights across paired arms. Negative tests must reject mixed UUIDs and shared-device timing. Preserve historical scripts/results and renew direction costs after the signed-word repair. |
| Actual DZ5 geometry callbacks and deadlines | External callbacks materialize tensors into NumPy and mutate Python telemetry; some limits depend on callback progress or treat elapsed guards as reporting-only. | Return tensor telemetry with value/score, aggregate it inside the numerical program and serialize after completion. Add an independent parent-enforced deadline and a declared bounded compile/run phase; do not manufacture semantic progress. Test blocked and healthy quiet workers, termination/cleanup, invalid-row accounting and actual fixed 18-batch consumers. |
| Whole-repository closure and integration | The source guard is partial, F01–F20 lack terminal endpoint dispositions, and remote main has advanced. | Map every active consumer endpoint to its implementation/dependencies and executable tests, refresh source coverage after integration, then run the required frozen-source suites and matched two-extent/three-process comparisons. A function's existence or source count cannot close a call-chain gap. |

The rank threshold in the original fit is
`math.ulp(1.0) * max(design.shape)` times the leading singular value. Increasing
the physical row count changes this cutoff as well as the reductions. Existing
active-row static dispatch in `ops/qr_lstsq_tf.py:77–86` and
`factor_correlation_geometry.py:666–683` is a useful implementation pattern,
but replacing this QR+SVD fit with a different COD solver is not justified by
those examples. Exact compact factorization behavior and compile cost both
remain acceptance questions.

## Executed diagnostics

All run paths below are under the shared
`/home/ubuntu/workspace/BayesFilter/docs/plans/artifacts/filter-gradient-repair-20260917/`.
Numbered `run.json` files preserve exact commands, source hashes, environment,
wall time and devices; `process.log` records TensorFlow 2.19.1, TF32 settings,
verified growth and managed-session GPU trust. GPU work used physical GPU2,
UUID `GPU-541e1e19-2df4-9064-4db9-9d0d2abc3eba`.

The stable command prefix is:

```text
/home/ubuntu/miniforge3/envs/tf-gpu/bin/python /tmp/bayesfilter-filter-gradient-xla-validation-20260918/scripts/run_filter_repair_campaign.py
```

Append `test --group <group> --device <device> --test-timeout-seconds 120`,
choosing CPU or GPU as recorded below. The groups and results are:

| Runs | Group | Result |
| --- | --- | --- |
|02627/02628|`gap_enclosure_diagnostics_cpu` / `_gpu`|Three checks pass on each device.|
|02629/02630|`gap_compiler_memory_graph_cpu` / `_xla_cpu`|One fresh-process fit per arm passes.|
|02631/02632|`gap_compiler_memory_graph_gpu` / `_xla_gpu`|One fresh-process fit per arm passes on the same UUID with clean preflights.|
|02633/02634|`gap_ownership_supervisor_cpu`|Two checks pass; 02634 repeats after import formatting and equivalent fixture-dictionary cleanup.|
|02635|`policy`|All 102 policy/controller checks pass.|
|02636|`audit` (static command)|2,990 working Python files / 2,989 parsed; one unchanged vendor-reference parse error.|

The fixed-capacity permutation uses capacity 17, active counts 0/1/2/7/16/17 and
seeds (173, 251)/(739, 911). All 24 device/count/seed cases match the existing CPU
stream exactly. Inactive suffixes stay unchanged, count and seed remain HLO
operands, and the executable is unchanged with one trace. This is a tested
diagnostic prototype, not installed runtime behavior.

The padding counterexample uses four rows of an exact quadratic and the same
rows padded to eight. Precision and rank 2 agree, but the intercept changes from
5.0 to 2.5 and loss from about 4.93e-32 to 6.25 on both devices. The dynamic-batch
probe at counts 3 and 11 returns correct results with one trace, but the count
operand disappears from HLO and the executable changes. Those are successful
blocker observations; neither test claims a repaired enclosing implementation.

The memory comparison uses D3, 24 training/8 holdout rows, identical frozen inputs
within each graph/XLA pair, and 20 warm repeats. The post-run analyzer
`analyze_gap_memory_20260922.py` issues `gap-memory-attribution-02632.json` after
checking complete result records at the unchanged 1e-10 absolute/relative
tolerance, exact discrete fields, matching source and device provenance.
Content-derived payload hashes remain independent per arm, as in the existing
fit comparison. Maximum numeric differences are 2.22e-15 CPU and 8.88e-16 GPU.

| Stage, resident host MiB from smaps | CPU graph | CPU XLA | GPU graph | GPU XLA |
| --- | ---: | ---: | ---: | ---: |
|Prepared inputs|574.172|574.121|960.832|960.141|
|Built|579.762|583.211|962.910|962.934|
|Traced|586.223|593.547|966.633|972.148|
|First execution|612.934|884.746|1133.504|1159.551|
|Reported|612.996|884.824|1133.570|1159.703|
|After 20 warm calls|613.078|884.953|1133.613|1159.785|
|After Python release|613.328|884.953|1133.988|1161.676|

The graph reference has 1019 nodes and XLA 1663; these arms use their respective
declared SVD implementations, so this is not an identical-graph JIT ablation.
First-execution times are 0.208/1.562 seconds CPU and 4.031/4.267 seconds GPU;
these single-process descriptive measurements cannot rank performance. GPU
allocator current bytes go from 9728 prepared to 39424 graph/25088 XLA after
execution, then 9728/9984 after release. Warm observed allocator peaks are
1,138,688/71,680 bytes. These are live allocator observations, not whole-device
reservation. All Python graph weak references expire, while RSS remains high.
`ru_maxrss` already exceeds `/proc` high water at preparation; use smaps stage
deltas for this attribution rather than interpreting it as a newly allocated
kernel peak. Native executable-cache versus allocator retention is unresolved.

The cache probe traces an actual factor-correlation sequential lifecycle. After
caller references are dropped, both callbacks, the lifecycle and its graph stay
alive. Clearing the lifecycle cache releases that program and graph, but both
callbacks remain. Clearing loaded inference dependency caches releases them.
Nineteen cache factories were populated by this fixture. No numerical lifecycle
was executed in this probe; it demonstrates Python ownership only.

## Current external source and limits

The inspected MacroFinance checkout is at `27d5c939652ea7e60731a001ed9b6e18c274edea`.
The old September 16 wrong-filter finding is stale:
`two_currency_double_zlb_credit_target.py:381` now calls
`tf_rectangular_srukf_value_and_score`, under
`rectangular_srukf_full_rank_identity_prepared_cir_v2`.
`two_currency_double_zlb_credit_estimation.py:137–146` explicitly describes its
qualification as target-only, with full-chain XLA diagnostic readiness false.
No external pin was changed and an actual HMC transition remains unqualified.

The separate geometry lane in `two_currency_double_zlb_dz5_identification.py:1597`
constructs scalar, three-start and fixed 18-batch runtimes. Callbacks in
`two_currency_double_zlb_dz5_sequential_recovery.py:461–482` and 875–944 call those
inner compiled targets but use `.numpy()`, NumPy reductions, Python counters
and `_record_rows:220–241`. They cannot be treated as enclosing-XLA callbacks.
Source trace establishes the incompatibility; an actual outer-call integration
test is still required after migration.

The exact source-excerpt supervisor test records source and constant-file SHA256
in `run-02634/gap-external-supervisor.json`. Its 650-second stage/3600-second total
guards only report crossings. At elapsed 3601 with advancing progress it does
not request termination; at 901 seconds without any semantic/stage report it
requests termination although the total guard has not fired. The boundary 900
case does not terminate. The parent consumes this boolean at
`scripts/run_two_currency_double_zlb_dz5_hierarchical_initializer.py:606`.
This is not an executed process-kill test.

Separately, `scripts/run_ccma_full_partition_center_sweep.py:464` enforces its
1800-second cap inside the progress callback; a same-process compiled call
cannot invoke that check until it returns. The center-first continuation
supervisor at `scripts/run_two_currency_double_zlb_dz5_center_first_continuation.py:693`
also polls semantic progress. These need independent parent deadlines before
buffered progress can be integrated safely. Initialization does not confer HMC
tuning authority; the public capability registry and tuning interface remain
binding for downstream consumers.

## Result review and next action

| Decision | Primary criterion status | Veto / uncertainty | Next justified action | Unsupported conclusion |
| --- | --- | --- | --- | --- |
| Keep the tensor-permutation repair candidate | Exact CPU-stream parity, runtime operands and stable HLO pass on both devices | Only capacity 17 and two seeds; public call/seed order not wired | Integrate with active-row geometry and expand count/boundary tests | Full initializer compilation |
| Reject naive padding and dynamic-slice reuse as closure | Numerical counterexample and count-specialized HLO reproduced on both devices | Static exact-shape branching may enlarge compilation | Prototype bounded exact-shape fitting and measure graph growth | A different solver or altered target calls are equivalent |
| Repair coordinated Python cache ownership | Outer-only eviction retains callbacks; dependency eviction releases them | Native memory behavior remains unisolated | Test bounded owner-scoped cache design plus signature-churn residency | All excess RSS is a leak or all of it is harmless caching |
| Migrate consumers and supervisors | Source trace and exact pure-supervisor tests expose incompatible telemetry and deadline semantics | Actual consumer/worker execution not yet tested | Tensor telemetry, parent deadline and real quiet/blocked-worker tests | Existing target-only qualification proves full transitions |
| Continue campaign; retain merge veto | Diagnostics identify implementation and harness debt, not failure of the scientific direction | F01–F20 and terminal comparisons remain open | Complete public wiring, costs, active call-chain audit and remote integration/retest | Repository-wide compliance, HMC readiness or merge readiness |

Post-run skeptical review: the strongest alternative explanation for retained
RSS is normal native allocation reuse, and the weakest memory evidence is one
small extent with no allocation-stack attribution. A distinct-signature churn
test after coordinated callback release can overturn a callback-only
explanation. The source-excerpt supervisor test cannot prove operating-system
termination behavior. These limitations block corresponding completion claims;
they do not invalidate the preserved diagnostics or authorize wider tolerances.

Final verification: focused Ruff for both new test modules and the post-run
analyzer, critical-error Ruff for the legacy driver, and whitespace checks pass.
The partial guard remains 213 sources/1306 exact exemptions. No runtime or
allow-list changes were made. All 14 diagnostic checks (including the two-check
repeat) pass, as do 102 policy checks. Total new charged time is 116.19 seconds,
including the 34.64-second static audit conservatively assigned to GPU by the
existing runner. Cumulative charges through 02636 are 52,486.56585271178 CPU /
49,656.2098863024 GPU seconds; 17.42 CPU / 38.21 GPU process-hours remain.

Remote main was refreshed to `01d67ec41062e6cf8759c5e099f45a5a7f3443f1`. Relative to
merge-base 3582b4ac, committed changes touch 408 repair and 183 remote code/test
paths, with overlap only in `bayesfilter/inference/hmc_warmup.py` and
`bayesfilter/nonlinear/experimental_batched_svd_sigma_point_tf.py`. This is a
path-overlap inspection, not a merge/conflict test. Integration and retesting
remain required; main is unmerged. The next implementation dependency is
active-count geometry with exact compact fit semantics.
