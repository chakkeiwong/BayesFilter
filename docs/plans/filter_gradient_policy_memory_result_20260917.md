# Filtering, analytical-gradient and XLA memory audit — 2026-09-17

**Repository-wide compliance fails.** The earlier Kalman/UKF numerical-loop
repair at `3582b4ac` does not cover the remaining particle, tensor-train,
preparation and gradient-consumer implementations. Several accessible paths
still use Python numerical iteration, NumPy outside reference work, or an
incomplete/non-default compilation boundary. Some score entry points use
autodiff and must not be described as analytical recursive LEDH scores.

**The four fresh GPU comparisons did not show a concerning XLA memory
increase or continuing warm device allocation growth.** They agree numerically.
This conclusion is limited to the measured fixtures; the unrolled TT/GenUT
programs and large histories have not received memory qualification here.

The complete [algorithm inventory](filter_gradient_algorithm_inventory_20260917.md)
lists the filtering families, variants, derivative mechanisms, model adapters
and reference implementations. The [plan](filter_gradient_policy_memory_audit_20260917.md)
defines the budget and evidence contract. Runtime repairs for the new findings
below remain open; this change adds the audit and measurement tools and evidence.

## Coverage and interpretation

The scanner parsed **2,694 of 2,695 tracked Python files**. The sole parse error
is a leading-zero literal in the vendored student file
`experiments/student_dpf_baselines/vendor/2026MLCOE/old_pt1_submission/filters.py:23`;
it is recorded, not silently omitted. The discovery includes 735 ordinary
owned modules, 713 document/benchmark harness modules, 74 package testing
modules, 911 tests, 245 external/vendor sources and 17 archived artifact
scripts. All owned modules parsed successfully.

The [static source inventory](artifacts/filter-gradient-policy-memory-20260917/static-complete.json.gz)
contains source hashes, functions/defaults/decorators, NumPy imports and calls,
Python loop sites, TF control/autodiff/host boundaries, lazy exports, and
84,301 resolved call or callable-reference edges. It records 24,784 loop syntax
sites and 244 NumPy import sites across the owned code and harnesses. **These
are search counts, not counts of policy violations.** In the package outside
`testing/`, 45 modules contain 46 NumPy import sites; reference and mixed roles
are distinguished below.

Static discovery covers the entire tracked Python tree. Contextual review
covered each algorithm family, the high-risk numerical loops and NumPy/JIT/AD
boundaries, and the consumer chains cited below. This is not a line-by-line
mathematical proof of every function or execution of every benchmark. Static
callback/factory resolution is best effort: dynamic dispatch, rebinding,
external callbacks and arbitrary consumer configurations remain limitations.
The [recorded static chains](artifacts/filter-gradient-policy-memory-20260917/reviewed-call-chains-complete.json)
are source evidence, not runtime wiring certification. Actual execution and
reachable GraphDef/HLO evidence are provided for the four memory fixtures.

Three different conditions matter:

1. Python numerical iteration executed eagerly, especially `.numpy()` feeding
   control flow, prevents a pure compiled enclosing filter.
2. A Python loop over static dates/parameters may be unrolled during tracing
   and then compiled by XLA. It still violates the requested ban on Python
   numerical loops and can multiply tracing/compiler memory. Successful JIT
   compilation alone does not repair this.
3. Schema construction, fixed tuple plumbing, artifact writing and host
   orchestration are not numerical filtering loops. Explicit independent
   reference/diagnostic NumPy is permitted. Numerical preparation and admission
   logic are not exempt merely because they execute on the host.

## Confirmed findings and repair priorities

`P1` means repair or enforce an explicit historical/reference boundary before
new default/candidate use. `P2` means preparation, API-default or consumer debt
that must be closed for the relevant complete route. These priorities concern
execution policy, not proof that the underlying mathematics is wrong.

| ID | Priority | Source evidence and affected route | Finding and consequence |
|---|---|---|---|
| F01 | P1 | `highdim/cubature_genut_batch_tf.py:2147`, `:2152`, `:1725`, `:329`, `:1401`, `:1455` | `batch_finite_value_score` performs a Python loop over parameters and calls `ForwardAccumulator`; statically shaped value evaluation unrolls time, Sinkhorn and higher-moment iterations. The actual consumer is `GenUTNeuTraTargetAdapter.neutra_batch_log_prob_and_grad_status` → `_posterior_value_score_status` (`cubature_genut_neutra_targets.py:394`) → this endpoint. The score is AD of the finite value program, not the required analytical recursive LEDH score. A separately named manual-JVP diagnostic is not authority to silently substitute it. |
| F02 | P1 | `highdim/ledh_contract_e_canonical_lgssm_tf.py:217`, `:243`, `:411`, `:526` | The factory defaults to XLA and the outer time recurrence is a `tf.while_loop`, but reachable Cholesky/solve/flow/weight derivative helpers still iterate over `PARAMETER_COUNT` in Python. This is numerical graph unrolling inside an otherwise compiled route. Vectorize the parameter dimension or carry it through a native TF loop. The older date loops at `:782` and `:1121` are separate helpers; they are not the factory's current time recurrence. |
| F03 | P1 | `highdim/squared_tt_adjoint_engine_tf.py:154`, `:315`, `:361`; `squared_tt_engine_v0_tf.py:234` | The manual analytical TT score uses Python forward dates, reverse dates and reverse ALS updates; its traced forward fitting also uses Python sweep/core iteration. `.numpy()` telemetry appears within the calculation. The adjoint benchmark at `docs/benchmarks/run_p2_t120_adjoint_stress_20260817.py:112` calls this implementation. It is not an end-to-end compiled analytical score. |
| F04 | P1 | `highdim/squared_tt_engine_xla_tf.py:119`; `squared_tt_engine_adapted_xla_tf.py:170`; `squared_tt_engine_gaussian_xla_tf.py:488`, `:672` | The `*_xla` engines compile per-step functions while retaining Python date recurrences, ALS unrolling and host-computed numerical controls. Gaussian tau construction consumes `float(rms.numpy())`. The names do not establish compilation of the complete filtering evaluation. The eager v0 engine already documents its limited role; compiled wrappers still inherit numerical preparation dependencies. |
| F05 | P1 | `highdim/zhao_cui_actual_sv_batched_tt_tf.py:213`, `:536`, `:568`, `:575`, `:699`, `:725`; `zhao_cui_fixed_adjacent_tt_tf.py:170` | Batch-native actual-SV TT value and analytical directional replay retain Python time, parameter and ALS-sweep loops. A separate value-score endpoint at `:345` uses `GradientTape`; distinguish the two APIs. Batch-native tensor algebra does not remove this unrolling. |
| F06 | P1 | `highdim/zhao_cui_frozen_proposal_apf_tf.py:855` | `FrozenProposalAPFProgram.compiled` defaults to XLA (`:756`), but `_evaluate_core` iterates through dates in Python. Its explicit recursive score is therefore still statically unrolled. In contrast, the predator-prey fixed-variant evaluator has a native time loop at `zhao_cui_predator_prey_fixed_variant_tf.py:1068`; do not transfer the APF finding to that recurrence. |
| F07 | P1 for reuse | Experimental `tf_tfp/filters/bootstrap_pf_tf.py:48`, `:58`; `dpf_ot_tf.py:50`; `ledh_pfpf_ot_tf.py:68`; `structural/structural_filter_tf.py:60`; `ledh_pfpf_alg1_ukf_tf.py:360`, `:396`, `:824`, `:1783` | General particle/flow filters use Python dates, particles, pseudo-time or covariance rows. Bootstrap resampling depends on `bool(...numpy())`. The general enclosing programs cannot be made fully XLA simply by adding a decorator. Explicit reference diagnostics may remain isolated; these functions also have runnable benchmark callers, e.g. `scripts/filtering_value_gradient_benchmark_run_p8_numeric.py:813`. |
| F08 | P1 for canonical claims | Experimental `tf_tfp/filters/experimental_batched_ledh_pfpf_ot_tf.py:1695`, `:1787`; streaming companion `:700` | Dense batched LEDH unrolls dates and its score wrapper uses `GradientTape`. Streaming has native TF loops but its enclosing score also uses a tape. File headers describing a production/default direction do not override the post-2026-08-21 analytical-score and canonical-rebuild policy. No historical LEDH result was reused as evidence here. |
| F09 | P2 | `highdim/sqmc_tf.py:74`, `:76`, `:88`, `:93`, `:112`, `:141`, `:168`; `ledh_pfpf_genut_initialization_tf.py:514` | Hilbert bit/axis/sort-word calculations and an initialization-design filter's time recursion use Python numerical loops. The SQMC routines are invoked by initialization/ancestry paths; they are not merely a plotting helper. Use native tensor/bitwise recurrences, preserving the same ordering. |
| F10 | P2 | `hardbound/dns_curve_tf.py:20`; `highdim/retained_moments_tf.py:48`; `squared_tt_engine_v0_tf.py:128` | NumPy constructs Gauss-Legendre rules; the TT engine additionally uses NumPy meshgrid/product. DNS has a direct chain from joint target → observation mean → yield curve → NumPy rule. Retained moments feed the adapted TT engine. A “host constants” comment does not satisfy the NumPy policy. Existing TF quadrature primitives are possible replacements after node/weight parity checks. |
| F11 | P2 | `highdim/retained_moments_tf.py:61`, `:82`, `:89`, `:92`; `nonlinear/fixed_sgqf_tf.py:737`, `:743`, `:746`, `:795` | Retained mean/covariance contractions and SGQF cloud/node/weight construction perform numerical loops in Python. SGQF's compiled filtering recurrence is separate and should not be mislabeled broken. Configuration-time arithmetic remains migration work under a strict all-numerical-loops interpretation; immutable tensor inputs should be prepared without a NumPy/runtime fallback. |
| F12 | P2 | `backends.py:120`, `:125`, `:128`; `highdim/ledh_contract_e_identity.py:295`, `:574`, `:576` | Spectral derivative/HMC certification and factory prepared-input serialization/finite/positive checks use NumPy. These are admission/identity paths, explicitly outside the diagnostic-only exemption. They need TF validation and standard-library/TF serialization, not an XLA-wrapped metadata system. |
| F13 | P1 on the derivative branch | `nonlinear/fixed_sgqf_structural_adapter_tf.py:100`, `:109` | `tf_predator_prey_to_fixed_sgqf_model(..., with_derivatives=True)` supplies `GradientTape.batch_jacobian/jacobian` without `experimental_use_pfor=False`. Thus model-local derivatives are AD with implicit pfor, even though the receiving SGQF recursion is analytical. The default `with_derivatives=False` branch does not execute them. No applicable written pfor exception was established by this review. |
| F14 | P2 / classification | `highdim/filtering.py:4562`; `ledh_contract_e_tp_structural_tf.py:526`; `highdim/models.py:1411`; `zhao_cui_austria_sir_parameter_density_training_tf.py:2772` | Further implicit-pfor Jacobian/vectorized-map sites exist. They include reference/model-local/training paths; their enclosing default use and any valid exception need individual classification. Presence of an API alone is not proof that every filter reaches it. |
| F15 | P1 for new training use | `inference/cpu_value_score_pool.py:102`, `:112`, `:133`, `:145`, `:221`, `:340`, `:384` | The pool defaults to scalar row dispatch, hardcodes non-XLA TF wrappers and uses NumPy for packing/concatenation/checks. `docs/benchmarks/run_ssl_lstm_neutra_complexity_training_2026_07_19.py:346` constructs the default scalar configuration. Even the optional batch-native branch hardcodes JIT off. The CPU sharding exception requires batched target shards; scalar workers do not qualify. This is a consumer violation, not a failure of repaired Kalman kernels. |
| F16 | P2 | `adapters/bgs.py:251`, `:260`; `inference/hmc_warmup.py:2574`, `:5950` | Public prior-value/score wrappers omit `jit_compile`; direct warmup/bracketing helpers default it to false. An outer compiled caller can absorb an inner TF function, so this is default-boundary debt, not proof that every current HMC chain runs without XLA. Existing compiled HMC qualification remains scoped to its actual endpoint. |
| F17 | P2 | `hardbound/joint_target_tf.py:89`, `:143`, `:210`, `:252`; `independent_score/sir_observation_simulator_tf.py:64`, `:103` | Latent joint-target and score-data simulation recurrences still use Python time/substep loops. The latter has an XLA wrapper but unrolls its body. These are adjacent differentiable/data algorithms, not additional marginal filters. |
| F18 | P2 / auxiliary migration | NumPy call inventory in `inference/{posterior_local_initializer,quadratic_geometry,sequential_map_covariance,joint_center,block_score_geometry,hmc_identity,target_failure_policy}.py` and other inference modules | NumPy also appears in target validity, geometry/preparation and identity functions. Diagnostic-only modules such as `backend_parity` require different treatment. The machine inventory lists every site; sampler/tuner-wide mathematical and runtime requalification is outside this filter audit. |
| F19 | P2 / signature boundary | `linear/kalman_covariance_derivatives_tf.py:181`; `linear/correlated_kalman_tf.py:724`; `highdim/ledh_contract_e_reset_tf.py:483`; `zhao_cui_moment_teacher_xla.py:243` | These public/helper decorators enable XLA but omit an explicit `input_signature`, often using `reduce_retracing=True`. This does not establish bounded tracing for arbitrary repeated direct calls. A stable outer factory can supply the required boundary, as in the memory workers; consumers without that boundary need signatures or documented bounded specialization. The static ledger records all declarations without claiming every such helper violates its enclosing call contract. |
| F20 | P2 | Experimental `tf_tfp/filters/experimental_batched_ledh_pfpf_ot_streaming_tf.py:764`, `:770` | `_is_static_all_false_bool_tensor` calls `.any()` on the NumPy array returned by `tf.get_static_value`. This is an example of NumPy numerical/control computation without an explicit NumPy import. Import counts are therefore not an upper bound on violations. Use a TF reduction followed by scalar materialization at the preparation boundary. |

The paths `highdim/`, `nonlinear/`, `linear/`, `hardbound/`, `inference/` and
`adapters/` in this table are under `bayesfilter/`; `tf_tfp/` is under
`experiments/dpf_implementation/`.

The factory-named LGSSM Contract E route also composes finite-program flow/reset
JVPs. Its name and reset identity do not establish the post-2026-08-21 required
analytical recursive LEDH score or full Algorithm-1/GenUT covariance lifecycle.
Those conformance questions remain separate from repairing F02's execution
loops; a compilation repair cannot grant scientific admission.

## Cases that must not become false positives

- The NumPy Kalman, sigma-point and bootstrap implementations in
  `bayesfilter/filters/` explicitly declare reference roles. Their companion
  `linear/types.py` and `results.py` are imported by those reference routines;
  the repaired TensorFlow kernels use their TF counterparts. Public exports
  still need clear reference naming/documentation, but these imports do not
  prove NumPy execution inside the repaired TensorFlow recurrence.
- `generalized_sv_sgqf_tf.py:272` uses NumPy in the explicitly named dense
  **reference** function. Its manual SGQF recurrence itself uses TensorFlow and
  `tf.while_loop`. However, the reference import is at module scope; separating
  it avoids contaminating runtime import closure. Similarly,
  `native_generalized_sv.py` and `sir_latent_preclip_reference_tf.py` explicitly
  document reference roles. `highdim/__init__.py:158` eagerly imports the former,
  so import isolation is still needed for strict admitted-route closure.
- `sv_mixture_cut4.py:3632` has a NumPy quadrature helper: classify the caller's
  TT/comparator/preparation role, rather than claiming every UKF in that module
  executes NumPy.
- SGQF's post-run conversion of tensor histories into result records is
  reporting. It is not the time recurrence. Its cloud construction is numerical
  preparation and is separately identified in F11.
- The scalar-SV Algorithm-1 module's explicit `jit_compile=False` declaration
  is a debug alternative; the public selector defaults to true at
  `ledh_pfpf_alg1_ukf_tf.py:1652`. A search for `False` alone gives the wrong
  default verdict.
- Non-JIT diagnostic telemetry in `predictive_equivalence.py`, HMC verification
  summaries and testing modules is not evidence that an enclosing target is
  uncompiled. `inference/hmc.py:2758` chooses the compiled branch when `use_xla`
  is true. Implicit-pfor sites are separately recorded; an explicit
  `experimental_use_pfor=False` is not a pfor violation.
- A Python loop over epochs that dispatches a complete batched compiled
  optimizer step is orchestration. A loop over samples inside a target or over
  dates/parameters inside a filtering calculation is numerical iteration.

## GPU memory comparison

Eight fresh processes ran on visible device 2, an NVIDIA GeForce RTX 4090,
using `/home/ubuntu/miniforge3/envs/tf-gpu/bin/python`, TensorFlow as recorded in
each JSON, float64, TF32 enabled (irrelevant to these FP64 arithmetic kernels),
two intra-op threads and one inter-op/OpenBLAS thread. Memory growth was
configured and verified before importing numerical fixtures. Fixtures were
deterministic, with no random sampling or tuning.

Non-XLA means `tf.function(jit_compile=False)`, not eager execution. Both arms
use the same numerical implementation, inputs, shapes and outputs. The off
arm has no reachable `_XlaMustCompile=True`, explicit XLA op or Python callback;
global automatic JIT is off. Each on arm executed with outer must-compile true
and exported optimized HLO. Source hashes match within each pair except import
formatting in the first rectangular worker; numerical module sources match.
Output elementwise checks use `atol=rtol=1e-10`; all values are finite.

| Fixture | Shapes | Peak TF GPU allocation, graph / XLA (MiB) | XLA / graph | Cold host RSS, graph / XLA (MiB) | Maximum absolute output difference |
|---|---|---:|---:|---:|---:|
| Rectangular SRUKF value + analytical score | B=8, P=3, state=2, observations=2, T=96; fixed full-rank chart | 0.100 / 0.059 | 0.586 | 1047.2 / 1028.2 | 7.11e-15 |
| Direct factor SRUKF value + analytical score | B=8, P=3, state=2, observations=2, T=96 | 0.131 / 0.064 | 0.488 | 1037.2 / 1018.5 | 8.88e-15 |
| Covariance Kalman value + analytical score | B=8, P=32, state=24, observations=12, T=96 | 15.573 / 16.173 | 1.039 | 999.8 / 1000.8 | 1.42e-13 |
| Analytical finite Sinkhorn JVP primitive | B=4, N=128, state=4, P=4, 16 iterations; K=N=128 | 12.138 / 8.203 | 0.676 | 1013.9 / 988.5 | 3.33e-16 |

Peak is the maximum of the cold and per-call warm TensorFlow allocator peaks;
it is not total driver reservation. Outputs are synchronized and released on
the same schedule in both arms. TensorFlow peak statistics are reset before
each measured phase/call. Tracing happens before the `after_trace` snapshot;
HLO text export happens after the measured phases.

Every arm traced exactly once. Over 20 warm calls, device **current** allocation
had a zero-byte range in all eight processes. Kalman/UKF host RSS grew by
0–36 KiB after cold-output release. Sinkhorn's host RSS grew by 648 KiB (graph)
and 756 KiB (XLA), then plateaued by iterations 10 and 5 respectively; host
serialization/allocation caches are a plausible contributor, not a proven
allocation attribution. No fixed-shape continuing growth was observed in the
measured window. The workers consumed 82.48 seconds total inside their recorded
measurement processes; preliminary attempts also remain well within the
20 GPU process-minute budget.

The slightly higher XLA Kalman device peak is **3.86%**, about 0.60 MiB, while
warm current allocation is lower (178,688 versus 2,925,312 bytes). This is
consistent with different temporary-buffer scheduling and lifetime; it is not
evidence of a leak. Exact buffer attribution was not measured. The small UKF
device footprints and approximately 1 GiB host RSS reflect that TF import,
device context, graph/runtime and compiler memory are separate from live
device tensors. None of the comparisons hits the predeclared >2x peak-device
or >256 MiB extra-host explanatory alarm.

One telemetry oddity remains explicit: `resource.ru_maxrss` is lower than
`/proc/self/status` RSS/HWM in these processes. Both are retained in the raw
JSON. The table consistently uses `/proc` metrics; this accounting discrepancy
must not be interpreted as filter memory growth or memory release.

This measurement does not cover singular-rank rectangular branches, FP32/TF32
production arithmetic, large particle counts, full-history reverse scores,
many-shape compilation caches, long-running training/HMC, or every TT/LEDH
route. No old LEDH artifact was used as a memory baseline. The Sinkhorn case is
a primitive diagnostic, not a canonical LEDH qualification. There is no
repository-wide leak-free or throughput claim.

## Systematic repair sequence

| Order | Concrete work | Evidence required before closing it |
|---|---|---|
| 1 | Classify each owned endpoint as runtime, preparation, explicit reference, historical, or reporting; record score semantics. Prevent AD/legacy LEDH wrappers from issuing analytical/canonical claims. | Consumer-to-implementation wiring checks, including factory-produced callbacks; source-grounded LEDH rebuild checks under current policy. |
| 2 | Replace Python time, particle, pseudo-time, parameter and ALS iterations in F01–F09 with batch-native tensor operations or bounded `tf.while_loop`/`tf.scan`. Pack variable TT core shapes with explicit masks where necessary. | Same-input value/analytical-score parity, fixed-branch numerical checks, graph-size scaling with horizon/parameter count, and complete-call GPU HLO. A decorator or renamed function is insufficient. |
| 3 | Remove NumPy from quadrature preparation, retained moments, admission/identity and runtime consumers; isolate independent references from admitted import closures. | Node/weight/moment parity and serialization/identity compatibility checks. Preserve historical identities/artifacts rather than silently rewriting their semantics. |
| 4 | Make stable-signature full numerical target factories default to XLA; retain explicit labeled diagnostic exceptions. Repair scalar/non-XLA target pools and pfor callbacks on their actual consumer paths. | Callable-default tests, executed complete GraphDef/HLO, no PyFunc/NumPy callbacks, bounded tracing, batch-size>1 training and analytical-gradient provenance. |
| 5 | Repeat fresh-process memory diagnostics on the repaired TT/GenUT routes at growing T, P, N and history modes. | Output parity first; separate tracing/compiler RSS, allocator peak/current, output lifetime, warm growth and shape-cache growth. Preserve exact chunk policy and memory growth. |
| 6 | Extend the existing Kalman/UKF guard to the classified complete filter/score graph and numerical preparation. | Fail on new unclassified numerical loops/NumPy/non-XLA defaults; permit narrowly identified schema/reporting/reference sites. Do not add an enormous automatic whitelist based on filenames. |

This sequence preserves score definitions and numerical branches. In
particular, replacing a finite-program derivative with a statistical score,
changing Contract E reset terms, substituting an unrelated TT route, or
retuning numerical controls is a scientific change, not a loop repair.

## Reproduction, failures and terminal assessment

Tools added:

- `scripts/audit_filter_gradient_policy.py` — standard-library AST discovery,
  including relative imports, lazy exports and callable-reference edges.
- `scripts/measure_filter_xla_memory.py` — fresh-process worker with explicit
  device placement, memory-growth verification, actual tracing, graph checks,
  source/command/environment manifest, outputs and HLO.
- `scripts/summarize_filter_xla_memory.py` — finite/output parity, source and
  graph-boundary checks plus comparison metrics.

Example worker command, substituting the four fixture names and `on|off` into
fresh output paths:

```bash
TF_FORCE_GPU_ALLOW_GROWTH=true CUDA_VISIBLE_DEVICES=2 \
TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=1 OPENBLAS_NUM_THREADS=1 \
timeout 180 /home/ubuntu/miniforge3/envs/tf-gpu/bin/python \
  scripts/measure_filter_xla_memory.py --device GPU --gpu 2 \
  --fixture covariance --jit on --output /tmp/filter-memory-new/covariance-on.json
```

Exact executed commands, environment, hardware, source hashes, phase memory,
timings and outputs are in [memory-v3](artifacts/filter-gradient-policy-memory-20260917/memory-v3).
The [comparison JSON](artifacts/filter-gradient-policy-memory-20260917/memory-comparison.json)
is the machine-readable authority for the table. `ruff check` passes for the
three new scripts. The earlier 205 Kalman/UKF test executions are prior scoped
evidence, not a new whole-repository test result.

The initial memory attempt imported TF constants before memory-growth setup
and correctly failed closed. The first successful tiny off-only attempt had a
misplaced trace snapshot and insufficient parity/closure checks; it is
preliminary and excluded. The corrected workers configure growth first,
measure actual tracing, inspect inner compilation and compare complete numeric
outputs. Initial static scans also overselected modules because the package
name contains “filter,” inferred loop roles too aggressively, omitted benchmark
source ownership and encountered non-dictionary lazy-export declarations.
Those heuristic counts are superseded by `static-complete`; they are not
violation evidence. Intermediate artifacts are retained for recovery.

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Reject a repository-wide Python-loop/NumPy/XLA compliance claim | Failed by the concrete source paths above | Numerical Python loops, nonreference NumPy and AD analytical-label risks remain | Arbitrary callbacks and historical/default consumer classification | Execute the ordered migration with route-specific parity and wiring checks | No blanket claim that every filter is wrong |
| Accept the four memory comparisons as bounded engineering evidence | All finite outputs match; correct graph/JIT distinction; one trace per arm | No parity, callback, retracing or continuing allocation-growth veto observed | One process per arm, limited sizes/dtypes/branches and 20 warm calls | Extend to repaired unrolled routes during their qualification | No repository-wide leak-free, performance or scientific admission claim |

Post-run review: the strongest alternative explanation for the reassuring
memory result is limited coverage—the problematic unrolled full TT/GenUT
programs were not the measured complete filters. Larger horizons or parameter
counts could expose compilation-memory growth that these fixtures miss.
Conversely, the concrete Python/NumPy source findings do not establish an XLA
compiler defect. The static review was performed by the primary agent;
independent multi-agent review was not performed.
