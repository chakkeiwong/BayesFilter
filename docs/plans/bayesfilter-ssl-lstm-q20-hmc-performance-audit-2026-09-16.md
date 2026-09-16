# q20 HMC execution and performance audit

Date: 2026-09-16. Status: completed; source audit and one bounded GPU diagnostic.

## Question and scope

Explain the actual master run's runtime: XLA coverage, Python loops, independent
chain scheduling, NumPy/callback use, target work and redundant evaluations.
Inspect the executed numerical source `71e0fba399489a8f25fcbdea1185f6bbb600e487`
in `/tmp/BayesFilter-q20-master-repair-20260916` (current HEAD `b6bf8c76` adds
documentation only). Preserve all concurrent main-worktree changes. This is a
performance diagnosis, not a change to the financial model or numerical policy.

## Evidence contract and intent ledger

The comparator is the existing four scalar-chain serial public HMC runner with
the q20/T30 float64 `tensorflow_eigh_strict` target, beta one, L=3 and epsilon
0.01. Those values reproduce the existing qualification; they are inherited
diagnostic settings, not a tuned posterior kernel. Inspect the existing batched
reusable runner with the same state bank and health fields as a diagnostic.
Its random stream topology differs, so identical seed integers do not establish
pathwise parity. Neither topology is promoted by this diagnostic.

Pass criteria: source inventory matches saved execution; the measured graph has
XLA enabled and no Python callback operations; timed results are synchronized
and valid. Report first-call and repeated-call times separately. Timings, graph
counts and low-level cost estimates are explanatory only. Invalid target or
proposal status, nonfinite outputs, source mismatch, absent GPU memory growth,
resource contention or the external deadline stop the numerical probe. A
failed diagnostic is a repair trigger, not evidence against HMC or NeuTra.
No posterior convergence, whitening, numerical-backend equivalence, statistical
speed ranking, or production-readiness conclusion is authorized by this work.

## Bounded measurement

Use the existing `tfgpu` environment and a trusted idle GPU of the same hardware
class. Set `TF_FORCE_GPU_ALLOW_GROWTH=true` before importing TensorFlow and verify
the repository memory policy. One diagnostic worker has a 595-second execution
deadline and five-second termination grace (600 seconds total); these are
convenience resource limits inherited from the master attempt cap. Allow a
further 60 seconds for readiness and settlement, charged within the existing
allowance. At most one numerical launch; a failure is reported before another
attempt is planned. No renewal of the old master's attempt allowance.

Record first call plus two repeated synchronized calls for target batches one
and four and for two-transition serial/batched HMC chunks. Two transitions and
two repeats are convenience limits for cost attribution, not statistical
evidence. Use qualification start generation and the existing scoped seed
derivation, preserving starts in the result. Inspect the reachable TensorFlow
graph functions, input signatures, XLA attribute and callback operations.
Optionally time the exact checked 80-by-80 factor/derivative operation on the
initial target covariance as a component diagnostic; do not extrapolate that
single covariance to an exact runtime share along all filter trajectories.

Command template:

```text
timeout --signal=TERM --kill-after=5s 595s env CUDA_VISIBLE_DEVICES=<idle GPU> TF_FORCE_GPU_ALLOW_GROWTH=true /home/ubuntu/anaconda3/envs/tfgpu/bin/python docs/benchmarks/diagnose_ssl_lstm_q20_hmc_performance_2026_09_16.py --source-root /tmp/BayesFilter-q20-master-repair-20260916 --output-dir docs/plans/artifacts/ssl-lstm-q20-hmc-performance-2026-09-16/attempt-001
```

Starting balances from the previous settled ledger: campaign
123883.99030228473 seconds, diagnostic 42907.537327354854 seconds. This plan's
660-second upper reservation leaves 123223.99030228473 and 42247.537327354854
seconds until measured charges settle. Diagnostics are included in campaign
time. Preserve the previous ledger; write this phase's settlement separately.

## Skeptical audit before execution

The existing 181-second public qualification is a first call, so it cannot be
called steady throughput. The existing 36-second repeated batched qualification
already excludes that explanation. Scalar versus batch timing alone cannot
prove a posterior speed improvement; the planned output reports execution cost
only. The 449-hour reservation includes training/heldout validation and a factor
of two, and is not a measured HMC duration. Compiler elimination may remove
duplicate source calls, so source counts will not be stated as exact executed
call counts. Fixed Jacobi loop counts are derivable from code, but device
profiling would be required to assign exact runtime percentages. Start-bank
setup, compilation and output synchronization are accounted for separately.
Verdict: proceed with the bounded diagnostic under these limits.

## Findings and result

The numerical HMC loop is XLA compiled. The actual public q20 runner leaves
independent chains serial, and expensive filter work is requested repeatedly
for value/score and health telemetry. Compilation alone does not fix either
problem. The existing batched reusable HMC implementation took about one third
of the serial execution time in this diagnostic with the health checks retained.

The worker completed in 354.691454 seconds on host GPU 1, an RTX 4080 SUPER,
TensorFlow 2.20.0, with verified memory growth. The source inventory exactly
matched the executed master. All 25 inspected source files also match main;
see [source inventory](artifacts/ssl-lstm-q20-hmc-performance-2026-09-16/source-audit.json).
No numerical runtime code or production configuration was edited.

### Measured execution cost

All repeated times below synchronize returned tensors. HMC times include its
host runner and diagnostic construction; subsequent health assertions are
checked outside the timer. Both HMC topologies have four chains, two transitions
per chain, L=3, epsilon=0.01, beta=1, the same physical start bank and the same
float64 strict backend. Scalar and batched random stream layouts differ.

| Operation | First call, seconds | Repeated calls, seconds |
| --- | ---: | ---: |
| One target value/score/status row | 8.584 | 1.172, 1.172 |
| Four rows evaluated together | 9.254 | 1.508, 1.508 |
| Four serial scalar-chain HMC runners | 130.114 | 53.532, 53.442 |
| One batched four-chain HMC runner | 35.991 | 16.568, 16.467 |
| Checked 80-by-80 factor and four derivative directions, batch four | 4.459 | 0.02216, 0.02194 |

The observed warm HMC time ratio is about 3.24. This is descriptive evidence
from two short repetitions, not a supported ranking of posterior efficiency or
a general speedup guarantee. First-call times mix tracing, compilation and
execution. Each target and HMC graph traced once across all three calls;
uncontrolled retracing was not observed. The separate scalar runners each own
a graph; the timings do not prove exactly four independent XLA compilations.

Every inspected concrete HMC/target function has `_XlaMustCompile=true`, a
static input signature and no `PyFunc`, `EagerPyFunc` or host-compute callback
operation in its GraphDef or nested function library. Both topologies passed
the existing accepted/proposed state, value, score and target-status checks.
Singleton versus batched target values and scores also passed the existing
1e-9 relative / 1e-10 absolute comparison for the first start. This checks
one input, not all possible posterior states. TensorFlow's measured allocator
peak was 268,792,576 bytes in this bounded run; this is not a long-chain memory
capacity result.

Machine-readable evidence:
[result](artifacts/ssl-lstm-q20-hmc-performance-2026-09-16/attempt-001/result.json),
[execution and readiness](artifacts/ssl-lstm-q20-hmc-performance-2026-09-16/execution.json),
and [diagnostic script](../benchmarks/diagnose_ssl_lstm_q20_hmc_performance_2026_09_16.py).

### The actual call path

| Layer | Checked source and consequence |
| --- | --- |
| Master qualification | `q20_hmc_qualification.py:50` builds a batched compiled diagnostic; line 95 explicitly runs the public chain runner with `mode="serial"`. |
| HMC pricing | `q20_master_stages.py:194` also explicitly selects serial mode. The final master stopped before reaching this downstream pricing stage. |
| Tuning and retained sampling | `q20_production_hmc.py:83` does not override `HMCCandidateExecutionConfig.chain_mode="serial"` (`hmc_candidate_set_execution.py:128`). `_run` at line 436 builds the independent-chain runner. Both retained blocks (retained module line 369) and the sequential member controller (line 131) call that binding. |
| Independent chains | `hmc.py:3116` constructs one reusable runner per chain; line 3212 calls them through a Python serial comprehension. The available threaded option is not selected. No multi-GPU chain distribution is present. |
| Numerical chain | `hmc.py:2942` creates TFP HMC and `tfp.mcmc.sample_chain`; line 2750 wraps the whole chain with stable-signature XLA. TFP's leapfrog integrator uses `tf.while_loop` (installed `leapfrog_integrator.py:291`). |
| Analytic target | `reviewed_value_score_target_fn` (`batched_value_score.py:272`) calls the adapter once and supplies its computed analytic score via a custom gradient. HMC does not reverse-differentiate the complete filter recursion. |
| Bridge and filtering | `FixedBetaBridgeAdapter.log_prob_and_grad_status` (`tempered_target_tf.py:585`) reshapes the supplied state into a batch. `ssl_lstm_complexity_batched_target_tf.py:560` enters the batched analytic filter through `batched_svd_sigma_point_tf.py:23`. The time recursion is `tf.while_loop` in `experimental_batched_svd_sigma_point_tf.py:3401`. |

Thus the answer to “fully JIT compiled” depends on scope: the repeated numerical
chain is compiled; the campaign, candidate selection, independent-chain
scheduler, chunk control, diagnostic summaries and file writing remain Python
or host TensorFlow operations. There is no Python loop dispatching each
leapfrog or each filter observation in the measured path. The outer chain loop
does dispatch a complete compiled scalar-chain chunk at a time.

The separate replica-exchange/ensemble implementation is different. It builds
batched within-temperature kernels (`q20_production_hmc.py:290`) and a compiled
transition program (`tempered_transitions_tf.py:1125`). Its Python temperature
loop constructs graph branches when traced; it is not a per-transition host
loop. This path was source-inspected, not timed or posterior-qualified here.
The filtering time axis and an individual Markov chain's transition axis are
sequential by their recurrences. Parallelism is available over independent
chains, proposals, sigma points and matrix operations.

### Expensive work inside the compiled target

q20 contains 20 latent, 20 hidden and 20 cell coordinates, giving a 60-dimensional
filter state. `_batched_components` adds 20 process-noise coordinates; the
placement covariance passed at filter line 2865 is **80-by-80**. The HMC parameter
dimension is four, but that is not the matrix dimension used by its likelihood.

For the selected `tensorflow_eigh_strict` backend,
`_checked_batched_principal_sqrt_factor_first_derivatives` (line 1217) requests:

1. A refined eigensystem to classify the covariance (line 1247).
2. Another refined eigensystem to construct its principal root (line 1402,
   entering `_tensorflow_strict_principal_sqrt` at line 808).
3. Another refined eigensystem of that root for its Sylvester derivative solve
   (`_tensorflow_strict_symmetric_sylvester_solve` at line 927).

Each refined solve first calls `tf.linalg.eigh`, then executes eight full
Jacobi sweeps (`_refined_symmetric_eigh`, line 473). For dimension 80 there are
79 rounds per sweep: **632 sequential rotation rounds per solve**. The schedule
is built in Python once while tracing; the rounds execute in a compiled
`tf.while_loop` (line 621). Every round permutes/gathers matrix elements,
performs rotations and updates eigenvectors. There is no residual-based early
exit. Across three solves and 30 observations, the source requests
`3 * 632 * 30 = 56,880` batched rotation rounds per target evaluation, in addition
to the initial eigensolves and all other filter/derivative algebra. This is a
source-derived work count, not a profiler count after compiler optimization.

The 22-millisecond isolated factor diagnostic uses the initial covariance and
four identity derivative directions. It shows a substantial primitive cost,
but multiplying it by 30 does not measure the actual fraction of target runtime:
later covariances differ, and fusion/hoisting in the enclosing graph can differ.
Exact device-time attribution remains unmeasured.

The eight-sweep choice was introduced for numerical accuracy: the solver's
source records a case where four sweeps fail a residual check. Reducing sweeps
or changing to float32 without equivalence evidence would not be a justified
repair. The current run uses float64 throughout; the manifest's `tf32=true`
does not accelerate float64 arithmetic.

The package also labels this principal-root path historical/reference-only
relative to its newer direct-factor SR-UKF default. The q20 master nevertheless
explicitly selects the frozen principal-root target. A new package default
does not silently replace that declared experiment. Switching factor geometry
can change a nonlinear UKF's finite value program and needs a separate checked
target decision; it is not equivalent to batching the same likelihood.

### Redundant target evaluation and kernel initialization

`FixedBetaBridgeAdapter.target_status_telemetry` at line 610 calls the full
combined value/score/status method and discards value and score. In the public
runner, `hmc.py:6603` requests proposed-state telemetry and line 6685 requests
accepted-state telemetry. These are two extra full target requests per traced
transition beyond the leapfrog work, even though the kernel already holds the
accepted/proposed values and gradients. The missing cached component is status.

At source level, two transitions with L=3 request approximately
`1 + 2*(3+2) = 11` evaluations per scalar chain, including initial bootstrap.
Using the standalone 1.172-second target cost gives
`4 * 11 * 1.172 = 51.6` seconds, close to the measured 53.5 seconds. This is a
useful consistency check, not an instrumented invocation count: compiler
elimination, trace setup and differing states can alter the actual work.

The batched one-step primitive used in qualification and ensemble kernels has
additional costs (`fixed_transport_hmc_mechanics_tf.py:566`): it calls
`kernel.bootstrap_results(state)` at every transition and then requests status
at the initial, accepted and proposed states. Qualification additionally calls
the target at the accepted endpoint (`q20_hmc_qualification.py:58`). For L=3 this
is eight source-level full requests per vector transition before possible
compiler elimination. Retaining kernel results and combined value/score/status
would avoid redundant requests when state coordinates and kernel identity have
not changed. Chart changes and swaps require the appropriate refreshed state;
blindly carrying gradients across them would be wrong.

These health checks are required. A repair should carry their already-computed
status through the kernel, not delete proposal checks or relax invalid-target
handling.

### NumPy and Python boundaries

There is no NumPy numerical computation or Python callback in the measured
target/HMC graph. `.numpy()` occurs at synchronization, host validation,
metadata and serialization boundaries; it copies/materializes tensors and is
not a call to a NumPy numerical solver. TensorFlow/TFP may import NumPy internally
for shape/constants infrastructure; that does not imply NumPy executes the
filter or leapfrog loop.

The inspected shared `hmc_tuning.py` does contain NumPy inside two separate
Gaussian-only diagnostic functions (lines 1775 and 1944). Neither is called by
this q20 route. Likewise, the non-batched scalar target has row-map helpers and
is constructed with `jit_compile=False` to obtain configuration/data/identity;
the active batched target does not invoke those helpers. Its own enclosing
graphs were measured with XLA enabled. The fixed-transport adapter's optional
`tf.vectorized_map` fallback is outside the required batch-native branch.

### Why the earlier campaign reservation was 449 hours

That number was **not an observed HMC time or complete campaign forecast**.
The saved cost review contains 63,267 seconds (17.57 hours) for the optimizer
floor plus 745,607 seconds (207.11 hours) for all planned heldout validation,
then applies the inherited engineering factor two: 449.37 hours total.
The master exited `UNDER_BUDGETED` during training pricing. It did not train a
full cohort or run posterior sampling/confirmation.

`_evaluate_rung` (`q20_production_training.py:74`) reconstructs baseline,
previous and current map evaluators each rung. For each validation look it
re-evaluates the whole common prefix for all three maps.
`HeldoutLoss` (`neutra_training_protocol.py:234`) uses a Python loop over batches,
generates CPU noise, invokes a compiled GPU value/score/status function and
synchronizes validity every batch. It computes the analytic score to preserve
the existing finite-score check even though the loss itself uses the value.

The maximal reservation covers `(768+3072+12288)/32 = 504` batches per map/look
ladder, three maps, four rungs and 36 positive-temperature candidate scopes:
217,728 batch evaluations. Saved batch-32 heldout calls cost about 3.4 seconds
each, similar to training updates. Adaptive early stopping may do less. Cached
immutable baseline/previous losses and incremental evaluation of bank extensions
can remove repeated work without changing the paired observations. Simply
calling this reservation the “minimum required runtime” would be wrong.

## Decision and next justified work

| Decision | Primary criterion status | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| HMC numerical graph is XLA | Confirmed by source, concrete graphs and successful execution | No callback or measured target-health failure | Other future route/shape combinations were not run | Preserve this coverage in repairs | Whole campaign is one compiled program |
| Serial chain scheduling wastes available batching | Confirmed configuration; 53.5 versus 16.5 seconds observed | Both small arms passed health checks | Small unrandomized timing sample; different random layouts | Wire a batched public candidate/retained runner with seed independence, trace shapes, health and resume checks | 3.24-fold posterior ESS/time improvement |
| Factor/telemetry work is a performance repair target | Repeated source operations confirmed; target cost measured | Accuracy gates remain binding | Exact device runtime fractions and finite-program equivalence | Reuse combined status/results; separately qualify existing eigensystem-reuse diagnostics | Safe to drop refinement, reduce precision or delete checks |
| Validation reservation needs work reuse | Prefix/baseline recomputation confirmed | Validation independence and pairing remain binding | Actual adaptive schedule length | Cache losses by exact target, map and bank identity; extend prefixes; then reprice | Campaign already affordable |

Prioritize batching independent chains, then reuse target status/kernel state,
then qualify eigensystem reuse under unchanged value/score/status tolerances.
Keep stable graphs across repeated calls; the candidate pool already caches by
L and chunk length, but owns separate scalar graphs and still builds new graph
families for changed L/count. Dynamic-L reuse is a later engineering option.
Validation cache work can proceed independently. Only after these changes pass
focused numerical and state/seed checks should the master be repriced.

| Inference status | Result |
| --- | --- |
| Hard veto screen | No measured nonfinite target/score/state or status failure; source matched and GPU growth verified. |
| Statistically supported ranking | None; two short timing repetitions are descriptive. |
| Descriptive-only differences | About 3.24 serial/batched HMC time ratio and 3.11 target throughput ratio. |
| Default readiness | Not established by this diagnostic; no default changed. |
| Next evidence needed | Controlled transition/seed/trace equivalence for batching and caching, representative-state target parity for numerical reuse, then longer timing and downstream sampler checks. |

Post-run red-team: the strongest alternative explanations for exact timing
ratios are different random trajectories, compiler reuse and uncontrolled clock
variation. Fixed loop counts, repeated target timings and two near-identical
warm timings support the diagnosis, but do not establish exact runtime shares.
The weakest numerical-performance evidence is the single initial-covariance
factor probe. A device profile or matched-noise controlled run could refine
attribution and overturn a proposed optimization's benefit. No evidence here
rejects HMC, NeuTra or the statistical research direction.

## Accounting and recovery

One numerical launch completed with exit zero. Preserve the worker's measured
354.691454 seconds separately from the tool-observed 381.229-second wall envelope.
Conservatively retain the full declared 660-second phase allocation for the run,
readiness, inspection and settlement rather than credit uncertain overhead.
Remaining campaign allowance is 123223.99030228473 seconds, including
42247.537327354854 diagnostic seconds. This is no increase in either budget.
The next agent must use the
[new settlement](artifacts/ssl-lstm-q20-hmc-performance-2026-09-16/settled-allowance.json)
and preserve its predecessor. This audit is the recovery note; resume from the
confirmed bottlenecks, not by repeating qualification or launching the full
unrepaired campaign.
