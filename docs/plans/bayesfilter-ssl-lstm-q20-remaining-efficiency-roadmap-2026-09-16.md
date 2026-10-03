# q20 remaining efficiency repairs

Date: 2026-09-16; execution updated 2026-09-17 Asia/Shanghai.
Status: roadmap reviewed; first status-reuse repair implemented and CPU/XLA
tested. The [September 17 master continuation](bayesfilter-ssl-lstm-q20-master-refresh-2026-09-17.md)
subsequently passed q20 GPU qualification and exact paired transition/health
checks. Training pricing/calibration is now supervised by the refreshed master.
The earlier resource-contended attempts below remain historical records.
Source inspected at main `d86dadf68ea57772642c6802990f46a3c6a04c30`.

## Objective and baseline

Reduce the total cost of obtaining reliable q20 posterior estimates. Measure
target evaluation, HMC transition, and end-to-end posterior cost separately.
Faster transitions do not establish improved mixing or adequate NeuTra training.

The engineering baseline is the current **batched** four-chain q20 runner,
T=30, float64, `tensorflow_eigh_strict`, TensorFlow/TFP with XLA, and all current
accepted/proposed health checks. Use identical data, starts, frozen maps,
epsilon/L, seed layout, hardware and precision within each controlled comparison.
The old serial runner is historical context, not the comparator for new savings.
Chain batching and immutable validation-prefix caching are already implemented.

The September 16 audit observed approximately 16.5 seconds for a warm batched
chunk of two transitions per chain at L=3, versus 53.5 seconds serial. These are
short descriptive timings of the earlier source, not current qualification or
a forecast for tuned posterior sampling. The latest integrated q20 source still
needs GPU qualification. Resource availability must be checked at launch.

Read together with the [performance audit](bayesfilter-ssl-lstm-q20-hmc-performance-audit-2026-09-16.md),
[batched repair](bayesfilter-ssl-lstm-q20-batched-execution-repair-2026-09-16.md),
[validation repair](bayesfilter-ssl-lstm-q20-validation-budget-repair-plan-2026-09-16.md),
and [public tuning interface](../reference/hmc-tuning-interface.md).
The capability registry was inspected: ordinary/affine HMC uses
`tune_hmc_kernel`; frozen nonlinear transports use
`tune_fixed_transport_hmc_kernel`. Runner optimizations issue no tuning authority.

## Repair order

| Order | Concrete work | Evidence required before adoption |
| --- | --- | --- |
| 0 | Preserve a current batched baseline and measure target/status work, compilation, warm execution and host/checkpoint overhead separately. | Actual synchronized q20 timings and a valid GPU/XLA run; no estimated component percentage presented as a profile. |
| 1 | Carry already-computed value, analytic score and health status through HMC; retain reusable kernel results in the ensemble path. | Same transition and health semantics, independent rows, rejected-proposal checks, exact deterministic continuation within the new implementation, and reduced measured work. |
| 2 | Revalidate the existing safe-factor eigensystem cache against the current refined strict solver. | Current-source value/score/status, spectral residual and transition checks, including near-degenerate and previously failing states. |
| 3 | Reuse compiled programs across tuning work and worker lifetimes where static contracts match. | Bounded tracing, no stale map/target/step/seed capture, unchanged evidence allocation, and measured total savings including startup. |
| 4 | Schedule independent training histories and prepared tuning scopes on persistent workers on available GPUs. | Reproducible scoped seeds, isolated outputs, correct aggregate budget accounting, restart checks and measured end-to-end throughput. |
| 5 | Run the existing training calibration stage and revise the remaining training allocation using its evidence. | Learning/validation uncertainty and downstream fixed-map HMC diagnostics; full replication and untouched posterior checks remain necessary for stronger claims. |

Orders 1 and 2 are separate candidates: measure each against the baseline, then
measure their combination. Do not multiply hypothetical speedups. Start with a
single qualified GPU; additional workers become useful after work per scope is
understood. If status propagation requires a large kernel rewrite, keep it
separate and test the existing factor-cache candidate first.

### 1. Preserve target results through the transition

`FixedBetaBridgeAdapter.target_status_telemetry` calls the combined target and
discards its value and score. The public runner requests this at accepted and
proposed states. The shared one-step primitive also bootstraps each transition
and checks its input state again. These are confirmed source-level requests;
XLA can eliminate unused computations, so they are not measured full-cost calls.

Make the combined `(value, score, status)` result explicit tensor state in the
shared numerical kernel. Retain both proposal evidence and accepted evidence;
select the accepted bundle with the same per-chain Metropolis mask as the state.
Rejecting a proposal must not hide an invalid proposal. Retain or reconstruct
the complete required state across chunk/restart boundaries. After a replica
swap, temperature change or chart change, refresh any cached quantity whose
target or coordinates changed; a physical state alone is not a valid cache key.

TFP's ordinary target callback exposes the density, with its score supplied by
the current custom-gradient adapter; it does not automatically return status in
kernel results. The first implementation check must establish a small, stateless
shared-kernel extension. A Python last-call cache or mutable variable side channel
is unsuitable for XLA, parallel chains and deterministic replay. Avoid a separate
q20 implementation of HMC. If a compact extension is unavailable, record the
engineering cost before expanding the sampler implementation.

Test the actual q20 consumers: qualification, pricing, public candidate tuning,
retained replay and ensemble transitions. Include accepted, rejected, invalid
and mixed-chain cases; compare proposals, momenta, acceptance calculations,
states and every existing health field. Use controlled identical random inputs
for implementation comparisons. Same integer seeds across different random
topologies do not imply paired trajectories.

A source-level model for n transitions is `1+n*(L+2)` target requests today and
`1+n*L` after endpoint-status reuse. For long chunks this removes at most 40%
of such requests at L=3, or about 7.4% at L=25. This is derived counting under an
equal-cost assumption, not a predicted speedup: compiler elimination, status-only
work and other costs must be measured. Ensemble bootstrap reuse is additional.

### 2. Reuse the safe factor's eigensystem

The candidate already exists as `tensorflow_eigh_strict_factor_cached` in
`experimental_batched_svd_sigma_point_tf.py`. It retains covariance classification
and diagonalizes the exact safe covariance passed to the root. It reuses that
basis for the derivative solve, removing one of three refined eigensystem
calculations in the placement factor/derivative operation. It does not cache
across filter observations or parameter values.

For safe SPD covariance C=V diag(lambda) V^T, let F=C^(1/2) and
Y=V^T X V, R'=V^T R V. The derivative equation FX+XF=R becomes
`(sqrt(lambda_i)+sqrt(lambda_j))*Y_ij=R'_ij`; the positive denominator gives
`Y_ij=R'_ij/(sqrt(lambda_i)+sqrt(lambda_j))`. Thus F's basis need not be computed
again in exact arithmetic. MathDevMCP's local SymPy obligation check simplified
`f_i*r_ij/(f_i+f_j)+r_ij*f_j/(f_i+f_j)-r_ij` to zero under a nonzero denominator.
This checks scalar algebra, not finite-precision equivalence or the full filter.

The [earlier reuse result](bayesfilter-ssl-lstm-q20-gpu-replay-batching-eigh-reuse-result-2026-09-04.md)
rejected the more aggressive raw-covariance cache for score disagreement.
The [factor mechanics result](bayesfilter-ssl-lstm-q20-factor-route-promotion-test-result-2026-09-04.md)
and later [scoped admission](bayesfilter-ssl-lstm-q20-factor-route-fresh-tuning-admission-result-2026-09-04.md)
support testing the conservative candidate, not transferring old tuning to the
current source. The eigensolver was subsequently repaired; historical timing
ratios and old admission cannot qualify this master.

Keep FP64, classification/repair semantics, refinement accuracy and all existing
vetoes. Compare the current full recursion and analytic score on regular,
near-degenerate, invalid and saved failure cases, then controlled transitions.
Use existing regression tolerances with their provenance; do not widen them to
make caching pass. Numerical differences must be assessed for their downstream
effect. Any adopted backend must have its correct scope identity and fresh
public tuning. Changing to Cholesky sigma-point geometry or lower precision is a
separate target/numerical decision, not this cache optimization.

### 3-4. Compilation reuse and useful concurrency

The q20 binding caches runners by `(L, chunk_count)`. `ReusableFullChainHMCRunner`
already supports dynamic L, while the current q20 caller fixes it. Measure using
that existing capability before introducing a new compiler-cache abstraction.
Dynamic L may trade compilation savings for steady-state performance. Preserve
static state shapes and current chunk evidence semantics; do not pad or truncate
draws simply to make shapes match. Across workers, keep programs alive for
compatible target/map scopes and report compilation separately from execution.

For parallel scheduling, keep four chains batched on each selected GPU and
distribute independent histories or independently prepared scopes. Preserve
candidate-cohort completion rules and fresh verification streams; completion
order must not change candidate selection. Parallelize same-cohort independent
work, not a repair child before its parent evidence exists. Classical mass
preparation is a separate workload and should be changed only if its measured
cost justifies it. Within-chain time and filter time remain compiled recurrences.

Use deterministic scope-derived seeds, distinct output directories, memory
growth and bounded persistent workers. GPU count is limited by current permitted
idle capacity and the existing budget. Report both elapsed time and summed
worker/device time: parallelism reduces elapsed time without automatically
reducing compute. One master must account for all worker reservations and
settle failures; do not multiply an allowance by the number of GPUs.

### 5. Calibrate the work that remains

Validation caching and decision-specific prefix expansion are already repaired.
The unresolved choices are the learning schedule and precision needed to make
useful decisions: 24 histories/36 beta scopes, batch 32, rungs 128/512/2048/8192,
floor 512, validation banks 768/3072/12288, loss delta .04 and half-width .02.
These are inherited hypotheses, not established requirements for q20.

Use the existing `calibrate` stage to observe the first declared root and first
rung across the width/LR/schedule inventory, preserving exact continuation into
the full protocol. Record paired-loss variance, learning progress, validation
cost and map reliability. The approximate fixed-comparison row requirement
`M=(1.96*s/h)^2` explains how precision costs scale; it neither calibrates h nor
provides anytime-valid inference on repeatedly inspected banks. Wider intervals
can suffice for a clear continue-training decision; uncertain plateaus remain
uncertain. Relate further training to actual transformed-HMC behavior using a
frozen map and the public tuner. One root or a short HMC diagnostic cannot rank
models or establish convergence. Keep identity and tuned classical comparators.

The existing historical-price calibration reservation is 2.762 hours, excluding
unmeasured startup/pricing overhead. It needs current pricing. Full training's
first-bank floor alone was quoted at 40.381 reserved hours, excluding HMC and
confirmation, so the full campaign remains unpriced and unaffordable under that
scenario. Savings are not yet evidence that the whole campaign fits.

## Evidence, budget and skeptical review

Engineering acceptance requires preserved target/transition/health behavior and
less measured work or total execution time under controlled conditions. Numerical
or health mismatches veto the candidate. Broken replay, corrupt evidence,
unqualified GPU execution or exhausted allocation stop the affected run. A failed
optimization triggers localization or the next independent repair; it does not
reject HMC or NeuTra. Timing and source operation counts are explanatory. Only
the separate posterior protocol can support MCSE, convergence, mode coverage or
effective-sample-per-time conclusions, with appropriate uncertainty.

For the first implementation package, write the exact focused test and paired
GPU commands into this same document once the implementation exists. Preserve a
fresh versioned output directory and a manifest with source, environment, hardware,
memory growth, seeds, commands, elapsed time and results. GPU work uses the
existing `tfgpu` environment with trusted access and `TF_FORCE_GPU_ALLOW_GROWTH=true`
before import. Tiny CPU reference checks explicitly set `CUDA_VISIBLE_DEVICES=-1`.
The existing performance/pricing scripts are starting points; the performance
script's historical baseline metadata must be updated for a current comparison.

No new compute allowance is created here. Settle the outstanding final 22.59-second
CPU regression from the previous phase before any next launch, then reserve the
smallest priced comparison from the remaining diagnostic allocation, which is
included in campaign time. Later stages receive allocations only after earlier
results determine their scope. Do not launch the full campaign against a partial
training-only forecast. Include preparation, compilation, tuning, warmup,
retained draws, ensemble work, checkpoints and confirmation in the final quote.

Skeptical review: the correct baseline is already batched; source calls can be
optimized away; component savings overlap; current solver changes invalidate
old backend qualification; status reuse can conceal rejected proposals unless
explicitly carried; and multi-GPU elapsed-time savings can hide greater total
compute. The stages above address each failure before adoption. No numerical
threshold is relaxed and no arbitrary reduced training count is promoted. This
roadmap is adequate for beginning bounded implementation; exact execution costs
and commands remain to be recorded before numerical runs.

## Active first implementation package

The owner requested continuation. The installed TFP HMC integrator only carries
target value and gradients; it has no public auxiliary-status callback. Avoid a
new integrator for this repair. A transparent wrapper can use TFP's existing
`MetropolisHastingsKernelResults.extra` field to carry accepted/proposed status.
Bootstrap status once, evaluate each proposal once, and select accepted status
with TFP's actual per-chain acceptance bit. This removes accepted-state trace
reevaluation, including when a proposal was rejected; proposal health remains
observable. Apply it to the reusable public runner and the shared one-step
primitive. Bootstrap and proposal target/status fusion remain future work and
must not be reported as completed.

Skeptical implementation audit: the wrapper must not change the target, TFP
integrator, RNG splits, result fields used by adaptation, or acceptance decision.
Status is deterministic at a fixed state/target. Never overwrite another kernel's
nonempty `extra`, drop optional diagnostics, or infer validity from acceptance.
Compatibility checks cover plain/adaptive/finite-guard results and no-status
routes. Dynamic chart/temperature states use fresh bootstrap as before.

Reserve at most 2400 seconds of the existing diagnostic allocation for this
package, including CPU tests, readiness, one bounded GPU diagnostic and
accounting. This is a convenience upper limit, not a required runtime. Charge the
previous outstanding 22.59-second CPU check first. Allocate at most 900 seconds
to CPU tests and repairs, at most 1200 seconds to a numerical GPU comparison,
and 300 seconds to readiness/accounting. Stop or record missing GPU timing if
the existing capacity policy finds no idle permitted device. Do not displace
another worker. Use fresh output roots beneath
`artifacts/ssl-lstm-q20-status-reuse-2026-09-16/`.

The first CPU reference command, with GPU intentionally hidden, will be:

```text
timeout --signal=TERM --kill-after=5s 595s env CUDA_VISIBLE_DEVICES=-1 TF_FORCE_GPU_ALLOW_GROWTH=true TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=2 /home/ubuntu/anaconda3/envs/tfgpu/bin/python -m pytest -q tests/test_hmc_status_reuse.py tests/test_q20_qualification.py tests/test_hmc_candidate_set_execution.py tests/test_q20_master_integration.py --junitxml=<fresh-attempt>/pytest.xml
```

The new test compares raw and wrapped TFP transitions with identical seeds,
checks changing/nontrivial status values through accepts/rejects and replay,
and verifies trace construction makes no new target-status calls. It includes
stable-signature CPU XLA as an explicitly tiny mechanics check. The GPU command
will be recorded here after its current-source diagnostic is implemented;
no old source qualification or old numerical tuning is transferred.

First focused test attempt: five tests passed and the adaptation fixture failed
because its test policy omitted the required `source` argument. This is a test
construction error, before adaptive execution. Add the missing provenance and
run the planned focused suite. Pytest reported 4.83 seconds; reserve 30 seconds
for the entire unsupervised startup/test envelope rather than treating startup
as zero. Compilation/import checks also passed. The wrapper is automatically
selected by the reusable runner's existing `per_chain_step` policy; it does not
add a new user-facing option or alter no-status execution.

The corrected focused suite passed 12 tests in 11.63 pytest seconds. Its
unsupervised startup envelope is charged another 30 seconds. The broader suite
uses the isolated checkout `/tmp/BayesFilter-q20-status-reuse-20260916` with the
current concurrent inference changes copied in before testing; it preserves
their behavior and a source inventory. Public source closure now includes
`hmc_status.py`, and runner identity reports the reuse policy.

GPU readiness at 2026-09-16T14:36:48Z passed on host GPU 1 (RTX 4080 SUPER),
TensorFlow 2.20.0, verified growth, with GPUs 0 and 2 already occupied. A fresh
probe immediately before launch must still find capacity. The numerical command
is now concrete:

```text
timeout --signal=TERM --kill-after=5s 1195s env CUDA_VISIBLE_DEVICES=<permitted-idle-gpu> TF_FORCE_GPU_ALLOW_GROWTH=true TF_NUM_INTRAOP_THREADS=2 TF_NUM_INTEROP_THREADS=2 /home/ubuntu/anaconda3/envs/tfgpu/bin/python docs/benchmarks/diagnose_q20_hmc_status_reuse_2026_09_16.py --output-dir <fresh-attempt>/worker
```

Run from the isolated checkout. The script performs current-source q20 GPU
qualification at beta .5 and 1, then compares the current public runner against
the same runner traced with only the status wrapper disabled. That baseline
override is explicit in the result. Both graphs are constructed before timing;
graph construction and first compilation/execution are recorded separately.
Use four chains, two transitions, L=3 and epsilon .01 from the prior bounded
diagnostic, with the same starts and paired stateless seeds. First call and two
warm calls are convenience diagnostic limits, not an uncertainty study. Alternate
execution order on repeats; record every timing and do not claim a ranking.
Compare all states and trace fields with the inherited 1e-9 relative/1e-10
absolute q20 regression tolerances, exact discrete fields, existing health
checks, one trace and no callbacks. No posterior run or default change occurs.

The isolated public-consumer suite passed all 83 tests in 268.208043 supervised
seconds (258.20 pytest seconds), with no source changes during execution. The
shared inference source and tests subsequently matched that checkout byte for
byte. Other worktree changes remain untouched.

GPU attempt 001 stopped before TensorFlow import because the launch command
mistakenly precreated the script's fresh output directory. Preserve the failed
attempt; no numerical work ran. Attempt 002 ran in the correct fresh root, but
another campaign shared GPU 1 after the external readiness check. Only the
owned diagnostic PID 1672223 was sent SIGTERM. Its supervised wall time was
192.521702 seconds. Qualification and paired timing had not completed. The
terminal disposition is `stopped_resource_contention`, not a numerical failure
or method rejection; no performance evidence is usable.

Repair the diagnostic launcher to run the installed readiness checker inside
the worker, after any approval/scheduling delay and before TensorFlow import.
Check for competing compute PIDs around qualification/timing, save memory/source
provenance before qualification, and preserve the result on SIGTERM. These are
localized resource/provenance repairs. One retry may use at most 970 seconds
including termination grace, within the original 1200-second numerical
allocation (192.522+970 < 1200). A busy-device result ends this phase with GPU
qualification pending. The readiness subprocess has a 60-second convenience
timeout inside that cap. Do not rerun the already-passing sampler tests for this
diagnostic-only change; check its syntax/CLI and execute its actual readiness
path instead. No scientific setting, precision or veto changes.

Final compatibility review found one overly broad application: legacy reusable
consumers that request accepted status only would gain proposal evaluations.
Restrict reuse to the existing combination `capture_candidate_health=True` and
`target_status_trace_policy=per_chain_step` (the q20 policy). Other consumers
retain their prior behavior, as does first-failure recording. Add one focused
regression and rerun the affected status/q20 qualification tests; the q20 branch
itself is unchanged. This is justified follow-up to a concrete review finding.

The resource-safe retry stopped before TensorFlow import after 0.286908 supervised
seconds with `no_idle_policy_permitted_gpu`. GPU qualification and timing remain
pending; no third numerical attempt is planned in this phase. The earlier
external ready checks did not reserve the device. Delayed execution and another
campaign made those observations stale, so actual-launch readiness is now checked
by the worker itself.

## First package result and continuation

The final compatibility suite passed 17 tests in 19.086155 supervised seconds
(16.53 pytest seconds), after the 83-test public-consumer suite. These suites
overlap. They cover TFP transition parity, accepted/rejected status, invalid
proposals, independent batch rows, dynamic seeds/steps/L, adaptation and finite
guards, checkpoint/replay, XLA traces without callbacks, public source closure,
q20 qualification, identity/classical/NeuTra consumers and replica exchange.
The final shared inference code/test files match the isolated tested checkout.
Syntax/CLI checks and `git diff --check` passed.

Implemented in `bayesfilter/inference/hmc_status.py`, the public reusable runner,
and the shared one-step primitive. Accepted status is selected from the initial
or proposed status using TFP's actual Metropolis decision. Proposal evidence is
preserved even when rejected. The q20 health policy selects the wrapper;
accepted-only legacy consumers and first-failure capture preserve their behavior.
The new source is included in public execution identity and source checks.

This is **partial removal of duplicate work**: each chunk initializes status,
then each transition evaluates proposal status once. It removes the separate
accepted-state status request. It does not fuse proposal status with the
integrator's value/score, retain ensemble kernel results across chart/swap
transitions, change the eigensolver, or introduce multi-GPU scheduling. No elapsed
time percentage is claimed. The tested CPU numerical/mechanics behavior supports
the implementation; GPU qualification is still required by the actual master.

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Retain accepted-status reuse implementation | Controlled TFP/health parity and actual public consumers pass | No numerical or replay failure in the focused CPU/XLA checks | q20 GPU behavior and net runtime effect are unmeasured | Complete the prepared paired GPU diagnostic in an idle slot | Full duplicate-target elimination or measured acceleration |
| Stop this GPU attempt | Resource validity failed | Other compute processes shared selected GPU; retry found no idle device | When capacity will be available | Use the worker's fresh launch check on the next attempt | Rejection of HMC, NeuTra or the cache |
| Continue roadmap | Kernel/status and conservative factor reuse remain separate mechanisms | Historical raw-basis gradient mismatch remains a veto | Current factor-cache full-recursion and downstream parity | Revalidate the existing safe-factor candidate after the status baseline is measured | Lower precision, fewer accuracy sweeps or altered sigma-point geometry is justified |

| Inference status | Finding |
| --- | --- |
| Hard veto screen | CPU/XLA engineering checks pass; GPU resource veto fired |
| Statistically supported ranking | None |
| Descriptive-only differences | Reduced source-level status request count; no completed timing comparison |
| Default-readiness | q20 wiring implemented; current-source GPU qualification remains pending; posterior readiness is not established |
| Next evidence needed | Uncontended paired q20 GPU execution, then separately checked factor reuse and complete campaign pricing |

Terminal red-team: source request counts do not determine device work because
XLA can remove duplicated or unused operations. This status wrapper adds an
initial status evaluation per chunk and changes scheduling of telemetry; short
chunks may show little benefit. It relies on the existing fixed deterministic
target contract, and status is not reused across chart/temperature changes.
The most consequential missing evidence is real q20 GPU transition parity and
cost on the final source. The isolated diagnostic and existing tests preserve a
direct comparator for that question.

Accounting is in
`artifacts/ssl-lstm-q20-status-reuse-2026-09-16/settled-allowance.json`.
The phase charges 480.102809 supervised worker seconds, two conservative
30-second startup/test envelopes and the predeclared 300-second
infrastructure/readiness/accounting hold: 840.102809 seconds total. Also settle
the earlier outstanding 22.59-second main regression. Release 1559.897191 unused
seconds of this phase's 2400-second reservation. Remaining allowance is
**33.56984 campaign hours, including 11.07638 diagnostic hours**. No allowance
was renewed. No posterior campaign or new training run was launched and no commit
was requested; unrelated concurrent changes remain untouched.
