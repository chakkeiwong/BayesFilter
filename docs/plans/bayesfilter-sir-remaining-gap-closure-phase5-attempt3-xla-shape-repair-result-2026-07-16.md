# SIR Phase 5 Attempt 3 XLA Shape Repair Result

Date: 2026-07-16

Status: `REPAIRED_LOCALLY_GPU_RETRY_BLOCKED_BY_EXHAUSTED_ATTEMPT_BUDGET`

Plan: `docs/plans/bayesfilter-sir-remaining-gap-closure-master-plan-2026-07-16.md`

Failed attempt root:
`docs/benchmarks/artifacts/sir_remaining_gap_closure_20260716/phase5_phase6_gpu_paired_attempt03/`

## What Failed

GPU attempt 3 first passed the exact frozen-input final-source Austria replay:

- device: `/GPU:0`;
- value CPU/GPU absolute delta: `0`;
- maximum score CPU/GPU absolute delta: `1.2523315717771766e-13`;
- all prepared hashes equal;
- valid and reset charts passed.

It then failed while compiling the first `J=2,T=1` LEDH row. For one
observation, the runner prepared `transition_noise` with shape
`[R,0,N,D]`. Although the transition branch is never executed at `T=1`, XLA
validates both branches of the `tf.cond` inside the horizon `tf.while_loop` and
rejected the unused slice `transition_noise[:, time_index-1, :, :]`.

Classification: localized harness/graph-shape failure. It did not invalidate the
target, teacher mathematics, LEDH equations, Contract E reset, Phase 3 replay,
or identity mechanism.

## Repair

For `T=1` only, the runner now prepares one zero-valued unused transition slot
with shape `[R,1,N,D]`. The horizon remains one observation, so the transition
branch does not execute and the padded value does not enter the mathematical
program. It exists only to make the statically traced unused branch shape-valid.

## Focused Evidence

- focused comparison/identity tests: `11 passed`;
- exact failed scale rerun on deliberate CPU XLA:
  `J=2,T=1,N=256,R=16`;
- XLA log: `Compiled cluster using XLA!`;
- value: `-7.131521311228654`;
- score: `[0,0,-0.6365866772711894]`;
- chart valid: true;
- padded transition shape: `[16,1,256,4]`;
- maximum padded transition magnitude: `0`;
- two-node identity:
  `0d634e09d006b4f94ebe3684575948862154d2bf255b64f59cb1396012f2ce33`;
- prepared-input identity:
  `0a012186779b2b15d5a7b75bac54a79736e7eae979dae5c83d74261c3215bac8`.

## Budget Ledger

The master plan allows at most three trusted GPU/XLA attempts.

| Attempt | Outcome | Classification |
| ---: | --- | --- |
| 1 | passed current candidate before identity wiring | engineering evidence, superseded by later source wiring |
| 2 | passed exact registered Austria route | exact-symbol Phase 3 evidence, later source expanded for separate `J=2` route |
| 3 | final-source Austria replay passed; `J=2,T=1` failed on zero-length unused transition slice | localized harness shape failure, repaired on CPU XLA |

The attempt budget is exhausted. A fourth trusted GPU launch would expand the
declared campaign budget and therefore requires owner approval. The proposed
retry does not change the scientific target, data-generation policy, method,
particle counts, replicates, intervals, vetoes, hardware class, or privacy
boundary.

## Exact Retry Scope

One additional trusted GPU/XLA attempt, at most two remaining campaign hours,
using:

```bash
env -u TF_FORCE_GPU_ALLOW_GROWTH \
BAYESFILTER_TF_GPU_MEMORY_LIMIT_MIB=8192 TF_CPP_MIN_LOG_LEVEL=1 \
MPLCONFIGDIR=/tmp/matplotlib-sir-gap \
/home/chakwong/anaconda3/bin/conda run -n tf-gpu python \
docs/benchmarks/run_sir_ledh_teacher_comparison_gpu.py \
--phase3-prepared docs/benchmarks/artifacts/sir_latent_preclip_repair_20260716/prepared_t2_n32_attempt01/prepared.json \
--phase3-cpu-result docs/benchmarks/artifacts/sir_remaining_gap_closure_20260716/phase3_cpu_xla_registered_final_source_attempt02/result.json \
--output-root docs/benchmarks/artifacts/sir_remaining_gap_closure_20260716/phase5_phase6_gpu_paired_attempt04
```

The retry must use a fresh output root, preserve attempt 3, checkpoint every
row, and retain all mismatch-only nonclaims.

The runner configures one TensorFlow logical GPU with an `8192 MiB` allocator
limit before importing model routes or constructing tensors. It rejects memory
growth, any other cap, a missing cap, or more than one visible GPU. This is a
hard TensorFlow allocator ceiling for this process; it is not evidence that the
complete ladder fits within the ceiling.

Before retry, the runner also writes a `RUNNING` manifest before Phase 3 and
catches TensorFlow resource exhaustion and other Python exceptions around the
comparison ladder. A caught failure writes `failure.json`, failed `progress.json`,
a terminal manifest, and artifact hashes before exiting nonzero. Completed rows
remain intact. Resource exhaustion is an engineering continuation veto and
cannot support a scientific claim about LEDH or the teacher. Native aborts,
`SIGKILL`, host OOM, and driver reset remain uncatchable by Python; the startup
manifest preserves launch provenance for those cases.

## Attempt 4 Skeptical Prelaunch Audit

Status: `PASS_TO_LAUNCH_WITH_FIXED_ENGINEERING_CAP`.

- The exact baselines remain teacher `N=128` versus `N=256` for refinement and
  LEDH `N=256` versus teacher `N=256` only after the refinement classification;
  runtime, ESS, and GPU memory remain explanatory diagnostics.
- The `8192 MiB` setting is an owner-selected resource ceiling, not an accuracy,
  promotion, or model-validity threshold. OOM rejects only the incomplete rung
  as engineering evidence and triggers memory repair or sharding.
- A statistically supported method disagreement is a promotion veto but not a
  continuation veto. It does not identify which method is closer to truth.
- A nonfinite result, invalid chart, malformed artifact, caught OOM, other
  execution exception, native abort, host kill, or exhausted two-hour budget is
  a continuation veto for the remaining ladder.
- `T=20` follows earlier `d=18` rungs only when they complete computationally.
  A completed but scientifically unfavorable earlier rung remains valid and is
  carried forward as a diagnostic trigger.
- The environment remains the reviewed `tf-gpu` TensorFlow float64 GPU/XLA
  route with one visible logical GPU. The cap probe verified the configured
  device has exactly `8192 MB`; it did not establish that every rung fits.
- Attempt 4 uses a fresh versioned output root and preserves all earlier
  evidence. The generated manifest, checkpoints, terminal result or failure,
  and hashes answer the stated execution question.

## Attempt 4 Result And Repair Trigger

Attempt 4 enforced the `8192 MiB` cap, passed the Phase 3 GPU replay, and
completed `two_node:T1`. That row reported no detected teacher refinement shift
and no detected LEDH--teacher disagreement at current precision. The next rung,
`two_node:T2`, failed before producing scientific output.

The structured failure record classified the exception as an engineering
continuation veto. TensorFlow peak allocation was only `770.313 MiB`, so the
failure was not OOM. The registered two-node `tf.function` first traced `T=1`
and then generalized its horizon axis during the `T=2` trace. The canonical core
used `int(prepared["observations"].shape[0])`; the relaxed graph shape was
`None`, causing `int(None)`. The repair replaces only that Python shape
conversion with the graph scalar `tf.shape(observations)[0]`. TensorArrays and
the existing `tf.while_loop` already accept the dynamic scalar, so the finite
program and Contract E mathematics are unchanged.

Focused verification after repair:

- exact registered two-node `T=1` then `T=2` CPU-XLA trace regression: pass;
- Contract E SIR value/score and same-scalar AD/FD tests: pass;
- canonical route-identity and dependency-closure tests: pass;
- online teacher, comparison-runner failure handling, and GPU-memory-policy
  tests: pass;
- combined focused result: `34 passed`;
- diff whitespace check: pass.

Attempt 4 consumed about `58.1 s` and reached only `770.313 MiB` TensorFlow peak
allocation. Its fresh output root and failure record remain immutable evidence.
The fourth-attempt count is exhausted even though the two-hour wall-time budget
has substantial headroom. A repaired retry must therefore be a separately
authorized fifth attempt under the unchanged scientific contract, hardware,
`8192 MiB` cap, and remaining wall-time budget. Its fresh output root must be:

`docs/benchmarks/artifacts/sir_remaining_gap_closure_20260716/phase5_phase6_gpu_paired_attempt05/`.

## Additional Retry Budget

Owner authorization on 2026-07-16 adds five trusted GPU/XLA attempts, numbered
5 through 9. The scientific target, methods, data, particle counts, replicates,
interval contract, promotion and continuation vetoes, GPU hardware class,
`8192 MiB` TensorFlow allocator cap, and remaining two-hour campaign wall-time
budget are unchanged. Each attempt must use a fresh numbered output root and
preserve every earlier result. Localized harness, XLA, serialization, or memory
failures may be repaired and retried within this budget after focused regression
checks. The budget does not authorize threshold changes, scientific promotion,
package changes, destructive operations, or external release.

## Attempts 5--6 Result And Ridge Blocker

Attempt 5 verified that the dynamic-horizon repair passes tracing and executes
the `T=2` route, then the runner failed while trying to construct a Student
interval from nonfinite paired samples. Peak TensorFlow allocation was only
`770.612 MiB`; this was not OOM. The runner was repaired to preserve nonfinite
samples as JSON `null`, report finite masks, omit unavailable intervals, emit a
computational-invalid row, and stop later horizons under a continuation veto.

Attempt 6 completed that diagnostic path. `two_node:T1` remained valid.
At `two_node:T2`, the teacher was finite for all `N=128,256` replicates and its
refinement intervals were available. LEDH was invalid for exactly seeds
`87202`, `87203`, and `87215`. The clipping-boundary chart passed for all
replicates, all quotient row masses were positive, replay was exact, and the
three invalid replicates failed Contract E reset validity only at time index 1.

The reset requires
`chol(Sigma_w - Sigma_plus + lambda I)`. The prepared `lambda=1e-6` was inherited
without SIR-specific provenance or a declared domain-feasibility argument. It
does not make the covariance gap positive definite for the three replicates.
This is a genuine invalid-chart veto under the documented Contract E--Chol
semantics. Increasing the ridge after seeing these outputs would be
candidate-dependent post-result tuning and is forbidden for the canonical
total-gradient claim. Attempts 7--9 must not retry the unchanged program merely
to reproduce the same invalidity or select a ridge from these audit seeds.

The next permitted diagnostic is telemetry-only: report the already-computed
flow, geometry, quotient, reset-finite, reset-factor-positive, and covariance-gap
eigenvalue histories without changing the scalar, score, ridge, reset, or
transport. Any future SIR ridge candidates require a separate pre-result design
over a declared parameter/input domain, with positivity and raw covariance-bias
requirements fixed independently of these audit outputs.

Attempt 7 added that telemetry without changing the finite program. For the
three invalid seeds, the minimum time-1 gap eigenvalues were `-0.149862`,
`-1.121298`, and `-0.311444`. Flow, geometry, quotient positivity, and clipping
charts all passed; reset finiteness and factor positivity failed. The raw finite
plan had column-marginal residual near `2e-15` but row-marginal residual about
`0.964--0.969`. Contract E consumes the row quotient `Y=Q/M`, which enforces
rows by dividing each plan row by its mass and therefore changes its column
marginal. The raw-plan column residual is not the relevant premise check for
the actual quotient cloud.

The next telemetry-only repair reports the exact post-quotient column mass
`sum_i P_ij/M_i` and its residual against `N*w_j`, using the existing streaming
block pass and only `O(B*N)` retained state. A tiny dense oracle and source/AST
guard pass. Attempt 8 may capture this diagnostic with the unchanged two-step
transport and `1e-6` ridge. It must not change either setting.

Attempt 8 executed Phase 3 and the `T=1` computation, then the route-identity
factory correctly rejected issuance because the telemetry helper was renamed
and the declared dependency closure still named the previous symbol. No
scientific row was emitted; peak allocation was `770.318 MiB`. The repair keeps
a historical one-output compatibility wrapper, binds the exact new two-output
reporter into canonical SIR identity, and passes identity closure, forgery,
dense-marginal, no-`N^2` allocation, and dynamic-route checks (`9 passed`).
Attempt 9 is reserved solely for the final post-quotient telemetry capture.

## Attempt 9 Terminal Result

Attempt 9 completed in `93.256 s` under the verified `8192 MiB` TensorFlow cap;
peak TensorFlow allocation was `770.657 MiB`. Phase 3 replay passed, fresh route
identity issued, and `two_node:T1` reproduced all earlier scalar, score,
validity, reset, and statistical fields exactly. `two_node:T2` reproduced the
same three invalid seeds and stopped the ladder under the declared continuation
veto. Artifact hashes and the terminal manifest were written.

The final post-quotient telemetry establishes the structural root cause. For all
16 `T=2` replicates, the raw two-step plan had row residual `0.843--0.969`; its
raw column residual was near machine precision. Dividing each row by its mass
created the cloud actually consumed by Contract E but changed the column
marginal. The post-quotient relative column residual was `1.298--5.788`. For
seeds `87202`, `87203`, and `87215`, the time-1 covariance-gap minimum
eigenvalues were respectively `-0.149862`, `-1.121298`, and `-0.311444`, so
`chol(G+1e-6 I)` failed as it should.

The claimed target is a Contract E positive first-order transform followed by
Contract E--Chol. The quantity actually computed is a row quotient of a finite
plan whose post-quotient column marginal is not `N*w`. They are different. The
positive-gap proposition therefore does not apply to the executed quotient
cloud. This is wrong relative to the stated Contract E positive-transform
premise, even though the code correctly differentiates its finite operations.
A larger ridge would hide the indefinite gap but would not restore the missing
first-order transport constraint.

Terminal affected-suite verification: `47 passed`, plus exact cross-attempt
invariance of all pre-existing `T=1` LEDH and teacher fields and a clean diff
check. GPU attempts 5--9 are exhausted. Phases 5--6 remain blocked; Austria
`d=18` was not executed and no all-model score, HMC, leaderboard, default, or
accuracy claim is available.
