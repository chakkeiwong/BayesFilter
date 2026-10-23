# Surrogate-HMC optimization implementation results

Status: complete. Final filter comparisons, training integration, saved-tensor
review and terminal source verification are finished. No compute job is pending.

The purpose is to reduce the cost of testing surrogate HMC while preserving the
finite LEDH likelihood, its analytical score, and the canonical NeuTra update.
The comparison uses commit `d189c429542eb3a682677ce7fc463c3b796b69ea` as the
preserved baseline. Numerical source hashes identify the uncommitted candidate.
The reviewed plan is [the implementation plan](surrogate-hmc-optimization-implementation-2026-10-21.md).
All execution evidence is in
`artifacts/surrogate-hmc-optimization-20261021-01/`.

## Retained changes and equivalence

The SIR adapter evaluates the finite LEDH log likelihood plus the scientific
prior and the corresponding analytical score in native batches. Training uses
that matched pair. The frozen map transforms the proposal force and the
independent MH endpoint separately, including their log-Jacobian terms. Data,
prior, observation chart, particle design, controls and numerical sources are
checked before reusing a map. A changed target rejects the map.

Native sharding preserves one logical optimizer update. Each shard uses the
same weights; its contribution is weighted by its real row count. Padding is
excluded. Clipping and Adam run once after accumulation. An invalid shard
rejects the entire update, and checkpoints include optimizer slots and iteration.

The nonshared Gaussian-mixture score uses exact weighted moments instead of
retaining pairwise tensors for every parameter direction. Shared polar geometry
also avoids repeating the primal calculation for each direction. These are
algebraic rearrangements of the same finite filter. Float64 independent density
and derivative checks cover moving particles, moving means, parameter-dependent
covariances and common translations. The polar check uses an independent value
formula and a finite-difference step ladder, including near-origin cases.

The triangular OT cost cache was implemented and **rejected as the default**.
Alternating measurements exposed a substantial slowdown, including LGSSM at
10.75s versus 4.61s. Its source and measurements are retained under
`rejected-triangular-cache/`. The final implementation restores the original
dense cache and retains the invariant feature calculation outside the repeated
tile reductions. This final change has no demonstrated material speed or memory
gain for the four shared-covariance models.

## Full-size filter evidence

Both arms use N4032, FP32 with TF32 disabled, native B4, exact K2016 chunks,
identical data/designs, and frozen controls. T50 applies to the first four models;
range bearing uses the saved T20 fixture and 16 flow steps. The four rows are
the central parameter and three small perturbations. Every row remains valid,
finishes the horizon and repeats deterministically.

| Model | Largest absolute log-likelihood difference | Largest absolute score difference | Baseline/candidate median seconds | Baseline/candidate peak MiB |
|---|---:|---:|---:|---:|
| LGSSM | 0 | 0 | 4.025 / 3.945 | 519.07 / 519.07 |
| KSC SV | 0 | 0 | 1.783 / 1.791 | 515.13 / 515.13 |
| Predator-prey | 0 | 0 | 2.784 / 2.827 | 518.17 / 518.17 |
| SIR | 0 | 0 | 8.144 / 8.169 | 551.05 / 551.05 |
| Range bearing | 0.000087738 | 0.000860691 | 2.178 / 1.576 | 765.12 / 517.10 |

Likelihood/score differences and allocator peaks come from separate processes;
times come from six alternating calls per arm in one process with both graphs
already compiled. Synchronization is included. Compilation is recorded
separately. The same RTX5080 was used with verified memory growth. Another
workload occupied the second GPU. The range-bearing observation is roughly 28%
lower runtime and 32% lower allocation; timings are descriptive measurements,
not a statistically supported ranking over datasets or hardware. The small
timing differences for the other models do not establish improvement or harm.

The range-bearing score's largest unit-scaled difference is 0.0001992, where
the denominator is `max(1, abs(baseline_component))`. This is **not** the ordinary
relative percentage of every component. The paired timing process has slightly
different FP32 rounding and is preserved separately. No fixed percentage rule
is used to infer scientific deterioration. This comparison isolates engineering
equivalence; it does not compare either filter to an exact nonlinear oracle or
estimate Monte Carlo error across particle designs.

The saved-tensor review checks every returned field, rather than only the two
headline quantities. For the first four models, values, scores, particles,
particle sensitivities, traces, score traces and flags are all bitwise identical.
For range bearing, six floating fields differ. The largest absolute particle
and particle-sensitivity differences are 0.0001312 and 0.0012461. The trace has
a largest absolute difference of 0.12964, but a largest unit-scaled difference
of 0.0002592; the score-trace maximum is 0.0004148. All discrete status fields
agree. The unchanged shared core is also bitwise identical in its saved paired
comparisons. Details are in `saved-tensor-equivalence.json`.

## SIR training and frozen-map integration

The final `training-sir_d18-candidate-03` pilot uses the real N4032, T50 LEDH
target with the canonical author IAF, logical batch B8 and native shard size S4.
It performs two updates from identical weights and noise, comparing the whole
batch with accumulated shards. Adam uses learning rate 0.001 and gradient clip
1; the narrow integration fixture uses scale 0.02 and seed 261021. These are
pilot choices, not a selected SIR training protocol.

| Quantity compared | Largest absolute difference |
|---|---:|
| Training objective | 0 |
| Complete parameter gradient | 0.00000005960 |
| Proposal score | 0 |
| Parameters and Adam state after update 1 | 0.000000003725 |
| Parameters and Adam state after update 2 | 0.000000007451 |
| Checkpoint-resumed state versus uninterrupted state | 0, bitwise exact |

Both routes reach optimizer iteration 2 with valid updates. Warm whole-batch
and sharded updates took 14.934 and 15.954 seconds, respectively: about 6.8%
observed overhead for sharding in this pilot. The allocator peak was
938,332,672 bytes (894.86 MiB) with both graphs and states retained. It is not an
isolated whole-versus-sharded memory comparison and does not establish an
optimal shard size.

Training retains the native batch dimension on GPU under XLA, without a
sample-wise target loop. The frozen map is also exercised through the public
surrogate-HMC binding, with the independent MH endpoint and its log Jacobian.
The map and HMC coordinates use FP64 under the shared HMC interface; the LEDH
force remains FP32. Both tested rows are finite with no force fallback. One
transition per row produces acceptance flags `[true, false]` and energy changes
`[0.14217, 2.15497]`; these check mechanics and are not an acceptance-rate or
convergence study. Two updates do not establish posterior coverage or improved
SIR mixing.

## Test and run records

The CPU diagnostic ledger contains 130 distinct tests, all passing in their
latest execution. Seven XML files retain 187 test executions, including reruns.
The earlier `final-focused.xml` retains one failed finite-difference reference
case; `final-polar.xml` records its successful rerun. `shared-core-final.xml`
adds ten distinct shared-core regression tests. The ledger deduplicates by
test identity and retains the latest status; it does not claim every historical
execution passed. The tests
cover all five model families, multiple rows, covariance structures, correction
on/off, complete gradients, clipped/unclipped Adam updates, uneven shards,
invalid final shards, exact continuation, mixed flow/target precision, frozen-map
pullbacks, actual public HMC binding and stale map rejection. GPU devices were
intentionally hidden for these tests. One initial polar finite-difference
reference failed from truncation at the coarsest step. Richardson extrapolation
resolved it without relaxing the assertion tolerance. Original logs are kept.

Four initial candidate workers completed calculations but failed while
serializing optional-module provenance. The harness was repaired and retried
into fresh directories. The first SIR integration pilot passed, but was
superseded by stronger map identity checks. Its second attempt correctly
rejected a stale identity after source edits during execution. The third attempt
passed with frozen numerical sources. These are harness failures, not
evidence against the scientific method, and all attempts consume the budget.

The ledger accounts for 2,758.012 seconds (45.97 minutes) of the six-hour worker
budget across 32 attempts: 16 complete, five failed and 11 superseded. It leaves
18,841.988 seconds unused. The completed checks answer this bounded plan, so
spending the remaining budget is unnecessary. The final training manifest's
ten source hashes, including the master program, match the current files.
Python AST parsing, JSON-ledger checks and source/document whitespace checks
also pass. Commit preparation includes 345 evidence files and excludes Python
bytecode caches. A check including the preserved raw logs/XML reports 37
formatting-whitespace findings in tool output; those original records are
kept unchanged. No source or documentation whitespace finding remains.

## Interpretation and remaining work

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | What is not established |
|---|---|---|---|---|---|
| Matched SIR adapter and frozen map | CPU call-chain checks and final GPU integration pass | Stale maps rejected; tested HMC rows finite | Trained-map coverage | Parent target-specific training protocol | Improved SIR mixing |
| Complete-update native sharding | Complete gradient/Adam/resume comparisons pass, including real SIR updates | Invalid batches reject all mutation | Wider batch sizes; observed 6.8% pilot overhead | Retain optional sharding; choose size under parent capacity protocol | Best shard size or isolated memory saving |
| Range-bearing algebra | Independent derivatives and full filter comparisons pass | All four rows healthy | Wider parameter/data regimes | Retain with recorded rounding limits | Better statistical accuracy |
| OT storage | Triangle loses to dense-cache baseline | Runtime veto fired for triangle | Benefit of feature hoist | Keep dense cache; preserve rejection evidence | General OT speedup |

| Inference status | Finding |
|---|---|
| Hard veto screen | Final filter rows and integration pilot healthy; all saved tensors reviewed; rejected triangular layout documented |
| Statistically supported ranking | None claimed |
| Descriptive differences | Range-bearing runtime/allocation reduced on this fixture; shared-model timings essentially unchanged |
| Default readiness | Engineering changes have bounded equivalence evidence; no new tuning-scope or posterior admission |
| Next evidence needed | Full target-specific SIR training protocol and cumulative HMC assessment to judge mixing |

The strongest alternative explanation for runtime differences is shared-machine
load or compiler layout choices. Alternating order reduces this confound;
isolated processes supply the allocator evidence. A regression on an unseen
healthy parameter region would overturn broad equivalence generalization. The
weakest empirical evidence is one particle design with nearby parameter rows.
Independent mathematical checks strengthen implementation confidence but do
not turn that fixture into a general statistical non-harm study.

The bounded implementation/equivalence master is finished. The existing parent
SIR training plan still requires its target-specific architecture/learning-rate
search, coverage assessment and cumulative adaptive HMC run. Those are a
separate research phase and were not launched by this plan. Its integration
work is now available, but the five-model posterior objective remains
incomplete. Two updates on a deliberately narrow fixture must not be presented
as a useful trained posterior map.
