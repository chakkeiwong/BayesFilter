# Five-model surrogate-HMC master, N=4032, 2026-10-11

## Terminal state — 2026-10-21

The allocated execution has ended: LGSSM, KSC SV, predator–prey and range
bearing passed their sampling rules; SIR exhausted its worker budget during
warm-up with no retained draws. The native batching repair and bounded
capacity testing are complete. The SIR NeuTra adapter, training and downstream
sampling remain pending. The fixed-percentage blocker is retired in favor of
reference comparison with uncertainty. See the
[final master summary](surrogate-hmc-master-final-summary-2026-10-21.md) and
[active reset memo](surrogate-hmc-five-model-n4032-reset-2026-10-11.md).
The dated execution states below are historical; no run is pending from them.

## Current execution state — 2026-10-16

The active question remains testing surrogate HMC on the five saved models.
The owner has authorized an additional 60 GPU-worker hours under
`surrogate-hmc-60h-continuation-2026-10-16.md`, executed by
`scripts/run_surrogate_hmc_60h_continuation.py`. Predator–prey is running and
SIR follows. The first 16-flow-step range-bearing attempt stopped after 250
warmup transitions solely because the proposal force used its deterministic
zero fallback; its endpoint, finiteness, energy, movement and divergence
checks passed. That failure remains archived, with zero retained draws.

The owner-authorized controller repair is
`surrogate-hmc-force-fallback-repair-2026-10-16.md`. For the bound deterministic
position-only proposal field, force fallbacks now produce recorded warnings
and counts. Independent Metropolis endpoint and sampler health checks remain
enforced. Fresh sampling, replay/resume and retained continuation use the same
policy. This does not establish exactness of a nonlinear endpoint likelihood.
The existing PP process retains its loaded controller; subsequent workers use
the repaired controller. The filter, endpoint, prior and tuned controls are unchanged.

The `enqueue-range-bearing` command restores range bearing after the existing
PP/SIR queue. It waits for the original supervisor to exit and for all its
worker charges to settle, then runs the existing master in a fresh retry
directory. It uses at most 22712.36 worker seconds remaining from the original
eight-hour range allocation, also bounded by the total 216000 seconds and six
attempts. This retry is part of the master. Sixty-two focused CPU-only tests
passed, including kernel, controller and queue checks. The shared output root
is `artifacts/surrogate-hmc-60h-continuation-20261016-01`; the durable retry
record is `queued-range-bearing.json` and the retry output is
`range-bearing-retry-01/`. `status` reports the original phase and retry;
`next-phase-plan.json` transfers to the retry supervisor after PP/SIR finish.
LGSSM/KSC results are preserved. Live progress and the exact next action are in
`surrogate-hmc-performance-repair-checkpoint-2026-10-11.md`.

Previous allocation terminal update checked00:24 UTC: all campaign processes had exited and the
86400-second allocation is exhausted. LGSSM and KSC completed2000 warmup and
1000 retained draws per chain, maximum modern R-hat1.00708 and1.00600,
respectively, with zero divergences/fallbacks. Predator–prey hit its stage
budget, SIR inference did not start, and the16-step range continuation passed
fresh tuning/mechanics but timed out during reported XLA compilation before
saved sequential samples. The five-model task remains incomplete. The
following queue narrative describes the earlier launch, not a live queue.
Result: `artifacts/surrogate-hmc-range-bearing-flow16-20261016-01/result.md`.

The owner selected 16 fixed flow steps for range bearing. The preparation
master carries that choice through fresh offline tuning and saved selection
consumption; other models retain their selections. The reviewed continuation
is `surrogate-hmc-range-bearing-flow16-2026-10-16.md`, implemented by
`scripts/run_surrogate_hmc_range_flow16.py`. CPU-only checks passed194 tests.
Its supervisor waited for the old master and then ran range tuning and GPU
mechanics, both passing. Inference exhausted the final allocation during
reported XLA compilation without saved sequential samples. Its terminal
state is `artifacts/surrogate-hmc-range-bearing-flow16-20261016-01/state.json`.
This repair remains part of the five-model master. The earlier eight-step
health failure and local 16/32-step CPU diagnostic evidence remain separate.

The owner directed adoption of literature priors on 2026-10-15, replacing the
earlier pending owner-prior specification. The implemented option is
`literature-model-priors-v1`, now the default in both this master and the
performance/preparation launcher. `benchmark-normal-v1` remains mechanics-only.
Scientific inference rejects it. The prior density, its analytic score, and
coordinate Jacobians are shared by the force and independent MH endpoint.

Source settings, explicit adaptations, downloaded papers, author-code checks
and derivations are in `../papers/surrogate-hmc-priors-2026-10-15/README.md`.
The adoption plan is `surrogate-hmc-literature-priors-2026-10-15.md`.
The prior and implementation hashes enter the tuning identity. Historical
benchmark-prior selections cannot be relabeled or consumed by the new scopes.

All five accepted-OT tuning/mechanics/consumer checks passed for their earlier
scopes. Those results remain historical evidence of that preparation; they do
not validate these newly defined posteriors. Keep N4032/K2016, FP32/TF32 off,
GPU/XLA, epsilon1/five OT steps, and SIR's FP64-recurrence/FP32-pair exception.
The preparation launcher's `--priors literature-model-priors-v1` option carries
the new choice through its existing accepted-OT execution-plan route. Use that
route to preserve the settled OT controls and model-specific precision when
issuing fresh selections. Use `continue-advance --advance-root ...` to consume
the preparation launcher's actual nested output. The older `--prepared-root`
adapter is for this master's flat preparation layout only.

Fresh literature-prior tuning and mechanics have passed for all five models.
The continuation
adapter validates actual selections, source/data/prior/precision identities,
endpoint replay, mechanics health and selection binding. It preserves SIR's
mixed precision and charges each inference worker to the original cumulative
GPU ledger. Its checked path reaches the public tuner and shared adaptive
controller; it does not repeat completed filter tuning or mechanics.

Adaptive HMC is now running as part of this same master, in order KSC,
range bearing, LGSSM, predator–prey and SIR, subject to remaining
compute. Retain the 86,400-second total campaign cap. The bounded continuation
allocation and attempt-limit revision are in the literature-prior plan. There
is no separate terminal validation chain. As of 15:36 UTC, KSC has completed
the declared sampler checks with 1000 retained draws per chain and maximum
modern R-hat 1.006. This is convergence evidence for its finite-grid endpoint;
posterior agreement and endpoint refinement remain pending. Range bearing
stopped after 250 warmup transitions per chain on a surrogate-force-fallback
health veto, with no retained draws; the other chunk health checks passed.
The diagnosis is now recorded in
`artifacts/range-bearing-warmup-fallback-20261015-01/result.md`: an eight-step
left-endpoint Euler flow nearly collapses one proposal covariance at
observation14. FP32 16/32-step full replays pass both inspected inputs. These
are repair candidates requiring fresh scope tuning and GPU/XLA checks before
retry, not new defaults or convergence evidence. The structural integration
candidate and cross-model checks are in that result. This repair belongs to
the same master; no live runtime or frozen selection was changed. LGSSM is
running; predator–prey and SIR are queued. The live ledger and checkpoint
supply current process/budget state.
Preparation results and the handoff audit are in
`artifacts/surrogate-hmc-literature-priors-20261015-01/continuation-result.md`.
This section and `surrogate-hmc-performance-repair-checkpoint-2026-10-11.md`
supersede older phase status and benchmark-prior instructions below.


The question is whether a frozen covariance-guided defensive-mixture filter's
analytical score can serve as a **surrogate proposal force** in Metropolis-
corrected HMC for LGSSM, KSC SV, predator–prey, SIR (all T=50), and the saved
range-bearing model (T=20). Ordinary FP32 with TF32 **off**, N=4032, GPU/XLA,
and TensorFlow memory growth are fixed for every force evaluation. An exact or
separately frozen endpoint potential, plus its prior and parameter chart, must
be stated per model. The finite endpoint is the MH target; where it is an
approximation, successful chains do not establish the original model posterior.
The force need not equal the endpoint gradient. Using the filter's own finite
value as the endpoint without a separate target would answer a different HMC
question and is forbidden in the surrogate-force arm.

The comparator is exact-gradient HMC on the **same endpoint** wherever its
gradient exists, and a checked random-walk Metropolis baseline otherwise.
LGSSM's Kalman likelihood is exact for the specified model; KSC uses a refined
grid with discretization checks; predator–prey and SIR need a frozen
independent finite filter endpoint; range bearing needs its full-support
fixed-design importance endpoint. All endpoint designs use N=4032 per
replicate. Independent higher-accuracy references are posterior-accuracy
diagnostics, never substituted silently for the fixed MH endpoint. Priors,
charts, seeds, data checksums and endpoint source closure must be bound before
sampling. Existing T=2 KSC and N=4096 range HMC bindings do not match these
scopes and cannot be reused as tuning authority.

The initial execution is a bounded **readiness and cost phase** (at most 3600
GPU-wall seconds, five model attempts, 900 seconds per attempt). It constructs
the five exact model/data scopes, verifies N/dtype/TF32/device/chunk policy,
and measures one complete force value plus every analytical score coordinate
on each scope. Saved FP64 controls may seed this measurement only. Its outputs
are diagnostic and cannot be called tuned HMC or posterior evidence. The
master refreshes a next-phase plan after every model. A full inference launch
requires fresh offline control tuning for each precise model/T/N/FP32/backend
scope on disjoint calibration and validation data, a frozen untouched claim
dataset, an endpoint/prior/chart audit, and a measured four-chain cost forecast
inside a separately recorded total budget. No inherited control or benchmark
pass is upgraded to tuning admission.

For an admitted HMC phase, the public `tune_hmc_kernel` position-field route
with repository-issued `bind_neural_force_hmc_tuning_runner` is the supported
mechanics tuner. Its typed position-field artifact is **conditional mechanics
evidence only**, not ordinary exact-score tuning authority. Warmup is adaptive
on a recent-window maximum rank-normalized split/folded R-hat, with minimum
2000 and maximum 10000 transitions per chain. Every healthy fixed-kernel
postwarmup draw, including public-tuner verification draws, is retained
cumulatively. Check health every 250 transitions and cumulative R-hat every
1000; stop at <=1.01 or 20000 per chain. No second validation sample is
required after a qualifying cumulative sample. Report acceptance, energy
errors, bulk/tail ESS, MCSE, between-chain movement, posterior-reference
differences and runtime. Acceptance and an isolated low ESS are explanatory;
invalid endpoint, nonfinite states, divergence, stale identity or broken
sample continuity are vetoes. A failed force or mixer triggers a documented
scope-specific repair, not rejection of the other four models.

The promotion criterion is healthy retained sampling with cumulative maximum
rank/folded R-hat <=1.01 and predeclared posterior agreement with the target's
independent reference at uncertainty-aware margins. Any poor conditional
performance against exact-force HMC or simple RWM blocks a practical-value
claim. No superiority ranking follows without paired uncertainty. For
predator–prey, SIR and range bearing, finite-endpoint convergence alone never
certifies the exact model posterior. The previous SIR FP32 score sensitivity
is an explicit repair trigger; it does not erase a finite likelihood result.

Skeptical audit before readiness execution: the old all-model master has no
five-target T=50 registry; the N1008 FP64 tuning artifacts are stale at N4032
ordinary FP32; KSC's old exact T2 enumeration is not a T50 endpoint; nonlinear
exact-likelihood references do not exist; and the saved range HMC endpoint
uses N4096. Reusing any of those as claim evidence would give a misleading
answer. The readiness phase uses only saved data and warm-start controls and
stops short of that claim. Its measured cost and failures determine whether a
full five-model HMC campaign fits a proposed budget. These limitations are
recorded as program statuses, not hidden by a success exit code.

Default audit: FP32/TF32-off and N4032 are explicit owner instructions;
T50 is explicit for four models, while T20 for range bearing is the saved-data
scope and is provisional until a new horizon is requested. The covariance
filter is an optional candidate, not the canonical LEDH route. Saved FP64
controls and model truth points are warm starts, not defaults. The 3600-second
readiness cap is an initial local-resource limit, not an inferred authorization
for an unbounded multi-day posterior campaign. A fast but invalid filter, an
imprecise endpoint, stale controls, or short chains could all make an apparent
pass meaningless; the earliest checks are scoped source/data hashes, status,
independent value/score references, and a measured transition pilot.

Program: `scripts/run_surrogate_hmc_five_model_master.py`.
Versioned output root: `docs/plans/artifacts/surrogate-hmc-five-model-n4032-20261011-01`.
The master stores the exact command, Git/source hashes, model/data seeds,
memory/device policy, wall time, result paths and a refreshed next-phase plan.
An invalid scientific target or corrupted artifact stops its dependent HMC
phase. Other model diagnostics continue inside the cap.

## Readiness attempt and bounded repair

Attempt 01 ran all five model workers within 122 seconds. LGSSM, KSC, predator–prey and SIR returned finite full-direction values and scores. Range bearing stopped before filtering because the master unpacked the range `specs()` pair in the wrong order; this is a harness error, not evidence against the model or method. The range unpacking is repaired and the worker now repeats the full score at fixed inputs to check deterministic replay and record warm-call cost. Attempt 02 uses a fresh output root `docs/plans/artifacts/surrogate-hmc-five-model-n4032-20261011-02` under the unchanged one-hour readiness cap. Attempt 01 remains preserved.

## Execution review and current contract (2026-10-11)

The complete runner is now implemented provisionally. The GPU force-readiness
phase has finished for all five models in attempt 02; HMC has not run.
Endpoint-only checks and focused program tests continue under the original
3600-second readiness allocation (324 seconds used before this review).
Posterior execution awaits the pending owner choices of prior and total wall
budget. Elapsed time is not approval for either choice.

The proposed benchmark parameterization is q ~ N(0,I), with nominal theta0 from
each fixture. LGSSM, KSC, SIR and range bearing use theta=theta0+scale*q, with
scales respectively (0.4,0.5,0.5,0.2), (0.5,0.5), (0.35,0.35,0.35) and
(log(2),log(2)). Predator-prey uses theta=theta0*exp(0.35*q). These are explicit
benchmark prior hypotheses, not recovered priors from earlier posterior runs.
The prior is defined in q; the induced physical-parameter prior needs no extra
Jacobian in the q target. The proposal force is q - J(theta,q)^T score(theta).
The sign follows the repository momentum update p <- p - epsilon*grad(U).

The independent endpoints are exact Kalman likelihood (LGSSM); the KSC mixture
model's 1201-node, bound-48 grid (with a 1601-node, bound-56 refinement check);
four independent frozen bootstrap designs of 4032 particles each (prey and
SIR); and four full-support trajectory-importance designs of 4032 particles
each (range). Particle replicates are combined as the mean likelihood, not
mean log likelihood. Their finite deterministic likelihoods define approximate
posterior targets. Passing HMC tests for these targets does not certify the
original nonlinear-model posterior. All force calculations use ordinary FP32,
TF32 off. Independent reference calculations and HMC coordinate/energy
arithmetic use FP64 as explicit reference/accumulation exceptions.

Program phases:
1. Recheck the independent endpoint, replay, nominal-point refinement and
   replicate uncertainty. This needs no posterior-prior decision.
2. Tune the proposal in its own model/T/N4032/FP32 scope using two fresh
   calibration and two disjoint validation datasets; keep the saved test
   dataset untouched. Compare inherited controls, doubled flow/reset
   resolutions and half transport epsilon, with calibrated versus equal
   defensive mixture weights. The loss nominates an exploratory force; it
   cannot promote posterior correctness or support a superiority ranking.
3. Run two transitions of four chains as a mechanics and cost check.
4. Use the public frozen-position-force tuner, adaptive warmup (minimum 2000,
   latest 1000 modern R-hat <=1.05, maximum 10000), then cumulative retained
   sampling (modern R-hat <=1.01, maximum 20000 per chain). Every chunk is
   archived and checked for finite quantities, target status, movement and
   energy errors. All retained draws are inference draws. There is no second
   validation phase. Failing a diagnostic window during retained sampling
   does not erase earlier draws or require every window to pass.

The first three phases precede expensive inference for all models. Each model
has separate results; a candidate failure does not invalidate another model.
The parent enforces the explicit wall budget and refreshes next-phase-plan.json
after each stage. Chunk archives survive interruption. Automatic restoration
from an arbitrary interrupted warmup is not yet implemented; the master must
not claim otherwise. Public healthy retained-sampling continuation may be used
only after its saved manifest passes the existing resume checks.

### Default and skeptical audit

| Choice | Provenance and role | Failure mode and early check |
|---|---|---|
| FP32, TF32 off; 4032 particles | Explicit current user instruction | Nonfinite or inaccurate force; full-score fixed-input replay and MH health |
| Range T20 | Existing saved range dataset; stated assumption | Wrong requested horizon; manifest exposes it |
| Benchmark priors above | Proposed hypothesis, pending owner choice | A different posterior; require explicit choice before HMC |
| Three proposal-control candidates | Bounded scope-calibration hypotheses | Incomplete search or poor posterior-region transfer; record each candidate and check downstream trajectories |
| Zero additional covariance correction | Inherited comparison arm, not promoted default | Missed robustness benefit; cannot claim optimized algorithm or default readiness |
| Mass windows 128/256, step adaptation 256; L<=4 | Initial tuning hypotheses | Short trajectories/mass mismatch; convergence, geometry and timing diagnostics |
| Shrinkage 0.1; zero covariance jitter | Shared affine-tuner hypothesis with relative eigenvalue checks | Near-singular estimate; fail-closed geometry checks, no silent numerical patch |
| Warmup and cumulative thresholds | Existing BayesFilter procedure plus user's 20000 cap | R-hat alone can miss target error; keep endpoint accuracy separate |
| Energy threshold 1000 | Existing shared health screen | Does not certify good integration; report actual errors and acceptance |
| Compilation-inclusive two-step timing | Explanatory planning estimate | Not a lower bound or tuned runtime forecast; label and refine before budget conclusions |

Promotion criterion: healthy cumulative HMC sampling for the explicitly stated
finite target, plus posterior/reference agreement before any scientific
posterior-accuracy claim. Hard validity failures veto the affected candidate.
Missing/corrupted target identity or archives veto dependent execution.
Insufficient mixing or a tuning failure triggers a scoped repair within budget;
it is not evidence rejecting surrogate HMC. Acceptance and runtime explain
behavior and do not certify convergence.

The comparator promised in the initial plan is not yet executed. Exact-gradient
HMC is available for smooth exact/refined references; bootstrap-resampling
endpoints are not smooth-gradient oracles. For those targets, a symmetric
random-walk MH comparator is appropriate. Prior-only proposal dynamics, a plain
Gaussian approximation and an independently mixed same-target chain are cheap
falsification controls for prior-dominated, concentrated and nonlinear regimes.
No efficiency/superiority/default claim is allowed until the applicable
controls and posterior agreement have actually been evaluated. The current
master's sampling status must be reported separately from these open checks.

Skeptical review caught and repaired: a potential-gradient sign error before any
HMC execution; a wrong GPU-memory helper import; range-reference tensors mixed
between FP32 and FP64; missing finite validation-loss checks; acceptance
probabilities mislabeled as realized acceptance; and an initial-run retained
cap that incorrectly allowed 20000 only on resume. Focused executable checks
are required before considering these repairs verified.

Pre-mortem: a high acceptance rate with tiny steps can still mix badly; a
converged finite-particle target can still differ from the scientific posterior;
a correct force at the nominal parameter can still fail away from it; and a
completed command can still represent a short mechanics check only. Reports
must expose each distinction rather than turn successful execution into
scientific promotion.

### Attempt 04 and focused repair

Attempt04 consumed about 757 GPU-worker/parent wall seconds. Four independent
endpoints passed; the LG Kalman wrapper needed AutoGraph conversion of the
existing TensorFlow time loop (no mathematical change). KSC, predator-prey and
SIR completed proposal-scope selection. Range calibration's beta optimizer
failed its convergence test for all three controls. The existing equal-mixture
comparator was incorrectly rejected merely because the separate calibrated
candidate had not converged. Repair: retain the failed optimization diagnostics,
reject that calibrated candidate, and permit the separately evaluated equal
mixture only when all context/finite checks pass. This implements the declared
comparator; it does not loosen the optimizer tolerance. Record beta objective,
gap and iteration count instead of discarding them.

The CPU-only Gaussian reference exercised the actual public tuner, adaptive
warmup and cumulative retained writer: warmup 2000, retained 1000 per chain,
maximum R-hat 1.0011184, no extra validation phase. Its final test assertion
had a float32/float64 constant mismatch, corrected in the test. This is
engineering evidence only, not a five-model posterior result.

A second focused regression and a fresh preparation attempt will verify these
localized repairs. Runtime source hashing deliberately invalidates the previous
scope selections after a runtime edit; reissue them by retuning, never by
relabeling old selections. The cumulative preparation allocation remains
3600 seconds; after readiness324 and attempt04~757, at most2400 seconds are
allocated to attempt05 including tests. The live parent stops on that cap.


### Preparation result and bounded mechanics continuation, 2026-10-11

Attempt06 completed endpoint replay and fresh own-scope FP32/N4032 filter
selection for all five saved datasets in 928.79 wall seconds; its terminal
status is PREPARATION_COMPLETE_HMC_NOT_RUN. The previous endpoint, calibration
and validation artifacts are reused only after scope, source, data, selection
hash, endpoint finiteness and exact replay checks. No real-model HMC chain
has run. Focused CPU tests exercise the actual public tuner on a known Gaussian
target; those draws are not evidence for any of the five models.

Skeptical cost audit: measured warm full analytical scores range from about
1.86 seconds (KSC) to 48.37 seconds (SIR) for one position on the requested
GPU. A four-chain step evaluates multiple forces and endpoint potentials.
Multiplying these measurements by the mandatory minimum 2000 warmup steps
suggests that all-five posterior completion may exceed a 24- or 48-hour
campaign even before retained draws. This extrapolation is diagnostic only:
the first compiled four-chain HMC steps will be timed directly and no
unaffordable full run will be silently launched. The current implementation
loops over chains and score coordinates; that is a likely cost driver.
Each inference model gets an equal share of the remaining wall budget, with
unused time carrying forward, so the first model cannot spend the entire
campaign allocation.

The next bounded stage is a four-chain, two-transition **mechanics and cost
diagnostic only** for each model, in the order KSC, range bearing, LGSSM,
predator-prey, SIR. Its proposed q~N(0,I) benchmark prior is an explicit
diagnostic fixture, not a decision on the final scientific posterior. It
tests finite endpoint and force, fallback/divergence and actual transition
wall time. Acceptance is explanatory only. No R-hat, ESS, posterior agreement,
or prior choice follows from two transitions. A finite but expensive pilot
triggers a feasibility repair; an invalid pilot triggers a local correctness
repair; neither is a rejection of the research direction. Stop at 1200 wall
seconds total in fresh versioned output root
`docs/plans/artifacts/surrogate-hmc-five-model-n4032-20261011-07`.
This keeps cumulative readiness, repair, preparation and focused-test effort
inside the previously stated 3600-second preparation allocation. Exact launch
(after CPU regression and escalated GPU availability check):

```bash
/home/chakwong/anaconda3/envs/tftwogpu/bin/python scripts/run_surrogate_hmc_five_model_master.py campaign --through mechanics --priors benchmark-normal-v1 --prepared-root docs/plans/artifacts/surrogate-hmc-five-model-n4032-20261011-06 --budget-seconds 1200 --gpu 1 --output docs/plans/artifacts/surrogate-hmc-five-model-n4032-20261011-07
```

Full retained inference remains a separate serious launch. Its target prior,
posterior-agreement reference, and total budget must be explicit; the prior
used in this mechanics check does not silently settle those choices. Prior
saved KSC HMC used Uniform(.1,.9) on physical parameters, so the proposed
normal benchmark is materially different. The current master does not compute
same-endpoint comparator chains or posterior agreement and cannot make an
efficiency, original-posterior, or default-readiness claim from a VERiFIED
controller status alone.


### Attempt07 mechanics/cost result, 2026-10-11

The focused CPU review passed 28 tests in 17.78 seconds. GPU availability was
verified before launch. Using the reusable preparation from attempt06, the
master ran the declared four-chain/two-transition mechanics diagnostic with
ordinary FP32, TF32 off, N=4032, XLA and memory growth. KSC, range bearing,
LGSSM and predator-prey completed with finite states, zero divergences and zero
force fallbacks. SIR reached its fair 492.7-second share without a durable
pilot artifact and was recorded as STAGE_TIME_BUDGET_STOP. The campaign ended
at 1200.19 seconds with status INCOMPLETE_SEE_MODEL_STATUS. The exact
interpretation and tables are in
docs/plans/surrogate-hmc-five-model-n4032-result-2026-10-11.md.

The measured two-transition costs imply minimum 2,000-warmup extrapolations of
about 11.1 hours for KSC, 15.4 hours for range bearing, 48.8 hours for
predator-prey and 71.1 hours for LGSSM. These include first XLA compilation and
are diagnostic rather than steady-state forecasts. They nevertheless veto an
unexamined claim that the unmodified all-five adaptive campaign is a routine
24-hour run. The next phase must either optimize and remeasure the serial
score-coordinate HMC path or receive an explicit larger compute allocation.
It must also settle the final prior: benchmark-normal-v1 was used only as a
mechanics fixture, whereas earlier KSC work used Uniform(.1,.9) on physical
parameters. Full inference was therefore not launched.
