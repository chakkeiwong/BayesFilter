# Simpler-model NeuTra master: first allocation and terminal review

Historical first-allocation closeout. The owner subsequently added 48 hours;
active continuation and subsequent findings are recorded in
`bayesfilter-neutra-warm-start-continuation-2026-09-29.md`. The costs, failed
attempts and conclusions below are preserved as the first-allocation record.

The resumable master is implemented and has executed 25 complete training runs
on five simpler targets. Five frozen-map configurations passed the declared
downstream HMC checks. The full study is incomplete: the first CPU allowance
has no conservatively established remainder, and the master now stops there.
No q20 experiment or default promotion follows from these results.

Entry point: `scripts/run_neutra_warm_start_master.py`.
Reviewed plan: `bayesfilter-neutra-warm-start-master-2026-09-29.md`.
Campaign: `artifacts/neutra-warm-start-master-2026-09-29/campaign-r1/`.
The campaign's `summary.json`, `result.md`, and individual attempt directories
preserve the measured results. `continuation-proposal.json` records the ordered
next commands; its additional allocation has not been applied.

## Implementation and checked scope

The master prepares correlated Gaussian, unequal Gaussian mixture, invertibly
warped mixture, pinned author-code wiggle, and ten-dimensional funnel targets.
It implements batched multistart optimization and score/curvature checks,
duplicate-mode merging, target-specific training calibration, forward warm
starts, classic RKL, before/after coverage diagnostics, and the standard
1,000-base-point score probe. Independent forward initializers are exact
examples, Gabrié sampling/training, ordinary SMC, waste-free SMC, AFT and CRAFT.
The separate discovered-mode Gabrié arm is implemented but unexecuted. The
Gabrié-to-SMC composition remains a later experiment, as in the design.

Every final learned map uses the shared configured author IAF. Actual arithmetic
in this checkout is FP64; the enabled TF32 flag does not change that fact.
GPU workers used host GPU 1 with memory growth before initialization. Reference
preparation intentionally hid GPUs. Training uses native batched TF/XLA kernels
and the shared trainers. Adaptive stage orchestration, checkpoint writing and
reporting stay outside those kernels. AFT/CRAFT still have host-controlled stage
and optimizer-call loops; they must not be described as fully fused training.

Frozen-map qualification calls the public fixed-transport tuner and shared
sequential NeuTra controller. Four chains run as one batch with identity mass
in latent coordinates. Warm-up draws are archived and excluded from estimates.
No new mass adaptation or private HMC tuner was added.

Preparation and calibration completed on all five targets. Mode discovery found
one Gaussian mode, two modes in each mixture, and three local maxima for the
actual author-code wiggle. The funnel density mode is at v=-9; initialization
uses scale-region representatives because the density maximum is atypical of
the probability mass. Finite multistart search does not prove complete mode
discovery on unknown targets. Wiggle's reference is checked quadrature, so its
exact-example oracle arm is deliberately absent.

The latest complete component run before the supervisor closeout repair passed
33 tests (`attempts/checks-ce4e5898d46d-r1/process.log`). These include target
finite differences, actual pinned-author controlled MH/MALA trajectories and
wiggle density/score, SMC weights and telescoping, AFT free-energy gradients and
pre-update selection, CRAFT update ordering, real shared optimizer updates,
clipping diagnostics, and runner behavior. The controlled author trajectory
agreed within approximately 2e-14. This checks selected numerical operations;
the full AFT/CRAFT adaptations are not established feature-equivalent ports of
the authors' JAX implementations.

The budget/recovery repair passed 12 focused supervisor tests, with three
unrelated TensorFlow tests deselected, in 0.45 CPU core-seconds. That routine
engineering verification is separately metered in
`artifacts/neutra-warm-start-master-2026-09-29/engineering-closeout-checks-r1/`.
A real master resume then exited with budget status 3, no worker launch and no
additional campaign attempt. The full repository route-discovery check has an
unrelated existing failure: `docs/benchmarks/run_q20_configured_hmc_2026_09_24.py`
is unregistered. The new qualification consumer's own route registration passes.

## What the first seed shows

All completed training uses confirmation seed 11. Seeds 23 and 37 remain
unexecuted. The completed set comprises all five plain-RKL, Gabrié, ordinary-SMC
and waste-free-SMC targets, four eligible oracle targets, and Gaussian AFT.

| Target | Initializer before RKL | Warm-up per chain | Retained per chain | Downstream result |
|---|---|---:|---:|---|
| Gaussian | Plain RKL | 2,000 | 1,000 | Declared checks passed |
| Gaussian | Exact examples | 2,000 | 2,000 | Declared checks passed |
| Gaussian | Gabrié | 2,000 | 1,000 | Declared checks passed |
| Warped mixture | Gabrié | 2,000 | 2,000 | Declared checks passed, diagnostic revision 2 |
| Funnel | Exact examples | 2,000 | 4,000 | Declared checks passed |

These checks include numerical health, modern R-hat, bulk/tail or binary-event
ESS, mean precision, and agreement with independent reference features. They
are finite operational checks on one frozen-map replication, not proofs of
convergence, exhaustive coverage, or method superiority. Full target-status
telemetry is absent for these analytic fixtures; finite target/score and
log-acceptance checks are available. Large finite energy errors remain recorded
explanatory diagnostics under the shared controller's current policy.

The ordinary mixture exposes an important failure of the proposed RKL phase.
The following numbers are sampled minority-component responsibility means;
the target value is 1/3. They are descriptive and do not rank viable methods.

| Initializer | Before RKL | After RKL | Coverage screen after RKL |
|---|---:|---:|---|
| Plain RKL | N/A | about 7e-11 | Failed |
| Exact examples | 0.358 | about 7e-11 | Failed |
| Gabrié | 0.361 | about 7e-11 | Failed |
| Ordinary SMC | 0.286 | about 8e-11 | Failed |
| Waste-free SMC | 0.365 | 0.402 | Passed coarse screen only |

The plain-RKL ordinary-mixture map had score-residual p95 of only 0.094 while
missing the minority basin in the generated sample. Plain RKL also missed a
warped-mixture basin with p95 0.165. Thus the 1,000-point base probe cannot
establish posterior coverage. Its small value can describe a well-fitted single
basin. The independent physical-region check detected both failures.

The source of the ordinary-mixture loss is localized to the tested RKL stage:
oracle, Gabrié and ordinary-SMC warm maps passed the coverage screen beforehand
and failed afterward. This weakens this specific RKL schedule, not all reverse
KL training. The implemented repair restores the exact pre-RKL map, uses the
already-calibrated lower learning rate .001, doubles batch 256 to 512, and keeps
the same 256/1,024/2,048 update ladder. It is a schedule/noise hypothesis, not a
proven remedy, and has not yet run. These three collapsing runs had zero
reported clipping in all their recorded training blocks, so excessive clipping
does not explain these particular failures.

Warped-mixture oracle, Gabrié, ordinary SMC and waste-free SMC retained coarse
coverage through RKL. Waste-free's minority estimate was 0.370; ordinary
mixture waste-free was 0.402. Passing a coarse region screen does not establish
accurate basin weights. The ordinary-mixture waste-free map also had a large
score-residual p95 (42.7); its usefulness requires actual downstream testing.

Other unresolved outcomes:

- Plain-RKL funnel found verified tuning candidates, then failed warm-up after
  1,000 transitions because some log-acceptance ratios were nonfinite. Retained
  sampling did not start. This is a numerical qualification failure of the
  tested map/kernel, even though stored states remained finite.
- Wiggle plain RKL failed to find a verified pair in the initial and wider
  bounded searches. Gabrié's initial wiggle search also failed. Other maps or
  repaired tuning remain untested; this does not reject the target or method.
- The warped oracle run reached 10,000 retained draws per chain. Its old
  diagnostic used continuous-quantile tail ESS for saturated responsibilities,
  which was inappropriate. Revision 2 uses the actual half-space event and
  binary-event ESS. The original failure is preserved; separate revised
  qualification is pending. It is not retroactively counted as a pass.
- AFT mixture completed four of eight stages before its 240-second cap; warped
  AFT completed three before interruption/recovery. These are incomplete cost
  outcomes. Gaussian AFT completed, but CRAFT and harder AFT targets have not
  completed GPU training. Their numerical mechanics tests are narrower evidence.

## AFT timing and accounting

The first price diagnostic used only one update at each of two partial stages.
Its first gradient/validation/advance calls took 2.07/7.82/4.00 seconds; second
calls took 1.51/.0015/1.88 seconds. Total process cost was 26.03 wall/GPU and
88.46 CPU seconds. This shows startup cost but does not explain the full-loop
stall. A single slow-period device snapshot showed zero GPU utilization while
the process consumed multiple CPU cores. Compilation/retracing/dispatch remain
unresolved hypotheses.

Pricing revision 2 is prepared to instrument all 32 updates, best-checkpoint
selection and all three populations over two actual AFT stages. It records
each active call, CPU/wall cost and trace count under the existing 120-second
cap. It deliberately synchronizes calls, so it is a localization diagnostic,
not a throughput benchmark. It has not executed. No speculative numerical
change was made to AFT on the basis of the first timing result.

| Resource | First ceiling | Current charge | Conservative remaining |
|---|---:|---:|---:|
| GPU process seconds | 7,200 | 2,969.54 | 4,230.46 |
| CPU core-seconds | 7,200 | 6,465.24 estimated; 7,379.24 upper charge | -179.24 |

One old recovery lost the warped-AFT final CPU counter. The point charge is
960 seconds; the recorded launch limit gives an upper charge of 1,874 seconds.
The other recorded CPU charges total 5,505.24 seconds. Actual total CPU usage
is therefore unknown, and compliance with the original ceiling cannot be
certified. The separate 0.45-second engineering verification is not hidden in
that estimate. This accounting failure does not change the saved numerical
results, but it blocks further campaign launches under the current allocation.

The repaired supervisor reserves the upper charge when deciding whether to
launch, exposes the estimate separately, preserves budget status across phase
wrappers, and reserves an interrupted worker's recorded launch allowance if
its final counter is unavailable. It does not silently treat the configured
TensorFlow thread count as the compiler's CPU usage. Unknown SIGKILL is still
an unclassified failure; explicit CPU-limit and timeout outcomes are separate.

## Terminal decision and skeptical review

| Decision | Primary criterion status | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Preserve five qualified configurations | One-seed downstream checks passed | No declared veto in these runs | Replication and target transfer | Independent seeds after funded repairs | Default or q20 readiness |
| Reject collapsed ordinary-mixture candidates | Region coverage failed after RKL | Candidate promotion veto | Schedule, batch noise and capacity | Restore pre-RKL map and test declared repair | Failure of every RKL method |
| Keep waste-free mixture candidate for testing | Coarse coverage passed | No training validity veto | Basin-weight error and difficult geometry | Fixed-map HMC qualification | Accurate posterior from map draws |
| Repair plain funnel map/kernel | Downstream health failed | Nonfinite log-acceptance ratio | Rare trajectory instability | Declared training/tuning repair | General NeuTra failure |
| Diagnose AFT cost | Full teacher incomplete | Per-job cost cap | Slow call not yet isolated | Full-inner-loop pricing revision 2 | Evidence against AFT mathematics |
| Close first allocation | Replicated study incomplete | Conservative CPU capacity exhausted | Lost orphan counter | New allocation or accounting reconciliation | Entire campaign complete |

| Inference status | Finding |
|---|---|
| Hard veto screen | Mixture coverage failures and funnel nonfinite acceptance are supported; AFT cost caps are not scientific rejections |
| Statistically supported ranking | None |
| Descriptive-only differences | Loss, region probabilities, score residual quantiles, timings and clipping rates |
| Default readiness | Not established |
| Next evidence needed | Repaired RKL runs, fixed-map qualification and independent training/confirmation replications |

The strongest alternative to a general failure of RKL is insufficiently tuned
optimization of an imperfect warm map. Several arms were still improving at
their final rung; none is certified fully trained. Capacity and optimizer
nomination used exact/quadrature benchmark references, so these settings are
oracle-calibrated fixture settings and cannot transfer directly to q20.
MALA sensibility was checked on the endpoint target, not separately optimized
for every SMC bridge. AFT/CRAFT stage-gradient clipping inherits the calibration
candidate from forward fitting and has not received its own sustained pilot.
Those are explicit hypotheses to inspect before interpreting an AFT/CRAFT
quality failure. Full JAX/TF training equivalence and general mode discovery
also remain unproved. The evidence's weakest part is the single confirmation
seed and incomplete difficult-arm execution.

These observations invalidate particular candidates and reveal a diagnostic
and resource-accounting defect; they do not invalidate the analytic targets,
checked derivatives, or every warm-start mechanism. The next planned repairs
remain scientifically justified. Their execution stops here because the
resource continuation veto is real.

## Concrete continuation proposal

Proposed additional allocation: **3,600 CPU core-seconds**, with **no additional
GPU allowance**. This would change the CPU ceiling from 7,200 to 10,800 seconds
and leave approximately 3,420.76 CPU seconds after conservatively charging the
first allocation. The one-hour increment is a bounded diagnostic allocation,
not a claim that it completes the three-seed study. Configuration is unchanged
until the owner authorizes it.

The executable queue in `campaign-r1/continuation-proposal.json` prioritizes:

1. Ordinary-mixture Gabrié/oracle RKL-collapse repairs and qualification of any
   viable repaired maps.
2. Qualification of the existing ordinary-mixture waste-free map.
3. Warped-oracle qualification with the corrected binary-region diagnostic.
4. Full-inner-loop AFT pricing before any further expensive AFT teacher retry.
5. Mixture training from discovered modes, then a Gaussian CRAFT control, only
   if the preceding work leaves budget. Diagnose any new infrastructure failure
   before dependent work; retain the unchanged scientific criteria.

Every command resumes the same campaign and writes fresh attempt directories.
Its existing per-job, attempt and global ceilings remain active. AFT/CRAFT
production-sized retries and the remaining two seeds are not silently admitted
by this proposal. The full matrix has 86 uncompleted training jobs; the observed
completed-training mean predicts roughly 8,064 CPU and 4,182 GPU seconds for
training alone, excluding repairs and HMC. That estimate omits censored slow
AFT jobs and unmeasured CRAFT costs and is not a completion forecast.

Reapproval is required by `AGENTS.md`'s campaign rule when the total budget
changes. Ordinary local implementation and repair checks are already complete;
approval would authorize this concrete additional execution queue.

## Recovery state

This result note also serves as the reset memo. The branch remains
`preserve/shared-main-before-fab-20260926`, at Git HEAD
`de80aaff5812ebfbed551977476c0868551a2c88`. Changes from this task are
uncommitted; preserve the large unrelated dirty worktree. Per-attempt source
copies and hashes identify the executed versions. Current source additionally
contains the tested conservative accounting repair and the unexecuted pricing
revision 2.

The campaign has 62 recorded attempts, 25 complete training results, and no
active worker. State is `budget_exhausted`. Resume the existing output root
only after the CPU allocation is reconciled or amended with owner approval.
Do not reset its counters or create a fresh root to replenish the allowance.
Do not reinterpret the old saturated-responsibility qualification as a pass,
the 1,000-point score probe as coverage evidence, or an AFT timeout as algorithm
rejection. The next numerical action is the first command in the saved
continuation proposal once its allocation is authorized.
