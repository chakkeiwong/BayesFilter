# SIR-UKF Target-Specific NeuTra Campaign Plan

Date: 2026-07-30

Status: `TERMINATED_OWNER_EXCLUDED_FROM_TESTING`

Owner termination, 2026-07-31: UKF does not work for SIR. This plan is
historical evidence only and must not be resumed. No geometry repair, NeuTra
retraining, tuning, or HMC follow-up is authorized for `SIR-UKF`.

Campaign ID: `bayesfilter-sir-ukf-neutra-target-specific-20260730`

## Objective

Determine whether the newly admitted three-parameter `SIR-UKF` posterior can
support a freshly trained, fixed-identity-mass NeuTra HMC kernel under the
repository's batch-native GPU/XLA policy. Target execution admission is already
established; training, HMC convergence, posterior validity, and default
readiness are not.

## Research Intent Ledger

| Field | Binding decision |
| --- | --- |
| Main question | Can a target-specific dense-IAF transport for the admitted `SIR-UKF` target produce at least one healthy fixed transport whose complete broad-grid HMC candidate set can proceed to valid retained sampling under the shared sequential controller? |
| Candidate mechanism | Batched GPU/XLA reverse-KL dense IAF followed by the generic broad fixed-mass `L`/epsilon protocol and exact transformed-gradient HMC |
| Exact comparator | Same current `SIR-UKF` target signature `31bc2c569933523b7bda5e337ae5b7a307a7ed90656a1d6791523c69fb6b5998`; retained NeuTra results must ultimately be compared with a separately tuned same-target plain-HMC reference before a posterior or efficiency claim |
| Expected failure mode | Transferred architecture hypotheses fail on `SIR-UKF`; prior geometry is too weak; target evaluation dominates the budget; training becomes nonfinite; frozen parity fails; broad-grid candidates fail sampler validity; or retained chains fail convergence |
| Promotion criterion | Engineering phases: paired-seed screen machinery and one-step GPU/XLA preflight pass. Training: all target/status/frozen-parity hard gates pass and a recipe remains viable under paired uncertainty. Sampler: at least one complete broad-grid candidate passes the shared warm-up and retained validity/convergence gates. Posterior claim additionally requires same-target plain-HMC compatibility under a prospective uncertainty analysis. |
| Promotion veto | Nonfinite loss/gradient/weights, invalid target status, batch size one, sample-wise target fallback, wrong device, non-XLA training, memory-growth failure, frozen/trainable mismatch, candidate divergence/status/energy failure, or failed modern R-hat/ESS gates |
| Continuation veto | Target/source identity drift, harness invalidity, output collision, GPU/XLA failure, inability to afford the minimum target-specific protocol inside the remaining 24-hour campaign budget, or exhausted campaign budget |
| Repair trigger | Local seed-pairing, serialization, target telemetry, XLA compilation, memory, or orchestration failure under the unchanged target/method/budget |
| Explanatory diagnostics | Training and heldout reverse KL, gradient norm, runtime, tuned epsilon, acceptance, energy-error magnitude, posterior moments, and truth-tail values outside their declared hard-gate role |
| Must not be concluded | Approximate-filter exactness, epidemiological calibration, a statistically supported recipe ranking from three seeds, NeuTra superiority, universal validity, or default readiness |

## Entry Evidence

`SIR-UKF` target execution passed the frozen parity gate in:

```text
docs/plans/artifacts/bayesfilter-neutra-remaining-models-20260730/
  sir-ukf-parity-cpu-attempt-02/
  sir-ukf-parity-gpu-attempt-02/
```

The trusted GPU result has:

- value scale-normalized gap `2.790646646294622e-15 <= 1e-8`;
- score scale-normalized gap `1.184323487686088e-09 <= 1e-7`;
- exact status agreement;
- finite, GPU-resident XLA outputs; and
- verified TensorFlow memory growth before logical GPU initialization.

This admits target execution only.

## Evidence Contract

### Training screen

- Geometry baseline: zero center and `0.5 I` factor in raw log-scale
  coordinates, derived from the exact independent `Normal(0, 0.5^2)` prior.
  It is a baseline, not a posterior-optimal default.
- The historical `SIR-SGQF` Laplace geometry is target-mismatched and is not
  eligible as the `SIR-UKF` geometry. It may inform a later explicit
  comparator only; it cannot be copied into this campaign.
- Four target-specific architecture/optimizer hypotheses:
  - `compact_shallow`: two stages, `(9, 9)`, learning rate `1e-3`;
  - `compact_deeper`: three stages, `(9, 9)`, learning rate `1e-3`;
  - `wide_conservative`: three stages, `(18, 18)`, learning rate `1e-3`;
  - `wide_faster`: three stages, `(18, 18)`, learning rate `5e-3`.
- Every recipe uses the same three training seeds. A recipe index must not
  change its training seed.
- Heldout base-noise batches are disjoint from training noise and identical
  across paired recipes. Recipe comparisons use per-training-seed heldout
  means, not heldout batches as independent replicates.
- Heldout reverse KL is proxy nomination or veto evidence only. It cannot
  establish a good HMC transport, convergence, or superiority.
- Hard training gates are finite objective/gradient/weights, valid target
  status, batch-native execution with batch size `128`, GPU/XLA placement,
  exact frozen/trainable parity, and complete source/seed/config artifacts.
- A recipe is viable when all three replications pass hard gates. Among viable
  recipes, a candidate may be nominated when its paired mean loss difference
  from the descriptively lowest arm is within two paired MCSEs. If several
  remain viable, prefer the smallest parameter count; call this a deterministic
  representative choice, not a statistically supported ranking.

### Budget ladder

- Run one one-step preflight and measure compile time plus steady-state step
  time using a short repeated-step diagnostic.
- Serious screen rung is `500` steps per recipe/seed only if all twelve screens
  plus two final trainings project to at most 12 hours. Otherwise `250` is
  allowed only as diagnostic nomination and cannot promote a recipe to final
  training without a refreshed budget. `100` is mechanics only.
- Final training uses two fresh seeds, each `5,000` steps, only if their
  projection plus at least 8 hours reserved for broad-grid tuning, retained
  sampling, and same-target comparison fits within the remaining campaign
  budget. Otherwise stop as under-budgeted; do not promote a shorter convenient
  run.
- Total elapsed serious-campaign budget is 24 hours, including failed attempts.
  Engineering diagnostics before the serious launch are capped at 90 minutes.

### HMC sequence

- Freeze every final transport and discard every training/tuning draw.
- Run `broad-grid-frozen` independently for each healthy final seed using
  primary `L=(3,5,9,13,18,25)`, independently tuned epsilon per primary, three
  fresh acceptance replications, and nonrecursive same-epsilon one-hop
  coverage.
- Preserve the complete unranked viable primary-plus-coverage set. Coverage
  failure does not veto a viable primary.
- Sequentially validate candidates with the shared controller. Warm-up and
  retained draws follow `bayesfilter_neutra_sequential_hmc_v1`; acceptance is
  explanatory only.
- Energy error is a declared health diagnostic and repair trigger, not an
  automatic scientific veto by itself. Nonfinite state/target/log acceptance,
  invalid target status, divergence, no chain movement, and declared modern
  R-hat/ESS failures remain vetoes.
- If at least one NeuTra kernel is valid, tune and retain a same-target plain-HMC
  reference under its own fixed target identity. Compare posterior parameters
  using MCSE-aware intervals; do not claim superiority from runtime or ESS
  point estimates alone.

## Default And Assumption Audit

| Choice | Provenance | Justification | Failure mode | Earliest diagnostic | Status |
| --- | --- | --- | --- | --- | --- |
| `0.5 I` affine factor | Exact `SIR-UKF` prior scale | Target-specific and does not borrow another posterior | Posterior is much narrower or correlated, making training look worse than the method | affine heldout loss and one-step gradient/finite checks | baseline |
| Four dense-IAF arms | Dimension-scaled variants of prior SIR/NeuTra work | Tests depth, width, and two learning-rate scales within bounded compute | All arms may share an unsuitable capacity or optimizer family | paired three-seed screen and loss trajectories | hypotheses, not defaults |
| ELU, `s_max=1`, init `0.02`, Adam, clip `10` | Existing BayesFilter dense-IAF implementation | Keeps the first target-specific experiment comparable to repository mechanics | These convenience choices may cause common-mode failure | nonfinite/clip telemetry and failure interpretation; no default promotion | inherited mechanics hypotheses |
| Batch `128` | Existing batch-native GPU training route | Satisfies policy and has working infrastructure | May underutilize or exhaust GPU; target throughput may dominate | trusted one-step and short throughput preflight | reviewed starting hypothesis |
| Three screen seeds | Statistical evidence policy | Separates recipe effect from one random initialization while bounded | Still weak for a formal ranking | paired MCSE; all ranking language remains prohibited | minimum diagnostic replication |
| Two final seeds | Multi-seed downstream validation requirement | Detects a seed-fragile transport before promotion | Two seeds remain insufficient for a universal method claim | independent frozen parity and HMC gates for each | minimum promotion replication |
| `500`/`5000` steps | Existing serious NeuTra protocol | Comparable budget ladder with final retraining from scratch | Insufficient convergence or unaffordable target | measured throughput and loss trajectory; stop under-budgeted | reviewed target-specific budget hypothesis |
| Truth tail | Synthetic fixture provenance | Useful downstream check against generating values | Approximate filter posterior need not center exactly at generating truth | classify as diagnostic unless sampler valid; no calibration claim | secondary promotion diagnostic |

## Pre-Mortem

The campaign could pass while misleading us if heldout reverse KL selects a
transport that gives invalid HMC geometry, if final-score cancellation hides
per-parameter disagreement, if common heldout rows are treated as independent
training replications, or if the approximate UKF posterior is judged solely by
generating truth. The downstream HMC gates, per-training-seed aggregation,
same-target plain-HMC comparator, and explicit nonclaims address these risks.

It could fail for engineering or tuning reasons rather than scientific ones if
the master confounds recipe and seed, copied geometry mismatches the target,
XLA compilation dominates a one-step timing, a source hash is incomplete, or a
localized artifact failure aborts the campaign. Common seeds, prior geometry,
compile-versus-steady timing, source closure, fresh roots, and one localized
retry distinguish those explanations.

## Phases

1. **Close target admission**
   - Record parity attempt 02 and mark the old score-parity blocker resolved.
   - Preserve attempt 01 and localization artifacts as negative evidence.
2. **Repair generic screen design and registry**
   - Add paired replicated screen support to the generic master.
   - Aggregate uncertainty by training seed, not heldout batch.
   - Add `SIR-UKF` with its current target signature, prior geometry, and the
     four hypothesis recipes; remove its obsolete parity blocker.
   - Add contract tests for seed pairing, replicate aggregation, target
     signature, geometry provenance, and registry counts.
3. **Trusted GPU/XLA preflight and throughput**
   - Run a fresh one-step end-to-end training/tuner preflight.
   - Run a short repeated-step timing rung that separates compile and steady
     runtime and records allocator/device provenance.
   - Project the complete protocol and stop if the minimum serious plan cannot
     fit the remaining budget.
4. **Paired three-seed recipe screen**
   - Run all twelve 500-step screens under fresh roots.
   - Apply hard gates first, then paired uncertainty nomination.
5. **Two-seed final training**
   - Retrain the deterministic representative recipe twice from scratch for
     5,000 steps.
   - Freeze, hash, and independently validate each transport.
6. **Generic broad-grid tuning and sequential HMC**
   - Run the approved generic broad-grid route for each healthy transport.
   - Validate the complete candidate union without descriptive ranking.
7. **Same-target comparator and terminal inference**
   - Run tuned same-target plain HMC if NeuTra has a valid retained kernel.
   - Report hard vetoes, viable kernels, whether any ranking is statistically
     supported, descriptive-only differences, and the next evidence needed.

## Skeptical Pre-Execution Audit

- **Wrong baseline:** fixed by using the current `SIR-UKF` target and its exact
  prior geometry; `SIR-SGQF` is not the comparator or geometry authority.
- **Proxy promotion:** heldout reverse KL can nominate only; HMC and same-target
  evidence remain mandatory.
- **Missing stops:** 24-hour total cap, 90-minute engineering cap, fresh roots,
  hard training/sampler gates, minimum affordable protocol, and retry limit are
  explicit.
- **Unfair comparison:** current master fails this check because recipe index
  changes the training seed. Phase 2 is a mandatory repair before launch.
- **Hidden assumptions:** geometry, architectures, optimizer controls, seeds,
  batch, steps, truth role, and target approximation are recorded above.
- **Stale context:** parity attempt 02 supersedes the July 18 blocked registry;
  completed models remain excluded from rerun.
- **Environment mismatch:** serious commands require trusted GPU/XLA, verified
  memory growth, and batch-native target execution.
- **Non-answering artifacts:** preflight records throughput; screens record
  per-seed results; final runs preserve frozen parity; HMC records complete
  candidate and retained diagnostics.

Audit verdict before Phase 2: `REVISE_BEFORE_SERIOUS_EXECUTION`. The scientific
sequence was sound, but the master seed policy was confounded and the registry
lacked the admitted target.

Phase 2 repair result: `PASS_FOR_PHASE_3_PREFLIGHT_ONLY`. The generic master now
uses common screen seeds across recipes, aggregates recipe uncertainty by
independent training replication, preserves heldout batches as within-fit Monte
Carlo diagnostics, and binds `SIR-UKF` to its admitted signature, exact-prior
geometry baseline, four hypothesis recipes, three screen seeds, and this plan.
The complete master/registry contract suite passed (`36 passed`) before the
preflight. Serious screening remains blocked until the Phase 3 timing projection
shows that twelve screens, two final trainings, and the downstream evidence
reserve fit the 24-hour budget.

Phase 3 preflight attempt 01 produced one finite, target-valid, batch-native
GPU/XLA optimizer update and froze the transport, then failed during the
affine heldout diagnostic. A nonfinite proposal covariance reached GPU
`SelfAdjointEigV2`, which aborted with `heevd info=7` before the existing target
classification could issue its finite invalid-row rejection. Classification:
`LOCALIZED_NUMERICAL_SAFETY_HARNESS_FAILURE`; this is not a training-quality or
NeuTra result. Repair: sanitize only nonfinite principal-root covariance rows
before eigendecomposition, then restore their invalid evidence so they remain
classified invalid with zero score. Valid finite rows and the target remain
unchanged. Attempt 02 uses a fresh output root.

Phase 3 preflight attempt 02 passed training, heldout scoring, freezing, and
frozen/trainable parity, then emitted `TUNING_HARNESS_INVALID` before HMC
mechanics. The rank-1 HMC status adapter called the raw binding rather than the
normalized batch-native status function, so it omitted the optional
`innovation_condition_estimate` field that the tuner requires. This is not a
candidate hard veto. Repair: route rank-1 telemetry through the same normalized
status callable already used for rank-2 training and supply the explicit
availability flag. Attempt 03 uses a fresh output root with unchanged target,
recipe, seed, hardware, and scientific gates.

Phase 3 preflight attempt 03 reached real XLA HMC mechanics. Its public
bootstrap summary records six timed rounds, finite maximum round runtime
`32.082676945996354s`, observed acceptance both below and above the band, and
`preflight_passed=true`; the deliberately one-step transport then exhausted
its bootstrap repair budget and was rejected downstream. The wrapper had
incorrectly treated this expected candidate rejection as an engineering
failure. Repair: engineering preflight now requires those public mechanics and
timing fields but does not require a one-step candidate to pass scientific
tuning. Serious tuning and promotion gates are unchanged. Attempt 04 writes a
complete terminal preflight artifact in a fresh root.

Phase 3 throughput attempt 01 preserved a real candidate hard veto. The
one-step `compact_shallow` rung was finite and status-valid, but the otherwise
identical 25-step rung reached an invalid exact-target status after a GPU
Cholesky failure on one batched covariance row. This rejects that recipe/seed
replication under its declared hard gate; it is not a GPU, XLA, target-
admission, or throughput-harness failure. Attempt 02 used the minimum planned
10-step rung and completed both rungs, heldout scoring, freezing, and frozen /
trainable parity on GPU/XLA. It measured:

- compile-plus-first program time `6.415961015009088s`;
- compile-plus-first end-to-end time `133.6626326330006s`;
- 10-step program time `20.84147335300804s`;
- 10-step end-to-end time `135.36536228499608s`;
- conservative additional-step time `2.084147335300804s`, the maximum of the
  repeated-program average and the compile-adjusted program/wall slopes; and
- TensorFlow allocator peak `3,494,942,464` bytes under verified memory growth.

The initial projection implementation incorrectly included fixed heldout /
parity overhead both in the per-job intercept and in the per-step average.
That double count was repaired before making a campaign decision. The audited
projection charges 22 fresh jobs (12 screens and 10 recoverable final-training
segments), 15,978 post-first optimization steps, the measured `133.6626s`
fixed cost per job, and `2.08415s` per additional step. Training therefore
projects to `36,241.084041s = 10.066968h`. Adding the required `8h` downstream
reserve gives `18.066968h`, leaving `5.933032h` inside the 24-hour cap.

The throughput run also exposed two master-program defects before serious
screening. First, `screen_only` performed final training before returning;
the branch now stops immediately after recipe selection and records that final
training and HMC were not launched. Second, heldout scoring used finite
rejection values without preserving target status. It now obtains value,
score, and status in one transformed GPU/XLA call, records invalid-row counts,
and hard-vetoes a replication when any heldout target status is invalid. The
attempt-02 heldout loss around `4e97` is therefore descriptive invalid-candidate
evidence, not a viable reverse-KL score. Candidate hard vetoes are preserved
per replication so they do not invalidate or abort the paired experiment.

A final trainer trace found that the compiled graph calculated target-status
and numerical diagnostics for every optimizer step but the host hard-gate
check inspected only sparse heartbeat rows. A transient invalid update between
heartbeats could therefore escape the declared every-update veto. The compiled
program now emits all-step aggregate numerical, target-value, and target-status
gates in addition to sparse progress rows. The stochastic sequence, update
count, target evaluations, and graph-native `tf.while_loop` are unchanged.
Focused deterministic, invalid-status, and master-contract regressions pass.

The skeptical execution audit passes for Phase 4: the current target signature,
prior geometry, common seeds, hard gates, statistical unit, no-ranking rule,
fresh output root, and budget cap remain unchanged; the artifacts answer the
screen question; no proxy or energy diagnostic was promoted; and target-status
failures reject individual candidates rather than the research direction.
An unrelated LGSSM rank-1 static-shape tracing test remains failing in its own
time loop and is recorded as non-SIR infrastructure debt. A broader CPU-only
nonlinear suite passed 91 tests and failed 12 custom-op CPU-XLA tests because
`SymmetricPrincipalSqrt` has no XLA-CPU kernel; that historical backend is not
the admitted SIR-UKF Newton-Schulz route. The narrowed SIR/master suite passes
all 64 tests, and the trainer suite passes all 20 tests.

The serious screen writes its run manifest at launch and atomically finalizes
it at terminal. The launch snapshot records command, Git commit, environment,
target signature, GPU memory policy, batch size, XLA/TF32, paired seeds, output
root, plan, and planned result file. A no-survivor outcome also writes an
explicit terminal `selection.json`.

Screen attempt 01 was launched in detached tmux session
`sir_ukf_screen_20260730`. Post-launch verification observed live Python PID
`659746`, phase `recipe_screen`, and `98%` GPU utilization. The initial manifest
hook had landed in the broad-grid path rather than the screen path; the source
was corrected without interrupting the loaded process, and an explicitly
post-launch launch snapshot was written from the verified process, run-state,
tmux, and GPU evidence. The running process will overwrite it with the normal
terminal manifest when the screen completes.

Because the authorized launch used the existing dirty academic worktree, a
`source_provenance.json` companion records hashes for the exact loaded master,
trainer, target, registry, nonlinear backend, and CLI sources. The loaded master
and current master differ only by relocation of the nonnumeric manifest call;
the exact diff and both hashes are recorded. Each completed training checkpoint
also records the repository-issued target callable and dependency-closure
hashes.

## Artifact Root

```text
docs/plans/artifacts/bayesfilter-sir-ukf-neutra-target-specific-20260730/
  engineering-preflight-attempt-01/
  throughput-attempt-01/
  throughput-attempt-02/
  screen-attempt-01/
  final-seed-01-attempt-01/
  final-seed-02-attempt-01/
  broad-grid-seed-01-attempt-01/
  broad-grid-seed-02-attempt-01/
  retained-validation-*/
  plain-hmc-attempt-01/
```
