# FAB plus canonical IAF training campaign

Date: 2026-09-26

This campaign asks one bounded question: does the source-anchored FAB
objective, applied to the canonical Hoffman IAF, produce a better q20/T30
transport than the same IAF trained with matched reverse KL, especially in
whitening geometry and target-region coverage?

The result is a training diagnostic. It cannot establish posterior correctness,
HMC readiness, exhaustive mode discovery, or a new default from a short run.

## Research intent ledger

| Field | Declaration |
|---|---|
| Main question | Does FAB plus the canonical IAF improve whitening and target support relative to matched reverse-KL training? |
| Candidate | TensorFlow port of pinned `fab-jax` revision `c9f9913`, alpha=2 AIS and prioritized replay, using the canonical `NeuTraTransport`. |
| Exact comparator | The same canonical `NeuTraTransport` configuration and initial parameter state, updated by `NeuTraTransportTrainer` standard reverse KL. |
| Expected failure | FAB replay weights remain concentrated, the auxiliary alpha-2 target has poor tails, the map saturates its conditional scale, or short training leaves both methods untrained. |
| Primary promotion screen | For a replicate, all finite/valid checks pass, the standard 1,000-point probe is complete, and the FAB arm has a lower score-residual RMS and lower centered log-density RMS than its paired reverse-KL arm. |
| Statistical interpretation | Three paired seeds provide descriptive evidence only. No superiority claim is allowed without a predeclared uncertainty analysis. |
| Coverage diagnostic | Independent 4,096-point target-weight evaluation reports effective sample size, maximum normalized weight, sign-region mass for `theta[2]`, and (on the analytic three-mode control) nearest-mode occupancy against exact component weights. |
| Promotion veto | Nonfinite state/target/gradient, invalid target status, failed checkpoint/hash, incomplete 1,000-point probe, missing optimizer updates, or an unrecorded configuration mismatch. |
| Continuation veto | The target or architecture changes, a required artifact cannot be written, GPU memory policy is not verified before TensorFlow initialization, or the declared compute cap is exhausted. |
| Repair trigger | Any implementation or harness failure is repaired and rerun in a fresh output directory under the same scientific contract; a candidate failure proceeds to the declared diagnostic interpretation. |
| Must not conclude | A short paired run does not prove convergence, posterior correctness, exhaustive mode discovery, or that FAB is better for other targets. |

## Default and assumption audit

| Choice | Provenance and status | Why used here | Earliest check and failure meaning |
|---|---|---|---|
| Canonical architecture | Repository owner policy and `NeuTraTransportConfig.hoffman_author_iaf` | Keeps the architecture fixed while changing only the training objective | Config identity and FAB constructor fail closed on drift |
| Dimension and target | Existing q20/T30 UKF bridge | This is the target for which the FAB port was prepared | Target and adapter signatures must match the manifest |
| Map dtype `float32`, target dtype `float64` | Existing FAB harness; FP32/no-TF32 parity passed | Tests the intended GPU transport with high-precision target evaluation | FP64 post-training probe and finite checks expose precision failure |
| TF32 disabled | FP32/no-TF32 equivalence passed; TF32 screen was not qualified | Avoids promoting a numerically unqualified parity mode | Manifest records the setting; any accidental TF32 run is excluded |
| Batch size 32 | Existing FAB calibration and canonical batched-training minimum | Matches both arms and preserves batch-native target evaluation | Batch shape and target backend are recorded |
| Adam `(0.9, 0.999, 1e-8)`, learning rate `1e-3` | Existing canonical/FAB configuration; treated as a hypothesis | Matched optimizer scale isolates the objective | Loss, gradient norm, clipping and update counters are recorded |
| No gradient clipping in both arms | FAB port default and direct objective comparison | Avoids conflating objective effects with the earlier clipping pathology | Nonfinite update is a hard veto; no clipping is silently enabled |
| Three seeds | Target-specific small replication ladder | Allows paired descriptive variability within the bounded budget | Seed-specific manifests and paired tables are required |
| FAB temperatures 10, one mutation per temperature, Metropolis proposal scale 0.3 | Prior q20 calibration showed this is the low-cost valid mutation; source implementation supports it | Makes replay initialization affordable before a later HMC FAB campaign | Acceptance, ESS and movement are explanatory; they do not promote a map |
| Replay capacity 4,096, minimum 1,280, four updates per pass | Source ordering with a bounded smaller buffer | Gives 41 initialization passes and then four updates per fresh pass | Replay size and optimizer update count must agree exactly |
| 101 total FAB passes (41 replay-fill passes plus 60 four-update passes = 240 updates); 240 reverse-KL updates | Bounded exploratory ladder, not a convergence default | Equalizes optimizer updates while charging replay fill separately | Stop at cap; do not call the result converged |
| 1,000-point whitening probe | Repository standard post-training procedure | Measures the declared residual on the exact exported map | Incomplete or invalid probe vetoes the arm |
| 4,096 independent coverage rows | Fixed diagnostic sample size | Resolves coarse region support and importance concentration cheaply | Descriptive only; no mode claim without an analytic reference |

## Execution phases

1. **Integration and source checks.** Merge the FAB branch into remote `main`,
   preserve the dirty shared checkout, run the focused FAB/JAX-equivalence
   tests, and record the merge commit.
2. **Harness preflight.** Add the paired campaign runner. Run a CPU-hidden
   shape/configuration smoke and a one-update GPU smoke before any ladder. The
   runner must set memory growth before importing TensorFlow, disable TF32,
   write a fresh versioned output directory, and stop on nonfinite values.
3. **Analytic three-mode control.** Train the same canonical IAF with FAB and
   matched reverse KL on the existing exact three-mode Gaussian-mixture target
   for a short, cheap control. Report nearest-mode occupancy and exact mixture
   mass error. This tests whether the coverage diagnostic can distinguish
   objective failure from q20-specific target difficulty.
4. **Paired q20 ladder.** Run three paired seeds, with one arm per process and
   one GPU per process where devices are available. Preserve every checkpoint,
   probe, coverage result, manifest, and source hash. If only two GPUs are
   available, queue the third seed without changing the contract.
5. **Independent diagnosis.** Re-evaluate initial and final maps with fresh
   target draws. Compute whitening residual summaries, ESS, max weight, sign
   region mass, and scale-saturation diagnostics. The analytic control gets the
   exact component-mass comparison; q20 gets only the declared descriptive
   region evidence.
6. **Result review.** Write a decision table separating hard vetoes, viable
   arms, descriptive differences, uncertainty, and the next justified action.
   A FAB failure rejects this bounded candidate or triggers a stated repair; it
   does not reject the FAB research direction.

## Compute and artifacts

The ladder cap is 3 seeds × 2 arms. The hard wall budget is 14,400 worker
seconds (4 hours) plus at most 1,800 seconds of setup/diagnostics. A worker
must stop before its own 2,400-second cap and preserve its latest complete
checkpoint. The analytic control is charged separately to a 600-second CPU/GPU
smoke allowance. Every attempt uses a fresh directory under
`docs/plans/artifacts/neutra-fab-iaf-training-2026-09-26/`.

Required artifacts per arm are `manifest.json`, `progress.json`,
`checkpoint.json`, `post-training-1000.json`, `coverage.json`, `frozen-map.json`,
and `result.json`. The terminal report records the exact command, git commit,
environment, GPU, memory policy, TF32/XLA settings, seed, wall time and hashes.

## Skeptical plan review

Review completed before execution on 2026-09-26.

- **Baseline risk:** the comparator uses the same configured IAF and copied
  initial parameter state; it is not the historical legacy trainer.
- **Proxy risk:** loss, acceptance, ESS and sign-region counts are diagnostic;
  only the finite 1,000-point probe and declared paired geometry screen can
  nominate a viable training result, and neither proves posterior correctness.
- **Coverage risk:** q20 has no repository-issued exhaustive mode labels. The
  q20 region statistic is therefore descriptive. Exact mode occupancy is
  required only on the analytic three-mode control.
- **Stopping risk:** update/pass caps are explicit and cannot be converted into
  plateau evidence. A run stopping at its cap is reported as capped.
- **Fairness risk:** both arms use the same seed, architecture, batch size,
  Adam constants, update count, dtype and TF32 policy. FAB's replay fill is
  reported separately rather than counted as an optimizer update.
- **Environment risk:** GPU memory growth and device provenance are checked
  before TensorFlow initialization. CPU-only tests are mechanics checks only.
- **Artifact risk:** output roots are fresh and hashes are recorded; no prior
  failed checkpoint is reused as a map handoff.

The audit passes. Execution may proceed under this bounded contract.

## Recovery audit and diagnostic repair, before scientific execution

The resumed audit **revised** the harness before any completed scientific arm.
The first preflight failed at JSON serialization after 15.960 worker seconds;
its artifacts remain debugging evidence and its map will not be reused.
Commit `d7194ce40` repairs that boundary (16 focused FAB tests passed).

Three additional errors would have made the original campaign misleading:

1. The runner selected dimension-wide conditioners (width 4), whereas the
   owner's q20 instance requires `(16,16)`. Both targets and both arms now use
   three stages and width 16 with the same initial state within a pair.
2. A draw `x ~ g` from the independent Gaussian/prior bank was assigned `p/q`
   weights. This is wrong: expectations from that bank require `p/g` weights.
   The repaired diagnostic records `p/g`, while `log p-log q` on those points
   is retained only as a support-mismatch diagnostic. Raw learned-map occupancy
   and exact-target responsibility masses are also required; importance
   reweighting alone can hide a map that allocates too little mass to a mode.
3. Initial geometry, finite coverage weights, exact mixture references and
   replay clipping/movement diagnostics were missing. They are now recorded.
   Coverage evaluation uses stable batches of 32, avoiding an unpriced
   4,096-row target compilation. This is a batch loop, not scalar target calls.

Analytic verification uses the identity E_p[r_k(X)] = mixture_probability_k
for target component responsibilities r_k. It reports raw proposal average
responsibilities, nearest-mean occupancy, weighted responsibilities and
absolute errors against the exact component weights. Nearest-mean cells are
not equated with mixture labels. Exact independent mixture draws provide a
reference cell occupancy and target-to-map log-density mismatch. For q20 the
independent reference remains prior importance sampling: low reference ESS
makes region coverage unresolved, regardless of a finite whitening probe.
The two arms share each diagnostic bank but banks are independent of training.

The declared Adam hyperparameters match, but FAB uses Optax-form Adam and the
existing reverse-KL trainer uses Keras Adam (epsilon placement differs). This
is a comparison of existing trainers, not perfect isolation of the objective.
The standard reverse-KL estimator and 240 updates are exploratory comparators,
not the previously calibrated path-gradient/4,096-update training recipe.
No superiority over that recipe, optimal hyperparameters, converged training,
or complete investigation of the FAB idea can follow from this short ladder.
Undertraining and concentrated AIS/replay weights trigger further training or
exploration design, rather than a negative verdict on FAB itself.

Pricing also shows the original 2,400-second per-worker cap may not fit 101
FAB passes (the earlier 26.52 seconds/pass already implies 2,679 seconds,
before diagnostics). Preserve the total 16,200 worker-second allowance but
allocate at most 3,600 seconds per FAB arm and 1,800 per reverse-KL arm,
including their diagnostics: 3*(3,600+1,800)=16,200. Workers reserve measured
diagnostic time and have an external timeout. Budget stops are incomplete
pairs, never matched-update evidence. Smoke/control allowance remains 600
worker seconds including the failed preflight. Analytic controls use 240
updates per arm; the one-update preflights use replay 128/min32/one update,
solely to exercise the entire artifact chain. Scientific arms use the original
4096/min1280/four-update settings. Seeds are explicitly 0, 1 and 2.

Exact launch form (absolute tfgpu interpreter; trusted GPU execution):

```sh
TF_FORCE_GPU_ALLOW_GROWTH=true timeout --signal=TERM --kill-after=10s LIMITs \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/run_fab_iaf_training_campaign_2026_09_26.py \
  --arm ARM --target TARGET --gpu GPU --seed SEED --updates UPDATES \
  --worker-seconds LIMIT --output FRESH_VERSIONED_DIRECTORY
```

Skeptical re-review: the repairs correct the sampling measure and canonical
configuration without changing the target or FAB authority. Initial/final
diagnostics distinguish undertraining from invalid implementation; explicit
raw occupancy prevents importance correction from masquerading as map fit.
The analytic reference, untrained map and independent Gaussian/prior proposal
are basic sanity comparators for the learned arms. They do not authorize a
new default. The bounded ladder may proceed; exhausted budget or failed
coverage yields an unresolved training-quality verdict, not promotion.

Measured scheduling follow-up: the initial q20 probe costs about 105--112
steady seconds for 50 native batches, so per-row extrapolation reserves
1,487--1,582 seconds. This is not a measurement of the 64-row coverage graph.
Before further launches, price that exact graph for three calls (one compile,
two steady calls) on an available GPU, without optimizer updates. This is a
debugging-only timing check, capped at 120 seconds and charged to the 16,200
total; it cannot support a fit or performance-ranking claim. Use its measured
batch cost plus a recorded margin for the remaining workers. Existing valid
prefixes are preserved, and any necessary continuation uses exact checkpoint
restore, the same target/configuration and remaining total budget.

The timing check completed in 30.022 seconds. Its compile-inclusive call cost
18.211 seconds; steady 64-row calls cost 5.204 and 5.089 seconds. New workers
reserve `1.1*(measured_1000_probe_steady + 128*5.204)+40` seconds: 10% is an
explicit scheduling margin, and 40 seconds is a compile allowance, neither a
scientific threshold. The first two workers had already loaded the larger
reserve; preserve their results as bounded prefixes if they stop early.
`--resume-from` validates initial state, target, arm, seed, optimizer config and
checkpoint hash, and resumes the optimizer/replay exactly in a fresh directory.
Only the unchanged initial probe is reused; final diagnostics are rerun on the
continued map. Continuation is charged to the same total allocation.
