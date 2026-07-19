# SSL-LSTM NeuTra Optuna And Plateau Training Repair Plan

Date: 2026-07-15

Status: `TRIAL0_GH_CONFIRMATION_PASSED_PHASE4_COMPLETE`

Execution reset, 2026-07-16:

- the frozen C/D confirmation completed with C vetoed and D passed; one pass
  does not promote the procedure;
- C's exported loss-selected step-100 checkpoint failed moderate-shell support
  (`5.527918 > 4.30`), while the read-only audit found that C checkpoints at
  steps 300, 600, 800, and 1,100 passed the same support screen;
- this supports a checkpoint-selection mismatch, not retrospective promotion
  of C or rejection of its whole training trajectory;
- the prospective repair makes finite state, saturation, roundtrip, and
  moderate-shell support necessary for checkpoint eligibility. The first
  eligible checkpoint initializes patience; thereafter only a meaningful loss
  improvement among eligible checkpoints resets patience or replaces best;
- C/D are repair-design evidence and cannot be reused as confirmation. Untouched
  E/F namespaces are frozen in the repaired policy; and
- HMC remains closed unless both repaired E/F runs pass.

E/F result update, 2026-07-16:

- the repaired E/F confirmation completed in `9,102.96` seconds (`2.5286`
  GPU-hours), below its `17,500`-second cap;
- E passed after plateau/LR repair, exporting its support-eligible step-600
  best and stopping at step 1,600;
- F's support-aware selector worked: step 100 was rejected for missing support,
  and later eligible improvements advanced best through step 700;
- F was nevertheless vetoed when saturation rose from `0.01432` at step 700
  to `0.02865`, `0.04948`, and `0.05339` at steps 800-1,000. This crossed
  the frozen `0.05` cap before plateau patience would reduce LR at step 1,200;
- therefore trial 2 is not confirmed. The result supports optimizer-policy
  instability on F, not a harness, selector, target, XLA, artifact, or general
  NeuTra-direction failure; and
- the predeclared trial-0 alternative is the next discriminating candidate.
  It tests configuration viability rather than a pure LR effect because it
  changes LR from `0.00189968` to `0.00112196` and initialization scale from
  `0.01` to `0.02`.

G/H resource update, 2026-07-16:

- trial-0 Fresh G passed with support-eligible best step 1,000, one LR reduction
  at step 1,500, plateau stop at step 2,000, and no vetoes;
- Fresh H reached optimizer step 446 before the `8,500`-second cumulative cap.
  Its latest completed step-400 validation is eligible: moderate-shell radius
  `3.87146`, saturation `0.01042`, roundtrip `8.88e-15`, and meaningful
  heldout improvement;
- H's atomic joint checkpoint validates and binds trainer step 446 to controller
  best step 400. The resource stop is not a candidate veto and has no scientific
  interpretation;
- total charged/conservatively counted program use is `35,905.30` seconds
  (`9.9737` GPU-hours), leaving only `94.70` seconds inside the authorized
  10-hour envelope; and
- execution stops at the resource boundary. A resume requires explicit
  authorization of a larger cumulative cap. The resume may change only the
  resource cap and runner resume mechanics; trainer/controller/target/policy
  numerical hashes must remain identical to the partial run.

Execution update, 2026-07-15:

- the distinct tuned `(32,32)` family, mutable serialized effective learning
  rate, deterministic plateau controller, Optuna harness, and fresh two-seed
  confirmation runner are implemented;
- `60` focused CPU-hidden tests pass, including immutable comparator
  regressions, compiled-update LR mutation, exact trainer/controller resume,
  interruption receipts, historical/fresh seed separation, and best-state
  export;
- a one-step trusted GPU/XLA smoke on physical GPU 1 completed in `302.79`
  seconds, including compile, real locked-target update, validation,
  freeze/reload, and support probes; its result SHA-256 is
  `6d9f06f70515ed71d83866117d0be62be480eaf1143b5b09755881f8f37a79ef`;
- an earlier 300-second smoke on contended GPU 0 ended by resource stop and is
  not candidate or timing evidence; the other lane remains untouched; and
- the user instruction to execute this plan, read with the previously discussed
  10 GPU-hour envelope, authorizes the bounded tuning study below. It does not
  authorize spending beyond that envelope or launching HMC.

## 1. Purpose And Boundary

Modify the BayesFilter SSL-LSTM NeuTra training lane so that:

1. Optuna nominates a small set of static optimizer hyperparameters for the
   fixed three-stage `(32,32)` IAF architecture; and
2. final training uses a serializable validation-plateau controller that halves
   the learning rate after one patience period without meaningful improvement
   and stops after a second such period, or at the maximum step/resource limit.

This is a BayesFilter capacity-plus-tuning adaptation. It is not the exact
Rotemberg/SGU `(4,4)` source procedure. The strict
`dsge_paper_dense_iaf` comparator and the completed
`ssl_lstm_capacity_dense_iaf` capacity-only comparator remain immutable.

This plan scopes local implementation and focused CPU-hidden smoke tests. No
implementation or execution has started. It does **not** authorize the material
Optuna GPU study, fresh-seed confirmation, HMC, predictive validation, a default
change, or a scientific claim. Obtain a separate resource authorization after
implementation timing is available.

## 2. Current Evidence

- The strict `(4,4)` source-procedure family passed direct `dsge_hmc`
  forward/logdet/gradient/update parity and artifact tests, but its two material
  seeds were unstable under the inherited schedule.
- The separate `(32,32)` family has 4,440 trainable parameters and passed 62
  focused engineering tests.
- Historical streams A and B both exceeded the `0.05` scale-saturation screen
  at step 100 under initial learning rate `0.01`: `0.21745` and `0.11719`.
- Both failures were finite, invertible, GPU/XLA-valid training outcomes. They
  reject width expansion under the original schedule, not `(32,32)` capacity or
  NeuTra generally.
- The measured cost was about `612` seconds per 100-step stream. A naive
  20-trial, two-stream study at only 100 steps would cost about `6.8` GPU-hours,
  before confirmation. Multi-fidelity pruning and an explicit wall-time cap are
  therefore required.
- Optuna `4.6.0` is installed in the `tfgpu` environment. The sibling
  `dsge_hmc/scripts/tune_neutra_frozen_rkl_schedule_optuna.py` is a design
  reference only: its runner, target, vetoes, score, and post-run pruning are
  not the SSL-LSTM contract and must not be imported as execution authority.

## 3. Research Intent Ledger And Evidence Contract

| Field | Prospective contract |
| --- | --- |
| Main question | Can a lower, tuned initial learning rate plus a plateau repair controller train the fixed `(32,32)` SSL-LSTM IAF without the rapid saturation observed at `0.01`, and do independently initialized confirmation runs remain admissible under the existing transport screens? |
| Exact baseline | Completed `(32,32)` capacity-only historical A/B diagnostic under `0.01 -> 0.001 -> 0.0001`; program result SHA-256 `5ae83bc90faf7463a5b74437cdaf904aa54112a8a9945fc2b8ebddc994b47a00`. The strict `(4,4)` source procedure remains a historical operator/procedure comparator, not the tuning baseline. |
| Candidate mechanism | Fixed three-stage `(32,32)` ELU autoregressive IAF with the existing masks, reverse mixing, fixed translation, reverse-KL objective, batch 480, `s_max=1`, Adam, per-variable clipping, and GPU/XLA `float64`; tune only the declared optimizer/initialization fields, then use plateau LR halving in final training. |
| Tuning data | Exact historical A/B initialization, training, and tuning-validation streams. These streams may nominate hyperparameters and calibrate patience; they cannot confirm the selected procedure. |
| Confirmation data | Two prospectively fresh, independent initialization/training/validation seed namespaces not inspected during tuning or patience calibration. |
| Primary Optuna nomination criterion | At least one trial survives all hard and nomination vetoes on both historical streams at a common terminal rung. Among survivors, Optuna's scalar objective nominates trials for fresh confirmation only; it does not establish superiority or correctness. |
| Primary confirmation criterion | Both fresh runs complete under the frozen selected hyperparameters and frozen plateau policy; remain finite; never exceed the prospective saturation cap; pass terminal support and paired heldout-improvement screens; and produce valid, exactly reloadable best-checkpoint artifacts. |
| Hard veto | Target/chart/topology/objective drift; mutation of either comparator family; nonfinite state; invalid target support; corrupt/mismatched artifact; resume divergence; controller-state mismatch; hidden tuning on confirmation seeds; CPU fallback; missing GPU/XLA evidence; or resource overrun. |
| Optuna nomination veto | Saturation above `0.05` at a checked rung; invalid moderate-shell support at the terminal rung; failure to establish terminal paired heldout improvement; or failure on either historical stream. A veto prunes/rejects the trial but does not stop the study unless the shared continuation veto fires. |
| Confirmation veto | Either fresh run fails an existing material transport gate, or the selected controller/hyperparameters are changed after seeing a fresh result. |
| Continuation veto | Invalid target/harness/math, inability to reproduce resume semantics, corrupted seed separation, missing required diagnostics, unavailable trusted GPU/XLA route, or exhausted authorized GPU/wall-time budget. A candidate trial failure is not a continuation veto. |
| Repair trigger | No Optuna survivor; all trials saturate; patience cannot be calibrated; excessive validation noise; inconsistent historical-stream behavior; or one/both fresh runs fail without an evidence-validity failure. Each triggers a new prospective candidate, not retrospective threshold changes. |
| Explanatory only | Raw training loss, gradient/clipping rates, loss trajectories, LR trajectory, trial ranking, stage-specific saturation, shell/tail radii, runtime, and descriptive A/B or fresh-seed differences. |
| Must not conclude | Posterior correctness, complete support/mode coverage, HMC readiness, NeuTra superiority, architecture optimality, predictive validity, scientific validity, dimensional scalability, or default/production readiness. |
| Result artifact | One structured Optuna study artifact plus one result note; if authorized later, one fresh-confirmation artifact plus an updated result note and run manifests. |

### Diagnostic Roles

| Diagnostic | Role |
| --- | --- |
| Nonfinite values, source/target mismatch, invalid artifact, resume mismatch, missing GPU/XLA, resource overrun | Hard and continuation veto as specified above |
| Scale saturation above `0.05` | Trial/confirmation promotion veto; repeated saturation may trigger a new mechanism but is not by itself a research-direction veto |
| Moderate-shell support and paired heldout improvement | Terminal nomination/promotion veto |
| Heldout reverse-KL scalar objective after all vetoes | Optuna nomination proxy only |
| Training loss, gradient norm, clipping, per-stage scale, runtime | Explanatory and repair diagnostics |
| HMC/predictive moments | Out of scope; later downstream evidence |

## 4. Frozen Design And Tunable Search Space

### 4.1 Frozen For The First Study

- dimension `4` and locked SSL-LSTM target/signatures;
- three autoregressive IAF stages;
- hidden layers `(32,32)` with ELU;
- existing degree masks, reverse-coordinate mixing, and prior-center fixed
  translation;
- standard-normal base and reverse-KL mean objective;
- batch size `480`, `s_max=1`, Adam `beta1=0.9`, `beta2=0.999`, and
  `epsilon=1e-7`;
- per-variable gradient clipping mode;
- `float64`, trusted GPU, XLA JIT on, TF32 state recorded; and
- fixed maximum of 5,000 steps for eventual confirmation.

Do not tune width, stage count, activation, batch size, `s_max`, objective,
Adam betas/epsilon, validation data, or promotion thresholds in the first
study. A no-survivor result may justify a new prospective study, but must not
expand this search retrospectively.

### 4.2 Initial Optuna Search Space

| Field | Search |
| --- | --- |
| Initial learning rate | log-uniform `[1e-4, 2e-3]` |
| Initialization scale | categorical `{0.005, 0.01, 0.02}` |
| Per-variable clip norm | categorical `{5.0, 10.0}` |

The sampler seed, pruner, trial count, rung steps, stream order, timeout, and
storage path must be recorded. Use persistent SQLite storage so an interrupted
study can resume without repeating completed trials.

Optuna selects static hyperparameters. It must not choose the final maximum
step from short trials, tune on fresh confirmation seeds, or treat pruning as
evidence of posterior invalidity.

## 5. Plateau Repair Contract

### 5.1 Monitored Statistic

At every validation check, evaluate the same immutable heldout base-noise batch
for that run. Compare the current per-sample reverse-KL vector with the vector
stored at the best admissible checkpoint using a paired one-sided 95% upper
confidence bound for `current - best`.

A checkpoint is a meaningful improvement only when:

```text
one_sided_95_upper(current - best) < -absolute_min_delta
```

Set `absolute_min_delta` prospectively after examining only historical tuning
streams. Prefer `0.0` if the paired interval is stable; otherwise calibrate a
nonzero threshold from repeated no-update validation evaluations. Record the
calibration and do not change it on confirmation data. Training minibatch loss
must not control plateau decisions.

Every checked candidate must also pass the finite and scale-saturation screens.
An invalid or saturated checkpoint cannot become `best` even when its loss is
lower.

### 5.2 Patience `n`

Define `n` in optimizer steps, not wall time:

```text
n = validation_check_every * plateau_patience_checks
```

Provisional values are `validation_check_every=100` and
`plateau_patience_checks=5`, giving `n=500` steps. Calibrate once using only
surviving historical Optuna histories:

1. collect gaps between meaningful best-checkpoint improvements at the common
   tuning rung;
2. choose the smallest multiple of the validation interval that is at least
   the historical 90th percentile gap, constrained to `[500, 1000]` steps;
3. if there are too few meaningful improvements to estimate a gap distribution,
   keep the conservative provisional `n=500` and record the uncertainty; and
4. freeze `n`, the validation interval, confidence rule, and minimum delta
   before fresh confirmation.

This calibration is problem-specific, not an architectural claim. It must not
be recomputed separately for favorable and unfavorable confirmation runs.

### 5.3 State Machine

Let `steps_since_best` count completed optimizer steps since the last meaningful
admissible improvement.

1. Start with the Optuna-selected initial LR and `lr_reductions=0`.
2. On meaningful admissible improvement, store an exact best trainer state,
   reset `steps_since_best=0`, and retain the current LR.
3. At the first validation check where `steps_since_best >= n` and no reduction
   has occurred for this plateau, set `LR := 0.5 * LR`, increment
   `lr_reductions`, and continue from the **current** model and Adam state.
4. Do not reset `steps_since_best` merely because LR was reduced. A later
   meaningful improvement resets it and permits one future reduction for the
   new plateau.
5. If no meaningful improvement occurs and `steps_since_best >= 2n` after the
   reduction, terminate with `plateau_after_lr_repair`.
6. If the maximum step or resource cap is reached first, terminate with the
   corresponding explicit reason.
7. Freeze/export the best admissible checkpoint, not automatically the final
   in-memory state.

Use a prospective LR floor of `initial_lr / 16`. If the requested 50% reduction
would cross the floor, record `minimum_learning_rate_reached` and terminate at
the next applicable plateau stop rather than silently changing the factor.

The phrase "two periods without improvement" means `2n` consecutive steps
since the last meaningful improvement, with exactly one LR reduction at `n`.
This interpretation must be tested explicitly.

### 5.4 Resume Contract

Checkpoint and restore, atomically with trainer state:

- current effective learning rate;
- LR reduction count and last reduction step;
- best validation statistic and full per-sample vector;
- best trainer-state hash and best step;
- steps since best;
- validation interval, patience, confidence level, and minimum delta;
- stop reason and maximum step/resource settings; and
- seed namespaces and next optimizer step.

A split run resumed immediately before a meaningful improvement, LR reduction,
and terminal plateau must match uninterrupted training in parameters, Adam
state, controller state, validation decisions, LR trajectory, and frozen best
artifact.

## 6. Implementation Phases

### Phase 1: Tuned Family And Mutable LR Surface

Objective: add a distinct tuned capacity family without weakening either
frozen comparator.

Artifacts:

- a new family/procedure label such as
  `ssl_lstm_tuned_capacity_dense_iaf` /
  `bayesfilter_ssl_lstm_tuned_capacity_32x32_neutra_v1`;
- config validation allowing only the declared tunable fields while preserving
  topology/target requirements; and
- an optimizer learning-rate variable or serializable schedule interface whose
  current value is authoritative inside XLA-compiled updates.

Required tests:

- strict `(4,4)` and capacity-only `(32,32)` configs/payloads remain byte- or
  value-equivalent under existing regression tests;
- selected LR/init scale/clip norm round-trip in config and frozen artifacts;
- effective LR before and after a 50% reduction is observed by the compiled
  update without retracing or resetting Adam moments; and
- invalid family/topology/schedule combinations fail closed.

### Phase 2: Plateau Controller

Objective: implement the state machine in Section 5 as a small BayesFilter-owned
component, separate from Optuna.

Artifacts:

- immutable controller config;
- mutable, hash-bound controller state;
- deterministic update/action records; and
- trainer-plus-controller checkpoint helpers or a runner-level atomic payload.

Required tests:

- meaningful improvement, insignificant decrease, regression, saturation, LR
  reduction, recovery after reduction, `2n` stop, max-step stop, and LR-floor
  behavior;
- one reduction per plateau and a new reduction only after a later meaningful
  improvement begins a new plateau;
- best admissible rather than terminal state is exported; and
- uninterrupted/resumed equivalence at all three state transitions.

### Phase 3: BayesFilter Optuna Harness

Objective: adapt the useful orchestration ideas from `dsge_hmc` without using
its target-specific scoring or delayed pruning.

Artifacts:

- BayesFilter runner under `docs/benchmarks`;
- persistent Optuna study and structured JSON summary;
- per-trial/per-stream immutable checkpoint artifacts; and
- resource ledger with actual charged GPU time.

Multi-fidelity order:

1. compile/XLA warmup and a tiny finite/invertibility smoke;
2. historical stream A at steps `50`, `100`, `200`, and `400`;
3. call `trial.report()` at every rung and prune on hard/nomination vetoes or
   the prospective Optuna pruner;
4. run historical stream B only for stream-A survivors at the same rungs; and
5. extend only joint survivors to a common terminal nomination rung chosen in
   the execution note from timing, with identical total steps for every trial
   included in the scalar comparison.

Use the worst-stream paired heldout-loss statistic as the scalar objective only
after both streams pass all vetoes. Do not compare trials stopped at different
rungs as if their raw objectives were commensurate. Record all pruned trials
and reasons.

### Phase 4: Historical Study And Policy Freeze

Objective: nominate a viable hyperparameter region and freeze plateau policy
using tuning streams only.

Prospective execution defaults, subject to a separately authorized cap:

- TPE sampler with a fixed seed;
- a median or successive-halving pruner with explicit warmup rungs;
- 6 initial trials, then at most 6 additional trials only if budget remains and
  the study has fewer than two joint survivors;
- stream A before B; and
- immediate stop on the shared wall-time/GPU-hour cap.

If multiple trials survive, do not claim a statistically supported ranking from
one paired A/B observation. Select a representative trial using the declared
Optuna scalar objective, while reporting viable alternatives and descriptive
uncertainty. If no trial survives, write a negative result and propose the next
smallest discriminating search; do not weaken vetoes after seeing results.

Freeze selected static hyperparameters, `n`, validation interval, confidence
rule, minimum delta, LR factor/floor, maximum steps, and confirmation seeds in a
policy artifact before Phase 5.

### Phase 5: Fresh Two-Seed Confirmation

Objective: test the frozen selected procedure on independent seed namespaces.

- Run both fresh seeds with identical selected hyperparameters and plateau
  policy.
- Use sequential stopping for hard/confirmation vetoes and the shared resource
  cap, not for favorable retrospective selection.
- Preserve terminal and best-admissible states distinctly.
- Verify exact reload/inverse/logdet/target binding for each best artifact.
- Require both runs to pass; one passing run cannot promote the procedure.

If one run fails validly, classify whether this is tuning instability,
controller failure, or evidence against the candidate. Do not reject the whole
NeuTra direction unless a continuation veto fires.

### Phase 6: Handoff

Only after Phase 5 passes may a separate plan bind the two frozen transports to
exact transformed HMC. HMC and predictive validation retain their existing
independent admission gates. Training loss or Optuna selection never substitutes
for those gates.

### Phase 5R: Support-Eligible Checkpoint Repair

Research question: does the frozen trial-2 training policy confirm on two
untouched streams when export candidates are prospectively restricted to states
that already pass the declared transport gates?

Exact comparator: unchanged trial-2 hyperparameters and plateau schedule from
the parent policy. The only candidate-mechanism change is support-aware
checkpoint eligibility; C/D are explanatory repair evidence, not comparators or
confirmation runs.

Primary pass: both E and F complete and export a checkpoint that is finite,
below the saturation cap, below the roundtrip threshold, below the
moderate-shell radius threshold, and meaningfully improves paired heldout loss
over the run's initial validation state.

Promotion vetoes: either fresh stream fails a transport gate, lacks an eligible
checkpoint, fails exact reload, or fails paired heldout improvement. A failed
stream rejects this repaired trial-2 procedure but does not by itself reject
NeuTra or authorize retrospective selection.

Continuation vetoes: target/harness/math invalidity, controller/resume mismatch,
seed overlap, corrupted artifacts, missing GPU/XLA evidence, or the shared
resource cap. Support-invalid intermediate checkpoints are candidate-state
rejections, not continuation vetoes.

Explanatory only: loss trajectory, time-to-first-eligible checkpoint, shell
radius trajectory, saturation, LR-reduction timing, runtime, and E/F
differences. No continuous metric ranks E and F.

Nonclaims: no posterior correctness, complete support, HMC readiness,
superiority, predictive validity, default readiness, or scientific validity.

### Phase 5A: Predeclared Trial-0 Alternative

Research question: does the other historically viable Optuna configuration,
trial 0, pass the unchanged support-aware training confirmation on two untouched
streams after trial 2 failed one fresh stream by late saturation?

Exact baseline/comparator: failed trial-2 E/F confirmation. Trial 0 changes only
the already tuned static fields to LR `0.0011219623709077644`, initialization
scale `0.02`, and clip norm `5.0`; architecture, objective, batch, controller,
support thresholds, plateau schedule, runtime path, and two-run pass rule remain
unchanged.

Primary pass: both untouched G/H runs complete, never cross a frozen transport
veto, establish paired heldout improvement, and export exactly reloadable
support-eligible best checkpoints.

Promotion vetoes: any saturation above `0.05`, no eligible checkpoint, support
or roundtrip failure, absent heldout improvement, reload mismatch, or one fresh
run failing. A trial-0 veto rejects this alternative configuration but does not
alone reject NeuTra.

Continuation vetoes: target/harness/controller invalidity, source or parent
trial identity drift, seed overlap with A-F, corrupted artifacts, unavailable
trusted GPU/XLA, or the prospectively authorized shared cap.

Explanatory only: loss, saturation trajectory, support radii, LR timing,
runtime, and descriptive trial-0/trial-2 differences. Because both LR and
initialization scale change, the experiment must not attribute any outcome to
LR alone.

Nonclaims: no ranking, posterior correctness, support completeness, HMC
readiness, predictive validity, default readiness, or scientific validity.

## 7. Resource And Command Boundary

Implementation and tests use the smallest CPU-hidden commands, with
`CUDA_VISIBLE_DEVICES=-1` set before TensorFlow import. A tiny trusted GPU/XLA
smoke may be requested after local tests, but cannot support training-quality
claims.

Before Phase 4, update this plan or write the execution note with:

- exact runner command and environment;
- trial/rung/pruner configuration;
- measured per-rung timing from the final implementation;
- an explicit GPU-hour and wall-time cap;
- cancellation and persistent-study resume behavior;
- output paths and confirmation seeds; and
- remaining budget reserved for Phase 5 rather than spending the entire cap on
  tuning.

GPU execution must follow the BayesFilter trusted managed-session GPU policy,
record physical/logical device, XLA HLO evidence, TF32 state, dtype, and the
trust basis `owner_designated_managed_session_visible_gpu_trusted`.

### Phase 4 Frozen Execution Note

| Field | Frozen value |
| --- | --- |
| Question | Does the bounded search contain at least one `(32,32)` static hyperparameter configuration that survives both historical streams through the common step-400 rung? |
| Device | Physical GPU 1 isolated with `CUDA_VISIBLE_DEVICES=1`; do not use or interrupt the other lane on physical GPU 0 |
| Environment | `/home/ubuntu/anaconda3/envs/tfgpu/bin/python`; TensorFlow/TFP GPU/XLA; `float64`; TF32 state recorded |
| Optuna | `4.6.0`; fixed-seed TPE; successive-halving pruner; persistent SQLite |
| Trials | Six maximum; no automatic additional six in this execution |
| Rungs | `50,100,200,400`; stream A before stream B; scalar comparison only after both reach step 400 |
| Shared cap | `14,400` wall/GPU seconds (`4.0` GPU-hours), including compilation, validation, freeze/reload, and probes |
| Cancellation | Stop at the cap with a structured resource receipt; preserve completed SQLite trials and per-rung checkpoints; do not classify the interrupted trial as a candidate veto |
| Output | `docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/optuna-plateau-repair-study-2026-07-15` |
| Exact command | `CUDA_VISIBLE_DEVICES=1 /home/ubuntu/anaconda3/envs/tfgpu/bin/python docs/benchmarks/run_ssl_lstm_neutra_optuna_tuning_2026_07_15.py --mode study --output-root docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/optuna-plateau-repair-study-2026-07-15 --rungs 50,100,200,400 --n-trials 6 --gpu-cap-seconds 14400 --timeout-seconds 14400 --sampler-seed 20260715 --study-name ssl_lstm_neutra_32x32_tuning_v1` |
| Promotion meaning | A joint survivor permits policy freeze and separately costed fresh confirmation only |
| Nonclaims | No statistically supported ranking, posterior correctness, HMC readiness, superiority, default readiness, or scientific validity |

The prior timing evidence and the historical 100-step capacity run imply a
worst-case six-trial/two-stream study near `5.2` GPU-hours if every trial
survives. The `4.0`-hour cap deliberately relies on early pruning and may stop
before six complete trials. Such a stop preserves evidence but cannot be called
a no-survivor scientific result unless the completed trials themselves support
that conclusion.

### Phase 5 Frozen Execution Note

| Field | Frozen value |
| --- | --- |
| Policy | `docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/optuna-plateau-repair-study-2026-07-15/frozen-tuning-policy.json`; SHA-256 `a73cfd1750f1cac369a76e5573decb3b4791c9120f0384dc1447ecec2d5e0195` |
| Representative configuration | Trial 2: LR `0.0018996814203826532`, init scale `0.01`, clip norm `5.0` |
| Plateau controller | Validate every 100 steps; paired one-sided 95% meaningful improvement; `n=500`; LR factor `0.5`; stop at `2n`; max 5,000; LR floor `initial/16`; export best admissible state |
| Fresh streams | C: init/train/validation `[7101]/[8101]/[8201]`; D: `[7102]/[8102]/[8202]`; exact seed pairs are stored in the policy and disjoint from historical A/B |
| Device | Physical GPU 1 isolated with `CUDA_VISIBLE_DEVICES=1`; physical GPU 0 remains outside this lane |
| Total confirmation cap | `27,000` seconds (`7.5` GPU-hours), cumulative across any resume attempts |
| Envelope accounting | Study `8,144.25 s` + successful trusted smoke `302.79 s` + incomplete trusted smoke at most `300 s` + confirmation cap `27,000 s` = at most `35,747.04 s`, below the previously discussed 10 GPU-hour (`36,000 s`) envelope |
| Sequential behavior | Run C then D; stop on the shared cap with an atomic trainer/controller/best checkpoint; completed streams are hash-verified and skipped on resume |
| Output | `docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/plateau-confirmation-2026-07-16` |
| Exact command | `CUDA_VISIBLE_DEVICES=1 /home/ubuntu/anaconda3/envs/tfgpu/bin/python docs/benchmarks/run_ssl_lstm_neutra_plateau_confirmation_2026_07_15.py --policy docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/optuna-plateau-repair-study-2026-07-15/frozen-tuning-policy.json --output-root docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/plateau-confirmation-2026-07-16 --gpu-cap-seconds 27000` |
| Resume command | Same command plus `--resume`; it restores the partial stream and subtracts prior charged time from the same total cap |
| Promotion meaning | Both fresh streams passing permits only a separate exact-transformed-HMC plan |
| Nonclaims | No posterior correctness, support completeness, HMC readiness, superiority, default readiness, predictive validity, or scientific validity |

Phase 5 skeptical audit verdict:
`FRESH_CONFIRMATION_EXECUTION_READY_WITH_CUMULATIVE_CAP`. Historical tuning
data do not enter fresh training or validation. The policy, seeds, thresholds,
controller, maximum steps, command, device, cap, output path, interruption
behavior, and nonclaims are frozen before launch.

### Phase 5R Frozen Execution Note

| Field | Frozen value |
| --- | --- |
| Parent policy | `docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/optuna-plateau-repair-study-2026-07-15/frozen-tuning-policy.json`; SHA-256 `a73cfd1750f1cac369a76e5573decb3b4791c9120f0384dc1447ecec2d5e0195` |
| Repair evidence | C support audit SHA-256 `7cc0a4a7fd3e29d29f42597601a8c2db96e411eb7436519126e2e13de1411998`; explanatory only |
| Repaired policy | `docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/support-eligible-policy-2026-07-16.json` |
| Frozen mechanism | Trial-2 hyperparameters and plateau schedule unchanged; checkpoint eligibility additionally requires finite probes, saturation `<=0.05`, roundtrip `<=1e-9`, and moderate-shell radius `<=4.30` |
| Patience semantics | First eligible state initializes best/patience; only meaningful heldout-loss improvement among eligible states resets patience; ineligible states never become best or reset patience |
| Untouched streams | E: init/train/validation `(20260716,7103)/(20260716,8103)/(20260716,8203)`; F uses role codes `7104/8104/8204`; all are disjoint from A-D |
| Device | Physical GPU 1 isolated with `CUDA_VISIBLE_DEVICES=1`; do not use or interrupt physical GPU 0 |
| Shared repair cap | `17,500` seconds (`4.8611` GPU-hours), cumulative across resumes; 60-second emergency-checkpoint margin |
| Budget basis | Prior charged study `8,144.25 s` + trusted one-step smoke `302.79 s` + bounded incomplete smoke `300 s` + C/D confirmation `9,513.14 s` + support audit `101.30 s` + repair cap `17,500 s` = `35,861.48 s`, below the discussed 10 GPU-hour envelope |
| Sequential behavior | Run E then F; stop only on hard/continuation veto or shared cap; completed result hashes are verified on resume |
| Output | `docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/support-eligible-confirmation-2026-07-16` |
| Exact command | `CUDA_VISIBLE_DEVICES=1 /home/ubuntu/anaconda3/envs/tfgpu/bin/python docs/benchmarks/run_ssl_lstm_neutra_support_eligible_confirmation_2026_07_16.py --policy docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/support-eligible-policy-2026-07-16.json --output-root docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/support-eligible-confirmation-2026-07-16 --gpu-cap-seconds 17500` |
| Resume | Same command plus `--resume`; cumulative charged time cannot exceed `17,500` seconds |
| Alternative | Trial 0 remains predeclared if the repaired trial-2 procedure is unstable; it requires a separate prospective plan and untouched seeds |

Phase 5R skeptical audit verdict:
`REPAIR_IMPLEMENTATION_REVIEW_PASSED_GPU_EXECUTION_READY`. The old plan's
wrong terminal-only support baseline is repaired, proxy loss cannot promote an
ineligible state, stop conditions and nonclaims are explicit, E/F are untouched,
and the artifact directly answers whether the prospective selector repairs the
observed mismatch. The focused suite passes `68` tests, compilation and diff
checks pass, and the bounded one-path Claude review returned `VERDICT: AGREE`.

### Phase 5A Frozen Execution Note

| Field | Frozen value |
| --- | --- |
| Candidate identity | Optuna trial 0 record SHA-256 `7a1022841220e3743d5af27efc0733c65506e10f2189fe3609d43f07ebc7dd63`; status `SURVIVED` on historical A/B |
| Candidate configuration | LR `0.0011219623709077644`; init scale `0.02`; clip norm `5.0`; all other trainer fields unchanged |
| Support policy | `docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/support-eligible-policy-2026-07-16.json`; SHA-256 `9f05aff2a333bdc5179408e397645b44b65c5dee5f6d8ad941cd4dbdde74285d` |
| Trial-0 policy | `docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/trial0-alternative-policy-2026-07-16.json` |
| Untouched streams | G: init/train/validation `(20260716,7105)/(20260716,8105)/(20260716,8205)`; H uses role codes `7106/8106/8206`; disjoint from A-F |
| Device | Physical GPU 1 isolated with `CUDA_VISIBLE_DEVICES=1`; physical GPU 0 remains outside this lane |
| Remaining-envelope cap | `8,500` cumulative seconds (`2.3611` GPU-hours); resource stop is resumable and is not a candidate veto |
| Budget accounting | Actual/conservatively charged prior work totals `27,464.44 s`; plus `8,500 s` = `35,964.44 s`, below the authorized 10 GPU-hour (`36,000 s`) envelope |
| Sequential behavior | Run G then H. A completed G is hash-verified and skipped on resume. A partial H writes atomic trainer/controller/best state before resource stop |
| Output | `docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/trial0-alternative-confirmation-2026-07-16` |
| Exact command | `CUDA_VISIBLE_DEVICES=1 /home/ubuntu/anaconda3/envs/tfgpu/bin/python docs/benchmarks/run_ssl_lstm_neutra_trial0_alternative_confirmation_2026_07_16.py --policy docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/trial0-alternative-policy-2026-07-16.json --output-root docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/trial0-alternative-confirmation-2026-07-16 --gpu-cap-seconds 8500` |
| Resume | Same command plus `--resume`; do not increase the cumulative cap without new user authorization |
| Promotion meaning | Both G/H passing permits only a separate exact transformed-target/HMC preflight plan |
| Nonclaims | No pure LR effect, ranking, posterior correctness, support completeness, HMC readiness, predictive validity, or default readiness |

Phase 5A skeptical audit verdict: `EXECUTION_READY_WITH_REMAINING_ENVELOPE`.
The baseline is the failed trial-2 E/F configuration; trial 0 was predeclared
before E/F; the test changes only already tuned static fields; proxy loss cannot
promote an invalid state; G/H are untouched; candidate and continuation vetoes
are separate; resource stop is resumable; and the artifact directly answers
configuration viability. The focused controller/confirmation suite passes `20`
tests, compilation passes, and `git diff --check` passes.

Historical interim status, superseded by the final result below:
`G_PASSED_H_RESOURCE_STOP`. At that point confirmation remained open, not
failed: H had no candidate veto and could resume from optimizer step 447. Based
on G's `6,682.76` seconds and H's first `446` steps, a new cumulative cap of
`15,500` seconds (an additional `7,000` seconds, `1.9444` GPU-hours) is the
recommended bounded extension. It provides a modest margin over a G-scale H
completion while preserving the existing atomic resource-stop behavior. The
exact resume command is the Phase 5A command with `--gpu-cap-seconds 15500
--resume`. No HMC or further candidate search is authorized by that extension.

Resume authorization, 2026-07-16: the owner authorized increasing the
cumulative trial-0 G/H cap from `8,500` to `15,500` seconds, allowing at most
`7,000` additional seconds solely to resume Fresh H from optimizer step 447.
The authorization does not permit HMC or another candidate search. Resume
preflight revalidated the joint checkpoint, confirmed trainer step 446 and
controller best step 400, passed `14` focused tests plus compilation/diff
checks, and found physical GPU 1 idle. Exact command:

```text
CUDA_VISIBLE_DEVICES=1 /home/ubuntu/anaconda3/envs/tfgpu/bin/python docs/benchmarks/run_ssl_lstm_neutra_trial0_alternative_confirmation_2026_07_16.py --policy docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/trial0-alternative-policy-2026-07-16.json --output-root docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/trial0-alternative-confirmation-2026-07-16 --gpu-cap-seconds 15500 --resume
```

The first authorized resume attempt stopped before training because the runner
compared persisted JSON seed lists with in-memory tuples. The checkpoint,
progress, and summary hashes were unchanged. Resume now compares a canonical
JSON stream payload, with a focused regression test, before relaunching from the
same step-446 checkpoint. This is a resume-serialization repair only and does
not change numerical training or the cumulative cap.

Final Phase 5A result, 2026-07-16:

- the resumed H trajectory completed under the cumulative `15,500`-second cap;
- Fresh G passed with best/terminal steps `1000/2000`, and Fresh H passed with
  best/terminal steps `900/1900`;
- both stopped by `plateau_after_lr_repair`, each had one LR reduction, and
  neither had a transport veto;
- G best diagnostics were radius `3.01857`, roundtrip `4.66e-15`, saturation
  `0.00911`, and paired heldout upper bound `-22.203`;
- H best diagnostics were radius `3.00731`, roundtrip `1.60e-14`, saturation
  `0.00651`, and paired heldout upper bound `-25.035`;
- cumulative trial-0 charge was `13,730.28` seconds (`3.81397` GPU-hours),
  below its `15,500`-second cap. Total program use was `41,194.72` seconds
  (`11.4430` GPU-hours), below the successively authorized total ceiling; and
- Phase 4 closes with trial 0 as a viable confirmed training procedure. This
  is not a ranking, posterior-correctness, support-completeness, HMC-readiness,
  predictive-validity, or default-readiness conclusion.

The next admissible phase is a separate exact transformed-target preflight
binding the immutable G/H best payloads. The current authorization does not
permit HMC execution, so no HMC mechanics canary, tuning, or retained sampling
is launched here.

## 8. Skeptical Pre-Execution Audit

| Audit risk | Finding and repair |
| --- | --- |
| Wrong baseline | Repaired: the exact `(32,32)` plus original schedule is the causal tuning baseline; `(4,4)` remains a historical source-procedure comparator. |
| Proxy promoted to correctness | Repaired: heldout reverse KL is nomination-only. Fresh transport gates, HMC, and predictive validation remain separate. |
| Tuning/confirmation leakage | Repaired: historical A/B streams tune and calibrate; prospectively fresh seeds confirm only after policy freeze. |
| Noisy plateau detection | Repaired: use paired per-sample validation uncertainty and meaningful improvement, not any scalar decrease. |
| Retrospective patience choice | Repaired: bound and freeze `n` using historical improvement gaps before fresh confirmation. |
| Unfair trial comparison | Repaired: scalar comparison includes only joint survivors at a common terminal rung; pruned-rung losses are not ranked together. |
| Expensive nominal Optuna study | Repaired: stream-A-first multi-fidelity rungs, real intermediate pruning, persistent storage, small initial study, and a separately authorized cap. |
| Rollback corrupts optimizer/sample lineage | Repaired: LR reduction continues from current Adam/model state; best state is restored only for final export. |
| Resume changes decisions | Addressed prospectively by serializing complete controller state and testing boundary resumes. |
| Too many simultaneous changes | Repaired: first search varies only LR, initialization scale, and clip norm; architecture/objective/batch/Adam constants remain frozen. |
| Static schedule conflicts with plateau repair | Repaired: introduce a distinctly labeled tuned family with mutable effective LR; do not weaken frozen comparator presets. |
| Max-step artifact mislabeled converged | Repaired: explicit stop reasons; max-step completion is not automatically convergence or promotion. |
| Study can pass while misleading | It may overfit historical streams or favor reverse-KL mode seeking. Fresh seeds and downstream HMC/predictive gates are required; no posterior claim follows. |
| Study can fail for tuning rather than idea | Narrow range, early rungs, validation noise, or resource exhaustion can cause no survivors. The result must separate candidate/search failure from NeuTra-direction failure. |
| Stale context/environment | Before implementation, verify current source hashes, `tfgpu` TensorFlow/TFP/Optuna versions, dirty worktree overlap, and GPU timing artifacts. |
| Artifact does not answer question | Repaired: trial records bind hyperparameters, rung diagnostics, vetoes, seeds, timing, controller policy, source hashes, and selected/fresh separation. |

Audit verdict: `PLAN_DESIGN_PASSES_WITH_EXECUTION_RESOURCE_GATE_OPEN`. The
implementation phases are justified. Material GPU tuning must wait for the
exact post-implementation timing and explicit resource authorization required
by Section 7.

## 9. Required Result Record

The result note must include:

- exact commands, commit/dirty status, environment, device/XLA/TF32/dtype,
  Optuna version, seeds, wall time, charged GPU time, and artifact paths;
- every trial state, rung reached, veto/prune reason, and common-rung objective;
- the frozen selected hyperparameters and plateau policy, or a no-survivor
  record;
- separate engineering, numerical/training, and scientific-interpretation
  ledgers;
- a decision table covering primary criterion, vetoes, uncertainty, next
  justified action, and nonclaims; and
- an inference-status table stating hard vetoes, viable candidates, whether
  any ranking is statistically supported, descriptive-only differences,
  default readiness, and next evidence required.

The post-run red team must state the strongest alternative explanation, what
would overturn the decision, and the weakest part of the evidence.

## 10. Stop And Handoff Conditions

Stop implementation for an unresolved source-preset regression, optimizer/XLA
incompatibility, non-reproducible resume, or inability to represent the plateau
state machine exactly.

Stop the study only for a continuation veto or the authorized resource cap.
Individual vetoed/pruned trials are expected evidence and do not stop later
trials.

Handoff to fresh confirmation requires a completed, valid Optuna artifact with
at least one joint survivor and a prospectively frozen selection/controller
policy. Handoff to HMC requires both fresh confirmation runs to pass all stated
gates and a new exact-transformed-HMC plan.
