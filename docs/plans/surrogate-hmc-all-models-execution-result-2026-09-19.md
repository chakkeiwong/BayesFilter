# Surrogate-HMC master: implementation and first execution

Continuation through September20 repaired the missing public TensorFlow tuning
authority and completed LG1 references, scoped controls, A/B tuning and retained
inference. A and B/R are INCONCLUSIVE after both planned looks with healthy
chains. The finite DCTR posterior D is independently qualified; D/R fails one
accuracy margin. Separate C tuning and retention completed; C/D and C/R are
INCONCLUSIVE at the planned limit, with healthy chains. LG2 reference and
consumer qualification passed; scoped controls selected multiplier 0.01.
Its optimistic measured B-only forecast exceeds the initial allocation, so
LG2 retention is deferred. See the
[tuner repair result](surrogate-hmc-tf-tuner-repair-result-2026-09-19.md) and
[LG1 retained result](surrogate-hmc-lg1-inference-result-2026-09-20.md), followed
by the [finite result](surrogate-hmc-lg1-finite-result-2026-09-20.md) and
[LG2 calibration result](surrogate-hmc-lg2-controls-result-2026-09-20.md).
The initial execution and stop below remain historical evidence; the task is
active under the existing authorization. No all-model inference claim is made.

C01 exact, half-strength and rotational Gaussian controls subsequently
completed 2,000 retained draws per chain and are INCONCLUSIVE; zero force
failed fresh tuning R-hat. All 36 mechanics checks and archive inspection
passed. Some uncertainty checks failed even for exact force, motivating a
known-reference diagnostic. See the
[control result](surrogate-hmc-gaussian-controls-result-2026-09-20.md).
The master preserves closed LG1/C01 attempts without new sampling.

At the initial P0/P1 execution recorded below, the full LG1 GPU/XLA consumer
passed its mechanics checks, but the available TensorFlow tuner lacked retained
admission authority. Its explicit handoff refusal was an implementation blocker.
That blocker was subsequently repaired. The tables below describe that initial
source revision and must not be read as the current implementation state.

The research goal remains testing whether surrogate HMC driven by an imperfect
full canonical DCTR force can recover model-parameter posteriors. Filtering
error, force disagreement and the historical tuner limitation do not answer that
question. The [master](surrogate-hmc-all-models-master-program-2026-09-18.md)
retains B/R, C/D, D/R and C/R as four separate posterior comparisons.

## Revision and implementation

The [Opus reply](surrogate-hmc-all-models-opus-reply-2026-09-18.md) correctly
withdraws the proposed force/endpoint input-equality requirement. Each binding
must remain frozen and internally consistent; they may differ. The master now
states that distinction explicitly and records separate precision by arm, proper
synthetic benchmark priors, and the owner's campaign authorization.

The [CLI](../../scripts/run_surrogate_hmc_program.py) now invokes stdlib-only
[orchestration](../../bayesfilter/inference/surrogate_hmc_program.py), source
discovery, a minimum 17-family register, LG1 scope construction and bounded GPU
qualification. It creates fresh output directories, preserves logs, stops and
charges timed-out work, carries prior expenditures into later launches, and
distinguishes execution failure from incomplete implementation. `--dry-run`
does not import TensorFlow or initialize CUDA. `--stage all` runs implemented
preflight stages and reports incomplete inference with exit code 2.

The [LG1 consumer](../../bayesfilter/testing/surrogate_hmc_lg1_tf.py) calls the
existing full canonical analytical route through the linear fixture. It binds
the DCTR proposal field separately from an independent TensorFlow Kalman
endpoint. It preserves original force validity and typed endpoint status.
The benchmark has stationary x0, Q=1, R=.5, transition before observation,
phi in (-.95,.95), a proper uniform prior and a bounded logit chart with its
Jacobian included once. The reference sees the identical observations promoted
from the DCTR FP32 representation. These are diagnostic fixtures; no scoped
DCTR control selection or inference admission is asserted. The GPU witness
exercises arm B's force/endpoint pairing; arm C's finite-DCTR endpoint binding
and its independent finite-target reference remain to be qualified.

Source discovery examined 2,610 Python files and retained 540 candidate-bearing
files. One historical vendor file could not be parsed:
`experiments/student_dpf_baselines/vendor/2026MLCOE/old_pt1_submission/filters.py`.
Its path and syntax error remain in the inventory. The index and minimum
register do not establish concrete-instance coverage closure. Only the LG1
diagnostic consumer is implemented; the other scopes remain unresolved.

## Executed evidence

| Check | Result | Limit |
|---|---|---|
| Focused CPU regression | 47 passed, 2 deselected | GPUs intentionally hidden; omitted implicit-pfor Jacobian and long Gaussian posterior test |
| Independent LG1 law/score checks | Kalman likelihood and total derivative agree with dense Gaussian observation law; chart and predictive probability checks pass | Diagnostic mathematical reference, not posterior sampling |
| First actual full LG1 probe | 11 mechanics checks passed | Four checked positions, T10/N1024, diagnostic controls |
| Reviewed full LG1 probe | 13 mechanics checks passed; scalar/batch value and force difference 0 at checked point | No all-model or all-branch qualification |
| Replay, row independence, endpoint/cache, per-chain energy | Passed on the actual GPU/XLA consumer | Does not establish convergence |
| Reversed proposal | Maximum absolute residual 5.55e-17 | Diagnostic tolerance was 32 FP32 ulps at the observed scale; independent inference-tolerance calibration remains open |
| Public tuner with actual full consumer | Diagnostic health passed, finite states, `admission_supported=false`, `handoff_eligible=false` | No artifact authority; outer tuner non-XLA diagnostic exception, nested force GPU/XLA |
| Retained-runner handoff | Rejected with `TensorFlow diagnostic tuning is not admission-capable and cannot build a retained runner` | Confirms the implementation blocker, not a sampler failure |

Both GPU runs used an idle RTX 4080 SUPER selected by UUID, escalated access,
verified memory growth before initialization, TensorFlow
`2.20.0-dev0+selfbuilt`, and FP32 DCTR with TF32/XLA inside FP64 integration and
FP64 Kalman endpoint evaluation. All runs preserve their actual source hashes,
data seeds, input identities, settings and commands. The public tuner's outer
non-XLA graph is explicitly diagnostic and cannot support default readiness.

The first dry-run regression failed because the package's custom-op preload
imported TensorFlow. The CLI now disables that preload before importing its
orchestration module. The repaired initial suite passed 12 tests; the expanded
47-test review also passed. The review additionally repaired budget resets,
launch-error accounting, overbroad LG-family readiness and raw-nonfinite
reporting. These were harness failures or review findings, not DCTR rejection.
The final CLI review also pins the probe's particle-count option to N1024,
preventing a caller from changing N while retaining the diagnostic scope label.
The actual GPU command already used N1024; its numerical execution is unchanged.
Copies of the executed uncommitted sources are preserved under the test evidence
directory's `source/`, with the small post-run CLI guard recorded in validation.

Complete evidence:

- [Initial run manifest](artifacts/surrogate-hmc-all-models-20260918T155904Z-01/run-manifest.json)
  and [initial diagnostics](artifacts/surrogate-hmc-all-models-20260918T155904Z-01/P1-LG1/diagnostics.json).
- [Reviewed run manifest](artifacts/surrogate-hmc-all-models-20260919-review-02/run-manifest.json),
  [diagnostics](artifacts/surrogate-hmc-all-models-20260919-review-02/P1-LG1/diagnostics.json),
  [public tuning artifact](artifacts/surrogate-hmc-all-models-20260919-review-02/P1-LG1/public-tuner/manifest.json),
  [complete worker log](artifacts/surrogate-hmc-all-models-20260919-review-02/P1-LG1/command.log),
  and [decision](artifacts/surrogate-hmc-all-models-20260919-review-02/decision.json).
- [Regression log](artifacts/surrogate-hmc-all-models-20260919-review-tests/surrogate-hmc-review-tests-20260919.log)
  and [test accounting](artifacts/surrogate-hmc-all-models-20260919-review-tests/accounting.json),
  including the preserved failed initial test and subsequent repair logs.

The exact second launch was:

```bash
env CUDA_VISIBLE_DEVICES=GPU-68251639-fe82-8f81-3ccc-2953c32e805b TF_FORCE_GPU_ALLOW_GROWTH=true /home/chakwong/anaconda3/envs/tftwogpu/bin/python scripts/run_surrogate_hmc_program.py --stage all --scope all --output docs/plans/artifacts/surrogate-hmc-all-models-20260919-review-02 --max-gpu-seconds 3600 --max-cpu-worker-seconds 7200 --probe-timeout 900
```

The worker exited 0. The master exited 2 because inference is incomplete.
Neither invocation overwrote earlier evidence. No process remains running.

## Cost and decisions

Actual GPU charges were 48.344 seconds and 68.189 seconds, totaling **116.533
GPU seconds**. Cumulative CPU accounting is **228.164 CPU-worker seconds**,
including conservative charges for early unmetered checks. The
[ledger](artifacts/surrogate-hmc-all-models-20260919-review-02/accounting.json)
contains each contribution. Remaining master allocation is approximately
**62.9676 GPU hours and 14.9366 CPU-worker hours**. Neither additional LG1
inference attempt has been consumed; indices 4 and 5 remain available.

Single steady-transition measurements were 1.711 and .946 seconds for four
chains and L=2. Conditional 2000-warm-up plus 2000-retained forecasts were
2.868 and 1.601 GPU hours with the 1.5 factor, **excluding full tuning and
reference costs**. The spread is descriptive, not an uncertainty interval or
speed comparison. No complete inference schedule has been shown to fit its
allocation. Reforecast at selected settings with repeated timings and the
whole comparison cost before launching inference.

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | What is not concluded |
|---|---|---|---|---|---|
| Accept recorded LG1 mechanics as P1 diagnostic evidence | Full-consumer checks pass at tested inputs | No observed replay/coupling/cache/energy failure there | Untested branches, tolerance calibration and remaining scopes | Preserve evidence and complete prerequisite repair | Posterior correctness or default readiness |
| Keep inference `INCOMPLETE_IMPLEMENTATION` | All B/R, C/D, D/R and C/R comparisons NOT_RUN | Required TF public tuning authority is absent | Scope-specific selection, reference qualification and complete cost | Repair shared TF ordinary tuning and its artifact handoff before retained runs | Failed surrogate-HMC idea or rejected force candidate |
| Keep all-model coverage incomplete | Source index and minimum register exist | Concrete scope/reference/consumer work is missing | Additional instances and vendor parse exception | Reconcile each model into executable scope cards | All models tested |

| Inference status | Evidence |
|---|---|
| Hard veto screen | No mechanics failure at tested LG1 points; missing admitted public TF tuner blocks retained inference |
| Statistically supported ranking | None; no posterior comparison ran |
| Descriptive-only differences | Single-call times, reversal residual and scalar/batch equality |
| Default-readiness | Not established |
| Next evidence needed | Eligible public tuning, scoped DCTR selection, qualified references, independent retained draws and the four posterior comparisons |

The claimed target is the declared model posterior. What was actually computed
was a set of deterministic consumer/mechanics checks and a three-stage tuning
diagnostic. These quantities are different; they do not estimate the posterior
comparison criteria. The force candidate remains untested for useful inference,
and no candidate or research-direction rejection follows.

Post-run red team: the strongest alternative to an apparent mechanics success
is an untested support/fallback branch or shape/precision scope. A failing
same-scope invariance or status test would overturn that limited success. The
weakest feasibility evidence is the pair of single-transition timings. The
tuner blocker is supported by source, public documentation and an actual
retained-handoff refusal. Repairing it, then demonstrating a complete affordable
inference schedule, would remove that blocker without changing the scientific
question. Flipping diagnostic admission flags would not constitute repair.
