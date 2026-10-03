# Phase 9B strict runtime health investigation and continuation

Date: 2026-09-11 (Asia/Shanghai; prefix launch was September 10 UTC).
Status: `RECOVERY_PASSED_FACTOR_RUNTIME_COMPLETE_STRICT_RUNTIME_ACTIVE`; P1 has not started.
Governing plan: `bayesfilter-ssl-lstm-q20-phase9b-recovery-runtime-8h-amendment-plan-2026-09-09.md`.
Campaign: `artifacts/ssl-lstm-q20-phase9b-recovery-runtime-2026-09-09/campaign-24gpuh-20260909T135803Z/`.

## Current evidence and skeptical audit

Latest check, **September 12, 02:45 Shanghai**: training/tuning complete for
both arms and actual interruption/resume canaries pass exact tensor/trace
equality. In r2 `runtime-c355a7aa08`, factor finishes both 500-transition chunks
with passing full health; strict passes its first chunk and runs its second.
No nonfinite-value/status veto appears in the three completed chunks. The
service remains active; let strict finish and evaluate the unchanged two-arm
affordability checks before P1. Sources, thresholds and budget remain frozen.
See the current amendment result for receipts and unsettled runtime accounting.

### September 12, 00:43 Shanghai: GPU validation and migration pass

Service launches successfully at 00:44:39 Shanghai as
`bayesfilter-q20-phase9b-eigh8-20260912-r1.service`. Both workers start in
`r2/launches/reference-67d95ab1f2/`: factor on non-display GPU 1 and strict on
non-display GPU 0, with verified growth. The launch reserves 7,200 aggregate
seconds for setup, leaving 69,676.30067716587 unreserved seconds. Sources and
active migration are frozen throughout execution; stage results remain pending.

00:48 Shanghai: both fresh charts are committed after training at beta 0,
0.5 and 1. All six 32-row chart preflights pass. Public tuning is active in
both worker processes; completed calls inspected so far have finite required
quantities and valid target status. No fresh recovery or full-runtime result
exists yet, so neither trajectory stability nor P1 admission is inferred.

`launches/eight-sweep-gpu-validation-5c70410169/` passes both saved 32-row banks,
both four-row original endpoint comparisons, and odd-dimensional sign controls
on non-display GPU 1. All four static functions trace once with recorded HLO;
memory growth is verified and allocator peak is 269,139,456 bytes. Maximum
bank likelihood residuals are 6.82121e-13 (factor) and 2.27374e-12 (strict),
and scaled analytic-score residuals 1.23425e-14 and 4.43408e-14 against the saved
independent native CPU reference. No invalid row remains on these inputs.
This is fixed-bank numerical equivalence, not tuning or sampler stability.

The supervised diagnostic costs **53.115703790001135 GPU-worker seconds**.
Ledger: 9,523.699322834134 consumed, zero reserved, 76,876.30067716587 remaining.
The eight-sweep migration is active in `numerical-repairs/eigh-refinement-r2/`;
`runtime-health-diagnostic-20260911-r1/eight-sweep-migration-7a685780c8/` archives
the old migration, current sources and validation. It verifies 124 bundles and
3,109 tensors while preserving original start/ledger bytes. All 137 existing
CPU regressions bind the same current numerical source; no source changed
after GPU validation.

The post-validation audit permits the unchanged P1 entrypoint to run the
fresh two-arm recovery/runtime sequence. Start service
`bayesfilter-q20-phase9b-eigh8-20260912-r1` with `Restart=no`,
`KillMode=control-group`, `TimeoutStopSec=45`, and the inherited 87,000-second
outer wall ceiling. The coordinator hides GPUs; children use the verified
UUID-pinned, memory-growth worker launcher and the owner GPU selection policy.
Standard output/error go to the fresh r2 execution root. Command:

```bash
/home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/run_ssl_lstm_q20_phase9b_p1_sequential_canary_2026_09_05.py \
  --campaign-root docs/plans/artifacts/ssl-lstm-q20-phase9b-recovery-runtime-2026-09-09/campaign-24gpuh-20260909T135803Z \
  --resume
```

The scientific target, derivative, training/tuning protocol, health checks,
8-hour per-arm cap and 24 aggregate GPU-hour allocation remain unchanged.
Never reuse r1 setup/streams or forecasts. After fresh parallel setup, the
coordinator performs real interruption/resume comparison, both full runtime
health checks and new affordability estimates before bounded P1. The
original NaN/preflight failures remain preserved. A new numerical failure
triggers localization; it does not authorize a relaxed gate or reseeding.

### Post-reboot continuation, September 11

The owner reports reboot complete and requests continuation. Trusted inventory
confirms matching loaded NVIDIA module and NVML **580.178.04**, boot time
18:01:12 Shanghai, and no reboot-required marker. Both non-display GPUs are
eligible at the initial check (3%/0% utilization, over 31 GiB free each); the
selector must recheck immediately before each launch. The driver blocker is
resolved, not the pending eight-sweep GPU numerical validation.

The skeptical audit permits one fixed-bank GPU diagnostic capped at 900
aggregate worker seconds from the existing ledger. Compare actual current
eight-sweep strict and factor-cached GPU/XLA likelihood and analytic score on
the saved 32-row banks against the saved independent native CPU results in
`preflight-eight-sweep-bank-r1/`; also repeat the original four-row failed
endpoint comparison and odd-dimensional sign controls. Require all rows valid,
the existing value/score tolerances, one trace per static signature, verified
memory growth and no source/data changes. No training or sampler runs in this
diagnostic. HLO, full outputs, sources, input hashes, GPU identity and allocator
readings are preserved under a fresh `launches/eight-sweep-gpu-validation-*`.
The 900-second cap is the inherited bounded-diagnostic ceiling, not a runtime
forecast; stop on a failed numerical, resource or source check and do not
retry with different inputs. CPU references are read-only comparators, not
runtime selection authorities. No stochastic ranking follows from residuals.

On a pass, archive the four-sweep migration and record the eight-sweep source
binding in a fresh `numerical-repairs/eigh-refinement-r2/` namespace, retaining
the original start/ledger and all prior results. Then execute the existing
parallel reference/tuning, actual interruption/resume, two-arm full runtime
and budget-checked P1 sequence. New namespace seeds are for a fresh numerical
scope, not a substitute for passing the previously failed fixed banks. Fresh
runtime measurements must determine affordability; doubling the eigensolver
sweep cap may increase cost and the old forecasts cannot establish it.

### Pre-reboot state (historical)

Final September 11, 15:36 Shanghai update: **137 focused CPU regressions pass**
(121.77 seconds, no failures/errors/skips); `git diff --check` passes. The
read-only resume check correctly rejects the stale four-sweep source migration
before any worker launch and preserves the original start, ledger and
migration byte-for-byte. Validation, artifact hashes and current source
snapshots are in `eight-sweep-validation-a8df05dcf2/validation.json` beneath
the diagnostic directory. The trusted NVIDIA recheck still fails with the
same loaded/user-space mismatch; the campaign service remains stopped. No
GPU seconds are added. The unchanged remaining budget is
**76,929.41638095587 aggregate seconds**, with zero reserved.

At September 11, **11:54 Asia/Shanghai**, trusted NVIDIA checks fail with
`Failed to initialize NVML: Driver/library version mismatch` (exit 18).
The loaded kernel module is **580.173.02**; the installed NVML library is
**580.178.04**. `/var/log/dpkg.log` records installation of the new driver
packages at **06:02:27**, after the 04:52 preflight attempt, and removal of the
old firmware at 06:02:59. This is a new host-environment blocker, not a cause
of the earlier preflight failures. No package changes or reboot were performed
by this campaign. The diagnostic supervisor fails at inventory, before any
reservation or worker launch, so GPU spending remains 9,470.583619044133 seconds.
Do not bypass missing GPU telemetry or alter system packages; a host operator
must restore matching loaded/user-space driver versions before GPU continuation.

The CPU native-eigensolver reference reconstructs both seeded 32-row banks and
passes all rows. Exact equality with GPU-generated parameters is not checked
because the failed GPU preflight did not save them. The refined non-JIT CPU
diagnostic saves a passing 32/32 factor preflight receipt, then reaches its
300-second timeout before a full bridge report; preserve `cpu-preflight-r1/`,
and do not interpret timeout as numerical failure.
The subsequent CPU/XLA diagnostic also reaches its 300-second cap. It saves
the complete factor bank result before timing out during strict evaluation:
31/32 rows pass; row 15 (zero-based) fails with one invalid placement and a
`-1e100` sentinel, not a measured eigenvalue. All 32 saved physical factor
inputs are exactly equal to the independent native CPU reference's inputs,
which pass 32/32. Thus the factor rejection is not GPU-only. Strict CPU/XLA
is incomplete; neither timeout establishes its numerical outcome. Preserve
both partial attempts. CPU-only evidence cannot clear GPU/XLA scope, runtime
or recovery. Trusted NVIDIA rechecking at 12:06 still confirms the mismatch.

### Bounded fixed-row localization audit, September 11

Completed at 12:20 Shanghai in `preflight-eight-sweep-bank-r1/` (411.4079 CPU
wall seconds, zero GPU seconds). The eight-sweep process-local prototype passes
both fixed 32-row banks against same-input native CPU reference. Factor maximum
likelihood residual is 2.38742e-12 and scaled analytic-score residual
1.00143e-13; strict gives 1.47793e-12 and 7.36548e-14. All 64 rows are valid.
This repairs the observed bank under CPU/XLA, not a universal guarantee or GPU
readiness claim. The old four-sweep fixed-row regression fails on both paths;
that failure is preserved in `four-sweep-preflight-regression-red.xml`.

The implementation changes only the refinement default sweep cap from four
to eight and its docstring; residual, orthogonality, SPD, derivative and health
thresholds are unchanged. Both fixed-row tests and the fail-closed convergence
control pass (three tests, 19.04 seconds), followed by all 137 focused
core/recovery regressions. The prior four-sweep migration is
stale by design; do not resume its workers or reuse its receipts. GPU restart
requires matching driver versions, a trusted GPU/XLA fixed-bank validation,
and a new eight-sweep numerical migration/versioned execution namespace that
preserves the original campaign start/ledger. This is a numerical-source
change, not serializer-only continuation.

Completed at 12:10 Shanghai in `preflight-row-localization-r1/` (13.6873 CPU
wall seconds, zero GPU seconds). The one-row uninstrumented and instrumented
filter return exactly equal values and scores and reproduce an invalid
placement at time index 8. The saved covariance is positive definite by the
native CPU eigensolver (minimum 1.7476690e-11). Four refinement sweeps leave
eigenpair residual 2.08864e-11, exceeding the unchanged 8.91555e-12 bound;
orthogonality passes. Eight and twelve sweeps both reduce the residual to
7.74426e-14 and pass. Thus this failure is insufficient Jacobi convergence,
not a nonfinite covariance, a true negative eigenvalue or swallowed exception.
One-row reproduction is not a bitwise whole-bank replay claim.

Next bounded diagnostic: evaluate an eight-sweep process-local prototype on
both saved 32-row physical banks, then compare full likelihood and analytic
score with native CPU reference evaluation at the same parameters. Include
the original four-row endpoint bank and the just-localized row in subsequent
regressions. Eight is a measured sufficient sweep count for the failed
covariance, still a candidate cap for other inputs; twelve's identical result
is a convergence cross-check, not a universal guarantee. Keep every residual,
orthogonality, SPD and invalid-row test unchanged. Numerical continuation
requires all rows valid, likelihood agreement within the existing 1e-10
absolute/1e-12 relative regression tolerances, and score agreement within
1e-9 absolute/1e-10 relative tolerances. These inherited tolerances are
equivalence screens, not evidence of stochastic superiority.

Use a fresh `preflight-eight-sweep-bank-*` output with start/completion
manifests and durable per-arm results. A 900-second CPU wall cap is a debugging
convenience bound informed by the 105-second native reference and incomplete
300-second refined run, not a GPU allocation or measured campaign estimate.
No training, sampler, reseeding or runtime-source edit occurs in this
prototype. If it passes, a minimal core change plus focused regressions is
allowed, but any later GPU restart needs a new honest numerical migration and
full GPU bank validation after the host driver is repaired. The audit passes:
same-input reference, unchanged scientific question and guards, explicit
nonpromotion and timeout/source-change stop conditions. A wider-bank failure
returns to localization, never to bypassing preflight.

Run a new CPU-only/XLA diagnostic of saved factor row 15; preserve its input
checksum and instrument a copy of the current recursion to save each incoming
placement covariance and invalid flag. No optimizer, sampler, new seed,
runtime-source edit, or GPU launch is allowed in this diagnostic. Compare
uninstrumented and instrumented numerical results; any instrumentation or
batch-shape difference is a limitation, not an exact whole-bank replay claim.
For the first rejected covariance, inspect the refinement's eigenpair residual,
orthogonality and finite-value checks against independent native CPU `eigh`.
Check additional refinement sweeps only as a diagnostic hypothesis; do not
weaken tolerances, erase a true negative eigenvalue, or promote a passing row
to repaired-scope clearance. The 300-second wall cap is a convenience bound
for localization, not a measured training requirement. Output is a fresh
`runtime-health-diagnostic-20260911-r1/preflight-row-localization-*` directory.

The skeptical audit passes for this limited question: the baseline uses the
same saved physical parameters rather than a reseeded bank; residuals diagnose
implementation failure, not scientific promotion; nonfinite input, source
changes and cap exhaustion stop this diagnostic. The main risk is compiler
or batching sensitivity. Failure to reproduce requires whole-bank tracing,
not declaring the original rejection spurious. The research question,
target, analytic derivative, health vetoes and total GPU budget stay fixed.

Latest execution: the detached service runs September 11, 04:52:31–04:52:55
Asia/Shanghai. **Both GPU workers actually start in parallel**, on non-display
GPUs 0 and 1, with verified memory growth and XLA. Both reject their fresh
beta-zero chart preflight before any optimizer update or HMC tuning. The
reference attempt is `numerical-repairs/eigh-refinement-r1/launches/reference-b6ab845731/`.
Factor consumes 19.757533525000326 seconds and strict 22.66759688500245;
the ledger now records **9,470.583619044133 consumed**, zero reserved,
**76,929.41638095587 remaining** seconds. Initial wording that the service
failed before workers started was incorrect; their startup receipts settle it.

Do not retry with different seeds or relax preflight. Its boolean failure did
not preserve per-row diagnostics, so the next smallest diagnostic reconstructs
the exact fresh chart and 32-row preoptimizer bank, saves physical parameters,
receipt, value/score/status, and localizes failing rows. Compare CPU reference
and refined arithmetic first, then the same GPU/XLA bank if needed. These
are debugging-only, with no optimizer or tuner and no candidate admission.
A GPU diagnostic may reserve at most 900 aggregate worker seconds from this
same ledger, must snapshot diagnostic code, and must preserve failed attempts.
Keep all runtime source hashes unchanged while localizing. The failure may
expose inadequate refinement on a wider bank or a genuine invalid candidate;
neither is assumed. Target/status/finite-value checks remain unchanged vetoes.

The earlier endpoint validation and migration audit remain valid within their
limited scope. The new failed bank is a repair trigger, not permission to
declare the campaign complete, abandon the research direction, or bypass the
screen. No recovery/runtime/P1 result exists for the repaired scope yet.

Current update, September 11, 04:18 Asia/Shanghai: the saved covariance is
positive definite, not indefinite. The XLA eigensolver's binary32 internal
Jacobi cutoff produces a false negative eigenvalue in this binary64 problem.
The shared binary64 refinement repair passes 122 CPU regressions and the
integrated GPU/XLA fixed-point check for both strict and factor-cached paths.
Each compiled path traces once, returns four valid rows, and agrees with the
independent CPU reference within 7.11e-15 in likelihood and 6.10e-15 in scaled
analytic score. Positive and negative odd-dimensional controls pass. Evidence:
`launches/candidate-diagnostic-a7e029690b/strict/`. This is numerical debugging
and equivalence evidence, not fresh tuning, recovery, convergence or posterior
admission. The following paragraphs retain the chronological investigation.

The candidate validation consumes 24.664144911977928 GPU-worker seconds.
Settled budget is 9,428.15848863413 consumed, zero reserved, and
76,971.84151136587 remaining aggregate seconds (21.3811 GPU-hours).

### Numerical migration and continuation audit

The old serializer-only migration cannot admit the edited numerical core.
Both backends share that core, so retaining either old chart, tuning handoff,
stream, completion marker or canary receipt would be a stale-scope error.
Use `numerical-repairs/eigh-refinement-r1/` beneath the original campaign for
fresh chart training, public fixed-transport tuning, reference, actual
SIGKILL/resume canaries, two-call runtime/health, and conditionally bounded P1.
Keep the original campaign start, 86,400-second ledger and coordinator lock;
this namespace receives no separate allocation. Preserve the serializer-only
migration before replacing the active migration with an explicitly numerical
one binding exactly the wrapper and shared eigensolver module changes.

Audit verdict: the continuation is justified as a localized arithmetic repair
under the unchanged scientific question and budget. Baselines are the saved
independent CPU principal-root target/score for the repair, and same-source,
same-device uninterrupted runs for recovery. Neither old factor timing nor
the fixed-point smoke supplies current runtime or scientific admission.
The original plan's target, data, four-chain schedule, health/energy/status
vetoes, public tuner, GPU policy and P1 boundary remain unchanged. The inherited
smaller-step tuning grid and four-sweep refinement remain explicit hypotheses;
fresh training/tuning and full health checks may reject them. Do not rank
methods or infer posterior correctness from passing these checks.

Before launch, test namespace isolation, fresh seeds/setup/streams, resume
from only current-source receipts, rejection of changed sources or stale
namespace contents, and byte-preservation of the original start/spent ledger.
Retain the original 3,600-second reference, 400-second interruption/resume,
28,800-second per-arm runtime ceilings; P1 requires fresh measured affordability
within the remaining aggregate budget. Source/target mismatch, corruption,
invalid memory allocation, missing health diagnostics and budget exhaustion
remain continuation vetoes. A candidate health failure is a repair trigger,
not evidence against the research direction. No P2 or promotion is authorized.

### Migration validation and launch command

The final combined CPU suite passes **135 tests**, including interrupted
execution-namespace creation, stale completion isolation, fresh chart/tuning/
stream seeds, actual same-ledger two-arm scheduling, and resumed same-source
work. Activation preserves the original start and ledger byte-for-byte and
verifies **101 committed bundles / 2,697 tensors**. Receipt and source snapshots:
`runtime-health-diagnostic-20260911-r1/migration-activation-9da3c94a37/`.
The preceding serializer-only migration is archived there. No old tuning,
stream or readiness marker is copied to the numerical namespace.

The trusted installed idle-only GPU probe reports no idle device because other
processes exist. This is not a CUDA failure. Trusted NVIDIA inventory shows
ample memory on all devices, GPU 0 at 2% utilization, GPU 1 at 46%, and the
display GPU at 40% at September 10, 20:50 UTC. The repository's owner-approved
40%-load/5-GiB-headroom selector, not an idle-only criterion, governs launch.
It will launch two processes only if two non-display devices are eligible;
otherwise work queues behind an eligible non-display device, using display
only when no non-display device is eligible. No unrelated process is stopped.

Launch in a detached user service, preserving stdout/stderr in the fresh
numerical namespace. The coordinator intentionally hides GPUs; child workers
pin the selected UUID and establish/verify growth before framework device
initialization. The outer **87,000-second wall ceiling** is a convenience bound
derived from the original 86,400-second allocation plus 600 seconds for parent
overhead; it does not add to the worker budget. No automatic service restart
or statistical retry is configured. Source code and the active migration are
frozen after this launch; only live result/master/reset notes may change.

```bash
systemd-run --user --unit=bayesfilter-q20-phase9b-eigh-20260911-r1 \
  --working-directory=/home/ubuntu/python/BayesFilter \
  --property=RuntimeMaxSec=87000 --property=TimeoutStopSec=45 \
  --property=KillMode=control-group --property=Restart=no \
  --property=StandardOutput=append:/home/ubuntu/python/BayesFilter/docs/plans/artifacts/ssl-lstm-q20-phase9b-recovery-runtime-2026-09-09/campaign-24gpuh-20260909T135803Z/numerical-repairs/eigh-refinement-r1/service.stdout.log \
  --property=StandardError=append:/home/ubuntu/python/BayesFilter/docs/plans/artifacts/ssl-lstm-q20-phase9b-recovery-runtime-2026-09-09/campaign-24gpuh-20260909T135803Z/numerical-repairs/eigh-refinement-r1/service.stderr.log \
  --setenv=CUDA_VISIBLE_DEVICES=-1 --setenv=TF_FORCE_GPU_ALLOW_GROWTH=true \
  --setenv=TF_NUM_INTRAOP_THREADS=1 --setenv=TF_NUM_INTEROP_THREADS=1 \
  --setenv=PYTHONUNBUFFERED=1 \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/run_ssl_lstm_q20_phase9b_p1_sequential_canary_2026_09_05.py \
  --campaign-root docs/plans/artifacts/ssl-lstm-q20-phase9b-recovery-runtime-2026-09-09/campaign-24gpuh-20260909T135803Z \
  --resume
```

The serializer repair completed the two-arm reference and actual SIGKILL/resume
canaries. Both arms have exact tensor/trace equality, reuse the first committed
chunk, and recompute the interrupted second chunk. This establishes recovery
mechanics for that eight-transition fixture, not sampler validity.

The following runtime wave, `runtime-c48f1f0535`, ended on September 10 at
17:34:11 Shanghai. Factor completed both timing calls. Strict committed its
first 500-transition chunk, then failed `nonfinite_log_accept_ratio` and
`nonfinite_delta_h`. All 2,000 sampled states and their target values remained
finite; sampled-state target status passed and all four chains moved. This
does not establish finite proposal states or scores, which the shared trace
does not expose. It is not the prior short-chain ESS serialization failure.

The audit rejects an unchanged restart: it would replay the same unhealthy
committed chunk and cannot answer why it failed. Acceptance and finite retained
states cannot override the declared proposal-energy health veto. First inspect
the checksum-verified stored trace with GPUs intentionally hidden, then trace
the actual fixed-transport/TFP code. A numerical reproduction is a separate,
bounded debugging step to be specified after that inspection. Do not change
the target, score, Metropolis correction, health thresholds, or tuning authority
to make a command pass. Do not reuse failed runtime draws for tuning or P1.

## Research intent and evidence contract

| Item | Definition |
| --- | --- |
| Main question | Does the strict runtime stop arise from corrupt reporting/checkpointing, an implementation defect, or a numerically invalid tuned candidate? |
| Candidate/mechanism | The frozen strict chart and public fixed-transport tuner handoff used by the failed runtime wave. |
| Exact comparator | The original checksum-verified strict chunk, its original inputs/seed and full health receipt; factor is contextual evidence only, not a like-for-like numerical baseline. |
| Expected failure mode | Invalid rejected proposals can leave sampled states finite; alternatively serialization or telemetry could misreport valid values. Neither explanation is assumed. |
| Pass criterion | Localize nonfinite values (NaN versus signed infinity, transition and chain), verify original health computation, and identify the first supported causal layer. A proposed repair needs a focused regression and fresh unchanged-contract health validation. |
| Promotion veto | Existing nonfinite state/target/score/acceptance/energy and target-status vetoes remain unchanged. |
| Continuation veto | Corrupt checkpoints, target or derivative mismatch, unverified GPU memory policy, exhausted budget, or a changed scientific contract. Failure of this candidate alone does not reject the direction. |
| Repair trigger | A localized code defect permits a regression-tested repair; a candidate failure triggers fresh scope-specific public tuning, not hand-edited epsilon or replay-until-pass. |
| Explanatory diagnostics | Acceptance, finite energy extremes, location of rejected proposals, timing, and parameter/score ranges. None establishes convergence or a candidate ranking. |
| Not concluded | Posterior validity, default readiness, superiority, or full P1 recovery coverage. |
| Preserved result | Fresh `runtime-health-diagnostic-20260911-r1/` under the campaign, this note, and master/reset/result updates. Existing evidence is not overwritten. |

## Defaults, assumptions, budget, and pre-mortem

The 28,800-second per-arm and 86,400 aggregate GPU-worker-second caps are owner
authorized, not measured requirements. The current ledger records
8,534.501830745052 consumed, zero reserved and 77,865.49816925495 remaining
seconds. Reconcile the ledger before any GPU launch; no new allocation is
created. The four chains, 500-transition chunks, float64 GPU/XLA target,
frozen chart, tuner scope, seeds, and health policy are inherited from the
governing plan; their continuation is a controlled comparison, not a claim of
optimality. The selected strict epsilon 0.0275 with L=3 is a tuned candidate,
not a universally stable default; its early diagnostic is precisely the failed
runtime health screen. Short canaries did not establish long-run stability.

Stored-trace inspection is an explicit CPU-only, post-run diagnostic exception.
Use `CUDA_VISIBLE_DEVICES=-1 TF_FORCE_GPU_ALLOW_GROWTH=true` before importing
TensorFlow in the `tfgpu` environment, with one intra/inter-op thread. It runs
no sampler or optimizer updates and spends no GPU budget. Preserve command,
commit, input checksums, wall time and report. The first stage needs no external
source claims or new mathematical assumptions. If installed TFP behavior is
used in an explanation, inspect the actual installed source.

The strongest misleading-success risk is mistaking rejected nonfinite
proposals for harmless diagnostics because sampled states remain finite.
The strongest misleading-failure risk is blaming the target for a diagnostic
encoding problem. Counts, locations, retained-state equality across rejected
transitions, checkpoint checksums, and the actual acceptance-ratio code
distinguish these before an expensive replay. Any later numerical-code source
migration must be described honestly; the existing serializer-only migration
cannot authorize a different numerical dependency closure.

## Stored-trace result and smallest proposal replay

Checksum-verified CPU inspection finds exactly one negative-infinite acceptance
ratio, zero NaNs and zero positive infinities: transition 159, chain 4
(one-based). It was rejected and the sampled state is bitwise unchanged. Both
factor chunks are finite. Recomputing the original health function reproduces
exactly the strict vetoes. The saved trace contains no proposal states or
proposal scores, so it cannot locate the internal cause by itself.

Installed TFP `metropolis_hastings.py:204` sums proposed minus accepted log
target and kinetic correction. `mcmc/internal/util.py:206` replaces a nonfinite
sum with negative infinity; `mcmc/hmc.py:864` uses the same operation for the
kinetic difference. The strict bridge deliberately returns NaN value/score
when its numerical status is invalid (`tempered_target_tf.py:586`). Thus
negative infinity does not distinguish raw NaN, infinity or an invalid-status
rejection. These are inspected implementation facts, not a root-cause verdict.

Next run at most two diagnostic one-step replays from stored pre-transition
states, using the original sample-chain seed splitting and unchanged public
handoff, first on CPU/XLA with GPUs intentionally hidden. Reconstruct transition
158 as a control and transition 159 as the failing case; export proposed state,
potential, gradients, momentum and raw acceptance components. This is a tiny
debugging exception, not CPU training or runtime evidence. A 240-second external
cap is a convenience diagnostic bound, not a measured requirement. Reuse only
already-committed chart/tuning checkpoints, never invoke new training/tuning.
Preserve output in `runtime-health-diagnostic-20260911-r1/cpu-replay-r1/`.

CPU/GPU random generation or compilation may differ. Compare the control
against the saved GPU sample/trace before attributing the reproduced proposal
to the original failure. A mismatch or timeout is a diagnostic limitation, not
target failure; then specify a same-GPU bounded replay. Do not change numerical
code based on a failed cross-device match. Replaying failed states is debugging
only and cannot produce a fresh tuning or P1 success.

The CPU replay completed in 51.91 seconds. Its control differs from the GPU
sample by at most 3.064e-9, and the failed transition becomes finite/accepted
on CPU. It therefore does **not** reproduce the original failure. Preserve it
as a cross-device diagnostic only. No health gate or numerical source changes.

Run a same-GPU/XLA diagnostic on the original strict GPU UUID, using the same
stored states/seeds and frozen handoff. Export the two one-step controls plus
L=1 and L=2 prefixes of the failing L=3 trajectory to locate the earliest invalid
potential, gradient or numerical-status field. Prefixes are explanatory, never
retuning candidates. Reserve **900 aggregate worker seconds** from the existing
ledger for one worker, including compilation and cleanup; this is a conservative
convenience cap, not a forecast. Maximum one launch at this rung; a failed replay
must be interpreted before further work. The parent settles measured lifetime,
keeps a fresh output directory, and preserves failed-attempt evidence. All code
changes are in diagnostic files outside the campaign's numerical dependency
closure; verify that closure before and after. Original GPU 1 is currently
non-display, 0% utilized with ample headroom. Recheck at launch, apply the
40%/5-GiB policy, enable/verify memory growth before device initialization and
monitor headroom. No unrelated GPU work is stopped. Only strict needs debugging;
rerunning the successful factor worker in parallel would waste compute.

The same-GPU proposal-local diagnostic completed in 42.782334198011085
GPU-worker seconds. It also did not reproduce the original negative infinity:
the control differs by 2.509e-9 and the original failed proposal is finite.
All tested L=1/2/3 prefix proposals have valid reported status. Memory growth
passed; TensorFlow peak allocator usage was 34,018,304 bytes. Remaining budget
is 77,822.71583505694 seconds. There is no evidence of a seed-splitting bug;
similar control values support the seed reconstruction. One-step replay
recomputes the initial gradient instead of carrying the original accepted
kernel result, and its compilation context differs. Neither it nor CPU replay
can remove the original veto or prove the original cause.

Next bounded diagnostic: call TFP `sample_chain` from the original initial
bank/seed through transition **159** (derived from the stored failure index),
preserving carried kernel results rather than bootstrapping at transition 158.
Keep the original target, epsilon/L and sampled-state status trace; add only
already-computed proposed state, score, potential and momentum to the trace.
Persist all tensors before interpreting them. Compare every prefix sample,
acceptance and target against the saved GPU prefix. Exact reproduction is
reported only if those comparisons agree; otherwise this is a diagnostic
trajectory with changed compilation/trace shape, not original-failure proof.
Report the first deviation and first nonfinite proposal separately. Evaluate
numerical status at finite bad-proposal endpoints only after saving the trace.

Reserve **1,800 worker seconds** for one launch, charged to the same ledger.
This convenience cap is above the descriptive 159/500 times 2,025.77-second
prior strict call (about 644 seconds), allowing compilation/cleanup but making
no statistical runtime guarantee. Stop on its cap, source/device/memory policy
failure, or missing/corrupt required trace. Do not retune or change numerical
code yet. This is debugging, not a new candidate health admission. Preserve the
previous diagnostic sources with their results before editing diagnostic code.

Live launch at September 10, 17:08:21 UTC:
`bayesfilter-q20-phase9b-prefix-diagnostic-20260911-r1.service`; worker 869730;
output `launches/prefix-diagnostic-83873ea777/`; supervisor command is the
campaign's `runtime-health-diagnostic-20260911-r1/supervise_proposal_diagnostic.py`
in `/home/ubuntu/anaconda3/envs/tfgpu/bin/python`. The job receipt preserves the
exact timeout/worker command, source hashes and GPU UUID. The service has a
separate 1,900-second outer wall limit; worker accounting and cleanup remain
inside 1,800 seconds. Memory growth and XLA are verified. At 17:10 UTC,
8,577.284164943063 seconds are settled consumed, 1,800 reserved and
76,022.71583505694 unreserved. The service stops after producing diagnostic
evidence; interpreting that evidence is required before any new tuning/P1 run.

## Exact replay result and repair audit

The exact carried-state replay completes at September 10, 17:19:17 UTC,
in 655.2250362480409 GPU-worker seconds. All 159 prefix samples, acceptance
decisions, log acceptance ratios and sampled target values match the original
bitwise (maximum sample residual zero). Chain 4's final proposed position,
momentum, potential and score are NaN; its initial momentum and carried accepted
score are finite. TFP maps the nonfinite kinetic/acceptance calculation to
negative infinity and rejects the proposal. The accepted state remains finite
and unchanged. This conclusively rejects the current candidate and rules out
checkpoint corruption or misreported health as explanations of this stop. It
does not yet distinguish a target implementation defect from insufficient
numerical tuning. The first invalid intermediate evaluation is not exposed.

An unlaunched tuning-r2 draft proposed a smaller epsilon grid. Its prelaunch
audit found stale strict recovery receipts/stream paths would be reused, and
the internal numerical cause remained unresolved. The draft is withdrawn:
no r2 tuner, reference, runtime or P1 worker launched. The wrapper and active
source migration are restored byte-for-byte to the validated serializer-only
closure (wrapper SHA-256 `60684794b77e27a9ea43ab6956859909df4c62ab116097b78d2417cd3772773c`).
The draft migration is archived separately. One draft compatibility regression
failed; its correction passed 56 CPU tests, but neither draft defines the active
route or a completed numerical repair. A future retuning repair must isolate
only strict, use fresh seed/checkpoint/completion namespaces, rerun its recovery
canaries, and justify the new grid before launch. Factor's valid evidence remains
reusable under its unchanged scope.

The new exact trace now preserves the original accepted gradient at transition
158. Use it, its stored target value, original state and seed to replay the
single failing transition with L=1, 2, 3 prefix diagnostics. Unlike the earlier
one-step diagnostic, replace the bootstrap gradient/value with these exact
carried values. Check the L=3 result against the exact trace before attributing
a prefix to the original failure. Do not substitute CPU gradients or adjust
epsilon. Reserve at most 900 worker seconds for one same-GPU/XLA diagnostic;
this inherited small-rung cap bounds compilation and four calls. It is not
retuning. Apply the same memory/headroom/source checks. Remaining budget before
launch is 77,167.4907988089 seconds, zero reserved, with 9,232.509201191104
settled consumed. No numerical source is changed.

## September 11 cached-gradient result and covariance diagnostic

`launches/cached-gradient-diagnostic-2dededa013/` completed in
36.94630356901325 GPU-worker seconds. The transition-158 preceding control
matches samples and acceptance exactly. The failing transition's L=1 endpoint
is valid. Its L=2 endpoint has finite parameters but one invalid placement
covariance: minimum eigenvalue -1.4256699903299468e-11, zero placement floors,
zero derivative-RHS nonfinites, and positive innovation minimum eigenvalue
0.688280339034077. The bridge intentionally returns NaN for an invalid row;
L=3 then has NaN position/momentum. The L=3 sampled states and failing chain
match the original; one unrelated finite acceptance ratio differs by about
4e-16. The full-prefix reproduction, not this latter comparison, supplies the
bitwise original-failure evidence. Budget is now 9,269.455504760117 consumed,
zero reserved, 77,130.54449523988 remaining seconds. The active serializer-only
closure passes 56 CPU tests, and 33 bundles/376 tensors verify.

### Next diagnostic contract and skeptical audit

Question: at the saved finite L=2 parameter batch, does the covariance lose
positive definiteness in the prediction, measurement-update subtraction, or
placement eigensolver? Trace the actual analytic-score recursion reached by
`batched_svd_sigma_point_tf.py`, not the distinct value/adjoint recursion.
Baseline: unchanged `tensorflow_eigh_strict`, float64, GPU/XLA, same batch of
four saved parameters and original observations/model. Instrument only a
diagnostic in-memory copy of the existing recursion with per-time tensor
outputs; save its generated source and hash. Do not edit a numerical module.
Check its values, scores and status against a separately compiled unchanged
call and against the saved invalid status. If instrumentation or fixed-point
compilation changes the result, record that limitation: the diagnostic may
explain a nearby computation, not replace the exact-prefix evidence.

Record incoming/predicted/updated covariances, innovation covariance, cross
covariance, gain, sigma-point residuals and original eigenvalue classification.
Compute spectra and subtraction residuals after saving these tensors. With the
existing nonnegative covariance weights, compare the subtraction with the
algebraic residual-sum form at the same saved operands:

`A_r = X_r - K Y_r`, `R_eff = S - sum_r w_r Y_r Y_r^T`,
`P_residual = sum_r w_r A_r A_r^T + K R_eff K^T`.

Expanding gives `P_pred - C K^T - K C^T + K S K^T`; it equals
`P_pred - K S K^T` when `K S = C`. Measure the solve residual and the
positive-semidefiniteness of `R_eff`; do not assume either numerically. This
is an explanatory local algebra check, not a replacement target, a derivative
repair, or admission of any candidate. A materially negative residual covariance,
negative weights, or mismatch of the actual model/operands invalidates the
simple cancellation explanation. No tolerance changes or eigenvalue clipping.

The audit passes for this diagnostic: exact comparator and arithmetic question
are explicit, no proxy is a promotion criterion, and trace-induced changes are
checked rather than ignored. Record all numerical constants from the original
source as inherited, not newly justified defaults. The failed endpoint is
debugging data only, not tuning or holdout admission data. A 900-worker-second
cap is the inherited convenience diagnostic bound, not a performance forecast;
allow one launch, charged to the existing ledger. Stop on cap, source mismatch,
invalid memory policy, inadequate headroom, or corrupt/missing required tensors.
Use the original non-display strict GPU if eligible (load <=40%, estimated
4096 MiB workload plus owner-required 5120 MiB headroom); do not switch devices
silently for an exact-comparator diagnostic. Launch via the existing supervised
accounting and headroom monitor, with memory growth before TensorFlow import.
Save fresh output under `launches/covariance-diagnostic-*/`, command/environment,
source hashes, data/parameter input hashes, no-new-randomness seed declaration,
GPU/memory/XLA provenance and settled wall time. Preserve prior diagnostic
sources beside their completed results before extending the supervisor.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Reject current strict runtime candidate | Exact original failure reproduced | Invalid placement covariance, nonfinite acceptance/energy | Why covariance becomes indefinite | Fixed-parameter covariance trace | Rejection of target or research direction |
| Continue bounded debugging | Integrity/source checks pass | No budget or artifact veto | Instrumentation may change rounding | Measure comparator agreement first | P1 clearance |

| Inference status | Current evidence |
| --- | --- |
| Hard veto screen | Strict's frozen runtime candidate fails; factor's two calls remain healthy |
| Statistically supported ranking | None |
| Descriptive-only differences | Timings and backend numerical differences |
| Default-readiness | Not established |
| Next evidence needed | Supported numerical repair or fresh tuning, then strict recovery/full health and downstream sequential validation |

### Covariance trace result: eigensolver discrepancy, not established indefiniteness

`launches/covariance-diagnostic-6a6c7c4a1b/` completed the numerical diagnostic
in 20.36170059099095 worker seconds. The supervisor rejected its manifest
because the diagnostic used status `completed_debugging_only` instead of the
required `completed`; worker exit was zero and both checksum-verified tensor
bundles plus summary/manifest exist. Preserve this engineering failure unchanged.
The script now uses the required status and a separate debugging evidence-role
field. Verify the existing tensors independently before interpreting them; no
scientific admission or rerun is implied by correcting future manifest output.

The instrumented and unchanged fixed-point value/score are bitwise identical,
and both reproduce the original invalid status/minimum eigenvalue exactly.
The classification occurs at filter time index 17, chain index 3 (zero-based).
However, a post-run eigensolve of the **same saved incoming matrix** returns a
positive minimum 4.6757110739367504e-14; the in-XLA classification returns
-1.4256699903299468e-11. Predicted and updated covariance spectra before that
point are positive in this post-run diagnostic. The residual-sum covariance
agrees with subtraction within 4.271744942646367e-15 at the preceding step,
with positive effective observation noise and a 1.13e-16 gain-solve residual.
Thus the earlier wording "invalid covariance" means **classified invalid by
the compiled eigensolver**, not yet a verified indefinite stored matrix. A
subtraction repair or smaller epsilon is not currently supported as the cause's
repair. The nonfinite and target-status vetoes remain valid operational stops.

Next smallest diagnostic: verify that one saved 80-by-80 matrix with independent
CPU TensorFlow/LAPACK and 60-decimal-digit mpmath Cholesky/eigenvalues. This is
an explicit tiny independent-reference exception (`CUDA_VISIBLE_DEVICES=-1`),
not training or a runtime fallback. NumPy/SciPy are confined to this diagnostic
reader. A 180-second wall limit and 60 decimal digits are convenience diagnostic
bounds; report residuals and precision limitations, not mathematical exactness
from a numerical reference. Preserve reference values/command/versions and
checkpoint checksum validation in a fresh `cpu-eigensystem-r1/` output.

If the reference confirms positive definiteness, compare the unchanged GPU/XLA
eigensolve with explicit XLA eigensolver tolerance on the **same fixed saved
batch**, never a recomputed filter path. Inspect actual compiler IR and installed
operator source/docs rather than assuming a library tolerance. A tighter
tolerance is only a diagnostic hypothesis until eigenpair/orthogonality/score,
runtime and downstream checks pass. Derive a proposed float64 tolerance from
machine precision and record the safety factor; do not widen the target's
1e-14 classification threshold. Keep the existing one-launch 900-second bound
for this next distinct diagnostic, settle it against the unchanged allocation,
and retain comparator failures. Current remaining balance: 77,110.18279464889
seconds; zero reserved. No target source changes or retuning have occurred.

The independent CPU reference confirms a positive minimum eigenvalue of
4.67714489018148805414485814275950147339710923154964473505419e-14
at 60 decimal digits. TensorFlow CPU and three LAPACK drivers agree on the
positive sign. The high-precision Cholesky reconstruction maximum error is
3.11e-61. These checks support an eigensolver classification error, not a truly
indefinite matrix. The r1 reference's maximum eigenvalue was misreported as zero
because mpmath does not support that negative-index lookup; r2 uses an explicit
last-row index and asserts positive ordered extrema. The minimum and Cholesky
checks are unchanged. Preserve both outputs; use r2 for reference provenance.

The next GPU operator comparison uses the original saved batch and four
diagnostic paths: TensorFlow default, explicit tolerance 1e-6, 1e-12, and
`8 * sys.float_info.epsilon` (1.7763568394002505e-15). The two decimal tolerances
are explanatory probes, not established library defaults; compiler IR will
settle the actual default. The factor eight is a convenience roundoff allowance
above binary64 machine epsilon, a hypothesis tested against the independent
matrix reference. The explicit 100-sweep maximum is a convenience termination
bound, not a convergence guarantee. Record eigenpair residuals, orthogonality,
reference error and actual HLO. A positive eigenvalue alone cannot admit a
replacement eigensolver or the full filter. No code/default/backend change
follows automatically; derivative/full-filter and runtime checks remain required.

### Explicit XLA tolerance does not repair the error

`launches/eigensystem-diagnostic-dc7911780f/` completes successfully, consuming
6.9189022650243714 worker seconds. Its saved TensorFlow HLO explicitly shows
`Eigh` configuration `1,1,14,0.000001`. The explicit 1e-6 path reproduces the
default eigenvalues exactly. The failing matrix's eigenpair Frobenius residual
is 9.97e-7. Tighter requested tolerances with 100 sweeps still return a negative
minimum (-1.566978374551501e-11) and residual 3.64e-8. Both small requested
tolerances appear as `0.000000` in HLO serialization and give the same result.
Changing the exposed tolerance is therefore a **failed repair hypothesis**,
not an implemented fix. No production source was changed. Remaining budget:
77,103.26389238387 seconds, zero reserved.

Next bounded diagnostic: test binary64 Jacobi residual refinement of the default
eigenvectors, entirely in a batched TensorFlow/XLA graph. This is an optional
diagnostic prototype, not a new runtime backend. Use the same four saved matrices
plus deterministic diagonal, repeated-eigenvalue and genuinely indefinite
controls. Start with the default orthogonal eigenvector matrix `V`, form
`B = V^T A V`, and apply disjoint plane rotations that diagonalize each 2-by-2
principal block of `B`, accumulating the same rotations into `V`. For block
entries `(a,b;b,d)`, let `delta=d-a`,
`t=2b/(delta+sign_plus(delta)*sqrt(delta^2+4b^2))`,
`c=1/sqrt(1+t^2)`, `s=t*c`; zero blocks use `t=0`. Scale numerator/denominator
before the square root to avoid overflow. Then `G=(c,s;-s,c)` gives
`G^T B G` with that off-diagonal entry zero. This follows by substituting the
rotation into the block: its off-diagonal is
`b(c^2-s^2)+(a-d)c*s`, which vanishes for that `t`. No eigenvalue clamp, target
tolerance relaxation, float32 cutoff or new regularizer is introduced.

Use a round-robin pairing covering each index pair once per sweep; a TensorFlow
`while_loop` has one native batched body, never a per-sample loop or pfor.
Three sweeps are a convenience diagnostic bound, not a convergence guarantee.
Measure eigenpair residual, orthogonality, smallest-eigenvalue/reference
agreement, covariance/square-root reconstruction and sign preservation for the
indefinite control. Compare timing descriptively; no throughput or sampler
claim from a tiny matrix benchmark. If accurate, the next necessary check is
the full filter and analytic derivative, followed by campaign-source migration,
fresh strict tuning/recovery/full health. If inaccurate, retain the failure and
investigate the solver rather than change target guards. Reserve at most 900
seconds for one new supervised original-GPU launch with unchanged headroom,
memory growth and source-integrity checks; preserve the prototype and outputs.

### Source-level cause and successful matrix-level refinement

The version-matched TensorFlow v2.20.0 XLA source is saved as
`runtime-health-diagnostic-20260911-r1/tensorflow-v2.20.0-eigh-expander.cc`
(download path: TensorFlow release tag, `third_party/xla/xla/hlo/transforms/expanders/eigh_expander.cc`).
At lines 129–133 its Jacobi rotation decides an off-diagonal is tiny using
`0.1f * std::numeric_limits<float>::epsilon()` **even for float64 inputs**.
At lines 255–257 it then zeros the corresponding off-diagonal. This is a
source-grounded explanation for the measured residual floor when tightening
the outer tolerance: the internal cutoff is still based on float32 precision.
This explains the observed false negative eigenvalue on this SPD matrix, not
every possible NaN or filter failure. The original matrix/compiled comparisons
and independent high-precision reference, not source inspection alone, establish
this failure. No installed library is modified or upgraded.

`launches/refinement-diagnostic-086ba0b540/` consumes 7.573121847992297 seconds
and completes. Three binary64 refinement sweeps return 4.677148228235773e-14
for the failed matrix, absolute difference 3.3380542853355316e-20 from the
60-digit reference. The four eigenpair Frobenius residuals are at most 7.63e-13,
versus 4.12e-6 for the default batch; orthogonality errors are at most 3.67e-15.
Diagonal, repeated-eigenvalue, near-singular-positive and indefinite controls
retain their exact eigenvalues; the negative control is not clipped. One sweep
is insufficient (about 1.22e-6 residual), so it is rejected as a repair setting.
These are deterministic matrix-level checks, not full-filter validation.

Next: an isolated full-filter prototype, never imported by the campaign. Clone
the existing core functions into a diagnostic namespace that substitutes only
the eigensolver with refinement; retain all model, placement/innovation guards,
analytic derivative expressions, observations and batch shape. Compare three
and four sweeps at the saved four model parameters; four is a convergence probe,
not a promoted default. Use a separate CPU non-XLA call of the unchanged target
as an independent full-filter reference. Compare likelihood and analytic scores
and status; use central differences at h=1e-4 and h=3e-5 for all four parameter
coordinates, without changing the four-row batch. These are convenience step
sizes probing finite-difference stability on the saved near-singular case.

Prototype continuation requires finite valid rows and coherent value/score
comparisons; relative score/finite-difference disagreement above 1e-4 (scaled by
max(1,abs(score))) is a repair trigger, not permission to promote. This diagnostic
threshold is a conservative comparison hypothesis around the existing nonlinear
test tolerances (1e-5 to 5e-5), not posterior accuracy. Compare both finite
difference steps and record all residuals even when smaller. Check three/four
sweep agreement before accepting the sweep count. Measure two warmed calls
only for an early runtime feasibility warning, not stochastic speed ranking.
Reserve 900 seconds for one supervised GPU launch, preserving sources and
durable intermediate results. CPU reference has a 180-second diagnostic cap.
If the full-filter or derivative checks fail, stop at that numerical repair
boundary; do not resume the HMC campaign. Source migration, fresh tuning,
recovery canaries and full runtime health still follow any successful repair.

The first CPU full-filter diagnostic computed its unchanged endpoint, then
failed before finite differences because a decimal point was included in a
checkpoint key. The diagnostic now encodes that point as `p`. During retry,
the agent mistakenly removed and reused that newly created CPU diagnostic
directory and redirected its log, contrary to the campaign's fresh-attempt
rule. This is an evidence-preservation error. The initial traceback is recorded
in the session tool output (`CheckpointError: invalid checkpoint key`, in
`DurableTensorCheckpoint._path`); it is not retained as an original on-disk log.
The current `cpu-full-filter-r1/` is the retry, not the original failed attempt.
No original campaign, failed strict runtime, prefix, covariance/eigensystem
bundle, source-migration file or budget evidence was removed. Do not repeat
this cleanup pattern: retain the retry and use fresh directories for all further
attempts. This loss limits the initial diagnostic's provenance, not the saved
matrix or original failure evidence used by the numerical investigation.

The strict full-filter prototype `launches/full-filter-diagnostic-928311a7e7/`
completes in 54.45377595600439 GPU-worker seconds. All four rows are valid with
zero roundoff repairs. Four-sweep likelihoods agree with the independent CPU
reference to 7.11e-15; maximum scaled analytic-score discrepancy is 6.09e-15.
Central-difference scaled errors are 9.48e-8 (h=1e-4) and 8.54e-9 (h=3e-5),
consistent with the CPU reference's step dependence. Three versus four sweeps
changes likelihood by about 1.33e-12 and scaled score by less than 8.87e-13.
Four-sweep warmed evaluations of this four-row batch take 1.331 and 1.330
seconds; these two calls are descriptive, not a campaign finish forecast.
No runtime source or tuning artifact has been changed. Remaining budget:
77,041.23699457987 seconds, zero reserved.

Source audit adds an important scope correction: the factor-cached sibling also
uses `tf.linalg.eigh` for classification and the covariance factor basis
(`experimental_batched_svd_sigma_point_tf.py`, `_batched_psd_eigh` and
`_tensorflow_strict_factor_cached_eigensystem`). Its healthy runtime calls
therefore do not exempt it from this primitive's precision defect. Preserve
those successful engineering/timing artifacts, but do not grant a repaired
shared primitive new-scope clearance from them. A strict-only repair without
auditing factor would be an incomplete numerical repair.

Run the same isolated fixed-parameter/derivative comparison for the factor-cached
path (unchanged and diagnostic-refined). Use the already saved CPU reference to
the same principal-root filter, retain factor caching rather than silently
substituting the strict derivative path, and use the same central differences,
diagnostic thresholds and four-row batch. This one additional 900-second-capped
GPU diagnostic answers whether both siblings need a shared primitive migration;
it is not a rerun of factor training or its successful runtime stream. Do not
reuse successful old-scope tuning or recovery receipts for either changed
numerical scope. Fresh two-arm public tuning can again be process-parallel after
the common repair is implemented, regression-tested and explicitly migrated.

### Shared primitive implementation audit

The factor-cached fixed-point diagnostic also reproduces the same false
negative with the unchanged eigensolver. Its refined four-sweep value equals
the strict prototype value; analytic scores match the CPU reference within
6.10e-15 scaled error, and finite differences within 9.48e-8 and 8.54e-9 at the
two declared steps. Warmed calls take 0.914/0.912 seconds descriptively. Worker
cost is 44.73133830202278 seconds. Thus **both numerical scopes need repair**;
factor's old healthy runtime cannot validate a newly repaired sibling. Budget
is 9,403.494343722152 consumed, zero reserved, 76,996.50565627785 remaining
seconds (21.3879 hours).

Implement a shared private binary64 eigen-refinement primitive and route the
TensorFlow strict/cached principal-root paths through it, including their
covariance classifier and Sylvester factor solve. Leave non-strict, compiled
custom-op and historical SVD paths unchanged. The primitive uses four sweeps
from the full-filter convergence comparison; this remains a measured local
repair setting, not a universal eigensolver-accuracy theorem. Support scalar
and odd dimensions with a round-robin idle participant, not padding that could
change the returned spectrum. Require static matrix dimensions and batch-native
TensorFlow operations, stable explicit signatures, bounded native while loops
and no sample mapping/pfor. Preserve real negative eigenvalues, repeated
eigenspaces, all target floors/thresholds and analytic derivative equations.

The implementation audit passes as a localized arithmetic repair. Before any
campaign launch it must pass focused scalar/even/odd/repeated/indefinite/ill-
conditioned eigenpair and reconstruction regressions, existing filter/score
tests, and a fresh GPU fixed-point comparison against the saved prototype.
Eigenpair/sign and target/score checks are validity vetoes; timing is descriptive
and subsequently enters the original affordability check. A fixed number of
sweeps cannot be silently trusted on arbitrary matrices; return nonfinite
failure sentinels if a measured normalized residual/orthogonality check exceeds
the declared binary64 bound. The residual bound is `64 * dimension * epsilon`
relative to matrix Frobenius norm (orthogonality uses the same dimension bound),
a conservative roundoff hypothesis, tested rather than a formal solver error
bound. It is not a replacement for the absolute SPD classification threshold.

An edited primitive creates a new numerical source closure. The old coordinator
must remain fail-closed until a recorded **numerical**, not serializer-only,
migration plus new chart/tuning/stream/canary namespaces is in place. Preserve
the current original-source diagnostic evidence before code edits. No old
readiness receipt can authorize the repaired closure. If tests fail, retain
the candidate patch as incomplete and do not start long tuning or P1.

The integrated repair passes 122 CPU regressions, including new scalar/even/
odd/repeated/near-singular/indefinite eigenpair checks, fail-closed residual and
nonfinite-input checks, and the full saved q=20 endpoint for both backends.
`eigen-repair-regression-r1.xml` and its log preserve the results. An intermediate
patch had one monkeypatch-signature compatibility failure (111 others passed)
and then a collection syntax error; both are repaired, not ignored. Odd sizes
now use an idle pairing participant; no artificial eigenvalue is padded. The
four-sweep residual/orthogonality checks retain the target's invalid-row guard.

Before campaign migration, run one bounded candidate-source GPU check of the
actual edited implementation (not the in-memory prototype): same stored matrix,
same four saved parameters, both strict and factor-cached backends. Compare to
the saved independent CPU reference and successful prototypes with the regression
tolerances (value 1e-10 absolute/1e-12 relative; score 1e-9 absolute/1e-10 relative).
Require zero row-class codes, finite outputs, one trace per explicit signature,
recorded XLA HLO, and negative-eigenvalue control preservation. No sampling or
tuning occurs. Reserve at most 900 worker seconds from the unchanged original
ledger, verify that only the intended core source differs from the previous
serializer-repaired closure, and record the **candidate** source hash separately
from the ledger's historical budget-binding hash. This diagnostic does not
activate or bypass the old coordinator's numerical-migration veto. Stop on any
source mismatch, failed comparison, memory/headroom failure or timeout.
