# Execution record: tempered reverse-KL transport ensemble

Date: 2026-08-28  
Last updated: 2026-09-02
Plan: `docs/plans/bayesfilter-ssl-lstm-q20-tempered-rkl-transport-ensemble-implementation-plan-2026-08-28.md`  
Status: `PHASES_0_TO_7_COMPLETE_PHASE_8_C0_TO_C5_COMPLETE_PHASE9A_LOCALIZED_REPAIR_COMPLETE_FULL_REPLAY_PENDING_PHASE9B_BLOCKED`

Governing master program:
`docs/plans/bayesfilter-ssl-lstm-q20-tempered-rkl-transport-ensemble-master-program-2026-09-02.md`.
This execution record is append-only evidence; it does not independently open
the next phase.

## Pre-execution audit

The plan was reread after the Claude adjudication and passes the repository's
skeptical audit. The research question, comparator ladder, promotion criteria,
vetoes, continuation vetoes, nonclaims, default provenance, and artifact
boundary are explicit. The earlier execution was bounded to routine
implementation, analytic/reference fixtures, and a non-claiming q=20 mechanics
smoke. The 2026-08-29 refresh adds the now-authorized Phase 8 campaign and
records its evidence separately below.

## Scope for this attempt

This execution implements and tests Phases 0--6, runs the bounded Phase 7
mechanics harness, and executes the authorized Phase 8 C0/C1 feasibility work
through the repository-default GPU launcher. It does not launch a retained-HMC
confirmation campaign. The C1 budget is now exhausted after three bounded GPU
attempts; every artifact is written below a fresh versioned campaign root
without overwriting prior evidence.

The shared-GPU preflight snapshot immediately before the authorized launch was
GPU 0: 337 MiB used, 31,893 MiB free, 32,760 MiB total, 1% utilization; the
existing `/usr/NX/bin/nxnode.bin` process accounted for 312 MiB. GPU 1 remained
occupied and is not exposed to the run. The smoke uses on-demand TensorFlow
memory growth and records the realized allocator peak; it remains an
implementation diagnostic, not capacity evidence for a larger campaign.

## Research intent ledger

- Question: can fresh-Gaussian reverse-KL lineages and frozen categorical chart
  kernels be represented without replay circularity or map averaging?
- Mechanisms: proper prior-likelihood bridge, transport bank, independent and
  optional joint reverse-KL objectives, bounded blind initialization preflight,
  fixed chart-kernel mixture, and bridge-generic replica exchange.
- Promotion: implementation identities and analytic invariance fixtures first;
  no sampler or scientific promotion from these tests.
- Vetoes: wrong target/bridge identity, nonfinite status, invalid inverse or
  determinant, state-dependent chart selection, stale scope, forbidden scalar
  or row-mapped training, missing XLA/memory policy, or artifact corruption.
- Nonclaims: no mode-discovery, posterior-correctness, convergence,
  high-dimensional scaling, or statistical-superiority claim.

## C3A execution and repair, 2026-08-30 to 2026-08-31

The pre-execution audit for
`docs/plans/bayesfilter-ssl-lstm-q20-phase8-c3-lineage-overlap-subplan-2026-08-30.md`
passed after the runner was repaired to require the completed C2 strict-backend
and B=8 parity receipts and to enforce a 4-GiB row allocator cap. The fresh
GPU0 attempt completed all eight rows in `1695.4819976739818` seconds with
status `PASS_PHASE8_C3_LINEAGE_OVERLAP`. Every row passed target/status,
checkpoint replay, proper-bridge overlap, and learned-map reliability checks;
peak row allocator usage was `1410923264` bytes.

The first C3 runner omitted covariance and sign-occupancy summaries named in
its own evidence contract. The between-phase repair subplan
`docs/plans/bayesfilter-ssl-lstm-q20-phase8-c3-diversity-repair-subplan-2026-08-31.md`
restored all beta-one checkpoints and evaluated fresh disjoint 256-row banks.
It passed in `4.9916944860015064` seconds. The repair showed no consistent
branching advantage: branching mean distance was lower for both compact-high
roots and higher for both compact-low roots; covariance distance was mixed;
sign occupancy remained close to 50/50. Therefore no branching arm or
architecture is nominated, and no statistical ranking is supported.

The terminal C3 result is
`docs/plans/bayesfilter-ssl-lstm-q20-phase8-c3-lineage-overlap-result-2026-08-31.md`.
The next action is a fresh audited C3B L5 ladder diagnostic with the same
pure-versus-single-restart comparison. Phase 9 confirmation, whitening
promotion, posterior claims, and HMC remain closed.

## Attempt log

| Attempt | Scope | Result | Repair/next action |
|---|---|---|---|
| 1 | Phase 7 CPU-debug harness, initial import | failed: `ModuleNotFoundError: bayesfilter` | Added repository-root path resolution; reran in a fresh directory. |
| 2 | Same harness, first optimizer update | failed: TensorFlow had no gradient for `SymmetricSylvester` | Added the reviewed analytic-score custom-gradient boundary; preserved the finite-program score path. |
| 3 | Same harness, full initial mechanics graph | timed out at the 300 s bound while redundantly tracing the q=20 graph | Reduced the smoke to one learned update, one direct chart transition, and one compact replica program; no prior artifact was overwritten. |
| 4 | Reduced harness, direct transformed state | failed: `[2,4]` versus `[4,4]` shape mismatch | Repaired the diagnostic offset to the static two-row state shape. |
| 5 | Reduced harness, replica program | failed: `StatelessCase` received an `int64` branch index | Cast the stateless categorical index to `int32` before `tf.switch_case`. |
| 6 | Reduced harness, complete CPU-debug mechanics | passed: `PASS_CPU_DEBUG_ONLY` | Preserved as the implementation diagnostic; it cannot close the required GPU/XLA exit. |
| GPU probe | Trusted Phase 7 admission | automatic idle probe failed: `no_idle_policy_permitted_gpu`; user-authorized GPU0 sharing exception recorded | Run only the bounded GPU0 smoke with memory growth and contention monitoring; preserve the idle-probe veto as scheduling evidence. |
| 7 | User-authorized GPU0 shared mechanics smoke | first attempt failed: XLA GPU does not support string `GatherV2` metadata; repaired by keeping compiled chart selection numeric and moving IDs to the eager boundary | Fresh attempt 8 passed with GPU0/XLA and memory growth; preserve attempt 7 as a localized graph repair record. |
| 8-C1 | Repository-default q20 cost pilot, validation 256 | attempt-02 timed out at 1,800 s after the B=8 beta-0 checkpoint | Localize the large static graph; no candidate conclusion. |
| 8-C1 repair | Target localization, B=8 then B=256 | attempt-03 completed both B=8 target calls, then timed out at B=256 beta=0 after 900 s | Bound validation bank and run the final feasibility retry. |
| 8-C1 final | Cost pilot, validation 8 | attempt-04 completed both B=8 charts and B=32 chart 0, then timed out at 2,700 s | Close C1; C2--C5 require a new budget or graph optimization. |

## Commands and artifacts

The output root is:

`docs/plans/artifacts/ssl-lstm-q20-tempered-rkl-transport-ensemble-2026-08-28/`

The active Phase 8 output root is:

`docs/plans/artifacts/ssl-lstm-q20-tempered-rkl-transport-ensemble-2026-08-29/phase8-calibration/`

The repository-default launcher is
`scripts/run_ssl_lstm_q20_tempered_rkl_phase8_gpu_default.sh`. It selects GPU 0
by default, sets memory growth before TensorFlow import, and does not call an
idle-GPU or Luna approval probe.

## Phase-close evidence

The implementation and focused fixtures closed Phases 0--6. The following
command was run with the repository TensorFlow environment and CPU explicitly
hidden:

```text
CUDA_VISIBLE_DEVICES=-1 TF_CPP_MIN_LOG_LEVEL=3 \
/home/ubuntu/anaconda3/envs/tfgpu/bin/python -m pytest -q --disable-warnings \
tests/test_tempered_transport_ensemble.py \
tests/test_tempered_lineage_transitions.py
```

Result: `20 passed, 8084 warnings in 10.67s`.

The route-policy check also passed:

```text
CUDA_VISIBLE_DEVICES=-1 TF_CPP_MIN_LOG_LEVEL=3 \
/home/ubuntu/anaconda3/envs/tfgpu/bin/python -m pytest -q \
tests/test_neutra_hmc_route_policy.py
```

Result: `6 passed`.

Additional compatibility checks were deliberately bounded. The q=20 protocol
file passed `11 passed in 1.83s`, and three targeted predictive-contract tests
passed in `2.15s`. The full complexity-target and predictive suites each
exceeded their 120-second tracing bound without emitting a test failure; they
are recorded as incomplete diagnostics, not passes. An earlier combined
compatibility invocation was interrupted after the same tracing stall. These
timeouts do not alter the focused Phase 0--6 decision, but a future GPU-enabled
campaign should rerun the full suites under its own measured budget.

An analytic TensorFlow/XLA reference check (with CUDA hidden) constructed fresh
preflighted charts and ran one independent and one joint reverse-KL update:
`CPU_XLA_TRAINER_PASS True True 12`, with one compiled trace for each trainer.
A companion stale-preflight probe was rejected with
`transport changed after initialization preflight`; this is the intended
identity guard, not a training failure.

Touched TensorFlow modules and the Phase 7 harness passed `py_compile`. A
static scan of the new route modules found no `tf.map_fn`,
`tf.vectorized_map`, `GradientTape.jacobian`,
`GradientTape.batch_jacobian`, or `pfor` use.

The successful bounded smoke was:

```text
CUDA_VISIBLE_DEVICES=-1 TF_CPP_MIN_LOG_LEVEL=3 timeout 300s \
/home/ubuntu/anaconda3/envs/tfgpu/bin/python \
docs/benchmarks/run_ssl_lstm_q20_tempered_rkl_transport_ensemble_phase7_smoke_2026_08_28.py \
  --cpu-debug \
  --output-dir docs/plans/artifacts/ssl-lstm-q20-tempered-rkl-transport-ensemble-2026-08-28/phase7-mechanics-smoke/attempt-06-cpu-debug \
  --principal-sqrt-backend compiled_custom_op
```

It produced `PASS_CPU_DEBUG_ONLY` in
`phase7-mechanics-smoke/attempt-06-cpu-debug/`. The manifest records q=20,
parameter dimension 4, internal filter state dimension 60, static training
batch 4, two components (`chart-0`, `chart-1`), two positive temperatures
(`beta=0.5` and `beta=1` in addition to the prior endpoint), finite endpoint
values and scores, a passing v2 learned-map reliability receipt, a finite
direct fixed-chart transition, and a passing proper-replica health record with
two swap proposals. The output is explicitly non-claim-bearing: it does not
establish GPU placement, XLA readiness, whitening, convergence, posterior
correctness, mode discovery, scaling, or default readiness.

The source-bound properness receipt in that manifest proves the normalized
bridge law and records the corresponding unnormalized runtime-kernel bound
`bar Z_beta <= A_prior max(1,M)`. The active plan now states both conventions;
the artifact's plan hash remains the immutable launch snapshot and is not
silently rewritten after the documentation repair.

The trusted GPU admission command was:

```text
/home/ubuntu/.codex/bin/codex-gpu-probe
```

It returned:

```json
{"error":"no_idle_policy_permitted_gpu","error_type":"ProbeError","framework":"tensorflow","requested_gpu":"auto","schema":"claudecodex.codex_gpu_probe.v1","scientific_authority":false,"status":"FAILED","timestamp_utc":"2026-08-28T16:35:05.199828Z","trusted_execution_required":true}
```

The structured copy is
`phase7-mechanics-smoke/gpu-probe-2026-08-29.json`. This is an execution
boundary veto, not evidence that the GPU, driver, TensorFlow installation, or
algorithm is defective.

The admission probe was repeated at `2026-08-28T17:10:29.500342Z` under the
same trusted command and returned the same `no_idle_policy_permitted_gpu`
failure. Its separate immutable record is
`phase7-mechanics-smoke/gpu-probe-2026-08-29-final.json`; no GPU/XLA workload
was launched around either veto. A further same-contract probe at
`2026-08-28T17:17:26.412746Z` returned the same error; its record is
`phase7-mechanics-smoke/gpu-probe-2026-08-29-latest.json`.
An explicit `--gpu 0` probe at `2026-08-28T17:18:55.716847Z` was also rejected
as `requested_gpu_compute_busy:0`; its record is
`phase7-mechanics-smoke/gpu-probe-2026-08-29-gpu0.json`.

Because the user explicitly authorized GPU0 sharing, the bounded GPU smoke was
then launched with `CUDA_VISIBLE_DEVICES=0` and
`TF_FORCE_GPU_ALLOW_GROWTH=true`. Attempt 7 reached GPU/XLA initialization but
failed at the replica graph's string chart-ID gather; the failure is preserved
at `phase7-mechanics-smoke/attempt-07-gpu0-shared/failure.json`. The localized
repair removed string metadata from the compiled tensor path while retaining
numeric chart indices. A fresh attempt 8 then completed:

```text
TF_FORCE_GPU_ALLOW_GROWTH=true CUDA_VISIBLE_DEVICES=0 TF_CPP_MIN_LOG_LEVEL=3 \
timeout 300s /home/ubuntu/anaconda3/envs/tfgpu/bin/python \
docs/benchmarks/run_ssl_lstm_q20_tempered_rkl_transport_ensemble_phase7_smoke_2026_08_28.py \
  --output-dir docs/plans/artifacts/ssl-lstm-q20-tempered-rkl-transport-ensemble-2026-08-28/phase7-mechanics-smoke/attempt-08-gpu0-shared \
  --principal-sqrt-backend compiled_custom_op
```

Result: `PASS_PHASE7_GPU_MECHANICS_SMOKE`. TensorFlow 2.20.0 and TFP 0.25.0
used one logical RTX 4080 SUPER GPU with XLA and TF32 enabled. All endpoint,
transition, and replica states were finite; the direct chart transition and
beta-one stream were on GPU0; the reliability screen passed; and both swap
proposals were accepted. TensorFlow's allocator reported current 366,336 bytes
and peak 67,494,144 bytes (about 64.4 MiB peak tensor allocation). The largest
driver-level process snapshot during the run was 496 MiB, leaving 31,392 MiB
free. These are capacity measurements for this bounded smoke only, not a
forecast for the long ensemble campaign.

## Phase 8 refresh and C0 execution, 2026-08-29

The user authorized continuation under the previously supplied 18-hour
budget. The active bounded campaign is now defined by
`bayesfilter-ssl-lstm-q20-tempered-rkl-transport-ensemble-phase8-calibration-subplan-2026-08-29.md`.
It freezes the baseline ladder, architecture and batch hypotheses, immutable
temperature checkpoints, non-tautological pullback-density and pullback-score
diagnostics, four-chain Phase 9 boundary, versioned output root, attempt cap,
and stop conditions. Its skeptical audit found and repaired three material
pre-launch risks: mutable maps could overwrite earlier temperature charts,
generated Gaussian inputs could be mislabeled as whitening evidence, and the
checkpoint identity did not initially bind data/backend/XLA/seed-bank scope.

C0 ran with GPU devices intentionally hidden. The immutable JUnit receipts are
under `phase8-calibration/attempt-00-compatibility/`:

| Suite | Result | JUnit time (s) | SHA-256 |
|---|---:|---:|---|
| Full complexity target | 9 passed | 260.340 | `7a8a4cf95ba3b3791ee769c85c153a3155ca164e2acec3bf5c0e9072c786cf54` |
| Full predictive target | 38 passed | 929.866 | `cb9c071098893839b3b0d8a61d56ce48a5975b032d171cd9c411f5ca15c067ed` |
| Focused tempered/route protocol | 39 passed | 11.678 | `5fb51816ac42364127fd3b9e1d90c6018a6feb7ec1bb4e559859c7e7817e0655` |

The full receipt time is 1,201.884 seconds, within the 3,600-second C0
allocation. After the checkpoint schema was strengthened, the affected
transport suite passed again (`13 passed` in 5.12 seconds), syntax and diff
checks passed, and the cost-pilot harness independently verified all three
JUnit receipts and the forbidden pfor/row-mapping scan. These are engineering
checks only; no NeuTra optimizer update ran on CPU and no whitening, posterior,
or sampler conclusion follows.

Neither historical trusted GPU admission request for C1 received a scientific or resource
verdict: the approval reviewer returned HTTP 503 before evaluating each one.
The second response, at `2026-08-29T17:53:51+08:00`, had request ID
`81c59d6b-3dba-4127-bb59-4048319c3862` and is preserved in
`phase8-calibration/gpu-admission-denial-02.json`. No probe process was created,
so these denials consumed no GPU attempt or GPU budget. The subsequent
repository-default launch reached GPU 0 without an idle probe or a new Luna
reviewer gate, then exited with code `124` at its 1,800-second cap after the
`B=8`, chart-0, `beta=0` checkpoint. The immutable partial output and timeout
classification are under
`phase8-calibration/attempt-02-default-gpu/timeout.json`. This is a bounded
compile/graph-cost repair trigger, not a candidate or target failure. The next
action is the fresh `target-localization` mode with a 900-second cap, followed
by one cost-pilot retry within the remaining C1 allocation.

The localization attempt (`attempt-03-target-localization`) completed the
`B=8` beta-0 and beta-0.5 target calls (finite values and scores), then timed
out with exit code `124` at the `B=256`, beta-0 call after 900 seconds. Its
start/done markers and timeout record are preserved in that attempt directory.
The result narrows the bottleneck to the large static validation graph; it does
not establish a target or candidate failure. The final C1 repair is a fresh
cost-pilot launch with `validation_size=8` and a 2,700-second cap. The reduced
bank is feasibility evidence only; the frozen 256-row bank remains required for
any claim-bearing diagnostic.

On 2026-08-29 the owner clarified that GPU is the repository default and should
not depend on a one-by-one idle-probe approval. The active route remains
`scripts/run_ssl_lstm_q20_tempered_rkl_phase8_gpu_default.sh`. It sets one
visible GPU and memory growth before TensorFlow import, writes a fresh versioned
directory, and invokes no idle or approval probe. A narrow persistent command
allowance is preferred over disabling approval for unrelated commands. The
reusable boundary and service-configuration distinction are recorded in
`docs/plans/bayesfilter-gpu-default-execution-boundary-2026-08-29.md`.

## Phase decision table

| Phase | Primary criterion | Veto status | Main uncertainty | Decision / next action | Nonclaim |
|---|---|---|---|---|---|
| 0 | Endpoint parity, exact score decomposition, source-bound properness receipt | pass | Receipt is tied to the current q=20 source facts | Closed; refresh if target or normalizer changes | No discovery or convergence |
| 1 | Stable categorical bank, log-density and cross-density fixtures | pass | Finite fixture size | Closed; measure larger-bank cost in Phase 8A | No density-identification theorem beyond the implemented formulas |
| 2 | Independent/joint reverse-KL value and gradient identities | pass | Joint arm remains optional and cost-limited | Closed; do not promote joint arm without measured envelope | No posterior-mass interpretation of alpha |
| 3 | Blind fixed-bank initialization and lineage semantics | pass | Finite initialization bank cannot prove coverage | Closed; use fresh scope-specific bank in calibration | No exhaustive mode discovery |
| 4 | Fixed-chart transformed HMC mechanics and learned-map reliability | pass | q=20 smoke uses one learned update and one exact affine chart | Closed mechanically; apply full screen before tuning | No HMC readiness |
| 5 | Proper bridge swaps and shared sequential transition abstraction | pass | CPU smoke uses identity within-temperature kernels for the compact replica program; direct chart transition is tested separately | Closed mechanically; GPU/XLA and full controller validation remain | No posterior validation |
| 6 | Analytic end-to-end invariance and counterexample fixtures | pass | Finite fixtures are not a stochastic comparison | Closed | No statistical ranking |
| 7 | Trusted GPU/XLA smoke with memory growth | pass under user-authorized GPU0 sharing; automatic idle probe remains a scheduling veto | Long-run contention and larger-campaign memory are unmeasured | Closed mechanically; proceed only to a fresh Phase 8 subplan, not directly to confirmation | Smoke is not whitening, convergence, discovery, or scaling evidence |
| 8 | Frozen calibration/search protocols and candidate selection | C0 passed; attempts 02--04 exhausted the C1 feasibility allocation by timeout | q20 static target/reliability graph cost | Pause at the continuation veto; require a new budget or reviewed graph optimization | No candidate promotion |

## Inference-status table

| Evidence class | Status | Interpretation |
|---|---|---|
| Hard veto screen | Implementation fixtures and GPU0 mechanics pass; C1 timed out before a complete receipt | No numerical hard veto observed; timeout invalidates C1 feasibility evidence only |
| Statistically supported ranking | None | No multi-seed or confirmation comparison was run |
| Descriptive-only differences | Smoke loss, gradient norm, swap count, and movement | Diagnostics only; they cannot rank methods |
| Default-readiness | Not established | No serious training, tuning, or retained posterior stream exists |
| Next evidence needed | New reviewed graph-optimization/cost plan, then a complete C1 receipt before C2--C5 or Phase 9 | Preserve untouched Phase 9 streams; use the repository-default GPU launcher |

## Between-phase repair and stop decision

The repair loop classified Attempts 1--5 as localized harness or graph defects;
each was repaired without changing the target, bridge, correction, hardware
class, or evidence contract. Attempt 6 passed the CPU-debug exception. Attempt 7
exposed an XLA-incompatible string metadata gather; the numeric-index repair was
focused and the fresh Attempt 8 GPU0 smoke passed. The automatic idle probe
remains scheduling evidence, while the user's explicit shared-GPU authorization
made this bounded smoke admissible. The 2026-08-29 Phase 8 subplan now freezes
the fresh compute budget, component count, temperature ladder,
architecture/optimizer search, joint-arm feasibility, ESS/MCSE and
region-travel targets, seeds, and attempt cap. C0 passed. C1 attempt-02 reached
the trusted GPU boundary and timed out after its beta-0 checkpoint; attempt-03
localized a timeout at the B=256 target call; and attempt-04 exhausted the
remaining small-bank feasibility budget after partial B=32 progress. No
confirmation stream has been consumed. The continuation veto is now active.

After the service switch, the independent UUID-pinned TensorFlow GPU probe
passed on GPU 0 with memory growth verified before logical-device creation.
The authorized graph-repair subplan then ran two fresh attempts: the first
timed out on an unnecessarily direct q=20 `B=32` parity graph; the repaired
`B=8`-only route completed a finite 32-row prefix in `192.694` seconds but
timed out before the remaining 224 rows. The terminal repair record is
`docs/plans/bayesfilter-ssl-lstm-q20-phase8-c1-graph-repair-result-2026-08-30.md`.
The gateway boundary is therefore clear, but C2--C5 remain closed until a new
graph-level optimization/execution plan and budget are reviewed.

The 2026-08-30 strict-backend repair passed the q=20 value/score/status parity
screen and the bounded GPU/XLA trainer localization. The fresh cost-rescue
subplan is now executing its one full-256 K=2 pilot. This is a new diagnostic
allocation; it does not reopen the exhausted compiled-backend C1 attempt or
consume Phase 9 confirmation streams.

The strict-backend cost-rescue pilot then passed its complete hard screen in
`261.52175762609113` seconds and selected B=32. The result is recorded at
`docs/plans/bayesfilter-ssl-lstm-q20-phase8-c1-strict-backend-cost-result-2026-08-30.md`.
This is a feasibility closeout, not a candidate or whitening result. The next
execution step is a refreshed, audited C2 calibration subplan with strict
backend identity and a larger-batch parity check; no confirmation stream has
been consumed.

C2 then passed on the strict route after the B=8 parity prerequisite. The
repaired attempt completed all eight rows (four architecture/rate hypotheses
times two independent initialization roots) in `1074.309582018992` seconds.
Every row was finite and replayable, all four architecture groups passed
learned-map reliability, and all within-row paired held-out reverse-KL
intervals were negative. The receipt and interpretation are in
`docs/plans/bayesfilter-ssl-lstm-q20-phase8-c2-strict-calibration-result-2026-08-30.md`.
The score residuals remain large, so this is candidate viability only, not
whitening or HMC evidence. Compact-high and compact-low remain the two
calibration representatives without a statistically supported ranking. The
next execution step is a fresh C3 branching/temperature-overlap subplan;
Phase 9 confirmation remains closed.

## Post-run red-team

The strongest alternative explanation is that the compact CPU graph passes
because it exercises only one optimizer update, one direct transition, and a
small identity-kernel replica program; it may still fail under GPU placement,
XLA compilation, longer chains, or target-specific tuning. The result would be
overturned as an implementation readiness signal by a trusted GPU smoke with a
nonfinite target/map/transition, a source-hash or identity mismatch, or a
failure of the same focused fixtures after the environment is admitted. The
weakest evidence is the finite smoke loss, gradient norm, and two accepted
swaps; none bears on whitening, mode coverage, posterior convergence, or
statistical superiority.

## Current execution state (historical snapshot; superseded by the Phase 9A closeout below)

The program is paused in Phase 8 at a real continuation blocker. C0 is complete,
the repository-default GPU launcher reached GPU 0 without an idle-probe or
per-run Luna gate, and attempts 02--04 exhausted the C1 allocation by bounded
timeouts. Partial checkpoints are feasibility evidence only; no claim-bearing
q=20 candidate or retained-HMC stream exists. C2--C5 cannot launch without a
new authorized budget or a reviewed optimization of the q20 target graph.

The preceding snapshot is superseded by the strict-backend C2 and C3 refresh
recorded above. As of 2026-08-31, C3A and its diversity repair are complete:
all eight C3 rows and all eight repair rows passed their hard screens, but no
branching arm was nominated because its descriptive diversity was not
consistently greater than pure continuation. The active continuation is C3B,
the audited L5 ladder diagnostic; Phase 9 confirmation and HMC remain closed.

## C3B L5 execution and closeout, 2026-08-31

The first C3B launch failed in the harness after completing the beta-0.25
component-0 checkpoints for all eight rows. The controller's `parent_index` is
a component index within the immediately preceding beta slice, but the runner
used it as a beta-list index. This was a localized indexing defect; no
candidate evidence was produced. The failed attempt is preserved under
`.../c3b-l5-ladder/attempt-01/`.

After the focused repair and `py_compile`/diff checks, the fresh GPU0/XLA
attempt completed all eight rows in 3,080.149 seconds with status
`PASS_PHASE8_C3B_L5_LADDER`. The strict target and bridge identities, static
batch, memory-growth policy, learned-map reliability, checkpoint replay, and
4-GiB row cap all passed. The full result is
`docs/plans/bayesfilter-ssl-lstm-q20-phase8-c3b-l5-ladder-result-2026-08-31.md`.

L5's minimum and median adjacent-acceptance diagnostics (0.3856 and 0.5101)
were descriptively above the C3A L3 values (0.2350 and 0.3482). The paired
branching mean-distance contrast was positive in all four architecture/root
rows, but covariance and sign-occupancy contrasts were mixed. These are finite
map-bank diagnostics, not mixing or mode evidence; no arm or architecture is
promoted. The C3B source dependency omission was repaired by the supplemental
metadata receipt at
`docs/plans/artifacts/ssl-lstm-q20-tempered-rkl-transport-ensemble-2026-08-30/c3b-l5-ladder/provenance-repair-2026-08-31/attempt-01/provenance_manifest.json`.

The next active boundary is a fresh, audited C4A feasibility subplan for the
optional K=4 joint mixture-RKL arm. Phase 9 tuning, retained sampling, HMC,
whitening promotion, and posterior claims remain closed.

## C4A joint-feasibility execution and closeout, 2026-08-31

The C4A subplan and runner passed the focused static checks and the 27-test
tempered transport/lineage suite before launch. The fresh GPU0/XLA attempt
completed in 454.691 seconds with status
`PASS_PHASE8_C4A_JOINT_FEASIBILITY`. The K=4 joint trainer made one target
call over 128 rows and exactly 512 cross-density work units per update; all
eight pilot updates were finite. The 16-update resource forecast was 117.723
seconds and the allocator peak was 3,401,816,064 bytes, below the 4-GiB cap.

The independent and joint arms were restored from identical beta-0.5 start
checkpoints. Both passed learned-map reliability and replay; alpha stayed
positive and nearly uniform. The joint held-out objective was lower on the
single fresh bank, but this is an eight-update, one-root descriptive result,
not evidence of superiority or whitening. The complete result is
`docs/plans/bayesfilter-ssl-lstm-q20-phase8-c4a-joint-feasibility-result-2026-08-31.md`.

C4A closes as a resource and implementation pass without promotion. The next
active boundary is C4B, a fresh-root/architecture replication of the K=4 joint
arm. C5 freeze, whitening promotion, Phase 9 tuning, retained sampling, HMC,
and posterior claims remain closed.

## C4B joint-replication execution and closeout, 2026-08-31

The C4B subplan survived the skeptical audit after the runner was repaired to
include itself in the forbidden-route-token scan. `py_compile`, `git diff
--check`, the protocol assertion, and the focused CPU reference suite all
passed (`85 passed`). The bounded GPU0/XLA command then completed both fresh
architecture/root rows in `876.4273084410233` seconds with status
`PASS_PHASE8_C4B_JOINT_REPLICATION`.

Every row passed exact K-squared-B work accounting, finite target/status
checks, checkpoint replay, learned-map reliability, positive normalized alpha,
distinct parameter-state hashes, memory-growth/XLA checks, the 4-GiB allocator
cap, and the 3,600-second forecast cap. The largest 16-update forecast was
`118.20146459259558` seconds and the largest allocator peak was
`3402234624` bytes. A bounded retracing warning occurred when row-local trainer
objects were constructed; the fixed signatures and numerical receipts remained
valid, but the warning is retained as performance debt.

The nominal joint-minus-independent held-out objective differences were
`+21.32596951341` for compact-high and `-6.0386412803620715` for compact-low.
The arms used separate held-out banks, so these are unpaired finite-bank
observations and cannot rank the objectives. Pullback score residuals stayed
large. The terminal result is
`docs/plans/bayesfilter-ssl-lstm-q20-phase8-c4b-joint-replication-result-2026-08-31.md`.

C4B closes as an implementation/resource replication with no arm promotion.
The next active boundary is the metadata-only C5 freeze refresh. Whitening,
Phase 9, retained sampling, HMC, posterior, and default gates remain closed.

## C5 freeze execution and closeout, 2026-08-31

The metadata-only C5 evaluator first completed in `attempt-01`; because the
subplan was then finalized with its closed status, that receipt is retained as
pre-closeout provenance evidence only. The unchanged evaluator was rerun in
fresh `attempt-02` and passed in `0.016574528999626637` seconds with a terminal
subplan-hash receipt. It selected the K=2 compact-high L3 pure-continuation
protocol with fixed uniform gamma and marked the optional K=4 joint arm
`NOT_RETAINED_FOR_PHASE9`.

The terminal result is
`docs/plans/bayesfilter-ssl-lstm-q20-phase8-c5-freeze-result-2026-08-31.md`.
Phase 9 tuning and sequential posterior validation are still closed until a
new subplan survives the skeptical pre-execution audit.

## Phase 9A fresh-map tuning preflight closeout, 2026-08-31

The Phase 9A subplan and runner were repaired and audited before execution.
Attempts 01-03 localized the checkpoint identity/import path and the first
epsilon cap.  Attempt 03 passed the chart-0/beta-0 handoff with selected
epsilon `0.810010`, acceptance `0.859967`, one trace per reusable HMC graph,
and `1402670592` bytes (about 1.31 GiB) peak allocation.  Complete attempt 04 passed all chart-0 scopes
but failed chart-1/beta-0 at the cap-1.0 repair boundary.  Complete attempt 05
repeated that failure under the final cap 2.0: the tuner observed acceptance
`0.998950` at epsilon `0.628978`, `0.939618` at epsilon `1.205189`, then
requested `2.410379` and emitted a no-viable-candidate veto.

The terminal result is
`docs/plans/bayesfilter-ssl-lstm-q20-phase9a-fresh-tuning-preflight-result-2026-08-31.md`.
This closes Phase 9A with a chart-specific tuning continuation veto.  The
proper bridge, strict backend, fresh-map reliability, and completed chart-0
scope mechanics remain valid diagnostic evidence; the six-scope handoff and
shared transition prerequisites were not met.  Phase 9B, retained sampling,
whitening, HMC, posterior, and default-readiness work remain blocked.  Attempts
04 and 05 failed before final manifest writing, so their tuner JSON and
failure records are authoritative but lack exact process-level wall and
allocator telemetry; future repair runners must write progress/failure
manifests before scope execution.

## Localized chart-1/beta-0 program repair, 2026-09-01

The repaired runner and GPU wrapper were exercised under the new scoped
subplan. Attempts 01 and 02 were preserved as resource timeouts; attempt-03
completed but exposed a result-note provenance defect; attempt-04 completed as
a deterministic provenance replay using the same v3 seeds. Fresh attempt-05
completed in `494.5689085649792` seconds. It measured all four declared joint
grid pairs for scope index 3, selected `(epsilon=0.55,L=3)` as a mechanics
candidate, and passed finite/movement/held-out checks. The manifest records
GPU0 memory growth before TensorFlow initialization and a `1402670592`-byte
peak allocator reading.

The result is localized mechanics evidence only. It does not complete the six
scope Phase 9A prerequisite and does not open Phase 9B. See
`docs/plans/bayesfilter-ssl-lstm-q20-phase9a-chart1-beta0-program-repair-result-2026-09-01.md`
and the reset memo with the next-plan boundary.
