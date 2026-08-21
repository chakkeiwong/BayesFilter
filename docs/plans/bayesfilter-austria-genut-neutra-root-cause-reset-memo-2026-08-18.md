# Austria GenUT NeuTra Restart Memo And Claude Code Handoff

Date: 2026-08-18

Audience: Claude Code or another fresh execution agent taking over the Austria
GenUT value/score repair campaign.

Purpose: preserve the complete scientific and engineering state, identify
which artifacts apply to which source revision, record the current trusted-GPU
launch blocker, and give the next agent an exact bounded resume procedure.

## Authoritative Current Status

> **Superseded 2026-08-18 (later the same day):** the bounded resume below was
> executed by Claude Code under its own trusted GPU boundary. GPU eager
> (attempt07) PASSED all gates; GPU graph (attempt08) FAILED within-mode
> `T=20` value identity (gap `0.56207275`), a compiler-mode confirmation
> failure. XLA and invariants were not run per the stop rule. See the updated
> execution result and checkpoint documents for the authoritative status:
> `GPU_EAGER_PASS_GPU_GRAPH_WITHIN_MODE_IDENTITY_FAIL_XLA_AND_INVARIANTS_NOT_RUN`.
> Do not re-run the resume procedure from this memo without a fresh reviewed
> localization plan.

`CPU_REPAIR_AND_DERIVATIVE_AUTHORITY_COMPLETE_CURRENT_SOURCE_GPU_CONFIRMATION_BLOCKED_BY_APPROVAL_REVIEWER_404`

The shared-primal repair and independent CPU forward-autodiff authority are
complete for the current source. A complete older GPU eager/graph/XLA artifact
exists, but it is stale because its recorded source hash differs from the
current source. The current source has not completed the required trusted GPU
endpoint confirmation.

The latest trusted NVIDIA occupancy probe succeeded. The subsequent current-
source TensorFlow eager launch was rejected before process creation because the
Codex approval reviewer attempted to use an unavailable model and returned
`404 Not Found`. This is an infrastructure/permission failure, not a CUDA,
TensorFlow, or scientific endpoint failure. No new process artifact exists.

Austria remains blocked from NeuTra training and HMC. Do not promote a tuning
artifact, claim dual-cap correctness, claim posterior correctness, or launch
NeuTra/HMC from this memo. The wired Austria callable is
`batch_diagonal_candidate`, not the promoted dual-cap route.

This memo supersedes older notes only for the current infrastructure status and
resume instructions. It does not upgrade or invalidate their preserved
scientific artifacts except where source-hash staleness is stated explicitly.

## Required Read Order

Before running or editing anything, Claude Code should read:

1. The repository `AGENTS.md`, especially the trusted GPU/CUDA, TensorFlow GPU
   memory-growth, XLA-default, NumPy diagnostic-only, NeuTra batching, and LEDH
   tuning policies.
2. This memo in full.
3. The execution result:
   `docs/plans/bayesfilter-austria-genut-neutra-root-cause-execution-result-2026-08-18.md`.
4. The execution checkpoint:
   `docs/plans/bayesfilter-austria-genut-neutra-root-cause-execution-checkpoint-2026-08-18.md`.
5. The reviewed plan:
   `docs/plans/bayesfilter-austria-genut-neutra-root-cause-hypotheses-fable-handoff-2026-08-17.md`.
6. The independent second review:
   `docs/plans/bayesfilter-austria-genut-neutra-root-cause-hypotheses-fable-second-review-reply-2026-08-17.md`.
7. The bounded runner and focused tests:
   `docs/benchmarks/run_genut_austria_endpoint_root_cause_20260817.py`,
   `tests/highdim/test_genut_batch_primal_parity.py`, and
   `tests/highdim/test_cubature_genut_batch.py`.

The older execution result records an HTTP 502 approval failure. That statement
was accurate for its attempt. The latest failure is the approval-reviewer 404
recorded here. Neither response is evidence that the GPU stack is broken.

## Scope And Authority

The user requested this handoff so Claude Code can continue the bounded GPU
confirmation. That authorizes the already reviewed local confirmation sequence
below, subject to Claude's own platform permissions. It does not authorize:

- changing the scientific target, model, frozen tensors, route, dtype, TF32,
  correction controls, or acceptance criteria;
- broad package or environment mutation;
- overwriting an existing artifact;
- reusing stale tuning;
- launching NeuTra training or HMC;
- promoting the diagonal route as dual-cap or as a repository default; or
- editing production source during the confirmation run.

If current-source confirmation fails, preserve the artifact and classify the
failure before proposing a new source edit. Do not repair and rerun inside the
same output directory.

## Research Intent Ledger

Main question: does the repaired public score execute the derivative of the
same finite `batch_finite_value` scalar on the repository GPU target, across
eager, graph, and XLA endpoint modes, for the frozen Austria scope?

Candidate under test: the current shared-primal implementation in
`bayesfilter/highdim/cubature_genut_batch_tf.py`, whose SHA-256 at handoff is
`ae8cbfb486fc90a4a38257702cf025569ca7651bb058558d341ad602cf8a976e`.

Expected failure mode: FP32/TF32 reduction or compiler layout may expose
mode-sensitive arithmetic, a remaining duplicated primal, a nonfinite tangent,
or a source/input identity mismatch.

Primary promotion criterion for this confirmation: within every execution
mode, the value-only endpoint and the value carried by the score endpoint are
exactly equal for both `T=20` with four correction steps and `T=1` with zero
correction steps. The values and scores must be finite and both endpoints must
report `program_valid=true`.

Promotion vetoes:

- any frozen target, adapter, or tensor hash mismatch;
- any launched artifact whose source hash differs from the pre-launch hash;
- `status` other than `COMPLETE`;
- wrong TensorFlow build, device, dtype, TF32 state, or unverified memory
  growth;
- nonfinite value or score;
- either endpoint reporting invalid status;
- non-exact within-mode value identity; or
- cross-mode drift of the `T=20` value scalar under the current checkpoint's
  confirmation rule.

Continuation vetoes: frozen identity failure, a nonfinite or invalid eager
result, or failed within-mode eager value identity. If one fires, do not run
graph, XLA, or invariants.

Repair trigger: a valid artifact tied to the current source that fails a
confirmation gate. Preserve it, identify whether the failure is source,
compiler-mode, numerical, harness, or environment related, and write a bounded
repair proposal. Do not invent a tolerance after seeing the result.

Explanatory diagnostics only: runtime, allocator growth, condition numbers,
coefficient magnitudes, central finite-difference regression quality, and the
magnitude of historical CPU/GPU or old-source mode differences. Cross-mode
score differences must be recorded; because no reviewed nonzero score
tolerance exists here, they cannot support promotion and should trigger a
documented interpretation before further work.

What must not be concluded even if all confirmation gates pass: exact nonlinear
Austria likelihood, dual-cap implementation correctness, covariance
restoration correctness beyond the tested route, posterior correctness,
NeuTra readiness, HMC readiness, tuning validity, default readiness,
cross-model generality, or statistical superiority.

Artifact contract: every process writes a fresh JSON file below
`docs/benchmarks/artifacts/genut_austria_endpoint_root_cause_20260817/` and
records command, commit, source hashes, target hashes, environment, device,
TF32, memory policy, wall time, and results. The execution result and checkpoint
must be updated after terminal interpretation.

## Frozen Scientific Target

All authority checks use this exact target:

- model: Austria SIR;
- horizon: `T=20` for the repaired endpoint gate;
- zero-correction control: `T=1`;
- particles: `N=1008`;
- parameter dimension: `3`;
- state dimension: `18`;
- observation dimension: `9`;
- dtype: TensorFlow `float32`;
- TensorFlow build: `2.20.0-dev0+selfbuilt`;
- GPU arithmetic: TF32 enabled;
- deterministic TensorFlow operations enabled;
- higher-moment correction steps: four at `T=20`, zero for the `T=1`
  upstream control;
- target signature:
  `4845e7322685e19650024e5886e47d89c8b9c4b70c5d36a639c9b1218d39b5c3`;
- adapter signature:
  `6a56c7a9cb9f488f2f2a44cf86316d4ad80be45ab86b74d33b019f720fd0fee6`;
- observations hash:
  `40c793fb374e84fcd347c66b189352b5997740cc753ea0be03441ecf32828009`;
- initial-noise hash:
  `21b49995edf6c72188de0870e1282348178b8ae1be1a63812933be3d30827e82`;
- process-noise hash:
  `98e6cf19066e5e3a480d41b5073d3224751a9031bfe793eac4acabf2ef9b526e`;
- design hash:
  `d8ad7e0b986cc7c90b6f55b3aaf1f582f7040b77b8cfa5ec7f5f48875f950edd`.

The runner constructs the frozen random target tensors on CPU and then executes
the endpoint explicitly on GPU. This is required because the TensorFlow build's
stateless random streams are device-specific; a previous GPU-side construction
changed the frozen hashes and invalidated that attempt.

## Route Identity And Nonclaims

The tested callable contains diagonal third/fourth-moment correction and final
affine restoration. It does not contain:

- pairwise co-skewness/co-kurtosis correction;
- pairwise row-RMS radial cap `2`; or
- coordinate cap `b=0.98,p=8`.

Therefore its route classification is `batch_diagonal_candidate`. The promoted
dual-cap algorithm is a separate future route. A pass here repairs and confirms
shared batch infrastructure and this diagonal candidate only.

The route identity ledger is:
`docs/benchmarks/artifacts/genut_austria_endpoint_root_cause_20260817/attempt01/route_identity.json`.

## Current Worktree Identity

Git commit at handoff:
`dae37183bf4421682b2ad991e2dc0d0f3c53f260`.

The worktree is dirty and contains unrelated edits from other agents. Preserve
them. Do not reset, checkout, stash, revert, or broadly reformat the worktree.
Inspect focused diffs before any future edit.

Relevant SHA-256 values at this handoff:

| Path | SHA-256 |
|---|---|
| `bayesfilter/highdim/cubature_genut_batch_tf.py` | `ae8cbfb486fc90a4a38257702cf025569ca7651bb058558d341ad602cf8a976e` |
| `bayesfilter/highdim/cubature_genut_neutra_targets.py` | `a2885cad32b4e0b7de50c6bb32e108e7939700490141947d81c93fa5e6c07793` |
| `bayesfilter/highdim/cubature_genut_batch_adapters.py` | `69537f61d8caead9b01fd20d276c30179f4b589d6aca72d1fdc3125492d05dcc` |
| `bayesfilter/highdim/higher_moment_contract_e.py` | `88337f82c3ff3e7ade9b5b9e356d7d29c4ac4dcebf79c5ffd6f992f6960d19bf` |
| `bayesfilter/highdim/cubature_genut_filter.py` | `5440f470efdd1cb18dc7f864e5f516a8d5d3f1ad6b00bb703f0d7617709572d1` |
| `docs/benchmarks/run_genut_austria_endpoint_root_cause_20260817.py` | `f57a1eddbf8077dd5d270e1f1888771f19de1fa07b69b7d091308cedb46ab4ea` |

Recompute these immediately before launching. If any relevant hash has changed,
do not assume this memo still applies. Inspect the diff, determine whether the
scientific target or numerical program changed, and update the evidence
contract before running.

## Root Cause History

The original failure was a finite scalar mismatch between tangent-free and
tangent-carrying endpoints. Both routes were finite and reported valid, but
they executed different primal arithmetic.

The first unequal particle-path tensor was the JVP route's redundant
standardization at the start of every higher-moment correction iteration. The
value route standardized once, while the JVP route standardized again from an
already standardized cloud. In FP32 this is not bitwise idempotent. The
ill-conditioned normal-equation moment solve amplified the small difference
into a large particle and likelihood mismatch.

The reset/Sinkhorn primal particles matched. The Contract-E covariance-gap
scalar had a small ULP-level asymmetry, while particles and validity agreed;
that was classified as a validity-only issue, not the initiating finite scalar
mismatch. With correction steps set to zero, the two endpoint values were
bitwise equal.

Holding the same first-step `J,r` fixed, direct least squares closely matched a
float64 reference and had a much smaller equation residual than the normal-
equation solve. Solver conditioning is therefore an amplifier after the
operation-order mismatch, not evidence that it initiates the mismatch. No
solver replacement was promoted.

The previous hand-coded recursive score was also wrong relative to the complete
finite value program. Its discrepancy began around `T=3`, grew to order `5` by
`T=10`, and reached order `10^2` by `T=20` in the preserved diagnostics.

## Repair Implemented

The repair is in `bayesfilter/highdim/cubature_genut_batch_tf.py`:

1. Higher-moment value and JVP routes use one shared primal correction core.
   The old independent hand-coded paths remain only under diagnostic names.
2. Public `batch_finite_value_score` computes one fixed-direction TensorFlow
   `ForwardAccumulator` JVP per static parameter direction of the complete
   `batch_finite_value` program.
3. Reverse-mode `GradientTape` was evaluated as a graph workaround and
   rejected. It is not numerically interchangeable with forward mode on the
   ill-conditioned long Austria recursion. Do not replace the forward score
   with reverse mode merely for graph convenience.
4. For statically known shapes, Sinkhorn iterations and horizon recurrences are
   Python-unrolled before tracing. Unknown-shape callers keep bounded
   `tf.while_loop` fallbacks.
5. The transition-first Austria policy uses a Python static branch, avoiding an
   unnecessary `tf.cond`; genuinely dynamic policies retain `tf.cond`.
6. Redundant `tf.ensure_shape` calls inside recursive bodies were removed
   because they caused TensorFlow forward-mode graph tracing failures.
7. Existing fail-closed semantics are preserved: tangent-only nonfiniteness
   produces a nonfinite value/score pair and `program_valid=false`.

Do not broad-refactor this file. It contains shared-worktree changes beyond
this campaign.

## Completed Current-Source Evidence

### CPU Derivative Authority

Terminal artifact:
`docs/benchmarks/artifacts/genut_austria_endpoint_root_cause_20260817/attempt04/derivative_cpu.json`.

This artifact is authoritative for the current handoff source hash
`ae8cbfb...a976e`. It records:

- `status=COMPLETE`;
- frozen identity guard `PASS`;
- CPU-only diagnostic with `CUDA_VISIBLE_DEVICES=-1`;
- TensorFlow `2.20.0-dev0+selfbuilt` from the `tftwogpu` environment;
- TF32 setting recorded but irrelevant to the CPU computation;
- wall time `342.0727` seconds; and
- public value/score versus independent TensorFlow forward autodiff at
  `T=1,2,20`.

| Horizon | Value identity | Maximum public-score error | Relative error | Approximate ULP error | Program valid |
|---:|---|---:|---:|---:|---|
| 1 | exact | 0 | 0 | 0 | true |
| 2 | exact | 0 | 0 | 0 | true |
| 20 | exact | 0 | 0 | 0 | true |

The central finite-difference `h^2` regression remains poor in FP32. It was
predeclared as explanatory only because the perturbation ladder is numerically
unstable for this route. It does not overturn the exact same-scalar forward-JVP
authority.

### Focused Tests And Static Checks

Recorded successful checks:

```text
python -m py_compile diagnostic runner and focused test: PASS

CUDA_VISIBLE_DEVICES=-1 pytest -q tests/highdim/test_genut_batch_primal_parity.py
5 passed

CUDA_VISIBLE_DEVICES=-1 pytest -q tests/highdim/test_cubature_genut_batch.py
4 passed

focused existing batch endpoint/FD tests plus new tests: 5 passed
git diff --check for execution-lane files: PASS
```

The earlier checkpoint records `3 passed` for the first version of the focused
parity suite; the final post-repair suite contains five passing tests. This is a
test-growth chronology, not a contradiction.

### Preserved CPU Invariants

The diagnostic lane established:

- injected tangent-only invalidity returned a NaN value/score pair with
  `program_valid=false`; and
- adding a second posterior row did not change row 0's value, score, or
  validity at the tested point.

These passed on the preserved diagnostic source state. The GPU invariant phase
must still be rerun after current-source eager/graph/XLA endpoint confirmation.

## Artifact History And Eligibility

Artifact eligibility is per JSON file and its recorded `source_sha256`, not per
directory name. Several files in `attempt04` were produced at different source
states. Never infer current eligibility from the attempt number alone.

| Artifact or attempt | Source state | Result | Current eligibility |
|---|---|---|---|
| `attempt01/route_identity.json` | initial diagnostic lane | Frozen route classified as `batch_diagonal_candidate` | Historical route evidence; recheck callable if source changes |
| `attempt01/failure.json` | initial harness | Wrong inherited GPU lane and memory-growth ordering; no scientific run | Infrastructure history only |
| `attempt02/failure.json` | repaired harness attempt | TensorFlow initialized before memory-growth configuration; no scientific run | Infrastructure history only |
| `attempt03/validity.json` | early GPU attempt | GPU-side random construction changed frozen hashes; hard veto, no result eligibility | Invalid scientific attempt; explains CPU construction fix |
| `attempt04/endpoint_eager_result.json` | `cubature_genut_batch_tf.py` hash `7e1a6a...` | Pre-repair GPU eager `T=20` mismatch: `-683.0018921` versus `-682.5823364`, gap `0.4195557`; `T=1`, zero steps exact | Historical pre-repair diagnostic only |
| `attempt04/localization_h1_result.json` | same older diagnostic source | GPU eager localization found first unequal boundary at redundant iteration-start standardization | Historical causal localization only |
| `attempt04/derivative_cpu.json` | current hash `ae8cbf...` | Current-source CPU derivative authority exact at `T=1,2,20` | Current CPU authority |
| `attempt05/endpoint_cpu.json` | diagnostic source | Intentionally stopped long CPU endpoint, `status=RUNNING` | Incomplete; never use as evidence |
| `repair_validation_attempt06/endpoint_modes_result.json` | older repaired hash `606897...` | GPU eager/graph/XLA complete; each mode had exact within-mode value identity and finite/valid results | Stale for current source and fails current cross-mode confirmation rule |
| proposed current-source `repair_validation_attempt07` | current hash `ae8cbf...` intended | Approval reviewer rejected process before creation | No directory or JSON artifact; no scientific compute consumed |

Historical `repair_validation_attempt06` details are important but cannot close
the current gate:

| Mode | Historical `T=20` value | Within-mode value identity | Finite/valid |
|---|---:|---|---|
| eager | `-683.0018921` | exact | yes |
| graph | `-683.0575562` | exact | yes |
| XLA | `-682.3775024` | exact | yes |

The cross-mode values differ. Under the current checkpoint, that drift is a
confirmation veto for that historical source. A later shared-worktree edit
changed `cubature_genut_batch_tf.py` from `606897...` to `ae8cbf...`, so fresh
current-source evidence is required rather than interpreting or rerunning the
old artifact in place.

Pre-final derivative artifacts under `attempt02`, `attempt03`, and
`repair_validation_attempt01` through `repair_validation_attempt05` are useful
debugging provenance but are superseded by the current-source terminal CPU
artifact. Preserve them; do not promote them.

## Known Unrelated Test Environment Failure

The target-factory test suite has an unrelated environment failure when it
constructs Austria: the existing
`bayesfilter/ops/_symmetric_sylvester_ops.so` has an undefined
TensorFlow/Abseil ABI symbol. Do not attribute that loader failure to this
GenUT repair. Do not rebuild or mutate the environment as part of this bounded
GPU confirmation. The relevant batch tests listed above pass.

## Current GPU And Approval State

Latest trusted occupancy probe on 2026-08-18:

```text
GPU 0: NVIDIA GeForce RTX 5080, 16303 MiB total, 4106 MiB used, 11% utilization
GPU 1: NVIDIA GeForce RTX 4080 SUPER, 16376 MiB total, 11 MiB used, 0% utilization
```

GPU 0 remains the frozen lane despite GPU 1 being emptier. Changing the GPU is
a scope change because the reviewed baseline binds the RTX 5080 lane.

The exact current-source eager command below was submitted with trusted GPU
permissions. The approval boundary rejected it before Python process creation:

```text
Automatic approval review failed: 404 Not Found.
The configured approval-review model gpt-5.6-luna was not available.
```

Post-rejection checks found no matching TensorFlow process, no
`repair_validation_attempt07` directory, and no endpoint JSON. Therefore:

- NVIDIA device visibility is working;
- TensorFlow execution of the current source is not checked;
- CUDA/TensorFlow health must not be declared broken or healthy from this
  rejection alone;
- the rejection consumes no scientific attempt or GPU compute budget; and
- Claude must use its own trusted/elevated GPU execution boundary and must not
  treat a sandbox-hidden GPU failure as machine evidence.

Earlier Codex sessions also made several bounded trusted-launch requests that
failed at the approval boundary with timeouts or HTTP 502 responses. Some prose
notes call those proposed attempts 07 through 09, but no corresponding process
directories were created. Attempt numbering is an artifact-path convention,
not evidence that a process ran. The current handoff may use the proposed
`repair_validation_attempt07` path only after rechecking that it is absent.

## Skeptical Resume Audit

The bounded resume remains scientifically meaningful because the runner:

- guards the exact target signature, adapter signature, and target tensor
  hashes before endpoint execution;
- constructs frozen random inputs on CPU and runs the endpoint on an explicitly
  selected GPU;
- compares value-only and value-plus-score endpoints for the same finite
  program;
- records finite status and `program_valid` separately from promotion;
- records commit, source hashes, TensorFlow build, dtype, TF32, device,
  deterministic operations, memory growth, command, output, and wall time;
- keeps graph and XLA arms endpoint-only, avoiding intrusive interior capture;
  and
- preserves dual-cap, tuning, NeuTra, HMC, posterior, and default nonclaims.

Wrong-baseline audit: the comparator is the current same-scalar value endpoint,
not UKF, SGQF, finite differences, or an old GPU artifact.

Proxy-metric audit: finiteness, runtime, FD quality, conditioning, and
acceptance-like diagnostics cannot replace exact within-mode value identity.

Environment audit: only `/home/chakwong/anaconda3/envs/tftwogpu/bin/python` is
eligible; the old `tf-gpu` stack is not Blackwell-capable for this purpose.

Default audit: `T=20`, `N=1008`, FP32, TF32 enabled, four diagonal correction
steps, current repository controls, deterministic operations, GPU 0, and the
`tftwogpu` environment are frozen baseline choices from the reviewed plan.
They are not universal defaults. Their main failure mode is mode-sensitive,
ill-conditioned arithmetic; the separate eager/graph/XLA endpoints are the
earliest relevant diagnostic.

Pre-mortem: the commands could succeed while testing the wrong source, wrong
random tensors, wrong GPU, stale output, disabled memory growth, or only
within-mode identity while hiding cross-mode drift. The manifest and fresh
hash checks below address those risks. A run can also fail because of approval
or harness infrastructure rather than science; no endpoint inference is allowed
without a process artifact.

The audit passes for a bounded retry. It does not authorize a broader campaign.

## Resume Budget

- One trusted occupancy probe immediately before execution.
- At most three endpoint processes: eager, graph, then XLA.
- One fresh output directory and JSON per process.
- Combined endpoint wall-time ceiling: 15 minutes.
- Run the invariant process only if all endpoint gates, including cross-mode
  identity, pass.
- Approval rejections before process creation do not consume scientific compute,
  and historical Codex approval failures do not consume Claude's fresh
  infrastructure retry audit. Starting from this handoff, stop after three
  consecutive Claude launch rejections and record the exact infrastructure
  condition.
- No cross-model run, tuning, NeuTra training, or HMC is included in this
  budget.

## Exact Claude Code Resume Procedure

### 1. Re-read Policy And Preserve The Worktree

```bash
git status --short
git rev-parse HEAD
git diff -- bayesfilter/highdim/cubature_genut_batch_tf.py
```

Do not reset or checkout. If the commit or relevant source differs from this
memo, stop and reconcile the change before launch.

### 2. Recompute Relevant Source Hashes

```bash
sha256sum \
  bayesfilter/highdim/cubature_genut_batch_tf.py \
  bayesfilter/highdim/cubature_genut_neutra_targets.py \
  bayesfilter/highdim/cubature_genut_batch_adapters.py \
  bayesfilter/highdim/higher_moment_contract_e.py \
  bayesfilter/highdim/cubature_genut_filter.py \
  docs/benchmarks/run_genut_austria_endpoint_root_cause_20260817.py
```

The output must match the handoff table. If not, do not launch against an
unreviewed source state.

### 3. Run Trusted GPU Occupancy And Framework Preflight

Any GPU/CUDA/NVIDIA command must use trusted/elevated device access.

```bash
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu \
  --format=csv,noheader
```

Use GPU 0 if it remains reasonably available. If it is materially occupied,
wait or record the capacity blocker; do not silently switch to GPU 1.

The endpoint runner itself performs the framework-specific GPU and memory-
growth checks. Do not add a separate untrusted TensorFlow probe and interpret
its failure.

### 4. Run Current-Source Eager Endpoint First

Use a fresh path. At handoff,
`repair_validation_attempt07/endpoint_gpu0_eager.json` does not exist.

```bash
TF_FORCE_GPU_ALLOW_GROWTH=true TF_DETERMINISTIC_OPS=1 \
MPLCONFIGDIR=/tmp/bayesfilter-matplotlib CUDA_VISIBLE_DEVICES=0 \
/home/chakwong/anaconda3/envs/tftwogpu/bin/python \
docs/benchmarks/run_genut_austria_endpoint_root_cause_20260817.py \
--device gpu --gpu-index 0 --phase endpoint --endpoint-modes eager \
--output docs/benchmarks/artifacts/genut_austria_endpoint_root_cause_20260817/repair_validation_attempt07/endpoint_gpu0_eager.json
```

Inspect the JSON before proceeding. Required eager checks:

- `status` is `COMPLETE`;
- source hashes match the pre-launch source;
- frozen identity guard is `PASS`;
- TensorFlow is `2.20.0-dev0+selfbuilt`;
- physical and logical GPU lists contain GPU 0;
- execution device is `/GPU:0`;
- memory policy is `memory_growth`, configured before logical initialization,
  and verified true for every visible physical GPU;
- TF32 is enabled and deterministic-op environment is `1`;
- both endpoint rows have finite values and scores;
- both validity arrays are true; and
- both value comparisons are `exact_equal=true` with zero absolute, relative,
  and approximate ULP error.

If any eager check fails, preserve the JSON, stop, classify the failure, and do
not run graph, XLA, or invariants.

### 5. Run Graph Endpoint In A Separate Process

```bash
TF_FORCE_GPU_ALLOW_GROWTH=true TF_DETERMINISTIC_OPS=1 \
MPLCONFIGDIR=/tmp/bayesfilter-matplotlib CUDA_VISIBLE_DEVICES=0 \
/home/chakwong/anaconda3/envs/tftwogpu/bin/python \
docs/benchmarks/run_genut_austria_endpoint_root_cause_20260817.py \
--device gpu --gpu-index 0 --phase endpoint --endpoint-modes graph \
--output docs/benchmarks/artifacts/genut_austria_endpoint_root_cause_20260817/repair_validation_attempt08/endpoint_gpu0_graph.json
```

Apply the same per-artifact checks. Stop on failure.

### 6. Run XLA Endpoint In A Separate Process

```bash
TF_FORCE_GPU_ALLOW_GROWTH=true TF_DETERMINISTIC_OPS=1 \
MPLCONFIGDIR=/tmp/bayesfilter-matplotlib CUDA_VISIBLE_DEVICES=0 \
/home/chakwong/anaconda3/envs/tftwogpu/bin/python \
docs/benchmarks/run_genut_austria_endpoint_root_cause_20260817.py \
--device gpu --gpu-index 0 --phase endpoint --endpoint-modes xla \
--output docs/benchmarks/artifacts/genut_austria_endpoint_root_cause_20260817/repair_validation_attempt09/endpoint_gpu0_xla.json
```

Apply the same per-artifact checks. XLA is the repository default target, but
eager and graph remain required diagnostic replication arms.

### 7. Perform Cross-Mode Comparison

Compare the exact `T=20`, four-correction `value_only` scalar and its hash
across eager, graph, and XLA. Under the current checkpoint, any cross-mode value
drift is a confirmation veto. Also report the score vectors and hashes for all
three modes. Do not create a tolerance after seeing their differences.

The `T=1`, zero-correction control should also remain exact across modes. A
difference there indicates an upstream/compiler arithmetic issue rather than a
higher-moment correction issue.

If the cross-mode gate fails, preserve all three artifacts, stop before
invariants, and write the exact values, hashes, absolute differences, source
identity, and likely classification into the execution result and checkpoint.

### 8. Run GPU Invariants Only After All Endpoint Gates Pass

```bash
TF_FORCE_GPU_ALLOW_GROWTH=true TF_DETERMINISTIC_OPS=1 \
MPLCONFIGDIR=/tmp/bayesfilter-matplotlib CUDA_VISIBLE_DEVICES=0 \
/home/chakwong/anaconda3/envs/tftwogpu/bin/python \
docs/benchmarks/run_genut_austria_endpoint_root_cause_20260817.py \
--device gpu --gpu-index 0 --phase invariants \
--output docs/benchmarks/artifacts/genut_austria_endpoint_root_cause_20260817/repair_validation_attempt10/endpoint_gpu0_invariants.json
```

Required invariant results:

- tangent-only invalidity returns nonfinite value and score,
  `program_valid=false`, and `pass_fail_closed=true`;
- adding a second posterior row leaves row 0 value and score exactly equal; and
- row 0 validity is unchanged.

### 9. Record Terminal Interpretation

Update both:

- `docs/plans/bayesfilter-austria-genut-neutra-root-cause-execution-result-2026-08-18.md`;
- `docs/plans/bayesfilter-austria-genut-neutra-root-cause-execution-checkpoint-2026-08-18.md`.

Record the exact commands, source hashes, GPU occupancy, TensorFlow and memory
policy, artifact paths, wall times, per-mode values and scores, all veto checks,
cross-mode comparison, decision, uncertainty, next justified action, and
nonclaims. Do not overwrite or delete older result text; append or carefully
supersede only the stale current-status statements.

If the current-source GPU confirmation passes, the next phase is cross-model
value/score regression under a new reviewed scope. It is not NeuTra or HMC.
Any score/value repair invalidates historical Austria tuning; fresh scope-
specific tuning with disjoint calibration/validation and untouched claim data
is mandatory before a NeuTra proposal.

## Decision Table For The Next Agent

| Observation | Classification | Required action | Forbidden conclusion |
|---|---|---|---|
| Approval or permission rejection before process creation | Infrastructure blocker | Preserve exact error, confirm no process/artifact, retry only within platform rules and blocked-attempt budget | GPU, CUDA, TensorFlow, or endpoint failure |
| Frozen target/source identity mismatch | Hard validity veto | Stop before endpoint interpretation and reconcile source/input state | Any scientific or numerical verdict |
| Eager nonfinite, invalid, or unequal within-mode value | Current-source confirmation failure | Preserve artifact; stop graph/XLA/invariants; localize under a bounded repair plan | Research direction or dual-cap rejection |
| Eager passes, later mode fails within-mode identity | Compiler-mode confirmation failure | Preserve all completed artifacts; stop later phases; localize mode-sensitive path | Tolerance-based acceptance |
| All modes pass internally but `T=20` value differs across modes | Cross-mode confirmation veto | Record exact drift and stop before invariants | Current GPU confirmation pass |
| Endpoints and cross-mode identity pass, invariants fail | Fail-closed or batch-semantics defect | Preserve artifact and plan the smallest repair | NeuTra/HMC readiness |
| Endpoints, cross-mode identity, and invariants all pass | Current diagonal-route GPU confirmation pass | Update terminal notes; plan separate cross-model regression and later fresh tuning | Dual-cap correctness, posterior correctness, or default readiness |

## Stop Conditions

Stop and record a blocker or failure if:

- three consecutive trusted launches are rejected before process creation;
- the target signature, adapter signature, target hashes, or relevant source
  hashes change;
- memory growth cannot be configured and verified before GPU initialization;
- the wrong GPU, TensorFlow build, dtype, TF32, or execution mode is used;
- eager produces nonfinite/invalid output or unequal within-mode value;
- graph or XLA produces nonfinite/invalid output or unequal within-mode value;
- cross-mode `T=20` scalar identity fails;
- a required artifact is missing, left `RUNNING`, or overwritten;
- the 15-minute endpoint budget is exhausted; or
- continuing would require a new scientific target, acceptance tolerance,
  route, package/environment change, broader compute, NeuTra, HMC, or default
  decision.

An approval timeout, 502, or approval-model 404 is not a scientific failure.
Do not work around the execution boundary with an untrusted GPU command.

## Final Handoff Summary

What is established:

- the wired route is diagonal plus affine restoration, not dual-cap;
- redundant JVP iteration-start standardization initiated the original primal
  mismatch, with the ill-conditioned solve acting as an amplifier;
- the prior manual recursive score was wrong relative to the complete finite
  scalar;
- the current public route uses a shared primal and forward autodiff;
- current-source CPU value and score match independent forward autodiff exactly
  at `T=1,2,20`;
- focused CPU tests and fail-closed/batch diagnostics pass; and
- trusted `nvidia-smi` sees both GPUs.

What remains unresolved:

- no trusted endpoint artifact exists for the current `ae8cbf...` source;
- the only complete three-mode GPU artifact is stale and had historical cross-
  mode value drift;
- current eager/graph/XLA endpoint identity and current GPU invariants remain
  open; and
- cross-model, tuning, NeuTra, HMC, posterior, dual-cap, and default-readiness
  gates remain outside this handoff.

The immediate next action is the trusted current-source eager endpoint command,
not a source edit, environment rebuild, cross-model run, tuning run, NeuTra
training run, or HMC launch.
