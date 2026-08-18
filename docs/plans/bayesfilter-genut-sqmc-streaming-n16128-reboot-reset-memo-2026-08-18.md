# GenUT SQMC Streaming N=16128 Claude Code Handoff and Reset Memo

Original reset memo: 2026-08-18 18:46 HKT

Claude Code handoff update: 2026-08-18 19:57 HKT

Claude Code execution update: 2026-08-19 03:20 HKT (see terminal section)

Repository: `/home/chakwong/BayesFilter`

Git HEAD: `dae37183bf4421682b2ad991e2dc0d0f3c53f260` (at handoff; execution ran
at `a13c481b` with all nine in-scope hashes verified identical)

Environment: `/home/chakwong/anaconda3/envs/tftwogpu`

Status: `EXECUTED_2026-08-19__N1008_PARITY_EXACT_PASS__N4032_PARITY_FAIL__N16128_NOT_LAUNCHED__AWAITING_OWNER_DIRECTION`

> 2026-08-19 terminal note: the ladder below was executed by Claude Code. The
> mandated GPU probe, focused suite, and both fresh pairs completed. `N=1008`
> passed with bit-exact equality; `N=4032` failed the frozen parity gate while
> passing every validity screen; `N=16128` was therefore not launched. Full
> evidence, diagnostics, budget accounting, and owner options are in the
> 2026-08-19 section of
> `docs/plans/bayesfilter-genut-sqmc-streaming-n16128-result-2026-08-18.md`.
> The historical sections below are preserved unchanged for provenance.

## Claude Code Executive Handoff

Resume the bounded GenUT SQMC exact-streaming `N=16128` campaign from this
memo. The implementation and CPU-hidden focused tests are complete, but no
post-repair GPU parity artifact exists and no `N=16128` row has been run.

The immediate blocker is an execution-permission gateway failure, not observed
CUDA, TensorFlow, OOM, numerical, or scientific failure. In the current Codex
session, trusted `nvidia-smi` succeeded, but the exact TensorFlow GPU probe was
rejected before the Python process was created because the automatic approval
reviewer model was unavailable. Claude Code must rerun that TensorFlow probe
with trusted/elevated GPU access before running tests or benchmarks.

Do not launch `N=16128` until fresh current-source dense and streamed pairs at
both `N=1008` and `N=4032` pass the frozen parity gates below. Do not use any of
`smoke_attempt01` through `smoke_attempt03` to clear those gates: all three
artifacts were created before the final one-block repair source hash.

### Current verdict by layer

| Layer | Current evidence | Verdict |
|---|---|---|
| Physical NVIDIA visibility | Trusted `nvidia-smi` succeeded at 19:55 HKT | `PASS` |
| Preferred GPU availability | GPU1 RTX 4080 SUPER: 0% utilization, 16035 MiB free of 16376 MiB | `PASS` |
| Current TensorFlow environment/device initialization | Exact probe rejected before process creation | `NOT CHECKED` |
| TensorFlow memory-growth policy on current session | Exact probe rejected before process creation | `NOT CHECKED` |
| Real TensorFlow GPU operation on current session | Exact probe rejected before process creation | `NOT CHECKED` |
| Historical TensorFlow GPU/XLA execution | Attempts 01-03 ran on logical `/GPU:0` with memory growth and TF32/XLA | `HISTORICAL PASS ONLY` |
| Post-repair one-block GPU parity | No post-repair artifact | `BLOCKED` |
| Current-source tiled `N=4032` GPU parity | No post-repair dense/streamed pair | `BLOCKED` |
| `N=16128` feasibility | Not launched | `NOT RUN` |

It is therefore wrong to say either "the GPU is broken" or "the GPU problem
is fully cleared." Device/driver visibility is currently healthy. TensorFlow
GPU execution is still unverified in the current session because the command
gateway, not TensorFlow, prevented the probe from starting.

## Current Session Evidence: 19:55 HKT

Trusted physical-device query:

```text
0, NVIDIA GeForce RTX 5080, 10, 11873, 16303
1, NVIDIA GeForce RTX 4080 SUPER, 0, 16035, 16376
```

The exact TensorFlow probe in the next section was then submitted with trusted
permissions and failed at `CreateProcess`, before interpreter startup, with:

```text
Automatic approval review failed: unexpected status 404 Not Found:
model is not available. model: gpt-5.6-luna
```

No TensorFlow process, GPU allocation, benchmark row, checkpoint, result file,
or new attempt directory was created by that rejected command. This new error
has the same operational classification as the earlier permission-review
timeouts and `502 Bad Gateway` responses: `APPROVAL_GATEWAY_FAILURE_BEFORE_PROCESS_CREATION`.

The Codex gateway response prohibited an automatic workaround or retry without
fresh explicit user approval after disclosure. That restriction applies to the
rejected Codex action. Claude Code should use its own trusted execution boundary
and still obey the repository GPU/CUDA and memory-growth policies.

## Exact Starting Sequence for Claude Code

1. Read the repository `AGENTS.md`, this memo, the plan, and the result note.
2. Run `git status --short --untracked-files=all`. Preserve every unrelated
   modified or untracked file. Do not reset, restore, stash, clean, or checkout
   the worktree. The streaming implementation, harness, plan, result, memo, and
   focused test are untracked relative to HEAD.
3. Verify that the in-scope files exist and recompute their SHA-256 hashes.
   Compare them with the current-source table below. If any code, harness,
   test, or plan hash differs, inspect the diff and refresh the parity baseline
   rather than silently using this memo's numbers.
4. Run the trusted `nvidia-smi` availability query. Prefer physical GPU1 only
   when utilization is below 50% and free memory exceeds 8 GiB. Otherwise use
   GPU0 if it meets the rule. Record the selected physical index.
5. Run the exact TensorFlow probe below with the selected physical GPU exposed
   through `CUDA_VISIBLE_DEVICES`. It must see exactly one physical and one
   logical GPU, verify memory growth before initialization, place the matmul on
   logical `/GPU:0`, and print result `512.0`.
6. If the probe passes, rerun the CPU-hidden focused suite, Python compilation,
   and whitespace checks. CPU-only status must remain explicit through
   `CUDA_VISIBLE_DEVICES=-1` for the focused suite.
7. Execute the exact resume ladder below in fresh automatically allocated
   attempt directories. Stop at the first failed continuation gate. Never
   overwrite attempts 01-03 and never manually precreate an attempt directory.
8. After each pair, compare raw manifests and rows before interpreting values.
   Update the result note and this memo with the actual commands, hashes,
   device/memory evidence, wall times, artifacts, gate decision, uncertainty,
   and nonclaims.

This handoff authorizes resumption only within the existing reviewed campaign:
Austria-SIR `T=20`, trust-region reset, frozen controls and tolerances, seed
`97701`, the reviewed particle counts and routes, the exact-divisor transport
policy, one visible GPU, TF32, XLA, and the existing compute ceiling. It does
not authorize a 16-seed `N=16128` campaign, changed equations or thresholds,
package/environment mutation, a different model, a different hardware class,
or scientific/default/HMC/NeuTra promotion.

## Historical Reason for the Reboot Memo

The repository and CUDA hardware were not the observed blocker. Escalated
`nvidia-smi` repeatedly succeeded and last reported:

| Physical index | Device | Utilization | Free memory | Availability |
|---:|---|---:|---:|---|
| 0 | RTX 5080 | 12% | 12113 MiB | available fallback |
| 1 | RTX 4080 SUPER | 0% | 16035 MiB | preferred |

Before this handoff, the benchmark launch repeatedly failed before process
creation with:

```text
The automatic permission approval review did not finish before its deadline.
```

The same timeout occurred through both the absolute `tftwogpu` Python path and
the approved `conda run` entrypoint. Two post-fix streamed retries also returned
`502 Bad Gateway`. After every failed launch, process and artifact checks found
no running harness and no new attempt directory. The current-session 404
approval-review failure described above shows that the permission gateway is
still an active blocker for Codex. None of these failures is evidence of CUDA
initialization failure, TensorFlow failure, OOM, or scientific failure.

## Mandatory Current-Session GPU Test

First check physical availability:

```bash
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.free,memory.total \
  --format=csv,noheader,nounits
```

Select GPU1 when it meets the rule. Then test the exact environment, device,
memory-growth policy, and one real GPU operation:

```bash
TF_FORCE_GPU_ALLOW_GROWTH=true CUDA_VISIBLE_DEVICES=1 \
/home/chakwong/anaconda3/envs/tftwogpu/bin/python - <<'PY'
import tensorflow as tf

physical = tf.config.list_physical_devices("GPU")
assert len(physical) == 1, physical
tf.config.experimental.set_memory_growth(physical[0], True)
assert tf.config.experimental.get_memory_growth(physical[0]) is True
logical = tf.config.list_logical_devices("GPU")
assert len(logical) == 1, logical
with tf.device("/GPU:0"):
    value = tf.linalg.matmul(tf.ones([512, 512]), tf.ones([512, 512]))
assert "GPU:0" in value.device, value.device
print(
    {
        "physical": [item.name for item in physical],
        "logical": [item.name for item in logical],
        "memory_growth": tf.config.experimental.get_memory_growth(physical[0]),
        "result": float(value[0, 0].numpy()),
        "device": value.device,
    }
)
PY
```

Interpret the outcome strictly:

| Outcome | Classification | Action |
|---|---|---|
| `nvidia-smi` cannot run | device/driver/session failure | stop; repair GPU visibility |
| `nvidia-smi` works but the command never creates a process | approval gateway failure | stop; repair gateway/session |
| TensorFlow has zero physical GPUs | `tftwogpu` visibility/environment failure | stop; inspect environment and `CUDA_VISIBLE_DEVICES` |
| memory growth cannot be set before initialization | repository memory-policy failure | stop; do not run benchmark |
| matmul executes on `/GPU:0` and prints `512.0` | GPU execution pass | proceed to focused tests and parity ladder |

Remember that physical GPU1 becomes TensorFlow logical `/GPU:0` because
`CUDA_VISIBLE_DEVICES=1` exposes only that device.

## Current Code State

The exact streaming implementation is present in:

- `bayesfilter/highdim/genut_guided_proposal_tf.py`
- `bayesfilter/highdim/ledh_pfpf_genut_initial_rqmc_tf.py`
- `docs/benchmarks/run_genut_sqmc_particle_trust_austria_20260817.py`
- `tests/highdim/test_genut_sqmc_score_blocking.py`

The important repair is the one-block rule. The active transport policy uses
`K=N` for `N<=3000`. In that case the streaming-labelled route now calls the
dense arithmetic helper directly, avoiding an XLA reduction-graph difference
that the nonlinear trust-region correction amplified. For `N>3000`, transport
remains genuinely tiled. Required chunk selections are:

| N | K | Grid |
|---:|---:|---:|
| 1008 | 1008 | `1 x 1`, dense arithmetic baseline |
| 4032 | 2016 | `2 x 2`, tiled streaming |
| 16128 | 2688 | `6 x 6`, tiled streaming |

Current source hashes:

| Path | SHA-256 |
|---|---|
| `bayesfilter/highdim/genut_guided_proposal_tf.py` | `9eec11e92a12145144a4f579fda85ffa87ed4a97b4a374a0df2bdf66286dc0cc` |
| `bayesfilter/highdim/genut_shape_lm_tf.py` | `17007e0484633f4882173a09f0da1a91ab1a1cbf4f65eeff67936fec1489a2af` |
| `bayesfilter/highdim/higher_moment_contract_e.py` | `88337f82c3ff3e7ade9b5b9e356d7d29c4ac4dcebf79c5ffd6f992f6960d19bf` |
| `bayesfilter/highdim/ledh_pfpf_genut_initial_rqmc_tf.py` | `6744354e449a4ff53cee177b98f6d75b52f92315fe3767135bde4bde223221df` |
| `bayesfilter/highdim/sqmc_tf.py` | `de1c335d5a3586cb2ece51663447ecbd56c6d7a5eb22c478cf366e3fcdc493eb` |
| `docs/benchmarks/run_genut_sqmc_particle_trust_austria_20260817.py` | `2c0639a72d184be89c6c92930caedff0cf276e9c1e0033ec7b477540c8a9b882` |
| `tests/highdim/test_genut_sqmc_score_blocking.py` | `66c8657fcee6aef66802e5672d2429cb7dc1f75666c19bd683f795e67d367059` |
| `docs/plans/bayesfilter-genut-sqmc-streaming-n16128-plan-2026-08-18.md` | `7608ac6b505f904c4fc916d5229dd1d571fab99b260b16bee51f4a264431835c` |
| `docs/plans/bayesfilter-genut-sqmc-streaming-n16128-result-2026-08-18.md` | `2fa401c49012d8d2421b0379ee9ed230e599742f629a1f0b7517516a7d43108b` |

Do not assume these hashes remain current after editing. Every resumed run must
record fresh hashes in its manifest. This memo intentionally does not list its
own hash because this handoff edit changes it; compute it at handoff time if a
snapshot checksum is useful.

## Verified Tests

The last completed focused run was CPU-hidden by design:

```bash
CUDA_VISIBLE_DEVICES=-1 \
/home/chakwong/anaconda3/envs/tftwogpu/bin/python -m pytest \
  tests/highdim/test_genut_sqmc_score_blocking.py -q
```

Result: `10 passed`, with only two TensorFlow Probability `distutils`
deprecation warnings. Python compilation and `git diff --check` also passed.

The ten passing tests and their evidence roles were:

| Test | Evidence role |
|---|---|
| `test_sqmc_harness_defaults_to_trust_region` | harness-default guard |
| `test_streaming_sinkhorn_has_no_dense_pairwise_state` | structural-memory guard |
| `test_streaming_sinkhorn_multitile_matches_dense_fp64` | direct multi-tile FP64 parity |
| `test_streaming_one_block_uses_dense_arithmetic_baseline` | exact `K=N` repair regression |
| `test_streaming_sinkhorn_multitile_compiles_with_cpu_xla` | streaming XLA compilation smoke |
| `test_streaming_sinkhorn_multitile_matches_dense_fp32_cpu_xla` | direct FP32/XLA parity |
| `test_austria_score_child_blocking_matches_dense` | analytical score-blocking parity |
| `test_sqmc_trust_region_reset_route_is_finite` | small reset validity smoke |
| `test_sqmc_streaming_reset_matches_dense_with_trust_region` | reset/diagnostic parity |
| `test_austria_streaming_route_matches_dense_value_and_score` | small full-route value/score/final-particle parity |

These tests establish bounded engineering evidence only. The last run was
deliberately CPU-hidden and does not prove current GPU visibility, GPU XLA
compilation, the `N=4032` tiled full route, or `N=16128` feasibility. No focused
suite was rerun after the 19:55 HKT gateway failure because the mandated
current-session TensorFlow GPU probe did not start.

After the mandatory trusted TensorFlow GPU probe passes, rerun this focused
suite and:

```bash
/home/chakwong/anaconda3/envs/tftwogpu/bin/python -m py_compile \
  bayesfilter/highdim/genut_guided_proposal_tf.py \
  bayesfilter/highdim/ledh_pfpf_genut_initial_rqmc_tf.py \
  docs/benchmarks/run_genut_sqmc_particle_trust_austria_20260817.py \
  tests/highdim/test_genut_sqmc_score_blocking.py
git diff --check
```

## Artifact Ledger

Artifact root:
`docs/benchmarks/artifacts/genut-sqmc-streaming-n16128-20260818/`

| Artifact | Content | Admission status |
|---|---|---|
| `smoke_attempt01/result.json` | pre-repair streamed `N=1008`, repaired permutation, seed `97701` | historical engineering diagnostic only |
| `smoke_attempt02/result.json` | pre-repair streamed `N=4032`, repaired permutation, seed `97701` | historical engineering diagnostic only |
| `smoke_attempt03/result.json` | pre-repair-source-hash dense `N=1008`, repaired permutation, seed `97701` | explains mismatch; not the post-repair parity baseline |
| `smoke_attempt04` and later | not created | required fresh current-source evidence after handoff |

No post-repair GPU artifact exists. Do not claim that the one-block repair has
passed GPU route parity. Do not use attempts 01-03 to clear the `N=16128` gate.

Reference numbers, descriptive only:

| Artifact | Value | Score `(j0,j1,j2)` | TV | Unique ancestors | Row seconds | Allocator peak bytes |
|---|---:|---|---:|---:|---:|---:|
| attempt01 streamed `N=1008` | `-682.6151123` | `(393.73398, -154.54193, 8.82004)` | `1.8714e-6` | `1008` | `74.9996` | `679926528` |
| attempt02 streamed `N=4032` | `-681.6522217` | `(-663.29675, 155.72716, 5.47599)` | `6.6462e-6` | `4032` | `151.3136` | `8285744896` |
| attempt03 dense `N=1008` | `-681.3641357` | `(294.18781, -133.56847, 5.83784)` | `1.6856e-6` | `1008` | `63.9098` | `679925760` |

All three rows were finite, program-valid, permutation-valid, had zero
state-map saturation, used `score_child_block_size=126`, ran with TF32 and XLA
on logical `/GPU:0`, and recorded verified memory growth. Their common runtime
environment was Python `3.11.15` and TensorFlow
`2.20.0-dev0+selfbuilt` under `tftwogpu`. Their total recorded row time was
about 290 seconds; rejected gateway launches consumed no benchmark GPU time.

Attempts 01-03 all recorded
`genut_guided_proposal_tf.py` hash
`c20292b14ea7e1b56e00e48b2f0162b426881c258f6d30c4a215de374f957bbf`.
The current post-repair hash is
`9eec11e92a12145144a4f579fda85ffa87ed4a97b4a374a0df2bdf66286dc0cc`.
That difference is decisive: attempt03 is the dense run that motivated the
repair, not a current-source dense baseline. Attempts 01 and 03 do have matching
initial, process, and ancestor hashes, which localizes their observed mismatch
to finite-program/source behavior rather than different sampled inputs, but it
does not admit either artifact after the repair.

The older dense claim artifact under
`docs/benchmarks/artifacts/genut-sqmc-particle-count-trust-region-20260817/`
also has different source hashes. It remains historical evidence and cannot
clear current-source streaming parity.

For completeness, its repaired-permutation seed-`97701` rows were
`N=1008`: value `-682.2589111`, score
`(-603.21265, 46.61590, 9.69924)`; and `N=4032`: value `-681.9028931`, score
`(167.98019, -104.87935, 5.87885)`. Its transport, wrapper, and harness hashes
all differ from current source. These numbers are historical source-drift
context only; do not compare them to a new streamed row as a parity oracle.

## Exact Resume Ladder

All commands below use seed `97701` because `--stage smoke` selects the single
reviewed smoke seed. Every command creates a fresh attempt directory through
the harness's exclusive `mkdir()` allocator. Attempts 01-03 must remain
untouched.

Before comparing outputs, require the dense and streamed artifacts in a pair to
match on all fields that should be invariant:

- Git commit and every current source hash in `run_manifest.source_sha256`;
- model `austria_sir_T20`, horizon `20`, particle count, seed `97701`, route,
  reset variant, reset route ID, and score child block size `126`;
- initial, process, and ancestor SHA-256 values;
- complete route and trust-region controls;
- one logical GPU, TF32 `true`, JIT `true`, verified memory growth, and the
  managed-session trust basis;
- finiteness, `program_valid`, `row_valid`, marginal diagnostics, saturation,
  and ancestry validity.

The intentional differences are `transport_plan`, `transport_plan_id`, and
the arithmetic path implied by the plan. If an invariant field differs, stop
with `INPUT_OR_PROVENANCE_MISMATCH`; do not calculate a parity verdict from the
values or scores.

### 1. Fresh post-repair N=1008 pair

Run dense:

```bash
TF_FORCE_GPU_ALLOW_GROWTH=true CUDA_VISIBLE_DEVICES=1 \
/home/chakwong/anaconda3/envs/tftwogpu/bin/python \
docs/benchmarks/run_genut_sqmc_particle_trust_austria_20260817.py \
  --stage smoke --particle-counts 1008 \
  --routes repaired_permutation --resets trust_region \
  --transport-plan dense
```

Run streamed:

```bash
TF_FORCE_GPU_ALLOW_GROWTH=true CUDA_VISIBLE_DEVICES=1 \
/home/chakwong/anaconda3/envs/tftwogpu/bin/python \
docs/benchmarks/run_genut_sqmc_particle_trust_austria_20260817.py \
  --stage smoke --particle-counts 1008 \
  --routes repaired_permutation --resets trust_region \
  --transport-plan streaming
```

Require matching input hashes, controls, source hashes, validity, ancestry,
and route identity. Because `K=N` now executes the same arithmetic helper,
expect exact output equality; at minimum enforce the frozen full-route
`atol=rtol=5e-4` bound. Stop if it fails.

### 2. Fresh post-repair N=4032 pair

Run the same two commands with `--particle-counts 4032`. The tiled comparator
must satisfy:

- `abs(value_streamed-value_dense) <= 0.05`;
- for each score coordinate,
  `abs(delta) / max(1, abs(score_dense)) <= 5e-3`;
- matching inputs, controls, source hashes, and trust route;
- finite and program-valid output;
- `TV <= 1e-4`;
- zero state-map saturation; and
- exactly 4032 unique ancestors with a valid permutation.

Do not continue if this gate fails. Diagnose the tiled reduction or trust
sensitivity instead.

### 3. N=16128 repaired-permutation feasibility row

Only after both parity gates pass:

```bash
TF_FORCE_GPU_ALLOW_GROWTH=true CUDA_VISIBLE_DEVICES=1 \
/home/chakwong/anaconda3/envs/tftwogpu/bin/python \
docs/benchmarks/run_genut_sqmc_particle_trust_austria_20260817.py \
  --stage smoke --particle-counts 16128 \
  --routes repaired_permutation --resets trust_region \
  --transport-plan streaming
```

Require `K=2688`, a `6 x 6` grid, finite/program-valid output, `TV<=1e-4`,
zero saturation, valid permutation, 16128 unique ancestors, and recorded
allocator peak/runtime. An OOM or invalid row is a retained feasibility result,
not permission to lower `N` silently.

### 4. Remaining N=16128 variants

If the repaired-permutation row passes resource and validity gates, run:

```bash
TF_FORCE_GPU_ALLOW_GROWTH=true CUDA_VISIBLE_DEVICES=1 \
/home/chakwong/anaconda3/envs/tftwogpu/bin/python \
docs/benchmarks/run_genut_sqmc_particle_trust_austria_20260817.py \
  --stage smoke --particle-counts 16128 \
  --routes iid_dual_cap previous_inverse_cdf \
           repaired_fixed_previous_controls \
  --resets trust_region --transport-plan streaming
```

This plan authorizes one seed across the four variants. It does not authorize a
16-seed `N=16128` campaign.

## Campaign Budget and Repair Boundaries

The reviewed plan's ceilings remain active:

| Work | Ceiling |
|---|---:|
| implementation tests and small smokes | 45 minutes |
| streamed/current-source `N=4032` parity work | 90 minutes |
| four `N=16128` rows | 8 GPU hours |
| localized infrastructure repairs without equation/scope changes | at most 2 |

Attempts 01-03 account for about 290 seconds of recorded GPU row time. No
`N=16128` GPU time has been consumed. Approval-gateway rejections created no
process and consumed no benchmark GPU time, but every rejection must still be
recorded as an infrastructure attempt. Do not expand the compute ceiling to
compensate for future failures without user direction.

A localized harness, serialization, XLA-resource, or execution-wrapper failure
may be repaired and retried only while the target, data, equations, controls,
particle counts, variants, seed, gates, hardware class, and total budget remain
unchanged. Use a fresh attempt directory and record the failure, repair,
focused regression, retry, wall time, and remaining budget. Stop for user
direction before any package/environment mutation, changed numerical method,
relaxed tolerance, extra seed, materially expanded compute, privacy change,
destructive operation, or scientific/product-direction change.

## Current Decision Table

| Decision | Primary criterion | Veto status | Next justified action | Not concluded |
|---|---|---|---|---|
| Retain exact tiled implementation | focused structural and direct parity tests | no focused-test veto | preserve code and rerun verification after GPU probe | GPU route parity |
| Admit one-block repair | fresh same-source GPU `N=1008` dense/streamed pair | `BLOCKED`: no post-repair artifact | run fresh pair and require exact equality if observed, with frozen `5e-4` bound as the formal veto | general tiled parity |
| Admit tiled route | fresh same-source GPU `N=4032` pair | `BLOCKED`: no post-repair pair | run pair and apply frozen value/score/validity gates | `N=16128` feasibility |
| Launch repaired-permutation `N=16128` | both lower parity gates pass | `NOT CLEARED` | launch only after both gates pass | remaining variants or ranking |
| Launch remaining three `N=16128` variants | repaired-permutation resource/validity row passes | `NOT CLEARED` | run one seed for the three named variants | statistical comparison |
| Promote method/default/HMC/NeuTra status | no promotion criterion exists in this campaign | `OUT OF SCOPE` | none | any promotion claim |

## Scientific Gates and Nonclaims

Successful completion establishes tested finite-program parity and large-count
feasibility in this scope only. It does not establish:

- an exact Austria-SIR observed-data score;
- statistical superiority or a ranking among SQMC variants;
- a formal SQMC variance rate;
- 16-seed behavior at `N=16128`;
- HMC, NeuTra, production, or default readiness; or
- equivalence to the separate annealed streaming OT algorithm.

Stop on wrong equations, hash/input mismatch, parity failure, nonfinite output,
invalid reset, TV failure, saturation, ancestry failure, corrupted artifact,
OOM outside the recorded budget, or another gateway failure before process
creation.

## Worktree Warning

The worktree contains many unrelated modified and untracked files belonging to
other active research tasks. Preserve them. Do not use `git reset --hard`,
`git checkout --`, broad `git restore`, broad cleanup, or a blanket stash. The
streaming files themselves are currently untracked relative to HEAD, so Git
HEAD alone does not preserve this work. Verify their presence and hashes after
handoff before running anything. Do not commit or publish unless separately
asked. A commit would not automatically include these untracked files.

## Required Terminal Documentation

After the last authorized run or the first true continuation veto, update
`docs/plans/bayesfilter-genut-sqmc-streaming-n16128-result-2026-08-18.md` and
this memo. Record:

- the exact command, environment, Git HEAD, selected physical GPU and logical
  device, TF32/XLA state, memory policy, TensorFlow version, source hashes,
  seed, wall time, and artifact path for every attempt;
- whether each failure happened before process creation, during TensorFlow
  initialization, during XLA compile, during execution, or during artifact
  validation;
- paired provenance checks and absolute/relative value and score differences;
- hard-veto status, viable rows, and whether any ranking has statistical
  support (`no` under the authorized one-seed scope);
- descriptive-only diagnostics, including TV, saturation, ancestry, allocator
  peak, and runtime;
- the decision, next justified action, strongest alternative explanation,
  weakest evidence, what would overturn the decision, and all nonclaims.

If only one seed is run, continuous differences remain descriptive. A row that
passes hard validity screens is a viable feasibility row, not evidence that it
is better than another route.

## Exact First Message for Claude Code

Use this exact handoff:

```text
Resume the bounded GenUT SQMC exact-streaming N=16128 campaign from
docs/plans/bayesfilter-genut-sqmc-streaming-n16128-reboot-reset-memo-2026-08-18.md.
Read AGENTS.md, that memo, its plan, and its result note before acting. Preserve
the dirty worktree and all historical artifacts; do not reset, restore, stash,
clean, checkout, overwrite attempts 01-03, install packages, or change the
environment. Codex's latest trusted nvidia-smi passed on physical GPU1, but the
TensorFlow probe was rejected before process creation because the approval
reviewer model returned 404. Treat that as a gateway failure, not a CUDA result.
First rerun nvidia-smi and the exact TensorFlow GPU/memory-growth/matmul probe
with trusted permissions. Prefer physical GPU1 when it remains below 50%
utilization with more than 8 GiB free. Then rerun the CPU-hidden focused tests
and compile checks. Do not launch N=16128 until fresh current-source dense and
streamed N=1008 and N=4032 pairs pass every provenance, numerical, validity,
memory-policy, and ancestry gate in the memo. Attempts 01-03 predate the final
one-block repair and are historical diagnostics only. Use fresh attempt
directories, stop on the first continuation veto, preserve the existing budget
and one-seed scope, and update the result/reset memo with exact evidence and
nonclaims.
```
