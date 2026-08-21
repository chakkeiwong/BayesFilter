# Claude Code handoff: SSL-LSTM q=20 NeuTra global mixing (2026-08-19)

## Handoff status

This memo is the complete continuation handoff for the SSL-LSTM q=20 NeuTra
global-mixing lane.  It is intentionally self-contained.  The current worktree
is shared with other agents and is dirty; preserve unrelated changes.

The immediate blocker is execution infrastructure, not a scientific result:

- a trusted GPU probe was rejected twice by the approval gateway with
  `404 Not Found: model is not available. model: gpt-5.6-luna`;
- a normal managed-session retry hid CUDA and correctly failed closed at the
  repository GPU memory-growth policy;
- no GPU training or HMC artifact was created.

Do not interpret either failure as evidence about TensorFlow, CUDA, the target,
or NeuTra.  Retry the trusted boundary only when the gateway is available.

## Research question

The scientific question is:

> Can a target-trained NeuTra transport make the exact SSL-LSTM q=20 pullback
> target globally traversable by one fixed-HMC kernel, without requiring an
> independently converged global posterior archive as training input?

The objective is not merely to obtain chains near each known mode.  The desired
object is one Markov chain kernel invariant for the exact transformed target,
with initialization forgotten and all material modes traversed.  Only that one
common-kernel run can provide retained posterior draws.

## Critical mathematical correction

Let `theta = T_phi(z)` be a bijective transport.  The exact transformed target
is

```text
pi_phi(z) proportional to pi_theta(T_phi(z)) * |det J_T_phi(z)|.
```

Fixed reversible HMC with a Metropolis correction leaves `pi_phi` invariant.
However, if chain `j` remains in region `A_j`, its empirical distribution is
`pi_phi(. | A_j)`.  Pooling mode-locked chains gives

```text
J^{-1} sum_j pi_phi(. | A_j),
```

where the weights are set by chain count and initialization, not by the target.
This is not generally `pi_phi`.

Therefore:

1. Mode-specific starts are allowed only as overdispersed initialization
   diagnostics.
2. Balanced occupancy caused by those starts is not a posterior-weight result.
3. Conditional chains that never transition must not be pooled.
4. Promotion requires one common exact pullback target/kernel to forget starts,
   cross all declared material regions, and pass modern R-hat/ESS/status gates.
5. A mode-locked result is a transport/HMC failure, not a posterior sample.

The new anti-pooling diagnostic enforces this distinction.

## Why a global posterior archive is not an input requirement

Forward KL is

```text
KL(pi || q_phi) = E_pi[-log q_phi] + constant.
```

For a full-support proposal `r`, target queries form a finite weighted estimator:

```text
w_i = exp(log-tilde-pi(theta_i) - log-r(theta_i))
E_pi[f] ~= sum_i w_i f(theta_i) / sum_i w_i.
```

Those rows are training evidence, not an independent posterior archive.  They
can be imperfect without changing the exact HMC invariant target: after the
transport is frozen, HMC still evaluates the original target and Jacobian.
Poor replay coverage may make the learned coordinates inefficient, so support
and global mixing remain explicit gates.

The previous 2026-08-18 plan incorrectly required an eligible global physical
posterior archive before NeuTra training.  That was circular and has been
superseded.

## Binding plans and notes

Primary corrected plan:

`docs/plans/bayesfilter-ssl-lstm-q20-neutra-global-mixing-repair-plan-2026-08-19.md`

Interim result:

`docs/plans/bayesfilter-ssl-lstm-q20-neutra-global-mixing-repair-interim-result-2026-08-19.md`

Reset memo:

`docs/plans/bayesfilter-ssl-lstm-q20-neutra-global-mixing-repair-reset-memo-2026-08-19.md`

The old plan/result/reset memo remain historical evidence, with the circular
boundary explicitly corrected:

- `docs/plans/bayesfilter-ssl-lstm-q20-gap-closure-plan-2026-08-18.md`
- `docs/plans/bayesfilter-ssl-lstm-q20-gap-closure-result-2026-08-18.md`
- `docs/plans/bayesfilter-ssl-lstm-q20-gap-closure-reset-memo-2026-08-18.md`

## Current code and artifact inventory

New code in this lane:

- `bayesfilter/inference/neutra_global_mixing.py`
  - `assess_retained_mode_mixing(region_labels, region_count, ...)`
  - expects integer labels shaped `[chain, retained_draw]`;
  - rejects invalid labels, chains that do not visit every region, and chains
    with no retained-state transition;
  - explicitly reports that it is a coverage diagnostic, not convergence proof.
- `tests/test_ssl_lstm_q20_neutra_global_mixing.py`
  - anti-pooling tests;
  - invalid-label and shape fail-closed tests;
  - GPU runner source checks for memory growth and correct draw/chain axis
    handling;
  - disjoint replay-bank and nonclaim checks.
- `docs/benchmarks/run_ssl_lstm_q20_neutra_global_mixing_replay_canary_2026_08_19.py`
  - CPU-only/XLA engineering smoke;
  - generates a small two-local-Gaussian proposal batch;
  - evaluates exact q=20 target/status;
  - runs weighted forward-KL updates;
  - binds the trained map to the exact pullback adapter at both known
    representatives;
  - no HMC or posterior claim.
- `docs/benchmarks/run_ssl_lstm_q20_neutra_weighted_replay_gpu_canary_2026_08_19.py`
  - prepared GPU/XLA canary;
  - loads eight existing terminal SMC populations as weighted replay only;
  - trains a weighted dense IAF;
  - constructs the exact pullback adapter;
  - runs a short common-kernel fixed-HMC screen;
  - applies the anti-pooling mode coverage diagnostic;
  - requires `TF_FORCE_GPU_ALLOW_GROWTH=true` and repository memory-growth
    verification before any GPU work.

Current CPU artifacts:

- `docs/plans/artifacts/ssl-lstm-q20-neutra-global-mixing-2026-08-19/replay-canary-r1/`
- `docs/plans/artifacts/ssl-lstm-q20-neutra-global-mixing-2026-08-19/replay-canary-r2/`

Use `r2`; `r1` predates the exact pullback parity addition.

## Exact CPU smoke result

Artifact:

`docs/plans/artifacts/ssl-lstm-q20-neutra-global-mixing-2026-08-19/replay-canary-r2/result.json`

Artifact SHA-256:

`afc665b884cf28de85f0572cfc2c65e586d1f93e101c48a49d309e640e132023`

Measured result:

| Quantity | Result | Role |
|---|---:|---|
| Proposal rows | 32 | Engineering smoke only |
| Valid target rows | 32/32 | Hard engineering screen |
| Proposal-weight ESS fraction | `0.5050560933` | Explanatory/support diagnostic |
| Target values finite | Yes | Hard engineering screen |
| Transport validation finite | Yes | Hard engineering screen |
| Weighted update | One finite update, step 1 | Mechanics only |
| Held-out weighted NLL | `18.5547412093` | Descriptive only |
| Gradient norm | `27.8291528829` | Descriptive only |
| Exact pullback value residual | `3.3584246495e-15` | Engineering parity |
| Pullback parity tolerance | `1e-10` | Reviewed engineering tolerance |
| Target status at representatives | Valid | Hard engineering screen |
| Values/scores finite at representatives | Yes | Hard engineering screen |
| XLA | Enabled | Execution provenance |
| GPU | Intentionally hidden (`CUDA_VISIBLE_DEVICES=-1`) | CPU diagnostic exception |
| Internal measured wall | `12.587055577 s` | Runtime description only |

Nonclaims in the artifact: not a posterior archive, not global mode discovery,
not HMC evidence, and not predictive equivalence.

The CPU smoke does not show that the transport is useful.  It only shows that
the target, weighted update, frozen map, Jacobian, and explicit score pullback
are wired consistently on a tiny batch.

## Existing SMC replay source

The prepared GPU canary uses the eight central terminal populations from:

`docs/plans/artifacts/ssl-lstm-q20-physical-annealed-smc-repair-2026-08-10/r2/central-00/` through `central-07/`

For each child, the runner discovers the highest terminal
`stage-XX-pre-theta.tftensor` and matching
`stage-XX-pre-normalized_weights.tftensor`, verifies shape `[100,4]` / `[100]`,
finite values, and normalized weights.  The current terminal stages are:

| Child | Terminal stage | ESS fraction | Max normalized weight |
|---:|---:|---:|---:|
| central-00 | 04 | `0.9316374` | `0.0335183` |
| central-01 | 03 | `0.9172212` | `0.0267945` |
| central-02 | 05 | `0.9999585` | `0.0102557` |
| central-03 | 03 | `0.9657693` | `0.0233593` |
| central-04 | 03 | `0.9310583` | `0.0299552` |
| central-05 | 03 | `0.9376688` | `0.0239921` |
| central-06 | 04 | `0.9795339` | `0.0189275` |
| central-07 | 04 | `0.8783133` | `0.0385412` |

All eight terminal tensor hashes match their corresponding stage JSON receipts.
The replay is concatenated as 800 rows.  Each child contributes one-eighth of
the aggregate measure by subtracting `log(8)` from its within-child log weight.
This is an optimization-bank construction, not a posterior authority.

Historical SMC result:

`docs/plans/bayesfilter-ssl-lstm-q20-physical-annealed-smc-material-result-2026-08-10.md`

Its scope is only relative mass over the two known proposal-supported sign
regions.  It does not prove exhaustive mode discovery, HMC stationarity, or a
full posterior.  Historical central mean is approximately `0.47087`, with
95% independent-batch interval `[0.40573, 0.53602]`; keep this as a comparator
only.  Do not use SMC occupancy or its weights as HMC mode weights.

## Exact SSL-LSTM target identity

Target constructor:

`bayesfilter.nonlinear.ssl_lstm_complexity_batched_target_tf:batch_native_complexity_posterior_target`

Use:

```python
target = batch_native_complexity_posterior_target(
    20, jit_compile=True, principal_sqrt_backend="tensorflow_eigh"
)
```

Expected target signature:

`9a86e60081f1b9cd288dbdb1dcbe1e9a5b5e23d9b5ef97afdb72ee95c23d7278`

Expected adapter signature:

`a8be6c212eb2a74ef926ccb9279871805949cae6e00c312a12525141638166f3`

The target is batch-native, graph-native, float64, and XLA-capable when created
with the above settings.  It exposes:

- `neutra_batch_log_prob_and_grad_status(theta)` for rank-2 target rows;
- status fields including `status_code` and
  `valid_pre_regularized_score`;
- no NumPy runtime path in the target implementation;
- no scalar/map fallback for the NeuTra training route.

Do not silently change `principal_sqrt_backend`, dtype, q, or JIT policy.  Any
change creates a new target scope and must be recorded.

## Existing geometry warm start

Geometry artifact:

`docs/plans/artifacts/ssl-lstm-q20-seed-b-neutra-mode-failure-root-cause-2026-08-10/r1/geometry.json`

SHA-256:

`dc3dd7b84566867bc49c11ad16f50778d21457adbb398a17c2a75f3c3b461eeb`

Known stationary representatives:

- plus: approximately
  `(0.7331136936, 0.1727323775, 0.5894250950, 0.1589205953)`;
- minus: approximately
  `(0.4466756367, -0.2413180384, -0.5876966020, 0.1198904131)`.

The geometry also contains source local precision matrices.  These are used to
construct a two-region Gaussian proposal only as a warm start.  They are not
posterior covariance or mode-weight authority.

Historical root-cause facts:

- old reverse-KL proposal had essentially no negative-region coverage (`3/100000`
  frozen base draws in the cited diagnostic);
- old negative representative mapped to a very remote latent tail;
- transformed representatives were approximately `23.707` latent units apart;
- negative local precision was much larger than positive (approximately `91.72`
  versus `1.16` maximum eigenvalue in the cited diagnostic);
- all old HMC starts were positive-region starts;
- old selected step size was locally unusable in the negative region;
- reducing the step size repaired local negative motion but did not solve the
  latent separation.

These facts explain why more old reverse-KL updates or more positive-mode HMC
transitions are not a repair.

## New anti-pooling diagnostic contract

Implementation:

`bayesfilter/inference/neutra_global_mixing.py`

API:

```python
report = assess_retained_mode_mixing(
    region_labels,                # int32 [chain, retained_draw]
    region_count=2,
    minimum_transitions_per_chain=1,
)
```

The report requires:

- valid labels in `[0, region_count)`;
- every chain has at least one retained draw in every declared region;
- every chain has at least one adjacent retained-label transition.

`report.passed` is only a canary coverage screen.  It must be combined with:

- finite state/target/Jacobian/score;
- target-status validity;
- modern rank-normalized split and folded R-hat;
- declared bulk/tail ESS;
- warm-up exclusion;
- exact target/transport identity and artifact provenance.

Focused tests currently pass:

```bash
CUDA_VISIBLE_DEVICES=-1 TF_CPP_MIN_LOG_LEVEL=3 \
python -m pytest -q \
  tests/test_ssl_lstm_q20_neutra_global_mixing.py \
  tests/test_ssl_lstm_q20_gap_closure_campaign.py \
  tests/test_ssl_lstm_q20_gap_closure_mode_discovery.py
```

Observed: `11 passed`.

The test cases include:

- balanced pooled labels from two mode-locked chains -> reject;
- one common kernel with crossings in every chain -> coverage pass;
- invalid label -> reject;
- rank/threshold contract failures -> raise;
- GPU runner axis-order and memory-growth source checks;
- disjoint replay-bank and nonclaim checks.

## Prepared GPU runner: review before launch

Runner:

`docs/benchmarks/run_ssl_lstm_q20_neutra_weighted_replay_gpu_canary_2026_08_19.py`

Intended command:

```bash
TF_FORCE_GPU_ALLOW_GROWTH=true \
/home/ubuntu/anaconda3/envs/tfgpu/bin/python \
docs/benchmarks/run_ssl_lstm_q20_neutra_weighted_replay_gpu_canary_2026_08_19.py \
  --device 1 \
  --updates 20 \
  --hidden-width 32 \
  --stages 3 \
  --seed 1 \
  --output-root \
  docs/plans/artifacts/ssl-lstm-q20-neutra-global-mixing-2026-08-19/gpu-canary-r2
```

Launch requirements:

1. Set `TF_FORCE_GPU_ALLOW_GROWTH=true` before Python/TensorFlow import.
2. Set `CUDA_VISIBLE_DEVICES=1` before import; the script also sets it before
   importing TensorFlow.
3. Use trusted/escalated GPU execution under the GPU policy.
4. Call `configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)` before
   listing logical devices or any tensor/operation initialization.
5. Require exactly one visible logical GPU.
6. Disable TF32; use float64 and XLA.
7. Write only to the unique versioned output root.
8. Preserve the failed gateway attempt as infrastructure evidence; do not make
   a CPU fallback look like a GPU result.

Runner design:

- train rows: first 600 concatenated replay rows;
- selection rows: rows 600:700;
- audit rows: rows 700:800;
- weighted IAF: configurable width/stages, default `(32,32)`, 3 stages, tanh;
- learning rate: `1e-3` inherited as a target-specific warm-start hypothesis,
  not a promoted default;
- updates: 20 for the canary;
- HMC: 64 retained + 64 burn-in, fixed `L=5`, step `0.10`, no NUTS;
- starts: mapped known representatives plus small local perturbations, solely
  to test initialization forgetting;
- no equal pooling if chains remain locked;
- output includes HMC diagnostics, mode labels, mixing report, transport state,
  GPU memory policy, XLA/TF32/dtype, target signatures, replay receipts, and
  nonclaims.

### Important runner review issue

Before launch, inspect the target adapter capability path.  The runner uses
`BatchNativeBoundAdapter` and `FixedTransportValueScoreAdapter`; verify that the
base adapter exposes `log_prob_and_grad` and status telemetry in the exact form
required by `FixedTransportValueScoreAdapter`.  The CPU smoke established this
path successfully at the two representatives, but the GPU runner has not run.

Also verify that HMC result shape is `[draw, chain, parameter]` before interpreting
mode labels.  The current runner explicitly transposes labels after flattening;
do not replace this with a direct `[chain, draw]` reshape.

### Expected interpretation

- If the runner fails before writing an artifact due to no GPU/memory policy,
  classify as infrastructure only.
- If target rows or transport values are nonfinite, classify as numerical/
  target-support failure and preserve exact diagnostics.
- If HMC is finite but chains remain mode-locked, reject the transport candidate;
  do not pool chains.
- If every chain crosses both signs in this short canary, call it only a
  mechanics nomination.  It does not establish convergence.
- Do not launch a material 2,000/1,000 HMC campaign from this canary without a
  reviewed capacity/seed screen and declared convergence gates.

## Trusted GPU approval failure evidence

Two trusted attempts were made.

Attempt 1: the full GPU canary command was rejected before process creation by
the approval service:

```text
CreateProcess rejected: Automatic approval review failed:
unexpected status 404 Not Found: model is not available.
model: gpt-5.6-luna
```

Attempt 2: a minimal trusted TensorFlow GPU probe with GPU 1 and memory growth
was rejected with the same gateway error:

```text
TF_FORCE_GPU_ALLOW_GROWTH=true CUDA_VISIBLE_DEVICES=1 \
/home/ubuntu/anaconda3/envs/tfgpu/bin/python -c "..."
```

The normal managed-session retry reached the process but saw no CUDA device and
failed closed in `configure_tensorflow_gpu_memory_growth`, which is the correct
policy behavior.  Do not try to bypass the gateway by changing the model,
removing escalation, or treating an untrusted CPU process as GPU evidence.

## Verification already completed

Syntax and focused tests:

```bash
python -m py_compile \
  bayesfilter/inference/neutra_global_mixing.py \
  tests/test_ssl_lstm_q20_neutra_global_mixing.py \
  docs/benchmarks/run_ssl_lstm_q20_neutra_global_mixing_replay_canary_2026_08_19.py \
  docs/benchmarks/run_ssl_lstm_q20_neutra_weighted_replay_gpu_canary_2026_08_19.py

CUDA_VISIBLE_DEVICES=-1 TF_CPP_MIN_LOG_LEVEL=3 \
python -m pytest -q \
  tests/test_ssl_lstm_q20_neutra_global_mixing.py \
  tests/test_ssl_lstm_q20_gap_closure_campaign.py \
  tests/test_ssl_lstm_q20_gap_closure_mode_discovery.py

git diff --check
```

Result: `11 passed`; `py_compile` passed; `git diff --check` passed for this
lane.

## Source hashes at handoff

These hashes are informational provenance for the current dirty worktree.  Do
not use them as a substitute for a commit or artifact manifest.

| File | SHA-256 |
|---|---|
| `bayesfilter/inference/neutra_global_mixing.py` | `ea1548d3986ea0ddc283a04289856c09f34fd726692a415b0dbcfc8d7b075af8` |
| `tests/test_ssl_lstm_q20_neutra_global_mixing.py` | `c06d051cea08557090b26e336488219db59c61b0f9912ac7700ee2e7ba1aa7e3` |
| CPU replay runner | `22329470ff2493c855c6d74c991f95d8732161b30bbaf343a6d841915a1394a7` |
| GPU replay runner | `0b90f4882976b0bdc61af97390b12d9e5eae46b6d1a9a1c8dbfb9d19ab8debd6` |
| Corrected plan | `532f38233152bf50358ab1cbdae76434c7d52191686ea2aff75c9b2593dd6c56` |
| Interim result | `b81b4b5b2e97705aa37ea55a520a60cbf661c52b2a27c2701cbc739fc57b5692` |
| Reset memo | `d5ded09c7b9cd21210e8d3286c785034ac69d00e4c02f68e2400ebc0e2edd19b` |
| CPU r2 result | `afc665b884cf28de85f0572cfc2c65e586d1f93e101c48a49d309e640e132023` |

Current `HEAD` when this memo was written:

`5699dafec23de9549a8092bec638997e7973593c`

The worktree has unrelated concurrent modifications.  Do not reset, clean,
checkout, or revert them.

## Claude execution protocol

Use bounded, read-only source review first.  Suggested first review targets:

1. this handoff memo;
2. the GPU runner;
3. `bayesfilter/inference/batched_value_score.py`;
4. `bayesfilter/inference/neutra_end_to_end.py`;
5. `bayesfilter/nonlinear/ssl_lstm_complexity_batched_target_tf.py`.

Review questions:

- Does the GPU runner bind the exact target and frozen transport without a
  scalar/map fallback?
- Does the HMC target include the original value plus the transport Jacobian and
  the full explicit score pullback?
- Are all status/finite checks preserved through the HMC trace?
- Is the chain/draw axis handling correct?
- Does the runner accidentally turn replay or initialization labels into mode
  weights?

After review, run only the minimal trusted GPU canary.  If it completes, write a
result note before any material run.  If it fails at the gateway, record the
exact gateway error and stop the GPU lane; do not improvise a bypass.

## Forbidden shortcuts

- Do not restore the circular “global archive before NeuTra” gate.
- Do not pool mode-specific conditional chains.
- Do not call SMC replay rows posterior samples.
- Do not infer global mixing from acceptance, loss, replay ESS, or pooled
  occupancy.
- Do not use old seed-B positive-mode NeuTra draws as global evidence.
- Do not use dense physical warm-up states as posterior or training samples.
- Do not use NUTS.
- Do not tune or run `L=1`; `L >= 2` only.
- Do not claim native TFP divergence counts when the kernel does not expose
  them; report `not_exposed_by_kernel`.
- Do not change target q, dtype, XLA, principal-square-root backend, or memory
  policy without a new reviewed scope.

## Success boundary for this handoff

This handoff is complete when Claude has either:

1. run the trusted GPU canary and written a versioned result note classifying it
   as mechanics nomination, candidate rejection, or infrastructure failure; or
2. documented that the approval gateway remains unavailable, with no scientific
   claim and a clean reset memo preserving the exact next command.

A passing canary is not the end of the SSL-LSTM task.  The subsequent sequence is:

1. target-specific capacity/learning-rate/seed screen;
2. disjoint audit and frozen transport artifact;
3. one common exact pullback fixed-HMC tuning grid;
4. global mixing, modern R-hat/ESS, status, and warm-up exclusion gates;
5. only after admission, posterior-predictive output-law tests with one posterior
   parameter draw per path, `n=1000`, `T=10,20,30,50,100`, five separate 1%
   tests, and no omnibus test.

