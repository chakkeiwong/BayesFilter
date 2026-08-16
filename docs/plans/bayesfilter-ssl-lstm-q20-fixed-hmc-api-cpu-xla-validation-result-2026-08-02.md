# q=20 Fixed-Transport TFP HMC API CPU/XLA Validation Result

Date: 2026-08-02  
Plan: `docs/plans/bayesfilter-ssl-lstm-q20-fixed-hmc-api-cpu-xla-validation-plan-2026-08-02.md`  
Status: `TUNING_COMPLETED_NO_KERNEL_ADMITTED_SEQUENTIAL_NOT_LAUNCHED`

## Outcome

The corrected BayesFilter fixed-transport tuning campaign executed successfully
for both trained q=20 charts. Neither chart admitted an `L=2` fixed HMC kernel
under the predeclared fresh-screen acceptance band `[0.65,0.75]`. The
sequential warm-up/retained HMC phase was therefore not launched.

This is a clean tuning failure, not an implementation, target, data, or
artifact failure. It does not establish that fixed HMC, either trained
transport, or the q=20 target is invalid. It establishes only that this
target-specific dual-averaging ladder did not produce a frozen kernel eligible
for the planned sequential test.

## Evidence Contract Result

| Evidence role | Result |
| --- | --- |
| Engineering question | Passed: public TensorFlow-only API, rank-2 four-chain target, shared scalar dual averaging, CPU/XLA execution, fixed zero initialization, target telemetry, and artifact writing all executed. |
| Kernel promotion criterion | Failed for both charts: no fresh 16-draw screen was inside `[0.65,0.75]`. |
| Hard veto screen | No nonfinite state/target/score/log acceptance and no invalid target status were reported. TFP native divergence was unavailable, recorded as unavailable/null, and not treated as zero or as a veto. |
| Sequential admission | Not checked. No kernel passed tuning, so the predeclared gate prohibited sequential launch. |
| Scientific interpretation | No posterior-validity, convergence, chart-ranking, model-adequacy, GPU-equivalence, or default-readiness conclusion. |

## Tuning Results

All values below are descriptive. The screens are short and have no uncertainty
interval, so differences must not be ranked.

| Chart | DA budgets | Tuned step sizes | Fresh screen acceptance | Final decision |
| --- | --- | --- | --- | --- |
| A | `8,16,32` | `0.647273, 0.540312, 0.589960` | `0.781646, 0.637710, 0.843694` | `NO_VIABLE_KERNEL` |
| B | `8,16,32` | `0.673863, 0.528120, 0.593117` | `0.829631, 0.939211, 0.817549` | `NO_VIABLE_KERNEL` |

For Chart A, the observed screens appeared on both sides of the acceptance
band. For Chart B, all observed screens were above it. Because the screens used
only 16 draws and fresh seeds, the nonmonotone Chart A values are compatible
with substantial Monte Carlo noise; they do not support a deterministic
step-size ordering.

Every ladder round recorded:

- `jit_compile=true`, TensorFlow `2.20.0`, TFP `0.25.0`, FP64;
- one rank-2 `[4,4]` chain bank with one shared scalar step;
- finite samples, target values, proposed target values, target scores, and log
  acceptance;
- valid per-transition target-status telemetry; and
- native divergence `not_exposed_by_kernel` with count `null`.

Finite log-accept/energy-proxy tails were explanatory only and did not veto a
candidate.

## Decision Table

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Reject the current Chart A tuning handoff | Failed acceptance screen | No hard mechanics veto | 16-draw screen noise and unstable DA handoff | Fresh-seed, denser fixed-step repair around the observed transition region, with longer screens | Chart A transport or fixed HMC invalidity |
| Reject the current Chart B tuning handoff | Failed acceptance screen | No hard mechanics veto | All tested handoffs screened above band | Fresh-seed step repair extending to larger steps, with longer screens | Chart B transport or fixed HMC invalidity |
| Do not launch sequential HMC | No admitted frozen kernel | Gate operated correctly | Sequential behavior remains unknown | Tune first; launch the tested four-process callback only after admission | Convergence, ESS, posterior validity, or HMC readiness |

## Inference Status

| Evidence class | Status |
| --- | --- |
| Hard veto screen | No hard numerical/status veto in completed tuning runs. Native divergence unavailable. |
| Statistically supported ranking | None. |
| Descriptive-only differences | Tuned steps, acceptance values, runtimes, and finite log-accept tails. |
| Default-readiness | Not ready. |
| Next evidence needed | Target-specific fixed-step repair with fresh seeds and longer screens, followed by fresh kernel verification and the full sequential R-hat/ESS policy. |

## Run Manifest

| Field | Value |
| --- | --- |
| Git commit at run | `b370dc89e6e79f3853e0fccd5ab5b4fa2cb9065d` |
| Worktree | Dirty; manifests preserve the exact status. Unrelated concurrent changes were not reverted. |
| Environment | `/home/ubuntu/anaconda3/envs/tfgpu`, Python `3.13.13`, TensorFlow `2.20.0`, TFP `0.25.0` |
| Hardware | CPU-only by explicit exception; CUDA hidden; physical GPU list empty |
| XLA | Enabled; both logs record an XLA-compiled cluster |
| Chart A affinity | CPUs `0..15` |
| Chart B affinity | CPUs `16..31` |
| Supervisor affinity | CPU `32` |
| Dtype | `float64` |
| Tuning seeds | API bases: tune `(20260625,100)`, screen `(20260625,200)`, verification `(20260625,300)`, with API-owned offsets |
| Chart A checkpoint | `checkpoint-1500.json`, SHA-256 `c87ee24874705bb12296cc05b82310326579694cc04c2a3682792f9bf18fb9ff` |
| Chart B checkpoint | `checkpoint-2500.json`, SHA-256 `849e33855d87dc34644e15757942bf872937d9f4d4b00a4f03855661827d761d` |
| Target signature | `9a86e60081f1b9cd288dbdb1dcbe1e9a5b5e23d9b5ef97afdb72ee95c23d7278` |
| Chart A transport hash | `caf6c9ec1a46d04253b2ae3922d83e619f38c824cea955d5da8ac419d2dfed7f` |
| Chart B transport hash | `09eb181289cdbe44a6ef8d9a423a64246a8bd557e08c013c3a6e76cd0a461ab0` |
| Tuning wall time | `2278.2646 s` concurrent supervisor wall; Chart A `1806.6523 s`, Chart B `2273.5219 s` |
| Prior conservative canary charge | `1900 s` |
| Cumulative campaign charge | `4178.2646 s` of `20000 s` |
| Unused authorized budget | `15821.7354 s` |
| RSS | Not recorded by this tuning harness; no value is inferred. |
| Tuning command | `taskset -c 32 timeout 3600s /home/ubuntu/anaconda3/envs/tfgpu/bin/python docs/benchmarks/run_ssl_lstm_q20_fixed_hmc_api_cpu_xla_validation_2026_08_02.py --mode tuning-supervisor --output-root docs/plans/artifacts/ssl-lstm-q20-fixed-hmc-api-cpu-xla-validation-2026-08-02/r4-tuning --cap-seconds 3600` |
| Tuning artifact | `docs/plans/artifacts/ssl-lstm-q20-fixed-hmc-api-cpu-xla-validation-2026-08-02/r4-tuning/summary.json`, SHA-256 `71f5498ee618babd1dc2ebe34402f7ec41e5b9467979894802d832bc2a6b1e34` |
| Chart A tuning artifact | `docs/plans/artifacts/ssl-lstm-q20-fixed-hmc-api-cpu-xla-validation-2026-08-02/r4-tuning/chart-a/tuning-result.json`, SHA-256 `e375284b8ebe6d494a5ebc38ee37c53679bada5ec8f696e1e019d220564d250c` |
| Chart B tuning artifact | `docs/plans/artifacts/ssl-lstm-q20-fixed-hmc-api-cpu-xla-validation-2026-08-02/r4-tuning/chart-b/tuning-result.json`, SHA-256 `e3622bd9c0cc302782f393e5b4a72347e8aa50c305f70faf532b09309354cbc2` |
| Sequential non-launch artifact | `docs/plans/artifacts/ssl-lstm-q20-fixed-hmc-api-cpu-xla-validation-2026-08-02/r5-sequential-not-launched/summary.json`, SHA-256 `31cce032700ccee51bd50d9b0e7a55d31ba062f5bac6294de124e26080cc86c0` |
| Tests | `36 passed` in `19.29 s` across tuner, controller, target status, and distributed callback contract tests |

## Engineering Ledger

- Corrected unavailable-native-divergence and finite-energy-tail tests now use
  the deterministic IID archive fixture, so the policy assertions are not
  confounded by an intentional modern R-hat failure.
- The campaign harness now contains a persistent four-process frozen-chain
  callback with deterministic per-chain seed folding, exact chain-axis tensor
  reassembly, admitted-kernel identity checks, and bounded supervisors.
- Four focused callback tests passed. The real target callback was not launched
  end to end because no kernel was admitted; that limitation is preserved.
- No TFP NUTS route was used or launched.

## Post-Run Red Team

Strongest alternative explanation: the 16-draw fresh screens are noisy enough
that a usable step may exist near the observed handoffs even though none landed
inside the narrow band. Chart A's nonmonotone screen values reinforce this
explanation.

What would overturn the decision: a fresh, longer fixed-step screen and
verification that passes `[0.65,0.75]` with finite required telemetry and no
available positive native divergence. Only then should sequential warm-up and
retained sampling start.

Weakest evidence: no uncertainty interval was computed for screen acceptance,
and TFP HMC exposes no native divergence boolean. Consequently, this result
supports rejection of the current tuning handoffs only, not a broad sampler or
transport conclusion.
