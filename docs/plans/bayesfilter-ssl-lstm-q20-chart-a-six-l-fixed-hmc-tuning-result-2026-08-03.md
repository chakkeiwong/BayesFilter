# q=20 Chart A Six-L Fixed-HMC Tuning Result

Date: 2026-08-03  
Plan: `docs/plans/bayesfilter-ssl-lstm-q20-chart-a-six-l-fixed-hmc-tuning-plan-2026-08-03.md`  
Status: `GRID_COMPLETED_CANDIDATE_NOMINATED`

## Outcome

The complete Chart A grid `(5,10,15,20,25,3)` finished successfully under the
12-hour cap. `L=10` was the only arm that passed both the public BayesFilter
tuner's fresh screen and its 64-result fresh verification:

```text
num_leapfrog_steps = 10
step_size = 0.4148806556986277
fresh verification mean acceptance probability = 0.7235869085131437
fresh verification binary acceptance rate = 0.7265625
```

The required state, target value, proposed target value, target score, and log
acceptance tensors were finite. Target-status telemetry was valid for every
recorded transition. TFP native divergence remained unavailable and is recorded
as unavailable, not zero.

This is a valid Chart A fixed-HMC kernel candidate. It is not an accepted
posterior result: no sequential warm-up, R-hat, ESS, posterior, or downstream
scientific validation was launched.

## Grid Results

| L | Cores | Wall time | Terminal tuning result | Selected step | Fresh verification acceptance |
| ---: | ---: | ---: | --- | ---: | ---: |
| `3` | `6` | `1.434 h` | Verification rejected | `0.5460429` | `0.4955847` |
| `5` | `6` | `1.498 h` | No screen-qualified step | N/A | N/A |
| `10` | `6` | `4.723 h` | **Candidate passed tuner** | `0.4148807` | `0.7235869` |
| `15` | `8` | `7.067 h` | Verification rejected | `0.3956525` | `0.8156956` |
| `20` | `16` | `5.996 h` | No screen-qualified step | N/A | N/A |
| `25` | `16` | `7.487 h` | No screen-qualified step | N/A | N/A |

The supervisor wall time was `26959.741117871017 s` (`7.488817 h`). All six
workers exited with code zero, and the supervisor did not time out.

Each arm is one stochastic realization without an uncertainty interval.
Consequently, runtime and acceptance differences are descriptive only. The
result supports viability of `L=10`; it does not statistically rank trajectory
lengths.

## Screen Details

| L | Budget-8 screen | Budget-16 screen | Budget-32 screen | Interpretation |
| ---: | ---: | ---: | ---: | --- |
| `3` | `0.6456472` | `0.9177205` | `0.7272110` | Budget-32 nominated a step; fresh verification fell below band. |
| `5` | `0.8032337` | `0.8317476` | `0.8500920` | All screens above `[0.65,0.75]`. |
| `10` | `0.4968768` | `0.7679008` | `0.6520731` | Budget-32 nominated a step; fresh verification passed. |
| `15` | `0.3884154` | `0.8222900` | `0.7139709` | Budget-32 nominated a step; fresh verification rose above band. |
| `20` | `0.5465462` | `0.8175362` | `0.9253964` | No screen-qualified step. |
| `25` | `0.5735965` | `0.8631105` | `0.8327218` | No screen-qualified step. |

The screen-to-verification movement for `L=3` and `L=15` demonstrates material
finite-run noise. Their rejection is a tuning result, not evidence against the
fixed-HMC direction, target, transport, or model.

## Evidence Contract Result

| Evidence role | Result |
| --- | --- |
| Scientific question | Answered for Chart A: one candidate was produced from the six-`L` grid. |
| Candidate criterion | Passed only by `L=10`, step `0.4148806556986277`. |
| Candidate vetoes | Selected `L=10` candidate has no hard veto. Rejected arms retain their own acceptance vetoes or repair triggers. |
| Continuation vetoes | None: all processes completed, source/config identities matched, affinity/thread assignments matched, GPUs were hidden, and XLA compiled in every worker. |
| Explanatory only | Runtime, per-round acceptance, binary acceptance, and finite log-accept/energy tails. |
| Not concluded | Chart B behavior, convergence, ESS, posterior validity, model adequacy, sampler superiority, GPU equivalence, or default-readiness. |

The merged artifact's aggregate `hard_vetoes` includes
`verification_acceptance_outside_pass_band` because rejected `L=3` and `L=15`
arms are preserved. This aggregate field does not apply to the selected
candidate: the selected `L=10` candidate has `hard_vetoes=[]`.

## Large Finite Log-Accept Tail

The selected `L=10` fresh verification records
`max_abs_log_accept_ratio=1e100`. The value is finite; required sampler and
target tensors are finite; target status is valid; and mean acceptance is
inside `[0.65,0.75]`. Under the predeclared policy, this tail is an explanatory
alert only and does not veto the candidate.

It materially limits interpretation. The candidate must be tested by the
sequential HMC controller with chunk-level numerical/status gates before any
convergence or posterior claim. This tuning result alone does not establish
stable long-run energy behavior.

## Decision Table

| Decision | Primary criterion status | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Preserve `L=10`, step `0.4148806556986277` as the Chart A kernel candidate | Passed screen and fresh verification | No candidate hard veto | One finite-run verification and extreme finite log-accept tail | Run bounded sequential HMC validation using this frozen kernel | Convergence or posterior validity |
| Reject current `L=3` and `L=15` handoffs | Fresh verification outside acceptance band | Acceptance veto only | Screen-to-verification variability | Retain as step-repair evidence if `L=10` later fails | Invalidity of those trajectory lengths |
| Reject current `L=5,20,25` handoffs | No screen-qualified step | No numerical/status veto | Short screen noise and coarse DA handoff | Fresh-step repair only if later needed | Fixed-HMC direction rejection |
| Do not run Chart B now | Outside user-directed scope | Not evaluated | Chart-specific transport behavior unknown | Tune Chart B only if a Chart B candidate is needed | Cross-chart generality |

## Inference Status

| Evidence class | Status |
| --- | --- |
| Hard veto screen | Selected `L=10` candidate has no hard veto. Rejected arms preserve their own acceptance outcomes. |
| Statistically supported ranking | None. Only one arm passed; no uncertainty-supported ranking was performed. |
| Descriptive-only differences | Per-arm runtime, screen acceptance, tuned steps, and log-accept tails. |
| Default-readiness | Not ready. |
| Next evidence needed | Sequential warm-up and cumulative retained sampling with modern R-hat, ESS, finite/status, movement, and declared posterior gates. |

## Engineering, Numerical, And Scientific Ledgers

| Ledger | Status |
| --- | --- |
| Engineering correctness | Passed focused tests and terminal artifact assertions; all six sharded public-API calls completed and merged with canonical indices/seeds. |
| Numerical/sampler validity | Tuning candidate only. Required tensors/status passed for `L=10`; native divergence unavailable; long-run behavior not checked. |
| Scientific interpretation | No posterior or model claim. One trained Chart A transport has a fixed-HMC kernel candidate worth sequential testing. |

## Run Manifest

| Field | Value |
| --- | --- |
| Git commit | `b370dc89e6e79f3853e0fccd5ab5b4fa2cb9065d` |
| Worktree | Dirty with concurrent agent work; unrelated files were not modified or reverted. |
| Command | Exact command in the plan; supervisor cap `43200 s`, outer timeout `43500 s`. |
| Environment | `/home/ubuntu/anaconda3/envs/tfgpu`, TensorFlow `2.20.0`, TFP `0.25.0` |
| Hardware | CPU-only diagnostic lane; `CUDA_VISIBLE_DEVICES=-1`; every worker recorded `physical_gpus=[]` |
| XLA | Enabled; every worker log records `Compiled cluster using XLA!` |
| Dtype | `float64` |
| Chart | Chart A checkpoint `1500` |
| Checkpoint SHA-256 | `c87ee24874705bb12296cc05b82310326579694cc04c2a3682792f9bf18fb9ff` |
| Target signature | `9a86e60081f1b9cd288dbdb1dcbe1e9a5b5e23d9b5ef97afdb72ee95c23d7278` |
| Transport hash | `caf6c9ec1a46d04253b2ae3922d83e619f38c824cea955d5da8ac419d2dfed7f` |
| Canonical grid/order | `(5,10,15,20,25,3)`; `L=3` appended to preserve default-grid seed streams |
| Core allocation | `L5:6`, `L10:6`, `L15:8`, `L20:16`, `L25:16`, `L3:6`; 58 worker cores total |
| Supervisor | CPU `127` |
| Chains | Four chains in one rank-2 `[4,4]` batch; shared scalar step |
| DA budgets | `(8,16,32)` |
| Screen | 16 results, 4 burn-in |
| Fresh verification | 64 results, 16 burn-in |
| Acceptance policy | Target `0.70`; pass band `[0.65,0.75]` |
| Random streams | Canonical candidate-index tune/screen/verification offsets preserved by each shard and verified before merge |
| Supervisor wall | `26959.741117871017 s` (`7.488817 h`) |
| Maximum worker RSS | `3450695680` bytes (`L=15`) |
| Tests | `21 passed` across public fixed-transport tuner and q=20 campaign harness tests |
| Supervisor source SHA-256 | `219c5d59830c231f7ce9bd066efaab9864990e5f0c564d944f1c5d67bca881ff` |
| Base harness SHA-256 | `b288274860c1b8ec122c80ad7b8e67527d34eb0a741dd84f91ab96b202366d1a` |
| Public wrapper SHA-256 | `2e720f467e39b3fc7977b7fb127b8f91ec7c8a975fce1fe1fa87d175dea5ff6b` |
| TensorFlow tuner SHA-256 | `8a627c283084e1d90908b5a3bf731f3e4e862c9189179c3c1da371779c9cbd42` |
| Plan SHA-256 at launch | `f0cefe0fa7dd331e1f965aef4bad5652342bc355f7a11f9a7a8172b60852c7e8` |
| Supervisor artifact | `docs/plans/artifacts/ssl-lstm-q20-chart-a-six-l-fixed-hmc-tuning-2026-08-03/r1/summary.json`, SHA-256 `76e3c716c4ab1daf4a7e4fce6852f99df71ca4263b7c193b9962661c85cb5dce` |
| Merged candidate artifact | `docs/plans/artifacts/ssl-lstm-q20-chart-a-six-l-fixed-hmc-tuning-2026-08-03/r1/merged-tuning-result.json`, SHA-256 `c3018064fcbbe040b3510165138bc7db7de1b378dd0eb4c1a1b8155af796fb19` |
| Selected kernel hash | `34b89acd551dd25bee9dd0a463be67ff9d06f08ea3f970da5ffa97b44438ca4d` |
| Sequential HMC | Not launched |

## Budget Accounting

| Charge | Seconds |
| --- | ---: |
| Prior campaign charge | `6440.7955` |
| Six-arm concurrent grid | `26959.7411` |
| Cumulative charge | `33400.5366` |
| New grid wall cap | `43200.0000` |
| Unused grid allowance | `16240.2589` |

The new 12-hour grid authorization superseded the older remaining
`13559.2045 s` allowance for this one Chart A campaign; otherwise the run could not have
completed. No Chart B or sequential-HMC budget was consumed.

## Post-Run Red Team

Strongest alternative explanation: `L=10` passed because one 64-result fresh
verification happened to land in the acceptance band, while the extreme finite
log-accept tail signals behavior that may fail in longer chunks. The result is
therefore a candidate nomination, not validation.

What would overturn the candidate: a sequential chunk with a nonfinite required
tensor, invalid target status, available positive native divergence, failed
movement, or persistent failure of the predeclared warm-up/retained convergence
and ESS gates.

Weakest evidence: no sequential sampling, no R-hat or ESS, no posterior
comparison, one chart, one seed stream per arm, and no uncertainty interval for
acceptance or runtime.
