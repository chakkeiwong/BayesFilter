# q=20 Chart A HMC Reboot Reset Memo

Date: 2026-08-03  
Prepared: 2026-08-03T22:16:29+08:00  
Status: `SAFE_TO_REBOOT_NO_BAYESFILTER_RUN_ACTIVE`

## Read This First

No BayesFilter training, tuning, HMC, TensorFlow, or pytest process is active.
There is no run to resume after reboot.

The last scientifically usable state is the completed Chart A six-`L`
fixed-HMC tuning campaign. It nominated one candidate:

```text
L = 10
step_size = 0.4148806556986277
mass = fixed identity in trained-transport z coordinates
kernel_hash = 34b89acd551dd25bee9dd0a463be67ff9d06f08ea3f970da5ffa97b44438ca4d
```

This is a tuning-nominated candidate only. It has not completed sequential
warm-up, retained sampling, R-hat, ESS, posterior comparison, or downstream
scientific validation.

## Invalid Run Removed

A later 16-transition-per-chain canary was incorrectly designed to reject the
candidate from per-chain acceptance. Sixteen draws per chain are not adequate
for that scientific decision, particularly after the tuning campaign already
showed large finite-run acceptance variability.

At the user's direction, that entire run was removed:

- run artifacts and logs;
- experiment plan;
- result and reset note;
- dedicated launcher; and
- dedicated tests and Python bytecode caches.

No path matching `chart-a-l10-sequential-hmc-validation` or
`chart_a_l10_sequential` remains. Do not reconstruct the deleted numbers from
conversation history, cite them as evidence, treat them as a candidate veto, or
charge them as a scientific HMC result. The deletion does not alter the valid
six-`L` tuning artifact.

## Authoritative Scientific State

| Ledger | State at reboot |
| --- | --- |
| Training | Chart A frozen checkpoint 1500 and trained NeuTra transport were successfully used by the completed tuner. No training process is active. |
| Fixed-HMC tuning | Complete for grid `(5,10,15,20,25,3)`. `L=10` was the only nominated candidate. |
| Candidate evidence | Fresh 64-result tuning verification reported pooled mean acceptance `0.7235869085131437`, finite required tensors, and valid target status. |
| Important warning | The same tuning verification recorded finite `max_abs_log_accept_ratio=1e100`. This is an explanatory warning, not a divergence count or standalone rejection gate. |
| Native divergence | Not exposed by the installed TFP HMC kernel. Unavailable is not zero. |
| Sequential HMC | Not run. |
| Warm-up | Not run. |
| Retained posterior draws | None. |
| R-hat / ESS | Not available. |
| Posterior validity | Not checked. |
| Candidate ranking | Not statistically supported. One arm passed its screen; descriptive arm differences are not a ranking. |
| Default readiness | Not established. |

The correct current claim is:

> Chart A has one fixed-HMC kernel candidate worth a properly designed
> sequential HMC test.

Do not claim that the candidate is accepted, rejected, converged, posterior
correct, scientifically valid, or ready as a default.

## Authoritative Artifacts

| Artifact | Path | SHA-256 at memo creation |
| --- | --- | --- |
| Six-`L` plan | `docs/plans/bayesfilter-ssl-lstm-q20-chart-a-six-l-fixed-hmc-tuning-plan-2026-08-03.md` | `bf5539097c9b6175c33529ef0ae671ef1aed5fa993a305a99f2faff4fe92cf62` |
| Six-`L` result | `docs/plans/bayesfilter-ssl-lstm-q20-chart-a-six-l-fixed-hmc-tuning-result-2026-08-03.md` | `f2072a314b8758cf72836d4623e92cb0fc5a35b515e52d377a399d69c2b6db28` |
| Supervisor summary | `docs/plans/artifacts/ssl-lstm-q20-chart-a-six-l-fixed-hmc-tuning-2026-08-03/r1/summary.json` | `76e3c716c4ab1daf4a7e4fce6852f99df71ca4263b7c193b9962661c85cb5dce` |
| Merged tuning result | `docs/plans/artifacts/ssl-lstm-q20-chart-a-six-l-fixed-hmc-tuning-2026-08-03/r1/merged-tuning-result.json` | `c3018064fcbbe040b3510165138bc7db7de1b378dd0eb4c1a1b8155af796fb19` |
| Six-`L` launcher | `docs/benchmarks/run_ssl_lstm_q20_chart_a_six_l_fixed_hmc_tuning_2026_08_03.py` | `219c5d59830c231f7ce9bd066efaab9864990e5f0c564d944f1c5d67bca881ff` |

Candidate identity bindings:

| Field | Frozen value |
| --- | --- |
| Target scope | `ssl_lstm_neutra_state_complexity_batch_native:q20:fixed_hmc_api:chart-a:claim_tuning_grid6` |
| Transformed adapter signature | `9772c5988104a9548e34eb138ffe4e950fb8354580f2395fd96718a35e60103e` |
| Base adapter signature | `a8be6c212eb2a74ef926ccb9279871805949cae6e00c312a12525141638166f3` |
| Fixed transport manifest hash | `dcb1ec65e7d91a382518a0eef382e3cd8efec78341445f22d4d6ac899ea685eb` |
| Chart checkpoint SHA-256 | `c87ee24874705bb12296cc05b82310326579694cc04c2a3682792f9bf18fb9ff` |
| Target signature | `9a86e60081f1b9cd288dbdb1dcbe1e9a5b5e23d9b5ef97afdb72ee95c23d7278` |
| Transport hash | `caf6c9ec1a46d04253b2ae3922d83e619f38c824cea955d5da8ac419d2dfed7f` |

Fail closed after reboot if these identities do not match. Do not use the old
loader that expects `chart-a/tuning-result.json`; the candidate is in the
merged six-`L` artifact.

## Repository State

- Git commit at memo creation:
  `b370dc89e6e79f3853e0fccd5ab5b4fa2cb9065d`.
- The worktree is intentionally dirty with concurrent agents' modified and
  untracked work.
- Do not run `git reset`, `git clean`, broad restore commands, or delete
  unrelated artifacts after reboot.
- Re-read `AGENTS.md` and current `git status --short` before editing.
- Do not assume the commit or dirty-file set is unchanged after reboot; other
  agents may finish work before shutdown.

## GPU And Thermal Boundary

GPU 0 is overheating and must not be used by this lane.

Trusted pre-reboot observation at approximately 22:16 Asia/Shanghai:

| GPU | UUID | Temperature | Utilization | Memory | Power |
| ---: | --- | ---: | ---: | ---: | ---: |
| 0 | `GPU-a008e90f-259e-df57-7988-63b6831fff68` | `90 C` | `1%` | `603 MiB` | `83.55 W` |
| 1 | `GPU-a1ea1946-07c0-8ed5-2ba1-d96f82c89cd3` | `38 C` | `39%` | `667 MiB` | `30.20 W` |

No BayesFilter or TensorFlow process was using GPU 0. Earlier trusted process
inspection attributed GPU 0 contexts to the display/remote-session stack,
including `gnome-remote-desktop-daemon` and `nxnode.bin`. This is a host thermal
problem, not evidence that the removed CPU-only HMC run used GPU 0.

Post-reboot rules:

1. Do not launch any framework merely to probe the GPU.
2. First run trusted `nvidia-smi` and record temperature, fan, power, clocks,
   utilization, memory, and processes.
3. Do not use GPU 0 even if it cools after reboot unless the user explicitly
   changes this boundary.
4. CPU-only TensorFlow commands must set `CUDA_VISIBLE_DEVICES=-1` before
   TensorFlow import.
5. If later work is explicitly approved for GPU, expose only GPU 1 with
   `CUDA_VISIBLE_DEVICES=1`, set `TF_FORCE_GPU_ALLOW_GROWTH=true` before import,
   and verify repository memory growth before device initialization.
6. Do not assume GPU 1 is available merely because it is cooler; check its
   current owners before launch.

The next HMC phase is currently planned as CPU/XLA work, so no GPU is required.

## Correct Next Research Step

Do not rerun a tiny acceptance canary and do not use a small number of draws to
accept or reject the candidate.

The next agent must first write and skeptically audit a replacement sequential
HMC plan. The plan should test the frozen `L=10` candidate using the repository
sequential policy itself:

- four chains;
- fixed kernel with no adaptation or retuning inside HMC;
- at least 2,000 discarded warm-up transitions per chain;
- readiness checked on the latest 1,000 warm-up transitions per chain using
  maximum rank-normalized split and folded R-hat `<=1.05`;
- cumulative retained sampling with at least 1,000 retained transitions per
  chain;
- retained maximum rank-normalized split and folded R-hat `<=1.01`;
- bulk ESS `>=400` and tail ESS `>=400`;
- declared warm-up and retained caps no greater than 10,000 per chain;
- warm-up draws archived but excluded from posterior estimates; and
- finite state, target, proposed target, score, log acceptance, target-status,
  and available positive native-divergence checks throughout.

Acceptance and finite log-accept/energy-tail magnitudes should be reported as
diagnostics. They must not be promoted into a short-run candidate decision
without a prospectively justified statistical design. Native-divergence
unavailability must remain unavailable rather than being converted to zero.

The replacement plan must answer these unresolved design questions before
execution:

1. whether to start the actual sequential controller directly or run a
   mechanics-only compilation/timing probe that cannot accept or reject the
   candidate;
2. how the four-chain state and random streams are preserved across persistent
   CPU/XLA worker processes;
3. how runtime is bounded without weakening the minimum warm-up or retained
   sample counts;
4. how partial chunks are archived safely if the wall cap stops the campaign;
5. which posterior/reference checks follow if R-hat and ESS pass; and
6. what precise evidence distinguishes candidate failure from harness,
   initialization, metric, or transport failure.

No NUTS is authorized. `L=1` is not allowed. No Chart B run, retuning, new
metric, relaxed gate, posterior claim, or default change is authorized by this
memo.

## Post-Reboot Checklist

Run read-only checks first:

```bash
cd /home/ubuntu/python/BayesFilter
date -Is
git rev-parse HEAD
git status --short
pgrep -af 'run_ssl_lstm|BayesFilter|tensorflow|pytest'
nvidia-smi
sha256sum \
  docs/plans/artifacts/ssl-lstm-q20-chart-a-six-l-fixed-hmc-tuning-2026-08-03/r1/summary.json \
  docs/plans/artifacts/ssl-lstm-q20-chart-a-six-l-fixed-hmc-tuning-2026-08-03/r1/merged-tuning-result.json
find docs/plans docs/benchmarks tests -maxdepth 3 \
  \( -iname '*l10*sequential*' -o -iname '*chart-a-l10-sequential*' \) -print
```

Expected state:

- no active BayesFilter/TensorFlow run;
- no deleted invalid-run path is present;
- authoritative artifact hashes match this memo;
- GPU 0 is not selected for work; and
- no HMC command is launched until a replacement plan passes the required
  skeptical audit and the user approves its actual sampling design.

## Nonclaims

This memo does not establish that `L=10` is a good kernel, that the target is
correct, that NeuTra converges, that HMC will fit a wall budget, that CPU and GPU
results are equivalent, that the posterior is valid, or that the model is
scientifically adequate. It preserves the last valid evidence and prevents the
deleted under-sampled run from contaminating the next decision.
