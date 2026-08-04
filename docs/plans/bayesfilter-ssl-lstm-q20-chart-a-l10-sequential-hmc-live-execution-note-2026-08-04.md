# q=20 Chart A L=10 Sequential HMC Live Execution Note

Date: 2026-08-04
Observed: 2026-08-04T02:25:01+08:00
Status: `RUNNING_FIRST_WARMUP_CHUNK_NO_SAMPLER_RESULT_YET`

## Terminal Correction

This note is a historical live snapshot. The service later terminated at
`2026-08-04T10:46:59+08:00`. It archived three 500-draw warm-up chunks per
chain. The third chunk hit the predeclared acceptance veto because chain 0 had
mean acceptance probability `0.2269945784`, below `0.35`. No retained sampling
or R-hat/ESS calculation occurred. The terminal evidence and a subsequent
reporting-only serialization failure are recorded in
`docs/plans/bayesfilter-ssl-lstm-q20-chart-a-l10-sequential-hmc-result-2026-08-04.md`.

## Outcome So Far

The reviewed Chart A sequential fixed-HMC campaign was launched successfully as
the user service:

```text
bayesfilter-q20-chart-a-l10-sequential-hmc-r1.service
```

It started at `2026-08-04T01:27:08+08:00` with a 24-hour internal campaign cap
and a `RuntimeMaxSec=87000` external service cap. At the observation time it had
run for 57 minutes. The supervisor and all four persistent workers were active.

No 500-transition chunk had completed. The only material artifacts were
`launch.json` and four worker logs. Therefore:

- completed warm-up transitions per chain: `0`;
- retained posterior transitions: `0`;
- R-hat and ESS: unavailable;
- numerical/status/acceptance chunk verdict: unavailable;
- candidate decision: unavailable.

Process liveness and CPU consumption are engineering evidence only. They are not
sampler evidence and must not be used to accept or reject `L=10`.

## Prelaunch Evidence

| Check | Result |
| --- | --- |
| Skeptical plan audit | `PASS_WITH_RECORDED_ROUTE_LEDGER_LIMITATION` |
| Focused harness/controller tests | `20 passed` |
| Exact-target preflight | `PREFLIGHT_PASSED` in `7.3107723 s` |
| Initial states | Four declared `[4,4]` states; finite value/score and target status code zero for all |
| Candidate | `L=10`, step `0.4148806556986277`, fixed identity `z` mass |
| Kernel hash | `34b89acd551dd25bee9dd0a463be67ff9d06f08ea3f970da5ffa97b44438ca4d` |
| Preflight summary SHA-256 | `ad3a847ade89a0819dac573108d38f14a5863333573822d5e6b5f1808c472b4d` |
| GPU status | `CUDA_VISIBLE_DEVICES=-1`; preflight recorded `physical_gpus=[]` |
| XLA status | Every worker log contains `Compiled cluster using XLA!` |

TensorFlow logged `CUDA_ERROR_NO_DEVICE` while probing its hidden CUDA backend.
This is consistent with the deliberate `CUDA_VISIBLE_DEVICES=-1` boundary. The
preflight and every worker readiness payload record no physical GPU. Neither GPU
0 nor GPU 1 is used by this campaign.

## Live Resource State

| Process | Role | CPU affinity | Observed CPU | Observed RSS |
| ---: | --- | --- | ---: | ---: |
| `44226` | Supervisor | `32` | `0.1%` | `452,040 KiB` |
| `44259` | Chain 0 | `0..7` | `122%` | `1,242,976 KiB` |
| `44260` | Chain 1 | `8..15` | `122%` | `1,240,892 KiB` |
| `44261` | Chain 2 | `16..23` | `122%` | `1,242,556 KiB` |
| `44262` | Chain 3 | `24..31` | `123%` | `1,242,648 KiB` |

The service reported 229 tasks, 3.9 GiB current memory, and 4 hours 44 minutes
aggregate CPU time after 57 minutes wall. Memory was stable across repeated
observations. The approximately 1.22-core usage per worker shows that allowing
eight cores per chain does not create eightfold within-chain acceleration; HMC
transitions remain sequential and the compiled target is the current limiting
work.

## Decision Table

| Decision | Primary criterion status | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Continue the bounded service | Not yet evaluable; first 500 warm-up draws are active | No continuation veto observed | Exact time per `L=10` transition | Wait for `progress/chunk-0000.json` and the first archived warm-up receipt | Kernel viability or failure |
| Do not estimate completion from `L=1` | Earlier mechanics rate is the wrong workload | N/A | First-call versus warm-call cost | Use the first completed exact chunk as the planning rate | 14.4-hour completion |
| Preserve zero scientific claims | No complete chunk, R-hat, ESS, or posterior draws | N/A | All sampler evidence pending | Interpret only after shared-controller artifact creation | Convergence, posterior correctness, or model validity |

## Inference Status

| Evidence class | Status |
| --- | --- |
| Hard veto screen | Not evaluated; the first chunk is incomplete |
| Statistically supported ranking | None; no ranking is planned |
| Descriptive-only differences | Service wall, CPU, RSS, process liveness, and XLA receipts |
| Default readiness | Not evaluated |
| Next evidence needed | First full 500-transition four-chain warm-up chunk, then policy minimum warm-up and retained R-hat/ESS |

## Engineering, Numerical, And Scientific Ledgers

| Ledger | Current state |
| --- | --- |
| Engineering | Service active; four expected workers active; affinities, CPU isolation, kernel identity, and XLA receipts passed |
| Numerical/sampler | Not checked because no full chunk has returned to the shared controller |
| Scientific | No evidence; posterior/reference validation remains a later separate phase even if sequential sampling passes |

## Run Manifest

| Field | Value |
| --- | --- |
| Git commit | `b370dc89e6e79f3853e0fccd5ab5b4fa2cb9065d` with unrelated dirty worktree preserved |
| Environment | `/home/ubuntu/anaconda3/envs/tfgpu`, TensorFlow/TFP CPU-only FP64 XLA |
| Unit | `bayesfilter-q20-chart-a-l10-sequential-hmc-r1.service` |
| Start | `2026-08-04T01:27:08+08:00` |
| Internal cap | `86,400 s` |
| External unit cap | `87,000 s` |
| CPU topology | Four workers on `0..31`; supervisor on `32` |
| GPU | Hidden before TensorFlow import; no visible physical GPUs |
| Chunk | 500 transitions per chain |
| Policy minima | 2,000 warm-up and 1,000 retained per chain |
| Policy caps | 10,000 warm-up and 10,000 retained per chain |
| Plan | `docs/plans/bayesfilter-ssl-lstm-q20-chart-a-l10-sequential-hmc-plan-2026-08-04.md` |
| Launcher | `docs/benchmarks/run_ssl_lstm_q20_chart_a_l10_sequential_hmc_2026_08_04.py` |
| Output | `docs/plans/artifacts/ssl-lstm-q20-chart-a-l10-sequential-hmc-2026-08-04/r1/` |

## Monitoring

Trusted service status:

```bash
systemctl --user status \
  bayesfilter-q20-chart-a-l10-sequential-hmc-r1.service --no-pager
```

Completed chunk or terminal artifact check:

```bash
find docs/plans/artifacts/ssl-lstm-q20-chart-a-l10-sequential-hmc-2026-08-04/r1 \
  -maxdepth 4 -type f -printf '%P %s\n' | sort
```

A completed first chunk creates `progress/chunk-0000.json` followed by shared-
controller warm-up archive files. A terminal run creates `summary.json` and
`sequential-result.json`.

If an explicit stop is required:

```bash
systemctl --user stop bayesfilter-q20-chart-a-l10-sequential-hmc-r1.service
```

Stopping during an active chunk will not create a partial scientific shard.
Previously completed immutable chunks, if any, remain preserved.

## Post-Run Red Team At This Boundary

The strongest alternative explanation for the long first call is exact `L=10`
target cost and XLA first-call work, not a deadlock: all workers sustain CPU use
and stable memory. This remains descriptive until a chunk completes. A worker
exit, stagnant CPU, new error log, service failure, or failure to emit the first
chunk within the external cap would overturn that interpretation and classify
the attempt as an engineering or budget failure rather than a candidate result.
