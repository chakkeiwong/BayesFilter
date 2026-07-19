# SSL-LSTM NeuTra `(32,32)` Capacity Diagnostic Result

Date: 2026-07-15

Status: `R2_CAPACITY_REPAIR_NOT_NOMINATED_CAPACITY_PLUS_SCHEDULE_REPAIR_TRIGGERED`

## Outcome

The `(32,32)` capacity implementation passed engineering checks and the paired
trusted GPU/XLA diagnostic completed validly. Both historical A/B streams hit
the prospective scale-saturation veto at step 100 under the inherited `0.01`
initial learning rate. The run stopped as designed after `1,224.85` charged
GPU-seconds (`0.3402` GPU-hours), within the `9,000`-second cap.

This rejects **width expansion alone under the original optimizer schedule**
as the repair candidate. It does not reject `(32,32)` capacity, NeuTra, or the
research direction generally. The next smallest discriminating candidate is
`(32,32)` with a lower initial learning rate, tested prospectively. No full
5,000-step confirmation or HMC run is justified from this result.

## Paired Evidence

| Diagnostic | Historical stream A | Historical stream B |
| --- | --- | --- |
| Decision | `R2_CAPACITY_REPAIR_NOT_NOMINATED` | `R2_CAPACITY_REPAIR_NOT_NOMINATED` |
| Veto | Saturation above `0.05` | Saturation above `0.05` |
| Veto step | 100 | 100 |
| Stored trainable parameters | 4,440 | 4,440 |
| Initial heldout mean loss | `62.99714` | `80.14121` |
| Step-100 heldout mean loss | `41.28614` | `41.47929` |
| Overall saturation | `0.21745` | `0.11719` |
| Stage 1 saturation | `0.23047` | `0.21484` |
| Stage 2 saturation | `0.25781` | `0.02734` |
| Stage 3 saturation | `0.16406` | `0.10938` |
| Moderate-shell maximum radius | `4.34472` | `3.48968` |
| Original-neighborhood maximum radius | `2.29710` | `2.13400` |
| Roundtrip maximum | `8.44e-15` | `4.44e-15` |
| Stream runtime | `612.83 s` | `611.98 s` |

The loss reductions and A/B differences are descriptive only. Both streams
failed the prospective saturation screen, so neither is ranked or nominated.
Stream A also had a step-100 shell radius above `4.30`, but shell was
prospectively explanatory before terminal step 1,200 and is not counted as a
promotion veto here.

## Decision Table

| Decision | Primary criterion status | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Stop `(32,32)` plus original schedule | Failed at step 100 on both streams | Candidate promotion veto; no hard evidence veto | Whether `0.01` overdrives the wider network or reverse KL would saturate it even at a stable rate | Prospectively test `(32,32)` with a lower initial rate on the same paired streams | `(32,32)` is intrinsically unsuitable; NeuTra failure; posterior/HMC conclusion |
| Preserve strict `(4,4)` source preset | Passed regression/parity tests and was not mutated | No implementation veto | It remains under-capacity for this target | Keep as immutable source-procedure comparator, not material default | `(4,4)` adequacy |
| Keep capacity adaptation implementation | Engineering and artifact contracts passed | No implementation/artifact veto | Candidate behavior under a stable schedule is untested | Reuse in one-change capacity-plus-schedule repair | Default readiness or candidate nomination |

## Inference Status

| Inference | Status |
| --- | --- |
| Hard veto screen | Passed; both streams were finite, GPU/XLA-resident, source-bound, serializable, and invertible |
| Candidate viability | `(32,32)` plus original `0.01` schedule not nominated |
| Statistically supported ranking | None |
| Descriptive-only differences | Loss, stage saturation, shell/tail radii, runtime, and all A/B differences |
| Default readiness | Not established |
| Next evidence needed | Paired `(32,32)` run with prospectively reduced initial learning rate; only then fresh two-seed full confirmation if nominated |

## Run Manifest

| Field | Value |
| --- | --- |
| Git commit | `20835ecf90bff78ca93c5d401f231e4aa94e63ce` (dirty worktree preserved) |
| Command | `/home/ubuntu/anaconda3/envs/tfgpu/bin/python docs/benchmarks/run_ssl_lstm_neutra_capacity_32x32_diagnostic_2026_07_15.py --output-root docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/capacity-32x32-diagnostic` |
| Environment | conda `tfgpu`; Python `3.13.13`; TensorFlow `2.20.0`; `float64` |
| Device | Physical GPU 1 selected; process-visible `/physical_device:GPU:0`; NVIDIA GeForce RTX 4080 SUPER |
| Runtime policy | XLA JIT on; TF32 enabled; soft placement disabled |
| Trust basis | `owner_designated_managed_session_visible_gpu_trusted` |
| Streams | Exact historical init/train/validation A `[4101]/[5101]/[5201]` and B `[4102]/[5102]/[5202]` namespaces |
| Shared cap / charged | `9,000 s` / `1,224.85 s` |
| Focused checks | `62 passed`; compilation and `git diff --check` passed |
| Plan | `docs/plans/bayesfilter-ssl-lstm-neutra-seed-instability-stabilization-repair-plan-2026-07-15.md` |
| Native review | `docs/reviews/bayesfilter-ssl-lstm-neutra-capacity-32x32-native-review-2026-07-15.md` |

Artifact SHA-256 values:

- program result: `5ae83bc90faf7463a5b74437cdaf904aa54112a8a9945fc2b8ebddc994b47a00`;
- stream A result: `c179bfb451b337c6328b212738dde34c085141c96bfe556ab6607a480756bc18`;
- stream B result: `a36e6c1cc16b75e01c8e02baf38e91198d9659a5d1243bd7bbf864deaba0908a`;
- stream A checkpoint: `3f84b98c4f830e7d514dc0b879b5b8e89143c4d9c5a645f29094eddacaadf89b`;
- stream B checkpoint: `ec0da4e00c3e612b978298e102f42f7e613506dbf8e9e561987529ba475cd1a5`.

## Negative-Result Classification

- Implementation failure: not supported.
- Runtime/XLA failure: not supported.
- Artifact failure: not supported.
- Tuning/candidate failure: supported; the inherited high initial rate and
  wider conditioner interact to drive all three stages toward scale bounds.
- Evidence against `(32,32)` capacity itself: not established because learning
  rate was deliberately held fixed to isolate the first capacity-only test.
- Evidence against NeuTra generally: not established.

## Post-Run Red Team

The strongest alternative explanation is that reverse-KL mode seeking, rather
than the learning rate, drives saturation even for a wider network. A lower-rate
paired counterfactual can distinguish rapid optimizer overdrive from an
objective/geometry failure but cannot establish posterior coverage. A pass at
lower rate would still require two fresh full seeds and downstream exact HMC,
replication, predictive, and calibration evidence.

What would overturn this result is evidence that the `(32,32)` family changed
something besides width, used different streams, failed target/source binding,
or did not execute on GPU/XLA. Focused tests and structured artifacts guard all
of those boundaries. The weakest evidence is duration: stopping at step 100 is
appropriate for the predeclared saturation veto but says nothing about a
different optimizer schedule.
