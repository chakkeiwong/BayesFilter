# q=20 CPU-XLA 32x1 Distributed NeuTra-HMC Validation Result

Date: 2026-08-01
Status: `TUNING_REPAIR_REQUIRED`

## Superseding Correction (2026-08-02)

The interpretation below is historical and is superseded. The launcher used a
hand-written `(L, step_size)` grid instead of BayesFilter's fixed-transport HMC
tuning API. It also treated finite `abs(log_accept_ratio) > 1000` as a hard
veto. That quantity is an energy-related health proxy, not native divergence
telemetry, and a finite exceedance is explanatory only under the corrected
gate.

Re-evaluating the raw `L>=2` arms using the declared tuning acceptance band
`[0.55, 0.85]` and nonfinite checks gives these descriptive viable arms:

- Chart A: `(L=2, epsilon=0.5656854249492381)`,
  `(L=2, epsilon=0.75)`, and `(L=4, epsilon=0.5656854249492381)`.
- Chart B: `(L=2, epsilon=0.75)`,
  `(L=4, epsilon=0.5656854249492381)`, and `(L=4, epsilon=0.75)`.

This does not admit a kernel. Chart A's only confirmation had per-chain mean
acceptance as low as `0.193247`, below its declared `0.35` lower bound. Chart B
was never confirmed because the incorrect selector rejected its arms first.
Native divergence was not exposed by the TFP fixed-HMC kernel, so the
divergence gate was not checked and must not be reported as passed or as zero
divergences. The corrected terminal classification is therefore
`INCONCLUSIVE_FIXED_HMC_ADMISSION`; no warm-up or retained result is supported.

## Outcome

The following is the original, superseded interpretation. The 32-worker CPU/XLA harness was engineering-valid, but neither identity-mass
fixed HMC candidate was admitted. The run stopped at the prospective tuning
gate. No warm-up or retained posterior samples were generated.

- Chart A had one arm pass both 32-transition tuning replications:
  `L=2`, step `0.5656854249492381`, pooled acceptance `0.815222`. Its
  independent 64-transition confirmation failed because chain 0 had
  `max(abs(log_accept_ratio))=1273.493`, and chain 5 had acceptance `0.193247`
  plus `max(abs(log_accept_ratio))=2045.184`.
- Chart B had no arm pass both replications. Its closest descriptive arms were
  `L=2`, step `0.5656854249492381` (pooled acceptance `0.848179`, one chain at
  `1476.6` maximum absolute log acceptance) and `L=2`, step `0.75` (pooled
  acceptance `0.730113`, but the second replication had two excessive
  log-accept chains).
- All workers were CPU-only, XLA-enabled, finite and moving at the mechanics
  preflight. Claim-bearing tuning states and final-state target audits remained
  finite and target-status valid. Native divergence telemetry was not exposed.

This is a fixed-kernel/metric tuning failure. It does not invalidate the target,
checkpoint payloads, transport implementation, trained charts, distributed
harness, or NeuTra direction.

## Decision Table

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Do not start sequential warm-up | No chart had a confirmed kernel | Chart A confirmation and all Chart B arms rejected | Whether a smaller `L=2` step controls rare extreme energy errors | Fresh bounded `L=2` refinement below the current midpoint, with the same hard veto and independent confirmation | Transport invalidity, posterior invalidity, or NeuTra failure |
| Preserve CPU/XLA topology | Full 32-worker exact-chunk preflight passed | No shared infrastructure/resource veto | Long-run throughput remains unmeasured because tuning stopped first | Reuse 16 chains/chart and CPU 32 supervisor for repair | CPU default or GPU equivalence |

## Inference Status

| Evidence class | Result |
| --- | --- |
| Hard veto screen | Candidate rejection supported by `abs(log_accept_ratio)>1000`; no nonfinite state/target, invalid target status, crash, affinity drift, or RSS veto |
| Statistically supported ranking | None; arm acceptance values are tuning diagnostics, not statistical superiority evidence |
| Descriptive-only differences | Per-arm pooled and per-chain acceptance, runtime, RSS, and log-accept magnitudes |
| Default-readiness | Not ready; CPU remains an explicit validation exception and no fixed kernel was confirmed |
| Next evidence needed | Fresh exact-payload `L=2` step refinement, two replications per arm, then independent 64-transition every-chain confirmation before any warm-up |

## Run Manifest

| Field | Value |
| --- | --- |
| Git commit | `882679796e8ee684b6b020b7cd84e3cfc1d92d58` (dirty worktree preserved) |
| Environment | `/home/ubuntu/anaconda3/envs/tfgpu/bin/python`, TensorFlow `2.20.0`, FP64 |
| Hardware | 32 one-chain workers on CPUs `0..31`; supervisor on CPU `32`; GPU intentionally hidden |
| XLA | Enabled in every worker; exact fixed-chunk preflight passed |
| Checkpoints | Chart A program/controller/trainer step `1500`; Chart B program/controller step `2500`, selected trainer internal step `2250`; hashes in `summary.json` |
| Wall time | `3328.4848 s` (`55.47 min`) |
| Campaign cap | `86400 s`; not approached |
| Raw posterior shards | None, because no kernel passed confirmation |
| Plan | `docs/plans/bayesfilter-ssl-lstm-q20-cpu-xla-32x1-distributed-hmc-validation-plan-2026-08-01.md` |
| Launcher SHA-256 | `6a4e313453ca31a383f022ebb38286ed0a0d243d2d0a3569842af6b97831c2e3` |
| Plan SHA-256 | `d0370cb527c51547853143f2cd44668857e42624cc9f6238530bb839a07b2787` |
| Preflight artifact | `docs/plans/artifacts/ssl-lstm-q20-cpu-xla-32x1-distributed-hmc-validation-2026-08-01/preflight-r3/summary.json`, SHA-256 `93bb3519c461af3e25c0d2057cefdd32deb1910126d31c3d93ceb6d2065ff5b8` |
| Result artifact | `docs/plans/artifacts/ssl-lstm-q20-cpu-xla-32x1-distributed-hmc-validation-2026-08-01/r1/summary.json`, SHA-256 `8df22ea1d306efcccf3db0e6d3fbf9c768d4baf821704f847d625032e22ac953` |

## Post-Run Red Team

- Strongest alternative explanation: identity mass plus the tested step grid,
  not the learned transport, may cause the rare extreme energy errors.
- What would overturn the decision: a fresh smaller-step `L=2` arm that passes
  both replications and a 16-chain confirmation with no extreme log-accept
  value, followed by sequential warm-up and retained diagnostics.
- Weakest evidence: each arm used short chains; acceptance and extreme-event
  rates remain descriptive. The hard numerical events reject the candidates,
  but they do not estimate their long-run frequency precisely.
- No conclusion is made about posterior correctness, stationarity, predictive
  validity, transport ranking, GPU performance, or scientific validity.
