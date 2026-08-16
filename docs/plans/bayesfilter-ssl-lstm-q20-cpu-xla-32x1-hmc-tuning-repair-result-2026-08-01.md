# q=20 CPU-XLA 32x1 HMC Tuning Repair Result

Date: 2026-08-01
Status: `TUNING_REPAIR_REQUIRED`

## Superseding Correction (2026-08-02)

This campaign is invalid as fixed-HMC tuning evidence. Its selected candidates
used `L=1`, which is now forbidden for this HMC lane. The finite
`abs(log_accept_ratio) > 1000` events were also misclassified as hard
divergences; they are energy-related explanatory telemetry and are not native
divergence events.

Both historical `L=1` confirmations were finite and inside the historical
per-chain acceptance bounds, but that cannot rescue them because the candidate
family itself is ineligible. Native divergence was not exposed, so the
divergence gate was not checked. Do not use this campaign to choose a kernel,
seed a later `L` grid, justify mass adaptation, or support warm-up, posterior,
HMC-validity, or scientific claims. Preserve the raw artifact only as evidence
of the superseded custom procedure.

## Outcome

The following is the original, superseded interpretation. The bounded short-trajectory repair completed its six-arm grid and independent
confirmation for both charts. It improved the numerical behavior substantially
relative to the original `L=2` grid, but neither chart passed confirmation under
the unchanged hard veto `abs(log_accept_ratio) <= 1000`.

- Chart A selected `L=1`, step `0.75`, with two tuning replications passing,
  pooled acceptance `0.6806`, and replication maximum absolute log acceptance
  `552.3` and `313.6`. Confirmation acceptance was within `[0.35,0.95]` for
  every chain except no chain; however chart A chain 12 had maximum absolute
  log acceptance `1036.299`, so the confirmation failed the hard numerical
  veto.
- Chart B selected `L=1`, step `0.875`, with two tuning replications passing,
  pooled acceptance `0.6928`, and replication maximum absolute log acceptance
  `56.9` and `430.1`. Confirmation acceptance was within `[0.35,0.95]` for
  every chain, but chart B chain 14 had maximum absolute log acceptance
  `1121.103`, so the confirmation failed the same hard veto.
- The exact CPU/XLA persistent chunk preflight passed before the repair. All
  states and target-status audits were finite and valid; there were no crashes,
  affinity failures, visible GPUs, or RSS cap events. Native divergence
  telemetry was unavailable.
- The tuning-only profile intentionally did not run warm-up or retained HMC.

The repair reduced the extreme-log-accept failure from widespread r1 events to
one chain per chart at confirmation, but it did not establish an admissible
kernel. This rejects the tested identity-mass short-trajectory candidates; it
does not reject the trained transports or the NeuTra research direction.

## Decision Table

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Do not launch warm-up | Neither chart passed confirmation | One hard log-accept veto per chart | Identity mass may be mismatched to chart tails; rare extreme-event rate is uncertain | Stop step-size fishing and test a reviewed mass/metric adaptation on fresh tuning partitions | Posterior invalidity, chart invalidity, or scientific failure |
| Retain shorter trajectory as repair evidence | Both selected arms passed two-arm tuning and had near-target confirmation acceptance | Confirmation still vetoed by extreme log-accept | Confirmation uses 64 transitions per chain, so tail event frequency is descriptive | Use `L=1` as the warm-start trajectory for a mass-matrix repair candidate, not as an admitted kernel |

## Inference Status

| Evidence class | Result |
| --- | --- |
| Hard veto screen | Supported: one confirmation chain per chart exceeded `abs(log_accept_ratio)=1000`; no other hard veto observed |
| Statistically supported ranking | None; acceptance and log-accept tails are descriptive short-run diagnostics |
| Descriptive-only differences | Repair-grid acceptance, per-chain confirmation acceptance, runtime, RSS, and extreme-log-accept magnitudes |
| Default-readiness | Not ready; no confirmed kernel and CPU remains an explicit validation exception |
| Next evidence needed | Fresh mass/metric tuning with exact payloads, `L=1` warm start, two independent replications, and the same confirmation gate |

## Run Manifest

| Field | Value |
| --- | --- |
| Git commit | `882679796e8ee684b6b020b7cd84e3cfc1d92d58` (dirty worktree preserved) |
| Environment | `/home/ubuntu/anaconda3/envs/tfgpu/bin/python`, TensorFlow `2.20.0`, FP64 |
| Hardware | 32 one-chain workers on CPUs `0..31`; supervisor on CPU `32`; GPU intentionally hidden |
| Profile | `short-trajectory-repair-v1`; six arms; two replications per arm; tuning-only terminal |
| Wall time | `1491.0617 s` (`24.85 min`) |
| Campaign cap | `7200 s`; not approached |
| Raw posterior shards | None, because confirmation failed and the profile forbids sequential sampling |
| Plan | `docs/plans/bayesfilter-ssl-lstm-q20-cpu-xla-32x1-hmc-tuning-repair-plan-2026-08-01.md` |
| Result artifact | `docs/plans/artifacts/ssl-lstm-q20-cpu-xla-32x1-hmc-tuning-repair-2026-08-01/r1/summary.json` |
| Result SHA-256 | `07cff83f4727ac045f130bc75566a4f8b2702f41ec2904e0e4f264d66faa3ab5` |

## Post-Run Red Team

- Strongest alternative explanation: the identity mass matrix is the limiting
  geometry, and one-chain extreme energy errors remain under-resolved by the
  short confirmation.
- What would overturn this decision: a fresh mass/metric candidate that passes
  both tuning replications and all 16-chain confirmation hard screens for each
  chart, followed by sequential warm-up and retained diagnostics.
- Weakest evidence: the confirmation is short and native divergence telemetry is
  unavailable; the strict log-accept veto is a numerical validity screen, not a
  posterior convergence proof.
- No conclusion is made about posterior correctness, stationarity, predictive
  validity, chart ranking, GPU performance, or scientific validity.
