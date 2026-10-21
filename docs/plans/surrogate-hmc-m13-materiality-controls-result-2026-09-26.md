# M13 materiality control result — 2026-09-26

No common adjacent pair passed both fresh partitions. All eight nominal arms
and all proposal probes were healthy. One of four stress rows was flagged in
every arm; stronger protection introduced no new failed stress row. This is
control-selection failure, not a failed posterior test or rejection of surrogate
HMC. No inference was launched. The40 focused tests passed; sources were unchanged.

| Stronger/weaker multiplier | Calibration max probability change | Validation max probability change | Common .01 pass |
|---|---:|---:|---|
| 1/.1 | .01735620 | .02004959 | No |
| .1/.01 | .00687630 | .03515692 | No |
| .01/.001 | .01462074 | .00595137 | No |

The independent partitions select different local pairs. Non-monotone sensitivity
may reflect finite-precision force geometry or controls that still materially
change the proposed trajectories. Neither explanation is established. The next
smallest test extends the geometric curve to weaker positive controls on fresh
streams, retaining the original tolerance. The source of these differences is
not the previously accepted tiny saved-likelihood discrepancy.

Artifact: `docs/plans/artifacts/surrogate-hmc-closure-20260922-m13-materiality-controls-02`.
Exact command: the control plan command with version02. Environment tftwogpu,
GPU/XLA, verified growth, FP32/TF32 force and FP64 endpoint. Worker wall1154.514334s;
master charge1267.659917 CPU/1154.514334 GPU seconds. Attempt01's pre-GPU Path
repair used6.579710 CPU seconds. All failures count toward the approved budget.

| Decision | Primary criterion | Veto diagnostics | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Candidate family not selected | No common materiality pair | No unflagged invalidity; nominal/proposal health pass | Finite probe coverage and non-monotone control response | Authorized bounded fresh-partition repair | Failure of surrogate HMC |

| Inference status | Assessment |
|---|---|
| Hard veto screen | No nominal/proposal validity veto; no selection issued |
| Statistically supported ranking | None |
| Descriptive-only differences | Probe maxima and timing |
| Default readiness | No new promotion; FP32 remains default |
| Next evidence needed | Selected scoped controls, public tuning, retained/reference test |

Post-run red team: this could be sensitivity to proposal trajectories rather
than harmful posterior bias. A weaker stable pair would overturn the need for
further control localization. These short probes cannot replace the actual
posterior comparison. This result follows the existing master; it is not a new
direction or a request to pause for Claude.
