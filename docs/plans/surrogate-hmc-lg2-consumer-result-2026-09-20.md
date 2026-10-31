# LG2 consumer qualification result

The actual LG2 consumer passed all fourteen qualification checks on the GPU.
It calls the full canonical analytical DCTR routine at N=1024, K=1024 and T=20,
with Contract E, GenUT, UKF per-particle covariance and dual-cap correction.
The independent FP64 matrix Kalman posterior supplies the endpoint. This is
engineering evidence that the intended surrogate-HMC computation is wired
correctly, not posterior-equivalence evidence.

The original field was healthy at all four declared calibration positions.
Direct-call parity, bitwise replay, permutation, row independence, endpoint
independence, sign and status, typed bindings, finite states, involution,
per-chain energy, endpoint health and cached endpoints all passed. Proposal
reversal error was 2.78e-17 against the declared scale-aware tolerance
9.78e-6. The field still uses an unselected protection hypothesis and cannot
authorize retained inference without LG2 calibration and public tuning.

The four-chain force replay took 1.776 seconds; a complete two-leap transition
took 4.326 seconds. These single measurements describe cost only. They imply
that the initial 2 GPU-hour / 0.5 CPU-worker-hour scope allocation needs a
complete forecast before tuning and retention. A shortened chain cannot be
substituted for the declared inference schedule to make the budget fit.

The qualification consumed 143.800454 GPU seconds and 198.952951 CPU-worker
seconds, including its watchdog accounting. LG2 preparation through this stage
totals 143.800454 GPU seconds and 253.471058 CPU-worker seconds. Whole-campaign
incremental accounting through LG1 archive inspection and this qualification
is 7860.208690 GPU seconds and 8108.672048 CPU-worker seconds, within the
unchanged 63 GPU-hour / 15 CPU-worker-hour ceiling.

The [plan](surrogate-hmc-lg2-consumer-qualification-2026-09-20.md) defines the
checks. The [report](artifacts/surrogate-hmc-all-models-20260920-lg2-consumer-01/qualification/report.json),
[run manifest](artifacts/surrogate-hmc-all-models-20260920-lg2-consumer-01/qualification/run-manifest.json)
and [accounting](artifacts/surrogate-hmc-all-models-20260920-lg2-consumer-01/accounting.json)
preserve the command, source snapshot, input identities, timings, device,
verified memory growth, TF32 and XLA settings. CPU preparation deliberately hid
GPU devices; this qualification used trusted GPU access.

| Decision | Primary criterion | Veto diagnostics | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Consumer qualified for scoped calibration | All fourteen checks passed | None observed at declared positions | Behavior outside these positions; control non-harm; total inference cost | LG2 calibration/validation curve and complete cost forecast | Posterior recovery or all-model completion |

| Inference status | Evidence |
|---|---|
| Hard veto screen | No numerical, replay, coupling or arithmetic veto in this bounded diagnostic |
| Statistically supported ranking | None |
| Descriptive-only differences | Force and transition timings; no efficiency comparison |
| Default readiness | Not established; heuristic dominance NOT_RUN |
| Next evidence needed | Selected LG2 controls, verified public tuning and retained comparisons against qualified references |

Post-run red team: the strongest alternative explanation for a later failure
is poor control selection or inadequate mixing rather than incorrect endpoint
correction. Four points and two transitions cannot resolve it. A failed
replay, call-chain or per-chain arithmetic check elsewhere would invalidate
this consumer's broader qualification and trigger a focused repair. A poor
force approximation alone would not reject the surrogate-HMC mechanism.
