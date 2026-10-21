# Actual-SV consumer result — 2026-09-20

The full canonical DCTR consumer for actual SV20 passes all fourteen GPU
mechanics checks. Every original force evaluation at the qualification probes
was healthy. Both four-chain transitions had no fallback, divergence or endpoint
failure. This is implementation evidence for testing surrogate HMC, not a
posterior-recovery result.

The consumer uses the raw-return weight density, stationary parameter-dependent
initialization, analytical recursion and the existing probit chart. The full
Contract E / GenUT / UKF / pairwise / coordinate-cap implementation is shared
with the canonical filter. An independent 1024-interval FP64 quadrature provides
the endpoint. Its model posterior was qualified in the preceding stage.

Evidence: `artifacts/surrogate-hmc-all-models-20260920-sv20-consumer-01`, including
`P3-SV20-qualify/report.json`, `consumer.json`, `binding.json`, source snapshots
and run manifest. The exact master command is in the launch manifest and
`surrogate-hmc-sv20-consumer-2026-09-20.md`. All 56 focused tests passed before
launch. The public capability registry was checked: later raw-coordinate field
tuning must use `tune_hmc_kernel` with the typed endpoint-corrected binding.

Replay, permutation, batch independence, actual call parity, force sign/status,
endpoint independence, reversal, energy accounting and cache checks passed.
Round-trip absolute error was zero versus a declared tolerance of 4.39e-6.
The compiled four-position force cost about 1.03 seconds; a two-leap transition
cost 2.85 seconds. These are descriptive timings, not efficiency comparisons.
Peak TensorFlow allocated memory was 134,381,056 bytes. The run used GPU1,
FP32/TF32 DCTR, FP64 integrator/reference, XLA and verified memory growth.

Actual master charges: 80.093701 GPU seconds / 81.620656 CPU-worker seconds.
The two preceding test jobs charged 22.551943 CPU seconds. Campaign cumulative
use is now 8520.168984 GPU / 9010.816930 CPU-worker seconds. All remain within
the original budgets. No local retry was needed.

| Decision | Primary criterion | Veto diagnostics | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Qualify actual consumer mechanics | 14/14 checks pass | No numerical, map or artifact failure detected | Only four calibration positions and short trajectories | SV-specific protection calibration and inference forecast | Posterior recovery, force accuracy, convergence, efficiency or default readiness |

| Inference status | Finding |
|---|---|
| Hard veto screen | No failure under tested mechanics |
| Statistically supported ranking | None |
| Descriptive-only differences | Timings and short-transition energy errors |
| Default-readiness | Not evaluated; controls remain unselected hypotheses |
| Next evidence needed | Own control calibration, public tuning, calibrated MCSE diagnostics and retained comparisons |

Post-run red team: healthy central probes do not establish force health over the
prior or the eventual trajectory. Boundary stress probes and independent
calibration/validation curves are next. Matching the direct canonical route
does not prove that finite DCTR approximates the model posterior; D/R remains a
separate comparison. The existing MCSE false-veto issue must be repaired
prospectively before new inferential decisions. Closed LG1/C01 outcomes stay
unchanged.
