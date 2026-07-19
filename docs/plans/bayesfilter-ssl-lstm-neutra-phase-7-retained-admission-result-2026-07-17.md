# SSL-LSTM NeuTra Phase 7 Retained Admission Result

Date: 2026-07-17

Status: `PHASE7_RETAINED_ADMISSION_PASSED_PHASE8_HANDOFF`

## Result

Fresh G and H independently admitted at the second prospective checkpoint,
with `512` retained draws per chain and four preserved chains per chart. Both
passed the frozen gates in chart-specific `z` and common mapped `theta`
coordinates. The predeclared mapped-parameter cross-replication stability
screen also passed.

Authoritative public receipt:

- path:
  `docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/phase-7-retained-admission/retained-acquisition.json`
- SHA-256:
  `b79e5f6041e284de40bbd3834cc909fd12f45d012f172e570acccaa62dbe31a5`

Stage A timing receipt SHA-256:
`647be960a5307d564d1777d9cee5488262f3345ac0fd46ae0a5aea05367841ef`.
Its mechanics samples were excluded from retained evidence.

## Decision Table

| Decision | Primary criterion | Veto diagnostics | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Hand admitted G/H archives to Phase 8 design refresh | Both charts independently passed R-hat, bulk/tail ESS, MCSE/SD, acceptance, movement, finite value/score, XLA/GPU, and hash-lineage gates; mapped-function stability passed | No hard veto; native divergence was unavailable, not zero | Finite chains and no posterior oracle; stability screen is not a formal equivalence test or mode-coverage proof | Audit and calibrate the Phase 8 predictive design before opening any confirmatory forecast bank | Posterior truth/correctness, stationarity proof, complete tail/mode coverage, predictive equivalence, superiority, model adequacy, or default readiness |

## Admission Evidence

| Chart | Draws/chain | Acceptance | Max R-hat `z/theta` | Min bulk ESS `z/theta` | Min tail ESS `z/theta` | Max MCSE/SD `z/theta` |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| G | 512 | 0.6689 | 1.0323 / 1.0310 | 832.4 / 1,027.2 | 242.7 / 190.3 | 0.0372 / 0.0319 |
| H | 512 | 0.7036 | 1.0401 / 1.0378 | 629.6 / 614.0 | 142.8 / 134.8 | 0.0421 / 0.0411 |

All per-chain cumulative acceptance rates were in the frozen `[0.55,0.85]`
band:

- G: `0.6660`, `0.6816`, `0.6621`, `0.6660`;
- H: `0.6738`, `0.7266`, `0.6875`, `0.7266`.

At `256` draws per chain, G extended because both `z` and `theta` tail ESS
were below `100`. H extended because `z`/`theta` R-hat and `theta` tail ESS
still failed. These were expected promotion vetoes and correctly triggered the
next frozen segment. Both passed at `512`.

Native divergence was `not_exposed_by_kernel` for all four retained segments.
This is unavailability and provides no zero-divergence evidence. All chains
moved, all core telemetry was finite, and every segment's full retained-point
value/score and mapped-`theta` XLA/GPU audits passed.

## Cross-Replication Evidence

The prospectively fixed 14 mapped-`theta` functionals were four coordinate
means and ten upper-triangular raw second moments. The maximum absolute G/H
difference divided by the combined mean MCSE was `1.1646`, below the frozen
`3.0` stability veto.

This means no predeclared material instability was detected at the available
Monte Carlo precision. It does not establish equality, equivalence, a method
ranking, posterior correctness, or complete mode/tail agreement. All
continuous G/H differences are descriptive.

## Inference Status

| Evidence class | Status |
| --- | --- |
| Hard veto screen | Passed; no artifact, target, value/score, movement, finite-telemetry, native-divergence, GPU/XLA, lineage, or resource veto |
| Viable candidates | G and H both independently admitted |
| Statistically supported ranking | None; G/H were not ranked |
| Descriptive-only differences | Acceptance, timing, marginal/second-moment values, individual diagnostic differences |
| Default readiness | Not assessed |
| Next evidence needed | Phase 8 comparator resolution and predictive-design calibration, then separately blinded Phase 9 confirmation |

## Run Manifest

| Field | Value |
| --- | --- |
| Git commit | `ffaaaf903354e095da126dbfa47878c34717c5b8` with unrelated dirty work preserved |
| Environment | conda `tfgpu`; Python `3.13.13`; TensorFlow `2.20.0`; TFP `0.25.0` |
| Device | physical GPU 1 exposed as logical `/device:GPU:0`; RTX 4080 SUPER |
| Numerical policy | TensorFlow/TFP `float64`; XLA JIT on; TF32 enabled and recorded |
| Kernel | Identity mass, `epsilon=0.8`, `L=4`, trajectory length `3.2` |
| Burn-in | `128` only before segment 0 for each chart |
| Seeds used | G `(8101,8102)`, `(8111,8112)`; H `(9101,9102)`, `(9111,9112)` |
| Wall time | `1008.0399` seconds versus `2100` second cap |
| Chart times | G `501.8583` seconds; H `506.1196` seconds, each below `1050` seconds |
| Trust basis | `owner_designated_managed_session_visible_gpu_trusted` |

All `16` private shard/sidecar/manifest files used for the four retained
segments and both continuation lineage edges were independently hash-verified
after the run. The public receipt contains no raw sample paths or raw samples.

## Scope Refinement

The historical master-program Phase 7 text listed predictive-feature and
broader ridge/covariance sensitivity comparisons. The reviewed live Phase 7
plan deliberately narrowed the pre-predictive gate to independent sampler
admission plus the 14 mapped-parameter functionals, because opening forecast
features before the Phase 8 calibration audit would contaminate predictive
design selection. Predictive features, controlled alternatives, covariance
regularization, and forecast sensitivity therefore remain required Phase 8
work; they are not claimed as completed here.

## Post-Run Red Team

| Question | Assessment |
| --- | --- |
| Strongest alternative explanation | Both global transports could emphasize the same posterior region while missing a common remote mode or tail; finite diagnostics and G/H stability cannot exclude this |
| What would overturn the handoff | Hash/source replay failure, a defect in chain/draw ordering or diagnostics, predictive calibration revealing an invalid shared forecast/statistics path, or later independent evidence of material mode/tail instability |
| Weakest evidence | No posterior oracle or formal mode-coverage proof; cross-replication threshold is a conservative screen rather than a calibrated equivalence test |
| Why proceed | The actual Phase 7 question was sampler admission and bounded independent-replication stability. Both passed every prospective gate, and Phase 8 is explicitly designed to add predictive falsification rather than relabel this as truth |

Phase 8 may use only the admitted retained archives and their exact lineage.
No Phase 8 confirmatory forecast outputs are opened by this result.
