# SSL-LSTM q=20 NeuTra global-mixing execution result (2026-08-19)

Plan: `docs/plans/bayesfilter-ssl-lstm-q20-neutra-global-mixing-execution-plan-2026-08-19.md`

Status: `TERMINAL_UNDER_BUDGETED_WITHOUT_POSTERIOR_ADMISSION`

## Outcome

The recovered campaign executed successfully through trusted GPU preflight,
the anti-pooling mechanics canary, the eight-cell batch-native NeuTra training
screen, frozen heldout audits, and two ordered fixed-HMC tuning attempts for
transport seed 2. It then stopped under the predeclared compute veto.

The terminal classification is `UNDER_BUDGETED_HMC`, not a harness failure and
not a rejection of the NeuTra research direction. The only scientifically
adjudicated HMC kernel is seed 2 with `L=3`; it is rejected because modern R-hat
failed. The `L=5` short screen was finite, but its long verification was
refused before GPU execution. No common-kernel sequential posterior bank was
admitted, so predictive validation was correctly not run.

## Claimed target and quantity computed

| Item | Verdict |
|---|---|
| Claimed target | Retained draws from one exact transformed-target fixed-HMC kernel satisfying `bayesfilter_neutra_sequential_hmc_v1`, all four parameter and sign-indicator R-hat/ESS gates, and direct per-chain cross-sign transitions. |
| Quantity actually computed | Two audited NeuTra transports; seed 2 `L=3` tune/screen plus 2,000-result-per-chain fixed-kernel verification; seed 2 `L=5` tune/screen only. |
| Relation | The `L=3` verification is valid tuning-veto evidence but fails the target gate. The `L=5` screen is tuning evidence only and is different from the required verification and canonical sequential posterior. |
| Artifact support | HMC result SHA-256 `714a51ef0f7179fced5e3e2972217b36e72ca017740b7173372f35f781d16402`; HMC manifest SHA-256 `56260c8896e8294db56af374c2cdf6648e47af42d53dd9f9e204ee9824c2de2e`. |
| Not established | `L=5`, `L=10`, or `L=15` convergence; transport seed 2 global failure; transport seed 3 performance; posterior correctness; mode weights; predictive equality; production or default readiness. |

## Phase results

| Phase | Evidence | Decision |
|---|---|---|
| Trusted GPU preflight | First XLA attempt consumed `9.5 s` and failed an over-strict roundoff check; repaired retry consumed `3.208 s`, verified memory growth/device/XLA, and has SHA-256 `28be07fcc83be539b9b643f2127094f025706d0a92b733873a85ddeb56b50a45`. | Local harness repair, then engineering pass. |
| Mechanics canary | Launch-invalid memory-policy-order attempt consumed `10.3 s`; repaired retry completed exact pullback and short fixed HMC in `790.290 s`. Every chain remained mode-locked although pooled occupancy was balanced. | Local harness repair, then anti-pooling pass; no convergence claim. |
| Training screen | Eight primary cells completed in `902.912 s`; seed 2 and seed 3 width-64, 3-stage, `3e-4` arms passed the frozen audit. | Two downstream nominees; no statistical ranking. |
| Seed 2, `L=3` | `2,000 x 4 x 4` finite samples; target status valid; descriptive acceptance `0.796313`; max modern R-hat `1.134678`. | Kernel rejected by R-hat hard veto. |
| Seed 2, `L=5` | Finite first-round screen; step size `0.249446`; descriptive acceptance `0.763694`. Long verification forecast `15,546.542 s` plus closeout versus `12,540.174 s` remaining. | Verification not run; resource stop only. |
| Canonical sequential HMC | Not run because no kernel completed tuning admission. | Not checked. |
| Predictive endpoint | Not run because no posterior bank was admitted. | Ineligible, not failed. |

## Decision table

| Decision | Primary criterion status | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Stop this campaign | No common kernel reached canonical sequential admission | Campaign resource continuation veto fired | Unrun kernels and seed 3 may behave differently | Preserve artifacts; use a fresh reviewed budget/root for any continuation | NeuTra failure or target invalidity |
| Reject seed 2 `L=3` kernel | Tuning R-hat gate failed | Rank R-hat `1.134678`; folded R-hat `1.129265` for observation weight | Longer verification might reduce Monte Carlo error but is not evidence from this run | Do not promote or reuse this kernel as admitted | Transport seed 2 is globally unusable |
| Classify seed 2 `L=5` | Required verification absent | Resource veto only; tuner missing-diagnostic labels are induced by the refused call | Verification R-hat, ESS, status, and mixing are unknown | Next smallest discriminating run is a fresh, adequately budgeted `L=5` verification | `L=5` passes or fails |
| Run predictive tests | Sampler prerequisite absent | Admission veto | No posterior-predictive draws exist | Keep predictive output absent | Predictive equality or inequality |

## Inference status

| Evidence class | Status |
|---|---|
| Hard veto screen | Seed 2 `L=3` fails modern R-hat. No numerical, target-status, identity, memory-policy, or artifact-integrity veto fired. |
| Viable candidates | Seed 2 `L=5`, seed 2 larger lengths, and transport seed 3 remain unadjudicated, not passed. |
| Statistically supported ranking | None. Training seeds and unrun kernels cannot be ranked from these data. |
| Descriptive-only differences | Training losses, HMC acceptance, energy-proxy alert, wall time, and short-screen behavior. |
| Default-readiness | Not established. No posterior bank or predictive endpoint exists. |
| Next evidence needed | Fresh `L=5` full verification, then canonical sequential warmup/retention with R-hat, ESS, finite/status, energy, and direct per-chain transition gates. |

## Engineering ledger

- The launch bound target signature
  `9a86e60081f1b9cd288dbdb1dcbe1e9a5b5e23d9b5ef97afdb72ee95c23d7278`,
  adapter signature
  `a8be6c212eb2a74ef926ccb9279871805949cae6e00c312a12525141638166f3`,
  and geometry SHA-256
  `dc3dd7b84566867bc49c11ad16f50778d21457adbb398a17c2a75f3c3b461eeb`.
- TensorFlow/TFP ran on physical GPU 1 as logical `GPU:0`, float64, XLA on,
  TF32 off, with memory growth configured and verified before initialization.
- The canonical route-policy ledger classified all 40 discovered routes and
  bound this runner directly to `bayesfilter_neutra_sequential_hmc_v1`.
- All nine terminal HMC hashes were recomputed after the run with zero
  mismatches. Output roots were fresh and no artifact was overwritten.
- TensorFlow retraced the distinct tuning chain shapes. This increased runtime
  and host-side graph cache but did not alter the target or invalidate results.
- The HMC runner omitted `19.8 s` of failed-attempt GPU wall from its prior-wall
  budget input. Its artifact therefore reports `1,696.410 s` rather than the
  correct `1,716.210 s`, and overstates the HMC remainder by `19.8 s`. Actual
  execution ended far below both the reported and corrected cap; the smaller
  corrected remainder strengthens rather than reverses the `L=5` refusal.

## Sampler ledger

- `L=3` produced finite states, target values, scores, and per-chain target
  status. Native divergence was not exposed and therefore is not claimed zero.
- `L=3` observation-weight rank R-hat `1.134678` and folded R-hat `1.129265`
  fail the `1.01` tuning threshold. Acceptance `0.796313` cannot override this.
- The maximum absolute log-accept energy proxy was capped at `1e100` and raised
  an explanatory alert. It is not the hard veto used for the decision.
- `L=5` passed its finite short screen, then the resource callback refused the
  10,320-leapfrog-transition verification before execution. The tuner encoded
  the resulting exception as missing verification diagnostics; the campaign
  terminal artifact correctly reclassifies those labels as resource-induced.
- Canonical warmup, retained sampling, ESS, sign-indicator diagnostics, and
  direct per-chain crossing were never reached.

## Scientific ledger

- The run rejects one kernel, seed 2 with `L=3`, relative to the stated tuning
  criterion.
- It does not reject the trained seed 2 transport, seed 3 transport, exact
  target, or NeuTra mechanism because the planned repair kernels were not fully
  tested.
- It provides no posterior probability measure and no posterior-predictive
  evidence. Pooled mode occupancy remains inadmissible.
- No stochastic ranking, superiority, exhaustive-mode, identifiability,
  production, or default-readiness claim is supported.

## Failure classification

| Class | Finding |
|---|---|
| Implementation failure | Budget accounting omitted `19.8 s` of failed-attempt GPU wall. It did not breach the campaign cap or change the decision. Identities, exact pullback, shapes, finiteness, route policy, XLA, memory growth, and artifact hashes passed. |
| Tuning failure | Seed 2 `L=3` failed the required modern R-hat gate. `L=5` tuning did not complete because verification was resource-refused. |
| Diagnostic failure | None for `L=3`. The `L=5` missing-diagnostic labels do not describe a broken diagnostic; their input chain was deliberately not run. |
| Scientific evidence | Evidence against the `L=3` kernel only. The broader NeuTra hypothesis remains unresolved. |
| Terminal cause | Campaign under-budgeted for the next indivisible required call. |

## Run manifest

| Field | Value |
|---|---|
| Git | Commit `5699dafec23de9549a8092bec638997e7973593c`; dirty shared worktree preserved |
| Environment | Conda `tfgpu`; Python `3.13.13`; TensorFlow/TFP GPU/XLA |
| Device | Physical GPU selector `1`, NVIDIA GeForce RTX 4080 SUPER; logical `GPU:0`; managed-session trust basis recorded |
| Numerical mode | float64; XLA JIT true; TF32 false; verified memory growth |
| Data/target | Receipt-verified q=20 replay inputs; target and adapter signatures listed above |
| Training seeds | Transport seeds 2 and 3; eight predeclared architecture/rate cells |
| HMC seeds | `L=3`: tune `(20260819,32000)`, screen `(20260819,42000)`, verification `(20260819,52000)`; `L=5`: `(20260819,32100)`, `(20260819,42100)`, `(20260819,52001)` |
| Wall time | Failed preflight `9.5 s`; successful preflight `3.208 s`; invalid canary `10.3 s`; successful canary `790.290 s`; training `902.912 s`; HMC `10,782.834 s`; corrected cumulative GPU wall `12,499.045 s` |
| HMC budget | Artifact reports internal `23,323 s` and external `23,503 s`; both were `19.8 s` too high because retry wall was omitted. Actual HMC wall was `10,782.834 s`, so no cap was breached. Predictive reserve `3,600 s` was not borrowed. |
| HMC output | `docs/plans/artifacts/ssl-lstm-q20-neutra-global-mixing-execution-2026-08-19/r1/hmc/` |
| Training output | `docs/plans/artifacts/ssl-lstm-q20-neutra-global-mixing-execution-2026-08-19/r1/training-screen/` |
| Plan at HMC launch | SHA-256 `309340acaf5a0702ffd0f8999f062aceba4c4e87bd595d34ac28e4cc11e2f81c` |

Exact HMC command:

```bash
TF_FORCE_GPU_ALLOW_GROWTH=true timeout 23503s \
/home/ubuntu/anaconda3/envs/tfgpu/bin/python \
docs/benchmarks/run_ssl_lstm_q20_neutra_global_mixing_hmc_2026_08_19.py \
  --device 1 --campaign-wall-cap-seconds 28800 \
  --predictive-reserve-seconds 3600 --time-cap-seconds 23323 \
  --training-root docs/plans/artifacts/ssl-lstm-q20-neutra-global-mixing-execution-2026-08-19/r1/training-screen \
  --output-root docs/plans/artifacts/ssl-lstm-q20-neutra-global-mixing-execution-2026-08-19/r1/hmc
```

## Verification

- Final CPU-only focused suite: `61 passed` across the canonical sequential
  controller, route-policy ledger, q=20 anti-pooling diagnostics, and fixed-HMC
  tuner. GPUs were intentionally hidden. The 228 warnings were dependency
  deprecations from TensorFlow Probability/Gast, not failures.
- Prelaunch expanded controller/campaign/end-to-end suite: `109 passed`, with
  two baseline failures caused by the absent historical ignored registry
  `docs/plans/artifacts/multimodel-neutra-filter-posterior-20260715/phase-p0/attempt-04-20260715T1658/target_registry.json`.
  Those tests do not exercise the changed q=20 route.
- HMC artifact hash inventory: nine entries recomputed, zero mismatches.
- `git diff --check` passed, and no predictive output directory exists.

## Next evidence and budget provenance

The next smallest discriminating artifact is a fresh seed 2 `L=5` full
verification in a new output root. The existing canary-derived conservative
forecast for that indivisible call is `15,546.542 s`, before the `180 s`
closeout reserve. If it passes, the canonical minimum adds 2,000 warmup and
2,000 retained results, or 20,000 more leapfrog transitions at `L=5`; the same
canary rate and 1.25 allowance imply about `30,129 s` more. Thus a complete
`L=5` verification-plus-minimum-sequential path needs roughly `45,676 s`
(`12.69 h`) plus rerun/orchestration overhead. This is a derived planning
estimate, not a performance guarantee and not authority to launch a new
campaign. It also excludes fallback lengths and transport seed 3.

## Post-run red team

The strongest alternative explanation for the `L=3` R-hat failure is that the
fixed verification length was insufficient for its slow observation-weight
coordinate; that possibility does not make the failed `1.01` gate pass. The
strongest alternative explanation for the campaign stop is simple compute
underestimation aggravated by TensorFlow retracing, not a mathematical or
transport failure. The post-run accounting audit also exposed a `19.8 s`
retry-wall omission; correcting it cannot rescue any failed gate or make the
refused call affordable.

Evidence that would overturn the terminal scientific nondecision is a fresh,
adequately budgeted common-kernel run that passes verification and the full
canonical sequential ledger. The weakest part of the present scientific
evidence is breadth: only one transport seed and one kernel length completed
long verification. The strongest part is the clean failure attribution: the
one rejected kernel failed a checked R-hat gate, while the unrun work is
preserved as unrun rather than mislabeled as failure.
