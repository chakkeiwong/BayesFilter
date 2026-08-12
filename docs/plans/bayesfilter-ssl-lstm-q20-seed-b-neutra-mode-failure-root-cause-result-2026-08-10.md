# SSL-LSTM q=20 seed-B NeuTra mode-failure root-cause result (2026-08-10)

## Verdict

The failed seed-B distribution has two established, interacting causes.

1. **NeuTra training missed the negative mode almost completely.** Among
   100,000 iid standard-normal base draws mapped through the frozen reverse-KL
   transport, only 3 entered the negative observation-weight half-space
   (`0.00003`). The negative source MAP mapped to
   `z=(-0.2746,-2.7569,-7.7138,-10.9031)`, with standard-normal log density
   `-96.704`; the positive MAP mapped near the origin, with base log density
   `-3.729`. This is direct proposal-coverage evidence for reverse-KL mode
   omission, not an exact posterior mode-weight estimate.
2. **The HMC kernel was tuned only for the positive-region geometry and is
   numerically unusable in the negative transformed mode.** The exact
   Jacobian-corrected pullback has two distinct stationary regions `23.707`
   latent units apart. At the negative transformed stationary point, the frozen
   step `epsilon=0.811521`, `L=3` accepted `0/32` local proposals; negative
   proposed target values ranged from approximately `-1.5e4` to `-1e100`.
   Negative-mode maximum local precision was `91.72047`, giving the quadratic
   leapfrog stability scale `2/sqrt(lambda_max)=0.208832`. A controlled rerun
   changing only the step to curvature-derived `epsilon=0.1` accepted `31/32`
   negative-region proposals and reduced maximum absolute log-acceptance ratio
   from `1e100` to `0.0566`. Thus the tested kernel failure is specifically a
   step-size/curvature mismatch, not an invalid negative target or merely a bad
   initial point.

All original material starts mapped to the positive half-space and lay only
`0.326` to `1.270` latent units from the inverse positive MAP, but `13.095` to
`13.882` units from the inverse negative MAP. The original low R-hat and high
ESS therefore measured local positive-region mixing only.

The correct scientific classification is:

- `transport_training_failure`: supported;
- `transformed_geometry_failure`: supported;
- `initialization_failure`: supported as a contributor;
- `frozen_kernel_local_failure_in_negative_region`: supported and causally
  localized to the selected step size;
- exact posterior mode weights: not established;
- NeuTra as a general method: not rejected.

## Claimed and computed quantities

| Item | Classification |
|---|---|
| Claimed target | Why the seed-B NeuTra/fixed-HMC archive omitted the known negative observation-weight region. |
| Quantity actually computed | Frozen-flow proposal sign coverage; inverse maps; exact pullback values/scores and optimized stationary points; stable local Hessians; two sampled path profiles; original-start mapping; frozen-kernel split starts; and a curvature-derived step-size control. |
| Relation | Directly diagnoses proposal coverage and this frozen kernel's local behavior. Local Laplace mass and sampled paths are explanatory approximations, not exact posterior-mass or minimum-barrier calculations. |
| Source anchor | Historical code commit `9ebaecc59f792f49bf7b946342ea512e71f5b3e4`, hash-verified checkpoint/trainer state, target signatures, archived trace tensors, and artifact hashes below. |
| Not proved | Exact basin probabilities, exhaustive mode discovery, minimum energy barrier, cross-mode mixing under a repaired kernel, posterior correctness, predictive validity, or an optimal replacement kernel. |

## Evidence summary

### Transport training and latent geometry

| Diagnostic | Positive region | Negative region | Interpretation |
|---|---:|---:|---|
| Flow proposal sign count, `n=100,000` | 99,997 | 3 | Reverse-KL proposal almost entirely omitted the negative half-space. |
| Inverse source-MAP `z` norm | `0.326` | `13.640` | Negative source MAP is an extreme standard-normal-tail event under the learned map. |
| Standard-normal log density at inverse source MAP | `-3.729` | `-96.704` | About 93 log units of base-density separation. |
| Exact transformed log density at inverse source MAP | `-38.769` | `-39.834` | The exact pullback retains meaningful density at the negative inverse despite proposal omission. |
| Optimized transformed stationary `z` | `(0.0304,0.0765,0.0027,0.2197)` | `(-0.2654,-4.2251,-7.4730,-21.8608)` | Distinct stationary regions; latent distance `23.707`. |
| Transformed stationary log density | `-38.746` | `-39.790` | Point densities alone do not determine masses. |
| Local Laplace two-mode fraction | `0.5112` | `0.4888` | Explanatory local approximation only; suggests the negative peak is not trivially negligible. |
| Maximum transformed precision eigenvalue | `1.1633` | `91.7205` | Strong region-dependent curvature remains after NeuTra. |
| Leapfrog quadratic stability scale | `1.8543` | `0.2088` | Frozen step `0.8115` is locally plausible near positive but 3.89 times the negative stability scale. |

The physical-straight inverse-mapped path had sampled potential rises `5.869`
from positive and `4.803` from negative. The latent-straight path gave `6.186`
and `5.121`. These are sampled path heuristics; neither is a minimum-barrier
bound, and neither explains the local numerical blow-up as directly as the
curvature/step control.

### Initialization and kernel controls

| Run | Starts | Epsilon, L | Positive binary acceptance | Negative binary acceptance | Cross-sign transitions | Verdict |
|---|---|---|---:|---:|---:|---|
| Original archive | Four local positive-region starts | `0.811521`, `3` | Archive overall about `0.70` | Not tested | 0 retained | Initial bank did not test global coverage. |
| Inverse-source-MAP canary | 16 per region, 8 transitions | `0.811521`, `3` | `0.7266` | `0` | 0 | Negative source-MAP inverse is unusable under frozen kernel, but not transformed-stationary. |
| Transformed-stationary canary | 8 per region, 4 transitions | `0.811521`, `3` | `0.7188` | `0` | 0 | Zero negative acceptance is not caused by using a nonstationary inverse source MAP. |
| Curvature-derived causal control | Same transformed starts and budgets | `0.1`, `3` | `1.0` | `0.96875` | 0 | Local negative proposals recover when the step is below the curvature stability scale. |

No cross-sign transition occurred in these short controls. That is expected and
does not test whether `epsilon=0.1`, `L=3` can mix globally; its trajectory
length is only `0.3`, versus latent separation `23.7`.

## Decision table

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | What is not concluded |
|---|---|---|---|---|---|
| Reject the frozen seed-B transport as a global multimodal chart | Supported: only `3/100,000` learned proposal draws entered the negative half-space | No numerical/target veto in geometry artifact | Exact posterior negative-mode mass remains unknown | Retrain with independently discovered, weighted coverage of both modes or use a genuinely multimodal transport/base | NeuTra cannot work when trained with global coverage |
| Reject frozen `epsilon=0.811521`, `L=3` for a two-region campaign | Supported: `0/32` negative stationary proposals accepted; control at `0.1` accepted `31/32` | Trace finite; extreme finite `-1e100` sentinel proposals expose numerical instability | Optimal region-aware/global kernel remains untuned | Tune with both regions represented and add a global transition mechanism | `epsilon=0.1`, `L=3` is an accepted replacement or mixes modes |
| Treat original R-hat/ESS as positive-region-only | Supported: all starts and all 4,000 retained draws were positive | No archive-integrity veto | Unarchived intermediate leapfrog states were not inspected | Require overdispersed/multimode starts and mode-occupancy diagnostics in future convergence gates | Original local ESS calculations were numerically wrong |
| Do not run predeclared 32-per-region, 64-transition rung | Under-budgeted after measured canary: linear estimate `47,245 s` | Campaign cap preserved | Large-batch scaling is approximate | Stop this root-cause campaign after the smaller causal control | A larger frozen-kernel run would repair the defect |

## Inference-status table

| Evidence class | Status |
|---|---|
| Hard veto screen | Geometry and admitted controls had finite samples/traces and passed source/checkpoint/target/parity gates. Frozen negative-region kernel is rejected by zero acceptance and catastrophic proposals; this is a candidate veto, not a campaign-invalidity veto. |
| Statistically supported ranking | None. No stochastic method ranking was attempted. |
| Descriptive-only differences | Flow sign fraction, local Laplace fractions, path profiles, finite-run acceptance rates, runtime, and tail maxima. |
| Default-readiness | Not evaluated. Neither `epsilon=0.1` nor a new NeuTra training protocol is promoted. |
| Next evidence needed | Weighted multimodal mode discovery/reference; multimode-aware transport training; region-aware tuning; and a tempered/global transition campaign with uncertainty-aware mode-mass and predictive validation. |

## Engineering, numerical, and scientific ledgers

| Ledger | Result |
|---|---|
| Engineering correctness | Focused tests `7 passed`; artifacts are versioned and non-overwriting; CPU-only XLA route used; custom op SHA recorded; tensor receipts are hashed. |
| Numerical/sampler validity | Archived standalone parity residuals were `1.522e-7` value and `2.079e-7` score under a measured-derived `5e-7` compatibility gate. The historical run's transformed manifest hash is unreproducible because it was launched from a dirty worktree; `historical_identity_exact=false` is retained. Frozen-step negative proposals are locally invalid; step control is finite and highly accepted. |
| Scientific interpretation | Reverse-KL mode omission and positive-only initialization prevented discovery, while severe residual negative-mode curvature made the positive-tuned kernel unusable even if initialized there. Exact global weights and repaired global mixing remain open. |

## Run manifest

| Field | Value |
|---|---|
| Git commit | `9ebaecc59f792f49bf7b946342ea512e71f5b3e4` |
| Commands | Plan commands plus `BAYESFILTER_CODE_ROOT=/tmp/BayesFilter-seed-b-root-cause-historical`; exact commands are embedded in each artifact |
| Environment | Conda `tfgpu`, Python/TensorFlow versions embedded per artifact |
| CPU/GPU status | CPU only; `CUDA_VISIBLE_DEVICES=-1`; GPU intentionally hidden |
| XLA | Enabled for target, optimization, and HMC |
| Custom op | Shared local `_symmetric_sylvester_ops.so`, SHA-256 recorded in geometry bindings |
| Random seeds | Flow `(20260810,1001)`; inverse canary `(20260810,2001)`; stationary canary/control `(20260810,4001)` |
| Admitted wall time | `5,436.6002 s` total: geometry `963.0557`, inverse canary `2,952.8311`, stationary canary `747.9841`, step control `772.7293` |
| Campaign cap | `12,000 s`; passed |
| Output root | `docs/plans/artifacts/ssl-lstm-q20-seed-b-neutra-mode-failure-root-cause-2026-08-10/r1/` |
| Plan | `docs/plans/bayesfilter-ssl-lstm-q20-seed-b-neutra-mode-failure-root-cause-plan-2026-08-10.md` |
| Result | This file |

Principal JSON SHA-256 values:

- `geometry.json`: `dc3dd7b84566867bc49c11ad16f50778d21457adbb398a17c2a75f3c3b461eeb`
- `split-canary.json`: `9e83e8b20de87c671e05994950a482851d62edc5a6f96bdf9e770b674b21aecd`
- `split-stationary-canary.json`: `bd304a3abc3cccddfd610d60b743300cb7a4f4e3a82a2c06e3f66118ec6c7478`
- `split-stationary-step-control.json`: `425b410ae9b362432727091a087d5e4472ec94fd3158f8c8b5a9ef5965182654`

## Negative-result classification

| Question | Answer |
|---|---|
| Did the implementation/harness fail? | Several bounded harness defects and stale identity assumptions were found and repaired before admission. Final artifacts passed their gates. |
| Did NeuTra optimization fail numerically? | Not established. The trained map is finite and locally useful for the positive region, but its global proposal coverage failed. |
| Did the tuning procedure fail? | Yes relative to multimodal use: tuning from positive-local starts selected a step incompatible with negative-mode curvature. |
| Is this evidence against NeuTra in general? | No. It is evidence against reverse-KL training without weighted multimodal coverage and against one global positive-tuned fixed kernel for this learned chart. |
| What rescues the direction? | Independent mode discovery/weights, multimode-aware training, region-aware or tempered transitions, and confirmation that repaired chains reproduce mode weights and predictive laws. |

## Post-run red team

The strongest alternative explanation is that the known negative stationary
region has negligible exact posterior mass despite the local Laplace estimate.
That would weaken the scientific consequence of missing it, but it would not
change the measured facts that the flow omitted it and the frozen kernel was
locally unusable there. A valid multimodal SMC/AIS/tempered authority assigning
negligible negative mass would overturn the implied importance of balanced
coverage, not the root-cause mechanics.

The result that would overturn the kernel conclusion is an exact reproduction
showing nonzero finite acceptance at the negative transformed stationary point
with frozen `epsilon=0.811521`, `L=3`, or showing that the stationary point and
curvature were computed from a different scalar target. The archived parity,
target signatures, checkpoint hashes, and direct `epsilon=0.1` control argue
against those alternatives.

The weakest evidence is the mode-mass statement: `0.4888/0.5112` is only a
two-mode local Laplace approximation. No exact global posterior authority was
constructed, and the sampled path profiles are neither optimized nor rigorous
barrier bounds.

