# SSL-LSTM q=20 NeuTra adaptive replay A/B result

Date: 2026-08-24

Status: `SCREEN_COMPLETED_NO_PROMOTION`

Plan: [A/B comparison plan](bayesfilter-ssl-lstm-q20-adaptive-replay-neutra-ab-comparison-plan-2026-08-24.md)

## Outcome

The amended plan was reviewed, repaired through bounded CPU retries, and
executed on the trusted GPU route. Candidate A and Candidate B both completed
the paired 24-update screen for seeds 17 and 29. No hard route, target-status,
finite-value, batch-native, freeze-order, memory-growth, XLA, or artifact
integrity veto fired in the final GPU run.

This is a viable implementation screen, not evidence that either transport is
well trained. The validation-block latent moments remain far from an IID standard
Gaussian. Neither arm is admitted to HMC or a posterior claim.

## Run manifest

| Field | Value |
|---|---|
| Git commit | `68ba5271989fe35740416dff599bb61c83dfa099` |
| Worktree | Dirty before and during the run; unrelated HMC edits were preserved |
| Environment | `tfgpu`, Python executable `/home/ubuntu/anaconda3/envs/tfgpu/bin/python` |
| GPU | Physical selector `1`, RTX 4080 SUPER, one visible logical GPU |
| Allocator | TensorFlow memory growth verified before logical-device initialization |
| Numerical mode | TensorFlow/XLA, float64, TF32 disabled, batch-native target |
| Seeds | Paired seeds `17` and `29` |
| Block/update configuration | 64 rows per block, four blocks per update, 24 updates, width 16, two tanh IAF stages, learning rate `3e-4` |
| Validation partition | One fixed per-seed validation block shared by A and B; no independent audit block in this bounded screen |
| Screen wall time | `2342.2090410579985 s` |
| Target signature | `9a86e60081f1b9cd288dbdb1dcbe1e9a5b5e23d9b5ef97afdb72ee95c23d7278` |
| Geometry SHA-256 | `dc3dd7b84566867bc49c11ad16f50778d21457adbb398a17c2a75f3c3b461eeb` |
| Runner SHA-256 | `ccf51bfef363cc0bbbdfe6770599e7400566cdda5d33e26c2e13dff1ade4c89a` |
| Plan SHA-256 bound at launch | `7c323f3d6d9e3ce913c8eac0339f3790cd7a08f276e61c05afe476f4711c9fca` |
| Terminal plan SHA-256 | `4df94a4744bfc77559831750b784c66e093367d21efccd543c2110f336130c08` |

The launch artifacts retain the plan hash that was present when the GPU
screen was started. The terminal hash includes the post-run status,
execution-record, and exact command/root annotation. Those edits record the
already executed command and do not change the scientific contract or rerun
semantics. Both hashes are preserved in the machine-readable manifest.

The machine-readable manifest is
`docs/plans/artifacts/ssl-lstm-q20-adaptive-replay-neutra-ab-2026-08-24/r1-gpu/manifest.json`.

## Decision table

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Candidate A route | Completed all 24 updates for both paired seeds with fixed-law whole-block refresh | No hard veto | Finite self-normalized blocks and only two seeds | Keep as the fixed-law control while repairing the estimator/geometry | Theorem-2 convergence, whitening, HMC, or superiority |
| Candidate B route | Completed all 24 updates for both paired seeds; 27 adaptive block records per seed had matching pre/post transport hashes | No hard veto | Adaptive proposal ESS and finite-block ratio variance are poor | Improve proposal/block estimator before any promotion attempt | Theorem-2A convergence, whitening, HMC, or superiority |
| A versus B | Both remained viable under the engineering screen | No route veto | Two paired seeds provide no ranking uncertainty analysis | Use a larger, independently replicated estimator/geometry study | B is better or worse |
| HMC admission | No HMC phase was authorized by this plan | Not eligible | No frozen transport passed a whitening/global-mixing gate | Write a separate HMC plan only after transport admission | Posterior correctness or mode mixing |

## Observed diagnostics

The following are descriptive, not promotion evidence.

| Arm/seed | Final validation loss | Max absolute latent mean | Max absolute covariance-minus-identity | Validation ESS fraction |
|---|---:|---:|---:|---:|
| A/17 | 12.867839 | 0.7032 | 11.6264 | 0.4295 |
| B/17 | 12.876355 | 0.7032 | 11.6376 | 0.4295 |
| A/29 | 21.469370 | 1.1421 | 24.8500 | 0.2031 |
| B/29 | 21.474676 | 1.1399 | 24.8663 | 0.2031 |

The paired B-minus-A validation-loss differences were `+0.008516` (seed 17) and
`+0.005306` (seed 29). With two seeds, no interval or test supports a ranking;
these differences are descriptive only. Adaptive preflight ESS fractions were
`0.0990` and `0.1717`, compared with fixed-proposal fractions `0.3425` and
`0.5260`. This is a repair signal for the adaptive proposal, not a posterior
ESS or a hard correctness veto.

## Inference-status table

| Evidence class | Status | Interpretation |
|---|---|---|
| Hard veto screen | Passed | All final A/B arms were finite and route-compliant; GPU policy and artifact checks passed |
| Statistically supported ranking | None | Two paired seeds and no uncertainty analysis cannot rank A and B |
| Descriptive-only differences | B had slightly higher validation loss; adaptive ESS was lower in preflight | Nomination/repair information only |
| Default readiness | Not ready | The estimator is self-normalized and finite-block; whitening is poor and no HMC gate was run |
| Next evidence needed | Fresh larger blocks, proposal-tail repair, theorem-bearing unnormalized estimator, independent validation, then frozen-transport HMC | Required before any scientific or default claim |

## Harness repairs

The first CPU smoke exposed two static-shape defects at the new block-combination
boundary. The first repair added explicit block shapes; the second made the
adaptive stale buffer fixed-capacity and gave A and B equal four-block batch
shapes. The failed roots are preserved under `r1`, `r1-retry-01`, and
`r1-retry-02`; the passing CPU smoke and GPU run are under `r1-retry-03` and
`r1-gpu`. These were localized implementation repairs under the same target,
method, data, hardware class, and budget; they are not scientific failures.

## Red-team conclusion

The strongest alternative explanation for the near-equality of A and B is that
the current screen is dominated by the same fixed validation block and a
self-normalized finite-block objective; it does not isolate long-run proposal
adaptation. The absence of an independent audit block also forbids a
generalization claim. A result that would overturn the current interpretation
would be a fresh, disjoint, unnormalized known-density/SMC-U study showing stable
improvement in an independent audit pullback-moment study and then a common transformed-target
HMC pass. The weakest evidence here is transport quality: the screen is short,
uses only two seeds, and the adaptive proposal has visibly poor importance
weight concentration.

## Nonclaims

This result does not establish IID Gaussian whitening, exhaustive mode
discovery, posterior correctness, HMC convergence, predictive equivalence,
optimizer convergence, statistical superiority, default readiness, or the
global strong-monotonicity assumptions in the mathematical note.
