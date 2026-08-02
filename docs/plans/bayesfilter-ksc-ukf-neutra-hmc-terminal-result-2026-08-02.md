# KSC-UKF Gaussian-Sum NeuTra/HMC Continuation Result

Date: 2026-08-02 to 2026-08-03  
Plan: `docs/plans/bayesfilter-ksc-ukf-neutra-hmc-continuation-plan-2026-08-02.md`

## Decision

KSC-UKF is admitted through the repaired filter, GPU/XLA canary, batch-native
adapter, NeuTra recipe screen, and 5,000-step final training. HMC broad-grid
tuning was not completed because three trusted GPU launch requests timed out at
the platform permission boundary before process creation. No CPU substitution
was made and no HMC convergence claim is issued.

| Phase | Primary criterion | Result | Status | Next action |
| --- | --- | --- | --- | --- |
| CPU repaired filter | Dense value/score/status, mass, permutation gates | Caps 7, 16, 32, 64, 128, 256 passed; cap 32 max value gap `1.29e-5`/observation, score gap `1.72e-3`, retained mass `1.0` | passed | GPU canary |
| GPU/XLA canary | Cap-32 value gap `<=1e-10`, score gap `<=1e-8`, finite/status | `ADMIT_KSC_GAUSSIAN_SUM_UKF_GPU_CANARY`; RTX 4080 SUPER, XLA, memory growth | passed | adapter |
| Batch-native target | Issued signature, no scalar/row fallback, finite/status | Signature `727718ec8c4b4a68e2bc59c5f88d33be8e24cc4b77095f9197a360f6c9e7114d`; focused tests `4 passed`; dynamic batch and required telemetry pass | passed | recipe screen |
| NeuTra screen | Four KSC recipes, 500 GPU/XLA steps, batch 128 | All four passed hard gates; one-seed losses are descriptively near-identical; smallest viable `ksc_narrow_lr1e3` selected without supported ranking | passed | final training |
| Final training | One selected recipe, 5,000 GPU/XLA updates, frozen parity | `PASS_TRAINING_HARD_GATES`; all held-out rows valid, zero floors, transport/logdet/pullback parity `0` to `5.6e-17` | passed | broad-grid tuning |
| Broad-grid HMC | Independent epsilon for `L=(3,5,9,13,18,25)`, three replications, 65 draws | Three attempts blocked before process creation by permission-review timeout; no tuning artifact | pending infrastructure | retry unchanged GPU launch |
| Sequential HMC | Unique viable pair and shared sequential convergence gates | Not launched | not assessed | broad-grid result required |

## Evidence Paths

- GPU admission: `docs/plans/artifacts/bayesfilter-ksc-ukf-gpu-admission-neutra-20260802/canary-attempt01/result.json`
- Recipe screen: `docs/plans/artifacts/bayesfilter-ksc-ukf-neutra-hmc-20260802/screen-attempt08/KSC-UKF-GAUSSIAN-SUM-T20/result.json`
- Final training: `docs/plans/artifacts/bayesfilter-ksc-ukf-neutra-hmc-20260802/final-training-attempt01/KSC-UKF-GAUSSIAN-SUM-T20/result.json`
- Frozen transport: `docs/plans/artifacts/bayesfilter-ksc-ukf-neutra-hmc-20260802/final-training-attempt01/KSC-UKF-GAUSSIAN-SUM-T20/final/segments/steps-004001-005000/frozen_transport.json`
- Transport SHA-256: `dbbaba3735404d9dd98b233e9419ab4fd3d82c8ac9a5922c9e47712d42e8bddb`

## Inference Status

| Evidence class | Status |
| --- | --- |
| Hard veto screen | CPU/GPU filter and training gates pass; no broad-grid or sequential HMC evidence exists |
| Statistically supported ranking | none; recipe differences are one-seed descriptive values only |
| Descriptive-only differences | recipe reverse-KL means, component counts/mass, runtime, and GPU compilation diagnostics |
| Default readiness | not assessed |
| Next evidence needed | trusted GPU broad-grid result with a complete viable pair set, then sequential HMC |

## Known Repairs

- Scope hashing now computes the transformed frozen observations on `/CPU:0`,
  making identity independent of GPU initialization.
- The public batch method is directly graph-native and no longer delegates via
  `self`, satisfying the repository batch-source audit.
- The Gaussian-sum recurrence declares `[None, component_cap]` loop invariants,
  allowing the NeuTra `[None,2]` batch signature.
- Required `floor_count_value` and `min_innovation_eigenvalue` telemetry is
  returned by the batch callable itself.
- The dedicated runner constructs static scope metadata without importing
  TensorFlow target code before memory growth configuration.

## Nonclaims

This result does not claim exact likelihood, exact score, source faithfulness,
posterior correctness, HMC convergence, statistical superiority, production or
default readiness, or transfer of any SVX-ZC tuning setting to KSC.

## Post-Run Red Team

The strongest alternative explanation is that the short `T=20` repaired target
is easier than a future KSC claim horizon and that the deterministic component
merge may define a target unlike an author-native KSC implementation. A failed
broad-grid or sequential health gate would overturn downstream HMC readiness but
would not invalidate the completed filter/adapter/training evidence.

