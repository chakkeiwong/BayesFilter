# NeuTra Banana Repair Reset Memo (2026-08-16)

## State

- The prior replication/HMC campaign remains authoritative for the Gaussian
  control and unresolved for banana under seeds `10,11,12`.
- This banana-only repair campaign used the reviewed plan
  `docs/plans/bayesfilter-neutra-banana-repair-plan-2026-08-16.md`.
- The corrected terminal artifacts are under
  `docs/plans/artifacts/neutra-banana-repair-2026-08-16-r3/`.
- The terminal runner is
  `docs/benchmarks/run_neutra_banana_repair_2026_08_16.py`.

## What Was Established

1. The root-preserving `(32,32)` banana transport with peak `LR=5e-4` passed
   all three fresh proposal-law audits for both 3,000 and 6,000 update arms on
   seeds `13,14,15`.
2. The 6,000-update arm is therefore a viable target-specific proposal-training
   candidate under the declared audit. This is not a universal budget/default
   claim.
3. Sequential HMC warm-up passed at 2,000 transitions per chain, but retained
   convergence failed at the first 500-draw chunk (`max R-hat=1.03027` versus
   the `1.01` threshold). The retained exact-law screen also failed coordinate
   mean index 5 (`3.75` standardized).
4. No banana HMC posterior result is valid or promoted. TFP native divergence
   telemetry was unavailable and is not treated as zero divergences.

## Artifact And Harness Notes

- Terminal wall time: `331.31 s`.
- GPU: logical GPU 0; TensorFlow memory growth verified before initialization.
- Numeric mode: float64, XLA enabled, TF32 disabled, batch size `4096`.
- Seeds and audit partitions are recorded in the terminal manifest and per-cell
  JSON files. All 25 terminal artifact hashes pass.
- An earlier `r2` run is preserved but non-promotable because its 6,000-update
  learning-rate phase boundaries moved with the horizon. The runner was fixed
  before `r3`; the fixed schedule horizon is recorded in the manifest.

## Next-Agent Constraints

- Do not transfer the 6,000-update banana setting to SSL-LSTM.
- Do not run HMC from a proposal that fails the exact-law audit.
- Do not interpret the `r3` HMC acceptance/ESS or warm-up pass as posterior
  correctness; retained convergence and exact-law agreement failed.
- Do not relax the retained R-hat, exact-law, divergence, finite-state, or
  energy gates to rescue this candidate.
- Any HMC repair must have a new target-specific plan and isolate one of:
  fixed identity mass, initial-state bank, kernel/grid, or transport geometry.
- Preserve the Gaussian candidate as the bounded exact-law HMC control; it does
  not establish banana or SSL-LSTM readiness.
