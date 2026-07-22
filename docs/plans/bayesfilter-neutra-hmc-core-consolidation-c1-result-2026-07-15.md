# NeuTra HMC Core Consolidation Phase C1 Result

Date: 2026-07-15  
Decision: `PASS_C1_SHARED_CONTROLLER`

## Outcome

`bayesfilter/inference/neutra_hmc.py` now owns the TensorFlow/TFP batched HMC
mechanics and sequential warm-up/retained controller under policy identifier
`bayesfilter_neutra_sequential_hmc_v1`.

The controller infers chain count and dimension, requires at least four chains
for sequential decisions, retains but excludes warm-up, checks a recent warm-up
window with modern rank-normalized split/folded R-hat, checks retained draws
cumulatively, permits a full-convergence callback, reuses one compiled program
per chunk size, applies health vetoes, and enforces 10,000-sample caps. Model
coordinates, target-status summaries, and archives enter through callbacks.

## Evidence

- Focused shared-core and route-policy tests passed, including a real
  CPU-hidden XLA Gaussian smoke.
- Static inspection found no NumPy, host callback, LGSSM token, or campaign
  path in the core.
- Warm-up/retained seed roots must be distinct, and deterministic chunk seed
  derivation is covered exactly.
- C1 establishes engineering behavior only. It does not establish posterior
  correctness, robustness, or default readiness by itself.

## Handoff

C2 may migrate the active LGSSM campaign while preserving its archive schema.
The shared controller, not a campaign copy, must remain the only reachable
sampler implementation.
