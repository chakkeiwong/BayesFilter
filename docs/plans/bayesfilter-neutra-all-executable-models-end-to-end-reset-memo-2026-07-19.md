# NeuTra All-Executable-Models Reset Memo

Date: 2026-07-19

The campaign created a reusable, batched GPU/XLA end-to-end composition and
completed fresh segmented training for four of five executable cells. The
implementation repair was successful: training segments are bitwise-equivalent
to uninterrupted training on a deterministic fixture, target status telemetry
is normalized through the existing BayesFilter binding, the real HMC runner is
used in preflight, and the PP-SGQF target identity is now CPU/GPU stable.

The campaign did not produce HMC samples. Every trained cell was rejected by
the current native tuning gate for one of: fixed-verifier folded R-hat, energy
error, ladder budget exhaustion, or nonfinite screen log-accept diagnostics.
These are distinct tuning/diagnostic outcomes, not one uniform NeuTra failure.

Most important planning defect: the tuner’s fixed 1,000-result modern R-hat
handoff veto can prevent the existing adaptive sequential sampler from extending
warm-up/retained draws to its declared 10,000-per-chain cap. The next plan must
resolve this contract explicitly, while keeping the native BayesFilter tuner
and final sequential folded-R-hat/ESS gates intact.

Preserved evidence roots:

- `docs/plans/artifacts/bayesfilter-neutra-all-executable-models-e2e-20260718/serious-attempt-02`
- `docs/plans/artifacts/bayesfilter-neutra-all-executable-models-e2e-20260718/serious-continuation-attempt-02`
- `docs/plans/artifacts/bayesfilter-neutra-all-executable-models-e2e-20260718/serious-continuation-attempt-01`

No historical artifact was overwritten. No claim about posterior correctness,
convergence, truth recovery, or NeuTra validity is authorized from this run.
