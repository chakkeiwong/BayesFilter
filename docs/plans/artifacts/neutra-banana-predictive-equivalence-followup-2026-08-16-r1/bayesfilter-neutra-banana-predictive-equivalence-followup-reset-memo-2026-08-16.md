# NeuTra Banana Predictive-Equivalence Follow-up Reset Memo (2026-08-16)

## State

The frozen banana candidate is unchanged: seed-15 learned transport, 6,000
training updates, identity z mass, `L=10`, step size `0.7709722545680272`.
The prior 1,024-draw predictive screen failed its descriptive MMD envelope.

The larger-window follow-up is complete at:

`docs/plans/artifacts/neutra-banana-predictive-equivalence-followup-2026-08-16-r1/`

It used 4,096 draws per chain, offsets 0 and 904 from the fixed 5,000-draw
archive, 128 exact-vs-exact calibration banks, 512 bootstrap replicates, block
lengths 32/64/128, GPU 0, float64/XLA, TF32 off, and verified memory growth.
All artifact hashes passed.

## Scientific verdict

The point MMD decreased by about fourfold versus the 1,024-draw run, and
coordinate/banana moments are close to exact values. Nevertheless, the
candidate upper interval exceeded the empirical exact-control q99 at all six
window/block cells by 4.7%-13.8%, and exceeded q95 in all cells. This is
persistent small finite-screen discrepancy evidence, not a formal equality
rejection. The two candidate windows overlap and are not independent.

The harness and candidate binding are valid. No training or kernel change is
authorized by this result alone.

## Next-session entry point

1. Keep the frozen HMC candidate and both predictive artifacts as holdout evidence.
2. Run a feature/tail decomposition of the MMD using latent-coordinate and
   model-coordinate projections plus predeclared nonlinear banana features.
3. Give the decomposition the same exact-vs-exact calibration treatment and
   preserve the block dependence handling.
4. If one feature family carries the excess, investigate that learned transport
   component; if no feature survives calibration, classify the original MMD
   excess as finite-sample/underpowered.
5. Do not retrain, retune, or transfer to SSL-LSTM until that diagnostic is complete.
