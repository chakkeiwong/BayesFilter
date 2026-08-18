# NeuTra Banana Predictive-Equivalence Reset Memo (2026-08-16)

## State

The frozen banana HMC candidate remains the seed-15 learned transport with
identity z mass, `L=10`, step size `0.7709722545680272`, and the sequential HMC
controller from the 2026-08-16 confirmation artifact. Its prior convergence and
limited exact-law moment screens passed for both start banks.

The new predictive/output-law diagnostic is complete at:

`docs/plans/artifacts/neutra-banana-predictive-equivalence-2026-08-16-r2/`

It used GPU 0 with verified TensorFlow memory growth, float64, XLA, TF32 off,
four chains, 1,024 draws per chain, offsets 0 and 1024, block lengths 32/64/128,
and 32 exact-vs-exact calibration banks per offset. Artifact hashes pass.

## Scientific verdict

The candidate's 99% block-bootstrap MMD upper interval exceeded the empirical
exact-vs-exact calibration envelope at both offsets and every block length. The
screen therefore failed under this finite target-specific diagnostic. This is a
repair trigger and a descriptive discrepancy, not a formal equality rejection,
because the custom calibration envelope has finite Monte Carlo support and no
formal p-value was defined.

The harness is valid: the candidate archive was hash-bound to the confirmation
result, exact banks used independent stateless seeds, the MMD point matched the
repository implementation to roundoff, and all values were finite. No HMC
candidate or training default is promoted or rejected solely on this result.

## Next-session entry point

1. Do not retune or retrain from this result alone.
2. Preserve the current candidate and archive as holdout evidence.
3. Run a larger retained-window and exact-vs-exact calibration diagnostic with
   an independent archive offset, predeclaring the larger calibration count and
   block policy.
4. If the excess persists, inspect tail/nonlinear projections and the learned
   transport's law error; if it disappears, classify the original screen as
   underpowered/descriptively noisy.
5. Do not transfer this banana result to SSL-LSTM.
