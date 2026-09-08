# Phase 14 Affine-Preconditioned NeuTra Subplan

Program: `docs/plans/bayesfilter-ssl-lstm-q20-particle-authority-master-program-2026-08-25.md`  
Status: `PASS_HARD_GATES_ROLE_LIMITED_FULLBANK_DIAGNOSTIC`  
Budget cap: `3600 s` within the unchanged global `64800 s` cap  
Input: Phase 8 metadata-bound/audited N=300 bank and Phase 13 affine oracle  
Output root:
`docs/plans/artifacts/ssl-lstm-q20-particle-authority-master-2026-08-25/phase14`

## Objective

Train the same NeuTra arms in the exact weighted affine-whitened coordinates,
then compose the affine map and its log-determinant back to physical space for
target/status checks. This tests whether the learned residuals are a
conditioning problem. It does not change the target or admit the affine map as
a posterior density.

## Skeptical audit and evidence contract

Phase 13 proves the affine covariance is positive definite and its finite
weighted moments whiten to machine precision. The preconditioned run must use
the same immutable rows/weights, explicit `mode_axis=2`, disjoint 180/60/60
split, GPU memory growth before device initialization, batch-native XLA
updates, and composed affine+flow logdet checks. Latent moments and loss are
explanatory only; no ranking or IID promotion is allowed.

Run the existing runner with `--precondition affine --profile tuning
--steps 300` and a fresh output root. If residuals improve substantially,
refresh a frozen representation candidate; if they remain poor, the evidence
shifts toward objective/measure support rather than simple conditioning.

Execution receipt: `phase14-attempt5-affine-fullbank2401` completed with all
hard gates passing. The affine map is exact for the measured first two moments,
but the best full-bank learned-flow residuals remain nonzero (mean max
`0.1197`, off-diagonal max `0.1243`, covariance Frobenius residual `0.3575`
for `compact_low_lr`). The validation subset is not a substitute for this
full-bank diagnostic. No IID, posterior, HMC, or default claim is admitted.
