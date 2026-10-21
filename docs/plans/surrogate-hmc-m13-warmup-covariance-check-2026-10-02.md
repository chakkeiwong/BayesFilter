# M13 covariance from archived warmup, 2026-10-02

Question: what covariance estimate would the existing 9,000 warmup transitions
per chain supply, compared with the frozen covariance actually used? Target is
the same M13/data1704/T20 frozen importance-likelihood posterior. This is a
CPU-only post-run diagnostic; no new HMC, adaptation, force/target evaluation,
posterior promotion, sample reclassification or stopping-rule change.

Evidence contract: compare the original saved covariance with the empirical
sample covariance of all four warmup chains (36,000 correlated states), and with
the same covariance processed by the actual repository regularizer: 90% sample
covariance plus 10% saved prior covariance, followed by the saved jitter/floor
policy. Verify original factors, checksums, aggregate/chunk equality, warmup
phase and count, state continuity, and finite SPD outputs. These validity checks
veto this diagnostic only. Covariance entries, SDs, correlations, eigenvalues,
and geometry of unchanged retained draws under alternative factors are
explanatory only. There is no sampler-performance promotion criterion.

Sensitivity views: all 9,000, last 4,500, and last 1,000 warmup transitions per
chain; also report each chain separately. Half-window and final-window summaries
are illustrative sensitivity checks, not a selected tuning window or new
default. Last 1,000 is also the archived warmup-readiness window. Report whether
the estimate changes across windows rather than asserting 9,000 steps suffice.
Retained draws are used only to describe transformed covariance under each
fixed retrospective factor, never to select a winning factor.

Default/assumption audit: unbiased finite sample covariance matches the existing
estimator algebra. The exact existing regularizer is the comparator; retain
its prior scale, shrinkage and eigenvalue settings, and separately show the
unregularized estimate to expose shrinkage effects. Warmup is not assumed to be
stationary and 36,000 correlated states are not 36,000 independent draws.
Pooled estimates can include between-chain separation; chain/window summaries
expose this. No counterfactual acceptance, ESS, R-hat, runtime or convergence
benefit can be inferred from re-expressing existing states. A continuously
adapting kernel would produce different draws. Heuristic-adversary performance
ranking is inapplicable to this descriptive covariance calculation.

Skeptical audit: no posterior samples enter the fitted covariance; use archived
raw warmup coordinates reconstructed with the original affine map, not latent
states mislabeled as raw parameters. Do not claim same-sample identity covariance
proves improved whitening. Saved-draw comparisons answer the user's numerical
question but cannot prove a future sampler repair. These controls address wrong
baselines, hidden shrinkage, sample-selection and stale-adaptation assumptions.
Audit passes; proceed.

Budget: five CPU minutes, at most two local attempts. CUDA_VISIBLE_DEVICES=-1
before TensorFlow import; TensorFlow/TFP calculations in a non-XLA post-run
reporting exception. No sampling budget is required or consumed. Preserve all
old evidence. Output root:
docs/plans/artifacts/surrogate-hmc-m13-warmup-covariance-check-20261002-01/.
Command: /home/chakwong/anaconda3/envs/tftwogpu/bin/python followed by that
root's analyze.py, with complete command log, structured report, result note
and run manifest. Sampling allocation remains expired.

Attempt 01 stopped before results: the independent estimator-parity wrapper
used a dynamic state count, but the repository Welford helper requires a static
count. Localized repair: use exact [36000,2] signature for the saved warmup.
Preserve attempt 01; retry output root ends in -02. No sampler code changed.
