# Multidimensional LGSSM SVD Score Wiring Demotion Result

Date: 2026-07-10

## Decision

The multidimensional lower-triangular LGSSM serious-HMC target now demotes the
historical QR derivative route as wrong relative to the active serious-HMC
target. The QR derivative route is not deleted, but it is no longer wired into
`bayesfilter/testing/multidim_triangular_lgssm_tf.py`.

## Reason

The previous serious LGSSM target called
`tf_qr_linear_gaussian_score`, which uses Python-unrolled time and parameter
loops in its derivative helper. For this `T=120`, `parameter_dim=18` target,
the observed large memory use is therefore consistent with static XLA graph
growth rather than model size.

The active serious-HMC route now calls
`tf_svd_linear_gaussian_score_first_order_graph_status`, which uses a
TensorFlow `tf.while_loop` Kalman recursion and vectorized first-derivative
tensors. Nonzero SVD/eigh status is treated as a hard HMC-gradient veto by
emitting NaN value/score tensors and preserving status telemetry.

## Prior DSGE Wiring Check

Checked `/home/chakwong/python` prior DSGE NeuTra/Kalman wiring:

- NK SVD-UKF NeuTra canary uses
  `NKSVDUKFBatchedPosteriorAdapter` in
  `/home/chakwong/python/scripts/train_nk_svd_ukf_neutra_phase2_canary.py`.
- Rotemberg SSM equivalence uses
  `bayesfilter.linear.tf_svd_linear_gaussian_score_first_order_graph_status`
  in `/home/chakwong/python/src/dsge_hmc/experiment_adapters/ssm_equivalence.py`.
- SGU linear Kalman HMC/XLA canary records
  `bayesfilter/linear/kalman_svd_derivatives_tf.py` as the BayesFilter linear
  SVD derivative authority in
  `/home/chakwong/python/scripts/run_sgu_linear_kalman_phase3_hmc_xla_worker_canary.py`.
- Legacy DSGE SVD Kalman filters live in
  `/home/chakwong/python/src/dsge_hmc/filters/_svd_filters.py`.

## Nonclaims

- No HMC burn-in or retained sampling was run.
- No posterior convergence, posterior recovery, HMC readiness, production
  readiness, or scientific validity claim is made.
- This is a wiring and compile-risk repair only.
