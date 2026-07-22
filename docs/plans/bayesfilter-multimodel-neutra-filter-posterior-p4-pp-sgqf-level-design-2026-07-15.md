# P4 PP-SGQF Level Design

Date: 2026-07-15

Status: `FROZEN_BEFORE_SGQF_RESULTS`

## Target

Use the same seed-81104 T=20 predator-prey model, six-probit Uniform-box
posterior, initial-observation-first time order, and stabilized PF1 reference as
the admitted PP-UKF target, but define a distinct deterministic fixed-SGQF
filter posterior and target identity.

Candidate sparse levels: 2, 3, 4. Reference level: 5. Level 1 is excluded
before results because its single point cannot represent a nonzero covariance.

## Admission Rules

At the four frozen PF audit points, a candidate level must simultaneously:

- have finite values/scores, strictly positive predictive/innovation/filtered
  covariance, zero status codes, and no active covariance floor;
- place every likelihood value inside the stabilized PF1 95% interval expanded
  by 1.0 log unit;
- agree with the next candidate and level 5 within 0.25 log unit in total
  likelihood; and
- agree with the next candidate and level 5 within 0.5 in every
  source-coordinate likelihood-score component.

Select the smallest passing candidate. These convergence margins are fixed
before results and are filter-admission vetoes, not superiority criteria.

## Engineering Gates

- graph-native batch `[B,6]`, TensorFlow float64;
- `tf.while_loop` over time and RK4 substeps, tensorized cloud and batch axes;
- manual Cholesky and moment forward sensitivities;
- analytic y0 assimilation before the first transition;
- centered-FD score stability, batch permutation, CPU XLA, trusted GPU XLA,
  memory growth, and no NumPy/callback/Python active-axis loops;
- independent posterior recomposition and cross-filter substitution negatives.

## Nonclaims And Stops

Passing admits one PP-SGQF approximate posterior identity only. It does not show
SGQF exactness, PF exactness, SGQF superiority to UKF, HMC convergence, NeuTra
quality, calibration, or readiness. If no level passes, block only PP-SGQF and
continue the PP-UKF comparator pipeline. Do not add levels or loosen margins
after inspection.
