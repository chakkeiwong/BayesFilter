# Phase 13 Exact Weighted Affine-Whitening Oracle Subplan

Program: `docs/plans/bayesfilter-ssl-lstm-q20-particle-authority-master-program-2026-08-25.md`  
Status: `PASS_AFFINE_ORACLE_ROLE_LIMITED`  
Budget cap: `1800 s` within the unchanged global `64800 s` cap  
Input: Phase 8 metadata-bound/audited N=300 bank  
Output root:
`docs/plans/artifacts/ssl-lstm-q20-particle-authority-master-2026-08-25/phase13`

## Objective

Compute the exact finite weighted affine map that centers the M0 particles and
restores their weighted covariance to the identity. This is an oracle for
first/second moments, not a learned density or posterior transport. It
separates a moment-conditioning problem from a NeuTra optimization/capacity
problem.

## Evidence contract and audit

Use TensorFlow CPU-hidden operations only, with no NumPy or HMC. Require finite
particles/weights, a positive-definite weighted covariance, and residuals at
most `1e-10` for the weighted mean and covariance after the Cholesky map. The
input hash and Phase 8 audit remain the authority; the affine map is a
diagnostic comparator and cannot promote M0 or NeuTra.

## Execution and refresh

Run
`docs/benchmarks/run_ssl_lstm_q20_particle_authority_affine_oracle_2026_08_25.py`
on the audited bank. If the oracle passes, refresh toward a preconditioned
NeuTra or representation audit; if it fails, classify covariance/support or
input integrity before changing the target. A passing oracle is not an IID
density proof.
