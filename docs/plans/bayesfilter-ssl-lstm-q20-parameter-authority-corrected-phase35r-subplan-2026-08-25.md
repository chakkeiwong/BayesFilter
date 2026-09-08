# Corrected Parameter-Authority Phase 35R Subplan

Parent: `bayesfilter-ssl-lstm-q20-parameter-authority-corrected-continuation-2026-08-25.md`  
Supersedes: initial Phase 35 full-bank affine comparison  
Entry gate: skeptical measure-mismatch audit completed  
Status: `PASS_NEUTRA_BOUNDARY_ROLE_LIMITED`
Local cap: 1800 s

## Why this repair is required

The initial Phase 35 factor used all 64 M0 rows, but the optimizer consumed a
40-row training split with floored and renormalized weights. Its affine oracle
therefore whitened a different empirical measure from the one optimized. The
initial receipt remains historical diagnostic evidence, but it cannot support
an identity-versus-affine whitening comparison.

## Question

When the affine factor is computed from the exact normalized training-subset
measure consumed by the optimizer, does affine preconditioning improve the
training-measure and held-out latent diagnostics relative to an identity arm
with the same frozen split, target, seed, and update budget?

## Mathematical contract

For frozen training indices `I` and optimizer weights
`\tilde w_i = max(w_i, 10^-300) / sum_{j in I} max(w_j, 10^-300)`, compute

`m_I = sum_{i in I} \tilde w_i theta_i`,

`C_I = sum_{i in I} \tilde w_i (theta_i-m_I)(theta_i-m_I)^T`,

and a lower Cholesky factor `L_I` of `C_I`. The chart is
`z=L_I^{-1}(theta-m_I)` and the density composition is

`log q_theta(theta) = log q_z(z) - log|det L_I|`.

The target is always evaluated at physical `theta`; the 60D UKF state and 20D
innovation remain internal. The chart is a conditioning device, not a density
or posterior theorem.

## Evidence contract and gates

Run identity and affine arms in separate fresh roots with the same Phase 28 M0
bank, deterministic split, seeds, 200 updates, GPU/XLA path, and batch size.
Require finite target/status, finite gradients, exact transport and affine
round trips, positive-definite `C_I`, and a train-measure affine oracle with
maximum mean and covariance residual at most `1e-10`. Report train-measure,
validation, and untouched audit diagnostics separately.

Loss and latent moments are explanatory. No arm ranking is supported by this
one paired seed. No HMC or canonical LEDH route is allowed.

## Commands

```text
TF_FORCE_GPU_ALLOW_GROWTH=true PYTHONUNBUFFERED=1 \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/run_ssl_lstm_q20_parameter_authority_corrected_phase31_2026_08_25.py \
  --authority-root docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase28-fresh-theta-pilot \
  --output-root docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase35r-affine-training-measure-repair/identity-final \
  --precondition identity --steps 200 --seed 20260825 3501

TF_FORCE_GPU_ALLOW_GROWTH=true PYTHONUNBUFFERED=1 \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/run_ssl_lstm_q20_parameter_authority_corrected_phase31_2026_08_25.py \
  --authority-root docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase28-fresh-theta-pilot \
  --output-root docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase35r-affine-training-measure-repair/affine-final \
  --precondition affine --steps 200 --seed 20260825 3501
```

## Repair and refresh

If either command fails a platform, target, serialization, or train-measure
gate, preserve the failed root and repair only that boundary in a new root.
If both pass but held-out moments remain poor, classify that as a candidate
limitation and proceed to Phase 36. It is not a continuation veto.

Execution receipt: `docs/plans/bayesfilter-ssl-lstm-q20-parameter-authority-corrected-phase35r-repair-refresh-2026-08-25.md`.
