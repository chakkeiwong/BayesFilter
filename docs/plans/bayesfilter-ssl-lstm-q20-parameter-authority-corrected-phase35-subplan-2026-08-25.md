# Corrected q=20 Affine Theta NeuTra Repair Phase 35 Subplan

Parent: `bayesfilter-ssl-lstm-q20-parameter-authority-corrected-continuation-2026-08-25.md`  
Entry gate: Phase 34 finite trace with persistent latent moment residuals  
Status: `SUPERSEDED_BY_PHASE35R_MEASURE_REPAIR`  
Local cap: 7200 s

## Question

Does an exact weighted affine preconditioner computed from the *same normalized
training-subset measure consumed by the optimizer* improve optimization
conditioning and latent moment residuals without changing the target, particle
dimension, or density composition?

## Mathematical contract

Let `I_train` be the frozen training split and let `w_i` be the same floored
weights passed to the optimizer, renormalized over `I_train`. Let `m` and `L`
be the weighted theta mean and lower Cholesky factor of that training-measure
covariance, with `L` nonsingular. Define `z=L^{-1}(theta-m)` and
`theta=m+L z`. The density composition is
`log q_theta(theta)=log q_z(z)-log|det L|`; target evaluations remain at
`theta`, and any chart determinant is recorded separately. The affine map is a
conditioning device, not a posterior transform theorem.

## Design and gates

The prior attempt used full-bank moments while training on a 40-row subset and
is therefore not a valid whitening comparison. Run the repaired Phase 31
runner in a fresh root with `--precondition affine --steps 200`; compute the
factor from the exact training split and emit train-measure oracle moments,
held-out validation moments, and audit status separately. Run a paired
identity arm under the same frozen split and seed. Require finite Cholesky,
round-trip/determinant parity, exact train-measure oracle residuals, batch size
>1, memory growth, XLA, target/status, and finite gradient gates. Compare
identity and affine traces descriptively; do not rank them without a
multi-seed uncertainty analysis.

## MathDevMCP audit

Before launch, run and preserve:

```text
PYTHONPATH=/home/ubuntu/python/MathDevMCP/src \
  /home/ubuntu/.venvs/mathdevmcp-mcp/bin/python -m mathdevmcp.cli \
  prove-or-counterexample \
  "log_q_theta(theta)=log_q_z(inv_L(theta-m))-log_abs_det_L" \
  --assumption "det(L) != 0"
```

Treat parser `inconclusive` or scope-limited output as a limitation, not a
certificate; the runner's finite numerical parity check remains the executable
gate.

## Command

```text
TF_FORCE_GPU_ALLOW_GROWTH=true PYTHONUNBUFFERED=1 \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/run_ssl_lstm_q20_parameter_authority_corrected_phase31_2026_08_25.py \
  --authority-root docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase28-fresh-theta-pilot \
  --output-root docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase35r-affine-training-measure-repair \
  --precondition affine --steps 200 --seed 20260825 3501
```

## Repair and refresh

An affine Cholesky, train-measure oracle, or composition failure is repaired in
a fresh root. A finite run with held-out residuals still present is a candidate
limitation and feeds Phase 36; it is not a target/measure blocker. No HMC or
canonical LEDH status may be inferred.

The prior full-bank attempt remains historical diagnostic evidence at
`phase35-affine-neutra-repair/attempt1`; it must not be used as the corrected
identity-versus-affine whitening comparison.
