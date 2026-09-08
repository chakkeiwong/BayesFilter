# Corrected q=20 Fresh-Theta ETPF Phase 29 Subplan

Parent: `bayesfilter-ssl-lstm-q20-parameter-authority-corrected-continuation-2026-08-25.md`  
Entry receipt: Phase 28 `PASS_THETA_MEASURE_PILOT`  
Status: `READY_TO_EXECUTE`  
Local cap: 3600 s

## Question

Does the source-faithful second-order ETPF map operate on a fresh weighted
theta bank and produce finite target/status-compatible rows, while retaining
its empirical-transform and no-density nonclaims?

## Design and gates

Load only the metadata-bound M0 `final_theta` and normalized weights from the
Phase 28 receipt. Verify every digest, shape `[N,4]`, target signature,
protocol hash, and weight normalization. Apply the source fixture map to a
deterministic 32-row subset to bound the quadratic diagnostic cost. Evaluate
the batch-native q=20 target on the transformed rows.

Hard gates are finite transform, Riccati convergence, source/analysis row and
column marginal residuals, finite target, and valid target status. Covariance
residual, support excursions, and negative correction fraction are explanatory
and role-limited. The transformed empirical cloud receives no proposal density
or Jacobian term.

## Command

```text
CUDA_VISIBLE_DEVICES=-1 TF_CPP_MIN_LOG_LEVEL=3 TF_FORCE_GPU_ALLOW_GROWTH=true \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/run_ssl_lstm_q20_parameter_authority_corrected_phase29_2026_08_25.py \
  --authority-root docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase28-fresh-theta-pilot \
  --output-root docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase29-fresh-theta-etpf
```

## Repair and refresh

If convergence or covariance residual fails, preserve the attempt and adjust
only the declared Riccati stopping control in a fresh root. If the target
rejects transformed rows, classify that as a role/candidate failure and do not
assign a density to them. A passing role receipt refreshes Phase 30 as a
parameter-space GenUT scope decision; it does not admit ETPF as an authority.

