# Corrected q=20 Parameter-Space GenUT Phase 30 Subplan

Parent: `bayesfilter-ssl-lstm-q20-parameter-authority-corrected-continuation-2026-08-25.md`  
Entry receipt: Phase 29 `PASS_FRESH_THETA_ETPF_ROLE_LIMITED`  
Status: `READY_TO_EXECUTE`  
Local cap: 3600 s

## Question

Is the generalized unscented transform feasible and target/status-compatible
when applied to the declared four-parameter theta cloud, and how does that
differ from the previously observed 20D/internal-state infeasibility?

## Design

Use the fresh Phase 28 M0 bank and its normalized weights. Run the 2d+1 GenUT
construction with `d=4`, then evaluate feasible sigma points with the q=20
target. If global feasibility fails, run sign-local subsets only as a scope
diagnostic. Do not clip negative weights or silently change the GenUT equations.

Moment residuals and mode fractions are explanatory. A global infeasibility is
a candidate scope result, not a theorem that parameter-space GenUT or all local
moment methods are impossible; it does not block Phase 31's independent
theta-bank boundary.

## Command

```text
CUDA_VISIBLE_DEVICES=-1 TF_CPP_MIN_LOG_LEVEL=3 TF_FORCE_GPU_ALLOW_GROWTH=true \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/run_ssl_lstm_q20_parameter_authority_corrected_phase30_2026_08_25.py \
  --authority-root docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase28-fresh-theta-pilot \
  --output-root docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase30-theta-genut-scope
```

## Repair and refresh

Classify failures as bank/hash, GenUT numerical, target/status, or harness.
Repair only ridge or artifact handling in a fresh root; never clip or alter
weights to force feasibility. Refresh Phase 31 with the measured global/local
scope result and preserve all nonclaims.

