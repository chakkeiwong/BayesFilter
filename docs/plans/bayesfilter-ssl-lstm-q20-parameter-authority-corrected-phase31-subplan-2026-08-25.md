# Corrected q=20 NeuTra Boundary Phase 31 Subplan

Parent: `bayesfilter-ssl-lstm-q20-parameter-authority-corrected-continuation-2026-08-25.md`  
Entry receipt: Phase 28 `PASS_THETA_MEASURE_PILOT` (Phase 30 GenUT candidate veto is nonblocking)  
Status: `READY_TO_EXECUTE`  
Local cap: 7200 s

## Question

Can the fresh four-parameter theta bank bind to the repository's GPU/XLA,
batch-native weighted NeuTra transport without accidentally training on the
60D UKF state or using a scalar target fallback?

## Skeptical boundary

This is a transport/data-boundary candidate screen, not HMC. It uses the
corrected M0 bank as a fixed empirical training measure. The target is called
on untouched theta audit rows only; training updates use the batched weighted
transport objective. A good loss or latent covariance is descriptive and does
not certify posterior correctness or IID whitening.

GPU memory growth must be configured and verified before logical-device
initialization. Every optimizer update must have batch size greater than one,
XLA enabled, finite gradients, and explicit target/status checks. CPU training
is not an allowed substitute for this claim-bearing lane.

## Command

Run with trusted GPU access:

```text
TF_FORCE_GPU_ALLOW_GROWTH=true PYTHONUNBUFFERED=1 \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/run_ssl_lstm_q20_parameter_authority_corrected_phase31_2026_08_25.py \
  --authority-root docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase28-fresh-theta-pilot \
  --output-root docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase31-neutra-boundary \
  --steps 3
```

## Repair and refresh

Classify a failure as GPU policy, XLA/operator, target/status, trainer,
serialization, or candidate. GPU/XLA failures may be repaired only without
changing the target or evidence contract, using a fresh root. A candidate
loss/whitening failure is not a whole-direction blocker; it refreshes Phase 32
with the measured transport evidence and leaves HMC deferred.

