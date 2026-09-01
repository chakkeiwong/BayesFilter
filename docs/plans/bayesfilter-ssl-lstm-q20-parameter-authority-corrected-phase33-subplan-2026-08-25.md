# Corrected q=20 Longer NeuTra Trace Phase 33 Subplan

Parent: `bayesfilter-ssl-lstm-q20-parameter-authority-corrected-continuation-2026-08-25.md`  
Entry gate: Phase 32 hard-gate replication passed  
Status: `READY_TO_EXECUTE`  
Local cap: 7200 s

## Question

Does a longer target-specific GPU/XLA batch-native weighted NeuTra trace
materially reduce the transport loss and latent moment residuals on the fresh
theta empirical bank, or do residuals persist after the short-screen phase?

## Design

Reuse the exact Phase 31 runner, target, split, two arms, and protocol, but run
20 optimizer updates in a fresh artifact root. This is a continuation of the
same empirical-measure boundary, not a claim-bearing HMC or posterior run.
Record every step, finite gradients, target/status audit, transport parity,
and device/memory policy. Compare only descriptively with the three-step
receipt; no ranking or superiority claim is allowed without a predeclared
multi-seed uncertainty analysis.

## Command

```text
TF_FORCE_GPU_ALLOW_GROWTH=true PYTHONUNBUFFERED=1 \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/run_ssl_lstm_q20_parameter_authority_corrected_phase31_2026_08_25.py \
  --authority-root docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase28-fresh-theta-pilot \
  --output-root docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase33-neutra-trace \
  --steps 20 --seed 20260825 3301
```

## Repair and refresh

Repair only localized GPU/XLA/trainer/harness failures in fresh roots under
the unchanged contract. If the longer trace remains finite but moment
residuals persist, classify that as a transport candidate limitation and carry
it to adjudication; it is not a proof that NeuTra or the research direction is
impossible. If a hard target/status or memory-policy veto fires, stop that arm
and preserve the failure.

