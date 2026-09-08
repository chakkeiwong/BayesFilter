# Corrected q=20 Extended NeuTra Repair Phase 34 Subplan

Parent: `bayesfilter-ssl-lstm-q20-parameter-authority-corrected-continuation-2026-08-25.md`  
Entry gate: Phase 33 finite GPU/XLA trace with persistent whitening residuals  
Status: `READY_TO_EXECUTE`  
Local cap: 7200 s

## Question

Do the latent mean and covariance residuals continue to improve under a much
longer target-specific optimization trace, with all target, transport, batch,
and device contracts unchanged?

## Design

Run the corrected Phase 31 runner for 200 optimizer updates in a fresh root,
using the same Phase 28 M0 bank, frozen split, two arms, XLA, and memory-growth
policy. This is an optimization-time repair, not a change in target or measure.
Capture checkpoints at every update and compare only descriptively with the
three- and twenty-step receipts. The desired IID Gaussian behavior is an
explanatory diagnostic here, not a promotion criterion.

## Command

```text
TF_FORCE_GPU_ALLOW_GROWTH=true PYTHONUNBUFFERED=1 \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/run_ssl_lstm_q20_parameter_authority_corrected_phase31_2026_08_25.py \
  --authority-root docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase28-fresh-theta-pilot \
  --output-root docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase34-neutra-extended-trace \
  --steps 200 --seed 20260825 3401
```

## Repair and refresh

If the run remains finite but residuals plateau, classify that as evidence for
a representation/tuning repair (for example, an affine-preconditioned arm)
under a new reviewed subplan. Do not claim the transport is wrong merely from
one trace, and do not claim IID whitening merely from loss reduction. A hard
GPU, target, or serialization failure is repaired in a fresh root without
altering the scientific contract.

