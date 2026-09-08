# Corrected q=20 Fresh-Seed Replication Phase 32 Subplan

Parent: `bayesfilter-ssl-lstm-q20-parameter-authority-corrected-continuation-2026-08-25.md`  
Entry gate: Phase 31 GPU/XLA boundary passed  
Status: `READY_TO_EXECUTE`  
Local cap: 7200 s

## Question

Are the corrected theta-measure C0/M0 finite/status and mode diagnostics
reproducible across fresh stateless seeds under the same frozen protocol?

## Design

Run the Phase 28 runner twice more with seeds `20260825 2802` and
`20260825 2803`, each in a new root and with the same N=64, schedule, target,
and proposal controls. The old geometry remains a warm-start calibration
only. Aggregate the three seed receipts (including seed 2801) with the
standard-library reporter. Require all hard gates and exact target/measure
signature parity. Report means, sample standard deviations, and MCSE as
descriptive uncertainty; do not rank C0 versus M0 or call M0 SMC-U authority.

## Commands

```text
CUDA_VISIBLE_DEVICES=-1 TF_CPP_MIN_LOG_LEVEL=3 TF_FORCE_GPU_ALLOW_GROWTH=true \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/run_ssl_lstm_q20_parameter_authority_corrected_phase28_2026_08_25.py \
  --output-root docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase32-replication/seed2802 \
  --particles 64 --calibration-particles 16 --arms both --seed 20260825 2802

CUDA_VISIBLE_DEVICES=-1 TF_CPP_MIN_LOG_LEVEL=3 TF_FORCE_GPU_ALLOW_GROWTH=true \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/run_ssl_lstm_q20_parameter_authority_corrected_phase28_2026_08_25.py \
  --output-root docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase32-replication/seed2803 \
  --particles 64 --calibration-particles 16 --arms both --seed 20260825 2803

/home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/aggregate_ssl_lstm_q20_parameter_authority_corrected_phase32_2026_08_25.py \
  --output-root docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase32-replication/aggregate \
  --pilot-root docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase28-fresh-theta-pilot \
  --pilot-root docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase32-replication/seed2802 \
  --pilot-root docs/plans/artifacts/ssl-lstm-q20-parameter-authority-corrected-2026-08-25/phase32-replication/seed2803
```

## Repair and refresh

A failed seed is preserved and repaired in a fresh root under unchanged
criteria. A hard signature/measure mismatch is a scientific veto for the
replication aggregate, not permission to pool incompatible runs. Low ESS or
mode variation remains descriptive and refreshes Phase 33 controls.

