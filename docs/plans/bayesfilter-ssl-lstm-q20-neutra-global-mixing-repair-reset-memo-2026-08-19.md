# SSL-LSTM q=20 NeuTra global-mixing repair reset memo (2026-08-19)

## Current state

- The old requirement for a converged physical posterior archive before NeuTra
  training is superseded as circular.
- Mode-aware starts may test initialization forgetting, but mode-locked
  conditional chains must never be pooled.
- The new anti-pooling diagnostic and focused suite pass (`11 passed`).
- CPU/XLA replay canary `r2` passed all finite/status/pullback wiring checks.
- No valid GPU training or HMC candidate exists yet.

## Binding files

- Plan: `docs/plans/bayesfilter-ssl-lstm-q20-neutra-global-mixing-repair-plan-2026-08-19.md`
- Interim result: `docs/plans/bayesfilter-ssl-lstm-q20-neutra-global-mixing-repair-interim-result-2026-08-19.md`
- Claude continuation handoff: `docs/plans/bayesfilter-ssl-lstm-q20-neutra-global-mixing-claude-handoff-2026-08-19.md`
- Mixing diagnostic: `bayesfilter/inference/neutra_global_mixing.py`
- Tests: `tests/test_ssl_lstm_q20_neutra_global_mixing.py`
- CPU runner: `docs/benchmarks/run_ssl_lstm_q20_neutra_global_mixing_replay_canary_2026_08_19.py`
- GPU runner: `docs/benchmarks/run_ssl_lstm_q20_neutra_weighted_replay_gpu_canary_2026_08_19.py`
- CPU artifact: `docs/plans/artifacts/ssl-lstm-q20-neutra-global-mixing-2026-08-19/replay-canary-r2/`

## Next command

After trusted GPU execution is explicitly available, run one versioned GPU/XLA
canary with memory growth:

```bash
TF_FORCE_GPU_ALLOW_GROWTH=true \
/home/ubuntu/anaconda3/envs/tfgpu/bin/python \
docs/benchmarks/run_ssl_lstm_q20_neutra_weighted_replay_gpu_canary_2026_08_19.py \
  --device 1 --updates 20 --hidden-width 32 --stages 3 \
  --output-root docs/plans/artifacts/ssl-lstm-q20-neutra-global-mixing-2026-08-19/gpu-canary-r2
```

This is a mechanics/capacity canary.  If it is finite but globally mode-locked,
reject that transport candidate and run the predeclared capacity/proposal repair;
do not pool its chains.  If it passes the short coverage screen, proceed to the
two-capacity/two-seed target-specific training screen and shared sequential HMC.

## Do not reuse or conclude

- Do not use old seed-B NeuTra draws as posterior evidence.
- Do not use dense physical warm-up states for training or prediction.
- Do not call SMC replay rows an unweighted posterior archive.
- Do not infer global mixing from loss, replay ESS, acceptance, pooled occupancy,
  or the CPU adapter smoke.
- No NUTS; `L=1` remains forbidden.
