# Phase 9 Corrected NeuTra Screen Subplan

Program: `docs/plans/bayesfilter-ssl-lstm-q20-particle-authority-master-program-2026-08-25.md`  
Status: `PASS_HARD_GATES_ROLE_LIMITED_WHITENING_UNRESOLVED`  
Budget cap: `3600 s` within the unchanged global `64800 s` cap  
Output root:
`docs/plans/artifacts/ssl-lstm-q20-particle-authority-master-2026-08-25/phase9`

## Objective

Rerun the target-specific batch-native NeuTra screen using the metadata-bound
N=300 bank that passed the independent finite-measure audit. This run uses the
pilot's explicit signed coordinate (`theta[:, 2]`) for stratification and the
index-aligned normalized weights. It tests downstream engineering and
transport validity only.

## Skeptical pre-execution audit

The input bank has passed all Phase 8 hash, ledger, terminal-weight, and
mode-axis checks. The runner's static tests pass after two repairs: weights are
gathered by partition indices and `MODE_AXIS=2`. Remaining risks are normalized
empirical-measure bias, finite mode coverage, short target-specific training,
and stochastic variation. Whitening, validation loss, and architecture choice
are explanatory; no one-seed ranking is permitted.

## Evidence contract

| Field | Choice |
|---|---|
| Comparator | compact and wide-low-learning-rate arms on the same metadata-bound bank and frozen 180/60/60 split |
| Hard criteria | trusted GPU memory growth before logical devices; XLA/batch-native updates; finite tensors; parity <= `1e-9`; valid transformed target/status on untouched audit rows; no HMC |
| Vetoes | input hash mismatch, wrong mode axis, weight/index mismatch, memory-policy failure, scalar training, non-finite/status/parity failure, audit leakage, incomplete artifact |
| Explanatory diagnostics | validation loss, latent moments/covariance, ESS, clipping, runtime |
| Nonclaims | no IID Gaussian whitening, posterior correctness, mode discovery, HMC convergence, predictive improvement, superiority, or default promotion |
| Artifact | unique GPU manifest, per-arm traces, decision/inference tables, repair note |

## Execution

```text
TF_FORCE_GPU_ALLOW_GROWTH=true PYTHONUNBUFFERED=1 \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python \
  docs/benchmarks/run_ssl_lstm_q20_particle_authority_neutra_screen_2026_08_25.py \
  --plan docs/plans/bayesfilter-ssl-lstm-q20-particle-authority-phase9-neutra-corrected-subplan-2026-08-25.md \
  --m0-root docs/plans/artifacts/ssl-lstm-q20-particle-authority-master-2026-08-25/phase6-attempt9-metadata-n300-seed2401 \
  --steps 20 --seed 20260825 6901 \
  --output-root docs/plans/artifacts/ssl-lstm-q20-particle-authority-master-2026-08-25/phase9-attempt1-bank2401
```

Run the focused static tests before and after the GPU command. A hard failure
triggers the companion repair note and a same-input rerun; it is not a whole
program blocker unless the master definition is met.
