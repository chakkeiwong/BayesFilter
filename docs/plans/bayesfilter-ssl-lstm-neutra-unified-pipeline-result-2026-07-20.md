# SSL-LSTM NeuTra/HMC Unified Pipeline Result

Date: 2026-07-20  
Plan: `docs/plans/bayesfilter-ssl-lstm-neutra-unified-pipeline-plan-2026-07-20.md`  
Decision: `WRAPPER_IMPLEMENTED_NO_MATERIAL_RUN`

## Implementation

Added `docs/benchmarks/run_ssl_lstm_neutra_hmc_pipeline_2026_07_20.py` as a
single-command orchestration layer. It launches, in order:

1. two-stream NeuTra training, including `--batch-size` and adaptive stopping;
2. transformed-target HMC preflight/tuning;
3. retained four-chain HMC with sequential checkpoints.

The wrapper does not merge numerical code or bypass stage gates. It requires two
`ADMITTED` training results before HMC tuning and `KERNELS_FROZEN` before
retained HMC. Each child has a dedicated log and the wrapper writes
`pipeline-summary.json`.

## Verification

| Check | Result |
| --- | --- |
| Contract smoke (`q=20`, batch 100) | Passed |
| Focused wrapper tests | `4 passed` |
| Python compilation | Passed |
| `git diff --check` | Passed |
| Material GPU/HMC execution | Not launched |

Example command:

```text
TF_FORCE_GPU_ALLOW_GROWTH=true \
python docs/benchmarks/run_ssl_lstm_neutra_hmc_pipeline_2026_07_20.py \
  --q 20 --batch-size 100 \
  --params-json <q20-params.json> \
  --output-root <fresh-run-root> \
  --training-cap-seconds <cap> \
  --hmc-tuning-cap-seconds <cap> \
  --retained-hmc-cap-seconds <cap> \
  --authorize-material-run
```

This result establishes repeatable orchestration only. It does not establish
training admission, HMC convergence, posterior correctness, or predictive
equivalence until a material run is separately authorized and passes its child
gates.
