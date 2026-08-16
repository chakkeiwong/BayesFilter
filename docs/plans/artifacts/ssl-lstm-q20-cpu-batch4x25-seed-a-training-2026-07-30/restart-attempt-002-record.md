# q=20 CPU Seed-A Resume Attempt 002

Date: 2026-07-30
Status: `RESUME_LAUNCH_RECORDED`

## Authorization And Budget

The owner authorized exactly `20,000 s` additional wall time for the existing
seed-A campaign. This changes only the bounded compute cap; the target, method,
data, seeds, architecture, topology, criteria, vetoes, and nonclaims remain
unchanged.

- Attempt-001 manifest cumulative wall: `31349.25759465 s`
- Charged prior cumulative wall, rounded upward: `31350 s`
- Additional authorized outer time: `20000 s`
- Active cumulative launcher cap: `51500 s`
- Accounting slack: `51500 - 31350 - 20000 = 150 s`

The outer timeout cannot consume more than the newly authorized `20,000 s`.
The 150-second difference is conservative rounding/finalization slack and is
not additional execution authority.

## Validated Checkpoint

- Path: `seed-a/checkpoint-1750.json`
- File SHA-256: `52ad277f01f98003c9eb12452aa6b21e2e78e102b9b4797ca741aa144a32014c`
- Joint checkpoint hash: `dfd65eb80c17e8b3f69fc692c80c428ec692677be2c2ddcd5821eb75c6d3af0f`
- Program step: `1750`
- Trainer step: `1500`
- Controller status: `running`
- Controller learning rate: `0.0002`
- Best step: `1500`

The trainer/program step difference is correct: the step-1750 controller
action restored the best step-1500 trainer state and reduced the learning rate.

## Source And Checks

- Repository `HEAD`: `882679796e8ee684b6b020b7cd84e3cfc1d92d58`
- Active plan SHA-256: `1194553c9473029a5e88dd7ed1a742487980afd85024c126d6d1b524070606f4`
- Launcher SHA-256: `3de4625a00dd50b709f5542e4d8096e87e9b1802346f1ed351934239a42c107f`
- Pool SHA-256: `cb279059cd7c84e709070585dee1e1211c36808efac4187bc76aded062fdfbe7`
- Target SHA-256: `4bca69298ae0d8065b4159da073bc6d2ff70d0a64ae3d841158a36bbe1184358`
- Focused launcher/control checks: `32 passed`
- Actual resume-loader reconstruction: passed

## Exact Command

```bash
timeout 20000 taskset -c 0-49 python \
  docs/benchmarks/run_ssl_lstm_q20_strict_cpu_batch_native_training_2026_07_22.py \
  --stream seed-a \
  --cpu-processes 25 \
  --batch-per-process 4 \
  --output-root \
  docs/plans/artifacts/ssl-lstm-q20-cpu-batch4x25-seed-a-training-2026-07-30 \
  --cap-seconds 51500 \
  --resume-checkpoint \
  docs/plans/artifacts/ssl-lstm-q20-cpu-batch4x25-seed-a-training-2026-07-30/seed-a/checkpoint-1750.json \
  --prior-wall-seconds 31350
```

## Stop Contract

Completion still requires the declared plateau stop or 2,000 program updates,
then finite final support and heldout audit checks. Any nonfinite value,
resource violation, corrupted artifact, configuration mismatch, or exhausted
cumulative budget vetoes continuation. This CPU-only stream cannot authorize
HMC, seed B, transport promotion, posterior claims, or a change to the GPU
NeuTra default.
