# Recovery checkpoint

The owner reported a machine crash during the compiled Kalman/UKF repair.
All source edits, the exact-loop lint exception inventory, and the `cpu-v1`,
`cpu-v2`, `cpu-v3`, and `gpu-v1` qualification artifacts survived. The baseline
remains Git `21828174`. No interrupted test/qualification process remained.
Temporary `/tmp` test logs did not survive and are not durable evidence.

The preceding test summaries remain conversation provenance only. Final focused
validation will write logs and JUnit receipts here. This is continuation of the
existing implementation/engineering plan, within its bounded compute budget;
no tuning, training, retained sampling, or scientific promotion is performed.

The surviving source includes the corrected complete-XLA autodiff diagnostic
closure in `testing/tf_hmc_readiness.py`. Its last reported failure was a
TensorFlow method-descriptor cache hashing a tensor-containing dataclass; the
closure avoids that descriptor. Runtime analytical scores remain analytical.

GPU devices 2 and 3 were idle at recovery inspection. Only a final bounded
qualification on one idle device is planned, with verified memory growth.

Validation environment: `/home/ubuntu/miniforge3/envs/tf-gpu/bin/python`,
TensorFlow/TFP from that existing environment, `TF_FORCE_GPU_ALLOW_GROWTH=true`,
`CUDA_VISIBLE_DEVICES=''` for CPU engineering checks,
`TF_NUM_INTRAOP_THREADS=2`, `TF_NUM_INTEROP_THREADS=1`,
`OPENBLAS_NUM_THREADS=1`. Every command has an explicit timeout.
