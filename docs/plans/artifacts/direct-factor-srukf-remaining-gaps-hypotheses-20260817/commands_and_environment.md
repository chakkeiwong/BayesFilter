# Commands and environment

- KSC CPU probe: `MPLCONFIGDIR=/tmp/bayesfilter-mpl XDG_CACHE_HOME=/tmp/xdg-cache CUDA_VISIBLE_DEVICES=-1 python docs/benchmarks/run_neutra_ksc_gaussian_sum_ukf_admission_20260731.py --output-root docs/plans/artifacts/direct-factor-srukf-remaining-gaps-hypotheses-20260817/ksc-gaussian-sum`
- KSC GPU/XLA canary: `TF_FORCE_GPU_ALLOW_GROWTH=true CUDA_VISIBLE_DEVICES=3 python docs/benchmarks/run_neutra_ksc_gaussian_sum_ukf_admission_20260731.py --output-root docs/plans/artifacts/direct-factor-srukf-remaining-gaps-hypotheses-20260817/ksc-gaussian-sum-gpu3 --gpu-canary`
- Exact-SV SGQF admission ladder: `TF_FORCE_GPU_ALLOW_GROWTH=true CUDA_VISIBLE_DEVICES=3 python docs/benchmarks/run_neutra_svx_sgqf_repair_admission_20260731.py --output-root docs/plans/artifacts/direct-factor-srukf-remaining-gaps-hypotheses-20260817/svx-sgqf-gpu3`
- Current SVX-ZC GPU/XLA gate: `TF_FORCE_GPU_ALLOW_GROWTH=true CUDA_VISIBLE_DEVICES=3 python scripts/run_svx_zc_gpu_xla_gate_20260817.py --output-root docs/plans/artifacts/direct-factor-srukf-remaining-gaps-hypotheses-20260817/svx-zc-gpu3`
- Hypothesis recorder: `CUDA_VISIBLE_DEVICES=-1 python scripts/run_direct_factor_srukf_remaining_gaps_hypotheses_20260817.py`
- GPU selection policy for this campaign: prefer physical devices `3, 2, 1, 0`; GPU 3 was available and selected.
- Every GPU process set `TF_FORCE_GPU_ALLOW_GROWTH=true`, verified memory growth before device initialization, and used XLA JIT.

The top-level recorder is a CPU/reference aggregation step. Its GPU claims are inherited only from the versioned GPU 3 sub-artifacts above.
