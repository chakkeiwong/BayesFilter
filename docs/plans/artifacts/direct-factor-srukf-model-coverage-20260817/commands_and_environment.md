# Commands and Environment

```text
CUDA_VISIBLE_DEVICES=-1 python scripts/run_direct_factor_srukf_model_coverage_20260817.py
CUDA_VISIBLE_DEVICES=-1 python -m pytest -q tests/test_block_qr_conditional_tf.py tests/test_rectangular_factor_tf.py tests/test_rectangular_srukf_tf.py tests/test_factor_srukf_tf.py tests/test_factor_srukf_model_parity.py tests/test_factor_srukf_route_guard.py tests/test_srukf_backend_policy.py
```

{
  "python": "3.11.15 | packaged by conda-forge | (main, Aug 11 2026, 10:26:29) [GCC 14.4.0]",
  "platform": "Linux-6.8.0-40-generic-x86_64-with-glibc2.35",
  "tensorflow": "2.19.1",
  "visible_gpus": [],
  "tf_force_gpu_allow_growth": "true",
  "artifact_root": "/home/ubuntu/workspace/BayesFilter/docs/plans/artifacts/direct-factor-srukf-model-coverage-20260817"
}
