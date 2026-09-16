# Whole-procedure review validation commands

Baseline: `9329cadf3296211ccfa9e2539235225e041ad36a`.
Working directory: `/home/ubuntu/python/BayesFilter`.
Environment: `/home/ubuntu/anaconda3/envs/tfgpu/bin/python`.
Plan/result: `docs/plans/bayesfilter-hmc-whole-procedure-gap-review-2026-09-14.md`.
No observed data; controller/fault fixtures and one standard Gaussian mechanics
counterexample. No sampler ranking or posterior claim.

The following commands were executed with GPUs deliberately hidden for all
TensorFlow work. The diagnostic scripts also set these environment variables
before importing TensorFlow. No GPU readiness check or workload was run.

```bash
CUDA_VISIBLE_DEVICES=-1 TF_FORCE_GPU_ALLOW_GROWTH=true BAYESFILTER_TEST_DEVICE_SCOPE=cpu /home/ubuntu/anaconda3/envs/tfgpu/bin/python docs/plans/artifacts/hmc-whole-procedure-gap-review-2026-09-14/reproduce_gaps.py

CUDA_VISIBLE_DEVICES=-1 TF_FORCE_GPU_ALLOW_GROWTH=true BAYESFILTER_TEST_DEVICE_SCOPE=cpu /home/ubuntu/anaconda3/envs/tfgpu/bin/python docs/plans/artifacts/hmc-whole-procedure-gap-review-2026-09-14/gaussian_oversized_step.py

CUDA_VISIBLE_DEVICES=-1 TF_FORCE_GPU_ALLOW_GROWTH=true BAYESFILTER_TEST_DEVICE_SCOPE=cpu /home/ubuntu/anaconda3/envs/tfgpu/bin/python -m pytest -q --disable-warnings tests/test_hmc_candidate_set_tuning.py tests/test_hmc_candidate_set_artifacts.py tests/test_hmc_candidate_set_adapters.py tests/test_hmc_tuning_documentation_contract.py tests/test_fixed_transport_hmc_tuning.py::test_modern_verification_folded_rhat_vetoes_in_band_acceptance tests/test_fixed_transport_hmc_tuning.py::test_diagnostic_modern_rhat_is_reported_without_vetoing_healthy_mechanics tests/test_fixed_transport_hmc_tuning.py::test_diagnostic_modern_rhat_computation_error_cannot_veto_mechanics --junitxml=docs/plans/artifacts/hmc-whole-procedure-gap-review-2026-09-14/focused-regressions.xml

CUDA_VISIBLE_DEVICES=-1 TF_FORCE_GPU_ALLOW_GROWTH=true BAYESFILTER_TEST_DEVICE_SCOPE=cpu /home/ubuntu/anaconda3/envs/tfgpu/bin/python -m pytest -q --disable-warnings tests/test_hmc_kernel_tuning_outer_loop.py::test_phase4_direct_queue_plan_precomputes_complete_unique_seed_map tests/test_hmc_kernel_tuning_outer_loop.py::test_phase4_direct_queue_continues_candidate_local_then_admits_second tests/test_hmc_kernel_tuning_public_api.py::test_public_tuner_accepts_high_rhat_tuning_diagnostic --junitxml=docs/plans/artifacts/hmc-whole-procedure-gap-review-2026-09-14/legacy-routing-regressions.xml

/home/ubuntu/anaconda3/envs/tfgpu/bin/python scripts/inventory_hmc_tuning_routes.py --check > docs/plans/artifacts/hmc-whole-procedure-gap-review-2026-09-14/route-inventory.json
```

| Check | Result | Observed wall time |
| --- | --- | --- |
| Initial deterministic diagnostics | 12 expected behaviors reproduced | 2.882 seconds |
| Expanded deterministic diagnostics | 13 expected behaviors reproduced, including loss of shared-invalidity scope | 3.015 seconds |
| Actual Gaussian public tuning call | Smaller-epsilon repair suppressed; one measurement, zero children | 4.783 seconds |
| Focused regressions | 53 passed, zero failures/errors/skips | 5.25 seconds |
| Legacy queue and ordinary R-hat regressions | 3 passed, zero failures/errors/skips | 3.68 seconds |
| Function-name inventory | 20 discovered/20 registered; no stale or unclassified names | 0.84 seconds command wall time |

Times for scripts include their Python/framework initialization; pytest times
are its reported suite durations. These are engineering run records, not
performance comparisons. The Gaussian seed and execution configuration are
saved in `gaussian-oversized-step-r1/result.md`, alongside actual tensor evidence.
The fault tests intentionally use fabricated traces and injected exceptions;
they do not report real memory exhaustion or real-target failure rates.

The Gaussian script uses a fresh output directory and will refuse a rerun into
its existing directory. A future execution must choose a new versioned suffix.
The synthetic overwrite fixture deliberately tests replacement within its own
temporary test output; it never targets a pre-existing campaign artifact.
