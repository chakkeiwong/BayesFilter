from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "docs/benchmarks/preflight_pp_ukf_true_hmc_validation_20260722.py"
SPEC = importlib.util.spec_from_file_location("pp_ukf_true_hmc_preflight", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
driver = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(driver)


def test_preflight_binds_each_candidate_to_its_own_controller(tmp_path):
    result = driver.build_preflight(output_root=tmp_path / "attempt")

    assert result["status"] == "blocked_before_sampling_missing_fresh_partition_and_budget"
    assert result["gpu_sampling_launched"] is False
    policies = result["sequential_controller_policies"]
    candidates = result["candidate_manifest"]["candidates"]
    assert len(policies) == len(candidates) == 10
    for candidate, binding in zip(candidates, policies):
        assert binding["candidate_id"] == candidate["candidate_id"]
        assert binding["num_leapfrog_steps"] == candidate["num_leapfrog_steps"]
        assert binding["step_size"] == candidate["step_size"]
        policy = binding["policy"]
        assert policy["chain_count"] == 4
        assert policy["num_leapfrog_steps"] == candidate["num_leapfrog_steps"]
        assert policy["step_size"] == candidate["step_size"]
        assert policy["warmup_min_results"] == 2000
        assert policy["retained_min_results"] == 1000
        assert policy["warmup_rhat_max"] == 1.05
        assert policy["retained_rhat_max"] == 1.01
        assert policy["jit_compile"] is True
    assert result["retained_ess_thresholds"]["status"] == "not_declared_before_launch"

