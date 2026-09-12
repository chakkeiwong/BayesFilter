from __future__ import annotations

import inspect

import bayesfilter.inference.hmc_kernel_tuning as hmc_kernel_tuning


def test_fixed_l_and_verification_builders_enable_traced_failure_capture() -> None:
    expected = {
        "_frozen_step_trajectory_screen_config": 'failure_capture_role="sampling"',
        "_windowed_stage_diagnostic_run_config": 'failure_capture_role="sampling"',
        "_run_phase7_injected_final_verification": 'failure_capture_role="verification"',
    }
    for name, role in expected.items():
        source = inspect.getsource(getattr(hmc_kernel_tuning, name))
        assert "capture_first_failure=config.chain_execution_mode == \"tf_function\" and not config.use_xla" in source
        assert role in source
