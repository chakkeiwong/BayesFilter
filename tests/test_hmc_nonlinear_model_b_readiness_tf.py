import os

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.testing import (
    ModelBNonlinearSVDTarget,
    run_model_b_nonlinear_svd_cut4_hmc_smoke,
)


pytestmark = pytest.mark.skipif(
    os.environ.get("BAYESFILTER_RUN_HMC_READINESS") != "1",
    reason="HMC readiness diagnostics are opt-in extended CPU tests.",
)


@pytest.mark.extended
@pytest.mark.hmc
def test_model_b_nonlinear_svd_cut4_target_value_gradient_and_branch_are_finite() -> None:
    target = ModelBNonlinearSVDTarget.default()
    value, gradient = target.target_log_prob_and_grad(target.initial_parameters)
    branch = target.branch_summary()

    assert np.isfinite(value.numpy())
    assert np.all(np.isfinite(gradient.numpy()))
    assert branch["ok_count"] == branch["total_count"]
    assert branch["active_floor_count"] == 0
    assert branch["weak_spectral_gap_count"] == 0
    assert branch["nonfinite_count"] == 0
    assert branch["failure_labels"] == ()


@pytest.mark.extended
@pytest.mark.hmc
def test_model_b_nonlinear_svd_cut4_compiled_and_eager_parity() -> None:
    target = ModelBNonlinearSVDTarget.default()
    params = target.initial_parameters
    eager_value, eager_gradient = target.target_log_prob_and_grad(params)

    @tf.function(reduce_retracing=True)
    def compiled(values: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        return target.target_log_prob_and_grad(values)

    compiled_value, compiled_gradient = compiled(params)

    np.testing.assert_allclose(compiled_value.numpy(), eager_value.numpy(), atol=1e-10)
    np.testing.assert_allclose(
        compiled_gradient.numpy(),
        eager_gradient.numpy(),
        rtol=1e-8,
        atol=1e-9,
    )


@pytest.mark.extended
@pytest.mark.hmc
def test_model_b_nonlinear_svd_cut4_hmc_smoke_completes_with_finite_diagnostics() -> None:
    diagnostics = run_model_b_nonlinear_svd_cut4_hmc_smoke(
        num_results=8,
        num_burnin_steps=4,
        step_size=0.005,
        num_leapfrog_steps=2,
    )

    assert int(diagnostics["nonfinite_sample_count"].numpy()) == 0
    assert int(diagnostics["finite_sample_count"].numpy()) == 8
    acceptance_rate = float(diagnostics["acceptance_rate"].numpy())
    assert 0.1 <= acceptance_rate <= 1.0
    assert bool(diagnostics["initial_gradient_finite"].numpy())
    assert int(diagnostics["branch_ok_count"].numpy()) == int(
        diagnostics["branch_total_count"].numpy()
    )
    assert int(diagnostics["branch_active_floor_count"].numpy()) == 0
    assert int(diagnostics["branch_weak_spectral_gap_count"].numpy()) == 0
    assert int(diagnostics["branch_nonfinite_count"].numpy()) == 0
    assert diagnostics["branch_failure_labels"] == ()
    assert np.all(np.isfinite(diagnostics["sample_mean"].numpy()))
    assert np.all(np.isfinite(diagnostics["sample_stddev"].numpy()))
    assert np.isfinite(float(diagnostics["min_target_log_prob"].numpy()))
    assert np.isfinite(float(diagnostics["max_target_log_prob"].numpy()))
    assert np.isfinite(float(diagnostics["max_abs_log_accept_ratio"].numpy()))
