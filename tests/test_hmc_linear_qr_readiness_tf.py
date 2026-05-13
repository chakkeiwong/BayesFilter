import os

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.testing import QRStaticLGSSMTarget, run_qr_static_lgssm_hmc_smoke


pytestmark = pytest.mark.skipif(
    os.environ.get("BAYESFILTER_RUN_HMC_READINESS") != "1",
    reason="HMC readiness diagnostics are opt-in extended CPU tests.",
)


@pytest.mark.extended
@pytest.mark.hmc
def test_qr_static_lgssm_target_value_gradient_and_curvature_are_finite() -> None:
    target = QRStaticLGSSMTarget.default()
    value, gradient = target.target_log_prob_and_grad(target.initial_parameters)
    curvature = target.curvature_diagnostics(target.initial_parameters)

    assert np.isfinite(value.numpy())
    assert np.all(np.isfinite(gradient.numpy()))
    assert float(curvature["value_residual"].numpy()) <= 1e-10
    assert float(curvature["score_residual"].numpy()) <= 1e-8
    assert float(curvature["hessian_residual"].numpy()) <= 1e-6
    assert float(curvature["hessian_symmetry_residual"].numpy()) <= 1e-10
    assert np.all(np.isfinite(curvature["negative_hessian_eigenvalues"].numpy()))


@pytest.mark.extended
@pytest.mark.hmc
def test_qr_static_lgssm_target_compiled_and_eager_parity() -> None:
    target = QRStaticLGSSMTarget.default()
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
def test_qr_static_lgssm_hmc_smoke_completes_with_finite_diagnostics() -> None:
    diagnostics = run_qr_static_lgssm_hmc_smoke(
        num_results=12,
        num_burnin_steps=6,
        step_size=0.03,
        num_leapfrog_steps=3,
    )

    assert int(diagnostics["nonfinite_sample_count"].numpy()) == 0
    assert int(diagnostics["finite_sample_count"].numpy()) == 12
    acceptance_rate = float(diagnostics["acceptance_rate"].numpy())
    assert 0.1 <= acceptance_rate <= 1.0
    assert bool(diagnostics["initial_gradient_finite"].numpy())
    assert np.all(np.isfinite(diagnostics["sample_mean"].numpy()))
    assert np.all(np.isfinite(diagnostics["sample_stddev"].numpy()))
    assert np.isfinite(float(diagnostics["min_target_log_prob"].numpy()))
    assert np.isfinite(float(diagnostics["max_target_log_prob"].numpy()))
    assert np.isfinite(float(diagnostics["max_abs_log_accept_ratio"].numpy()))
