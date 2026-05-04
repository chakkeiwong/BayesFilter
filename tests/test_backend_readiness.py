import numpy as np

from bayesfilter import audit_factor_backend, certify_spectral_derivative_region


def test_factor_backend_audit_separates_value_derivative_and_hmc_status():
    value_only = audit_factor_backend(
        "covariance_kalman",
        value_exact=True,
        derivative_checked=False,
        compiled_supported=False,
    )
    hmc_candidate = audit_factor_backend(
        "qr_differentiated_kalman",
        value_exact=True,
        derivative_checked=True,
        compiled_supported=True,
        min_spectral_gap=None,
    )
    approximation = audit_factor_backend(
        "enlarged_full_state_sigma_point",
        value_exact=False,
        approximation_label="full_state_mixed_approximation",
    )

    assert value_only.value_status == "exact_value"
    assert value_only.derivative_status == "not_certified"
    assert value_only.hmc_status == "blocked"
    assert hmc_candidate.hmc_status == "target_candidate"
    assert approximation.value_status == "approximation_only"
    assert approximation.approximation_label == "full_state_mixed_approximation"


def test_factor_backend_audit_blocks_unlabeled_approximation():
    result = audit_factor_backend(
        "unknown_square_root_fallback",
        value_exact=False,
    )

    assert result.hmc_status == "blocked"
    assert "requires an approximation label" in result.blockers[0]


def test_spectral_derivative_certification_requires_gap_and_numerical_checks():
    certified = certify_spectral_derivative_region(
        np.array([0.1, 0.5, 1.2]),
        finite_difference_checked=True,
        jvp_vjp_checked=True,
    )
    blocked = certify_spectral_derivative_region(
        np.array([1.0, 1.0 + 1e-9]),
        gap_tolerance=1e-6,
        finite_difference_checked=True,
        jvp_vjp_checked=False,
    )

    assert certified.derivative_certified is True
    assert certified.hmc_eligible is True
    assert blocked.derivative_certified is False
    assert blocked.hmc_eligible is False
    assert blocked.warning_label == "spectral_gap_too_small"
    assert any("minimum spectral gap" in blocker for blocker in blocked.blockers)
    assert any("JVP/VJP" in blocker for blocker in blocked.blockers)
