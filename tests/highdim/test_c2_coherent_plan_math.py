"""CPU-only checks for identities added to the coherent C2 plan.

These tests exercise closed-form exposition identities and fixture geometry.
They are diagnostic/reference checks, not production numerical code.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from scipy.integrate import quad
from scipy.special import eval_hermitenorm, gamma


ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "docs/benchmarks/fixtures/c2_sv_n4_seed52_obs42_t20_frozen_v1.json"


def _phi(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def _rbf(x: float, center: float, width: float) -> float:
    return math.exp(-0.5 * ((x - center) / width) ** 2)


def _rbf_gram_closed(center: float, width: float, other: float, other_width: float) -> float:
    a = 1.0 + width ** -2 + other_width ** -2
    b = center / width ** 2 + other / other_width ** 2
    c = center ** 2 / width ** 2 + other ** 2 / other_width ** 2
    return a ** -0.5 * math.exp(-0.5 * (c - b * b / a))


def test_gaussian_rbf_gram_closed_form_matches_quadrature() -> None:
    center, width, other, other_width = 0.7, 0.8, -1.1, 1.35
    numerical = quad(
        lambda x: _rbf(x, center, width) * _rbf(x, other, other_width) * _phi(x),
        -math.inf,
        math.inf,
        epsabs=2e-12,
        epsrel=2e-12,
    )[0]
    assert math.isclose(numerical, _rbf_gram_closed(center, width, other, other_width), rel_tol=2e-11, abs_tol=2e-12)


def test_hermite_antiderivative_matches_derivative_and_lower_endpoint() -> None:
    for degree in range(4):
        value = quad(
            lambda x: eval_hermitenorm(degree, x) * _phi(x),
            -math.inf,
            0.37,
            epsabs=2e-12,
            epsrel=2e-12,
        )[0]
        expected = 0.5 * (1.0 + math.erf(0.37 / math.sqrt(2.0))) if degree == 0 else -_phi(0.37) * eval_hermitenorm(degree - 1, 0.37)
        assert math.isclose(value, expected, rel_tol=2e-10, abs_tol=2e-12)


def test_duplicate_constant_is_singular_and_student_boundary_is_explicit() -> None:
    assert math.isclose(1.0 * 1.0 - 1.0 * 1.0, 0.0, abs_tol=0.0)
    # A degree-d polynomial Gram requires moments through order 2d.
    assert 8.0 > 2.0 * 3.0
    assert not (6.0 > 2.0 * 3.0)
    # The Student tail exponent makes the order-k moment integrable iff k < nu.
    assert (2.0 * 3.0) < 8.0
    assert not ((2.0 * 3.0) < 6.0)


def test_c2_fixture_has_nonzero_observations_and_matches_envelope_stationary_point() -> None:
    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    observations = fixture["observations"]
    assert all(abs(float(value)) > 0.0 for row in observations for value in row)
    for row in observations:
        for raw in row:
            y = float(raw)
            v_star = y * y
            log_likelihood = -0.5 * math.log(2.0 * math.pi * v_star) - 0.5 * y * y / v_star
            expected = -0.5 * math.log(2.0 * math.pi) - math.log(abs(y)) - 0.5
            assert math.isclose(log_likelihood, expected, rel_tol=2e-12, abs_tol=2e-12)

