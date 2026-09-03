"""Student-t defensive floor tests (campaign plan C2; review CF5).

U-T-FLOOR-1: no-fire check (LGSSM with t-floor vs reference floor).
U-T-FLOOR-2: margin closed-form vs numerical quadrature at probe nus.
U-T-FLOOR-3: two-sided criterion well-posedness (largest nu under cap).
U-T-FLOOR-4: lane parity under the t-floor (eager vs XLA).
"""

import math

import numpy as np
import tensorflow as tf

from bayesfilter.highdim.squared_tt_engine_gaussian_tf import (
    run_value_filter_branch_axis_gaussian,
    student_t_margin,
    student_t_nu_criterion,
)
from bayesfilter.highdim.squared_tt_engine_gaussian_xla_tf import (
    run_value_filter_branch_axis_gaussian_xla,
)
from bayesfilter.highdim.squared_tt_engine_v0_tf import EngineConfig

# Reuse the oracle fixture construction
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))
import test_c2_gaussian_engine_oracle as T  # noqa: E402


def test_student_t_floor_no_fire_lgssm() -> None:
    """U-T-FLOOR-1: LGSSM oracle gaps unchanged with t-floor active."""

    adapter, ys, steps, model = T._lgssm_fixture(2, 12, 44)
    ih, ph = T._exact_hint_factories(model)
    config = EngineConfig(
        basis_degree=6, rank=3, row_count=2048, sweeps=16,
        ridge=1e-10, tau=1e-6, coordinate_half_width=3.0,
        seed=96001, row_design="sobol",
    )
    # Reference floor: oracle-clean. t-floor: the represented shape is
    # legitimately perturbed at O(tau) per step (the floor is the
    # program, not an approximation), so the no-fire envelope is the
    # predicted O(T * tau_min) = 1.2e-5 with margin — NOT bitwise
    # identity. A gap far above the envelope would mean the floor
    # machinery (mass identity, marginals, telescoping) is wrong.
    gates = {None: 1e-6, 5.0: 5e-5}
    for nu, gate in gates.items():
        value, diags = run_value_filter_branch_axis_gaussian(
            adapter, ys, config,
            predictive_moment_hint=ph, initial_moment_hint=ih,
            defensive_nu=nu,
        )
        gap = abs(
            float(value.numpy())
            - sum(math.log1p(d["tau_t"]) for d in diags)
            - sum(steps)
        )
        assert gap < gate, f"no-fire failed at nu={nu}: gap {gap:.3e}"


def test_student_t_margin_vs_dense_grid_max() -> None:
    """U-T-FLOOR-2: closed-form sup vs dense-grid maximum.

    The margin is a supremum, so the numerical validator is a grid
    maximum of the same objective, not a quadrature."""

    def margin_grid(nu: float, alpha: float) -> float:
        u = np.linspace(0.0, 200.0, 2_000_001)
        log_ratio_const = (
            math.lgamma((nu + 1) / 2)
            - math.lgamma(nu / 2)
            + 0.5 * math.log(2 / nu)
        )
        objective = (
            alpha * u**2 / 2
            - (
                log_ratio_const
                + 0.5 * u**2
                - ((nu + 1) / 2) * np.log1p(u**2 / nu)
            )
        )
        return float(objective.max())

    for nu in (3.0, 10.0, 50.0):
        for alpha in (0.1, 0.5, 0.9):
            closed = student_t_margin(nu, alpha)
            grid = margin_grid(nu, alpha)
            assert abs(closed - grid) < 1e-6, (
                f"margin nu={nu} alpha={alpha}: "
                f"closed={closed:.10f} grid={grid:.10f}"
            )


def test_student_t_nu_criterion_well_posed() -> None:
    """U-T-FLOOR-3: two-sided criterion returns the largest admissible nu."""

    alpha_max, cap = 0.8, 2.0
    nu_result = student_t_nu_criterion(alpha_max, cap)
    assert 1.5 < nu_result < 500, f"nu out of search range: {nu_result}"
    # At the returned nu, the margin must be <= cap
    assert student_t_margin(nu_result, alpha_max) <= cap + 1e-9
    # A slightly larger nu must exceed the cap (largest property)
    assert student_t_margin(nu_result + 0.1, alpha_max) > cap - 1e-9


def test_student_t_floor_lane_parity() -> None:
    """U-T-FLOOR-4: eager vs XLA parity with t-floor active."""

    adapter, ys, steps, model = T._lgssm_fixture(2, 4, 44)
    ih, ph = T._exact_hint_factories(model)
    config = EngineConfig(
        basis_degree=6, rank=3, row_count=2048, sweeps=16,
        ridge=1e-10, tau=1e-6, coordinate_half_width=3.0,
        seed=96002, row_design="sobol",
    )
    nu = 8.0
    ve, de = run_value_filter_branch_axis_gaussian(
        adapter, ys, config,
        predictive_moment_hint=ph, initial_moment_hint=ih, defensive_nu=nu,
    )
    vx, dx = run_value_filter_branch_axis_gaussian_xla(
        adapter, ys, config,
        predictive_moment_hint=ph, initial_moment_hint=ih, defensive_nu=nu,
    )
    step_gap = max(
        abs(a["log_increment"] - b["log_increment"])
        for a, b in zip(de, dx)
    )
    assert step_gap < 1e-12, f"lane parity nu={nu}: step_gap {step_gap:.3e}"
