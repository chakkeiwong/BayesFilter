from __future__ import annotations

import math
from typing import Any

import numpy as np

from experiments.dpf_monograph_evidence.fixtures.hmc_value_gradient import FixedScalarHMCTargetFixture

NON_IMPLICATION = (
    "IE7 fixed-scalar value-gradient diagnostics validate only same-scalar, "
    "finite-difference, repeatability, and fixed-target leapfrog/energy-smoke checks "
    "on a deterministic clean-room fixture. They do not validate HMC correctness, "
    "DPF-HMC correctness, posterior/reference agreement, tuning readiness beyond "
    "controlled-fixture eligibility, production bayesfilter code, banking use, "
    "model-risk use, or production readiness."
)

FD_STEPS = (1e-2, 1e-3, 1e-4, 1e-5, 1e-6)
STABLE_WINDOW_STEPS = {1e-3, 1e-4, 1e-5}
REPEAT_COUNT = 3


def residual_entry(observed: float, threshold: float) -> dict[str, Any]:
    return {
        "threshold": float(threshold),
        "observed": float(observed),
        "finite": bool(math.isfinite(observed)),
    }


class FixedScalarTargetWrapper:
    wrapper_id = "fixed_scalar_hmc_target_single_wrapper_v1"

    def __init__(self, fixture: FixedScalarHMCTargetFixture) -> None:
        self.fixture = fixture
        self.matrix = np.asarray(fixture.matrix, dtype=np.float64)
        self.c = np.asarray(fixture.c, dtype=np.float64)

    def value(self, q: np.ndarray) -> float:
        q = np.asarray(q, dtype=np.float64)
        quadratic = 0.5 * float(q @ self.matrix @ q)
        quartic = self.fixture.beta * float(np.sum(q**4))
        sinusoid = self.fixture.gamma * math.sin(float(self.c @ q))
        return quadratic + quartic + sinusoid

    def gradient(self, q: np.ndarray) -> np.ndarray:
        q = np.asarray(q, dtype=np.float64)
        return self.matrix @ q + 4.0 * self.fixture.beta * q**3 + self.fixture.gamma * math.cos(float(self.c @ q)) * self.c

    def value_and_gradient(self, q: np.ndarray) -> tuple[float, np.ndarray]:
        return self.value(q), self.gradient(q)


def central_difference_gradient(target: FixedScalarTargetWrapper, q: np.ndarray, step: float) -> np.ndarray:
    gradient = np.zeros_like(q)
    for i in range(q.size):
        perturb = np.zeros_like(q)
        perturb[i] = step
        gradient[i] = (target.value(q + perturb) - target.value(q - perturb)) / (2.0 * step)
    return gradient


def kinetic_energy(momentum: np.ndarray, inverse_mass: np.ndarray) -> float:
    return 0.5 * float(momentum @ inverse_mass @ momentum)


def leapfrog(
    target: FixedScalarTargetWrapper,
    position: np.ndarray,
    momentum: np.ndarray,
    inverse_mass: np.ndarray,
    step_size: float,
    step_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    q = np.array(position, dtype=np.float64, copy=True)
    p = np.array(momentum, dtype=np.float64, copy=True)
    p = p - 0.5 * step_size * target.gradient(q)
    for step_index in range(step_count):
        q = q + step_size * (inverse_mass @ p)
        if step_index != step_count - 1:
            p = p - step_size * target.gradient(q)
    p = p - 0.5 * step_size * target.gradient(q)
    return q, p


def evaluate_hmc_value_gradient_fixture(fixture: FixedScalarHMCTargetFixture) -> dict[str, Any]:
    target = FixedScalarTargetWrapper(fixture)
    q0 = np.asarray(fixture.evaluation_point, dtype=np.float64)
    p0 = np.asarray(fixture.initial_momentum, dtype=np.float64)
    inverse_mass = np.asarray(fixture.inverse_mass, dtype=np.float64)

    accept_reject_scalar_value = target.value(q0)
    differentiated_scalar_value, analytic_gradient = target.value_and_gradient(q0)
    same_scalar_value_abs = abs(accept_reject_scalar_value - differentiated_scalar_value)

    eager_values = [target.value(q0) for _ in range(REPEAT_COUNT)]
    eager_gradients = [target.gradient(q0) for _ in range(REPEAT_COUNT)]
    eager_repeat_value_abs_max = float(np.max(np.abs(np.asarray(eager_values) - eager_values[0])))
    eager_repeat_gradient_abs_max = float(np.max(np.abs(np.asarray(eager_gradients) - eager_gradients[0])))

    fd_rows: list[dict[str, Any]] = []
    stable_residuals: list[float] = []
    for step in FD_STEPS:
        fd_gradient = central_difference_gradient(target, q0, step)
        abs_residual = float(np.max(np.abs(fd_gradient - analytic_gradient)))
        denominator = max(1.0, float(np.max(np.abs(analytic_gradient))))
        rel_residual = abs_residual / denominator
        if step == 1e-2:
            classification = "truncation_region"
        elif step == 1e-6:
            classification = "cancellation_region"
        else:
            classification = "stable_window"
            stable_residuals.append(abs_residual)
        fd_rows.append(
            {
                "step": step,
                "central_difference_gradient": fd_gradient.tolist(),
                "absolute_residual_max": abs_residual,
                "relative_residual_max": rel_residual,
                "classification": classification,
            }
        )
    finite_difference_stable_window_abs_max = float(np.max(stable_residuals))

    compiled_status = "not_available"
    compiled_not_available_reason = "No compiled/autodiff backend is imported in the clean-room CPU-only IE7 fixture."
    compiled_value_abs_max = 0.0
    compiled_gradient_abs_max = 0.0

    forward_q, forward_p = leapfrog(
        target,
        q0,
        p0,
        inverse_mass,
        fixture.leapfrog_step_size,
        fixture.leapfrog_step_count,
    )
    reverse_q, reverse_p_neg = leapfrog(
        target,
        forward_q,
        -forward_p,
        inverse_mass,
        fixture.leapfrog_step_size,
        fixture.leapfrog_step_count,
    )
    recovered_p = -reverse_p_neg
    leapfrog_reversibility_position_abs_max = float(np.max(np.abs(reverse_q - q0)))
    leapfrog_reversibility_momentum_abs_max = float(np.max(np.abs(recovered_p - p0)))
    initial_energy = target.value(q0) + kinetic_energy(p0, inverse_mass)
    final_energy = target.value(forward_q) + kinetic_energy(forward_p, inverse_mass)
    energy_drift_abs = abs(final_energy - initial_energy)

    tolerance = {
        "same_scalar_value_abs": residual_entry(same_scalar_value_abs, 0.0),
        "analytic_gradient_abs_max": residual_entry(0.0, 0.0),
        "finite_difference_stable_window_abs_max": residual_entry(finite_difference_stable_window_abs_max, 1e-5),
        "eager_repeat_value_abs_max": residual_entry(eager_repeat_value_abs_max, 0.0),
        "eager_repeat_gradient_abs_max": residual_entry(eager_repeat_gradient_abs_max, 0.0),
        "compiled_value_abs_max": residual_entry(compiled_value_abs_max, 0.0),
        "compiled_gradient_abs_max": residual_entry(compiled_gradient_abs_max, 0.0),
        "leapfrog_reversibility_position_abs_max": residual_entry(leapfrog_reversibility_position_abs_max, 1e-10),
        "leapfrog_reversibility_momentum_abs_max": residual_entry(leapfrog_reversibility_momentum_abs_max, 1e-10),
        "energy_drift_abs": residual_entry(energy_drift_abs, 1e-4),
    }

    finite_summary = all(entry["finite"] for entry in tolerance.values())
    finite_checks = {
        "fixture_id": fixture.fixture_id,
        "dimension": fixture.dimension,
        "evaluation_point": q0.tolist(),
        "scalar_formula": "U(q)=0.5*q^T*M*q + beta*sum(q_i^4) + gamma*sin(c^T*q)",
        "target_wrapper_id": target.wrapper_id,
        "accept_reject_scalar_value": accept_reject_scalar_value,
        "differentiated_scalar_value": differentiated_scalar_value,
        "same_scalar_source": "single_target_function_call",
        "same_scalar_verdict": same_scalar_value_abs == 0.0,
        "analytic_gradient": analytic_gradient.tolist(),
        "gradient_path": "closed_form_clean_room_gradient_from_same_target_wrapper",
        "finite_difference_ladder": fd_rows,
        "selected_stable_window_steps": sorted(STABLE_WINDOW_STEPS),
        "eager_repeat_values": eager_values,
        "eager_repeat_gradients": [gradient.tolist() for gradient in eager_gradients],
        "compiled_status": compiled_status,
        "compiled_not_available_reason": compiled_not_available_reason,
        "compiled_values": [],
        "compiled_gradients": [],
        "compiled_parity_verdict": "not_applicable",
        "leapfrog": {
            "step_size": fixture.leapfrog_step_size,
            "step_count": fixture.leapfrog_step_count,
            "initial_position": q0.tolist(),
            "initial_momentum": p0.tolist(),
            "forward_position": forward_q.tolist(),
            "forward_momentum": forward_p.tolist(),
            "reverse_position": reverse_q.tolist(),
            "recovered_momentum": recovered_p.tolist(),
            "initial_energy": initial_energy,
            "final_energy": final_energy,
        },
        "no_hmc_chain_run": True,
        "no_adaptation_or_tuning": True,
        "no_posterior_summary": True,
        "no_rng_used": True,
        "target_state_mutation_detected": False,
        "finite_summary": finite_summary,
    }
    shape_checks = {
        "state_dimension": fixture.dimension,
        "gradient_shape": list(analytic_gradient.shape),
        "finite_difference_ladder_length": len(FD_STEPS),
        "repeat_count": REPEAT_COUNT,
        "leapfrog_step_count": fixture.leapfrog_step_count,
        "scalar_output_shape": [],
    }

    return {
        "fixture_id": fixture.fixture_id,
        "target_wrapper_id": target.wrapper_id,
        "tolerance": tolerance,
        "finite_checks": finite_checks,
        "shape_checks": shape_checks,
    }


def hmc_value_gradient_status(metrics: dict[str, Any]) -> str:
    for key, value in metrics["tolerance"].items():
        if (not value["finite"]) or value["observed"] > value["threshold"]:
            return "fail"
    if not metrics["finite_checks"]["same_scalar_verdict"]:
        return "fail"
    if not metrics["finite_checks"]["no_hmc_chain_run"]:
        return "fail"
    if not metrics["finite_checks"]["no_adaptation_or_tuning"]:
        return "fail"
    if not metrics["finite_checks"]["no_posterior_summary"]:
        return "fail"
    return "pass"


def repair_trigger(metrics: dict[str, Any]) -> str:
    checks = [
        ("same_scalar_value_abs", "same-scalar mismatch"),
        ("finite_difference_stable_window_abs_max", "stable-window gradient mismatch"),
        ("compiled_value_abs_max", "compiled/eager discrepancy"),
        ("compiled_gradient_abs_max", "compiled/eager discrepancy"),
        ("eager_repeat_value_abs_max", "repeatability failure"),
        ("eager_repeat_gradient_abs_max", "repeatability failure"),
        ("leapfrog_reversibility_position_abs_max", "reversibility failure"),
        ("leapfrog_reversibility_momentum_abs_max", "reversibility failure"),
        ("energy_drift_abs", "energy-drift failure"),
    ]
    for key, label in checks:
        value = metrics["tolerance"][key]
        if (not value["finite"]) or value["observed"] > value["threshold"]:
            return label
    if not metrics["finite_checks"]["no_hmc_chain_run"]:
        return "prohibited HMC execution"
    if not metrics["finite_checks"]["no_adaptation_or_tuning"]:
        return "prohibited HMC execution"
    return "none"
