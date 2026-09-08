"""Focused CPU checks for the smooth Student-defensive UKF/APF route."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest
import tensorflow as tf

from bayesfilter.highdim.c2_mixture_ukf_apf_c2_adapter import (
    DEFENSIVE_MIXTURE_ROUTE_ID,
    compile_c2_per_ancestor_ukf_apf_defensive_mixture,
)
from bayesfilter.highdim.c2_sv_frozen_proposal_apf_tf import (
    C2StochasticVolatilityFrozenAPFModel,
)
from bayesfilter.highdim.zhao_cui_frozen_proposal_apf_tf import (
    prepare_frozen_proposal_apf_program,
)


DTYPE = tf.float64
ROOT = Path(__file__).resolve().parents[2]
FIXTURE = ROOT / "docs/benchmarks/fixtures/c2_sv_n4_seed52_obs42_t20_frozen_v1.json"


def _fixture(horizon: int = 4):
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    theta = tf.constant([payload["gamma"], math.log(payload["beta"])], DTYPE)
    transition = tf.constant(payload["transition_matrix"], DTYPE)
    coupling = transition - theta[0] * tf.eye(4, dtype=DTYPE)
    model = C2StochasticVolatilityFrozenAPFModel(
        coupling_matrix=coupling, sigma=float(payload["sigma"])
    )
    observations = tf.constant(payload["observations"][:horizon], DTYPE)
    return model, theta, observations


def _compile(component_count: int, *, jit_compile: bool = False):
    model, theta, observations = _fixture()
    return model, theta, compile_c2_per_ancestor_ukf_apf_defensive_mixture(
        model=model,
        observations=observations,
        theta_reference=theta,
        particle_count=20,
        seed=9510 + component_count,
        local_component_count=component_count,
        offset=0.35,
        nu=8.0,
        epsilon_min=0.05,
        epsilon_max=0.20,
        jit_compile=jit_compile,
    )


@pytest.mark.parametrize("component_count", [1, 2, 4])
def test_defensive_route_has_complete_density_and_smooth_gate(component_count: int):
    model, theta, compilation = _compile(component_count)
    assert compilation.manifest["route_id"] == DEFENSIVE_MIXTURE_ROUTE_ID
    assert compilation.manifest["local_component_count"] == component_count
    assert compilation.manifest["complete_mixture_density"] is True
    assert compilation.manifest["full_support"] is True
    assert len(compilation.proposal_diagnostics) == 3
    for row in compilation.proposal_diagnostics:
        assert bool(row["proposal_finite"].numpy())
        assert bool(row["exact_prefix_finite"].numpy())
        assert float(tf.reduce_min(row["component_minimum_eigenvalue"]).numpy()) > 0.0
        assert float(row["local_moment_recomposition_max_abs"].numpy()) <= 2.0e-12
        assert float(row["proposal_density_recomposition_max_abs"].numpy()) <= 2.0e-12
        assert float(row["label_permutation_max_abs"].numpy()) <= 2.0e-12
        epsilon = row["epsilon"]
        assert float(tf.reduce_min(epsilon).numpy()) > 0.05
        assert float(tf.reduce_max(epsilon).numpy()) < 0.20


@pytest.mark.parametrize("component_count", [1, 2, 4])
def test_defensive_frozen_score_matches_central_difference(component_count: int):
    model, theta, compilation = _compile(component_count)
    program = prepare_frozen_proposal_apf_program(model, compilation.branch)
    result = program.evaluate(theta)
    finite_difference = []
    for index in range(2):
        direction = tf.one_hot(index, 2, dtype=DTYPE)
        step = 1.0e-5
        finite_difference.append(
            (
                program.evaluate(theta + step * direction)["log_likelihood"]
                - program.evaluate(theta - step * direction)["log_likelihood"]
            )
            / (2.0 * step)
        )
    tf.debugging.assert_near(
        result["score"], tf.stack(finite_difference), atol=2.0e-7, rtol=2.0e-7
    )
    assert bool(result["finite"].numpy())


def test_defensive_gate_changes_with_observation():
    model, theta, observations = _fixture(horizon=3)
    first = compile_c2_per_ancestor_ukf_apf_defensive_mixture(
        model=model,
        observations=observations,
        theta_reference=theta,
        particle_count=20,
        seed=9540,
        local_component_count=1,
        jit_compile=False,
    )
    shifted = tf.tensor_scatter_nd_add(
        observations,
        tf.constant([[1, 0]], tf.int32),
        tf.constant([0.7], DTYPE),
    )
    second = compile_c2_per_ancestor_ukf_apf_defensive_mixture(
        model=model,
        observations=shifted,
        theta_reference=theta,
        particle_count=20,
        seed=9540,
        local_component_count=1,
        jit_compile=False,
    )
    first_row = first.proposal_diagnostics[0]
    second_row = second.proposal_diagnostics[0]
    assert float(
        tf.reduce_max(tf.abs(first_row["epsilon"] - second_row["epsilon"])).numpy()
    ) > 1.0e-5
    assert float(first_row["epsilon_spread"].numpy()) > 1.0e-8


def test_defensive_xla_and_eager_frozen_branch_match():
    _, _, xla = _compile(1, jit_compile=True)
    _, _, eager = _compile(1, jit_compile=False)
    tf.debugging.assert_near(xla.branch.states, eager.branch.states, atol=2.0e-10)
    tf.debugging.assert_equal(xla.branch.ancestors, eager.branch.ancestors)
    tf.debugging.assert_near(
        xla.branch.transition_log_proposal_density,
        eager.branch.transition_log_proposal_density,
        atol=2.0e-10,
    )
