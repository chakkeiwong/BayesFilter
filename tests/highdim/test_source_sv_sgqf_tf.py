from __future__ import annotations

from dataclasses import replace

import pytest
import tensorflow as tf

from bayesfilter.highdim.source_sv_sgqf_tf import (
    SOURCE_SV_OBSERVATION_SHA256,
    SOURCE_SV_STATE_SHA256,
    generate_source_order_sv_dataset_tf,
    make_source_order_sv_dataset,
)
from scripts.filtering_value_gradient_benchmark_generate_p8_datasets import _sv_dataset


def test_source_sv_fixture_is_transition_then_observe_and_deterministic() -> None:
    states, observations = generate_source_order_sv_dataset_tf()
    route = make_source_order_sv_dataset()

    assert states.shape == (1001, 1)
    assert observations.shape == (1000, 1)
    tf.debugging.assert_equal(states, route.states)
    tf.debugging.assert_equal(observations, route.observations)
    assert route.manifest["state_sha256"] == SOURCE_SV_STATE_SHA256
    assert route.manifest["observation_sha256"] == SOURCE_SV_OBSERVATION_SHA256
    assert route.manifest["time_order"] == (
        "x0_then_1000_transition_then_observe_steps_y1_y1000"
    )


def test_source_sv_fixture_rejects_old_initial_observation_first_data() -> None:
    route = make_source_order_sv_dataset()
    old = _sv_dataset(81101)

    with pytest.raises(ValueError, match="states require shape"):
        replace(route, states=tf.convert_to_tensor(old["states"], tf.float64))
    with pytest.raises(ValueError, match="observation identity rejected"):
        replace(
            route,
            observations=tf.convert_to_tensor(old["observations"], tf.float64),
        )


def test_source_sv_fixture_keeps_matlab_replay_as_nonclaim() -> None:
    route = make_source_order_sv_dataset()
    assert route.manifest["seed"] == 81101
    assert route.manifest["rng_classification"] == (
        "tensorflow_source_model_synthetic_replication"
    )
    assert "not MATLAB rng(1) bitwise replay" in route.manifest["nonclaims"]
