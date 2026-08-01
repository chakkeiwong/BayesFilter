from __future__ import annotations

import math
import hashlib
import json
from pathlib import Path

import pytest
import tensorflow as tf

from bayesfilter.highdim.cubature_genut_adapters import (
    diagonal_lgssm_candidate_adapter,
)
from bayesfilter.highdim.cubature_genut_candidate import (
    gaussian_genut_design,
    replicate_positive_genut,
)
from docs.benchmarks import run_genut_antithetic_lgssm_sv as campaign


def test_lgssm_adapter_shapes_and_recursive_score() -> None:
    adapter = diagonal_lgssm_candidate_adapter(
        observation_matrix=tf.eye(3, dtype=tf.float32)
    )
    theta = tf.constant([0.4, 0.3, 0.2, 0.5, 0.6], tf.float32)
    initial = tf.random.stateless_normal([12, 3], [1, 2])
    process = tf.random.stateless_normal([2, 12, 3], [3, 4])
    observations = tf.random.stateless_normal([2, 3], [5, 6])
    design = replicate_positive_genut(
        gaussian_genut_design(dim=3), num_particles=12
    )
    from bayesfilter.highdim.cubature_genut_filter import finite_value_score

    value, score, diagnostics = finite_value_score(
        adapter,
        theta,
        observations,
        initial,
        process,
        design,
        sinkhorn_steps=2,
    )
    assert value.shape == ()
    assert score.shape == (5,)
    assert diagnostics["score_increments"].shape == (2, 5)
    assert bool(tf.reduce_all(tf.math.is_finite(score)).numpy())


def test_model_dgps_replay_and_are_distinct() -> None:
    for model in campaign.MODELS:
        first = campaign._generate_dataset(model, 7001)  # noqa: SLF001
        replay = campaign._generate_dataset(model, 7001)  # noqa: SLF001
        changed = campaign._generate_dataset(model, 7002)  # noqa: SLF001
        tf.debugging.assert_equal(first["observations"], replay["observations"])
        assert not bool(
            tf.reduce_all(tf.equal(first["observations"], changed["observations"])).numpy()
        )
        assert first["observations"].shape[0] == campaign.HORIZON
        assert bool(tf.reduce_all(tf.math.is_finite(first["observations"])).numpy())


def test_sv_generator_obeys_declared_observation_equation() -> None:
    generated = campaign._generate_dataset("sv", 7101)  # noqa: SLF001
    theta = tf.constant(campaign.SV_THETA, tf.float32)
    beta = tf.exp(theta[1])
    reconstructed = (
        generated["states"]
        + 2.0 * tf.math.log(beta)
        + tf.math.log(tf.square(generated["observation_noise"]))
    )
    tf.debugging.assert_near(generated["observations"][:, 0], reconstructed)


def test_pair_estimators_are_equal_cost_and_correct() -> None:
    first = {"value": 2.0, "score": [1.0, 3.0]}
    second = {"value": 4.0, "score": [5.0, 7.0]}
    negative = {"value": 0.0, "score": [-1.0, 1.0]}
    estimators = campaign._pair_estimators(first, second, negative)  # noqa: SLF001
    assert estimators["standard"]["complete_run_count"] == 1
    assert estimators["independent_pair"]["complete_run_count"] == 2
    assert estimators["antithetic_pair"]["complete_run_count"] == 2
    assert estimators["independent_pair"]["value"] == 3.0
    assert estimators["antithetic_pair"]["score"] == [0.0, 2.0]


def test_outer_variance_summary_detects_reduction() -> None:
    datasets = []
    for dataset_index in range(8):
        independent = [
            [float(seed), 2.0 * seed, -0.5 * seed] for seed in range(16)
        ]
        antithetic = [
            [
                0.2 * seed + 0.01 * dataset_index,
                0.4 * seed,
                -0.1 * seed + 0.01 * dataset_index,
            ]
            for seed in range(16)
        ]
        datasets.append(
            campaign._dataset_statistics(  # noqa: SLF001
                "sv", dataset_index, independent, antithetic, independent
            )
        )
    summary = campaign._outer_summary("sv", datasets)  # noqa: SLF001
    for row in summary["variance_ratio_antithetic_over_independent_pair"]:
        assert row["geometric_ratio"] < 1.0
        assert row["familywise_interval"]["upper"] < 0.0
        assert row["coordinate_nomination"] is True


@pytest.mark.parametrize("model,count", [("lgssm", 1008), ("sv", 1002)])
def test_genut_design_has_exact_mean_and_population_covariance(
    model: str, count: int
) -> None:
    design = campaign._genut_design(model)  # noqa: SLF001
    assert design.shape[0] == count
    tf.debugging.assert_near(tf.reduce_mean(design, axis=0), tf.zeros([design.shape[1]]), atol=1e-7)
    covariance = tf.transpose(design) @ design / tf.cast(count, tf.float32)
    tf.debugging.assert_near(covariance, tf.eye(design.shape[1]), atol=2e-6)
    if model == "sv":
        fourth = float(tf.reduce_mean(tf.pow(design[:, 0], 4)).numpy())
        assert math.isclose(fourth, 3.0, abs_tol=2e-6)


def test_terminal_artifact_is_equal_cost_and_dgp_separated_when_present() -> None:
    root = Path(
        "docs/benchmarks/artifacts/"
        "genut_antithetic_lgssm_sv_20260722/attempt02"
    )
    result_path = root / "result.json"
    if not result_path.exists():
        pytest.skip("terminal antithetic campaign artifact is not present")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    assert manifest["result_sha256"] == hashlib.sha256(
        result_path.read_bytes()
    ).hexdigest()
    assert result["hard_valid"] is True
    assert result["decision"]["default_changed"] is False
    for model in campaign.MODELS:
        tuning = result["tuning"][model]["partitions"]
        assert set(tuning["calibration"]).isdisjoint(tuning["validation"])
        assert set(tuning["calibration"]).isdisjoint(tuning["claim"])
        assert set(tuning["validation"]).isdisjoint(tuning["claim"])
        raw = json.loads((root / f"raw_{model}.json").read_text(encoding="utf-8"))
        assert len(raw) == 8 * 16
        assert {row["dataset_seed"] for row in raw} == set(tuning["claim"])
        assert all(
            row["estimators"]["independent_pair"]["complete_run_count"]
            == row["estimators"]["antithetic_pair"]["complete_run_count"]
            == 2
            for row in raw
        )
