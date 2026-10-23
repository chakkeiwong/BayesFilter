"""Independent conditional-moment, paired-cloud and corruption diagnostics."""

import copy
import importlib.util
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from bayesfilter.score_study.conditional_cloud_design import (
    ARMS, FIXED_BETAS, PARTICLE_COUNTS, cloud_seed, data_seed, evaluation_cases, make_clouds)

path = Path(__file__).resolve().parents[2] / "scripts/analyze_ledh_conditional_clouds.py"
spec = importlib.util.spec_from_file_location("cloud_analysis", path)
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def rows():
    output = []
    for particles in PARTICLE_COUNTS:
        for regime, partition, dataset, replicate in evaluation_cases():
            prefix = str((regime, partition, dataset, replicate))
            for arm in ARMS:
                policy = arm.removeprefix("current_").removeprefix("future_")
                beta = FIXED_BETAS.get(policy, [.2, .3, .5])
                score = (1. if arm.startswith("future") else 2.) * 32. / particles
                output.append({"particles": particles, "regime": regime, "partition": partition,
                    "dataset": dataset, "replicate": replicate, "arm": arm,
                    "data_seed": data_seed(regime, partition, dataset),
                    "cloud_seed": cloud_seed(regime, partition, dataset, replicate, particles),
                    "data_sha256": str((regime, partition, dataset)), "prefix32_sha256": prefix,
                    "paired_input_sha256": prefix if particles == 32 else prefix + "64",
                    "beta": list(beta), "score": [score] * 4, "reference_score": [0.] * 4,
                    "squared_error": [score ** 2] * 4, "value_error": 0.,
                    "trace_min": {"valid": 1., "ess": 8.}, "trace_max": {"max_weight": .25}})
    return output


def test_conditional_moments_independent_identity_and_negative_bias_estimate():
    result = analysis.conditional_moments([[1.], [2.], [3.]])
    np.testing.assert_allclose(result["mean_error"], [2.])
    np.testing.assert_allclose(result["variance"], [1.])
    np.testing.assert_allclose(result["unbiased_squared_bias_estimate"], [11. / 3.])
    np.testing.assert_allclose(result["mean_squared_error"], [14. / 3.])
    centered = analysis.conditional_moments([[-1.], [1.]])
    assert centered["unbiased_squared_bias_estimate"] == [-1.]


def test_known_paired_guidance_and_particle_effects():
    result = analysis.analyze(rows(), resamples=20)
    assert result["rows"] == 832
    assert result["heuristic_dominance_verdict"] == "PASS_DESCRIPTIVE_SCREEN_ONLY"
    for record in result["contrasts"]:
        if record["contrast"] == "guidance_at_global":
            expected = -3. if record["particles"] == 32 else -.75
            assert record["mean_squared_error_difference"] == [expected] * 4
            assert record["paired_cloud_interval_95"] == [[expected, expected]] * 4
        if record["contrast"] == "particle_count_guidance_interaction_global":
            assert record["mean_squared_error_difference"] == [2.25] * 4
    assert not result["default_ready"]
    assert not result["population_ranking_supported"]


@pytest.mark.parametrize("damage", ["missing", "duplicate", "invalid", "beta", "input",
                                    "prefix", "reference", "data", "seed", "cloud_seed", "score", "nonfinite", "reused_cloud"])
def test_corruption_is_rejected(damage):
    data = copy.deepcopy(rows())
    if damage == "missing":
        data.pop()
    elif damage == "duplicate":
        data.append(data[0])
    elif damage == "invalid":
        data[0]["trace_min"]["valid"] = 0.
    elif damage == "beta":
        for row in data:
            if row["arm"] == "future_global":
                row["beta"] = [.2, .8, 0.]
    elif damage == "input":
        data[1]["paired_input_sha256"] = "different"
    elif damage == "prefix":
        for row in data:
            if row["particles"] == 64:
                row["prefix32_sha256"] += "wrong"
    elif damage == "reference":
        data[0]["reference_score"][0] = 1.
        data[0]["squared_error"][0] = 1.
    elif damage == "data":
        data[0]["data_sha256"] = "changed"
    elif damage == "seed":
        data[0]["data_seed"] += 1
    elif damage == "cloud_seed":
        data[0]["cloud_seed"] += 1
    elif damage == "score":
        data[0]["score"][0] = 9.
    elif damage == "reused_cloud":
        for row in data:
            if row["particles"] == 32 and row["partition"] == 2:
                row["paired_input_sha256"] = "one cloud"
                row["prefix32_sha256"] = "one cloud"
    else:
        data[0]["score"][0] = float("nan")
    with pytest.raises(ValueError):
        analysis.analyze(data, resamples=20)


def test_tensorflow_clouds_have_exact_shared_prefix_and_separate_calibration_seeds():
    import tensorflow as tf
    seed = cloud_seed(1, 2, 0, 3, 32)
    lower = make_clouds(tf, 32, seed)
    upper = make_clouds(tf, 64, seed)
    for first, second in zip(lower, (upper[0][:32], upper[1][:, :32], upper[2][:, :32])):
        np.testing.assert_array_equal(first.numpy(), second.numpy())
    assert cloud_seed(1, 0, 0, 0, 32) != cloud_seed(1, 0, 0, 0, 64)
    assert cloud_seed(1, 2, 0, 3, 32) == cloud_seed(1, 2, 0, 3, 64)


@pytest.mark.parametrize("damage", [None, "calibration", "certificate", "observation", "scope"])
def test_saved_scope_binds_calibration_and_observations(tmp_path, damage):
    data = [row for row in rows() if row["particles"] == 32]
    observations = [[0., 1.]] * 5
    digest = hashlib.sha256(json.dumps([observations], allow_nan=False).encode()).hexdigest()
    for row in data:
        row["data_sha256"] = digest
    datasets = [{"regime": regime, "partition": partition, "dataset": dataset,
                 "observations": observations, "data_sha256": digest, "reference_score": [0.] * 4}
                for regime in range(2) for partition in (1, 2) for dataset in range(2)]
    calibration = [{"regime": regime, "guide": guide, "particles": 32, "beta": [.2, .3, .5],
                    "converged": True, "objective": 1., "gap": 0.}
                   for regime in range(2) for guide in ("current", "exact")]
    manifest = {"status": "complete", "particles": 32, "row_count": 416}
    if damage == "calibration":
        calibration[0]["beta"] = [.2, .4, .4]
    elif damage == "certificate":
        calibration[0]["gap"] = .1
    elif damage == "observation":
        datasets[0] = {**datasets[0], "observations": [[1., 2.]] * 5}
    elif damage == "scope":
        calibration[0]["particles"] = 64
    for name, value in (("manifest.json", manifest), ("calibration.json", calibration)):
        (tmp_path / name).write_text(json.dumps(value))
    for name, values in (("rows.jsonl", data), ("datasets.jsonl", datasets)):
        (tmp_path / name).write_text("\n".join(json.dumps(value) for value in values))
    if damage:
        with pytest.raises(ValueError):
            analysis.load_scope(tmp_path, 32)
    else:
        _, loaded = analysis.load_scope(tmp_path, 32)
        assert len(loaded) == 416
