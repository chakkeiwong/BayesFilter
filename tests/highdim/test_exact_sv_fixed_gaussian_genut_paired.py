from __future__ import annotations

import math
import hashlib
import json
from pathlib import Path

import pytest
import tensorflow as tf

from docs.benchmarks import run_exact_sv_fixed_gaussian_genut_paired as paired


@pytest.mark.parametrize("particle_count", [6, 12, 1002, 1998])
def test_paired_designs_have_exact_declared_scalar_moments(particle_count: int) -> None:
    cubature, genut = paired.paired_rank_designs(particle_count, seed=19)
    cubature_moments = paired.design_moments(cubature)
    genut_moments = paired.design_moments(genut)
    assert cubature.shape == genut.shape == (particle_count, 1)
    assert cubature_moments["moment_1"] == pytest.approx(0.0, abs=1e-7)
    assert cubature_moments["moment_2"] == pytest.approx(1.0, abs=1e-7)
    assert cubature_moments["moment_3"] == pytest.approx(0.0, abs=1e-7)
    assert cubature_moments["moment_4"] == pytest.approx(1.0, abs=1e-7)
    assert genut_moments["moment_1"] == pytest.approx(0.0, abs=1e-7)
    assert genut_moments["moment_2"] == pytest.approx(1.0, abs=1e-6)
    assert genut_moments["moment_3"] == pytest.approx(0.0, abs=1e-7)
    assert genut_moments["moment_4"] == pytest.approx(3.0, abs=2e-6)
    assert int(tf.math.count_nonzero(tf.equal(genut, 0.0)).numpy()) == 2 * particle_count // 3


def test_paired_design_is_deterministic_and_seed_changes_row_coupling() -> None:
    first = paired.paired_rank_designs(12, seed=7)
    replay = paired.paired_rank_designs(12, seed=7)
    changed = paired.paired_rank_designs(12, seed=8)
    tf.debugging.assert_equal(first[0], replay[0])
    tf.debugging.assert_equal(first[1], replay[1])
    assert not bool(tf.reduce_all(tf.equal(first[0], changed[0])).numpy())
    assert not bool(tf.reduce_all(tf.equal(first[1], changed[1])).numpy())


@pytest.mark.parametrize("particle_count", [1, 2, 4, 7, 1000, 2000])
def test_paired_design_requires_exact_six_divisibility(particle_count: int) -> None:
    with pytest.raises(ValueError, match="divisible by six"):
        paired.paired_rank_designs(particle_count, seed=1)


def test_fresh_dgp_observations_are_finite_and_replayable() -> None:
    theta = tf.constant([0.25, -0.15], tf.float32)
    first, first_states = paired._fresh_dgp_observations(theta)  # noqa: SLF001
    replay, replay_states = paired._fresh_dgp_observations(theta)  # noqa: SLF001
    assert first.shape == first_states.shape == (paired.HORIZON, 1)
    tf.debugging.assert_equal(first, replay)
    tf.debugging.assert_equal(first_states, replay_states)
    assert bool(tf.reduce_all(tf.math.is_finite(first)).numpy())
    assert math.isfinite(float(tf.reduce_mean(first).numpy()))


def test_paired_bootstrap_sign_and_labels() -> None:
    cubature = [1.0 + 0.01 * index for index in range(16)]
    genut = [0.25 + 0.002 * index for index in range(16)]
    result = paired._paired_bootstrap(  # noqa: SLF001
        cubature,
        genut,
        statistic="absolute_mean_error",
        seed_offset=0,
    )
    assert result["statistic"] == "abs_mean_genut_error_minus_abs_mean_cubature_error"
    assert result["observed"] < 0.0
    assert result["upper"] < 0.0
    assert result["genut_reduction_supported"] is True


def test_terminal_artifact_integrity_when_present() -> None:
    root = Path(
        "docs/benchmarks/artifacts/"
        "exact_sv_fixed_gaussian_genut_paired_20260721/attempt02"
    )
    result_path = root / "result.json"
    if not result_path.exists():
        pytest.skip("terminal campaign artifact is not present")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    manifest = json.loads((root / "run_manifest.json").read_text(encoding="utf-8"))
    correction = json.loads(
        (root.parent / "correction_20260722.json").read_text(encoding="utf-8")
    )
    assert manifest["result_sha256"] == hashlib.sha256(result_path.read_bytes()).hexdigest()
    # The old hard_valid field certifies engineering execution only after the
    # 2026-07-22 non-DGP correction; it is not scientific admission.
    assert result["hard_valid"] is True
    assert result["engineering_valid"] is True
    assert correction["scientifically_eligible_dataset_keys"] == ["fresh_dgp"]
    assert correction["revoked_source_field"] == "mechanism_support"
    assert result["manifest"]["claim_seeds"] == list(paired.CLAIM_SEEDS)
    assert set(result["summaries"]) == {
        paired._scope_key(dataset, particle_count, steps, design)  # noqa: SLF001
        for dataset in ("original", "fresh_dgp")
        for particle_count in paired.PARTICLE_COUNTS
        for steps in paired.SINKHORN_STEPS
        for design in paired.DESIGN_FAMILIES
    }
    for key in result["summaries"]:
        rows = json.loads((root / f"rows_{key}.json").read_text(encoding="utf-8"))
        assert [row["seed"] for row in rows] == list(paired.CLAIM_SEEDS)
        assert all(row["gpu_placement"] for row in rows)
        assert all(row["finite"] for row in rows)
    eligible_summaries = {
        key: value
        for key, value in result["summaries"].items()
        if key.startswith("fresh_dgp_")
    }
    assert len(eligible_summaries) == 8
