"""Diagnostic provenance test for the frozen C2 model/data/hint fixture."""

import hashlib
import json
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_DIR = ROOT / "docs/benchmarks"
FIXTURE_PATH = BENCHMARK_DIR / "fixtures/c2_sv_n4_seed52_obs42_t20_frozen_v1.json"
SOURCE_PATH = BENCHMARK_DIR / "sv_fixture_c2_20260826.py"
sys.path.insert(0, str(BENCHMARK_DIR))
import sv_fixture_c2_20260826 as source  # noqa: E402


def test_frozen_fixture_exactly_reproduces_seeded_source() -> None:
    frozen = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    assert frozen["schema_id"] == "bayesfilter.c2_sv_frozen_fixture.v1"
    assert frozen["classification"] == "cpu_only_numpy_diagnostic_fixture_freeze"
    assert frozen["source_generator_sha256"] == hashlib.sha256(
        SOURCE_PATH.read_bytes()
    ).hexdigest()

    model = source.sv_model(4, 52)
    observations = source.sv_simulate(model, 20, 42)
    np.testing.assert_array_equal(frozen["transition_matrix"], model["A"])
    np.testing.assert_array_equal(frozen["process_covariance"], model["Q"])
    np.testing.assert_array_equal(frozen["stationary_covariance"], model["P0"])
    np.testing.assert_array_equal(frozen["observations"], observations)

    initial_hint, predictive_hint = source.sv_gh_hint_factory(model, gh_points=9)
    mean, covariance = initial_hint(observations[0])
    np.testing.assert_array_equal(frozen["moment_hints"][0]["mean"], mean.numpy())
    np.testing.assert_array_equal(
        frozen["moment_hints"][0]["covariance"], covariance.numpy()
    )
    for time_index in range(1, 20):
        mean, covariance = predictive_hint(time_index, observations[time_index])
        np.testing.assert_array_equal(
            frozen["moment_hints"][time_index]["mean"], mean.numpy()
        )
        np.testing.assert_array_equal(
            frozen["moment_hints"][time_index]["covariance"], covariance.numpy()
        )
