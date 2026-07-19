import importlib.util
from pathlib import Path

import pytest
import tensorflow as tf


PATH = Path("docs/benchmarks/run_ssl_lstm_complexity_ladder_2026_07_18.py")
SPEC = importlib.util.spec_from_file_location("ssl_lstm_ladder", PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.mark.parametrize(
    ("q", "state_dim", "parameter_dim"),
    ((1, 3, 24), (2, 6, 64), (5, 15, 292), (10, 30, 1032), (20, 60, 3862)),
)
def test_ladder_dimension_formula_matches_source(q: int, state_dim: int, parameter_dim: int) -> None:
    config = MODULE.make_config(q)
    assert config.augmented_state_dim == state_dim
    assert config.parameter_dim == parameter_dim
    assert MODULE.closed_form_parameter_dim(q) == parameter_dim


@pytest.mark.parametrize("q", MODULE.RUNGS)
def test_fixture_has_correct_shapes_and_positive_covariances(q: int) -> None:
    config = MODULE.make_config(q)
    theta = MODULE.make_fixture_theta(config)
    params = MODULE.unpack_ssl_lstm_parameters(theta, config)
    assert theta.shape == (config.parameter_dim,)
    assert params.lstm_input.shape == (4, q, q)
    assert params.lstm_recurrent.shape == (4, q, q)
    assert params.initial_covariance.shape == (3 * q, 3 * q)
    assert bool(tf.reduce_all(tf.linalg.diag_part(params.initial_covariance) > 0.0).numpy())
    assert bool(tf.reduce_all(tf.linalg.diag_part(params.ukf_innovation_covariance) > 0.0).numpy())


def test_closed_form_parameter_dim_rejects_invalid_q() -> None:
    for value in (0, -1, True, 1.5):
        with pytest.raises(ValueError):
            MODULE.closed_form_parameter_dim(value)


def test_q1_rung_passes_full_engineering_contract_cpu_reference() -> None:
    row = MODULE.run_rung(1)
    assert row["status"] == "PASSED"
    assert row["parameter_dim"] == 24
    assert row["minimum_filtered_covariance_eigenvalue"] >= -1.0e-10
    assert not row["hard_vetoes"]


def test_subprocess_timeout_is_a_resource_veto(monkeypatch, tmp_path: Path) -> None:
    def timeout(*args, **kwargs):
        raise MODULE.subprocess.TimeoutExpired(cmd=args[0], timeout=kwargs["timeout"])

    monkeypatch.setattr(MODULE.subprocess, "run", timeout)
    row = MODULE._run_rung_subprocess(
        1,
        output_dir=tmp_path,
        timeout_seconds=0.25,
    )
    assert row["status"] == "FAILED"
    assert row["hard_vetoes"] == ["per_rung_subprocess_timeout_at_remaining_wall_cap"]
    assert row["failure_classification"].startswith("resource_continuation_veto")
