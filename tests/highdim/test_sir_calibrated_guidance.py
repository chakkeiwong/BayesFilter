"""Synthetic calibration-contract tests, not scientific evidence."""

import importlib.util
from pathlib import Path

import pytest


MODULE_PATH = Path(__file__).resolve().parents[2] / "scripts/run_sir_calibrated_guidance.py"
SPEC = importlib.util.spec_from_file_location("sir_calibrated_runner", MODULE_PATH)
RUNNER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(RUNNER)


def records():
    return [dict(data_seed=26100831, arm=arm, beta=beta, valid=True, converged=True, gap=1e-9, objective=2.)
            for arm, beta in (("plain", [.4, .3, .3]), ("csmc_quadratic", [.7, .2, .1]), ("kernel", [.2, .4, .4]))]


def test_matched_controls_use_identical_frozen_probabilities():
    arms = RUNNER.calibrated_arms(records(), 26100831)
    assert len(arms) == 7
    assert arms["csmc_quadratic"] == arms["plain_quadratic_beta"]
    assert arms["kernel"] == arms["plain_kernel_beta"]
    assert arms["plain"] != arms["kernel"]
    assert arms["prior"] == [1., 0., 0.]


@pytest.mark.parametrize("field,value", [("valid", False), ("converged", False), ("gap", 1.),
    ("gap", float("nan")), ("objective", float("inf")),
    ("beta", [.05, .5, .45]), ("beta", [.4, .3, .4]), ("beta", [.4, float("nan"), .6])])
def test_invalid_calibration_is_rejected(field, value):
    calibration = records()
    calibration[0][field] = value
    with pytest.raises(ValueError):
        RUNNER.calibrated_arms(calibration, 26100831)


def test_duplicate_calibration_is_rejected():
    calibration = records()
    calibration.append(calibration[0])
    with pytest.raises(ValueError):
        RUNNER.calibrated_arms(calibration, 26100831)
