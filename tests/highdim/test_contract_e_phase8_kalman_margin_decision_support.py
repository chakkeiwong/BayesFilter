from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import tensorflow as tf


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "docs/benchmarks/emit_contract_e_phase8_kalman_margin_decision_support.py"
SPEC = importlib.util.spec_from_file_location("phase8_kalman_support", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
support = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(support)


def test_coordinate_roundtrip_and_chain_factors() -> None:
    theta = tf.constant(support.THETA, tf.float64)
    u = support._physical_to_hmc(theta)
    tf.debugging.assert_near(
        support._hmc_to_physical(u), theta, atol=2e-15, rtol=2e-15
    )
    tf.debugging.assert_near(
        support._chain_factors(theta),
        tf.constant([0.4816, 0.6975, 0.8775, 0.35, 0.45], tf.float64),
        atol=1e-15,
        rtol=1e-15,
    )


def test_proposal_radii_are_frozen_values() -> None:
    radii = support._proposal_radii(tf.constant(support.THETA, tf.float64))
    tf.debugging.assert_near(
        radii,
        tf.constant(
            [
                2.739425806383947,
                2.4501621366392863,
                2.1972245773362187,
                1.945910149055313,
                2.197224577336219,
            ],
            tf.float64,
        ),
        atol=1e-15,
        rtol=1e-15,
    )


def test_budget_coefficient_inverts_component_metric() -> None:
    radii = support._proposal_radii(tf.constant(support.THETA, tf.float64))
    s_oracle = tf.constant(3.25, tf.float64)
    delta = tf.constant(0.07, tf.float64)
    absolute_budget = delta * s_oracle / radii
    reconstructed = radii * absolute_budget / s_oracle
    tf.debugging.assert_near(
        reconstructed,
        tf.fill([5], delta),
        atol=1e-15,
        rtol=1e-15,
    )


def test_forbidden_dependency_audits_are_empty() -> None:
    assert support._forbidden_source_imports() == []
    probe = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import importlib.util,json,pathlib;"
                f"p=pathlib.Path({str(SCRIPT)!r});"
                "s=importlib.util.spec_from_file_location('phase8_kalman_probe',p);"
                "m=importlib.util.module_from_spec(s);"
                "s.loader.exec_module(m);"
                "print(json.dumps(m._forbidden_loaded_modules()))"
            ),
        ],
        cwd=ROOT,
        env={**os.environ, "CUDA_VISIBLE_DEVICES": "-1"},
        check=True,
        capture_output=True,
        text=True,
    )
    assert json.loads(probe.stdout.strip()) == []


def test_chain_tolerance_is_componentwise_and_scale_aware() -> None:
    direct = tf.constant([0.0, 2.0, -4.0], tf.float64)
    chained = tf.constant([0.5, 3.0, -2.0], tf.float64)
    expected_scale = tf.constant([1.0, 3.0, 4.0], tf.float64)
    tf.debugging.assert_equal(
        support._chain_tolerance(direct, chained),
        tf.constant(
            support.CHAIN_TOLERANCE_MULTIPLIER * support.FLOAT64_UNIT_ROUNDOFF,
            tf.float64,
        )
        * expected_scale,
    )
