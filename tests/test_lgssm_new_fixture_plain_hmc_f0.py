from __future__ import annotations

import ast
import os
from pathlib import Path

import tensorflow as tf

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

from bayesfilter.testing import lgssm_new_fixture_plain_hmc_f0_tf as campaign


def test_f0_inputs_bind_new_target_and_mass() -> None:
    bundle, mass, adapter = campaign._load_inputs()
    assert bundle.target_signature == campaign.EXPECTED_TARGET_SIGNATURE
    assert mass["artifact_hash"] == campaign.EXPECTED_MASS_ARTIFACT_HASH
    assert adapter.adapter_signature()


def test_f0_affine_value_score_matches_chain_rule() -> None:
    bundle, _mass, adapter = campaign._load_inputs()
    z = tf.zeros((4, 18), tf.float64)
    theta = adapter.forward(z)
    raw_value, raw_score = bundle.adapter.log_prob_and_grad(theta)
    value, score = adapter.log_prob_and_grad(z)
    tf.debugging.assert_near(value, raw_value + adapter.log_abs_det)
    tf.debugging.assert_near(score, tf.tensordot(raw_score, adapter.factor, axes=[[-1], [0]]))


def test_f0_route_has_no_numpy_or_local_tfp_sampler() -> None:
    path = Path(campaign.__file__)
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert not any(name == "numpy" or name.startswith("numpy.") for name in imported)
    assert "tf.numpy_function" not in source
    assert "tf.py_function" not in source
    assert "tfp.mcmc.sample_chain" not in source
    assert "tfp.mcmc.HamiltonianMonteCarlo" not in source


def test_f0_repair_grid_is_inside_failed_bracket_and_preserves_attempt() -> None:
    assert min(campaign.STEP_SIZES) > 0.2
    assert max(campaign.STEP_SIZES) < 0.4
    assert campaign.FAILED_COMPARATOR_ROOT.joinpath("result.json").is_file()
    assert campaign.COMPARATOR_ROOT != campaign.FAILED_COMPARATOR_ROOT
