"""Locator preparation preserves the original disconnected public geometry."""

import importlib.util
import subprocess
import sys

import pytest
import tensorflow as tf

from bayesfilter.inference import sequential_map_covariance as candidate
from tests.test_filter_repair_batched_locator import _batch
from tests.test_filter_repair_fixed_stability import _compare
from tests.test_filter_repair_sequential_locator import _target

D = tf.float64


@pytest.fixture(scope="module")
def original():
    source = subprocess.check_output(["git", "show",
        "3582b4ac:bayesfilter/inference/sequential_map_covariance.py"], text=True)
    spec = importlib.util.spec_from_loader("locator_frozen_original_reference", loader=None)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    exec(compile(source, spec.name, "exec"), module.__dict__)  # noqa: S102
    return module


@pytest.mark.parametrize("mode", ["scalar", "batched", "buffered"])
def test_original_locator_records_remain_frozen_under_external_tape(original, mode):
    records = []
    for module in (original, candidate):
        starts = tf.constant([[.4, -.5], [-.3, .2]], D)
        scale = tf.constant([.7, 1.3], D)
        with tf.GradientTape() as tape:
            tape.watch((starts, scale))
            result = module.estimate_sequential_map_covariance(_target("quadratic"), starts,
                batched_locator_value_and_score_fn=None if mode == "scalar" else _batch("quadratic"),
                scale=scale, progress_callback=(lambda _: None) if mode == "buffered" else None,
                config=module.SequentialMapCovarianceConfig(locator_max_iterations=2,
                    locator_max_line_search_iterations=4, max_exact_evaluations=1))
            assert result.map_candidate is not None
            value = tf.reduce_sum(tf.convert_to_tensor(result.map_candidate, D))
        assert tape.gradient(value, (starts, scale)) == (None, None)
        records.append(result.payload())
    _compare(records[1], records[0])
