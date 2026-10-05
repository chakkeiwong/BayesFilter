"""Diagnostic parity for the merged shared Hermite inverse and retired routes."""

import functools
import subprocess
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.highdim import c2_gaussian_hermite_proposal_tf as candidate
from tests.test_filter_repair_remaining_routes import _graph

D = tf.float64
REMOTE = '5e16df06f586c16bc58fb76bc62d4f6451e7690d'
ROOT = Path(__file__).resolve().parents[1]


@functools.lru_cache(maxsize=1)
def reference():
    path = 'bayesfilter/highdim/c2_gaussian_hermite_proposal_tf.py'
    source = subprocess.check_output(['git', 'show', f'{REMOTE}:{path}'], cwd=ROOT, text=True)
    module = types.ModuleType('filter_repair_remote_hermite_reference')
    sys.modules[module.__name__] = module
    exec(compile(source, f'{REMOTE}:{path}', 'exec'), module.__dict__)  # noqa: S102 - pinned diagnostic authority
    return module


@pytest.mark.parametrize('reverse', [False, True])
@pytest.mark.parametrize('batched', [False, True])
def test_shared_inverse_preserves_remote_batched_and_reverse_records(reverse, batched):
    ranks = (1, 2, 3)
    cores = tuple(tf.reshape(.3 + .1 * tf.sin(tf.cast(tf.range(ranks[i]*3*ranks[i+1]), D)),
        [ranks[i], 3, ranks[i+1]]) for i in range(2))
    if batched:
        cores = tuple(tf.stack([core, core * 1.1, core * .9, core * 1.05]) for core in cores)
    suffix = tf.linalg.diag(tf.constant([1.2, .8, 1.1], D))
    if reverse:
        # A distinct conditioning suffix for every particle must remain an operand.
        suffix = tf.stack([suffix * 1.1, suffix * .9, suffix * 1.2, suffix])
    uniforms = tf.constant([[.2, .8], [.4, .6], [.7, .3], [.1, .9]], D)
    signature = (tuple(tf.TensorSpec(c.shape, D) for c in cores),
        tf.TensorSpec(suffix.shape, D), tf.TensorSpec(uniforms.shape, D))
    program = candidate._inverse_program(signature, 64, reverse)
    assert program._jit_compile is True
    hlo = program.experimental_get_compiler_ir(cores, suffix, uniforms)(stage='hlo')
    for factor in (1., 1.03, 1.):
        values = tuple(c * factor for c in cores)
        actual = program(values, suffix, uniforms)
        expected = reference().inverse_hermite_polynomial_kr(values, suffix, uniforms, reverse=reverse)
        assert bool(actual['finite']) and bool(actual['cdf_bracket_valid'])
        for key in expected:
            if expected[key].dtype == tf.bool:
                np.testing.assert_array_equal(actual[key], expected[key])
            else:
                np.testing.assert_allclose(actual[key], expected[key], atol=1e-10, rtol=1e-10)
    assert program.experimental_get_tracing_count() == 1
    assert program.experimental_get_compiler_ir(values, suffix, uniforms)(stage='hlo') == hlo
    assert not program.get_concrete_function().captured_inputs
    _graph(program)


def test_remote_genut_bootstrap_retirement_is_preserved():
    for name in ('cubature_genut_batch_tf', 'cubature_genut_neutra_targets'):
        assert not (ROOT / 'bayesfilter/highdim' / f'{name}.py').exists()
    assert (ROOT / 'bayesfilter/highdim/ledh_canonical_score_tf.py').is_file()


def test_internal_warmup_attempt_defaults_to_xla():
    import inspect

    from bayesfilter.inference.hmc_warmup import (
        _run_operational_windowed_warmup_attempt,
    )

    assert inspect.signature(_run_operational_windowed_warmup_attempt).parameters['jit_compile'].default is True
