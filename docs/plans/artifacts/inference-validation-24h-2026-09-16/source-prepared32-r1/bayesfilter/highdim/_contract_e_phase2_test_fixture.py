"""Private source-binding fixture for Contract E schema-v2 tests.

These functions are not filtering implementations and are never registered by
the production identity factory.
"""

from __future__ import annotations

import math
import packaging.version
import tensorflow as tf


FIXTURE_SCALE = 2.0


def fixture_dependency(value: float) -> float:
    return math.fsum((float(value), 1.0))


def fixture_reset(value: float) -> float:
    return FIXTURE_SCALE * fixture_dependency(value)


def fixture_value(value: float) -> float:
    return fixture_reset(value) + math.fabs(float(value))


def fixture_gradient(value: float) -> float:
    return fixture_reset(value) - math.fabs(float(value))


def fixture_raw_route(value: float) -> float:
    return float(value)


def fixture_external(value: float) -> float:
    return float(value) + float(len(packaging.version.Version("1.2").release))


@tf.function(jit_compile=True)
def fixture_tf_reset(value):
    return fixture_reset(value)


@tf.function(jit_compile=True)
def fixture_tf_value(value):
    return fixture_value(value)


@tf.function(jit_compile=True)
def fixture_tf_gradient(value):
    return fixture_gradient(value)
