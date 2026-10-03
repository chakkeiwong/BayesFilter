"""Independent CPU FP64 parity diagnostics; never a candidate/runtime path."""
from __future__ import annotations

import ast
import csv
import inspect
import json
import os
from pathlib import Path
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["BAYESFILTER_PRELOAD_CUSTOM_OP"] = "0"
repo, fixture, output = (Path(p).resolve() for p in sys.argv[1:4])
sys.path.insert(0, str(repo))
import tensorflow as tf
from bayesfilter.score_study import fitted_twist_tf as twist
from bayesfilter.score_study import iapf_adapter, iapf_fit_tf


def matrix(name):
    with (fixture / f"{name}.csv").open() as stream:
        return tf.constant([[float(x) for x in row] for row in csv.reader(stream)], tf.float64)


def vector(name):
    return tf.reshape(matrix(name), [-1])


means, Q, V, center = matrix("means"), matrix("Q"), matrix("V"), vector("center")
log_floor = tf.reshape(matrix("log_floor"), [])
noise, uniform = matrix("noise"), vector("uniform")
points, targets, parameters = matrix("points"), vector("log_targets"), vector("parameters")


@tf.function(input_signature=[
    tf.TensorSpec([5, 2], tf.float64), tf.TensorSpec([2, 2], tf.float64),
    tf.TensorSpec([2, 2], tf.float64), tf.TensorSpec([2], tf.float64),
    tf.TensorSpec([], tf.float64), tf.TensorSpec([5, 2], tf.float64),
    tf.TensorSpec([5], tf.float64), tf.TensorSpec([25, 2], tf.float64),
    tf.TensorSpec([25], tf.float64), tf.TensorSpec([4], tf.float64)], jit_compile=False)
def evaluate(means, Q, V, center, floor, noise, uniform, points, targets, parameters):
    dm, dq = tf.zeros([6, 5, 2], tf.float64), tf.zeros([6, 2, 2], tf.float64)
    log_integral, _, probability = twist.normalizer(means, dm, Q, dq, center, V, floor)
    draws, _ = twist.twisted_transition(means, dm, Q, dq, center, V, probability, noise, uniform)
    coordinates = tf.concat([parameters[:2], parameters[2:]/2], 0)
    normalized_targets = targets-tf.reduce_max(targets)
    density = iapf_fit_tf._density_profile(points, normalized_targets, coordinates, "density_l2")
    relative = iapf_fit_tf._density_profile(points, normalized_targets, coordinates, "relative_shape")
    derivative_conversion = tf.constant([1., 1., .5, .5], tf.float64)
    return (log_integral, probability, draws, density[0], density[1]*derivative_conversion,
            density[3], relative[0], relative[1]*derivative_conversion)


actual = evaluate(means, Q, V, center, log_floor, noise, uniform, points, targets, parameters)
names = ("log_integral", "gaussian_probability", "draws", "density_loss", "density_gradient",
         "shape_error", "relative_loss", "relative_gradient")
checks = []
for name, computed in zip(names, actual):
    reference = tf.reshape(matrix(name), computed.shape)
    error = tf.abs(computed-reference)
    passed = bool(tf.reduce_all(tf.math.is_finite(computed) &
                  (error <= 1e-10+1e-10*tf.abs(reference))).numpy())
    checks.append(dict(name=name, passed=passed, maximum_absolute_error=float(tf.reduce_max(error).numpy())))

# Executable source-binding checks, plus direct tracing above, establish which
# shared implementation the consumer resolves. They do not claim a successful
# multidimensional end-to-end execution; the real endpoint test checks its veto.
def called_names(function):
    tree = ast.parse(inspect.getsource(function))
    return {node.func.id for node in ast.walk(tree)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)}


adapter_tree = ast.parse(inspect.getsource(iapf_adapter.execute_iapf))
imports = {alias.name: node.module for node in ast.walk(adapter_tree)
           if isinstance(node, ast.ImportFrom) and node.level == 1 for alias in node.names}
checks.append(dict(name="consumer_resolves_shared_factories", passed=(
    imports.get("make_fitted_twist_kernel") == "fitted_twist_tf" and
    imports.get("make_density_recursive_fit_kernel") == "iapf_fit_tf" and
    {"make_fitted_twist_kernel", "make_density_recursive_fit_kernel"} <= called_names(iapf_adapter.execute_iapf))))
checks.append(dict(name="density_factory_resolves_actual_profile", passed=(
    "bounded_density_fit" in called_names(iapf_fit_tf.make_density_recursive_fit_kernel.__wrapped__) and
    "_density_profile" in called_names(iapf_fit_tf.bounded_density_fit) and
    iapf_fit_tf.bounded_density_fit.__globals__["_density_profile"] is iapf_fit_tf._density_profile)))
checks.append(dict(name="filter_factory_resolves_shared_twisting", passed=(
    {"normalizer", "twisted_transition"} <= called_names(twist.make_fitted_twist_kernel.__wrapped__) and
    twist.make_fitted_twist_kernel.__wrapped__.__globals__["normalizer"] is twist.normalizer and
    twist.make_fitted_twist_kernel.__wrapped__.__globals__["twisted_transition"] is twist.twisted_transition)))
try:
    iapf_adapter.execute_iapf({"role": "diagnostic", "model": "linear_gaussian"},
        dict(dimension=2, observation_dimension=2, particles=25, horizon=1),
        tf.zeros([6], tf.float64), tf.zeros([1, 2], tf.float64), None)
except ValueError as error:
    reason = str(error)
    scalar_veto = "requires scalar state and observation" in reason
else:
    reason, scalar_veto = "endpoint unexpectedly accepted dimensions", False
checks.append(dict(name="actual_endpoint_scalar_restriction", passed=scalar_veto, message=reason))
report = dict(status="pass" if all(c["passed"] for c in checks) else "fail", checks=checks,
    tensorflow_version=tf.__version__, cpu_only=True, gpu_intentionally_hidden=True,
    jit_compile=False, dtype="float64", seed="N/A deterministic fixture",
    role="shared_component_and_wiring_diagnostic", full_filter_parity_checked=False,
    original_paper_replication=False, default_readiness=False)
output.write_text(json.dumps(report, indent=2)+"\n")
print(json.dumps(report))
raise SystemExit(0 if report["status"] == "pass" else 1)
