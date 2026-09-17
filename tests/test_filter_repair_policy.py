"""Focused enforcement for admitted filter/gradient execution routes."""

from __future__ import annotations

import ast
import importlib.util
import inspect
import json
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _guard():
    spec = importlib.util.spec_from_file_location("filter_policy_guard", ROOT / "scripts/enforce_filter_gradient_policy.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reviewed_source_closure_satisfies_exact_policy():
    report = _guard().verify()
    assert report["passed"], report


@pytest.mark.parametrize("source,kind", [
    ("def evaluate(xs):\n    for x in xs:\n        yield x * x\n", "python_iteration"),
    ("def evaluate(xs):\n    return [x * x for x in xs]\n", "python_iteration"),
    ("import numpy as np\nx = np.sum([1., 2.])\n", "numpy_numerics"),
    ("from numpy import sum as total\nx = total([1., 2.])\n", "numpy_import"),
    ("import tensorflow as tf\nx = tf.py_function(f, x, tf.float64)\n", "python_numerical_callback"),
    ("from tensorflow import numpy_function as callback\nx = callback(f, x, T)\n", "python_numerical_callback"),
    ("def evaluate(x, *, jit_compile=False):\n    return x\n", "default_jit_disabled"),
    ("import tensorflow as tf\nf = tf.function(evaluate, jit_compile=False)\n", "non_xla_function"),
    ("import tensorflow as tf\n@tf.function()\ndef evaluate(x):\n    return x\n", "non_xla_function"),
    ("import tensorflow as tf\n@tf.function\ndef evaluate(x):\n    return x\n", "non_xla_function"),
    ("x = tape.jacobian(y, theta)\n", "implicit_pfor"),
])
def test_guard_rejects_unreviewed_runtime_operations(source, kind):
    findings = _guard().inspect_source("kernel.py", source)
    assert kind in {finding["kind"] for finding in findings}


def test_changed_or_deleted_exception_fails_closed(tmp_path):
    guard = _guard()
    source = "def report(xs):\n    return [str(x) for x in xs]\n"
    kernel = tmp_path / "kernel.py"
    kernel.write_text(source)
    finding, = guard.inspect_source("kernel.py", source)
    policy = tmp_path / "policy.json"
    policy.write_text(json.dumps({"scope_note": "Fixture host report only.",
        "sources": {"kernel.py": None}, "exceptions": [{**finding,
            "role": "host_reporting", "reason": "Completed host report formatting."}]}))
    assert guard.verify(tmp_path, policy)["passed"]
    kernel.write_text(source.replace("str(x)", "x * x"))
    changed = guard.verify(tmp_path, policy)
    assert not changed["passed"]
    assert len(changed["violations"]) == len(changed["stale_exceptions"]) == 1
    kernel.write_text("def report(xs):\n    return xs\n")
    deleted = guard.verify(tmp_path, policy)
    assert not deleted["passed"] and deleted["stale_exceptions"]


def _function(path: str, name: str) -> ast.FunctionDef:
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == name:
            return node
    raise AssertionError(f"missing function {path}:{name}")


def _loop_nodes(function: ast.FunctionDef) -> list[ast.AST]:
    return [node for node in ast.walk(function) if isinstance(node, (ast.For, ast.AsyncFor, ast.While))]


def test_complete_admitted_recursions_have_native_control_flow():
    admitted = (
        ("experiments/dpf_implementation/tf_tfp/filters/bootstrap_pf_tf.py", "evaluate"),
        ("experiments/dpf_implementation/tf_tfp/filters/experimental_batched_ledh_pfpf_ot_tf.py", "batched_ledh_pfpf_ot_value_core_tf"),
        ("bayesfilter/highdim/zhao_cui_frozen_proposal_apf_tf.py", "_evaluate_core"),
        ("bayesfilter/highdim/ledh_pfpf_genut_initialization_tf.py", "finite_value_standard_score_ledh_pfpf_genut"),
    )
    for path, name in admitted:
        assert not _loop_nodes(_function(path, name)), f"Python numerical loop remains in {path}:{name}"


def test_admitted_runtime_modules_are_numpy_free():
    paths = (
        "experiments/dpf_implementation/tf_tfp/filters/bootstrap_pf_tf.py",
        "experiments/dpf_implementation/tf_tfp/filters/experimental_batched_ledh_pfpf_ot_tf.py",
        "experiments/dpf_implementation/tf_tfp/resampling/sinkhorn_tf.py",
        "bayesfilter/highdim/zhao_cui_frozen_proposal_apf_tf.py",
        "bayesfilter/highdim/ledh_pfpf_genut_initialization_tf.py",
    )
    for relative in paths:
        tree = ast.parse((ROOT / relative).read_text(encoding="utf-8"))
        imports = [node for node in ast.walk(tree) if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("numpy")]
        imports += [node for node in ast.walk(tree) if isinstance(node, ast.Import) and any(alias.name.startswith("numpy") for alias in node.names)]
        assert not imports, f"NumPy import in admitted module {relative}"


def test_ledh_score_provenance_blocks_unsupported_canonical_admission():
    from experiments.dpf_implementation.tf_tfp.filters import experimental_batched_ledh_pfpf_ot_tf as dense
    from bayesfilter.highdim import ledh_contract_e_canonical_lgssm_tf as contract_e

    assert dense.precision_policy_metadata()["canonical_algorithm_admitted"] is False
    assert dense.precision_policy_metadata()["canonical_rebuild_required"] is True
    assert contract_e.CANONICAL_LEDH_ADMITTED is False
    assert contract_e.SCORE_PROVENANCE.endswith("diagnostic_only")


def test_stable_native_factories_default_to_xla():
    from experiments.dpf_implementation.tf_tfp.filters.bootstrap_pf_tf import make_bootstrap_particle_filter_tf
    from experiments.dpf_implementation.tf_tfp.resampling.sinkhorn_tf import make_sinkhorn_resample_tf

    bootstrap_signature = inspect.signature(make_bootstrap_particle_filter_tf)
    sinkhorn_signature = inspect.signature(make_sinkhorn_resample_tf)
    assert bootstrap_signature.parameters["jit_compile"].default is True
    assert sinkhorn_signature.parameters["jit_compile"].default is True
