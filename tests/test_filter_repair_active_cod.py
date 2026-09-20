"""Diagnostic-only shared-body COD row-shape trial; no runtime solver change."""

import ast
import importlib.util
import inspect
import json
import subprocess
import sys
import textwrap
import time
from collections import namedtuple
from pathlib import Path

import pytest
import tensorflow as tf

from bayesfilter.inference import factor_correlation_geometry as factor
from tests import filter_repair_shared_cod_reference as qr
from tests.test_filter_repair_factor_capacity import _memory
from tests.test_filter_repair_initializer_rounding import _record_differences
from tests.test_filter_repair_padded_factor import _data, _public_numerics


def _module(source, name):
    spec = importlib.util.spec_from_loader(name, loader=None)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    exec(compile(source, name, "exec"), module.__dict__)  # noqa: S102 - diagnostic source clone
    return module


def _replace(source, old, new):
    assert source.count(old) == 1, old
    return source.replace(old, new)


def _shared_body_candidate():
    source = subprocess.check_output(["git", "show", "7d08c68e:bayesfilter/ops/qr_lstsq_tf.py"], text=True)
    source += '''

def _active_sum(values, active_rows, minimum_rows):
    def shape_case(rows):
        def reduction():
            return tf.reduce_sum(values[:rows], axis=0)
        return reduction
    branches = tuple(shape_case(rows) for rows in range(minimum_rows, int(values.shape[0]) + 1))
    return tf.switch_case(active_rows - minimum_rows, branches)


def _active_matvec(matrix, vector, active_rows, minimum_rows):
    def shape_case(rows):
        def product():
            return tf.linalg.matvec(matrix[:rows], vector[:rows], transpose_a=True)
        return product
    branches = tuple(shape_case(rows) for rows in range(minimum_rows, int(matrix.shape[0]) + 1))
    return tf.switch_case(active_rows - minimum_rows, branches)
'''
    source = _replace(source, "def _reflector(vector, pivot, active):",
        "def _reflector(vector, pivot, active, active_rows=None, minimum_rows=None):")
    source = _replace(source, "    square = tf.reduce_sum(tail * tail)",
        "    square = (tf.reduce_sum(tail * tail) if active_rows is None\n"
        "              else _active_sum(tail * tail, active_rows, minimum_rows))")
    # This diagnostic observes only the frozen initializer primal. It does not
    # propose a new derivative or claim qualification of the shared COD API.
    source = _replace(source, "@tf.custom_gradient\ndef complete_orthogonal_lstsq(matrix, rhs):",
        "def complete_orthogonal_lstsq(matrix, rhs, active_rows=None):")
    source = _replace(source, "    return solution, grad\n\n\ndef _active_sum",
        "    return solution\n\n\ndef _active_sum")
    source = _replace(source, """        norms = tf.reduce_sum(
            tf.where(row_indices[:, None] >= k, a * a, tf.zeros_like(a)), axis=0
        )""", """        squared = tf.where(row_indices[:, None] >= k, a * a, tf.zeros_like(a))
        norms = (tf.reduce_sum(squared, axis=0) if active_rows is None
                 else _active_sum(squared, active_rows, 2 * cols))""")
    source = _replace(source,
        "        v, tau = _reflector(tf.gather(a, k, axis=1), k, row_indices >= k)",
        "        v, tau = _reflector(tf.gather(a, k, axis=1), k, row_indices >= k, active_rows, 2 * cols)")
    source = _replace(source,
        "        a = a - tau * v[:, None] * tf.linalg.matvec(a, v, transpose_a=True)[None, :]",
        "        product_a = (tf.linalg.matvec(a, v, transpose_a=True) if active_rows is None\n"
        "                     else _active_matvec(a, v, active_rows, 2 * cols))\n"
        "        a = a - tau * v[:, None] * product_a[None, :]")
    source = _replace(source,
        "        b = b - tau * v[:, None] * tf.linalg.matvec(b, v, transpose_a=True)[None, :]",
        "        product_b = (tf.linalg.matvec(b, v, transpose_a=True) if active_rows is None\n"
        "                     else _active_matvec(b, v, active_rows, 2 * cols))\n"
        "        b = b - tau * v[:, None] * product_b[None, :]")
    solver = _module(source, "active_cod_shared_body_diagnostic")
    source = subprocess.check_output(["git", "show", "7d08c68e:bayesfilter/inference/factor_correlation_geometry.py"], text=True)
    source = _replace(source, "            jit_compile=jit_compile,\n        )\n        dense_covariance",
        "            jit_compile=jit_compile,\n"
        "            active_training_rows=active_training_rows,\n        )\n        dense_covariance")
    source = _replace(source, "    jit_compile: bool = True,\n) -> tf.Tensor:\n    root_weight",
        "    jit_compile: bool = True,\n    active_training_rows=None,\n) -> tf.Tensor:\n    root_weight")
    source = _replace(source, "    raw = complete_orthogonal_lstsq(offsets * root_weight, responses * root_weight)",
        "    raw = complete_orthogonal_lstsq(offsets * root_weight, responses * root_weight, active_training_rows)")
    module = _module(source, "factor_active_cod_shared_body_diagnostic")
    module.complete_orthogonal_lstsq = solver.complete_orthogonal_lstsq
    return module


def test_shared_cod_row_arithmetic_localization(request):
    candidate = _shared_body_candidate()
    compact, padded = _data(5, 1, 32)
    config = factor.FactorCorrelationGeometryConfig(factor_count=2)
    observations = []
    for implementation, arguments, rows in ((factor, (*compact, None), 11), (candidate, padded, 42)):
        program = implementation._make_factor_program(5, rows, 10, config, True,
            implementation._prediction_jacobian_diagnostics, padded_training=rows == 42)
        before = _memory()
        started = time.perf_counter()
        result = program(*arguments)
        public = tf.nest.map_structure(lambda value: value.numpy().tolist(), _public_numerics(result))
        elapsed = time.perf_counter() - started
        after = _memory()
        graph = program.get_concrete_function().graph.as_graph_def()
        nodes = list(graph.node) + [node for function in graph.library.function for node in function.node_def]
        hlo = program.experimental_get_compiler_ir(*arguments)(stage="hlo")
        observations.append({"public": public, "raw": result["optimizer"].position.numpy().tolist(),
            "cold_seconds": elapsed, "before_memory": before, "after_memory": after,
            "nodes": len(nodes), "hlo_bytes": len(hlo.encode())})
    differences = _record_differences(observations[1]["public"], observations[0]["public"])
    report = {"role": "explanatory_shared_cod_primal_only", "observations": observations,
        "differences": differences, "runtime_changed": False,
        "derivatives_qualified": False}
    directory = Path(request.config.getoption("xmlpath")).parent
    with (directory / "shared-cod-row-arithmetic.json").open("x") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print("SHARED_COD_ROW_ARITHMETIC " + json.dumps({key: value for key, value in report.items()
        if key != "observations"}, sort_keys=True))


@pytest.mark.parametrize("arm", ["compact", "shared"])
def test_shared_cod_fresh_process_memory(arm, request):
    before = _memory()
    implementation = factor if arm == "compact" else _shared_body_candidate()
    compact, padded = _data(5, 1, 32)
    arguments = (*compact, None) if arm == "compact" else padded
    config = factor.FactorCorrelationGeometryConfig(factor_count=2)
    program = implementation._make_factor_program(5, 11 if arm == "compact" else 42, 10,
        config, True, implementation._prediction_jacobian_diagnostics, padded_training=arm == "shared")
    built = _memory()
    observations = []
    for index in range(21):
        tf.config.experimental.reset_memory_stats("GPU:0")
        started = time.perf_counter()
        result = program(*arguments)
        public = tf.nest.map_structure(lambda value: value.numpy().tolist(), _public_numerics(result))
        observations.append({"seconds": time.perf_counter() - started, "memory": _memory()})
        if index == 0:
            first = public
        else:
            assert first == public
    report = {"role": "explanatory_shared_cod_memory", "arm": arm,
        "before": before, "built": built, "samples": observations, "result": first,
        "after": _memory(), "trace_count": program.experimental_get_tracing_count(),
        "runtime_modified": False, "derivatives_qualified": False}
    directory = Path(request.config.getoption("xmlpath")).parent
    with (directory / "shared-cod-fresh-memory.json").open("x") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print("SHARED_COD_FRESH_MEMORY " + json.dumps(report, sort_keys=True))


@pytest.mark.parametrize("reused,capacity", [(0, 4), (1, 4), (1, 32), (4, 32)])
def test_active_cod_enclosure_localization(reused, capacity, request, monkeypatch):
    """Observe initializer and same-state objective without accepting a fit."""
    candidate = _shared_body_candidate()
    compact, padded = _data(5, reused, capacity)
    config = factor.FactorCorrelationGeometryConfig(factor_count=2)
    observed = namedtuple("ActiveCodEnclosure", (
        "position", "objective_value", "objective_gradient", "converged", "failed",
        "weights", "offsets", "responses"))

    def observe(objective, *, initial_position, **_):
        loss = inspect.getclosurevars(objective.python_function).nonlocals["loss"]
        closure = inspect.getclosurevars(loss).nonlocals
        value, gradient = objective(initial_position)
        return observed(initial_position, value, gradient, tf.constant(False),
            tf.constant(False), closure["weights"], closure["train_z"], closure["train_response"])

    reports = []
    frozen = None
    for fixed in (False, True):
        with monkeypatch.context() as context:
            context.setattr(factor.tfp.optimizer, "lbfgs_minimize", observe)
            if fixed:
                context.setattr(factor, "_encode_state", lambda *_, state=frozen: state)
                context.setattr(candidate, "_encode_state", lambda *_, state=frozen: state)
            factor._make_factor_program.cache_clear()
            candidate._make_factor_program.cache_clear()
            try:
                for label, implementation, arguments, masked in (
                    ("compact", factor, compact, False),
                    ("pinned_shared", candidate, padded, True),
                    ("runtime_active", factor, padded, True),
                ):
                    program = implementation._make_factor_program(5,
                        10 + (capacity if masked else reused), 10, config, True,
                        implementation._prediction_jacobian_diagnostics, padded_training=masked)
                    state = program(*arguments)["optimizer"]
                    fields = {name: getattr(state, name).numpy().tolist() for name in (
                        "position", "objective_value", "objective_gradient", "weights", "offsets", "responses")}
                    if label == "compact":
                        baseline = state
                        if not fixed:
                            frozen = state.position
                    errors = {name: float(tf.reduce_max(tf.abs(getattr(baseline, name) -
                        (getattr(state, name)[:10 + reused] if name in ("weights", "offsets", "responses")
                         else getattr(state, name))))) for name in fields}
                    reports.append({"fixed_initial": fixed, "arm": label, "fields": fields, "errors": errors})
            finally:
                factor._make_factor_program.cache_clear()
                candidate._make_factor_program.cache_clear()
    directory = Path(request.config.getoption("xmlpath")).parent
    report = {"role": "explanatory_active_cod_enclosure", "reused": reused,
        "capacity": capacity, "observations": reports, "runtime_qualified": False}
    with (directory / f"active-cod-enclosure-{reused}-{capacity}.json").open("x") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print("ACTIVE_COD_ENCLOSURE " + json.dumps({**report, "observations": [
        {key: value for key, value in row.items() if key != "fields"} for row in reports]}, sort_keys=True))


def _local_square_candidate():
    source = Path(qr.__file__).read_text()
    source = _replace(source, "else _active_sum(tail * tail, active_rows, minimum_rows)",
        "else _active_square_sum(tail, active_rows, minimum_rows)")
    source = _replace(source, "else _active_sum(squared, active_rows, 2 * cols)",
        "else _active_square_sum(tf.where(row_indices[:, None] >= k, a, tf.zeros_like(a)), active_rows, 2 * cols)")
    source += '''

def _active_square_sum(values, active_rows, minimum_rows):
    return _active_row_operation(lambda rows: tf.reduce_sum(tf.square(values[:rows]), axis=0),
        active_rows, minimum_rows, int(values.shape[0]))
'''
    return _module(source, "local_squared_reduction_diagnostic"), source


def _materialized_dot_candidate():
    source = Path(qr.__file__).read_text()
    old = inspect.getsource(qr._active_matvec)
    source = _replace(source, old, '''def _active_matvec(matrix, vector, active_rows, minimum_rows):
    from tensorflow.compiler.tf2xla.ops.gen_xla_ops import xla_optimization_barrier
    def multiply(rows):
        compact_matrix, compact_vector = xla_optimization_barrier(input=[matrix[:rows], vector[:rows]])
        return tf.linalg.matvec(compact_matrix, compact_vector, transpose_a=True)
    return _active_row_operation(multiply, active_rows, minimum_rows, int(matrix.shape[0]))
''')
    return _module(source, "materialized_dot_diagnostic"), source


def _compact_step_candidate():
    source = Path(qr.__file__).read_text()
    tree = ast.parse(source)
    outer = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
        and node.name == "_complete_orthogonal_lstsq")
    step = next(node for node in outer.body if isinstance(node, ast.FunctionDef) and node.name == "factor")
    old = ast.get_source_segment(source, step)
    original = old.replace("def factor(k, a, b, permutation):", "def compact_factor(k, a, b, permutation):\n        row_indices = tf.range(a.shape[0])\n        active_rows = None")
    new = original + '''

    def factor(k, a, b, permutation):
        if active_rows is None:
            return compact_factor(k, a, b, permutation)
        def execute(count):
            next_k, compact_a, compact_b, next_permutation = compact_factor(k, a[:count], b[:count], permutation)
            return next_k, tf.pad(compact_a, [[0, rows-count], [0, 0]]), tf.pad(compact_b, [[0, rows-count], [0, 0]]), next_permutation
        return _active_row_operation(execute, active_rows, 2*cols, rows)
'''
    source = _replace(source, old, new)
    return _module(source, "compact_cpqr_step_diagnostic"), source


def _compact_cpqr_candidate():
    source = Path(qr.__file__).read_text()
    start = source.index("    def factor(k, a, b, permutation):")
    end = source.index("    upper = tf.pad(upper[:diagonal_size]", start)
    block = source[start:end]
    replacement = ("    def compact_cpqr(matrix, rhs):\n"
        "        row_indices = tf.range(matrix.shape[0])\n"
        "        active_rows = None\n" + textwrap.indent(block, "    ") +
        "        return upper[:diagonal_size], transformed[:diagonal_size], permutation\n\n"
        "    if active_rows is None:\n"
        "        upper, transformed, permutation = compact_cpqr(matrix, rhs)\n"
        "    else:\n"
        "        upper, transformed, permutation = _active_row_operation(\n"
        "            lambda count: compact_cpqr(matrix[:count], rhs[:count]), active_rows, 2*cols, rows)\n")
    source = source[:start] + replacement + source[end:]
    return _module(source, "compact_cpqr_loop_diagnostic"), source


@pytest.mark.parametrize("reused,capacity", [(0, 4), (1, 4), (4, 4), (0, 32), (1, 32), (4, 32)])
@pytest.mark.parametrize("extent", ["step", "cpqr"])
def test_compact_cod_step_complete_fit(reused, capacity, extent, request, monkeypatch):
    module, _ = _compact_step_candidate() if extent == "step" else _compact_cpqr_candidate()
    compact, padded = _data(5, reused, capacity)
    config = factor.FactorCorrelationGeometryConfig(factor_count=2)
    with monkeypatch.context() as context:
        context.setattr(factor, "complete_orthogonal_lstsq_active_rows", module.complete_orthogonal_lstsq_active_rows)
        factor._make_factor_program.cache_clear()
        try:
            before = factor._make_factor_program(5, 10+reused, 10, config, True,
                factor._prediction_jacobian_diagnostics)(*compact)
            after = factor._make_factor_program(5, 10+capacity, 10, config, True,
                factor._prediction_jacobian_diagnostics, padded_training=True)(*padded)
        finally:
            factor._make_factor_program.cache_clear()
    public = [tf.nest.map_structure(lambda x: x.numpy().tolist(), _public_numerics(row)) for row in (before, after)]
    differences = _record_differences(public[1], public[0])
    report = {"role": "diagnostic_compact_cpqr_fit", "extent": extent, "reused": reused,
        "capacity": capacity, "records": public, "differences": differences}
    directory = Path(request.config.getoption("xmlpath")).parent
    with (directory / f"compact-{extent}-fit-{reused}-{capacity}.json").open("x") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print("COMPACT_STEP_FIT " + json.dumps({k:v for k,v in report.items() if k != "records"}, sort_keys=True))
    if extent == "cpqr":
        assert not differences


def _step_program(module, source, rows, masked):
    outer = next(node for node in ast.parse(source).body
        if isinstance(node, ast.FunctionDef) and node.name == "_complete_orthogonal_lstsq")
    nested = next(node for node in outer.body if isinstance(node, ast.FunctionDef) and node.name == "factor")
    body = textwrap.dedent(ast.get_source_segment(source, nested))
    body = _replace(body, "return k + 1, a, b, permutation",
        "return dict(norms=norms, pivot=pivot, v=v, tau=tau, product_a=product_a, product_b=product_b, a=a, b=b, permutation=permutation)")
    body = "def step(k, matrix, rhs, permutation, active_rows=None):\n" + textwrap.indent(
        "rows, cols = matrix.shape\nrow_indices, col_indices = tf.range(rows), tf.range(cols)\n" +
        body + "\nreturn factor(k, matrix, rhs, permutation)\n", "    ")
    namespace = dict(module.__dict__)
    exec(compile(body, "cod_step_diagnostic", "exec"), namespace)  # noqa: S102 - source instrumentation only
    signature = [tf.TensorSpec([], tf.int32), tf.TensorSpec([rows, 5], tf.float64),
        tf.TensorSpec([rows, 5], tf.float64), tf.TensorSpec([5], tf.int32)]
    if masked:
        signature.append(tf.TensorSpec([], tf.int32))
    return tf.function(namespace["step"], input_signature=signature, jit_compile=True, autograph=False)


def test_active_cod_step_localization(request):
    compact, _ = _data(5, 1, 4)
    center, offsets, scores, _, _, weights = compact
    weights /= tf.reduce_sum(weights)
    matrix = offsets * tf.sqrt(weights)[:, None]
    rhs = (center[None, :] - scores) * tf.sqrt(weights)[:, None]
    local_module, local_source = _local_square_candidate()
    materialized_module, materialized_source = _materialized_dot_candidate()
    source = Path(qr.__file__).read_text()
    programs = {"compact": _step_program(qr, source, 11, False),
        "padded_plain": _step_program(qr, source, 14, False),
        "padded_shared": _step_program(qr, source, 14, True),
        "padded_local_square": _step_program(local_module, local_source, 14, True),
        "padded_materialized_dot": _step_program(materialized_module, materialized_source, 14, True)}
    permutation = tf.range(5)
    reports = []
    directory = Path(request.config.getoption("xmlpath")).parent
    for step in range(5):
        baseline = programs["compact"](tf.constant(step), matrix, rhs, permutation)
        for label, program in programs.items():
            padded = label != "compact"
            args = (tf.constant(step), tf.pad(matrix, [[0, 3], [0, 0]]) if padded else matrix,
                tf.pad(rhs, [[0, 3], [0, 0]]) if padded else rhs, permutation)
            if label not in ("compact", "padded_plain"):
                args += (tf.constant(11),)
            result = program(*args)
            errors = {name: float(tf.reduce_max(tf.abs(tf.cast(value, tf.float64) -
                tf.cast(result[name][:11] if name in ("a", "b", "v") else result[name], tf.float64))))
                for name, value in baseline.items()}
            reports.append({"step": step, "arm": label, "errors": errors,
                "fields": {name: value.numpy().tolist() for name, value in result.items()}})
            if step == 0:
                with (directory / f"cod-step-{label}-optimized-hlo.txt").open("x") as handle:
                    handle.write(program.experimental_get_compiler_ir(*args, **({} if len(args) == 5 else {"active_rows": None}))(stage="optimized_hlo"))
        matrix, rhs, permutation = baseline["a"], baseline["b"], baseline["permutation"]
    with (directory / "active-cod-step-localization.json").open("x") as handle:
        json.dump({"role": "explanatory_same_predecessor_cpqr_steps", "observations": reports}, handle, indent=2)
        handle.write("\n")
    print("ACTIVE_COD_STEP " + json.dumps([{key: value for key, value in row.items() if key != "fields"}
        for row in reports], sort_keys=True))
