"""Diagnostic resource cost of changed-data XLA specialization in factor fits."""

import hashlib
import importlib.util
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import pytest
import tensorflow as tf

from bayesfilter.inference import factor_correlation_geometry as current
from bayesfilter.ops import qr_lstsq_tf
from tests.test_filter_repair_padded_factor import (
    _assert_runtime_parameters,
    _data,
    _public_numerics,
)

CHECKPOINT = "1e9afd2c"


def _pinned(path, name):
    source = subprocess.check_output(["git", "show", f"{CHECKPOINT}:{path}"], text=True)
    spec = importlib.util.spec_from_loader(name, loader=None)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    exec(compile(source, name, "exec"), module.__dict__)  # noqa: S102 - pinned diagnostic
    return module, hashlib.sha256(source.encode()).hexdigest()


def _memory():
    host = {row.split()[0].rstrip(":"): int(row.split()[1]) * 1024
        for row in Path("/proc/self/status").read_text().splitlines()
        if row.startswith(("VmRSS:", "VmHWM:"))}
    return {"host": host, "gpu": tf.config.experimental.get_memory_info("GPU:0")}


@pytest.mark.parametrize("arm", ["checkpoint", "candidate"])
@pytest.mark.parametrize("jit", [False, True])
def test_changing_training_cloud_compilation_memory(arm, jit, request):
    before = _memory()
    if arm == "checkpoint":
        factor, factor_hash = _pinned("bayesfilter/inference/factor_correlation_geometry.py", "factor_compilation_reference")
        solver, solver_hash = _pinned("bayesfilter/ops/qr_lstsq_tf.py", "factor_compilation_solver_reference")
        factor.complete_orthogonal_lstsq = solver.complete_orthogonal_lstsq
    else:
        factor = current
        factor_hash, solver_hash = (hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
            for module in (current, qr_lstsq_tf))
    config = factor.FactorCorrelationGeometryConfig(factor_count=1, max_iterations=4)
    inputs, _ = _data(3, 4, 4)
    program = factor._make_factor_program(3, 10, 6, config, jit, factor._prediction_jacobian_diagnostics)
    built = _memory()
    samples = []

    def observe(arguments, phase, cloud):
        tf.config.experimental.reset_memory_stats("GPU:0")
        started = time.perf_counter()
        result = program(*arguments)
        # Synchronize and retain the same complete public fields in both arms.
        public = tf.nest.map_structure(lambda value: value.numpy().tolist(), _public_numerics(result))
        elapsed = time.perf_counter() - started
        sample = {"phase": phase, "cloud": cloud, "seconds": elapsed, "memory": _memory(), "result": public}
        samples.append(sample)
        assert public["finite"]

    observe(inputs, "cold", 0)
    for _ in range(3):
        observe(inputs, "same_input_warm", 0)
    for cloud in range(1, 5):
        # Same shapes, new offsets, scores and weights. No RNG stream is used.
        changed = (inputs[0], inputs[1] * (1. + .01 * cloud), inputs[2] * (1. + .015 * cloud),
            *inputs[3:5], inputs[5] * tf.linspace(tf.constant(1., tf.float64),
                tf.constant(1. + .02 * cloud, tf.float64), 10))
        observe(changed, "changed_input_first", cloud)
        observe(changed, "changed_input_warm", cloud)
    for _ in range(10):
        observe(changed, "last_input_warm", 4)
    measured = _memory()
    compiler = []
    directory = Path(request.config.getoption("xmlpath")).parent
    if jit:
        for index, arguments in enumerate((inputs, changed)):
            if arm == "candidate":
                arguments = (*arguments, None)
            hlo = program.experimental_get_compiler_ir(*arguments)(stage="hlo")
            entry = hlo[hlo.rfind("\nENTRY "):]
            path = directory / f"factor-memory-{index}-hlo.txt"
            with path.open("x") as handle:
                handle.write(hlo)
            compiler.append({"path": str(path), "sha256": hashlib.sha256(hlo.encode()).hexdigest(),
                "runtime_parameters": len(re.findall(r"\bparameter\(\d+\)", entry)),
                "internal_guard_resources": 1 if arm == "candidate" else 0})
            if arm == "candidate":
                _assert_runtime_parameters(program, arguments, hlo)
        if arm == "candidate":
            assert all(row["runtime_parameters"] == 7 for row in compiler)
            assert compiler[0]["sha256"] == compiler[1]["sha256"]
    assert program.experimental_get_tracing_count() == 1
    report = {"role": "explanatory_changed_input_compilation_memory_only", "arm": arm,
        "checkpoint": CHECKPOINT, "jit_compile": jit, "dimension": 3, "factor_count": 1,
        "max_iterations": 4, "timing_scope": "complete_numerical_call_and_public_field_materialization",
        "non_jit_role": None if jit else "explicit_graph_reference_exception",
        "source_sha256": {"factor": factor_hash, "solver": solver_hash},
        "stages": {"before": before, "built": built, "after_measurement": measured},
        "samples": samples, "compiler": compiler, "trace_count": program.experimental_get_tracing_count(),
        "nonclaims": ["No terminal repeats or statistically supported timing ranking.",
            "Known complete-fit record gates remain open; finite diagnostic output does not qualify a fitter.",
            "Compiler inspection runs after the timed/memory sequence."]}
    print("FACTOR_COMPILATION_MEMORY " + json.dumps(report, sort_keys=True))
