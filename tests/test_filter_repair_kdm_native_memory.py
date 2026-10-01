"""Linux native-memory and enclosing-XLA diagnostics, never runtime code."""

import ctypes
import gc
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import weakref
import xml.etree.ElementTree as ET
from collections import Counter
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import tensorflow as tf
from tensorflow.python.eager import context

from tests.test_filter_repair_gap_diagnostics import memory_snapshot
from tests.test_filter_repair_kdm_auxiliary import (
    _assert_record_equal,
    _fixture,
    _json,
    _original,
    _owner,
    _public_record,
    kdm_module,
)


class _Mallinfo2(ctypes.Structure):
    _fields_ = [(name, ctypes.c_size_t) for name in (
        "arena", "ordblks", "smblks", "hblks", "hblkhd", "usmblks",
        "fsmblks", "uordblks", "fordblks", "keepcost")]


def _native_snapshot(gpu):
    """Read glibc counters and summed mapping categories without allocator edits."""
    call = ctypes.CDLL(None).mallinfo2
    call.argtypes, call.restype = [], _Mallinfo2
    info = call()
    mappings = {}
    category = None
    for line in Path("/proc/self/smaps").read_text().splitlines():
        if re.match(r"^[0-9a-f]+-[0-9a-f]+ ", line):
            fields = line.split(maxsplit=5)
            name = fields[5] if len(fields) == 6 else ""
            category = ("anonymous" if not name else "heap" if name == "[heap]"
                        else "stack" if name.startswith("[stack")
                        else "special" if name.startswith("[") else "file")
            mappings.setdefault(category, Counter())["mapping_count"] += 1
        elif category and line.split(":", 1)[0] in (
                "Size", "Rss", "Pss", "Private_Dirty", "Private_Clean",
                "Anonymous", "Swap"):
            field, value, unit = line.split()
            assert unit == "kB"
            mappings[category][field.rstrip(":") + "_bytes"] += int(value) * 1024
    return {**memory_snapshot(gpu),
            "registered_functions": len(context.context().list_function_names()),
            "mallinfo2": {name: getattr(info, name) for name, _ in info._fields_},
            "mappings": mappings}


def _make_owner(model, inputs, options, jit_compile, ownership):
    original = kdm_module.make_linear_gaussian_kdm_normalizer_kernel

    def enclosing_only(**kwargs):
        # Diagnostic configuration of the same implementation. The enclosing
        # owner traces these tensor ops and retains jit_compile=True by default.
        return original(**kwargs).python_function

    factory = original if ownership == "nested" else enclosing_only
    with patch.object(kdm_module, "make_linear_gaussian_kdm_normalizer_kernel", factory):
        return _owner(model, inputs, options, jit_compile)


@pytest.mark.parametrize("jit_compile", [False, True], ids=["graph", "xla"])
@pytest.mark.parametrize("ownership", ["nested", "enclosing"])
def test_native_allocation_categories(ownership, jit_compile, request):
    model, inputs, options = _fixture()
    gpu = os.environ.get("BAYESFILTER_TEST_DEVICE_SCOPE") == "visible"
    output = Path(request.config.getoption("xmlpath")).parent / "kdm-native-allocation.json"
    report = {"role": "native_allocation_diagnostic_no_cache_owner_claim",
              "ownership": ownership, "jit_compile": jit_compile,
              "before": _native_snapshot(gpu), "rounds": []}
    first = None
    for index in range(6):
        owner = _make_owner(model, inputs, options, jit_compile, ownership)
        begin = time.perf_counter()
        result = owner(*inputs)
        arrays = _json(result)
        elapsed = time.perf_counter() - begin
        replay = _json(owner(*inputs))
        assert replay == arrays
        if first is None:
            first = arrays
        assert first == arrays
        traces = owner.experimental_get_tracing_count()
        refs = {"owner": weakref.ref(owner),
                "graph": weakref.ref(owner.get_concrete_function().graph)}
        compiled = _native_snapshot(gpu)
        del owner, result
        collections = []
        for _ in range(3):
            collected = gc.collect()
            collections.append({"collected": collected,
                "released": {name: ref() is None for name, ref in refs.items()}})
        report["rounds"].append({"index": index + 1, "cold_seconds": elapsed,
                                 "traces": traces, "compiled": compiled,
                                 "collections": collections,
                                 "released": _native_snapshot(gpu)})
        report["first"] = first
        output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
        assert traces == 1
    assert first["valid"] is True
    assert all(row["collections"][-1]["released"]["owner"] for row in report["rounds"])


def test_enclosing_qualification(request):
    """Full record and compiler checks isolated from construction measurements."""
    original, source_sha = _original()
    directory = Path(request.config.getoption("xmlpath")).parent
    report = {"reference_sha256": source_sha, "reset_sha256": original.reset_source_sha256,
              "controls": []}
    for reset_steps in (2, 8):
        model, inputs, options = _fixture(reset_steps)
        reference = original.canonical_linear_gaussian_kdm_auxiliary(
            model, *inputs, canonical_options=options, jit_compile=False)
        control = {"reset_steps": reset_steps, "reference": _json(reference), "arms": {}}
        nested = None
        for ownership in ("nested", "enclosing"):
            owner = _make_owner(model, inputs, options, True, ownership)
            first = owner(*inputs)
            _assert_record_equal(_public_record(first), reference)
            _assert_record_equal(owner(*inputs), first, atol=0., rtol=0.)
            if nested is None:
                nested = first
            else:
                _assert_record_equal(first, nested)
            changed = (*inputs[:4], inputs[4] + tf.constant(0.01, tf.float64), *inputs[5:])
            changed_reference = original.canonical_linear_gaussian_kdm_auxiliary(
                model, *changed, canonical_options=options, jit_compile=False)
            changed_result = owner(*changed)
            _assert_record_equal(_public_record(changed_result), changed_reference)
            invalid_inputs = (*inputs[:5], inputs[5] * 2., *inputs[6:])
            invalid = owner(*invalid_inputs)
            assert not bool(invalid["valid"].numpy())
            assert not bool(invalid["model_valid"].numpy())
            assert bool(tf.math.is_nan(invalid["kdm_auxiliary_value"]).numpy())
            derivative = None
            if reset_steps == 8:
                epsilon = 2.e-5
                plus, minus = owner(inputs[0] + epsilon, *inputs[1:]), owner(inputs[0] - epsilon, *inputs[1:])
                derivative = (plus["kdm_auxiliary_value"] - minus["kdm_auxiliary_value"]) / (2 * epsilon)
                np.testing.assert_allclose(first["kdm_auxiliary_score"].numpy()[0],
                                          derivative.numpy(), rtol=4.e-4, atol=4.e-5)
            graph = owner.get_concrete_function().graph.as_graph_def()
            nodes = [*graph.node, *(n for f in graph.library.function for n in f.node_def)]
            ops = Counter(n.op for n in nodes)
            assert not {"PyFunc", "PyFuncStateless", "EagerPyFunc"}.intersection(ops)
            assert {"While", "StatelessWhile"}.intersection(ops)
            hlo = owner.experimental_get_compiler_ir(*inputs)(stage="optimized_hlo")
            stem = f"kdm-{reset_steps}-{ownership}"
            (directory / f"{stem}.hlo").write_text(hlo)
            (directory / f"{stem}.pb").write_bytes(graph.SerializeToString())
            hlo_ops = Counter(re.findall(r"= .*? ([a-z][a-z0-9-]*)\(", hlo))
            assert hlo_ops["while"] > 0
            assert "PyFunc" not in hlo
            control["arms"][ownership] = {
                "first": _json(first), "changed": _json(changed_result),
                "changed_reference": _json(changed_reference), "invalid": _json(invalid),
                "finite_difference": _json(derivative),
                "traces": owner.experimental_get_tracing_count(), "ops": ops,
                "hlo_ops": hlo_ops, "hlo_bytes": len(hlo.encode()),
                "hlo_sha256": hashlib.sha256(hlo.encode()).hexdigest()}
            assert owner.experimental_get_tracing_count() == 1
        report["controls"].append(control)
        (directory / "kdm-enclosing-qualification.json").write_text(
            json.dumps(report, indent=2, allow_nan=False) + "\n")


@pytest.mark.parametrize("scale", [1., 1.01], ids=["base", "changed"])
def test_retained_owner_child(scale, request):
    """One frozen callback configuration per process; tensor operands may change."""
    base_model, inputs, options = _fixture()
    model = replace(base_model,
        transition_mean_fn=lambda theta, points: scale * base_model.transition_mean_fn(theta, points),
        transition_mean_tangent_fn=lambda theta, points, tangent: scale * base_model.transition_mean_tangent_fn(theta, points, tangent))
    reference_module, source_sha = _original()
    reference = reference_module.canonical_linear_gaussian_kdm_auxiliary(
        model, *inputs, canonical_options=options, jit_compile=False)
    owner = _owner(model, inputs, options, True)
    gpu = os.environ.get("BAYESFILTER_TEST_DEVICE_SCOPE") == "visible"
    begin = time.perf_counter()
    first = owner(*inputs)
    _assert_record_equal(_public_record(first), reference)
    first_json = _json(first)
    cold_seconds = time.perf_counter() - begin
    snapshots = [{"call": 0, **_native_snapshot(gpu)}]
    timings = []
    for index in range(20):
        begin = time.perf_counter()
        assert _json(owner(*inputs)) == first_json
        timings.append(time.perf_counter() - begin)
        if index in (0, 4, 19):
            snapshots.append({"call": index + 1, **_native_snapshot(gpu)})
    changed = (*inputs[:4], inputs[4] + tf.constant(0.01, tf.float64), *inputs[5:])
    changed_result = owner(*changed)
    assert _json(changed_result) != first_json
    assert _json(owner(*inputs)) == first_json
    assert owner.experimental_get_tracing_count() == 1
    output = Path(request.config.getoption("xmlpath")).parent / "kdm-retained-child.json"
    output.write_text(json.dumps({"pid": os.getpid(), "callback_scale": scale,
        "reference_sha256": source_sha, "reset_sha256": reference_module.reset_source_sha256,
        "first": first_json, "reference": _json(reference), "changed": _json(changed_result),
        "cold_seconds": cold_seconds, "warm_seconds": timings, "snapshots": snapshots,
        "traces": owner.experimental_get_tracing_count()}, indent=2, allow_nan=False) + "\n")


def test_bounded_process_lifetime(request):
    """Diagnostic harness: explicit process lifetime, no implicit API fallback."""
    directory = Path(request.config.getoption("xmlpath")).parent
    gpu = os.environ.get("BAYESFILTER_TEST_DEVICE_SCOPE") == "visible"
    reports, rounds = [], []
    before = _native_snapshot(gpu)
    for index, control in enumerate(("base", "changed", "base")):
        child_dir = directory / f"child-{index + 1}-{control}"
        child_dir.mkdir()
        command = [sys.executable, "scripts/filter_repair_test_worker.py", "-q",
            f"tests/test_filter_repair_kdm_native_memory.py::test_retained_owner_child[{control}]",
            f"--junitxml={child_dir / 'junit.xml'}"]
        begin = time.perf_counter()
        with (child_dir / "process.log").open("x") as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                     cwd=Path(__file__).resolve().parents[1])
            try:
                code = child.wait(timeout=60)
            except subprocess.TimeoutExpired:
                child.terminate()
                try:
                    child.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait(timeout=5)
                raise
        assert code == 0, (child_dir / "process.log").read_text()[-4000:]
        cases = list(ET.parse(child_dir / "junit.xml").getroot().iter("testcase"))
        assert len(cases) == 1 and not list(cases[0])
        report = json.loads((child_dir / "kdm-retained-child.json").read_text())
        assert report["pid"] == child.pid
        assert not Path(f"/proc/{child.pid}").exists()
        assert report["traces"] == 1
        reports.append(report)
        gc.collect()
        rounds.append({"index": index + 1, "command": command, "pid": child.pid,
                       "exit_code": code, "reaped": True,
                       "wall_seconds": time.perf_counter() - begin,
                       "parent": _native_snapshot(gpu),
                       "child_report": str(child_dir / "kdm-retained-child.json")})
    assert reports[0]["first"] == reports[2]["first"]
    assert reports[0]["first"] != reports[1]["first"]
    (directory / "kdm-process-lifetime.json").write_text(json.dumps({
        "role": "bounded_diagnostic_lifetime_not_automatic_runtime_wrapper",
        "before": before, "rounds": rounds,
        "same_configuration_exact_replay_across_processes": True,
        "changed_callback_matches_original": True}, indent=2, allow_nan=False) + "\n")
