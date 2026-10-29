"""Diagnostic frozen-source authorities for LEDH streaming-memory execution.

NumPy, Git-loaded historical owners and Python fixture loops are independent
test/reporting code only. They confer no canonical or scientific admission.
"""

import hashlib
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.highdim.ledh_canonical_filter_tf import (
    make_seeded_canonical_value_program,
)
from bayesfilter.highdim.ledh_canonical_value_program_tf import (
    make_canonical_value_program,
)
from bayesfilter.ops.ledh_random_compat_tf import integer_seed_words
from tests.test_filter_repair_ledh_seeded_cost import _callbacks_with_frozen_constants
from tests.test_filter_repair_ledh_seeded_public import (
    _fixture,
    _reference_generator,
    _reference_inputs,
)
from tests.test_filter_repair_ledh_seeded_public import authorities as _authorities
from tests.test_filter_repair_ledh_value_native import _compare, _frozen_reference

ROOT = Path(__file__).resolve().parents[1]
BUFFERED_COMMIT = "c7c0b88c2"


@pytest.fixture(scope="module")
def authorities():
    yield from _authorities.__wrapped__()


@pytest.fixture(scope="module")
def buffered_authority():
    modules, sources = {}, {}
    for short in ("ledh_canonical_value_program_tf", "ledh_canonical_filter_tf"):
        path = f"bayesfilter/highdim/{short}.py"
        source = subprocess.check_output(
            ["git", "show", f"{BUFFERED_COMMIT}:{path}"], cwd=ROOT, text=True)
        module = ModuleType(f"_frozen_streaming_comparator_{short}")
        sys.modules[module.__name__] = module
        exec(compile(source, f"<diagnostic-{short}>", "exec"), module.__dict__)  # noqa: S102 -- fixed Git reference only
        modules[short] = module
        sources[path] = source
    owner = modules["ledh_canonical_filter_tf"]
    owner.make_canonical_value_program = modules["ledh_canonical_value_program_tf"].make_canonical_value_program
    yield owner, sources
    for module in modules.values():
        sys.modules.pop(module.__name__, None)


def _write(request, name, record):
    directory = Path(request.config.getoption("xmlpath")).parent
    (directory / f"{name}.json").write_text(json.dumps(record, indent=2) + "\n")
    return directory


def _numeric(result):
    return {key: value.numpy().tolist() for key, value in result.items()}


class _TrackedGenerator:
    def __init__(self, seed):
        self.generator = _reference_generator(seed)
        self.draw_calls = []

    def normal(self, shape, dtype):
        self.draw_calls.append(tuple(shape))
        return self.generator.normal(shape, dtype=dtype)


def _graph_evidence(owner, args, directory, label, horizon, count):
    graph = owner.get_concrete_function().graph.as_graph_def(add_shapes=True)
    nodes = [("outer", node) for node in graph.node]
    nodes.extend((fn.signature.name, node) for fn in graph.library.function for node in fn.node_def)
    operations = {node.op for _, node in nodes}
    assert not operations & {"PyFunc", "EagerPyFunc", "PyFuncStateless", "XlaHostCompute"}
    buffer_nodes = []
    for scope, node in nodes:
        for shape in node.attr["_output_shapes"].list.shape:
            dims = [dim.size for dim in shape.dim]
            if dims == [horizon, count, 2]:
                buffer_nodes.append({"scope": scope, "name": node.name, "op": node.op, "shape": dims})
    evidence = {"process_buffer_shape_nodes": buffer_nodes,
                "trace_count": owner.experimental_get_tracing_count(),
                "no_host_callbacks": True, "hlo": {}}
    for stage in ("hlo", "optimized_hlo"):
        hlo = owner.experimental_get_compiler_ir(*args)(stage=stage)
        path = directory / f"{label}-{stage}.txt"
        path.write_text(hlo)
        evidence["hlo"][stage] = {"sha256": hashlib.sha256(hlo.encode()).hexdigest(),
            "path": str(path), "process_shape_occurrences": hlo.count(f"f64[{horizon},{count},2]")}
    return evidence


@pytest.mark.parametrize("case", ["composed", "annealed", "dual_trust"])
def test_freeze_buffered_owner(authorities, buffered_authority, case, request):
    baseline, callbacks, observations, controls, _ = _fixture(authorities, case)
    frozen, sources = buffered_authority
    spec = tf.TensorSpec(observations.shape, observations.dtype)
    owner = frozen.make_seeded_canonical_value_program(callbacks, spec, particle_count=8, **controls)
    records = []
    for seed, shift in ((123, 0.), (124, .1)):
        args = (observations + shift, tf.constant([seed], tf.uint32), tf.constant([seed - 106], tf.uint32))
        actual = owner(*args)
        expected = _frozen_reference(baseline, callbacks, args[0],
            _reference_inputs(seed, seed - 106, tf.float64, stages=controls["temper_stages"]), controls)
        records.append({"seed": seed, "observation_shift": shift, "record": _numeric(actual),
                        "eager_comparison": _compare(actual, expected)})
    report = {"baseline_commit": BUFFERED_COMMIT, "case": case, "records": records,
              "device": actual["value"].device, "source_sha256": {
                  path: hashlib.sha256(source.encode()).hexdigest() for path, source in sources.items()}}
    directory = _write(request, f"buffered-freeze-{case}", report)
    for path, source in sources.items():
        (directory / Path(path).name).write_text(source)
    report["graph"] = _graph_evidence(owner, args, directory, case, 3, 8)
    assert report["graph"]["process_buffer_shape_nodes"]
    assert report["graph"]["hlo"]["optimized_hlo"]["process_shape_occurrences"] > 0
    _write(request, f"buffered-freeze-{case}", report)


def _records_compare(actual, expected):
    """Complete shared record: unchanged healthy gate; rejected numerics diagnostic."""
    healthy = bool(expected["program_valid"])
    errors = {}
    for key, reference in expected.items():
        value = actual[key]
        if value.dtype.is_floating:
            lhs, rhs = value.numpy(), reference.numpy()
            np.testing.assert_array_equal(np.isnan(lhs), np.isnan(rhs), err_msg=key)
            if (healthy and key != "per_step_reset_scaled_system_condition") or key == "value":
                np.testing.assert_allclose(lhs, rhs, atol=1e-6, rtol=1e-6, equal_nan=True, err_msg=key)
            finite = np.isfinite(lhs) & np.isfinite(rhs)
            errors[key] = float(np.max(np.abs(lhs[finite] - rhs[finite]))) if np.any(finite) else None
        else:
            np.testing.assert_array_equal(value, reference, err_msg=key)
    return {"healthy": healthy, "max_absolute_errors": errors}


@pytest.mark.parametrize("case", ["composed", "annealed", "dual_trust", "invalid_initial",
    "invalid_prediction", "first_invalid_prediction", "time_dependent", "large_seeds"])
def test_streamed_seeded_owner(authorities, buffered_authority, case, request):
    fixture_case = "annealed" if case == "large_seeds" else case
    baseline, callbacks, observations, controls, _ = _fixture(authorities, fixture_case)
    if case == "first_invalid_prediction":
        transition = callbacks.transition_mean_fn
        callbacks = replace(callbacks, transition_mean_fn=lambda points, time: tf.where(
            time == 0, tf.fill(tf.shape(points), tf.constant(float("nan"), tf.float64)),
            transition(points, time)))
    elif case == "time_dependent":
        transition, observe = callbacks.transition_mean_fn, callbacks.observation_fn
        callbacks = replace(callbacks,
            transition_mean_fn=lambda points, time: transition(points, time) + tf.cast(time, tf.float64) * .01,
            observation_fn=lambda points, time: observe(points, time) + tf.cast(time, tf.float64) * .02)
        callbacks = _callbacks_with_frozen_constants(callbacks)
    frozen, _ = buffered_authority
    word_count = 5 if case == "large_seeds" else 1
    spec = tf.TensorSpec(observations.shape, observations.dtype)
    owner = make_seeded_canonical_value_program(callbacks, spec, particle_count=8,
        seed_word_count=word_count, resample_seed_word_count=word_count, **controls)
    buffered = frozen.make_seeded_canonical_value_program(callbacks, spec, particle_count=8,
        seed_word_count=word_count, resample_seed_word_count=word_count, **controls)
    supplied = make_canonical_value_program(callbacks, spec, particle_count=8, **controls)
    records = []
    for seed, shift in ((123, 0.), (124, .1)):
        if case == "large_seeds":
            seed = 2**160 - 1 if seed == 123 else 2**128 + 42
            resample_seed = seed
        else:
            resample_seed = seed - 106
        args = (observations + shift, tf.constant(integer_seed_words(seed), tf.uint32),
                tf.constant(integer_seed_words(resample_seed), tf.uint32))
        inputs = _reference_inputs(seed, resample_seed, tf.float64, stages=controls["temper_stages"])
        actual, previous = owner(*args), buffered(*args)
        record = {"seed": seed, "resample_seed": resample_seed, "observation_shift": shift,
                  "actual": _numeric(actual), "buffered": _numeric(previous)}
        records.append(record)
        _write(request, f"streamed-{case}", {"records": records})
        record["buffered_comparison"] = _records_compare(actual, previous)
        record["supplied_comparison"] = _records_compare(actual, supplied(args[0], *inputs))
        expected = _frozen_reference(baseline, callbacks, args[0], inputs, controls)
        record["eager_comparison"] = _compare(actual, expected)

        # Observe the original stateful generator's calls across actual gates,
        # independently of the native owner's completed counter.
        tracked = _TrackedGenerator(seed)
        baseline.np = np
        baseline._replication_generator = lambda seed, tracked=tracked: tracked
        baseline.canonical_value_and_diagnostics(callbacks, args[0], particle_count=8,
            seed=seed, resample_seed=resample_seed, **controls)
        consumed = len(tracked.draw_calls) - 1
        assert int(actual["process_draws_consumed"]) == consumed
        np.testing.assert_array_equal(actual["final_philox_state"], tf.bitcast(tracked.generator.state, tf.uint64))
        assert int(actual["resampling_draws_consumed"]) == consumed * (
            controls["temper_stages"] if controls["annealed_resampling"] else 0)
        if case in {"invalid_initial", "first_invalid_prediction"}:
            assert consumed == 0
        if case == "invalid_prediction":
            assert consumed == 1
        record["original_process_draws"] = consumed
        record["exact_final_philox_state"] = True
        for key, value in owner(*args).items():
            np.testing.assert_array_equal(value, actual[key], err_msg=key)
    directory = _write(request, f"streamed-{case}", {"records": records})
    graph = _graph_evidence(owner, args, directory, case, 3, 8)
    assert not graph["process_buffer_shape_nodes"]
    assert graph["hlo"]["optimized_hlo"]["process_shape_occurrences"] == 0
    assert graph["trace_count"] == 1
    _write(request, f"streamed-{case}", {"records": records, "graph": graph,
        "device": actual["value"].device, "baseline_commit": BUFFERED_COMMIT})
