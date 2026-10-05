"""Complete compact/padded structured fit and enclosing preparation checks."""

import gc
import importlib.util
import inspect
import json
import re
import subprocess
import sys
import weakref
from pathlib import Path

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.inference import factor_correlation_geometry as factor
from bayesfilter.inference import sequential_map_covariance as sequential
from bayesfilter.inference import sequential_structured_fit_tf as native
from bayesfilter.inference.sequential_structured_preparation_tf import (
    structured_data_program,
)
from tests.test_filter_repair_fixed_stability import _compare
from tests.test_filter_repair_structured_preparation import (
    _arguments,
    _before,
    _public_data,
)
from tests.test_filter_repair_structured_preparation import (
    baseline as _baseline,
)

D = tf.float64


@pytest.fixture(scope="module")
def frozen():
    baseline = _baseline.__wrapped__()
    source = subprocess.check_output(["git", "show",
        "f06fd505:bayesfilter/inference/factor_correlation_geometry.py"], text=True)
    spec = importlib.util.spec_from_loader("structured_fit_frozen_factor", loader=None)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    exec(compile(source, spec.name, "exec"), module.__dict__)  # noqa: S102 - frozen diagnostic oracle
    baseline.fit_factor_correlation_score_geometry = module.fit_factor_correlation_score_geometry
    baseline.FactorCorrelationGeometryConfig = module.FactorCorrelationGeometryConfig
    return baseline


def _prepared_arguments(data):
    return (data["center_score_z"], data["training_offsets_z"], data["training_scores_z"],
        data["holdout_offsets_z"], data["holdout_scores_z"], data["training_weights"],
        data["active_training_rows"])


@pytest.mark.parametrize("dimension,capacity,factors", [(3, 4, 1), (5, 32, 2)])
@pytest.mark.parametrize("occupancy", ["none", "partial", "three", "full"])
def test_complete_structured_records_match_frozen(frozen, dimension, capacity, factors, occupancy, request):
    scalar, batched, args = _arguments(dimension, capacity, occupancy)
    original, before_count = _before(frozen, scalar, batched, args, 4 * dimension, True)
    prepared, after_count = _before(sequential, scalar, batched, args, 4 * dimension, True)
    assert before_count == after_count
    active = int(prepared["_native_factor_data"]["active_training_rows"])
    compact_view = {name: value for name, value in prepared.items() if name != "_native_factor_data"}
    for name in ("training_offsets_z", "training_scores_z", "training_weights"):
        assert compact_view[name].shape[0] == active
    _compare(compact_view, original)
    cfg = sequential.SequentialMapCovarianceConfig()
    before = frozen._fit_factor_from_data(original, factor_count=factors, config=cfg)
    after = sequential._fit_factor_from_data(prepared, factor_count=factors, config=cfg)
    path = Path(request.config.getoption("xmlpath")).parent / f"structured-fit-{dimension}-{occupancy}.json"
    with path.open("x") as handle:
        json.dump({"dimension": dimension, "capacity": capacity, "occupancy": occupancy,
            "before": before, "after": after}, handle, indent=2)
    if dimension == 3:
        assert before["covariance_z"] is not None and after["covariance_z"] is not None
    else:
        assert before["status"] == after["status"] == "factor_optimizer_failed"
        assert before["covariance_z"] is None and after["covariance_z"] is None
        # As qualified in the guard repair, counts describe the continued
        # already-rejected trajectory; every pre-existing rejection field agrees.
        assert before["diagnostics"].pop("invalid_covariance_evaluations") > 0
        assert after["diagnostics"].pop("invalid_covariance_evaluations") > 0
    _compare(after, before)
    if dimension == 3:
        assert after["diagnostics"]["training_row_count"] == active


@pytest.mark.parametrize("fault", ["center", "training", "holdout", "weights", "count"])
def test_nonfinite_inputs_reject_before_optimizer(fault):
    scalar, batched, args = _arguments(3, 4, "none")
    data = structured_data_program(scalar, batched, 3, 12, 4, True)(*args)
    arguments = list(_prepared_arguments(data))
    if fault == "center":
        arguments[0] = tf.fill([3], tf.constant(float("nan"), D))
    elif fault == "training":
        arguments[2] = tf.tensor_scatter_nd_update(arguments[2], [[0, 1]], [float("nan")])
    elif fault == "holdout":
        arguments[4] = tf.fill([6, 3], tf.constant(float("nan"), D))
    elif fault == "weights":
        arguments[5] = tf.zeros_like(arguments[5])
    else:
        arguments[6] = tf.constant(11)
    result = native.structured_fit_data_program(3, 10, 6, factor.FactorCorrelationGeometryConfig())(*arguments)
    assert int(result["input_status"]) == (3 if fault == "weights" else 4 if fault == "count" else 2)
    assert int(result["fit"]["optimizer"].num_objective_evaluations) == 0
    assert not bool(result["fit"]["finite"])


def test_dimension_rejection_matches_public_record(frozen):
    scalar, batched, args = _arguments(2, 4, "none")
    old, _ = _before(frozen, scalar, batched, args, 8, True)
    new, _ = _before(sequential, scalar, batched, args, 8, True)
    cfg = sequential.SequentialMapCovarianceConfig()
    _compare(sequential._fit_factor_from_data(new, factor_count=2, config=cfg),
        frozen._fit_factor_from_data(old, factor_count=2, config=cfg))


def test_structured_preparation_arithmetic_localization(frozen, request):
    from tests.test_filter_repair_initializer_rounding import _record_differences

    reports = []
    for occupancy in ("partial", "full"):
        scalar, batched, args = _arguments(3, 4, occupancy)
        before, _ = _before(frozen, scalar, batched, args, 12, True)
        prepared, _ = _before(sequential, scalar, batched, args, 12, True)
        active = int(prepared["_native_factor_data"]["active_training_rows"])
        original_padded = {**before, "_native_factor_data": {**before, "active_training_rows": tf.constant(active)}}
        differences = {}
        for name in ("center_score_z", "training_offsets_z", "training_scores_z",
                "holdout_offsets_z", "holdout_scores_z", "training_weights"):
            old, new = before[name], prepared[name]
            if name in ("training_offsets_z", "training_scores_z", "training_weights"):
                new = new[:active]
                original_padded["_native_factor_data"][name] = tf.pad(old, [[0, 10 - active]] + [[0, 0]] * (old.shape.rank - 1))
            differences[name] = {"maximum_error": float(tf.reduce_max(tf.abs(old - new))),
                "before": old.numpy().tolist(), "after": new.numpy().tolist()}
        cfg = sequential.SequentialMapCovarianceConfig()
        compact_fit = frozen._fit_factor_from_data(before, factor_count=1, config=cfg)
        unchanged_data_fit = sequential._fit_factor_from_data(original_padded, factor_count=1, config=cfg)
        prepared_fit = sequential._fit_factor_from_data(prepared, factor_count=1, config=cfg)
        reports.append({"occupancy": occupancy, "preparation": differences,
            "same_data_padding": _record_differences(unchanged_data_fit, compact_fit),
            "enclosed_preparation": _record_differences(prepared_fit, unchanged_data_fit)})
    path = Path(request.config.getoption("xmlpath")).parent / "structured-arithmetic.json"
    with path.open("x") as handle:
        json.dump(reports, handle, indent=2)


def test_preparation_stage_localization(frozen, request):
    from bayesfilter.inference import (
        sequential_structured_preparation_tf as preparation,
    )
    from bayesfilter.inference.sequential_preparation_tf import evaluation_program

    source = inspect.getsource(preparation).replace("return {'center_score_z':",
        "return {'stage_positions': positions, 'stage_scores': scores, 'center_score_z':")
    namespace = {}
    exec(compile(source, "structured_stage_diagnostic", "exec"), namespace)  # noqa: S102 - instrumentation only
    scalar, batched, args = _arguments(3, 4, "partial")
    before, _ = _before(frozen, scalar, batched, args, 12, True)
    after = namespace["structured_data_program"](scalar, batched, 3, 12, 4, True)(*args)
    cloud = tf.concat([before["training_offsets_z"][:6], before["holdout_offsets_z"]], 0)
    old_positions = args[0][None, :] + cloud * args[2][None, :]
    standalone = evaluation_program(scalar, batched, 12, 3)
    old_scores = standalone(old_positions)[1]
    candidate_positions_scores = standalone(after["stage_positions"])[1]
    expected_scores = tf.concat([before["training_scores_z"][:6], before["holdout_scores_z"]], 0)
    candidate_scores = tf.concat([after["training_scores_z"][:6], after["holdout_scores_z"]], 0)

    def difference(left, right):
        return {"maximum_error": float(tf.reduce_max(tf.abs(left - right))),
            "before": left.numpy().tolist(), "after": right.numpy().tolist()}

    result = {"positions": difference(old_positions, after["stage_positions"]),
        "callback_same_positions": difference(candidate_positions_scores, after["stage_scores"]),
        "callback_position_effect": difference(old_scores, candidate_positions_scores),
        "score_scaling": difference(after["stage_scores"] * args[2][None, :], candidate_scores),
        "total": difference(expected_scores, candidate_scores)}
    path = Path(request.config.getoption("xmlpath")).parent / "structured-preparation-stages.json"
    with path.open("x") as handle:
        json.dump(result, handle, indent=2)


def test_enclosing_preparation_fit_has_runtime_inputs_and_frozen_geometry(request):
    scalar, batched, first = _arguments(3, 4, "none")
    prepare = structured_data_program(scalar, batched, 3, 12, 4, True)
    cfg = factor.FactorCorrelationGeometryConfig(max_iterations=4)
    # Prebind outside tracing; the factory must still bind ownership to this consumer.
    native.structured_fit_data_program(3, 10, 6, cfg)

    @tf.function(input_signature=prepare.input_signature, jit_compile=True, autograph=False)
    def enclosing(*args):
        data = prepare.python_function(*args)
        fit = native.structured_fit_data_program(3, 10, 6, cfg).python_function(*_prepared_arguments(data))
        return data, fit

    with tf.GradientTape() as tape:
        tape.watch(first[0])
        data, result = enclosing(*first)
        objective = tf.reduce_sum(result["fit"]["precision"])
    gradient = tape.gradient(objective, first[0])
    if gradient is not None:
        np.testing.assert_array_equal(gradient, 0.)

    @tf.function(input_signature=prepare.input_signature, jit_compile=True, autograph=False)
    def geometry_only(*args):
        data = prepare.python_function(*args)
        return native.structured_fit_data_program(3, 10, 6, cfg).python_function(*_prepared_arguments(data))

    with tf.GradientTape() as tape:
        tape.watch(first[0])
        geometry = geometry_only(*first)
        objective = tf.reduce_sum(geometry["fit"]["precision"])
    assert tape.gradient(objective, first[0]) is None
    del geometry_only
    assert int(result["input_status"]) == 0
    assert bool(result["fit"]["finite"])
    hlo = enclosing.experimental_get_compiler_ir(*first)(stage="hlo")
    path = Path(request.config.getoption("xmlpath")).parent / "structured-complete-hlo.txt"
    with path.open("x") as handle:
        handle.write(hlo)
    entry = hlo[hlo.rfind("\nENTRY "):]
    assert sorted(int(index) for index in re.findall(r"\bparameter\((\d+)\)", entry)) == list(range(8))
    for occupancy in ("partial", "full"):
        _, _, args = _arguments(3, 4, occupancy)
        data, actual = enclosing(*args)
        separate_data = prepare(*args)
        separate = native.structured_fit_data_program(3, 10, 6, cfg)(*_prepared_arguments(separate_data))
        _compare(_public_data(data), _public_data(separate_data))
        _compare(tf.nest.map_structure(lambda value: value.numpy().tolist(), actual),
            tf.nest.map_structure(lambda value: value.numpy().tolist(), separate))
        assert enclosing.experimental_get_compiler_ir(*args)(stage="hlo") == hlo
    assert enclosing.experimental_get_tracing_count() == 1
    graph = enclosing.get_concrete_function().graph
    state = graph.variables[0]
    ref, graph_ref = weakref.ref(state), weakref.ref(graph)
    factor._make_factor_program.cache_clear()
    native._cached_structured_fit.cache_clear()
    del state
    gc.collect()
    assert ref() is not None
    repeated = enclosing(*first)[1]
    np.testing.assert_array_equal(repeated["fit"]["precision"], result["fit"]["precision"])
    del graph, enclosing, tape
    gc.collect()
    assert graph_ref() is None and ref() is None

def _healthy_arguments(dimension, capacity, occupancy):
    # Independent explicit factor covariance: no runtime fitter helper constructs it.
    scales = np.array([.8, 1.2, 1.4] if dimension == 3 else [.8, 1.1, .9, 1.2, .7])
    loadings = np.array([[.2], [-.1], [.3]] if dimension == 3 else
        [[.3, 0.], [.12, .25], [-.2, .1], [.15, -.1], [.08, .2]])
    covariance = scales[:, None] * (np.diag(1. - np.sum(loadings ** 2, axis=1))
        + loadings @ loadings.T) * scales[None, :]
    frozen_precision = np.linalg.inv(covariance)

    def batched(rows):
        score = -(rows @ tf.constant(frozen_precision, D))
        return .5 * tf.reduce_sum(rows * score, axis=1), score

    def scalar(row):
        value, score = batched(row[None, :])
        return value[0], score[0]

    _, _, args = _arguments(dimension, capacity, occupancy)
    args = (args[0], scalar(args[0])[1], *args[2:6], batched(args[5])[1])
    return scalar, batched, args


@pytest.mark.parametrize("dimension,capacity,factors", [(3, 4, 1), (5, 32, 2)])
@pytest.mark.parametrize("occupancy", ["none", "partial", "three", "full"])
def test_healthy_complete_records_match_frozen(frozen, dimension, capacity, factors, occupancy, request):
    scalar, batched, args = _healthy_arguments(dimension, capacity, occupancy)
    old, _ = _before(frozen, scalar, batched, args, 4 * dimension, True)
    new, _ = _before(sequential, scalar, batched, args, 4 * dimension, True)
    cfg = sequential.SequentialMapCovarianceConfig()
    before = frozen._fit_factor_from_data(old, factor_count=factors, config=cfg)
    after = sequential._fit_factor_from_data(new, factor_count=factors, config=cfg)
    path = Path(request.config.getoption("xmlpath")).parent / f"structured-healthy-{dimension}-{occupancy}.json"
    with path.open("x") as handle:
        json.dump({"before": before, "after": after}, handle, indent=2)
    assert before["status"] == after["status"] == "usable"
    assert before["covariance_z"] is not None and after["covariance_z"] is not None
    _compare(after, before)
