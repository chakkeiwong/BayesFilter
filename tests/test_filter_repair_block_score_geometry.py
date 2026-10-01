"""Independent pinned records and complete graph/XLA block-score diagnostics."""

import importlib.util
import json
import subprocess
import sys

import numpy as np
import pytest
import tensorflow as tf

from bayesfilter.inference import block_score_geometry as candidate
from bayesfilter.inference import block_score_geometry_tf as native

D = tf.float64
BASELINE = "3582b4ac5fea67fb5da7fa60a1cbfaf19df35adf"


def _load_baseline(name):
    source = subprocess.check_output(["git", "show", f"{BASELINE}:bayesfilter/inference/{name}.py"], text=True)
    spec = importlib.util.spec_from_loader(name + "_block_score_pinned_reference", loader=None)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    exec(compile(source, spec.name, "exec"), module.__dict__)  # noqa: S102 - isolated diagnostic oracle
    return module


@pytest.fixture(scope="module")
def baseline():
    module = _load_baseline("block_score_geometry")
    geometry = _load_baseline("fixed_center_curvature")
    module.compare_precision_geometry = geometry.compare_precision_geometry
    assert module.compare_precision_geometry.__module__ == geometry.__name__
    return module


def _compare(actual, expected):
    if isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _compare(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected, strict=True):
            _compare(left, right)
    elif isinstance(expected, (str, bool, int)) or expected is None:
        assert actual == expected
    else:
        np.testing.assert_allclose(actual, expected, atol=1e-10, rtol=1e-10, equal_nan=True)


def _inputs(widths=(2, 3), replicates=3, rows=12):
    dimension = sum(widths)
    rng = np.random.default_rng(20519)
    center = np.linspace(-.2, .3, dimension)
    precision = np.diag(np.linspace(1., 4., dimension))
    partition = []
    start = 0
    for width in widths:
        precision[start:start + width, start:start + width] += .15
        partition.append((start, start + width))
        start += width
    training = rng.normal(size=(replicates, rows, dimension)) * .2
    selection = rng.normal(size=(replicates, 9, dimension)) * .2
    audit = rng.normal(size=(11, dimension)) * .2
    inputs = {"center_score_z": center, "training_offsets_z": training,
        "training_scores_z": center - training @ precision.T,
        "selection_offsets_z": selection, "selection_scores_z": center - selection @ precision.T,
        "audit_offsets_z": audit, "audit_scores_z": center - audit @ precision.T}
    return inputs, partition, precision


def _fit(module, inputs, partition, **overrides):
    options = {"ridge": 1e-12, "principal_subspace_rank": 2}
    options.update(overrides)
    return module.fit_block_diagonal_score_geometry(**inputs,
        blocks=tuple(module.ScoreGeometryBlock(str(index), *bounds) for index, bounds in enumerate(partition)),
        config=module.BlockScoreGeometryConfig(**options))


@pytest.mark.parametrize("widths,replicates", [((2, 3), 3), ((1, 2, 1), 2), ((2, 2, 2), 4)])
def test_complete_pinned_records_and_independent_block_identity(baseline, widths, replicates):
    inputs, partition, precision = _inputs(widths, replicates)
    before, after = (_fit(module, inputs, partition) for module in (baseline, candidate))
    assert before.accepted and after.accepted
    _compare(after.payload(include_matrices=True), before.payload(include_matrices=True))
    np.testing.assert_allclose(after.precision_z, precision, atol=1e-10, rtol=1e-10)
    np.testing.assert_allclose(after.precision_z @ after.covariance_z, np.eye(sum(widths)), atol=1e-10)
    scale = np.linspace(.5, 2., sum(widths))
    physical = after.position_geometry(scale)
    _compare(physical, before.position_geometry(scale))
    np.testing.assert_allclose(physical["factor"] @ tf.transpose(physical["factor"]),
        np.diag(scale) @ np.linalg.inv(precision) @ np.diag(scale), atol=1e-10, rtol=1e-10)


@pytest.mark.parametrize("field", ["center_score_z", "training_offsets_z", "training_scores_z",
    "selection_offsets_z", "selection_scores_z", "audit_offsets_z", "audit_scores_z"])
def test_nonfinite_inputs_reject_before_any_block(baseline, field):
    inputs, partition, _ = _inputs()
    inputs[field].flat[-1] = np.nan if "scores" in field else np.inf
    before, after = (_fit(module, inputs, partition) for module in (baseline, candidate))
    assert after.status == "nonfinite_fit_inputs"
    assert after.diagnostics == {}
    _compare(after.payload(include_matrices=True), before.payload(include_matrices=True))


@pytest.mark.parametrize("failure,replicate,block_index", [
    ("rank", 0, 0), ("rank", 1, 1), ("not_spd", 0, 1), ("not_spd", 2, 1),
    ("condition", 1, 0), ("condition", 2, 1),
])
def test_first_failure_preserves_replicate_and_block_truncation(baseline, failure, replicate, block_index):
    inputs, partition, precision = _inputs()
    start, stop = partition[block_index]
    if failure == "rank":
        inputs["training_offsets_z"][replicate, :, start:stop] = 0.
    else:
        precision[start:stop, start:stop] = np.diag(np.linspace(1., 2., stop - start))
        precision[start, start] = -1. if failure == "not_spd" else .0001
        inputs["training_scores_z"][replicate] = inputs["center_score_z"] - inputs["training_offsets_z"][replicate] @ precision.T
    options = {"max_condition_number": 100.}
    before, after = (_fit(module, inputs, partition, **options) for module in (baseline, candidate))
    assert after.status == {"rank": "block_design_rank_deficient", "not_spd": "raw_block_precision_not_spd",
        "condition": "block_condition_number_rejected"}[failure]
    reports = after.diagnostics["replicates"]
    assert len(reports) == replicate + 1
    assert len(reports[-1]["blocks"]) == block_index + 1
    assert "selection_relative_rmse" not in reports[-1]
    _compare(after.payload(include_matrices=True), before.payload(include_matrices=True))


@pytest.mark.parametrize("case,status", [
    ("selection", "selection_score_fit_rejected"),
    ("stability", "replicate_stability_rejected"),
    ("selection_and_stability", "selection_score_fit_rejected"),
    ("audit", "audit_score_fit_rejected"),
    ("offblock", "offblock_curvature_rejected"),
])
def test_qualification_and_precedence_preserve_complete_records(baseline, case, status):
    inputs, partition, precision = _inputs()
    options = {}
    if "stability" in case:
        for index, scale in enumerate((.7, 1., 1.3)):
            inputs["training_scores_z"][index] = inputs["center_score_z"] - inputs["training_offsets_z"][index] @ (scale * precision).T
            inputs["selection_scores_z"][index] = inputs["center_score_z"] - inputs["selection_offsets_z"][index] @ (scale * precision).T
        options["selection_relative_rmse_cap"] = 1. if case == "stability" else .001
    elif case == "selection":
        inputs["selection_scores_z"] += .4
        inputs["audit_scores_z"] += .4  # Selection must reject before audit.
    else:
        inputs["audit_scores_z"] += .4
        if case == "offblock":
            options["audit_relative_rmse_cap"] = 2.
    before, after = (_fit(module, inputs, partition, **options) for module in (baseline, candidate))
    assert before.status == after.status == status
    _compare(after.payload(include_matrices=True), before.payload(include_matrices=True))
    assert after.precision_z is not None and after.covariance_z is not None


@pytest.mark.parametrize("metric,field", [
    ("generalized_eigenvalue_spread", "generalized_eigenvalue_spread_cap"),
    ("trace_normalized_frobenius", "trace_normalized_frobenius_cap"),
    ("trace_normalized_operator", "trace_normalized_operator_cap"),
    ("maximum_principal_angle_degrees", "principal_angle_degrees_cap"),
])
def test_stability_caps_preserve_both_sides_of_boundary(baseline, metric, field):
    inputs, partition, precision = _inputs(replicates=2)
    rotation = np.eye(5)
    rotation[-2:, -2:] = [[np.cos(.2), -np.sin(.2)], [np.sin(.2), np.cos(.2)]]
    rotated = rotation @ precision @ rotation.T
    inputs["training_scores_z"][1] = inputs["center_score_z"] - inputs["training_offsets_z"][1] @ rotated.T
    inputs["selection_scores_z"][1] = inputs["center_score_z"] - inputs["selection_offsets_z"][1] @ rotated.T
    options = {"selection_relative_rmse_cap": 2., "audit_relative_rmse_cap": 2.,
        "unexplained_response_fraction_cap": 2., "principal_subspace_rank": 1,
        "trace_normalized_frobenius_cap": 1., "trace_normalized_operator_cap": 1.,
        "principal_angle_degrees_cap": 90.}
    reference = _fit(baseline, inputs, partition, **options)
    assert reference.accepted
    pair = reference.diagnostics["stability"]["comparisons"][0]["metrics"]
    boundary = pair["generalized_eigenvalues"]["spread"] if metric == "generalized_eigenvalue_spread" else pair[metric]
    assert boundary > 0.
    for factor, accepted in ((1. - 1e-6, False), (1. + 1e-6, True)):
        limited = {**options, field: boundary * factor}
        before, after = (_fit(module, inputs, partition, **limited) for module in (baseline, candidate))
        assert before.accepted == after.accepted == accepted
        _compare(after.payload(include_matrices=True), before.payload(include_matrices=True))


@pytest.mark.parametrize("failed_fit", [False, True])
def test_dimension_one_keeps_rank_exception_after_fit_only(baseline, failed_fit):
    inputs, partition, _ = _inputs((1,), 2)
    if failed_fit:
        inputs["training_offsets_z"] *= 0.
    for module in (baseline, candidate):
        if failed_fit:
            assert _fit(module, inputs, partition).status == "block_design_rank_deficient"
        else:
            with pytest.raises(ValueError, match="subspace_rank must lie"):
                _fit(module, inputs, partition)


@pytest.mark.parametrize("empty", ["training", "selection", "audit"])
def test_empty_rows_keep_existing_decisions_and_reports(baseline, empty):
    inputs, partition, _ = _inputs()
    axis = 0 if empty == "audit" else 1
    for suffix in ("offsets_z", "scores_z"):
        name = empty + "_" + suffix
        inputs[name] = np.take(inputs[name], [], axis=axis)
    before, after = (_fit(module, inputs, partition) for module in (baseline, candidate))
    _compare(after.payload(include_matrices=True), before.payload(include_matrices=True))


@pytest.mark.parametrize("scale", [[1., 1.], [1., 1., 0., 1., 1.], [1., 1., -1., 1., 1.],
    [1., 1., np.nan, 1., 1.], [1., 1., np.inf, 1., 1.]])
def test_coordinate_scale_rejections_remain_explicit(baseline, scale):
    inputs, partition, _ = _inputs()
    for module in (baseline, candidate):
        result = _fit(module, inputs, partition)
        with pytest.raises(ValueError, match="scale"):
            result.position_geometry(scale)


def _native_inputs(inputs):
    names = ("center_score_z", "training_offsets_z", "training_scores_z", "selection_offsets_z",
        "selection_scores_z", "audit_offsets_z", "audit_scores_z")
    config = candidate.BlockScoreGeometryConfig(ridge=1e-12, principal_subspace_rank=2)
    return (*tuple(tf.constant(inputs[name], D) for name in names),
        tf.constant([getattr(config, name) for name in native.CONTROL_FIELDS], D),
        tf.constant(config.principal_subspace_rank, tf.int32))


def _nodes(program):
    graph = program.get_concrete_function().graph.as_graph_def()
    return list(graph.node) + [node for function in graph.library.function for node in function.node_def], graph


def test_complete_graph_and_xla_have_bounded_loops_and_no_hidden_xla():
    counts = []
    for widths, replicates in (((2, 2), 2), ((2, 2, 2, 2), 4)):
        inputs, partition, _ = _inputs(widths, replicates)
        arguments = _native_inputs(inputs)
        programs = [native.fit_program(sum(widths), replicates, 12, 9, 11, tuple(partition),
            jit_compile=jit) for jit in (False, True)]
        results = [program(*arguments) for program in programs]
        for key in results[0]:
            np.testing.assert_allclose(results[1][key], results[0][key], rtol=1e-10, atol=1e-10)
        assert results[1]["status"] == 0
        assert "HloModule" in programs[1].experimental_get_compiler_ir(*arguments)(stage="hlo")
        size_counts = []
        for program in programs:
            nodes, graph = _nodes(program)
            assert not any(node.op in {"PyFunc", "EagerPyFunc", "NumpyFunction"} for node in nodes)
            assert sum(node.op in {"While", "StatelessWhile"} for node in nodes) >= 4
            assert program.experimental_get_tracing_count() == 1
            size_counts.append(len(nodes))
        _, graph = _nodes(programs[0])
        assert not any(function.attr.get("_XlaMustCompile", False).b for function in graph.library.function
            if "_XlaMustCompile" in function.attr)
        counts.append(size_counts)
    np.testing.assert_array_less(np.abs(np.subtract(counts[1], counts[0])), 30)


def test_pair_eigensystem_localization():
    from bayesfilter.inference.mass_matrix_tf import _eigh

    _, _, matrix = _inputs(replicates=2)
    rotation = np.eye(5)
    rotation[-2:, -2:] = [[np.cos(.2), -np.sin(.2)], [np.sin(.2), np.cos(.2)]]
    first, second = tf.constant(matrix, D), tf.constant(rotation @ matrix @ rotation.T, D)

    def inspect(eigensystem):
        def compute(left, right):
            lv, lq = eigensystem(left)
            rv, rq = eigensystem(right)
            overlap = tf.matmul(lq[:, -1:], rq[:, -1:], transpose_a=True)
            angle = tf.acos(tf.clip_by_value(tf.abs(overlap[0, 0]), -1., 1.)) * (180. / np.pi)
            return tf.stack([angle, tf.reduce_max(tf.abs(left @ lq - lq * lv)),
                tf.reduce_max(tf.abs(right @ rq - rq * rv))])
        return compute

    plain = inspect(tf.linalg.eigh)
    signature = [tf.TensorSpec([5, 5], D), tf.TensorSpec([5, 5], D)]
    raw = tf.function(plain, input_signature=signature, autograph=False, jit_compile=True)
    refined = tf.function(inspect(_eigh), input_signature=signature, autograph=False, jit_compile=True)
    result = {"eager_angle_and_residuals": plain(first, second).numpy().tolist(),
        "xla_angle_and_residuals": raw(first, second).numpy().tolist(),
        "refined_angle_and_residuals": refined(first, second).numpy().tolist()}
    print(json.dumps(result, sort_keys=True))
