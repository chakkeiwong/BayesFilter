"""Prepared movement recurrence against exact pinned host-loop statements."""

import ast
import dataclasses
import hashlib
import io
import subprocess
import tarfile
from functools import lru_cache
from pathlib import Path

import pytest
import tensorflow as tf

from bayesfilter.inference import posterior_local_initializer as original
from bayesfilter.inference.posterior_movement_tf import (
    STATUSES,
    make_posterior_movement_program,
)
from bayesfilter.inference.program_cache_scope import ProgramCacheScope
from bayesfilter.inference.quadratic_geometry import LowRankSPDQuadraticGeometryConfig
from bayesfilter.inference.quadratic_geometry_fit_tf import SOURCE_ROLES
from bayesfilter.inference.quadratic_geometry_fit_tf import STATUSES as FIT_STATUSES
from bayesfilter.inference.quadratic_geometry_full_report import geometry_result
from bayesfilter.inference.quadratic_geometry_full_tf import (
    STAGES,
    make_geometry_program,
    prepare_geometry_inputs,
)
from tests.filter_repair_frozen_checkpoint import FrozenCheckpoint
from tests.test_filter_repair_block_capture import stable_hlo
from tests.test_filter_repair_geometry_control import clean, save
from tests.test_filter_repair_quadratic_batches import _equal_records

D = tf.float64
REVISION = "031692a0b"
FROZEN_GRAPH_DEPENDENCIES = {
    "bayesfilter/inference/fixed_center_fitting_tf.py",
    "bayesfilter/inference/fixed_center_selection_tf.py",
    "bayesfilter/inference/fixed_center_stability_tf.py",
}


@lru_cache(maxsize=1)
def frozen_posterior_module():
    """Execute every numerical dependency from Git under isolated module names."""
    checkpoint = FrozenCheckpoint(REVISION, "posterior_public")
    module = checkpoint.load("bayesfilter.inference.posterior_local_initializer")
    assert FROZEN_GRAPH_DEPENDENCIES <= checkpoint.sources.keys()
    assert not any("functional_control_flow" in path for path in checkpoint.sources)
    return module, checkpoint


def _verify_public_helper_identity(frozen, current):
    """Only the full frozen public module may replace these reviewed symbols.

    Complete-public comparisons execute every statement from Git. Stage
    references additionally share the unchanged helpers checked below; none
    calls the replaced endpoint, result class or eigen reporter.
    """
    replaced = {"initialize_posterior_local_location_scale", "PosteriorLocalInitializerResult",
        "_eigen_summary", "_initialize_posterior_wall_diagnostic"}

    def retained(source):
        return [ast.dump(node, include_attributes=False) for node in ast.parse(source).body
            if not (isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in replaced)]

    assert retained(frozen) == retained(current), "frozen posterior helper/import drift"


@lru_cache(maxsize=1)
def verified_reference_tree():
    root = Path(__file__).resolve().parents[1]
    archive = subprocess.check_output(["git", "archive", REVISION, "bayesfilter"], cwd=root)
    _, checkpoint = frozen_posterior_module()
    sources, hashes = {}, {}
    with tarfile.open(fileobj=io.BytesIO(archive)) as tar:
        for member in tar.getmembers():
            if member.isfile() and member.name.endswith(".py"):
                source = tar.extractfile(member).read()
                current = (root / member.name).read_bytes()
                if member.name in FROZEN_GRAPH_DEPENDENCIES:
                    # Both copies use the pinned numerical authority. The
                    # reference executes an independent module/cache instance.
                    assert checkpoint.sources[member.name].encode() == source
                    assert current == source, member.name
                elif member.name == "bayesfilter/inference/posterior_local_initializer.py" and current != source:
                    _verify_public_helper_identity(source.decode(), current.decode())
                else:
                    assert current == source, member.name
                sources[member.name] = source.decode()
                hashes[member.name] = hashlib.sha256(source).hexdigest()
    assert all(hashes[path] == digest for path, digest in checkpoint.hashes().items())
    return sources, hashes


def original_movement(evaluator, config, movement_config, prepared):
    sources, hashes = verified_reference_tree()
    path = "bayesfilter/inference/posterior_local_initializer.py"
    source = sources[path]
    function = next(node for node in ast.parse(source).body
                    if isinstance(node, ast.FunctionDef) and node.name == "initialize_posterior_local_location_scale")
    recorder = next(node for node in function.body if isinstance(node, ast.FunctionDef) and node.name == "record_candidate")
    loop = next(node for node in function.body if isinstance(node, ast.For) and isinstance(node.target, ast.Name)
                and node.target.id == "attempt")
    recorder_source = "\n".join(source.splitlines()[recorder.lineno - 1:recorder.end_lineno])
    loop_source = "\n".join(source.splitlines()[loop.lineno - 1:loop.end_lineno])
    geometry_records = []

    def fit(callback, center, *, batched_value_and_score_fn, scale, config):
        index = config.seed[1] - movement_config.seed[1]
        program = make_geometry_program(callback, int(center.shape[0]), config,
            batched_callback=batched_value_and_score_fn)
        raw = program(center, scale, *(cloud[index] for cloud in prepared))
        geometry_records.append(raw)
        return geometry_result(raw, center, scale, config, batched=batched_value_and_score_fn is not None)

    def capture(**kwargs):
        return {"status": kwargs["status"], "center": kwargs["center"],
            "value": kwargs["center_value"], "score": kwargs["center_score"],
            "movement_fits": kwargs["movement_fits"], "ledger": kwargs["ledger"]}

    text = '''def run(initial, value, score, units):
    cfg, move_cfg = config, movement_config
    batched_value_and_score_fn = evaluator.batched_fn
    candidates, ledger, movement_rows = [], [], []
    locator_payload = {}
'''+recorder_source+'''
    record_candidate(initial, float(value), score, "initial")
    successful_fits = 0
    final_material_move = False
    base_seed = tuple(int(value) for value in move_cfg.seed)
'''+loop_source+'''
    incumbent = select_exact_incumbent(candidates)
    return {"status": "insufficient_successful_movement_fits" if successful_fits < cfg.min_movement_fits else
        "movement_not_centered_within_attempt_budget" if final_material_move else "movement_complete",
        "center": incumbent.position, "value": incumbent.value, "score": incumbent.score,
        "movement_fits": movement_rows, "ledger": ledger}
'''
    reference, _ = frozen_posterior_module()
    namespace = {**vars(reference), "config": config, "movement_config": movement_config,
        "evaluator": evaluator, "fit_low_rank_spd_quadratic_geometry": fit, "_build_result": capture}
    exec(compile(text, REVISION + ":prepared_movement_reference", "exec"), namespace)  # noqa: S102
    return namespace["run"], geometry_records, {"source_hashes": hashes,
        "recorder_sha256": hashlib.sha256(recorder_source.encode()).hexdigest(),
        "loop_sha256": hashlib.sha256(loop_source.encode()).hexdigest(),
        "adapter": "Only prepared geometry inputs and result capture; original recorder/loop statements unchanged."}


def target_fixture(dimension, batched, case, max_rows):
    calls = tf.Variable(0, dtype=tf.int64)
    rows = tf.Variable(0, dtype=tf.int64)
    points = tf.Variable(tf.zeros([1024, dimension], D))
    extents = tf.Variable(tf.zeros([1024], tf.int64))

    def batch(cloud):
        count = cloud.shape[0]
        call_index = calls.assign_add(1) - 1
        start = rows.assign_add(count) - count
        events = (points.scatter_nd_update((start + tf.range(count, dtype=tf.int64))[:, None], cloud),
                  extents.scatter_nd_update(call_index[None, None], tf.constant([count], tf.int64)))
        with tf.control_dependencies(events):
            delta = cloud - .13
            score = -delta * (tf.cast(tf.range(dimension), D) + 2.)
            value = .5 * tf.reduce_sum(delta * score, axis=1)
        if case == "nonlinear":
            value -= .03 * tf.reduce_sum(delta ** 4, axis=1)
            score -= .12 * delta ** 3
        if case == "invalid":
            value = tf.where(cloud[:, 0] > .1, tf.constant(float("nan"), D), value)
        return value, score

    def scalar(point):
        values, scores = batch(point[None])
        return values[0], scores[0]

    tracker = original._EligibilityTrackingEvaluator(scalar, dimension=dimension, max_rows=max_rows,
        batched_fn=batch if batched else None, eligibility_fn=(lambda p: p[0] < 0) if case == "mismatch" else None,
        batched_eligibility_fn=(lambda p: p[:, 0] < 0) if case == "mismatch" and batched else None)

    def reset():
        for resource in (calls, rows, points, extents, tracker.evaluated_rows, tracker.invalid_rows,
                         tracker.mismatch_rows, tracker.budget_exhausted):
            resource.assign(tf.zeros_like(resource))

    def record():
        return {"points": points[:int(rows)], "extents": extents[:int(calls)], "tracker": tracker.diagnostics()}

    return tracker, reset, record


def completed_record(raw, scale, movement_config):
    rows = []
    for index in range(int(raw["attempt_count"])):
        geometry = tf.nest.map_structure(lambda x, row=index: x[row], raw["geometry_history"])
        stage = int(geometry["stage"])
        rows.append({"attempt": index, "seed": f"{movement_config.seed[0]}:{movement_config.seed[1] + index}",
            "fit_center": raw["fit_centers"][index],
            "geometry_status": FIT_STATUSES[int(geometry["fit_result"]["status"])] if stage == 3 else STAGES[stage],
            "random_stream": original.STREAM_ID, "geometry_accepted": bool(raw["geometry_accepted"][index]),
            "geometry_exact_evaluation_count": int(geometry["evaluation_count"]),
            "geometry_best_source": SOURCE_ROLES[int(raw["geometry_best_sources"][index])] if bool(raw["geometry_best_present"][index]) else None,
            "candidate_promoted": bool(raw["candidate_promoted"][index]),
            "center_moved": bool(raw["center_moved"][index]), "material_center_move": bool(raw["material_moves"][index]),
            "scaled_center_move": float(raw["scaled_moves"][index]), "objective_improvement": float(raw["improvements"][index]),
            "successful_fit_count": int(raw["successful_fit_counts"][index]), "covariance_handoff_eligible": False})
    ledger = []
    for index in range(int(raw["ledger_count"])):
        attempt, source = int(raw["ledger_attempts"][index]), int(raw["ledger_sources"][index])
        ledger.append({"ledger_index": index, "source": "initial" if index == 0 else
            f"movement_fit[{attempt}]_{SOURCE_ROLES[source]}_replay", "position": raw["ledger_positions"][index],
            "value": float(raw["ledger_values"][index]),
            "score_l2": float(raw["ledger_scaled_score_l2"][index]),
            "promoted": bool(raw["ledger_promoted"][index])})
    return clean({"status": STATUSES[int(raw["status"])], "center": raw["center"], "value": raw["value"],
                  "score": raw["score"], "movement_fits": rows, "ledger": ledger})


@pytest.mark.parametrize("dimension,batched,case", [(1, False, "healthy"), (3, True, "healthy"),
    (1, True, "nonlinear"), (3, False, "budget"), (1, True, "invalid"), (1, True, "mismatch"),
    (1, True, "stationary")])
def test_prepared_movement_original_records(dimension, batched, case, request):
    cfg = original.PosteriorLocalInitializerConfig(max_movement_attempts=3, min_movement_fits=2)
    move_cfg = LowRankSPDQuadraticGeometryConfig(rank=1, sample_count=16, min_samples_per_parameter=1,
        pilot_direction_count=6, holdout_fraction=.25, trust_radius=.3,
        holdout_rmse_abs_tolerance=.1, holdout_rmse_rel_tolerance=.1,
        constrain_center_refinement_to_trust_region=True, seed=(12, 34))
    clouds = [prepare_geometry_inputs(dimension, dataclasses.replace(move_cfg, seed=(12, 34 + i))) for i in range(3)]
    prepared = tuple(tf.stack([row[i] for row in clouds]) for i in range(3))
    tracker, reset, record = target_fixture(dimension, batched, case, 1 if case == "budget" else 1000)
    reference, geometry_records, provenance = original_movement(tracker, cfg, move_cfg, prepared)
    scope = ProgramCacheScope()
    with scope.activate():
        program = make_posterior_movement_program(tracker, dimension, cfg, move_cfg)
    observations = []
    for shift in (0., .01, 0.):
        initial = tf.fill([dimension], tf.constant((.13 if case == "stationary" else -.2) + shift, D))
        scale = tf.cast(tf.range(dimension), D) * .1 + .8 + shift
        score = -(initial - .13) * (tf.cast(tf.range(dimension), D) + 2.)
        value = .5 * tf.reduce_sum((initial - .13) * score)
        if case == "nonlinear":
            value -= .03 * tf.reduce_sum((initial - .13) ** 4)
            score -= .12 * (initial - .13) ** 3
        reset()
        geometry_records.clear()
        expected = clean(reference(initial, value, score, scale))
        expected_calls = clean(record())
        reset()
        with scope.activate(), tf.GradientTape() as tape:
            tape.watch((initial, value, score, scale))
            raw = program(initial, value, score, scale, *prepared)
            returned = raw["value"] + tf.reduce_sum(raw["center"] + raw["score"])
        assert tape.gradient(returned, (initial, value, score, scale)) == (None, None, None, None)
        actual = completed_record(raw, scale, move_cfg)
        observations.append({"expected": expected, "actual": actual,
            "expected_calls": expected_calls, "actual_calls": clean(record())})
    save(request, f"posterior-movement-{dimension}-{batched}-{case}.json", {
        "comparisons": observations, "provenance": provenance, "public_installed": False})
    for observation in observations:
        _equal_records(observation["actual"], observation["expected"])
        _equal_records(observation["actual_calls"], observation["expected_calls"])
    assert observations[0] == observations[2]
    assert program.experimental_get_tracing_count() == 1
    hlo = stable_hlo(program.experimental_get_compiler_ir(initial, value, score, scale, *prepared)(stage="hlo"))
    changed_hlo = stable_hlo(program.experimental_get_compiler_ir(initial + .02, value, score, scale, *prepared)(stage="hlo"))
    assert hlo == changed_hlo
    if case == "stationary":
        assert len(observations[0]["actual"]["movement_fits"]) == 2
        assert observations[0]["actual"]["status"] == "movement_complete"


def test_real_compilation_error_has_no_eager_fallback(request):
    count = tf.Variable(0, dtype=tf.int64)
    eager_calls = []

    def callback(point):
        if tf.executing_eagerly():
            eager_calls.append(True)
        increment = count.assign_add(1)
        with tf.control_dependencies([increment]):
            unsupported = tf.strings.to_number(tf.strings.as_string(point[0], precision=17), out_type=D)
            return -.5 * unsupported ** 2, -point

    tracker = original._EligibilityTrackingEvaluator(callback, dimension=1, max_rows=100,
        batched_fn=None, eligibility_fn=None, batched_eligibility_fn=None)
    cfg = original.PosteriorLocalInitializerConfig(max_movement_attempts=2, min_movement_fits=2)
    move_cfg = LowRankSPDQuadraticGeometryConfig(rank=1, sample_count=8,
        min_samples_per_parameter=1, constrain_center_refinement_to_trust_region=True, seed=(7, 12))
    clouds = [prepare_geometry_inputs(1, dataclasses.replace(move_cfg, seed=(7, 12 + i))) for i in range(2)]
    prepared = tuple(tf.stack([row[i] for row in clouds]) for i in range(3))
    with ProgramCacheScope().activate():
        program = make_posterior_movement_program(tracker, 1, cfg, move_cfg)
        with pytest.raises(tf.errors.OpError, match="AsString|StringToNumber") as caught:
            program(tf.constant([-.2], D), tf.constant(-.02, D), tf.constant([.2], D),
                    tf.constant([1.], D), *prepared)
    assert not eager_calls and int(count) == int(tracker.evaluated_rows) == 0
    save(request, "posterior-movement-compiler-error.json", {"error_type": type(caught.value).__name__,
        "eager_calls": eager_calls, "target_rows": int(count), "tracker": tracker.diagnostics()})
