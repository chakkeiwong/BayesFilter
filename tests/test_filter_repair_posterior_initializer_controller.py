"""Complete pinned public-module reference for the prepared native initializer."""

import gc
import hashlib
import sys
import types
import weakref

import pytest
import tensorflow as tf

from bayesfilter.inference.joint_center import JointCenterLocatorConfig
from bayesfilter.inference.joint_center_tf import joint_center_result
from bayesfilter.inference.posterior_cloud_preparation_tf import (
    PosteriorCloudPreparation,
    posterior_seed_keys,
)
from bayesfilter.inference.posterior_initializer_controller_tf import (
    PreparedPosteriorInitializer,
)
from bayesfilter.inference.posterior_local_initializer import (
    PosteriorLocalInitializerConfig,
    _json_ready,
)
from bayesfilter.inference.posterior_movement_tf import STATUSES as MOVEMENT_STATUSES
from bayesfilter.inference.quadratic_geometry import LowRankSPDQuadraticGeometryConfig
from tests.test_filter_repair_block_capture import stable_hlo
from tests.test_filter_repair_geometry_control import clean, save
from tests.test_filter_repair_posterior_curvature_controller import (
    complete as curvature_record,
)
from tests.test_filter_repair_posterior_curvature_controller import fixture
from tests.test_filter_repair_posterior_movement import (
    REVISION,
    verified_reference_tree,
)
from tests.test_filter_repair_posterior_movement import (
    completed_record as movement_record,
)
from tests.test_filter_repair_quadratic_batches import _equal_records
from tests.test_posterior_local_initializer import _thresholds

D = tf.float64


def original_module():
    sources, hashes = verified_reference_tree()
    path = "bayesfilter/inference/posterior_local_initializer.py"
    name = f"posterior_public_frozen_{REVISION}"
    module = types.ModuleType(name)
    module.__file__ = f"{REVISION}:{path}"
    sys.modules[name] = module
    exec(compile(sources[path], module.__file__, "exec"), module.__dict__)  # noqa: S102
    return module, hashes


def completed_payload(raw, config, movement_config, thresholds, *, batched, eligibility=False):
    """Diagnostic formatting from completed tensors only; no target decisions."""
    from bayesfilter.inference.posterior_local_initializer import (
        POSTERIOR_LOCAL_INITIALIZER_NONCLAIMS,
        STREAM_ID,
    )

    stage, accepted = int(raw["stage"]), bool(raw["accepted"])
    accounting = {**clean(raw["accounting"]), "status_callback_supplied": eligibility,
        "batched_status_callback_supplied": eligibility and batched, "finite_status_agreement_required": True}
    location = {"status": "not_run"}
    if stage >= 1:
        location = {**joint_center_result(raw["locator"], jit_compile=config.locator_config.jit_compile).payload(),
            "chart": "z = radius * tanh(u / radius)", "chart_radius": config.locator_box_radius,
            "chart_center_role": "truth_blind_initial_position"}
    ledger = []
    for index in range(int(raw["initial_ledger"]["recorded_count"])):
        ledger.append({"ledger_index": index,
            "source": "initial_replay" if index == 0 else "bounded_locator_best_replay",
            "position": raw["initial_positions"][index], "value": raw["initial_values"][index],
            "score_l2": raw["initial_ledger"]["scaled_score_l2"][index],
            "promoted": raw["initial_ledger"]["promoted"][index]})
    movement = []
    curvature = None
    extra = {}
    if stage == 0:
        status = "eligibility_contract_mismatch" if accounting["mismatch_rows"] else "initial_target_invalid"
    elif stage == 1:
        status = "eligibility_contract_mismatch" if int(raw["tracker_status_after_locator"]) == 1 else "exact_evaluation_budget_exhausted"
    else:
        moved = movement_record(raw["movement"], raw["scale"], movement_config)
        movement = moved["movement_fits"]
        for row in moved["ledger"][1:]:
            ledger.append({**row, "ledger_index": len(ledger)})
        status = MOVEMENT_STATUSES[int(raw["movement"]["status"])]
        if stage == 3:
            curved = curvature_record(raw["curvature"], raw["scale"], config, thresholds)
            for row in curved["ledger"][1:]:
                ledger.append({**row, "ledger_index": len(ledger)})
            curvature, status = curved["curvature"], curved["status"]
            extra["curvature_attempts"] = curved["attempts"]
    if accepted:
        movement[-1] = {**movement[-1], "covariance_handoff_eligible": True}
    payload = {
        "schema": "bayesfilter.posterior_local_initializer.v1", "accepted": accepted,
        "status": status, "dimension": int(raw["center"].shape[0]), "center_value": raw["value"],
        "locator": location, "movement_fits": movement, "curvature": curvature,
        "exact_evaluation_ledger": ledger, "exact_evaluation_count": accounting["evaluated_rows"],
        "diagnostics": {"classification": "posterior_local_initializer_accepted" if accepted else "posterior_local_initializer_rejected",
            "eligibility_contract": accounting, "random_stream": STREAM_ID,
            "hmc_rejection_policy_permitted": False, "base_distribution_changed": False,
            "base_distribution_contract": "IID standard normal remains external to this initializer",
            "full_covariance_installed_as_fixed_transport": False, "terminal_stationarity_required": False,
            "global_map_claim": False, "config": config.payload(), "movement_config": movement_config.payload(), **extra},
        "nonclaims": POSTERIOR_LOCAL_INITIALIZER_NONCLAIMS,
        "center": raw["center"], "center_score": raw["score"], "scale": raw["scale"],
        "precision_z": raw["curvature"]["fit"]["fit"]["selection"]["precision"] if accepted else None,
        "covariance_z": raw["curvature"]["fit"]["fit"]["covariance"] if accepted else None,
        "precision_theta": raw["curvature"]["precision_theta"] if accepted else None,
        "covariance_theta": raw["curvature"]["covariance_theta"] if accepted else None,
        "marginal_standard_deviations": raw["curvature"]["marginal"] if accepted else None,
        "initial_output_shift": raw["center"] if accepted else None,
        "initial_output_scale_log": raw["curvature"]["scale_log"] if accepted else None,
    }
    if accepted:
        payload.update(precision_z_eigen_summary=raw["precision_eigen_summary"],
            covariance_theta_eigen_summary=raw["covariance_eigen_summary"])
    return _json_ready(payload)


CASES = ((1, True, "stationary"), (3, True, "stationary"),
    (1, False, "moving"), (3, True, "nonlinear"),
    (1, False, "invalid_center"), (1, True, "invalid_partial"),
    (1, True, "budget"), (1, True, "mismatch"))


@pytest.mark.parametrize("dimension,batched,case", CASES)
def test_prepared_complete_public_reference(dimension, batched, case, request):
    module, hashes = original_module()
    config = PosteriorLocalInitializerConfig(factor_max=2, seed=(31, 43),
        max_exact_evaluations=4 if case == "budget" else 5000,
        locator_config=JointCenterLocatorConfig(max_iterations=10,
            gradient_tolerance=1e-10, max_objective_evaluations=60))
    movement = LowRankSPDQuadraticGeometryConfig(rank=1, sample_count=12 * dimension,
        min_samples_per_parameter=1, fit_max_iterations=8, pilot_direction_count=6,
        trust_radius=.3, holdout_fraction=.25, holdout_rmse_abs_tolerance=.1,
        holdout_rmse_rel_tolerance=.1, constrain_center_refinement_to_trust_region=True, seed=(12, 34))
    thresholds = _thresholds(dimension)
    monitor, reset, record, _ = fixture(dimension, batched, case, config)
    generate = PosteriorCloudPreparation(dimension, config, movement)
    prepared = generate(*posterior_seed_keys(dimension, config, movement),
        tf.constant(movement.trust_radius, D), tf.constant(config.curvature_radius, D))
    initial = tf.fill([dimension], tf.constant(.13, D))
    scale = tf.cast(tf.range(dimension), D) * .1 + .8
    callbacks = {"batched_callback": monitor.batched_fn,
        "eligibility_callback": monitor.eligibility_fn,
        "batched_eligibility_callback": monitor.batched_eligibility_fn}
    reset()
    # Construction must remain safe even when the caller already has a tape.
    with tf.GradientTape() as construction_tape:
        construction_tape.watch((initial, scale))
        program = PreparedPosteriorInitializer(monitor.scalar_fn, dimension, config, movement, thresholds,
            **callbacks)
    assert not record()["extents"]
    observations, hlos = [], []
    for shift in (0., .01, 0.):
        initial = tf.fill([dimension], tf.constant((-.2 if case == "moving" else .13) + shift, D))
        scale = tf.cast(tf.range(dimension), D) * .1 + .8 + shift
        reset()
        expected = module.initialize_posterior_local_location_scale(monitor.scalar_fn, initial, scale=scale,
            batched_value_and_score_fn=monitor.batched_fn, eligibility_fn=monitor.eligibility_fn,
            batched_eligibility_fn=monitor.batched_eligibility_fn, config=config, movement_config=movement,
            curvature_thresholds=thresholds).payload(include_arrays=True)
        expected_calls = {k: v for k, v in record().items() if k != "tracker"}
        reset()
        operands = (initial, scale, prepared["directions"], prepared["movement_offsets"],
            prepared["permutation_keys"], prepared["curvature_offsets"])
        with tf.GradientTape() as tape:
            tape.watch((initial, scale))
            raw = program(*operands)
            total = raw["value"] + tf.reduce_sum(raw["center"] + raw["score"] + raw["curvature"]["marginal"])
        assert tape.gradient(total, (initial, scale)) == (None, None)
        actual = completed_payload(raw, config, movement, thresholds, batched=batched,
            eligibility=case == "mismatch")
        actual_calls = {k: v for k, v in record().items() if k != "tracker"}
        observations.append({"expected": clean(expected), "actual": actual,
            "expected_calls": expected_calls, "actual_calls": actual_calls})
        hlos.append(stable_hlo(program.compiled.experimental_get_compiler_ir(*operands)(stage="hlo")))
    save(request, f"posterior-complete-public-reference-{dimension}-{batched}-{case}.json", {
        "reference_revision": REVISION, "reference_sha256": hashes, "reference_entire_module": True,
        "comparisons": observations, "trace_count": program.compiled.experimental_get_tracing_count(),
        "hlo_sha256": [hashlib.sha256(hlo.encode()).hexdigest() for hlo in hlos], "public_installed": False})
    for comparison in observations:
        _equal_records(comparison["actual"], comparison["expected"])
        _equal_records(comparison["actual_calls"], comparison["expected_calls"])
    assert observations[0] == observations[2]
    assert program.compiled.experimental_get_tracing_count() == 1
    assert hlos[0] == hlos[1] == hlos[2]


def test_complete_compiler_failure_and_owner_release(request):
    count = tf.Variable(0, dtype=tf.int64)
    eager_calls = []

    def callback(point):
        if tf.executing_eagerly():
            eager_calls.append(True)
        increment = count.assign_add(1)
        with tf.control_dependencies([increment]):
            value = tf.strings.to_number(tf.strings.as_string(point[0], precision=17), out_type=D)
            return -.5 * value ** 2, -point

    config = PosteriorLocalInitializerConfig()
    movement = LowRankSPDQuadraticGeometryConfig(sample_count=8, min_samples_per_parameter=1,
        constrain_center_refinement_to_trust_region=True)
    owner = PreparedPosteriorInitializer(callback, 1, config, movement, _thresholds(1))
    owner_ref, program_ref, scope_ref = weakref.ref(owner), weakref.ref(owner.compiled), weakref.ref(owner.dependency_scope)
    with pytest.raises(tf.errors.OpError, match="AsString|StringToNumber") as caught:
        owner(tf.constant([.13], D), tf.constant([.8], D), tf.zeros([3, 0, 1], D),
            tf.zeros([3, 8, 1], D), tf.zeros([3, 2], tf.int32), tf.zeros([2, 5, 2, 1], D))
    assert not eager_calls and int(count) == 0
    error_type = type(caught.value).__name__
    del caught, owner
    gc.collect()
    assert owner_ref() is None and program_ref() is None and scope_ref() is None
    save(request, "posterior-complete-compiler-owner.json", {"error_type": error_type,
        "eager_calls": eager_calls, "target_rows": int(count),
        "python_owner_collected": True, "native_executable_eviction_proved": False})
