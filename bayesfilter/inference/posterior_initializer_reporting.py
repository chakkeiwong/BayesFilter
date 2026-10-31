"""Completed posterior-initializer reporting; no numerical feedback or targets.

Host loops decode completed tensor histories and static names only. Selection,
stopping, norms, eigensummaries and physical coordinate arithmetic are supplied
by the enclosing program.
"""

import tensorflow as tf

from bayesfilter.inference import fixed_center_curvature as fixed
from bayesfilter.inference.dense_initializer_reporting import _raise_partition_error
from bayesfilter.inference.factor_correlation_geometry import (
    FactorCorrelationGeometryConfig,
)
from bayesfilter.inference.joint_center_tf import joint_center_result
from bayesfilter.inference.posterior_curvature_controller_tf import (
    STATUSES as CURVATURE_STATUSES,
)
from bayesfilter.inference.posterior_local_initializer import (
    STREAM_ID,
    PosteriorLocalInitializerResult,
    _cloud_row_counts,
)
from bayesfilter.inference.posterior_movement_tf import STATUSES as MOVEMENT_STATUSES
from bayesfilter.inference.quadratic_geometry_fit_tf import SOURCE_ROLES
from bayesfilter.inference.quadratic_geometry_fit_tf import STATUSES as FIT_STATUSES
from bayesfilter.inference.quadratic_geometry_full_tf import STAGES


def movement_records(raw, config):
    """Decode stage records without running a geometry fit or a target."""
    rows, ledger = [], []
    for index in range(int(raw["attempt_count"])):
        geometry = tf.nest.map_structure(lambda x, i=index: x[i], raw["geometry_history"])
        stage = int(geometry["stage"])
        rows.append({"attempt": index, "seed": f"{config.seed[0]}:{config.seed[1] + index}",
            "fit_center": raw["fit_centers"][index],
            "geometry_status": FIT_STATUSES[int(geometry["fit_result"]["status"])] if stage == 3 else STAGES[stage],
            "random_stream": STREAM_ID, "geometry_accepted": bool(raw["geometry_accepted"][index]),
            "geometry_exact_evaluation_count": int(geometry["evaluation_count"]),
            "geometry_best_source": SOURCE_ROLES[int(raw["geometry_best_sources"][index])] if bool(raw["geometry_best_present"][index]) else None,
            "candidate_promoted": bool(raw["candidate_promoted"][index]),
            "center_moved": bool(raw["center_moved"][index]), "material_center_move": bool(raw["material_moves"][index]),
            "scaled_center_move": float(raw["scaled_moves"][index]), "objective_improvement": float(raw["improvements"][index]),
            "successful_fit_count": int(raw["successful_fit_counts"][index]), "covariance_handoff_eligible": False})
    # Row zero represents the already recorded incumbent from the prior stage.
    for index in range(1, int(raw["ledger_count"])):
        attempt, source = int(raw["ledger_attempts"][index]), int(raw["ledger_sources"][index])
        ledger.append({"source": f"movement_fit[{attempt}]_{SOURCE_ROLES[source]}_replay",
            "position": raw["ledger_positions"][index], "value": float(raw["ledger_values"][index]),
            "score_l2": float(raw["ledger_scaled_score_l2"][index]), "promoted": bool(raw["ledger_promoted"][index])})
    return rows, ledger


def curvature_records(raw, dimension, config, thresholds):
    """Decode completed curvature records, retaining original fit exceptions."""
    replicates = config.replicate_count
    rows = _cloud_row_counts(config, dimension)
    names = [*[f"training[{i}]" for i in range(replicates)],
        *[f"selection[{i}]" for i in range(replicates)], "audit"]

    def seeds(attempt):
        return [(config.seed[0] + attempt, config.seed[1] + 1000 * attempt + i) for i in range(len(names))]

    records = [{"attempt": i, "center": raw["centers"][i], "partition_names": names,
        "partition_seeds": seeds(i), "random_stream": STREAM_ID,
        "partition_rows": [rows[0]] * replicates + [rows[1]] * replicates + [rows[2]],
        "center_moved": bool(raw["center_moved"][i]), "scaled_center_move": float(raw["scaled_moves"][i]),
        "objective_improvement": float(raw["improvements"][i]), "fit_attempted": bool(raw["fit_attempts"][i])}
        for i in range(int(raw["record_count"]))]
    ledger = []
    for i in range(1, int(raw["ledger_count"])):
        attempt, part = int(raw["ledger_attempts"][i]), int(raw["ledger_partitions"][i])
        name = f"curvature[{attempt}]_" + ("center_replay" if part == -1 else f"{names[part]}_row")
        ledger.append({"source": name, "position": raw["ledger_positions"][i],
            "value": float(raw["ledger_values"][i]), "score_l2": float(raw["ledger_scaled_score_l2"][i]),
            "promoted": bool(raw["ledger_promoted"][i])})
    curvature = None
    if bool(raw["fit_attempted"]):
        _raise_partition_error(raw["fit"]["validation"], replicates)
        if bool(raw["configuration_error"]):
            FactorCorrelationGeometryConfig(factor_count=1,
                max_condition_number=config.max_condition_number,
                holdout_score_relative_rmse=thresholds.selection_holdout_relative_rmse_cap)
            raise RuntimeError("missing deferred factor configuration error")
        curvature = fixed._fixed_center_result_from_native(raw["fit"]["fit"], raw["center"], raw["center_score_z"],
            dimension=dimension, replicates=replicates, training_rows=rows[0], selection_rows=rows[1],
            audit_rows=rows[2], thresholds=thresholds, factor_max=config.factor_max,
            weights=config.shrinkage_weights, structured_target_family=config.structured_target_family,
            jit_compile=bool(raw["fit_jit_compile"]),
            lineage={"role": "posterior_local_terminal_curvature", "curvature_attempt": int(raw["attempt_count"]) - 1,
                "partition_seeds": seeds(int(raw["attempt_count"]) - 1), "random_stream": STREAM_ID,
                "fit_center_equals_exact_incumbent": True})
    status = CURVATURE_STATUSES[int(raw["status"])]
    if status == "curvature_result":
        status = f"curvature_{curvature.status}"
    return curvature, records, ledger, status


def posterior_initializer_result(raw, config, movement_config, thresholds, *,
        eligibility_supplied=False, batched_eligibility_supplied=False):
    """Construct the existing public result from an entirely completed program."""
    stage, accepted = int(raw["stage"]), bool(raw["accepted"])
    if stage == 4:
        _cloud_row_counts(config, int(raw["center"].shape[0]))
        raise RuntimeError("missing deferred curvature configuration error")
    accounting = {"evaluated_rows": int(raw["accounting"]["evaluated_rows"]),
        "invalid_rows": int(raw["accounting"]["invalid_rows"]),
        "mismatch_rows": int(raw["accounting"]["mismatch_rows"]),
        "budget_exhausted": bool(raw["accounting"]["budget_exhausted"]),
        "status_callback_supplied": eligibility_supplied,
        "batched_status_callback_supplied": batched_eligibility_supplied,
        "finite_status_agreement_required": True}
    location = {"status": "not_run"}
    if stage >= 1:
        location = {**joint_center_result(raw["locator"], jit_compile=config.locator_config.jit_compile).payload(),
            "chart": "z = radius * tanh(u / radius)", "chart_radius": config.locator_box_radius,
            "chart_center_role": "truth_blind_initial_position"}
    ledger = []
    for index in range(int(raw["initial_ledger"]["recorded_count"])):
        ledger.append({"ledger_index": index,
            "source": "initial_replay" if index == 0 else "bounded_locator_best_replay",
            "position": raw["initial_positions"][index], "value": float(raw["initial_values"][index]),
            "score_l2": float(raw["initial_ledger"]["scaled_score_l2"][index]),
            "promoted": bool(raw["initial_ledger"]["promoted"][index])})
    movement, curvature, extra = [], None, {}
    if stage == 0:
        status = "eligibility_contract_mismatch" if accounting["mismatch_rows"] else "initial_target_invalid"
    elif stage == 1:
        status = "eligibility_contract_mismatch" if int(raw["tracker_status_after_locator"]) == 1 else "exact_evaluation_budget_exhausted"
    else:
        movement, additions = movement_records(raw["movement"], movement_config)
        for row in additions:
            ledger.append({**row, "ledger_index": len(ledger)})
        status = MOVEMENT_STATUSES[int(raw["movement"]["status"])]
        if stage == 3:
            curvature, attempts, additions, status = curvature_records(raw["curvature"],
                int(raw["center"].shape[0]), config, thresholds)
            for row in additions:
                ledger.append({**row, "ledger_index": len(ledger)})
            extra["curvature_attempts"] = attempts
    if accepted:
        movement[-1] = {**movement[-1], "covariance_handoff_eligible": True}
    return PosteriorLocalInitializerResult(
        accepted=accepted, status=status, dimension=int(raw["center"].shape[0]),
        center=raw["center"], center_value=float(raw["value"]), center_score=raw["score"], scale=raw["scale"],
        precision_z=raw["curvature"]["fit"]["fit"]["selection"]["precision"] if accepted else None,
        covariance_z=raw["curvature"]["fit"]["fit"]["covariance"] if accepted else None,
        precision_theta=raw["curvature"]["precision_theta"] if accepted else None,
        covariance_theta=raw["curvature"]["covariance_theta"] if accepted else None,
        marginal_standard_deviations=raw["curvature"]["marginal"] if accepted else None,
        initial_output_shift=raw["center"] if accepted else None,
        initial_output_scale_log=raw["curvature"]["scale_log"] if accepted else None,
        locator=location, movement_fits=tuple(movement), curvature=curvature,
        exact_evaluation_ledger=tuple(ledger), exact_evaluation_count=accounting["evaluated_rows"],
        diagnostics={"classification": "posterior_local_initializer_accepted" if accepted else "posterior_local_initializer_rejected",
            "eligibility_contract": accounting, "random_stream": STREAM_ID,
            "hmc_rejection_policy_permitted": False, "base_distribution_changed": False,
            "base_distribution_contract": "IID standard normal remains external to this initializer",
            "full_covariance_installed_as_fixed_transport": False, "terminal_stationarity_required": False,
            "global_map_claim": False, "config": config.payload(), "movement_config": movement_config.payload(), **extra},
        _eigen_summaries={"precision_z_eigen_summary": raw["precision_eigen_summary"],
            "covariance_theta_eigen_summary": raw["covariance_eigen_summary"]} if accepted else None)
