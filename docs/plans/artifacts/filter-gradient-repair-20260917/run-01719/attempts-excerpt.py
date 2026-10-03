def run(value_and_score_fn, dimension, cfg, center, center_value, center_score, scale_tf, radius, stalled, fit, structured_data):
    history, row_diag, evaluations = [], {}, 0
    first_usable = fit["status"] == "usable"
    first_precision = (tf.convert_to_tensor(fit["projected_precision_z"], tf.float64)
        if first_usable else tf.zeros([dimension, dimension], tf.float64))
    if cfg.refinement_geometry_policy == "factor_correlation" and cfg.structured_max_factors == 2:
        numerical = structured_data["_native_factor_data"]
        second = second_factor_program(dimension, int(numerical["training_offsets_z"].shape[0]),
            int(numerical["holdout_offsets_z"].shape[0]), cfg)
        second_inputs = (numerical["center_score_z"], numerical["training_offsets_z"], numerical["training_scores_z"],
            numerical["holdout_offsets_z"], numerical["holdout_scores_z"], numerical["training_weights"],
            numerical["active_training_rows"])
    else:
        second, second_inputs = empty_second_program(dimension), ()
    proposals = attempts_program(value_and_score_fn, second, dimension, cfg)(
        center, tf.constant(center_value, tf.float64), center_score, scale_tf,
        tf.constant(radius, tf.float64), tf.constant(stalled), tf.constant(first_usable),
        first_precision, second_inputs)
    attempted = int(proposals["attempted"])
    evaluations += int(proposals["evaluations"])
    selected_fit = fit
    if attempted == 2:
        factor_config = FactorCorrelationGeometryConfig(factor_count=2,
            max_condition_number=cfg.max_condition_number,
            holdout_score_relative_rmse=cfg.structured_holdout_score_relative_rmse)
        second_result = _factor_result_from_native(proposals["second_fit"]["computed"], factor_config,
            dimension, int(numerical["active_training_rows"]), int(numerical["holdout_offsets_z"].shape[0]))
        selected_fit = _factor_fit_payload(second_result, structured_data)
    last = proposals["last"]
    accepted = bool(proposals["accepted"])
    evaluated = bool(proposals["last_evaluated"])
    evaluated_proposal = last["position"] if evaluated else None
    evaluated_proposal_value = float(last["value"]) if evaluated else None
    evaluated_proposal_score = last["score"] if evaluated else None
    actual, predicted, rho = float(last["actual"]), float(last["predicted"]), float(last["rho"])
    old_norm, new_norm = float(last["old_norm"]), float(last["new_norm"])
    step_info = {"boundary_active": bool(last["boundary"])}
    proposal_score_gate = (_proposal_gate_payload(last, cfg) if evaluated else {
        "policy": cfg.proposal_score_acceptance_policy, "active": cfg.require_proposal_score_reduction,
        "passed": not cfg.require_proposal_score_reduction})
    # Restore reporting rows only after all numerical attempts, decisions,
    # incumbent selection and radius/stall updates have completed in XLA.
    # Transfer each completed column once; indexing these host records must
    # not dispatch one eager TensorFlow slicing kernel for every field.
    history_values = {key: values.numpy().tolist() for key, values in proposals["histories"].items()}
    evaluated_rows = proposals["evaluated"].numpy().tolist()
    proposal_rows = []
    for index in range(attempted):
        row_fit = fit if index == 0 else selected_fit
        if not evaluated_rows[index]:
            proposal_rows.append({"factor_count": index + 1,
                "fit_status": row_fit["status"], "proposal_evaluated": False})
            continue
        row = {key: values[index] for key, values in history_values.items()}
        proposal_rows.append({"factor_count": index + 1 if cfg.refinement_geometry_policy == "factor_correlation" else None,
            "fit_status": row_fit["status"], "proposal_evaluated": True,
            "actual_improvement": float(row["actual"]), "predicted_improvement": float(row["predicted"]),
            "rho": float(row["rho"]), "score_norm_before": float(row["old_norm"]),
            "score_norm_after": float(row["new_norm"]), "score_reduction_passed": bool(row["score_passed"]),
            "proposal_score_gate": _proposal_gate_payload(row, cfg), "accepted": bool(row["accepted"])})
    incumbent_promoted_without_model_acceptance = bool(proposals["promoted_without_acceptance"])
    center_value, center, center_score = float(proposals["center_value"]), proposals["center"], proposals["center_score"]
    stalled = int(proposals["stalled"])
    radius = float(proposals["radius_after"])
    radius_action = ("contract", "expand", "retain")[int(proposals["radius_action"])]
    history.append({
        **row_diag,
        "fit": selected_fit,
        "action": "proposal_accepted" if accepted else "proposal_rejected",
        "actual_improvement": actual, "predicted_improvement": predicted,
        "rho": rho, "score_norm_before": old_norm, "score_norm_after": new_norm,
        "proposal_score_gate": proposal_score_gate,
        "boundary_active": bool(step_info["boundary_active"]),
        "radius_action": radius_action, "radius_after": radius,
        "proposal_attempts": proposal_rows,
        "exact_incumbent_promoted_without_model_acceptance": (
            incumbent_promoted_without_model_acceptance
        ),
        **(
            {
                "refinement_movement": _refinement_movement_payload(
                    refinement_origin=refinement_origin,
                    scale=scale_tf,
                    pre_center=pre_center,
                    pre_value=pre_value,
                    pre_score=pre_score,
                    search_center=selected_center,
                    search_value=selected_value,
                    search_score=selected_score,
                    evaluated_proposal=evaluated_proposal,
                    evaluated_proposal_value=evaluated_proposal_value,
                    evaluated_proposal_score=evaluated_proposal_score,
                    terminal_center=center,
                    terminal_value=center_value,
                    terminal_score=center_score,
                    radius_before=float(row_diag["radius_before"]),
                    radius_after=radius,
                    search_recentered=recentered,
                    proposal_accepted=accepted,
                )
            }
            if cfg.record_refinement_movement_diagnostics
            else {}
        ),
    })
    return {"history": history, "evaluations": evaluations, "center": center, "center_value": center_value, "center_score": center_score, "stalled": stalled, "radius": radius}
