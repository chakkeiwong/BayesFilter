def run(value_and_score_fn, dimension, cfg, center, center_value, center_score, scale_tf, radius, stalled, fit, structured_data):
    history, row_diag, evaluations = [], {}, 0
    proposal_rows: list[Mapping[str, Any]] = []
    accepted = False
    evaluated_proposal: tf.Tensor | None = None
    evaluated_proposal_value: float | None = None
    evaluated_proposal_score: tf.Tensor | None = None
    best_proposal: tf.Tensor | None = None
    best_proposal_value = center_value
    best_proposal_score: tf.Tensor | None = None
    selected_fit = fit
    actual = float("-inf")
    predicted = float("-inf")
    rho = float("-inf")
    old_norm = float(tf.linalg.norm(scale_tf * center_score).numpy())
    new_norm = old_norm
    step_info: Mapping[str, Any] = {"boundary_active": False}
    proposal_score_gate: Mapping[str, Any] = {
        "policy": cfg.proposal_score_acceptance_policy,
        "active": cfg.require_proposal_score_reduction,
        "passed": not cfg.require_proposal_score_reduction,
    }
    factor_candidates = [1]
    if (
        cfg.refinement_geometry_policy == "factor_correlation"
        and cfg.structured_max_factors == 2
    ):
        factor_candidates.append(2)
    for factor_count in factor_candidates:
        if factor_count == 2:
            if accepted:
                break
            selected_fit = _fit_factor_from_data(
                structured_data, factor_count=2, config=cfg
            )
            if selected_fit["status"] != "usable":
                proposal_rows.append(
                    {
                        "factor_count": 2,
                        "fit_status": selected_fit["status"],
                        "proposal_evaluated": False,
                    }
                )
                continue
        if selected_fit["status"] != "usable":
            proposal_rows.append(
                {
                    "factor_count": factor_count,
                    "fit_status": selected_fit["status"],
                    "proposal_evaluated": False,
                }
            )
            continue
        proposed = proposal_program(value_and_score_fn, dimension,
            cfg.proposal_score_acceptance_policy, cfg.require_proposal_score_reduction)(
            center, tf.constant(center_value, tf.float64), center_score, scale_tf,
            tf.convert_to_tensor(selected_fit["projected_precision_z"], tf.float64),
            tf.constant(radius, tf.float64), tf.constant(cfg.score_reduction_factor, tf.float64),
            tf.constant(cfg.acceptance_ratio, tf.float64))
        proposal, proposal_score = proposed["position"], proposed["score"]
        step_info = {"boundary_active": bool(proposed["boundary"])}
        evaluations += 1
        proposal_value = float(proposed["value"])
        evaluated_proposal = proposal
        evaluated_proposal_value = proposal_value
        evaluated_proposal_score = proposal_score
        actual, predicted, rho = float(proposed["actual"]), float(proposed["predicted"]), float(proposed["rho"])
        old_norm, new_norm = float(proposed["old_norm"]), float(proposed["new_norm"])
        finite = bool(proposed["finite"])
        if finite and proposal_value > best_proposal_value:
            best_proposal = proposal
            best_proposal_value = proposal_value
            best_proposal_score = proposal_score
        score_reduction_passed = bool(proposed["score_passed"])
        proposal_score_gate = {
            "policy": cfg.proposal_score_acceptance_policy,
            "active": cfg.require_proposal_score_reduction,
            "passed": score_reduction_passed,
            "legacy_fractional_passed": bool(proposed["legacy_passed"]),
            "required_score_norm_max": (float(proposed["required_norm_max"])
                if cfg.require_proposal_score_reduction else None),
            "numerical_resolution_floor": (float(proposed["resolution_floor"])
                if cfg.require_proposal_score_reduction
                and cfg.proposal_score_acceptance_policy == "resolvable_decrease" else None),
        }
        accepted = bool(proposed["accepted"])
        proposal_rows.append(
            {
                "factor_count": (
                    factor_count
                    if cfg.refinement_geometry_policy == "factor_correlation"
                    else None
                ),
                "fit_status": selected_fit["status"],
                "proposal_evaluated": True,
                "actual_improvement": actual,
                "predicted_improvement": predicted,
                "rho": rho,
                "score_norm_before": old_norm,
                "score_norm_after": new_norm,
                "score_reduction_passed": score_reduction_passed,
                "proposal_score_gate": proposal_score_gate,
                "accepted": accepted,
            }
        )
        if accepted:
            break
    incumbent_promoted_without_model_acceptance = bool(
        best_proposal is not None
        and best_proposal_score is not None
        and (
            not accepted
            or evaluated_proposal_value is None
            or best_proposal_value > evaluated_proposal_value
        )
    )
    if best_proposal is not None and best_proposal_score is not None:
        center_value = best_proposal_value
        center = best_proposal
        center_score = best_proposal_score
        stalled = 0
    else:
        stalled += 1
    if rho < cfg.shrink_threshold or not accepted:
        radius *= cfg.shrink_factor
        radius_action = "contract"
    elif rho >= cfg.expansion_threshold and bool(step_info["boundary_active"]):
        radius = min(cfg.maximum_radius, radius * cfg.expansion_factor)
        radius_action = "expand"
    else:
        radius_action = "retain"
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
