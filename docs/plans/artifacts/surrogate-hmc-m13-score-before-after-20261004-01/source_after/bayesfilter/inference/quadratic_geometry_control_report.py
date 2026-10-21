"""Host formatting for completed quadratic proposal and replay records only."""

from bayesfilter.inference.quadratic_geometry_control_tf import (
    REJECTION_REASONS,
    STEP_FAILURES,
)


def center_refinement_report(record, config, center_value, center_score_norm):
    constrained = bool(config.constrain_center_refinement_to_trust_region)
    method = "exact_spd_quadratic_trust_region" if constrained else "unconstrained_spd_quadratic_solve"
    host = {key: value.numpy().tolist() for key, value in record.items()
            if key not in ("refined_center", "refined_score")}
    status = host["status"]
    if 1 <= status <= 5:
        return {"accepted": False, "reason": STEP_FAILURES[status - 1],
                **({"step_method": method, "trust_region_constrained": True} if constrained else {})}
    result = {"accepted": host["accepted"], "reason": "refined_value_or_score_nonfinite",
        "z_norm": host["z_norm"], "refined_center": record["refined_center"],
        "step_method": method, "trust_region_constrained": constrained,
        "boundary_active": host["boundary"], "lagrange_multiplier": host["multiplier"],
        "predicted_improvement": host["predicted"], "exact_evaluation_count": 1,
        "refined_target_finite": host["target_finite"]}
    if status == 6:
        return result
    reasons = ";".join(reason for reason, failed in zip(REJECTION_REASONS, host["reason_mask"], strict=True) if failed)
    return {**result, "reason": "accepted" if host["accepted"] else reasons,
        "actual_improvement": host["actual"],
        "actual_to_predicted_improvement_ratio": host["ratio"] if host["has_ratio"] else None,
        "center_log_prob": float(center_value), "refined_log_prob": host["refined_value"],
        "center_score_norm": float(center_score_norm), "refined_score_norm": host["score_norm"],
        "refined_score": record["refined_score"]}


def exact_replay_report(record, *, evaluation_index, source):
    attempted, valid, matches = bool(record["attempted"]), bool(record["valid"]), bool(record["matches"])
    result = {"attempted": attempted, "valid": valid, "matches": matches}
    if attempted:
        result.update(evaluation_index=int(evaluation_index), value=float(record["value"]), source=source)
    return result
