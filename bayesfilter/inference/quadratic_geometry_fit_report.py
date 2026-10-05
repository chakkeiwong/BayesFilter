"""Host records for the completed low-rank geometry fit and decision program."""

from bayesfilter.inference.quadratic_geometry import (
    LowRankSPDQuadraticGeometryResult,
    _artifact_hash,
    _json_ready,
)
from bayesfilter.inference.quadratic_geometry_control_report import (
    center_refinement_report,
    exact_replay_report,
)
from bayesfilter.inference.quadratic_geometry_fit_tf import SOURCE_ROLES, STATUSES

_TOP_FIELDS = ("accepted", "status", "best_evaluated_position", "best_evaluated_value",
    "best_evaluated_score", "best_evaluated_index", "best_evaluated_source", "exact_evaluation_count",
    "center_refinement_accepted", "refined_center", "precision", "covariance", "q_basis",
    "linear_term", "intercept", "lambda0", "mu")


def _spectral_report(record):
    values, statistics, finite, positive = record
    low, high, condition = statistics.numpy().tolist()
    return {"finite": bool(finite), "positive": bool(positive), "min": low,
            "max": high, "condition_number": condition, "eigenvalues": values.numpy().tolist()}


def geometry_fit_report(raw, config, center_value, center_score_norm, *, holdout_rows):
    """Materialize the exact public result suffix after all decisions complete."""
    fit = raw["fit"]
    present = bool(raw["has_incumbent"])
    source = SOURCE_ROLES[int(raw["best_source"])] if present else None
    result = {"accepted": int(raw["status"]) == 1, "status": STATUSES[int(raw["status"])],
        "best_evaluated_position": raw["best_position"] if present else None,
        "best_evaluated_value": float(raw["best_value"]) if present else None,
        "best_evaluated_score": raw["best_score"] if present else None,
        "best_evaluated_index": int(raw["best_index"]) if present else None,
        "best_evaluated_source": source, "exact_evaluation_count": int(raw["evaluation_count"]),
        "center_refinement_accepted": False, "refined_center": None,
        "precision": None, "covariance": None, "q_basis": None, "linear_term": None,
        "intercept": None, "lambda0": None, "mu": None}
    if not bool(fit["finite"]):
        return {**result, "fit": {"status": "fit_nonfinite"}}
    if not bool(fit["design_resolved"]):
        return {**result, "fit": {"status": "fit_design_ill_conditioned",
            "score_design_rank": int(fit["score_design_rank"]),
            "score_design_retained_condition_number": float(fit["retained_design_condition"]),
            "score_design_roundoff_indicator": float(fit["design_roundoff_indicator"]),
            "score_design_roundoff_limit": float(fit["design_roundoff_limit"])}}
    report = {name: value.numpy().tolist() for name, value in fit.items()
              if name not in ("finite", "singular_values", "residual_sum_squares", "design_resolved",
                              "retained_design_condition", "design_roundoff_indicator", "design_roundoff_limit")}
    # Formatting uses precomputed scalar diagnostics supplied by the program.
    report.update(status="usable", fit_method="score_difference_linear_least_squares_with_value_intercept",
        optimizer_converged=True, optimizer_failed=False, optimizer_iterations=0,
        score_design_condition_number=float(raw["design_condition"]) if bool(raw["design_condition_defined"]) else None,
        score_lstsq_residual_sum_squares=float(fit["residual_sum_squares"]) if bool(raw["lstsq_residual_defined"]) else None)
    report["precision"] = None
    proposal = center_refinement_report(raw["refinement"], config, center_value, center_score_norm)
    result.update(fit=report, train_rmse=float(raw["train_rmse"]), holdout_count=holdout_rows,
        holdout_rmse=float(raw["holdout_rmse"]) if holdout_rows else None,
        holdout_threshold=float(raw["holdout_threshold"]) if holdout_rows else None,
        holdout_passed=bool(raw["holdout_passed"]), precision_eigen_summary=_spectral_report(raw["precision_summary"]),
        covariance_eigen_summary=_spectral_report(raw["covariance_summary"]), center_refinement=proposal,
        best_evaluated_replay=exact_replay_report(raw["replay"], evaluation_index=int(raw["replay_index"]), source=source))
    if result["accepted"]:
        result.update(precision=fit["precision"], covariance=raw["covariance"],
            q_basis=raw["basis"], linear_term=fit["linear_term"], intercept=float(fit["intercept"]),
            lambda0=float(fit["lambda0"]), mu=fit["mu"],
            refined_center=raw["refinement"]["refined_center"] if proposal["accepted"] else None,
            center_refinement_accepted=proposal["accepted"])
    return result


def geometry_fit_result(raw, config, center, scale, prefix_diagnostics, *, holdout_rows):
    """Restore the original result schema and hash after the native suffix.

    Prefix diagnostics describe already completed center/pilot/design evaluation.
    No numerical decision is made from that reporting metadata.
    """
    report = geometry_fit_report(raw, config, prefix_diagnostics["center_log_prob"],
        prefix_diagnostics["center_score_norm"], holdout_rows=holdout_rows)
    diagnostics = {**prefix_diagnostics, **{key: value for key, value in report.items() if key not in _TOP_FIELDS}}
    if bool(raw["fit"]["finite"]) and bool(raw["fit"]["design_resolved"]):
        diagnostics["artifact_hash"] = _artifact_hash({"config": config.payload(),
            "center": center, "scale": scale, "precision": raw["fit"]["precision"],
            "linear": raw["fit"]["linear_term"], "q_basis": raw["basis"], "diagnostics": {
                "finite_sample_count": prefix_diagnostics["finite_sample_count"],
                "train_rmse": report["train_rmse"], "holdout_rmse": report["holdout_rmse"]}})
    if not report["accepted"]:
        diagnostics["rejection_status"] = report["status"]
    return LowRankSPDQuadraticGeometryResult(dimension=center.shape[0], rank=raw["basis"].shape[1],
        center=center, scale=scale, diagnostics=diagnostics, **{key: report[key] for key in _TOP_FIELDS})


def geometry_fit_payload(result):
    """Format the full array payload using the completed native summaries.

    The public result method currently recomputes eigenvalues eagerly. This
    internal formatter retains its schema without repeating that numerical work.
    """
    names = ("accepted", "status", "dimension", "rank", "center_refinement_accepted",
        "intercept", "lambda0", "best_evaluated_value", "best_evaluated_source",
        "best_evaluated_index", "exact_evaluation_count", "diagnostics", "nonclaims",
        "center", "scale", "precision", "covariance", "q_basis", "linear_term",
        "refined_center", "best_evaluated_position", "best_evaluated_score")
    payload = {"schema": "bayesfilter.low_rank_spd_quadratic_geometry.v2",
               **{name: getattr(result, name) for name in names}}
    if result.mu is not None:
        payload["mu"] = result.mu
    if result.precision is not None:
        payload["precision_eigen_summary"] = result.diagnostics["precision_eigen_summary"]
    if result.covariance is not None:
        payload["covariance_eigen_summary"] = result.diagnostics["covariance_eigen_summary"]
    return _json_ready(payload)
