"""Assess test response separately from sampler findings and execution errors."""


def response(design, assessment):
    control = design.scenario.control
    finding = assessment.get("finding")
    if design.engine == "reference_mean":
        return {"control": control, "status": (
                    "reference_mean_rate_screen_passed" if assessment["rate_screen_passed"]
                    else "reference_mean_rate_screen_not_passed"),
                "rate_screen_passed": assessment["rate_screen_passed"],
                "unavailable": assessment["unavailable"], "power_established": False,
                "interpretation": "complete planned denominator; one cell alone does not establish calibrated null and defect rates"}
    if finding in {None, "invalid", "unavailable", "incomplete", "calibration_incomplete",
                   "no_verified_members", "posterior_incomplete"}:
        return {"control": control, "status": "unassessed", "reason": finding,
                "power_established": False,
                "interpretation": "missing or invalid evidence is not defect detection"}
    discrepancy = finding in {"mechanics_discrepancy", "inventory_discrepancy",
        "pipeline_discrepancy", "discrepancy_detected", "reference_discrepancy",
        "diagnostic_discrepancy"}
    if control in {"baseline", "noop"}:
        return {"control":control, "status":"discrepancy_under_baseline" if discrepancy else "baseline_observed",
                "power_established":False}
    if control in {"identity", "two_cycle"} or (control=="wrong_score" and design.engine=="invariance"):
        return {"control":control, "status":("discrepancy_under_invariance_preserving_control"
                                             if discrepancy else "invariance_preserving_control_observed"),
                "invariance_rejection_required":False, "power_established":False}
    return {"control":control, "status":"intended_discrepancy_detected" if discrepancy else "not_detected",
            "activation":assessment.get("mutation_activation", "explicit isolated diagnostic control"),
            "power_established":False,
            "interpretation":"one outcome is not a detection probability; execution failure never counts as detection"}
