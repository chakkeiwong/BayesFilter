"""Assess test response separately from sampler findings and execution errors."""


def response(design, assessment):
    control = design.scenario.control
    finding = assessment.get("finding")
    discrepancy = finding in {"mechanics_discrepancy", "inventory_discrepancy",
        "pipeline_discrepancy", "discrepancy_detected"}
    if control in {"baseline", "noop"}:
        return {"control":control, "status":"discrepancy_under_baseline" if discrepancy else "baseline_observed",
                "power_established":False}
    if control in {"identity", "two_cycle"} or (control=="wrong_score" and design.engine=="invariance"):
        return {"control":control, "status":"invariance_preserving_control_observed",
                "invariance_rejection_required":False, "power_established":False}
    return {"control":control, "status":"intended_discrepancy_detected" if discrepancy else "not_detected",
            "activation":assessment.get("mutation_activation", "explicit isolated diagnostic control"),
            "power_established":False,
            "interpretation":"one outcome is not a detection probability; execution failure never counts as detection"}
