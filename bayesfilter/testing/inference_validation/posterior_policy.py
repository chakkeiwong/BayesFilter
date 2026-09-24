"""Framework-free validation and shared settings for posterior-only experiments.

These controls configure existing estimators and readiness checks. They cannot
change tuning membership, and their availability is not evidence of calibration.
"""
import math


def validate_posterior_options(options):
    precision = options.get("posterior_precision_settings", {})
    assessment = options.get("posterior_assessment_settings", {})
    if not isinstance(precision, dict) or not set(precision) <= {
            "batch_size", "min_batches", "lugsail_r", "lugsail_c"}:
        raise ValueError("posterior_precision_settings supports batch and lugsail controls only")
    for key, lower in (("batch_size", 1), ("min_batches", 2), ("lugsail_r", 1)):
        if key in precision and not (key == "batch_size" and precision[key] is None):
            if type(precision[key]) is not int or precision[key] < lower:
                raise ValueError(f"posterior_precision_settings {key} requires an integer >= {lower}")
    c = precision.get("lugsail_c", .5)
    if type(c) not in (int, float) or not math.isfinite(c) or not 0 <= c < 1:
        raise ValueError("posterior_precision_settings lugsail_c must be finite in [0,1)")
    floors = {"warmup_bulk_ess_min", "warmup_tail_ess_min",
              "retained_bulk_ess_min", "retained_tail_ess_min"}
    if not isinstance(assessment, dict) or not set(assessment) <= floors | {"warmup_consecutive_checks"}:
        raise ValueError("posterior_assessment_settings supports ESS floors and persistence only")
    for key in floors & set(assessment):
        value = assessment[key]
        if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
            raise ValueError(f"posterior_assessment_settings {key} must be finite and nonnegative")
    checks = assessment.get("warmup_consecutive_checks", 1)
    if type(checks) is not int or checks < 1:
        raise ValueError("posterior_assessment_settings warmup_consecutive_checks must be a positive integer")


def mean_precision_options(design):
    """Use the identical declared estimator in the controller and both comparators."""
    validate_posterior_options(design.options)
    return {"method": design.options.get("posterior_precision_method", "lugsail"),
            "jit_compile": design.device == "gpu",
            **design.options.get("posterior_precision_settings", {})}
