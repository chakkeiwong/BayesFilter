"""Endpoint checks for learned transports; posterior validation stays downstream.

TensorFlow is imported only inside the numerical probe so host-side consumers
can validate reports before selecting a GPU or initializing a framework.
"""
from __future__ import annotations

import math

POST_TRAINING_SCHEMA = "bayesfilter.neutra.post_training.v2"
PROBE_SCHEMA = "bayesfilter.neutra.post_training_probe.v2"
POST_TRAINING_POINTS = 1000  # Owner's post-run verification requirement.
POST_TRAINING_BATCH_SIZE = 20  # Exact divisor; September 22 GPU diagnostic.
QUANTILE_PROBABILITIES = (0., .01, .05, .25, .5, .75, .95, .99, 1.)
QUANTILE_DEFINITION = "nearest_lower_index_floor(p*(number_of_summarized_rows-1))"
EXCEEDANCE_LEVELS = (.1, .5, 1., 2., 5., 10.)  # Historical explanatory table.
SCALE_SLOPE_ALERT = 0.1  # Existing tenfold-attenuation alert; explanatory only.


def post_training_settings(config, *, sanity_only=False):
    """Short banks are explicit non-admission smoke/sanity exceptions."""
    rows = (config["validation"]["reliability_rows"]
            if sanity_only or config["role"] == "smoke" else POST_TRAINING_POINTS)
    batch = math.gcd(rows, POST_TRAINING_BATCH_SIZE)
    if batch <= 1:
        raise ValueError("post-training diagnostic requires batches larger than one")
    return {"rows": rows, "batch_size": batch}


def _serializable(value):
    if isinstance(value, dict):
        return {key: _serializable(v) for key, v in value.items()}
    if isinstance(value, list):
        return [_serializable(v) for v in value]
    return None if isinstance(value, float) and not math.isfinite(value) else value


class PostTrainingProbe:
    """One stable, batch-native graph per immutable map and probe shape."""

    def __init__(self, transport, bridge, beta, *, rows=POST_TRAINING_POINTS,
                 batch_size=None, jit_compile=True):
        import tensorflow as tf
        from bayesfilter.inference.tempered_transport_ensemble_tf import (
            pullback_gaussianization_diagnostic,
        )
        if type(rows) is not int or rows <= 1:
            raise ValueError("post-training probe requires more than one row")
        batch_size = math.gcd(rows, POST_TRAINING_BATCH_SIZE) if batch_size is None else batch_size
        if type(batch_size) is not int or batch_size <= 1 or rows % batch_size:
            raise ValueError("post-training probe requires whole batches larger than one")
        self.rows, self.dimension = rows, bridge.parameter_dim
        self.batch_size = batch_size
        self.jit_compile = bool(jit_compile)
        inner = getattr(transport, "inner", transport)
        self.scale_diagnostic_kinds = [
            "dsf_diagonal_log_derivative_no_conditional_cap" if getattr(s, "kind", None) == "naf_dsf"
            else "iaf_conditional_scale" for s in inner.stages]

        def evaluate(latent):
            diagnostic = pullback_gaussianization_diagnostic(
                transport, bridge, beta=beta, latent=latent)
            score = diagnostic.pullback_score_residual
            inner = getattr(transport, "inner", transport)
            z, means, slopes = latent, [], []
            # Static architecture traversal during tracing, never a sample loop.
            for i, layer in enumerate(inner.stages):
                if hasattr(layer, "scale_diagnostics"):
                    scale, slope = layer.scale_diagnostics(z)
                elif hasattr(layer, "weights"):
                    from bayesfilter.inference.neutra_transport_core import iaf_parameters
                    scale, _, slope, _, _ = iaf_parameters(layer, z)
                else:  # Explicit diagnostic fixtures, not an alternative training kernel.
                    scale, _ = layer._network(z)
                    slope = 1. - tf.square(scale / layer.s_max)
                means.append(tf.reduce_mean(scale, axis=0))
                slopes.append(tf.reduce_mean(tf.cast(
                    slope < tf.constant(SCALE_SLOPE_ALERT, tf.float64), tf.float64), axis=0))
                z, _ = layer.forward_and_logdet(z)
                if i + 1 < len(inner.stages):
                    z = inner._between_stage_permutation(z)
            scales = tf.stack(means)
            return {
                "finite": diagnostic.finite & tf.reduce_all(tf.math.is_finite(scales)),
                "valid_rows": diagnostic.valid_row_count,
                "latent": latent,
                "score_residual": score,
                # The historical r excludes the Gaussian normalizing constant.
                "log_ratio": diagnostic.pullback_log_density_residual - tf.constant(
                    .5*self.dimension*math.log(2.*math.pi), tf.float64),
                "mean_log_scale": scales,
                "fraction_slope_below_point1": tf.stack(slopes),
            }

        self.compiled = tf.function(evaluate,
            input_signature=(tf.TensorSpec([batch_size, self.dimension], tf.float64),),
            jit_compile=jit_compile, reduce_retracing=False)

        def summarize(latent, score, log_ratio, scale_means, scale_slopes):
            norm = tf.linalg.norm(score, axis=1)
            latent_norm = tf.linalg.norm(latent, axis=1)
            def statistics(values):
                ordered = tf.sort(values)
                indices = [int(math.floor(p*(rows-1))) for p in QUANTILE_PROBABILITIES]
                mean = tf.reduce_mean(values)
                variance = tf.reduce_mean(tf.square(values-mean))
                return {"min": ordered[0], "median": ordered[(rows-1)//2],
                    "mean": mean, "p95": ordered[indices[-3]], "p99": ordered[indices[-2]],
                    "max": ordered[-1], "quantiles": tf.gather(ordered, indices),
                    "rms": tf.sqrt(tf.reduce_mean(tf.square(values))), "std": tf.sqrt(variance),
                    "mean_standard_error": tf.sqrt(variance/tf.cast(rows-1, tf.float64))}
            score_statistics = statistics(norm)
            score_statistics["exceedance_fraction"] = {str(v): tf.reduce_mean(tf.cast(
                norm > tf.constant(v, tf.float64), tf.float64)) for v in EXCEEDANCE_LEVELS}
            score_statistics["fraction_larger_than_gaussian_score"] = tf.reduce_mean(tf.cast(
                norm > latent_norm, tf.float64))
            log_statistics = statistics(log_ratio)
            log_statistics["range"] = log_statistics["max"] - log_statistics["min"]
            return {"score_residual_norm": score_statistics,
                "latent_norm": statistics(latent_norm),
                "r_log_target_over_gaussian_up_to_constant": log_statistics,
                "score_residual_rms": tf.sqrt(tf.reduce_mean(tf.square(score))),
                "score_residual_rms_per_coordinate": tf.sqrt(tf.reduce_mean(tf.square(score), axis=0)),
                "score_residual_maximum_row_norm": tf.reduce_max(norm),
                "centered_log_density_rms": tf.sqrt(tf.reduce_mean(
                    tf.square(log_ratio-tf.reduce_mean(log_ratio)))),
                "mean_log_scale": tf.reduce_mean(scale_means, axis=0),
                "fraction_slope_below_point1": tf.reduce_mean(scale_slopes, axis=0)}

        scale_shape = [rows//batch_size, len(getattr(transport, "inner", transport).stages), self.dimension]
        self.summary_graph = tf.function(summarize, input_signature=(
            tf.TensorSpec([rows, self.dimension], tf.float64),
            tf.TensorSpec([rows, self.dimension], tf.float64), tf.TensorSpec([rows], tf.float64),
            tf.TensorSpec(scale_shape, tf.float64), tf.TensorSpec(scale_shape, tf.float64)),
            jit_compile=jit_compile, reduce_retracing=False)

    def latent_bank(self, seed):
        import tensorflow as tf
        with tf.device("/CPU:0"):
            return tf.random.stateless_normal([self.rows, self.dimension], seed, dtype=tf.float64)

    def batch(self, latent):
        """Host artifact boundary around one batch-native numerical call."""
        import tensorflow as tf
        try:
            return _serializable({key: value.numpy().tolist()
                                  for key, value in self.compiled(latent).items()})
        except tf.errors.InvalidArgumentError as error:
            if "had NaN" not in error.message and "had Inf" not in error.message:
                raise
            return {"finite": False, "valid_rows": None,
                "latent": latent.numpy().tolist(), "numerical_error": error.message[-1000:]}

    def summarize(self, blocks, seed):
        import tensorflow as tf
        if len(blocks)*self.batch_size != self.rows:
            raise ValueError("post-training probe is incomplete; a prefix cannot pass")
        valid_rows = (None if any(b.get("valid_rows") is None for b in blocks)
                      else sum(b["valid_rows"] for b in blocks))
        report = {"schema": PROBE_SCHEMA, "complete": True,
            "finite": all(b["finite"] for b in blocks), "valid_rows": valid_rows,
            "rows": self.rows, "batch_size": self.batch_size, "batches": len(blocks), "seed": list(seed),
            "dimension": self.dimension, "jit_compile": self.jit_compile,
            "traces": self.compiled.experimental_get_tracing_count(),
            "distribution": "standard_normal_base_draws_not_posterior_draws",
            "verification_scope": ("standard_1000_point" if self.rows == POST_TRAINING_POINTS
                                   else "short_or_nonstandard_diagnostic_only"),
            "quantile_definition": QUANTILE_DEFINITION,
            "quantile_probabilities": list(QUANTILE_PROBABILITIES),
            "score_residual_definition": "grad_z(log_pi_beta(T(z)) + log_abs_det_J_T(z)) + z",
            "geometry_role": "explanatory_only_no_calibrated_finite_cutoff",
            "score_residual_rms_definition": "per_coordinate_rms_not_mean_vector_norm",
            "scale_slope_alert": SCALE_SLOPE_ALERT, "scale_diagnostics": []}
        if not report["finite"]:
            report["summary_status"] = "invalid_rows_no_gaussianization_summary"
            return report
        tensors = [tf.concat([tf.constant(b[key], tf.float64) for b in blocks], axis=0)
                   for key in ("latent", "score_residual", "log_ratio")]
        tensors += [tf.constant([b[key] for b in blocks], tf.float64)
                    for key in ("mean_log_scale", "fraction_slope_below_point1")]
        def to_host(value):
            if isinstance(value, dict):
                return {key: to_host(v) for key, v in value.items()}
            return value.numpy().tolist()
        summary = self.summary_graph(*tensors)
        if not all(bool(tf.reduce_all(tf.math.is_finite(tensor)).numpy())
                   for tensor in tf.nest.flatten(summary)):
            report.update(finite=False, summary_status="nonfinite_summary")
            return report
        report.update(_serializable(to_host(summary)))
        for key in ("score_residual_norm", "latent_norm", "r_log_target_over_gaussian_up_to_constant"):
            report[key]["rows"] = self.rows
        # Equal, complete batches: each mean has the same number of rows.
        means, slopes = report.pop("mean_log_scale"), report.pop("fraction_slope_below_point1")
        report["scale_diagnostics"] = [{"stage": i, "mean_log_scale": mean,
                "kind": self.scale_diagnostic_kinds[i],
                "fraction_slope_below_point1": (None if self.scale_diagnostic_kinds[i].startswith("dsf_") else slope)}
                for i, (mean, slope) in enumerate(zip(means, slopes, strict=True))]
        report["summary_status"] = "all_declared_rows_summarized"
        return report

    def __call__(self, seed):
        latent = self.latent_bank(seed)
        return self.summarize([self.batch(latent[start:start+self.batch_size])
            for start in range(0, self.rows, self.batch_size)], seed)


def assess_post_training(*, state, decision, parity, probe, history, sanity_only=False):
    """Assemble numerical vetoes and repair signals without ranking maps."""
    rows = [r for r in history if r["status"] == "accepted"]
    clipped = sum(bool(r["clipping_applied"]) for r in rows)
    fraction = clipped / len(rows) if rows else None
    numerical_ok = (parity.get("passed") is True and probe.get("finite") is True
                    and probe.get("complete") is True
                    and probe.get("valid_rows") == probe.get("rows")
                    and probe.get("rows", 0) > 1)
    saturated = any(v is not None and v > 0 for row in probe["scale_diagnostics"]
                    for v in (row["fraction_slope_below_point1"] or ()))
    triggers = []
    if fraction is not None and fraction > .5:
        triggers.append("clipping_on_majority_of_updates")
    if saturated:
        triggers.append("scale_parameterization_saturation")
    if decision["status"] == "deterioration_repair_trigger":
        triggers.append("heldout_loss_deterioration")
    eligible = numerical_ok and not sanity_only and decision.get("hmc_trial_eligible", False)
    if not numerical_ok:
        action = "repair_numerical_failure"
    elif sanity_only:
        action = "complete_sanity_pilot_not_training_qualification"
    elif eligible:
        action = "fresh_fixed_transport_hmc_tuning"
    elif decision["status"] == "deterioration_repair_trigger":
        action = "repair_training"
    elif decision.get("at_training_cap"):
        action = "review_training_budget_or_recipe"
    else:
        action = "continue_training_within_budget"
    return {"schema": POST_TRAINING_SCHEMA,
        "training_state_hash": state["state_hash"],
        "map_hash": state["map"]["transport_state_hash"], "beta": state["map"]["beta"],
        "numerical_check_passed": numerical_ok, "hmc_trial_eligible": bool(eligible),
        "learning": {**{key: decision.get(key) for key in (
            "status", "continuing_improvement", "plateau_observed", "at_training_cap",
            "distinct_increment_observed")},
            "status": decision["status"] if numerical_ok else "numerically_invalid"},
        "clipping": {"observed_updates": len(rows), "clipped_updates": clipped,
            "fraction": fraction, "role": "majority_is_repair_trigger_not_optimizer_quality_test"},
        "geometry": probe, "repair_triggers": triggers, "next_action": action,
        "continuing_training_indicated": bool(decision.get("continuing_improvement")),
        "downstream": {"fresh_hmc_tuning": "not_run_for_this_checkpoint",
            "posterior_validation": "not_run_for_this_checkpoint",
            "mode_coverage": "not_established"},
        "training_quality_established": False, "posterior_qualified": False}


def require_post_training_assessment(record):
    """Actual HMC consumers require checks of the exact exported checkpoint."""
    assessment = record["assessment"]
    report = assessment.get("post_training")
    if not isinstance(report, dict) or report.get("schema") != POST_TRAINING_SCHEMA:
        raise ValueError("HMC trial requires the automatic post-training assessment")
    state_hash = record.get("training_checkpoint_hash")
    if (not state_hash or report.get("training_state_hash") != state_hash
            or record["frozen_transport"].get("training_state_hash") != state_hash
            or not assessment.get("current_map_hash")
            or report.get("map_hash") != assessment["current_map_hash"]
            or report.get("beta") != assessment["beta"]):
        raise ValueError("HMC trial post-training assessment does not match the exported map")
    probe = report.get("geometry", {})
    if (report.get("numerical_check_passed") is not True
            or report.get("hmc_trial_eligible") is not True
            or probe.get("schema") != PROBE_SCHEMA or probe.get("finite") is not True
            or probe.get("complete") is not True
            or probe.get("rows") != POST_TRAINING_POINTS or probe.get("valid_rows") != POST_TRAINING_POINTS
            or probe.get("summary_status") != "all_declared_rows_summarized"):
        raise ValueError("HMC trial post-training numerical check failed or is incomplete")
    norm, density = probe.get("score_residual_norm", {}), probe.get("r_log_target_over_gaussian_up_to_constant", {})
    required = [norm.get(key) for key in ("min", "median", "mean", "rms", "p95", "p99", "max",
                                         "fraction_larger_than_gaussian_score")]
    required += [norm.get("exceedance_fraction", {}).get("1.0"), density.get("range")]
    if (norm.get("rows") != POST_TRAINING_POINTS or density.get("rows") != POST_TRAINING_POINTS
            or any(type(x) not in (float, int) or not math.isfinite(x) for x in required)):
        raise ValueError("HMC trial post-training 1000-point statistics are missing or nonfinite")
    return report
