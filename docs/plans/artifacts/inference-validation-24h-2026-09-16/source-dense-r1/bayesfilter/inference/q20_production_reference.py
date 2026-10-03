"""Independent full-support prior importance integration for the q20 target.

For theta~p0 and W=L(theta), E_pi[f]=E_p0[W*f]/E_p0[W]. The bridge's
finite likelihood bound and Gaussian prior give finite weighted polynomial
moments. Replicate MCSE and quantile intervals remain asymptotic; the finite
bank/concentration/stability checks cannot prove global mode coverage.
"""
from __future__ import annotations

import math
from pathlib import Path
import shutil

import tensorflow as tf
import tensorflow_probability as tfp

from bayesfilter.inference.q20_production_config import frozen_scope_hash, scoped_seed, write_json
from bayesfilter.inference.q20_production_comparison import quantity_layout, quantity_key
from bayesfilter.inference.q20_production_training import source_snapshot
from bayesfilter.runtime.durable_tensor_checkpoint import DurableTensorCheckpoint


def _weighted_quantile(x, weights, probability):
    order = tf.argsort(x, stable=True)
    x, w = tf.gather(x, order), tf.gather(weights, order)
    cumulative = tf.cumsum(w) / tf.reduce_sum(w)
    index = tf.minimum(tf.searchsorted(cumulative, tf.constant([probability], tf.float64))[0], tf.size(x)-1)
    return x[index]


def importance_summary(config, values, log_weights):
    """Post-run reporting: equal-sized banks, replicate influence MCSE.

    Eager TensorFlow here is a reporting exception; repeated target evaluations
    use the stable compiled batch kernel below.
    """
    values, logs = tf.convert_to_tensor(values, tf.float64), tf.convert_to_tensor(log_weights, tf.float64)
    if values.shape.rank != 3 or logs.shape != values.shape[:2] or values.shape[0] < 2:
        raise ValueError("reference requires [bank,row,coordinate] and matching log weights")
    tf.debugging.assert_all_finite(values, "reference proposals")
    tf.debugging.assert_all_finite(logs, "reference log weights")
    banks = int(values.shape[0])
    weights = tf.exp(logs - tf.reduce_max(logs))
    bank_weights = tf.exp(logs - tf.reduce_max(logs, axis=1, keepdims=True))
    sums = tf.reduce_sum(weights, axis=1)
    flat, w = tf.reshape(values, [-1, values.shape[-1]]), tf.reshape(weights, [-1])
    normalized = w / tf.reduce_sum(w)
    mean = tf.reduce_sum(flat * normalized[:, None], axis=0)
    sd = tf.sqrt(tf.reduce_sum(tf.square(flat-mean) * normalized[:, None], axis=0))
    ess = tf.square(tf.reduce_sum(bank_weights, axis=1)) / tf.reduce_sum(tf.square(bank_weights), axis=1)
    bank_mass_cv = tf.math.reduce_std(sums) / tf.reduce_mean(sums)
    quantities = {}
    tail_ok = True
    for row in quantity_layout(config):
        coordinate, probability = row["coordinate"], row["probability"]
        if coordinate is None:
            f = tf.cast(values[..., config["posterior"]["event_coordinate"]] > 0, tf.float64)
            estimate = tf.reduce_sum(weights*f) / tf.reduce_sum(weights)
        elif row["kind"] == "quantile":
            estimate = _weighted_quantile(flat[:, coordinate], w, probability)
            f = tf.cast(values[..., coordinate] <= estimate, tf.float64)
        else:
            f = values[..., coordinate]
            estimate = mean[coordinate]
        if row["kind"] == "quantile":
            bank_estimates = tf.stack([_weighted_quantile(values[i,:,coordinate], bank_weights[i], probability) for i in range(banks)])
            mcse = tf.math.reduce_std(bank_estimates) / tf.sqrt(tf.cast(banks-1, tf.float64))
        else:
            influence = tf.reduce_sum(weights*(f-estimate), axis=1) / tf.reduce_mean(sums)
            mcse = tf.math.reduce_std(influence) / tf.sqrt(tf.cast(banks-1, tf.float64))
        # No zero uncertainty for an unvisited sign or quantile tail. Effective
        # tail counts are weight-concentration screens, not binomial sample sizes.
        tail_valid = True
        if coordinate is None or row["kind"] == "quantile":
            for indicator in (f, 1-f):
                part = weights*indicator
                count = tf.square(tf.reduce_sum(part)) / tf.reduce_sum(tf.square(part))
                tail_valid = tail_valid and bool(tf.math.is_finite(count) & (count >= config["reference"]["minimum_tail_rows"]))
        tail_ok = tail_ok and tail_valid
        scale = None if coordinate is None else float(sd[coordinate])
        allowance = config["comparison"]["reference_error_fraction"] * (
            config["posterior"]["event_mcse"] if coordinate is None else scale *
            config["posterior"]["quantile_mcse_sd" if row["kind"] == "quantile" else "mean_mcse_sd"])
        uncertainty = float(mcse)
        valid = tail_valid and math.isfinite(uncertainty) and uncertainty > 0 and math.isfinite(float(estimate))
        quantities[quantity_key(row["name"], row["kind"], probability)] = {
            "name": row["name"], "kind": row["kind"], "probability": probability,
            "estimate": float(estimate), "mcse": uncertainty if valid else None,
            "posterior_sd": scale, "allowance": allowance, "valid": valid,
            "precision_passed": bool(valid and uncertainty <= allowance)}
    return {"quantities": quantities, "bank_ess": ess.numpy().tolist(),
            "minimum_bank_ess": float(tf.reduce_min(ess)), "bank_normalizer_cv": float(bank_mass_cv),
            "maximum_normalized_weight": float(tf.reduce_max(normalized)),
            "concentration_passed": bool(tf.reduce_all(tf.math.is_finite(ess) & (ess >= config["reference"]["ess_min"]) & (sums > 0))),
            "tails_observed": tail_ok, "precision_passed": all(r["precision_passed"] for r in quantities.values())}


def run_reference(config, bridge, root, *, resume_chunks=None):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=False)
    chunks = root / "chunks"
    if resume_chunks is not None:
        shutil.copytree(resume_chunks, chunks)
    r = config["reference"]
    batch, dimension = r["batch_size"], bridge.parameter_dim
    @tf.function(input_signature=(tf.TensorSpec([batch, dimension], tf.float64),),
                 jit_compile=config["jit_compile"], reduce_retracing=False)
    def evaluate(points):
        likelihood, score, prior, prior_score, status = bridge.component_terms(points)
        valid = (status["valid_pre_regularized_score"] & (status["status_code"] == 0)
                 & tf.reduce_all(tf.math.is_finite(score), axis=-1))
        return likelihood, valid
    scope = {"frozen_scope_hash": frozen_scope_hash(config), "target_signature": bridge.target_signature,
             "sources": source_snapshot(), "seed": scoped_seed(config, "independent-reference")}
    values, logs, history = [[] for _ in range(r["banks"])], [[] for _ in range(r["banks"])], []
    previous = None
    critical = float(tfp.distributions.Normal(tf.constant(0., tf.float64), tf.constant(1., tf.float64)).quantile(
        (1 + config["comparison"]["interval_probability"]) / 2))
    with DurableTensorCheckpoint(chunks, scope) as store:
        for rung in r["rungs"]:
            for bank in range(r["banks"]):
                for offset in range(len(values[bank])*batch, rung, batch):
                    seed = scoped_seed(config, "independent-reference", bank, offset)
                    def compute():
                        with tf.device("/CPU:0"):
                            z = tf.random.stateless_normal([batch, dimension], seed, dtype=tf.float64)
                            points = bridge.prior_center + tf.sqrt(tf.constant(bridge.prior_variance, tf.float64))*z
                        likelihood, valid = evaluate(points)
                        tf.debugging.assert_equal(tf.reduce_all(valid), True, message="invalid reference target; cannot discard prior proposals")
                        tf.debugging.assert_all_finite(likelihood, "reference likelihood")
                        return points, likelihood
                    points, likelihood = store.run(f"bank-{bank}-offset-{offset}", {"seed": seed, "batch": batch}, compute)
                    values[bank].append(points)
                    logs[bank].append(likelihood)
            summary = importance_summary(config, tf.stack([tf.concat(x,0) for x in values]), tf.stack([tf.concat(x,0) for x in logs]))
            stability = {}
            if previous is not None:
                for key, row in summary["quantities"].items():
                    before = previous["quantities"][key]
                    valid = row["mcse"] is not None and before["mcse"] is not None
                    # Cumulative rungs are correlated. Sum of standard errors
                    # is a conservative scale bound; no independent-pair claim.
                    bound = abs(row["estimate"]-before["estimate"]) + critical*(row["mcse"]+before["mcse"]) if valid else None
                    stability[key] = {"bound": bound, "passed": bool(valid and bound <= 2*row["allowance"])}
            qualified = bool(previous is not None and summary["precision_passed"] and summary["concentration_passed"]
                             and summary["tails_observed"] and all(x["passed"] for x in stability.values()))
            history.append({"rows_per_bank": rung, **summary, "stability": stability, "qualified": qualified})
            write_json(root / f"rung-{rung}.json", history[-1])
            previous = summary
            if qualified:
                break
    result = {"schema": "bayesfilter.q20.prior_importance_reference.v1", **summary,
              "status": "reference_qualified" if qualified else "reference_unqualified_at_cap", "qualified": qualified,
              "target_signature": bridge.target_signature, "frozen_scope_hash": frozen_scope_hash(config),
              "method": "full_support_prior_importance_independent_banks", "rungs": history,
              "error_interpretation": "asymptotic_MCSE_and_finite_bank_stability_not_certified_global_coverage",
              "proposal_generation": "batched_stateless_CPU_single_process_exception_in_reviewed_plan",
              "likelihood_implementation": "shared_checked_target; independent_integration_only",
              "role": config["role"], "sources": scope["sources"]}
    write_json(root / "result.json", result)
    return result
