"""Read-only M26 posterior diagnosis and exact confirmation-cost planning.

Saved-chain reassessment is explanatory; it never changes the actual stop.
SciPy is used only as an independent statistical planning reference.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    started = time.monotonic()
    from bayesfilter.testing.inference_validation.storage import (
        read_json, read_tensor, write_json, file_hash, json_ready)
    from bayesfilter.testing.inference_validation.designs import ValidationDesign
    from bayesfilter.testing.inference_validation.execution import configure_worker, source_state

    design = ValidationDesign.from_payload(read_json(
        args.root / "rotated_gaussian-autocorrelation-pilot-design.json"))
    runtime = configure_worker(design)
    import tensorflow as tf
    from scipy.stats import beta, binom
    from bayesfilter.inference.hmc_convergence import rank_normalized_split_rhat_summary
    from bayesfilter.inference.hmc_posterior_diagnostics import rank_normalized_bulk_tail_ess
    from bayesfilter.inference.hmc_precision import mean_precision, quantile_precision

    path = args.root / "rotated_gaussian-autocorrelation-pilot-r1/fits/replication-0000"
    native = read_json(path / "pipeline.json")
    member = next(m for m in native["members"] if m["status"] == "assessed")
    records = []
    for arm, tensor_path in (("saved_warmup", member["warmup_path"]),
                             ("independent_fixed", member["fixed_comparator"]["draws_path"])):
        values = read_tensor(tensor_path)
        windows = sorted({1000, 5000, 10000, int(values.shape[0])})
        for count in windows:
            if count > values.shape[0]:
                continue
            x = values[-count:]
            rhat = rank_normalized_split_rhat_summary(x, rhat_max=1.05)
            ess = rank_normalized_bulk_tail_ess(tf.transpose(x, (1, 0, 2)))
            means = {method: mean_precision(x, method=method, jit_compile=False)
                     for method in ("lugsail", "autocorrelation")}
            median = quantile_precision(x, .5)
            records.append({"arm": arm, "window_per_chain": count,
                "tensor": tensor_path, "sha256": file_hash(tensor_path),
                "rhat": rhat["max_finite_rhat"], "bulk_ess": ess["bulk"], "tail_ess": ess["tail"],
                "mean_mcse": {method: report["mcse"] for method, report in means.items()},
                "median_mcse": median["mcse"],
                "plug_in_retained_requirement": {
                    method: tf.reduce_max(tf.concat((report["mcse"], median["mcse"]), axis=0)**2)
                            * count / design.mcse_tolerance**2
                    for method, report in means.items()},
                "role": "explanatory_only_not_a_replacement_stop"})

    observed_costs = {}
    for target in ("gaussian", "beta_binomial", "rotated_gaussian"):
        costs = [read_json(args.root / f"{target}-{method}-pilot-r1/execution.json")["cpu_worker_seconds"]
                 for method in ("lugsail", "autocorrelation")]
        observed_costs[target] = {"min_seconds": min(costs), "max_seconds": max(costs), "observations": 2}
    proposals = []
    for count in (64, 128, 256, 384, 512):
        critical = next(k for k in range(1, count+1) if beta.ppf(.025, k, count-k+1) >= .90)
        probabilities = {str(p): float(binom.sf(critical-1, count, p)) for p in (.90, .95, .975)}
        proposals.append({"fits_per_model_and_estimator": count, "required_successes": critical,
            "passing_lower_bound": float(beta.ppf(.025, critical, count-critical+1)),
            "pass_probability_at_hypothetical_true_rate": probabilities,
            "all_four_quantities_union_lower_bound_at_true_rate_095": max(0., 1.-4.*(1.-probabilities["0.95"])),
            "cpu_hours_per_model_one_estimator": {target: count*cost["max_seconds"]/3600.
                for target, cost in observed_costs.items()},
            "cpu_hours_three_models_one_estimator": count*sum(c["max_seconds"] for c in observed_costs.values())/3600.})

    result = {"source": source_state(), "runtime": runtime,
        "selected_member": {key: member[key] for key in ("candidate_id", "L", "epsilon")},
        "actual_stop": {key: member["posterior"][key] for key in
            ("passed", "hard_vetoes", "warmup_cap_hit", "retained_cap_hit",
             "warmup_results_per_chain", "retained_results_per_chain")},
        "saved_chain_diagnosis": records, "pilot_costs": observed_costs,
        "confirmation_planning": proposals,
        "planning_derivation": "Invert the exact binomial tail at alpha/2=.025 for the inherited two-sided 95% interval; sum Binomial(n,p) probability from the first passing success count to n.",
        "limits": ["hypothetical rates are sensitivity cases, not inferred from these pilots",
            "four-quantity joint bound uses the union bound and requires all marginal true rates >=.95",
            "pointwise screens do not establish simultaneous 95% coverage or anytime validity",
            "two observed costs do not establish a runtime ceiling",
            "larger windows do not retrospectively pass the failed actual stop"],
        "elapsed_seconds": time.monotonic()-started}
    write_json(args.output, json_ready(result))
    print({"saved_windows": len(records), "planning_inventories": len(proposals),
           "elapsed_seconds": result["elapsed_seconds"]}, flush=True)


if __name__ == "__main__":
    main()
