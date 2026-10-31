"""Same-input numerical safety diagnosis, not training or efficacy evidence."""

import argparse
import hashlib
import json
from pathlib import Path
import time

from bayesfilter.score_study.moment_training_campaign import Campaign, save, tf


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    arguments = parser.parse_args()
    started = time.monotonic()
    campaign = Campaign(argparse.Namespace(output=arguments.output, budget_seconds=1500., smoke=False))
    from bayesfilter.highdim.ledh_backward_guide_tf import build_backward_guided_ledh_kernel
    from bayesfilter.score_study.moment_training_tf import make_moment_endpoint

    controls = dict(campaign.controls, moment_safety=True)
    design = tf.random.stateless_normal([64, 18], [262029901, 0], dtype=tf.float64)
    candidate = build_backward_guided_ledh_kernel(
        lambda parameter, direction: campaign.spec.model(parameter, direction)[0],
        None, dynamic_quadratic=True, horizon=5, particle_count=64,
        state_dimension=18, observation_dimension=9, parameter_dimension=3,
        dtype=tf.float64, jit_compile=True, return_trace=True, reset_design=design, **controls)
    candidate_endpoint = make_moment_endpoint(candidate, campaign.objective, 18)
    rows = []
    try:
        for dataset_seed, particle_seed in ((262023017, 274000034), (262023001, 274000002)):
            campaign.check_budget()
            data = campaign.dataset(dataset_seed)
            initial = campaign.particles_for(particle_seed)
            states, covariances, noises, d_states, d_covariances = initial
            precision, linear = (tf.zeros_like(value) for value in data["csmc"])
            saved = json.loads((arguments.baseline / f"dataset-{dataset_seed}.json").read_text())
            matches = {name: bool(tf.reduce_all(data[name] == tf.constant(saved[name], tf.float64)))
                       for name in ("observations", "label", "score_label")}
            if not all(matches.values()):
                raise ValueError("Saved inputs changed")

            def inputs(theta, direction):
                return (theta, direction, states, covariances, noises, data["observations"],
                        d_states, d_covariances, precision, linear)

            for name, kernel, endpoint in (("off", campaign.kernels["csmc"], campaign.endpoints["csmc"]),
                                           ("on", candidate, candidate_endpoint)):
                campaign.check_budget()
                value, derivative, trace = kernel(*inputs(campaign.theta, tf.one_hot(0, 3, dtype=tf.float64)))
                output = endpoint(*inputs(campaign.theta, tf.one_hot(0, 3, dtype=tf.float64)), data["label"])
                save(arguments.output / f"trace-{name}-{dataset_seed}.json", trace)
                nonfinite = [{key: int(tf.reduce_sum(tf.cast(~tf.math.is_finite(tensor), tf.int32)))
                    for key, tensor in step.items() if tensor.dtype.is_floating
                    and not bool(tf.reduce_all(tf.math.is_finite(tensor)))} for step in trace]
                row = dict(dataset_seed=dataset_seed, particle_seed=particle_seed, moment_safety=name,
                    output=output, inputs_match=matches, trace_nonfinite=nonfinite,
                    covariance_residual=[step["higher_moment_affine_restoration_relative_covariance_residual"]
                                         for step in trace],
                    shape_available=[step["higher_moment_materialized_shape_available"] for step in trace],
                    ess=[1 / tf.reduce_sum(step["posterior_weights"] ** 2) for step in trace],
                    feasibility=[step["higher_moment_minimum_finite_particle_upper_margin"] for step in trace],
                    finite_differences=[])
                if name == "off":
                    reference = json.loads((arguments.baseline / f"trace-{dataset_seed}.json").read_text())
                    row["unchanged_states"] = all(bool(tf.reduce_all(step["states_after_reset"] ==
                        tf.constant(old["states_after_reset"], tf.float64))) for step, old in zip(trace, reference))
                if name == "on" and bool(output["valid"]):
                    for coordinate in range(3):
                        direction = tf.one_hot(coordinate, 3, dtype=tf.float64)
                        score_output = endpoint(*inputs(campaign.theta, direction), data["label"])
                        for step_size in (1e-4, 1e-5):
                            campaign.check_budget()
                            plus = endpoint(*inputs(campaign.theta + step_size * direction, direction), data["label"])
                            minus = endpoint(*inputs(campaign.theta - step_size * direction, direction), data["label"])
                            difference = (plus["value"] - minus["value"]) / (2 * step_size)
                            row["finite_differences"].append(dict(coordinate=coordinate, step_size=step_size,
                                valid=score_output["valid"] & plus["valid"] & minus["valid"],
                                analytical=score_output["score"], finite_difference=difference,
                                normalized_error=tf.abs(difference - score_output["score"]) /
                                    tf.maximum(1., tf.abs(score_output["score"]))))
                rows.append(row)
                save(arguments.output / "results.json", rows)
                print(json.dumps(dict(dataset_seed=dataset_seed, moment_safety=name,
                    valid=bool(output["valid"]), nonfinite_steps=[index + 1 for index, errors in enumerate(nonfinite) if errors],
                    maximum_covariance_residual=max(float(item) for item in row["covariance_residual"])), allow_nan=False), flush=True)
        candidate_rows = [row for row in rows if row["moment_safety"] == "on"]
        verdict = dict(candidate_valid=all(bool(row["output"]["valid"]) for row in candidate_rows),
            traces_finite=all(not errors for row in candidate_rows for errors in row["trace_nonfinite"]),
            covariance_pass=all(float(value) <= 1e-6 for row in candidate_rows for value in row["covariance_residual"]),
            derivatives_pass=all(len(row["finite_differences"]) == 6 and all(bool(check["valid"])
                and float(check["normalized_error"]) <= 1e-3 for check in row["finite_differences"]) for row in candidate_rows),
            no_default_or_efficacy_promotion=True)
        save(arguments.output / "verdict.json", verdict)
        print(json.dumps(verdict), flush=True)
    finally:
        paths = [Path(__file__), Path("bayesfilter/highdim/higher_moment_contract_e.py"),
                 Path("bayesfilter/highdim/ledh_unified_correction_tf.py"),
                 Path("bayesfilter/highdim/moment_safety_tf.py"), Path("bayesfilter/score_study/moment_training_tf.py")]
        manifest = dict(campaign.manifest, controls=controls, baseline=str(arguments.baseline),
            plan="docs/plans/ledh-moment-collapse-repair-20261020.md", role="same-input numerical repair diagnostic",
            seeds=[[262023017, 274000034], [262023001, 274000002]], training_updates=0,
            result=str(arguments.output / "results.json"), wall_seconds=time.monotonic() - started,
            repair_source_hashes={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
            allocator=tf.config.experimental.get_memory_info("GPU:0"))
        save(arguments.output / "manifest.json", manifest)


if __name__ == "__main__":
    main()
