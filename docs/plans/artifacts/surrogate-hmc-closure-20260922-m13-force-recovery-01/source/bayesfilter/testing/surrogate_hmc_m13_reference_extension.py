"""Bounded independent M13 likelihood diagnostic; no posterior qualification."""
import math
from pathlib import Path
import time

import tensorflow as tf
import tensorflow_probability as tfp

from bayesfilter.inference.surrogate_hmc_program import ROOT, file_hash, write_json
from bayesfilter.testing.surrogate_hmc_m13_reference import likelihood_interval, finite_list
from bayesfilter.testing.surrogate_hmc_m13_trajectory_reference import trajectory_importance_reference

PLAN = "docs/plans/surrogate-hmc-m13-reference-extension-2026-09-24.md"
PARTICLES, REPLICATES = 4096, 8
SEED = 2026092401


def reference_extension_stage(output):
    from bayesfilter.testing.surrogate_hmc_m13_controls import partition_fixture
    fixture = partition_fixture("calibration")
    out = Path(output).parent
    out.mkdir(parents=True, exist_ok=True)
    data = out/"data.json"
    write_json(data, dict(model=fixture.model_definition(),
        observations=fixture.observations.numpy().tolist(), data_seed=1701,
        role="reused calibration diagnostic; not untouched claim"))
    files = ["bayesfilter/testing/surrogate_hmc_m13_reference_extension.py",
             "bayesfilter/testing/surrogate_hmc_m13_wrapped_reference.py",
             "bayesfilter/testing/surrogate_hmc_m13_trajectory_reference.py",
             "bayesfilter/testing/surrogate_hmc_m13_reference.py",
             "bayesfilter/testing/surrogate_hmc_m13_controls.py",
             "experiments/dpf_implementation/tf_tfp/fixtures/range_bearing_tf.py"]
    sources = {name: file_hash(ROOT/name) for name in files}
    extension = trajectory_importance_reference(fixture, PARTICLES, REPLICATES, full_support=True)
    old = trajectory_importance_reference(fixture, PARTICLES, REPLICATES)
    critical = float(tfp.distributions.StudentT(df=tf.constant(REPLICATES-1., tf.float64),
        loc=tf.constant(0., tf.float64), scale=tf.constant(1., tf.float64)
        ).quantile(tf.constant(.975, tf.float64)).numpy())
    rows = []
    for index, r in enumerate((0., math.log(2.), 1.5, 3., 4.5)):
        theta = tf.constant([0., r], tf.float64)
        seed = tf.constant([SEED, 8301+4*index], tf.int32)
        start = time.monotonic()
        values = extension(theta, seed)
        logs = values["log_likelihood"]
        interval = likelihood_interval(logs, critical)
        finite = bool(tf.reduce_all(tf.math.is_finite(logs)).numpy())
        valid = bool(tf.reduce_all(values["valid"]).numpy()) and finite
        row = dict(theta=theta.numpy().tolist(), seed=seed.numpy().tolist(), valid=valid,
            angular_sigma=float((tf.sqrt(fixture.R[1, 1])*tf.exp(theta[1])).numpy()),
            likelihood=interval, log_likelihood_mcse_delta=(
                interval["coefficient_of_variation"]/math.sqrt(REPLICATES) if finite else None),
            minimum_ess=finite_list(values["minimum_ess"]),
            maximum_normalized_weight=finite_list(values["proposal_maximum_normalized_weight"]),
            fit_first=finite_list(values["proposal_first_diagnostics"]),
            fit_full=finite_list(values["proposal_full_diagnostics"]))
        if index < 2:
            baseline = old(theta, seed)
            row["old_reference_valid"] = bool(tf.reduce_all(baseline["valid"]).numpy())
            difference = tf.reduce_max(tf.abs(logs-baseline["log_likelihood"]))
            row["old_new_max_log_difference_common_seeds"] = (
                float(difference.numpy()) if bool(tf.math.is_finite(difference).numpy()) else None)
            row["old_domain_agreement"] = (valid and row["old_reference_valid"] and
                row["old_new_max_log_difference_common_seeds"] is not None and
                row["old_new_max_log_difference_common_seeds"] <= 1e-7)
        row["wall_seconds"] = time.monotonic()-start
        rows.append(row)
        write_json(out/f"point-{index}.json", row)
        print("M13 reference extension", row["theta"], "valid", valid,
              "log MCSE", row["log_likelihood_mcse_delta"], flush=True)
    if any(file_hash(ROOT/name) != digest for name, digest in sources.items()):
        raise ValueError("source changed during M13 reference extension")
    agreement = all(row.get("old_domain_agreement", True) for row in rows)
    valid = all(row["valid"] for row in rows)
    return dict(status=("M13_REFERENCE_EXTENSION_DIAGNOSTIC_COMPLETED" if valid and agreement
                        else "M13_REFERENCE_EXTENSION_REPAIR_REQUIRED"),
        qualified=False, diagnostic_completed=True, all_points_valid=valid,
        old_domain_agreement=agreement, posterior_reference_qualified=False,
        likelihood_reference_qualified=False, inference_launched=False, all_models_validated=False,
        posterior_comparisons="NOT_RUN", prior_support="R^2", parameter_prior="independent N(0,log(2)^2)",
        particles=PARTICLES, replicates=REPLICATES, dtype="float64",
        reference_exception=True, jit_compile=True, data_sha256=file_hash(data),
        source_sha256=sources, mathematical_likelihood_independent_of_dctr=True,
        points=rows, plan=PLAN, ranking_supported=False,
        next_action="Existing master: inspect proposal validity and MCSE, then qualify posterior-wide integration and a declared acceptance endpoint.")
