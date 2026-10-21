"""AR2 known-law calibration of the same diagnostics used on retained chains."""
import json
import math
from pathlib import Path
import time

import tensorflow as tf

from bayesfilter.inference.surrogate_hmc_program import ROOT, file_hash, write_json
from bayesfilter.inference.surrogate_hmc_equivalence import finite_report
from bayesfilter.testing.surrogate_hmc_ar2_sampling import CONTROLS_ROOT, load_reference, grid_law
from bayesfilter.testing.surrogate_hmc_closure_diagnostics import wilson
from bayesfilter.testing.surrogate_hmc_nonlinear_inference_metrics import scalar_functionals
from bayesfilter.testing.surrogate_hmc_scoped_assessment import critical_value, diagnostic_calculator

COST_ROOT = ROOT/"docs/plans/artifacts/surrogate-hmc-closure-20260922-ar2-validation-02/ar2-cost"
INITIAL_CALIBRATION = ROOT/"docs/plans/artifacts/surrogate-hmc-closure-20260922-ar2-validation-03/ar2-calibration"


def validate_cost_contract(cost):
    if (cost.get("qualified") is not True or cost.get("status") != "AR2_COMPLETE_TRANSITION_COST_MEASURED"
            or cost.get("precision", {}).get("precision_forecast_sufficient") is not True):
        raise ValueError("AR2 measured feasible precision schedule required")
    calibration_protocol(cost["precision"]["retained_looks"])
    for arm in ("A", "B"):
        for key in ("cpu_worker_seconds", "wall_seconds"):
            value = cost["conservative_inference_forecast"][arm][key]
            if not math.isfinite(value) or value <= 0:
                raise ValueError("finite positive inference cost forecast required")


def load_cost_contract(cost_root):
    cost_root = Path(cost_root)
    cost = json.loads((cost_root/"report.json").read_text())
    validate_cost_contract(cost)
    if cost["selection_sha256"] != file_hash(CONTROLS_ROOT/"selection.json"):
        raise ValueError("cost selection changed")
    manifest = json.loads((cost_root.parent/"run-manifest.json").read_text())
    for path in ("bayesfilter/testing/surrogate_hmc_ar2_sampling.py",
                 "bayesfilter/testing/surrogate_hmc_scoped_assessment.py",
                 "bayesfilter/testing/surrogate_hmc_mcse_randomization.py",
                 "bayesfilter/testing/surrogate_hmc_nonlinear_inference_metrics.py",
                 "bayesfilter/inference/hmc_posterior_diagnostics.py"):
        if file_hash(ROOT/path) != manifest["source_hashes"].get(path):
            raise ValueError("cost or prospective diagnostic source drift: "+path)
    return cost


def calibration_budget_check(manifest, cost, *, confirmation=False):
    # Reserve the full new calibration/check allowances and a subsequent
    # inference preflight before declaring the primary B experiment affordable.
    future = dict(cpu_worker_seconds=1030. if confirmation else 1880.,
                  gpu_seconds=1100. if confirmation else 2700.)
    caps = dict(cpu_worker_seconds=54000., gpu_seconds=226800.)
    available = {k: caps[k]-manifest["previous_charge"][k]-future[k] for k in caps}
    forecast = cost["conservative_inference_forecast"]["B"]
    if forecast["cpu_worker_seconds"] > available["cpu_worker_seconds"] or forecast["wall_seconds"] > available["gpu_seconds"]:
        raise ValueError("primary AR2 B/R forecast does not fit after calibration and checks")
    return dict(after_reserved_checks_and_calibration=available, primary_B_forecast=forecast,
                exact_gradient_A_allocation="deferred until actual B use leaves its forecast affordable")


def calibration_protocol(looks):
    if tuple(looks) not in ((2000, 4000), (4000, 8000)):
        raise ValueError("AR2 requires the cost stage's declared two-look schedule")
    return dict(groups=[("iid", 0., 0, 512), ("refresh_half", .5, 0, 512),
        ("refresh_nine_tenths", .9, 0, 512), ("sticky", .99, 0, 128),
        ("chain_offsets", 0., 1, 128), ("drift", 0., 2, 128)], looks=list(looks),
        seed_base=2026092300, function_count=13, family_count=1, family_alpha=.05,
        permutations=1023, minimum_admitted=64, failure_upper_limit=.05,
        negative_control_rejection_lower_limit=.95, false_reference_shift=1.25,
        law="normalized trapezoidal AR2 reference grid node masses; exact finite diagnostic law")


def reference_generator(count, arrays):
    table, functions, probability = grid_law(arrays)
    cdf = tf.cumsum(probability)
    cdf = tf.concat([cdf[:-1], tf.ones([1], tf.float64)], 0)

    @tf.function(input_signature=[tf.TensorSpec([2], tf.int32), tf.TensorSpec([], tf.float64),
                                 tf.TensorSpec([], tf.int32)], jit_compile=True, autograph=False)
    def generate(seed, rho, regime):
        uniforms = tf.random.stateless_uniform([count, 4, 2], seed, dtype=tf.float64)
        fresh = tf.reshape(tf.searchsorted(cdf, tf.reshape(uniforms[:, :, 0], [-1]), side="right"), [count, 4])
        indices = tf.scan(lambda previous, pair: tf.where(pair[1] >= rho, pair[0], previous),
                         (fresh[1:], uniforms[1:, :, 1]), initializer=fresh[0])
        indices = tf.concat([fresh[:1], indices], 0)
        points, values = tf.gather(table, indices), tf.gather(functions, indices)
        offsets = tf.constant([-.8, -.25, .25, .8], tf.float64)[None, :]
        drift = tf.linspace(tf.constant(-1.5, tf.float64), tf.constant(1.5, tf.float64), count)[:, None]
        shift = tf.cast(regime == 1, tf.float64)*offsets+tf.cast(regime == 2, tf.float64)*drift
        points += shift[:, :, None]*tf.constant([1., 0.], tf.float64)*arrays["scales"]
        # Negative controls deliberately spoil moments/CDFs. Predictions retain
        # their original values; a spoiled single parameter must still veto.
        values = tf.reshape(scalar_functionals(tf.reshape(points, [-1, 2]), tf.reshape(values[:, :, -2:], [-1, 2]),
                            arrays["means"], arrays["scales"], arrays["cuts"]), [count, 4, 13])
        return tf.transpose(points, [1, 0, 2]), tf.transpose(values, [1, 0, 2])
    return generate


def confirmation_evidence(previous, looks):
    """One fresh precision confirmation, with no change to the diagnostic."""
    expected = {row[0] for row in calibration_protocol(looks)["groups"]}
    failed = {name for name, row in previous.get("groups", {}).items() if row.get("passed") is not True}
    if (previous.get("status") != "AR2_UNCERTAINTY_REPAIR_REQUIRED"
            or set(previous.get("groups", {})) != expected
            or failed != {"refresh_nine_tenths"}
            or previous["protocol"]["looks"] != list(looks)):
        raise ValueError("confirmation requires the preserved single-regime precision shortfall")
    return {name: row for name, row in previous["groups"].items() if name not in failed}


def calibration_stage(output, cost_root, *, confirmation=False):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    cost_path = Path(cost_root)/"report.json"
    cost = load_cost_contract(cost_root)
    manifest = json.loads((output.parent.parent/"run-manifest.json").read_text())
    budget = calibration_budget_check(manifest, cost, confirmation=confirmation)
    protocol = calibration_protocol(cost["precision"]["retained_looks"])
    preserved, provenance = {}, None
    if confirmation:
        previous_path = INITIAL_CALIBRATION/"report.json"
        previous = json.loads(previous_path.read_text())
        preserved = confirmation_evidence(previous, protocol["looks"])
        old_manifest = json.loads((INITIAL_CALIBRATION.parent/"run-manifest.json").read_text())
        for path in ("bayesfilter/testing/surrogate_hmc_scoped_assessment.py",
                     "bayesfilter/testing/surrogate_hmc_mcse_randomization.py",
                     "bayesfilter/testing/surrogate_hmc_nonlinear_inference_metrics.py",
                     "bayesfilter/inference/hmc_posterior_diagnostics.py",
                     "bayesfilter/testing/surrogate_hmc_ar2_sampling.py"):
            if old_manifest["source_hashes"].get(path) != file_hash(ROOT/path):
                raise ValueError("diagnostic changed before precision confirmation: "+path)
        if previous["cost_report_sha256"] != file_hash(cost_path):
            raise ValueError("cost scope changed before precision confirmation")
        protocol.update(groups=[("refresh_nine_tenths", .9, 0, 1024)], seed_base=2027092300,
                        confirmation_only=True, further_unchanged_confirmation_allowed=False)
        provenance = dict(report=str(previous_path), sha256=file_hash(previous_path),
                          preserved_groups=list(preserved), failed_original_group=previous["groups"]["refresh_nine_tenths"],
                          pooling="none; the fresh 1024 trials alone must meet the original criterion")
    _, reference, arrays = load_reference()
    _, functions, probability = grid_law(arrays)
    truth = tf.reduce_sum(probability[:, None]*functions, 0)
    margins = tf.constant(reference["margins"], tf.float64)
    critical = critical_value()
    generate = reference_generator(protocol["looks"][-1], arrays)
    calculators = {n: diagnostic_calculator(n) for n in protocol["looks"]}
    report = dict(status="RUNNING", qualified=False, protocol=protocol, groups=preserved,
        prior_calibration=provenance, prospective_budget=budget,
        cost_report=str(cost_path), cost_report_sha256=file_hash(cost_path),
        truth=truth.numpy().tolist(), margins=margins.numpy().tolist(), critical_value=float(critical),
        reference_error="zero relative to the explicitly finite diagnostic law; continuous integration error is separate",
        diagnostic_callable="surrogate_hmc_scoped_assessment.diagnostic_calculator (also called by retained assess)",
        generator_jit_compile=True, batch_and_randomization_jit_compile=True, rank_diagnostic_jit_compile=False,
        rank_exception="repository tie-aware rank diagnostic graph", inference_launched=False)
    write_json(output, report)
    start = time.monotonic()
    for group_index, (name, rho, regime, replications) in enumerate(protocol["groups"]):
        counts = dict(any_veto=0, both_veto=0, admitted=0, interval_failure=0, admitted_failure=0, false_equivalence=0)
        with (output.parent/(name+"-replications.jsonl")).open("x") as stream:
            for index in range(replications):
                seed = [protocol["seed_base"]+group_index*10000+index, 0]
                points, values = generate(tf.constant(seed, tf.int32), tf.constant(rho, tf.float64), tf.constant(regime, tf.int32))
                rows = []
                for n in protocol["looks"]:
                    result = calculators[n](points[:, :n], values[:, :n], tf.constant([seed[0], 100+n], tf.int32))
                    radius = critical*result["mcse"]
                    difference = result["means"]-truth
                    qualified = bool(result["qualified"])
                    covered = bool(tf.reduce_all(tf.abs(difference) <= radius))
                    false = False
                    for function in (0, 7):
                        wrong = truth+tf.one_hot(function, 13, dtype=tf.float64)*1.25*margins
                        false |= qualified and bool(tf.reduce_all(tf.abs(result["means"]-wrong)+radius <= margins))
                    rows.append(dict(draws=n, qualified=qualified, all_functions_covered=covered, false_equivalence=false,
                        diagnostics=finite_report({key: value.numpy().tolist() for key, value in result.items()})))
                admitted = any(row["qualified"] for row in rows)
                counts["admitted"] += admitted
                counts["any_veto"] += any(not row["qualified"] for row in rows)
                counts["both_veto"] += all(not row["qualified"] for row in rows)
                counts["interval_failure"] += any(not row["all_functions_covered"] for row in rows)
                counts["admitted_failure"] += any(row["qualified"] and not row["all_functions_covered"] for row in rows)
                counts["false_equivalence"] += any(row["false_equivalence"] for row in rows)
                stream.write(json.dumps(dict(replication=index, seed=seed, looks=rows), allow_nan=False)+"\n")
                if time.monotonic()-start > (1050 if confirmation else 2600):
                    raise TimeoutError("AR2 calibration wall reservation reached")
        group = dict(counts=counts, replications=replications,
            any_look_veto=wilson(counts["any_veto"], replications), both_look_veto=wilson(counts["both_veto"], replications),
            unconditional_interval_failure=wilson(counts["interval_failure"], replications),
            admitted_interval_failure=wilson(counts["admitted_failure"], counts["admitted"]),
            false_equivalence=wilson(counts["false_equivalence"], replications))
        if regime == 0 and rho <= .9:
            group["passed"] = (counts["admitted"] >= 64 and group["unconditional_interval_failure"]["upper"] <= .05
                and group["admitted_interval_failure"]["upper"] <= .05 and group["false_equivalence"]["upper"] <= .05
                and (name != "iid" or group["any_look_veto"]["upper"] <= .05))
        else:
            group["passed"] = group["both_look_veto"]["lower"] >= .95
        report["groups"][name] = group
        write_json(output, report)
        print("AR2_CALIBRATION", name, group["passed"], flush=True)
    report.update(qualified=all(group["passed"] for group in report["groups"].values()),
        statistical_scope="per-regime Wilson bounds; empirical qualification, not an all-chain or all-model coverage theorem",
        ranking="NOT_ESTABLISHED", default_readiness=False,
        next_action="Existing master: public AR2 HMC tuning and fresh retained B/R comparison if scope qualification passes")
    report["status"] = "AR2_SCOPED_UNCERTAINTY_QUALIFIED" if report["qualified"] else "AR2_UNCERTAINTY_REPAIR_REQUIRED"
    return report
