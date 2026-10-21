"""Prospective AR2 cost and posterior diagnostics; public tuners own HMC tuning."""
import json
from pathlib import Path
import resource
import time

import tensorflow as tf

from bayesfilter.inference.surrogate_hmc_program import ROOT, file_hash, write_json
from bayesfilter.inference.surrogate_hmc_dctr_selection import load_selection
from bayesfilter.testing.surrogate_hmc_ar2_controls import REFERENCE_ROOT, PARTICLE_COUNT, ar2_scope, source_hashes
from bayesfilter.testing.surrogate_hmc_ar2_consumer import make_ar2_consumer
from bayesfilter.testing.surrogate_hmc_linear_posterior_reference import trapezoid_weights
from bayesfilter.testing.surrogate_hmc_nonlinear_inference_metrics import scalar_functionals
from bayesfilter.testing.surrogate_hmc_scoped_assessment import critical_value

CONTROLS_ROOT = ROOT/"docs/plans/artifacts/surrogate-hmc-closure-20260922-structural-04/ar2-controls"


def load_reference(partition="claim"):
    report = json.loads((REFERENCE_ROOT/"report.json").read_text())
    manifest = json.loads((REFERENCE_ROOT.parent/"run-manifest.json").read_text())
    for path in ("bayesfilter/testing/surrogate_hmc_ar2_reference.py", "bayesfilter/testing/surrogate_hmc_linear_posterior_reference.py"):
        if file_hash(ROOT/path) != manifest["source_hashes"].get(path):
            raise ValueError("reference source drift: "+path)
    reference = report["partitions"][partition]
    if not report["qualified"] or not reference["qualified"]:
        raise ValueError("qualified AR2 reference required")
    data = json.loads((REFERENCE_ROOT/"datasets.json").read_text())[partition] | dict(attempt=1, replicate=0)
    arrays = {name: tf.io.parse_tensor(tf.io.read_file(str(REFERENCE_ROOT/partition/(name+".tensor"))), tf.float64)
              for name in ("axis", "density", "predictive", "means", "scales", "cuts")}
    return data, reference, arrays


def grid_law(arrays):
    axis = arrays["axis"]
    weights = trapezoid_weights(axis)
    probability = tf.reshape(arrays["density"]*weights[:, None]*weights[None, :], [-1])
    tf.debugging.assert_all_finite(probability, "AR2 reference grid probability")
    tf.debugging.assert_non_negative(probability)
    tf.debugging.assert_positive(tf.reduce_sum(probability))
    probability /= tf.reduce_sum(probability)
    a, b = tf.meshgrid(axis, axis, indexing="ij")
    points = tf.stack([tf.reshape(a, [-1]), tf.reshape(b, [-1])], -1)
    functions = scalar_functionals(points, tf.reshape(arrays["predictive"], [-1, 2]),
                                   arrays["means"], arrays["scales"], arrays["cuts"])
    return points, functions, probability


def selected_consumer(data):
    controls_report = json.loads((CONTROLS_ROOT/"report.json").read_text())
    if (controls_report.get("qualified") is not True or controls_report.get("precision_passed") is not True
            or controls_report.get("stress_healthy") is not True
            or controls_report["reference_report_sha256"] != file_hash(REFERENCE_ROOT/"report.json")):
        raise ValueError("scoped AR2 controls/precision/reference prerequisite failed")
    saved = json.loads((CONTROLS_ROOT/"selection.json").read_text())
    consumer = make_ar2_consumer(data, controls=saved["controls"], particle_count=PARTICLE_COUNT, batch_size=4)
    load_selection(CONTROLS_ROOT/"selection.json", expected_scope=ar2_scope(),
                   expected_source_hashes=source_hashes(consumer.fixture))
    return consumer


def precision_design(variance, margins, allowance):
    tf.debugging.assert_all_finite(variance, "reference function variance")
    tf.debugging.assert_positive(margins-allowance)
    required = critical_value()**2*variance/(margins-allowance)**2
    # Margin sizing is a planning approximation, not a power guarantee.
    target = float(tf.reduce_max(required))  # four chains, assumed IAT=4 cancel
    looks = next((pair for pair in ((2000, 4000), (4000, 8000), (8000, 16000)) if pair[1] >= target), (8000, 16000))
    return dict(required_effective_draws_total=required.numpy().tolist(), retained_looks=list(looks),
        assumed_iat=4., maximum_per_chain_required_at_iat={str(iat): float(tf.reduce_max(required))*iat/4 for iat in (1, 4, 10)},
        precision_forecast_sufficient=looks[-1] >= target, forecast_is_guarantee=False,
        design_rule="first two-look pair with final per-chain count >= max variance-based effective requirement at IAT4")


def _cpu():
    value = resource.getrusage(resource.RUSAGE_SELF)
    return value.ru_utime+value.ru_stime


def _timed(call, repetitions=6):
    rows = []
    result = None
    for index in range(repetitions):
        wall, cpu = time.monotonic(), _cpu()
        result = call(index)
        for tensor in tf.nest.flatten(result):
            if tf.is_tensor(tensor):
                tensor.numpy()
        rows.append(dict(wall_seconds=time.monotonic()-wall, cpu_worker_seconds=_cpu()-cpu, compilation_included=index == 0))
    return rows, result


def cost_stage(output):
    from bayesfilter.inference.neural_force_hmc import NeuralForceHMCConfig, neural_force_hmc_transition
    data, reference, arrays = load_reference()
    consumer = selected_consumer(data)
    directory = Path(output).parent
    directory.mkdir(parents=True, exist_ok=True)
    write_json(directory/"consumer.json", consumer.manifest)
    write_json(directory/"data.json", data)
    _, functions, probabilities = grid_law(arrays)
    truth = tf.reduce_sum(probabilities[:, None]*functions, 0)
    variance = tf.reduce_sum(probabilities[:, None]*(functions-truth)**2, 0)
    precision = precision_design(variance, tf.constant(reference["margins"], tf.float64),
                                  tf.constant(reference["empirical_allowance"], tf.float64))
    points = arrays["means"][None]+tf.constant([[0., 0.], [1., 0.], [0., 1.], [-.5, -.5]], tf.float64)*arrays["scales"][None]
    costs = {}
    for name, function in (("surrogate_force", consumer.force.function), ("exact_force", consumer.exact_force.function),
                            ("model_endpoint", consumer.endpoint.function)):
        costs[name], result = _timed(lambda _: function(points))
        if hasattr(result, "force_valid") and not bool(tf.reduce_all(result.force_valid)):
            raise ValueError("invalid cost-probe force: "+name)
    potential = consumer.endpoint.function(points).potential
    transitions = {}
    inverse_mass = tuple(float(v*v) for v in arrays["scales"])
    for arm, force in (("A", consumer.exact_force), ("B", consumer.force)):
        for leapfrog in (1, 8):
            config = NeuralForceHMCConfig(.05, leapfrog, inverse_mass)

            @tf.function(input_signature=[tf.TensorSpec([4, 2], tf.float64), tf.TensorSpec([4], tf.float64),
                                          tf.TensorSpec([2], tf.int32)], jit_compile=True, autograph=False)
            def step(q, u, seed):
                return neural_force_hmc_transition(q, u, force, consumer.endpoint, config, seed)

            timings, result = _timed(lambda i: step(points, potential, tf.constant([20260923, 900+i], tf.int32)))
            if not bool(tf.reduce_all(result.trace.finite_status)) or bool(tf.reduce_any(result.trace.divergence)):
                raise ValueError("invalid complete transition cost probe")
            transitions[f"{arm}_L{leapfrog}"] = dict(costs=timings,
                trace=tf.nest.map_structure(lambda v: v.numpy().tolist(), result.trace._asdict()))
    forecasts = {}
    for arm in ("A", "B"):
        rows = transitions[f"{arm}_L8"]["costs"]
        forecasts[arm] = {key: 1.5*(max(r[key] for r in rows[1:])*(precision["retained_looks"][-1]+7000)+rows[0][key]+600.)
                          for key in ("wall_seconds", "cpu_worker_seconds")}
    return dict(status="AR2_COMPLETE_TRANSITION_COST_MEASURED", qualified=True, costs=costs, transitions=transitions,
        precision=precision, conservative_inference_forecast=forecasts,
        forecast_assumptions="L8 upper trajectory bound; retained final look plus 7000 tuning/verification transitions; 1.5 safety factor and 600s extra compilation per arm",
        exact_force_comparator_cost_ratio=max(r["wall_seconds"] for r in transitions["B_L8"]["costs"][1:])/max(r["wall_seconds"] for r in transitions["A_L8"]["costs"][1:]),
        ranking="NOT_ESTABLISHED", acceleration_claim=False, inference_launched=False,
        reference_report_sha256=file_hash(REFERENCE_ROOT/"report.json"), selection_sha256=file_hash(CONTROLS_ROOT/"selection.json"),
        reference_variances=variance.numpy().tolist(), source_hashes=source_hashes(consumer.fixture),
        next_action="Existing master: qualify uncertainty at the measured precision schedule, publicly tune and execute AR2 B/R posterior test")
