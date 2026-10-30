"""Scoped numerical protection and precision diagnostics for canonical AR2."""
import json
import math
from pathlib import Path
import time

import tensorflow as tf

from bayesfilter.highdim.ledh_structural_ar2_target_tf import make_ar2_target
from bayesfilter.inference.surrogate_hmc_dctr_selection import MULTIPLIERS, issue_selection, protection_controls
from bayesfilter.inference.surrogate_hmc_program import ROOT, file_hash, write_json
from bayesfilter.testing.surrogate_hmc_ar2_reference import PRIOR_SD

REFERENCE_ROOT = ROOT/"docs/plans/artifacts/surrogate-hmc-closure-20260922-structural-03/ar2-reference"
PARTICLE_COUNT = 64
MOMENTS = ("filtering_mean", "filtering_covariance", "post_reset_mean", "post_reset_covariance")


def ar2_scope():
    return dict(model="AR2", horizon=10, attempt=1, state_dimension=2, observation_dimension=1,
        innovation_dimension=1, parameter_dimension=2, particle_count=PARTICLE_COUNT, chunk_extent=PARTICLE_COUNT,
        dtype="float32", tf32=True, jit_compile=True, route="canonical_batch_fused_value_score",
        reset_contract_id="contract_e_chol_v1", initial_law="N(0,I2)",
        law="F=[[.45,-.15],[1,0]]; Q=.09*exp(2*lambda_Q)*BB'; R=.04*exp(2*lambda_R); B=[1,0]'",
        prior="independent N(0,log(2)^2) on both log scales", draw_design="scrambled_halton_gaussian",
        data_regime="AR2 T10 independent calibration/validation/claim; once-rounded FP32 observations",
        control_family="full_canonical_geometric_protection_curve_v1")


def source_hashes(target):
    sources = dict(target.manifest["source_sha256"])
    for name in ("bayesfilter/testing/surrogate_hmc_ar2_consumer.py", "bayesfilter/testing/surrogate_hmc_ar2_reference.py",
                 "bayesfilter/testing/surrogate_hmc_ar2_controls.py", "bayesfilter/testing/surrogate_hmc_lg1_tf.py",
                 "bayesfilter/inference/surrogate_hmc_dctr_selection.py", "bayesfilter/inference/neural_force_hmc.py"):
        sources[name] = file_hash(ROOT/name)
    return sources


def finite_json(value):
    if isinstance(value, dict):
        return {k: finite_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [finite_json(v) for v in value]
    return None if isinstance(value, float) and not math.isfinite(value) else value


def components(output):
    value, score, status = output
    return dict(value=value, score=score, **{name: status[name] for name in MOMENTS})


def describe(output):
    value, score, status = output
    tensors = components(output)
    finite = tf.ones_like(status["value_valid"])
    for tensor in tensors.values():
        finite &= tf.reduce_all(tf.math.is_finite(tensor), list(range(1, tensor.shape.rank)))
    valid = status["value_valid"] & status["force_valid"]
    healthy = valid & finite
    flagged = ~valid
    return dict(all_healthy=bool(tf.reduce_all(healthy)), healthy_rows=healthy.numpy().tolist(),
        all_failures_flagged=bool(tf.reduce_all(healthy | flagged)),
        values=value.numpy().tolist(), scores=score.numpy().tolist(),
        status={k: v.numpy().tolist() for k, v in status.items()},
        moments={name: status[name].numpy().tolist() for name in MOMENTS})


def normalized_changes(left, right, mask=None):
    a, b = components(left), components(right)
    changes = {}
    for key in a:
        x, y = tf.cast(a[key], tf.float64), tf.cast(b[key], tf.float64)
        if mask is not None:
            x, y = tf.boolean_mask(x, mask), tf.boolean_mask(y, mask)
        changes[key] = float(tf.reduce_max(tf.abs(x-y)/(1+tf.abs(y)))) if int(tf.size(x)) else 0.
    return changes


def controls_stage(output):
    out = Path(output).parent
    out.mkdir(parents=True, exist_ok=True)
    saved = json.loads((REFERENCE_ROOT/"report.json").read_text())
    if saved.get("qualified") is not True:
        raise ValueError("qualified AR2 reference prerequisite missing")
    reference_manifest = json.loads((REFERENCE_ROOT.parent/"run-manifest.json").read_text())
    for name in ("bayesfilter/testing/surrogate_hmc_ar2_reference.py", "bayesfilter/testing/surrogate_hmc_linear_posterior_reference.py"):
        if file_hash(ROOT/name) != reference_manifest["source_hashes"].get(name):
            raise ValueError("AR2 reference source drift: "+name)
    datasets = json.loads((REFERENCE_ROOT/"datasets.json").read_text())
    scope, curves, paths, all_outputs, sources, points_by_partition = ar2_scope(), {}, {}, {}, None, {}
    stress = tf.constant([[-3., -3.], [-3., 3.], [3., -3.], [3., 3.]], tf.float32)*tf.cast(PRIOR_SD, tf.float32)
    for partition in ("calibration", "validation"):
        directory = out/partition
        directory.mkdir()
        data = datasets[partition] | dict(attempt=1, replicate=0)
        paths[partition] = directory/"data.json"
        write_json(paths[partition], data)
        reference = saved["partitions"][partition]
        mean, sd = (tf.constant(reference[key], tf.float32) for key in ("posterior_mean", "posterior_sd"))
        nominal = mean[None]+tf.constant([[0., 0.], [1., 0.], [0., 1.], [-.5, -.5]], tf.float32)*sd[None]
        points_by_partition[partition] = (nominal, stress)
        outputs, diagnostics = [], []
        for index, multiplier in enumerate(MULTIPLIERS):
            target = make_ar2_target(data, particle_count=PARTICLE_COUNT, controls=protection_controls(multiplier))
            current_sources = source_hashes(target)
            if sources is not None and sources != current_sources:
                raise ValueError("source changed during control curve")
            sources = current_sources
            evaluate = target.compile(4, filter_moments=True)
            begin = time.monotonic()
            results = [evaluate(nominal), evaluate(stress)]
            results[-1][0].numpy()
            row = dict(multiplier=multiplier, controls=protection_controls(multiplier),
                target=dict(target.manifest), wall_seconds=time.monotonic()-begin,
                situations={name: describe(result) for name, result in zip(("nominal", "stress"), results)})
            write_json(directory/f"arm-{index}.json", finite_json(row))
            print(partition, multiplier, {k: v["all_healthy"] for k, v in row["situations"].items()}, row["wall_seconds"], flush=True)
            outputs.append(results)
            diagnostics.append(row)
        all_outputs[partition] = outputs
        pairs = []
        for index in range(3):
            changes = {name: normalized_changes(left, right) for name, left, right in
                       zip(("nominal", "stress"), outputs[index], outputs[index+1])}
            maximum = max(changes["nominal"].values())
            healthy = all(diagnostics[i]["situations"]["nominal"]["all_healthy"] for i in (index, index+1))
            safe = all(diagnostics[i]["situations"]["stress"]["all_failures_flagged"] for i in (index, index+1))
            strong, weak = (diagnostics[i]["situations"]["stress"]["healthy_rows"] for i in (index, index+1))
            no_new_failure = all(not b or a for a, b in zip(strong, weak))
            pairs.append(dict(healthy_nominal=healthy, stress_safe=safe, no_new_stress_failure=no_new_failure,
                normalized_maximum_change=maximum, tolerance=1e-3, component_changes=changes,
                non_harm_passed=healthy and safe and no_new_failure and math.isfinite(maximum) and maximum <= 1e-3))
        curves[partition] = directory/"curve.json"
        write_json(curves[partition], finite_json(dict(scope=scope, source_hashes=sources,
            data_sha256=file_hash(paths[partition]), pairs=pairs,
            arms=[dict(path=str(directory/f"arm-{i}.json"), sha256=file_hash(directory/f"arm-{i}.json")) for i in range(4)])))
    try:
        selection = issue_selection(output=out/"selection.json", scope=scope, evidence_paths=curves,
                    partition_data_paths=paths, source_hashes=sources)
    except ValueError as exc:
        if str(exc) != "NO_STABLE_CONTROL_PAIR":
            raise
        return dict(status="AR2_CONTROL_REPAIR_REQUIRED", qualified=False, scope=scope,
                    reason=str(exc), posterior_comparisons="NOT_RUN", all_models_validated=False)
    precision = {}
    for partition in ("calibration", "validation"):
        data = datasets[partition] | dict(attempt=1, replicate=0)
        target = make_ar2_target(data, particle_count=PARTICLE_COUNT, controls=selection["controls"], dtype=tf.float64)
        evaluate = target.compile(4, filter_moments=True)
        comparisons = {}
        for name, points, fp32 in zip(("nominal", "stress"), points_by_partition[partition],
                                     all_outputs[partition][selection["selected_index"]]):
            fp64 = evaluate(tf.cast(points, tf.float64))
            validity_equal = bool(tf.reduce_all((fp32[2]["value_valid"] == fp64[2]["value_valid"])
                                      & (fp32[2]["force_valid"] == fp64[2]["force_valid"])))
            healthy = fp32[2]["program_valid"] & fp64[2]["program_valid"]
            changes = normalized_changes(fp32, fp64, healthy)
            comparisons[name] = dict(validity_equal=validity_equal, component_changes=changes,
                passed=validity_equal and all(math.isfinite(v) and v <= 2e-3 for v in changes.values()),
                fp64=describe(fp64))
        precision[partition] = comparisons
    precision_passed = all(row["passed"] for part in precision.values() for row in part.values())
    stress_healthy = all(bool(tf.reduce_all(all_outputs[p][selection["selected_index"]][1][2]["program_valid"]))
                         for p in all_outputs)
    passed = precision_passed and stress_healthy
    status = ("AR2_FORCE_CONTROLS_QUALIFIED" if passed else "AR2_STRESS_REPAIR_REQUIRED"
              if precision_passed else "AR2_PRECISION_REPAIR_REQUIRED")
    report = dict(status=status, precision_passed=precision_passed, stress_healthy=stress_healthy,
        qualified=passed, scope=scope, selection_path=str(out/"selection.json"), selected_multiplier=selection["selected_multiplier"],
        precision=precision, source_hashes=sources, reference_root=str(REFERENCE_ROOT),
        reference_report_sha256=file_hash(REFERENCE_ROOT/"report.json"), data_sha256=file_hash(REFERENCE_ROOT/"datasets.json"),
        all_models_validated=False, posterior_comparisons="NOT_RUN", jit_compile=True,
        next_action="Existing master: forecast precision/cost, public HMC tuning, retained B/R; separately qualify finite endpoint/reference")
    return finite_json(report)
