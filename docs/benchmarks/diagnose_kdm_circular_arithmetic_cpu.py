"""CPU-only factorial arithmetic localization for the KDM circular consumer."""
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["BAYESFILTER_PRELOAD_CUSTOM_OP"] = "0"
import argparse
import hashlib
import itertools
import json
import math
import shutil
import signal
import subprocess
import traceback
import sys
import time
from pathlib import Path

ROOT = Path("/home/chakwong/BayesFilter")
sys.path.insert(0, str(ROOT))

def host(x):
    if isinstance(x, dict):
        return {k: host(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)):
        return [host(v) for v in x]
    if hasattr(x, "numpy"):
        return x.numpy().tolist()
    return x

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    previous = sorted(output.parent.glob("kdm-circular-arithmetic-20261007-*/manifest.json"))
    if len(previous) >= 3:
        raise RuntimeError("Three-attempt diagnostic budget exhausted")
    used = sum(json.loads(path.read_text())["wall_seconds"] for path in previous)
    if used >= 600:
        raise RuntimeError("CPU diagnostic time budget exhausted")
    signal.alarm(max(1, int(600 - used)))
    started = time.monotonic()
    import tensorflow as tf
    from bayesfilter.replication.kdm_normal_von_mises_tf import NormalVonMisesKDM

    fixture = ROOT / "docs/plans/artifacts/controlled-smc-kdm-replication-20260928/kdm-circular-20261006-02/author.json"
    reference = json.loads(fixture.read_text())
    implementation = ROOT / "bayesfilter/replication/kdm_normal_von_mises_tf.py"
    plan = ROOT / "docs/plans/controlled-smc-kdm-circular-arithmetic-20261007.md"
    source_hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                     for p in (Path(__file__), implementation, plan, fixture)}
    author = ROOT / ".localresources/code/younis-mdpf-neurips-2023"
    for relative, digest in reference["source_hashes"].items():
        actual_path = author / relative
        if hashlib.sha256(actual_path.read_bytes()).hexdigest() != digest:
            raise RuntimeError("Author source changed: " + relative)
        source_hashes[str(actual_path)] = digest
    vm_path = Path(reference["von_mises_implementation"])
    if hashlib.sha256(vm_path.read_bytes()).hexdigest() != reference["von_mises_sha256"]:
        raise RuntimeError("Torch von Mises source changed")
    source_hashes[str(vm_path)] = reference["von_mises_sha256"]
    normal_path = vm_path.with_name("normal.py")
    source_hashes[str(normal_path)] = hashlib.sha256(normal_path.read_bytes()).hexdigest()
    snapshots = output / "sources"
    snapshots.mkdir()
    for source_path in (Path(__file__), implementation, plan, normal_path, vm_path):
        shutil.copyfile(source_path, snapshots / source_path.name)
    old = 128 * 2**-23
    checks = []
    arms = []
    for variance_first, broadcast_band, left_sum in itertools.product((False, True), repeat=3):
        arms.append(dict(variance_first=variance_first, broadcast_band=broadcast_band,
                         left_sum=left_sum))

    class ArithmeticVariant(NormalVonMisesKDM):
        def __init__(self, *args, variance_first, broadcast_band, left_sum, **kwargs):
            self._variance_first = variance_first
            self._broadcast_band = broadcast_band
            self._left_sum = left_sum
            super().__init__(*args, **kwargs)

        def _components(self, locations, bandwidth, points, *, resampling):
            residual = points[:, :, None, :] - locations[:, None, :, :]
            band = bandwidth[:, None, None, :]
            normal_band = band
            if self._broadcast_band:
                particle_band = tf.broadcast_to(
                    bandwidth[:, None, :], tf.shape(locations))
                normal_band = particle_band[:, None, :, :]
            if self._variance_first:
                variance = tf.square(normal_band)
                normal = -tf.square(residual) / (2 * variance) - tf.math.log(normal_band)
            else:
                normal = -.5 * tf.square(residual / normal_band) - tf.math.log(normal_band)
            normal -= self._constant(math.log(math.sqrt(2 * math.pi))
                                     if self._variance_first else .5 * math.log(2 * math.pi))
            concentration = 1. / (band + self.epsilon if resampling else band)
            concentration = tf.broadcast_to(concentration[:, 0, :, :],
                                            tf.shape(locations))
            log_i0, special_valid = self._log_i0(concentration)
            circular = concentration[:, None, :, :] * tf.cos(residual)
            circular -= self._constant(math.log(2 * math.pi))
            circular -= log_i0[:, None, :, :]
            selected = tf.where(self.circular, circular, normal)
            if self._left_sum:
                component = selected[:, :, :, 0] + selected[:, :, :, 1]
                component = component + selected[:, :, :, 2]
            else:
                component = tf.reduce_sum(selected, -1)
            valid = special_valid & tf.reduce_all(bandwidth > 0)
            valid &= tf.reduce_all(tf.math.is_finite(locations))
            valid &= tf.reduce_all(tf.math.is_finite(points))
            valid &= tf.reduce_all(tf.math.is_finite(component))
            return component, valid

    def compare(name, actual, expected, dtype_name):
        dtype = getattr(tf, dtype_name)
        if isinstance(actual, (tuple, list)):
            if not isinstance(expected, (tuple, list)) or len(actual) != len(expected):
                checks.append(dict(name=name + "/group", pass_check=False))
                return
            for i, (x, y) in enumerate(zip(actual, expected)):
                compare(name + "/" + str(i), x, y, dtype_name)
            return
        if isinstance(actual, dict):
            for key in actual:
                if key in expected:
                    compare(name + "/" + key, actual[key], expected[key], dtype_name)
            return
        if isinstance(expected, bool):
            checks.append(dict(name=name, pass_check=bool(actual) == expected))
            return
        a = tf.convert_to_tensor(actual, dtype=dtype)
        e = tf.convert_to_tensor(expected, dtype=dtype)
        tol = old if dtype_name == "float32" else 1e-9
        bound = tf.cast(tol, dtype) * (1 + tf.abs(e))
        delta = tf.abs(a - e)
        finite = tf.reduce_all(tf.math.is_finite(a))
        passed = bool((finite & tf.reduce_all(delta <= bound)).numpy())
        checks.append(dict(name=name, pass_check=passed,
                           max_absolute_error=float(tf.reduce_max(delta).numpy()),
                           max_scaled_error=float(tf.reduce_max(delta / bound).numpy()),
                           tolerance=tol))


    saved_outputs = {}
    analytic_checks, conditioning = [], []
    double_kernel = ArithmeticVariant(
        batch_size=2, particles=25, circular_dimensions=[False, False, True],
        dtype=tf.float64, jit_compile=False, variance_first=True,
        broadcast_band=True, left_sum=True)

    def poly_dual(argument, coefficients):
        coefficients = tf.constant(coefficients, tf.float64)
        i = tf.size(coefficients) - 2
        value = tf.fill(tf.shape(argument), coefficients[-1])
        slope = tf.zeros_like(value)
        def body(i, value, slope):
            return i - 1, coefficients[i] + argument * value, value + argument * slope
        _, value, slope = tf.while_loop(
            lambda i, value, slope: i >= 0, body, (i, value, slope),
            maximum_iterations=8)
        return value, slope

    def log_i0_derivative(kappa):
        small_x = tf.square(kappa / 3.75)
        ps, ds = poly_dual(small_x, [1., 3.5156229, 3.0899424, 1.2067492,
                                    .2659732, .0360768, .0045813])
        large_x = 3.75 / kappa
        pl, dl = poly_dual(large_x, [.39894228, .01328592, .00225319, -.00157565,
                                    .00916281, -.02057706, .02635537, -.01647633,
                                    .00392377])
        return tf.where(kappa < 3.75, ds / ps * (2 * kappa / (3.75**2)),
                        1 - .5 / kappa - dl / pl * (3.75 / tf.square(kappa)))

    cloud64 = tf.TensorSpec([2, 25, 3], tf.float64)
    weight64 = tf.TensorSpec([2, 25], tf.float64)
    band64 = tf.TensorSpec([2, 3], tf.float64)
    state64 = tf.TensorSpec([2, 3], tf.float64)

    @tf.function(input_signature=[cloud64, weight64, band64, cloud64,
                                  cloud64, weight64, band64, state64],
                 jit_compile=False, autograph=False)
    def analytic(locations, weights, bandwidth, samples, posterior_locations,
                 model_weights, posterior_bandwidth, states):
        eps = tf.constant(1e-8, tf.float64)
        component, _ = double_kernel._components(
            locations, bandwidth, samples, resampling=True)
        rho = (weights + eps) / tf.reduce_sum(weights + eps, -1, keepdims=True)
        pi = tf.nn.softmax(component + tf.math.log(rho[:, None, :]), -1)
        posterior_component, _ = double_kernel._components(
            posterior_locations, posterior_bandwidth, states[:, None, :],
            resampling=False)
        nu = model_weights / tf.reduce_sum(model_weights, -1, keepdims=True)
        likelihood = tf.exp(posterior_component[:, 0, :])
        density = tf.reduce_sum(likelihood * nu, -1, keepdims=True)
        responsibilities = tf.math.divide_no_nan(likelihood * nu, density)
        raw = tf.reduce_sum(tf.square(samples) + .3 * samples, -1)
        normalized = (raw - tf.reduce_mean(raw, -1, keepdims=True)) / (25 * (1 + eps))
        connected = (nu - responsibilities) * density / (density + eps) / (2 * (1 + eps))
        residual = samples[:, :, None, :] - locations[:, None, :, :]
        band = bandwidth[:, None, None, :]
        kappa = 1 / (band + eps)
        normal_positive = tf.square(residual) / (band**3)
        normal_negative = -tf.ones_like(residual) / band
        circular_positive = tf.square(kappa) * log_i0_derivative(kappa)
        circular_negative = -tf.square(kappa) * tf.cos(residual)
        positive = tf.where(double_kernel.circular, circular_positive, normal_positive)
        negative = tf.where(double_kernel.circular, circular_negative, normal_negative)
        location_score = tf.where(double_kernel.circular,
                                  kappa * tf.sin(residual), residual / tf.square(band))
        results = {}
        for objective, adjoint in (("raw", raw), ("normalized", normalized),
                                   ("consumer", connected)):
            mass = adjoint[:, :, None, None] * pi[:, :, :, None]
            wp = adjoint[:, :, None] * pi / (weights[:, None, :] + eps)
            wn = -adjoint[:, :, None] * rho[:, None, :] / (weights[:, None, :] + eps)
            wgrad = tf.reduce_sum(wp + wn, axis=1)
            wscale = tf.reduce_sum(tf.abs(wp) + tf.abs(wn), axis=1)
            bp, bn = mass * positive, mass * negative
            bgrad = tf.reduce_sum(bp + bn, axis=[1, 2])
            bscale = tf.reduce_sum(tf.abs(bp) + tf.abs(bn), axis=[1, 2])
            lgrad = tf.reduce_sum(mass * location_score, axis=1)
            results[objective] = dict(gradient=[lgrad, wgrad, bgrad],
                                      weight_term_scale=wscale, band_term_scale=bscale)
        return results

    # Build one traced kernel per arm/dtype; do not retrace for every fixture.
    kernels = {}
    for arm in arms:
        arm_name = "v%s_b%s_l%s" % tuple(int(arm[k]) for k in ("variance_first", "broadcast_band", "left_sum"))
        for dtype_name in ("float32", "float64"):
            kernels[(arm_name, dtype_name)] = ArithmeticVariant(
                batch_size=2, particles=25, circular_dimensions=[False, False, True],
                dtype=getattr(tf, dtype_name), jit_compile=False, **arm)
    for arm in arms:
        arm_name = "v%s_b%s_l%s" % tuple(int(arm[k]) for k in ("variance_first", "broadcast_band", "left_sum"))
        for case in reference["cases"]:
            dtype_name = case["dtype"]
            dtype = getattr(tf, dtype_name)
            kernel = kernels[(arm_name, dtype_name)]
            values = {k: tf.constant(v, dtype) for k, v in case["inputs"].items()}
            locations, weights, bandwidth, samples = [values[k] for k in
                ("locations", "weights", "bandwidth", "samples")]
            results = {
                "resampling": kernel.resampling_score(locations, weights, bandwidth, samples),
                "posterior": kernel.posterior_score(locations, weights, bandwidth, values["states"]),
                "consumer": kernel.consumer_score(
                    locations, weights, bandwidth, samples,
                    values["posterior_locations"], values["model_weights"],
                    values["posterior_bandwidth"], values["states"]),
            }
            saved_outputs[(arm_name, case["id"])] = host(results)
            for category, result in results.items():
                expected = case[category]
                for key in expected:
                    compare(arm_name + "/" + case["id"] + "/" + category + "/" + key,
                            result[key], expected[key], dtype_name)
                checks.append(dict(name=arm_name + "/" + case["id"] + "/" + category + "/valid",
                                   pass_check=bool(result["valid"].numpy())))

    trace_counts = {
        arm_name + "/" + dtype_name: {
            name: getattr(kernel, name).experimental_get_tracing_count()
            for name in ("resampling_score", "posterior_score", "consumer_score")}
        for (arm_name, dtype_name), kernel in kernels.items()}
    for case in reference["cases"]:
        if case["dtype"] != "float32":
            continue
        # Cast the exact float32 fixtures to double, not the separately sampled f64 cases.
        v = {k: tf.cast(tf.constant(x, tf.float32), tf.float64)
             for k, x in case["inputs"].items()}
        args64 = [v[k] for k in ("locations", "weights", "bandwidth", "samples",
                  "posterior_locations", "model_weights", "posterior_bandwidth", "states")]
        analysis = analytic(*args64)
        reference_resampling = double_kernel.resampling_score(*args64[:4])
        reference_consumer = double_kernel.consumer_score(*args64)
        objective_gradients = {
            "raw": reference_resampling["raw_gradient"],
            "normalized": reference_resampling["normalized_gradient"],
            "consumer": reference_consumer["gradient"][:3]}
        for objective, autograd in objective_gradients.items():
            for group, actual in enumerate(analysis[objective]["gradient"]):
                expected = autograd[group]
                error = tf.abs(actual - expected)
                bound = tf.constant(1e-9, tf.float64) * (1 + tf.abs(expected))
                analytic_checks.append(dict(case=case["id"], objective=objective,
                    group=group, pass_check=bool(tf.reduce_all(error <= bound).numpy()),
                    max_absolute_error=float(tf.reduce_max(error).numpy()),
                    max_scaled_error=float(tf.reduce_max(error / bound).numpy())))
            category = "consumer" if objective == "consumer" else "resampling"
            field = "gradient" if objective == "consumer" else objective + "_gradient"
            for group, scale_key in ((1, "weight_term_scale"), (2, "band_term_scale")):
                source = tf.constant(case[category][field][group], tf.float64)
                double = analysis[objective]["gradient"][group]
                scale = analysis[objective][scale_key]
                for arm_name in sorted(set(key[0] for key in saved_outputs)):
                    candidate = tf.constant(saved_outputs[(arm_name, case["id"])][category][field][group], tf.float64)
                    discrepancy = tf.abs(candidate - source)
                    old_bound = old * (1 + tf.abs(source))
                    coordinate = int(tf.argmax(tf.reshape(discrepancy / old_bound, [-1])).numpy())
                    flatten = lambda x: float(tf.reshape(x, [-1])[coordinate].numpy())
                    term_scale, derivative = flatten(scale), flatten(double)
                    conditioning.append(dict(
                        case=case["id"], arm=arm_name, objective=objective, group=group,
                        flat_coordinate=coordinate, source=flatten(source),
                        candidate=flatten(candidate), analytic_float64=derivative,
                        sum_absolute_terms=term_scale,
                        cancellation_ratio=term_scale / abs(derivative) if derivative else None,
                        error_scaled_by_original_bound=flatten(discrepancy / old_bound),
                        error_over_epsilon_term_scale=flatten(discrepancy) / ((2**-23) * term_scale)
                            if term_scale else None,
                        source_error_from_float64=abs(flatten(source) - derivative),
                        candidate_error_from_float64=abs(flatten(candidate) - derivative)))
    if any(not c["pass_check"] for c in analytic_checks):
        validity = "ANALYTIC_DERIVATIVE_CHECK_FAILED"
    elif any(not c["pass_check"] for c in checks if "/float64_" in c["name"]):
        validity = "FLOAT64_REFERENCE_FAILED"
    elif any(not c["pass_check"] for c in checks if c["name"].endswith("/valid")):
        validity = "INVALID_DENSITY_OR_GRADIENT"
    elif any(count != 1 for scopes in trace_counts.values() for count in scopes.values()):
        validity = "UNBOUNDED_TRACING"
    else:
        validity = "PASS"
    if hashlib.sha256(implementation.read_bytes()).hexdigest() != source_hashes[str(implementation)]:
        validity = "RUNTIME_SOURCE_CHANGED"
    by_arm = {}
    for check in checks:
        arm_name = check["name"].split("/", 1)[0]
        by_arm.setdefault(arm_name, []).append(check)
    summary = dict(
        schema="kdm_circular_arithmetic_localization_v1",
        status="PASS" if validity == "PASS" else "FAIL",
        diagnostic_validity=validity,
        analytic_checks=analytic_checks,
        trace_counts=trace_counts,
        scope="CPU_NON_XLA_ARITHMETIC_LOCALIZATION",
        replication="INCOMPLETE",
        fixture=str(fixture),
        fixture_sha256=source_hashes[str(fixture)],
        source_hashes=source_hashes,
        dtype_tolerances={"float32": old, "float64": 1e-9},
        arms=arms,
        checks=checks,
        arm_summary={name: dict(pass_check=all(c["pass_check"] for c in cs),
                                failed=sum(not c["pass_check"] for c in cs),
                                max_scaled_error=max((c.get("max_scaled_error", 0) for c in cs), default=0))
                     for name, cs in by_arm.items()},
        tensorflow=tf.__version__, cpu_only=True, gpu_intentionally_hidden=True,
        jit_compile=False, gpu_default_readiness_evidence=False,
        wall_seconds=time.monotonic() - started)
    summary["numerical_screen_passed"] = all(c["pass_check"] for c in checks)
    summary["scientific_verdict"] = (
        "ALL_ARMS_MATCH" if summary["numerical_screen_passed"]
        else "ARITHMETIC_ARM_DIFFERENCES_REMAIN")
    (output / "conditioning.json").write_text(json.dumps(conditioning, indent=2, allow_nan=False) + "\n")
    (output / "outputs.json").write_text(json.dumps(
        {arm + "/" + case: values for (arm, case), values in saved_outputs.items()},
        indent=2, allow_nan=False) + "\n")
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    (output / "manifest.json").write_text(json.dumps(dict(
        schema="kdm_circular_arithmetic_localization_manifest_v1",
        command=[sys.executable] + sys.argv, plan=str(plan), result=str(output / "summary.json"),
        environment={k: os.environ.get(k) for k in
                     ("CUDA_VISIBLE_DEVICES", "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS",
                      "BAYESFILTER_PRELOAD_CUSTOM_OP")},
        status=summary["status"], wall_seconds=summary["wall_seconds"],
        source_hashes=source_hashes, no_gpu=True,
        git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        branch=subprocess.check_output(["git", "branch", "--show-current"], cwd=ROOT, text=True).strip(),
        seeds="N/A: deterministic replay of preserved fixture",
        data_version=source_hashes[str(fixture)],
        attempt_timeout_seconds=int(600-used), diagnostic_validity=validity) , indent=2) + "\n")
    print(json.dumps({"status": summary["status"], "verdict": summary["scientific_verdict"],
                      "arms": summary["arm_summary"], "validity": validity,
                      "analytic_checks": len(analytic_checks),
                      "analytic_failures": sum(not c["pass_check"] for c in analytic_checks)}))
    signal.alarm(0)
    if validity != "PASS":
        raise SystemExit(1)

if __name__ == "__main__":
    main()
