"""Frozen-target diagnostic only: separate TT fitting, rank and inherited error.

Tensor-product integration and algebraic TT-SVD here are independent references,
not runtime filtering routes or candidates. No training/particle campaign runs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
SOURCE = ROOT / "docs/benchmarks/artifacts/observation_aware_tt_complete_20260913"
PLAN = ROOT / "docs/plans/observation-tt-first-transition-root-cause-20260914.md"


def plain(x):
    if hasattr(x, "numpy"):
        return plain(x.numpy().tolist())
    if isinstance(x, dict):
        return {str(k): plain(v) for k, v in x.items()}
    if isinstance(x, (tuple, list)):
        return [plain(v) for v in x]
    return x


def write(path, value):
    path.write_text(json.dumps(plain(value), indent=2, allow_nan=False) + "\n")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--wall-budget-seconds", type=int, default=600)
    args = parser.parse_args()
    out = Path(args.output_root).resolve()
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    manifest = dict(started_utc=datetime.now(timezone.utc).isoformat(),
                    command=[sys.executable, *sys.argv], plan=str(PLAN),
                    result=str(out / "result.json"), environment=sys.prefix,
                    git_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
                    gpu_intentionally_hidden=False, jit_compile=True,
                    diagnostic_only=True, status="STARTING")
    paths = [Path(__file__).resolve(), PLAN, SOURCE / "c2_fixture.json",
             SOURCE / "campaign-01/d4/tt_guided_proposals.json",
             SOURCE / "campaign-01/d4/tt_guided_fit_t01.json",
             ROOT / "bayesfilter/highdim/observation_guided_tt_tf.py",
             ROOT / "bayesfilter/highdim/c2_gaussian_hermite_proposal_tf.py"]
    manifest["source_hashes"] = {str(p.relative_to(ROOT)): sha(p) for p in paths}
    write(out / "run_manifest.json", manifest)
    try:
        import tensorflow as tf
        devices = tf.config.list_physical_devices("GPU")
        if not devices:
            raise RuntimeError("GPU required for the bounded diagnostic")
        growth = []
        for device in devices:
            tf.config.experimental.set_memory_growth(device, True)
            if not tf.config.experimental.get_memory_growth(device):
                raise RuntimeError("memory growth not established")
            growth.append(dict(name=device.name, memory_growth=True,
                               details=tf.config.experimental.get_device_details(device)))
        from bayesfilter.highdim import observation_guided_tt_tf as lib
        from bayesfilter.nonlinear.fixed_sgqf_tf import tf_standard_normal_ghq_level_rule
        D = tf.float64
        manifest.update(tensorflow=tf.__version__, numerical_dtype="float64",
                        gpu_memory_policy=dict(mode="memory_growth", devices=growth),
                        tf32_enabled=tf.config.experimental.tensor_float_32_execution_enabled(),
                        jit_exception="diagnostic SVD, quadrature construction and host reductions outside XLA",
                        random_seeds="none; frozen snapshots and deterministic quadrature", status="RUNNING")
        write(out / "run_manifest.json", manifest)
        fixture = json.loads((SOURCE / "c2_fixture.json").read_text())
        data = fixture["dimensions"]["4"]
        model = lib.SVModel(tf.constant(data["A"], D), tf.constant(data["P0"], D), fixture["beta"], fixture["sigma"])
        y0, y1 = (tf.constant(x, D) for x in data["observations"][:2])
        snapshots = json.loads((SOURCE / "campaign-01/d4/tt_guided_proposals.json").read_text())
        steps = []
        for snap in snapshots[:2]:
            current = lib.Chart(tf.constant(snap["mean"], D), tf.constant(snap["factor"], D))
            condition = None if snap["time"] == 0 else lib.Chart(tf.constant(snap["condition_mean"], D), tf.constant(snap["condition_factor"], D))
            steps.append(lib.TTStep(tuple(tf.constant(c, D) for c in snap["cores"]), current,
                                    condition, tf.constant(snap["tau"], D), {}, snap["time"]))
        zero, one = steps
        prior = zero.retained()
        tf.debugging.assert_equal(one.conditioning_chart.mean, zero.current_chart.mean)
        tf.debugging.assert_equal(one.conditioning_chart.factor, zero.current_chart.factor)
        scale = tf.constant(json.loads((SOURCE / "campaign-01/d4/tt_guided_fit_t01.json").read_text())["log_target_scale"], D)
        mass0 = lib._paired_right_environments(zero.cores, tf.ones([1, 1], D))[0][0, 0]
        mass1 = lib._paired_right_environments(one.cores, tf.ones([1, 1], D))[0][0, 0]

        def basis(x):
            values = lib.features(x, 3)
            result = tf.ones([tf.shape(x)[0], 1], D)
            for k in range(4):
                result = tf.reshape(result[:, :, None] * values[:, None, k, :], [tf.shape(x)[0], -1])
            return result

        def coefficient_tensor(cores):
            value = cores[0][0]
            for c in cores[1:]:
                value = tf.tensordot(value, c, axes=[[-1], [0]])
            return value[..., 0]

        fitted_coefficients = tf.reshape(coefficient_tensor(one.cores), [256, 256])

        def tt_svd(coefficients, order, rank=3):
            value = tf.transpose(tf.reshape(coefficients, [4] * 8), order)
            cores, previous_rank = [], 1
            for _ in range(7):
                matrix = tf.reshape(value, [previous_rank * 4, -1])
                singular, left, right = tf.linalg.svd(matrix, full_matrices=False)
                k = min(rank, int(singular.shape[0]))
                cores.append(tf.reshape(left[:, :k], [previous_rank, 4, k]))
                value = singular[:k, None] * tf.transpose(right[:, :k])
                previous_rank = k
            cores.append(tf.reshape(value, [previous_rank, 4, 1]))
            inverse_order = [order.index(k) for k in range(8)]
            return tf.reshape(tf.transpose(coefficient_tensor(cores), inverse_order), [256, 256])

        signature = [tf.TensorSpec([None, 4], D), tf.TensorSpec([None], D),
                     tf.TensorSpec([None, 4], D), tf.TensorSpec([None], D),
                     tf.TensorSpec([None], D), tf.TensorSpec([None], D)]

        @tf.function(input_signature=signature, jit_compile=True, autograph=False)
        def block(u, wu, v, wv, ratio_a, ratio_b):
            x, z = one.current_chart.forward(u), one.conditioning_chart.forward(v)
            az = tf.linalg.matmul(z, model.transition, transpose_b=True)
            square = tf.reduce_sum(x*x, axis=1)[:, None] + tf.reduce_sum(az*az, axis=1)[None, :] - 2*tf.linalg.matmul(x, az, transpose_b=True)
            log_transition = -.5*square/model.sigma**2 - 4*(.5*lib.LOG2PI + tf.math.log(tf.constant(model.sigma, D)))
            common = model.observation_log_prob(x, y1) + one.current_chart.logdet - lib._log_standard_normal(u)
            base = .5*(common[:, None] + log_transition - scale + tf.math.log(wu)[:, None] + tf.math.log(wv)[None, :])
            a = tf.exp(base + .5*ratio_a[None, :])
            b = tf.exp(base + .5*ratio_b[None, :])
            bu, bv = basis(u)*tf.sqrt(wu)[:, None], basis(v)*tf.sqrt(wv)[:, None]
            ca = tf.linalg.matmul(bu, tf.linalg.matmul(a, bv), transpose_a=True)
            cb = tf.linalg.matmul(bu, tf.linalg.matmul(b, bv), transpose_a=True)
            fitted = tf.linalg.matmul(tf.linalg.matmul(bu, fitted_coefficients), bv, transpose_b=True)
            scalars = tf.stack([tf.reduce_sum(a*a), tf.reduce_sum(b*b), tf.reduce_sum(a*b),
                                tf.reduce_sum((a-fitted)**2), tf.reduce_sum((b-fitted)**2), tf.reduce_sum(fitted*fitted)])
            moments = []
            for amplitude in (a, b):
                density = amplitude*amplitude
                row, col = tf.reduce_sum(density, axis=1), tf.reduce_sum(density, axis=0)
                mean = tf.concat([tf.linalg.matvec(x, row, transpose_a=True), tf.linalg.matvec(z, col, transpose_a=True)], 0)
                xx = tf.linalg.matmul(x*row[:, None], x, transpose_a=True)
                zz = tf.linalg.matmul(z*col[:, None], z, transpose_a=True)
                xz = tf.linalg.matmul(x, tf.linalg.matmul(density, z), transpose_a=True)
                second = tf.concat([tf.concat([xx, xz], 1), tf.concat([tf.transpose(xz), zz], 1)], 0)
                moments.append(tf.concat([mean, tf.reshape(second, [-1])], 0))
            return ca, cb, scalars, tf.stack(moments)

        results = []
        def budget():
            if time.monotonic() - started > args.wall_budget_seconds:
                raise TimeoutError("frozen diagnostic budget exhausted")

        for order in (5, 7, 9, 11):
            budget()
            if order == 11:
                prev, last = results[-2:]
                monitored = ("saved_relative_rms", "degree3_projection_relative_rms",
                             "grouped_tt_svd_rank3_relative_rms", "paired_tt_svd_rank3_relative_rms", "log_integral")
                changes = [abs(last[target][key]-prev[target][key])
                           for target in ("target_a", "target_b") for key in monitored]
                changes += [abs(last[key]-prev[key]) for key in (
                    "inherited_initial_amplitude_distance", "inherited_joint_amplitude_distance", "prior_exact_log_integral")]
                change = max(changes)
                if change <= .001 or args.wall_budget_seconds - (time.monotonic()-started) < 180:
                    break
            stage_start = time.monotonic()
            rule = tf_standard_normal_ghq_level_rule((order+1)//2)
            mesh = tf.meshgrid(*([rule.nodes]*4), indexing="ij")
            nodes = tf.stack([tf.reshape(m, [-1]) for m in mesh], axis=1)
            wmesh = tf.meshgrid(*([rule.weights]*4), indexing="ij")
            weights = tf.reduce_prod(tf.stack(wmesh), axis=0)
            weights = tf.reshape(weights, [-1])
            tf.debugging.assert_near(tf.reduce_sum(weights), tf.constant(1., D), atol=1e-12)
            z = one.conditioning_chart.forward(nodes)
            initial_amplitude = lib._prefix_row_vectors(zero.cores, nodes)[:, 0]
            ratio_a = tf.math.log(initial_amplitude**2+zero.tau)-tf.math.log(mass0+zero.tau)
            unnormalized_b = model.prior_log_prob(z)+model.observation_log_prob(z, y0)+one.conditioning_chart.logdet-lib._log_standard_normal(nodes)
            log_z0 = tf.reduce_logsumexp(unnormalized_b+tf.math.log(weights))
            ratio_b = unnormalized_b-log_z0
            physical_check = prior.physical_log_density(z)+one.conditioning_chart.logdet-lib._log_standard_normal(nodes)
            tf.debugging.assert_near(ratio_a, physical_check, atol=2e-10)
            # Independent same-row transition evaluation checks the matrix identity.
            xcheck = one.current_chart.forward(nodes[:8])
            zcheck = z[-8:]
            azcheck = tf.linalg.matmul(zcheck, model.transition, transpose_b=True)
            distance = tf.reduce_sum(xcheck*xcheck, 1)+tf.reduce_sum(azcheck*azcheck, 1)-2*tf.reduce_sum(xcheck*azcheck, 1)
            expanded = -.5*distance/model.sigma**2-4*(.5*lib.LOG2PI+tf.math.log(tf.constant(model.sigma, D)))
            tf.debugging.assert_near(expanded, model.transition_log_prob(xcheck, zcheck), atol=2e-10)
            bs = basis(nodes)*tf.sqrt(weights)[:, None]
            gram_error = tf.reduce_max(tf.abs(tf.linalg.matmul(bs, bs, transpose_a=True)-tf.eye(256, dtype=D)))
            tf.debugging.assert_less(gram_error, tf.constant(1e-10, D))
            ca, cb = tf.zeros([256, 256], D), tf.zeros([256, 256], D)
            scalars, moments = tf.zeros([6], D), tf.zeros([2, 72], D)
            for start in range(0, int(nodes.shape[0]), 64):
                budget()
                values = block(nodes[start:start+64], weights[start:start+64], nodes, weights, ratio_a, ratio_b)
                ca, cb, scalars, moments = [x+y for x,y in zip((ca, cb, scalars, moments), values)]
            tf.debugging.assert_near(scalars[5], mass1, atol=1e-9)
            entry = dict(order=order, points_per_block=int(nodes.shape[0]), basis_gram_max_error=float(gram_error),
                         prior_exact_log_integral=float(log_z0), prior_retained_integral=float(tf.reduce_sum(weights*tf.exp(ratio_a))),
                         inherited_initial_amplitude_distance=float(tf.sqrt(tf.reduce_sum(weights*(tf.exp(.5*ratio_a)-tf.exp(.5*ratio_b))**2))),
                         inherited_joint_amplitude_distance=float(tf.sqrt(tf.maximum(0., 2.-2.*scalars[2]/tf.sqrt(scalars[0]*scalars[1])))))
            for index, (name, coeff) in enumerate((("target_a", ca), ("target_b", cb))):
                norm = scalars[index]
                with tf.device("/CPU:0"):
                    singular = tf.linalg.svd(coeff, compute_uv=False)
                    grouped = tt_svd(coeff, list(range(8)))
                    paired = tt_svd(coeff, [0, 4, 1, 5, 2, 6, 3, 7])
                captured = tf.reduce_sum(coeff*coeff)
                tf.debugging.assert_less_equal(captured, norm*(1+tf.constant(1e-9, D)))
                outside = tf.maximum(0., norm-captured)
                algebraic_saved_error = outside+tf.reduce_sum((coeff-fitted_coefficients)**2)
                tf.debugging.assert_near(algebraic_saved_error, scalars[3+index], atol=1e-8)
                mean = moments[index, :8]/norm
                covariance = tf.reshape(moments[index, 8:], [8, 8])/norm-mean[:, None]*mean[None, :]
                lx, lz = tf.linalg.cholesky(covariance[:4, :4]), tf.linalg.cholesky(covariance[4:, 4:])
                correlation = tf.linalg.triangular_solve(lx, covariance[:4, 4:])
                correlation = tf.transpose(tf.linalg.triangular_solve(lz, tf.transpose(correlation)))
                ranks = {}
                for rank in (1, 2, 3, 4, 6, 8, 12):
                    tail = tf.maximum(0., captured-tf.reduce_sum(singular[:rank]**2))
                    ranks[str(rank)] = dict(polynomial_block_rank_relative_rms=float(tf.sqrt((outside+tail)/norm)),
                                           unrestricted_block_rank_lower_bound=float(tf.sqrt(tail/norm)))
                entry[name] = dict(log_integral=float(tf.math.log(norm)+scale),
                                  saved_polynomial_log_integral=float(tf.math.log(mass1)+scale),
                                  saved_relative_rms=float(tf.sqrt(scalars[3+index]/norm)),
                                  degree3_projection_relative_rms=float(tf.sqrt(outside/norm)),
                                  rank_diagnostics=ranks,
                                  grouped_tt_svd_rank3_relative_rms=float(tf.sqrt((outside+tf.reduce_sum((coeff-grouped)**2))/norm)),
                                  paired_tt_svd_rank3_relative_rms=float(tf.sqrt((outside+tf.reduce_sum((coeff-paired)**2))/norm)),
                                  canonical_correlations=tf.linalg.svd(correlation, compute_uv=False),
                                  joint_mean=mean, joint_covariance=covariance,
                                  projection_identity_error=float(tf.abs(algebraic_saved_error-scalars[3+index])))
            entry["wall_seconds"] = time.monotonic()-stage_start
            write(out/f"order-{order:02d}.json", entry)
            results.append(entry)
            a = entry["target_a"]
            print(json.dumps(dict(order=order, saved_rms=a["saved_relative_rms"], poly_floor=a["degree3_projection_relative_rms"],
                                  rank3_floor=a["rank_diagnostics"]["3"], paired_rms=a["paired_tt_svd_rank3_relative_rms"],
                                  inherited_distance=entry["inherited_joint_amplitude_distance"], wall_seconds=entry["wall_seconds"])), flush=True)
        result = dict(status="EXECUTED", diagnostic_only=True, source_campaign=str(SOURCE / "campaign-01"),
                      orders=results, continuous_space_certificate=False, default_ready=False,
                      interpretation="finite quadrature rank constraints and frozen-target localization; no retuning or filtering promotion")
        write(out/"result.json", result)
        manifest.update(status="EXECUTED", gpu_allocator=tf.config.experimental.get_memory_info("GPU:0"))
    except Exception:
        manifest["status"] = "FAILED"
        (out/"traceback.log").write_text(traceback.format_exc())
        raise
    finally:
        manifest.update(wall_seconds=time.monotonic()-started, completed_utc=datetime.now(timezone.utc).isoformat())
        write(out/"run_manifest.json", manifest)


if __name__ == "__main__":
    main()
