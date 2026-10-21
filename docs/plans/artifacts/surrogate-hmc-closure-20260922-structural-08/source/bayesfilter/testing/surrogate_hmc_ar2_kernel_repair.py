"""Numerical non-harm and compiled-consumer cost for the shared kernel repair.

This diagnostic deliberately evaluates an unqualified candidate. It cannot
issue a control selection, tune HMC, or launch posterior sampling.
"""
import json
from pathlib import Path
import re

import tensorflow as tf

from bayesfilter.highdim.ledh_numerical_safety_tf import safe_cholesky
from bayesfilter.highdim.ledh_small_matrix_tf import cholesky_solve
from bayesfilter.highdim.ledh_structural_ar2_target_tf import make_ar2_target
from bayesfilter.inference.surrogate_hmc_program import ROOT, file_hash, write_json
from bayesfilter.testing.surrogate_hmc_ar2_controls import PARTICLE_COUNT, protection_controls
from bayesfilter.testing.surrogate_hmc_ar2_sampling import load_reference, _timed

BASELINE = ROOT/"docs/plans/artifacts/surrogate-hmc-closure-20260922-structural-07/ar2-profile/report.json"


def primitive_checks():
    rows = []
    for dtype in (tf.float32, tf.float64):
        for order in (1, 2, 3):
            @tf.function(input_signature=[tf.TensorSpec([4, order, order], dtype), tf.TensorSpec([4, order, 2], dtype)], jit_compile=True, autograph=False)
            def evaluate(matrix, rhs):
                valid, factor = safe_cholesky(matrix)
                return valid, factor, cholesky_solve(factor, rhs), tf.linalg.cholesky(matrix), tf.linalg.cholesky_solve(tf.linalg.cholesky(matrix), rhs)
            raw = tf.random.stateless_normal([4, order, order], [64, order], dtype=dtype)
            scales = tf.constant([1e-15, 1., 1e8, 1e15], dtype)[:, None, None]
            matrix = (raw @ tf.linalg.matrix_transpose(raw)+tf.eye(order, dtype=dtype))*scales
            rhs = tf.random.stateless_normal([4, order, 2], [65, order], dtype=dtype)*scales
            valid, factor, solved, reference_factor, reference_solved = evaluate(matrix, rhs)
            factor_error = float(tf.reduce_max(tf.abs(factor-reference_factor)/(tf.sqrt(scales)+tf.abs(reference_factor))))
            solve_error = float(tf.reduce_max(tf.abs(solved-reference_solved)/(1+tf.abs(reference_solved))))
            tolerance = 5e-6 if dtype == tf.float32 else 5e-13
            rows.append(dict(dtype=dtype.name, order=order, factor_error=factor_error, solve_error=solve_error,
                tolerance=tolerance, passed=bool(tf.reduce_all(valid)) and max(factor_error, solve_error) <= tolerance))
    @tf.function(input_signature=[tf.TensorSpec([4, 2, 2], tf.float32)], jit_compile=True, autograph=False)
    def rejected(matrix):
        return safe_cholesky(matrix)
    validity, factor = rejected(tf.constant([[[1., 1.], [1., 1.]], [[1., 2.], [2., 1.]],
        [[0., 0.], [0., 1.]], [[1., float("nan")], [0., 1.]]], tf.float32))
    rows.append(dict(case="invalid_covariances", passed=not bool(tf.reduce_any(validity)) and
                    bool(tf.reduce_all(factor == tf.eye(2, batch_shape=[4], dtype=tf.float32)))))
    return rows


def kernel_repair_stage(output):
    out = Path(output).parent
    out.mkdir(parents=True, exist_ok=True)
    old = json.loads(BASELINE.read_text())
    if not old["qualified"]:
        raise ValueError("qualified unchanged-force baseline required")
    data, _, arrays = load_reference()
    target = make_ar2_target(data, particle_count=PARTICLE_COUNT, controls=protection_controls(1.))
    points_list = (arrays["means"][None]+tf.constant([[0., 0.], [1., 0.], [0., 1.], [-.5, -.5]], tf.float64)*arrays["scales"][None]).numpy().tolist()
    report = dict(status="RUNNING", qualified=False, baseline_sha256=file_hash(BASELINE), target_manifest=dict(target.manifest),
        arms={}, primitive_checks=[], controls_selection_qualified=False, inference_launched=False,
        cpu_role="explicit numerical/cost reference only", ranking="NOT_ESTABLISHED")
    devices = [d.name for d in tf.config.list_logical_devices("GPU")]+["/CPU:0"]
    for device in devices:
        with tf.device(device):
            if "GPU" in device and not report["primitive_checks"]:
                report["primitive_checks"] = primitive_checks()
            points = tf.constant(points_list, tf.float32)
            evaluate = target.compile(4)
            costs, result = _timed(lambda _: evaluate(points), repetitions=4)
            value, score, status = result
            original = old["arms"][device]
            value_error = float(tf.reduce_max(tf.abs(value-tf.constant(original["values"], tf.float32))/(1+tf.abs(value))))
            score_error = float(tf.reduce_max(tf.abs(score-tf.constant(original["scores"], tf.float32))/(1+tf.abs(score))))
            valid = bool(tf.reduce_all(status["program_valid"]))
            hlo = evaluate.experimental_get_compiler_ir(points)(stage="optimized_hlo")
            (out/(device.replace("/", "").replace(":", "-")+".hlo")).write_text(hlo)
            calls = re.findall(r'custom_call_target="([^"]+)"', hlo)
            targeted = [c for c in calls if "cholesky" in c.lower() or "triangularsolve" in c.lower()]
            eliminated = not targeted if "GPU" in device else True
            arm = dict(device=device, output_device=value.device, trace_count=evaluate.experimental_get_tracing_count(), costs=costs,
                valid=valid, value_relative_error=value_error, score_relative_error=score_error,
                precision_parity=valid and original["valid"] and max(value_error, score_error) <= .002,
                targeted_custom_calls=targeted, targeted_calls_eliminated=eliminated,
                values=value.numpy().tolist(), scores=score.numpy().tolist())
            if "GPU" in device:
                arm["allocator"] = tf.config.experimental.get_memory_info(device)
            report["arms"][device] = arm
            write_json(output, report)
            print("AR2_KERNEL_REPAIR", device, "parity", arm["precision_parity"], "calls eliminated", eliminated,
                  "steady", [r["wall_seconds"] for r in costs[1:]], flush=True)
    passed = all(r["passed"] for r in report["primitive_checks"]) and all(a["precision_parity"] and a["targeted_calls_eliminated"] for a in report["arms"].values())
    report.update(qualified=passed, status="AR2_KERNEL_NONHARM_PASSED" if passed else "AR2_KERNEL_REPAIR_FAILED",
        next_action="Existing master: requalify scoped controls and full-transition cost before calibration and posterior sampling",
        all_models_validated=False, default_readiness=False)
    return report
