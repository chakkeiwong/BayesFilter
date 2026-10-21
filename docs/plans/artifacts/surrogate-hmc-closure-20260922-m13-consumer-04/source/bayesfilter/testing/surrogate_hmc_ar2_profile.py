"""Explicit-device cost localization of the unchanged canonical AR2 force."""
from pathlib import Path
import time

import tensorflow as tf

from bayesfilter.inference.surrogate_hmc_program import write_json
from bayesfilter.testing.surrogate_hmc_ar2_sampling import load_reference, selected_consumer, _timed


def profile_stage(output):
    out = Path(output).parent
    out.mkdir(parents=True, exist_ok=True)
    data, _, arrays = load_reference()
    consumer = selected_consumer(data)
    values = (arrays["means"][None]+tf.constant([[0., 0.], [1., 0.], [0., 1.], [-.5, -.5]], tf.float64)*arrays["scales"][None]).numpy().tolist()
    physical = tf.config.list_physical_devices("GPU")
    devices = [d.name for d in tf.config.list_logical_devices("GPU")]+["/CPU:0"]
    report = dict(status="RUNNING", qualified=False, arms={}, device_details=[dict(name=d.name,
        details=tf.config.experimental.get_device_details(d)) for d in physical],
        inference_launched=False, controls_changed=False, cpu_role="explicit reference/localization only; GPU remains default")
    baseline = None
    for index, device in enumerate(devices):
        with tf.device(device):
            points = tf.constant(values, tf.float32)
            # Own the outer device-bound graph, while retaining exactly the
            # same canonical target callable and frozen source/input identity.
            @tf.function(input_signature=[tf.TensorSpec([4, 2], tf.float32)], jit_compile=True, autograph=False)
            def evaluate(q):
                return consumer.fixture.evaluate(q)

            costs, result = _timed(lambda _: evaluate(points), repetitions=4)
            value, score, status = result
            if baseline is None:
                baseline = (value, score, status["program_valid"])
            value_change = tf.reduce_max(tf.abs(value-baseline[0])/(1+tf.abs(baseline[0])))
            score_change = tf.reduce_max(tf.abs(score-baseline[1])/(1+tf.abs(baseline[1])))
            valid = bool(tf.reduce_all(status["program_valid"]))
            parity = bool(tf.reduce_all(status["program_valid"] == baseline[2])) and float(value_change) <= .002 and float(score_change) <= .002
            arm = dict(device=device, output_device=value.device, input_device=points.device, costs=costs,
                valid=valid, precision_parity=parity, value_relative_change=float(value_change), score_relative_change=float(score_change),
                graph_trace_count=evaluate.experimental_get_tracing_count(), values=value.numpy().tolist(), scores=score.numpy().tolist())
            begin = time.monotonic()
            try:
                hlo = evaluate.experimental_get_compiler_ir(points)(stage="optimized_hlo")
                (out/f"device-{index}-optimized.hlo").write_text(hlo)
                arm["hlo"] = dict(bytes=len(hlo), while_ops=hlo.count(" while("), custom_calls=hlo.count("custom-call("),
                    fusion_ops=hlo.count(" fusion("), inspection_seconds=time.monotonic()-begin)
            except (ValueError, tf.errors.OpError) as exc:
                arm["hlo_inspection_error"] = str(exc)
            if "GPU" in device:
                arm["allocator"] = tf.config.experimental.get_memory_info(device)
            report["arms"][device] = arm
            write_json(output, report)
            print("AR2_PROFILE", device, "valid", valid, "parity", parity, "steady", [r["wall_seconds"] for r in costs[1:]], flush=True)
    report.update(status="AR2_COST_LOCALIZATION_COMPLETED", qualified=all(row["valid"] and row["precision_parity"] for row in report["arms"].values()),
        next_action="Existing master: use measured placement/compiler evidence to repair cost, then refresh full-transition forecast before allocating chains",
        ranking="NOT_ESTABLISHED", all_models_validated=False, default_readiness=False)
    return report
