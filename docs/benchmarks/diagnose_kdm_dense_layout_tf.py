"""Diagnostic intervention at the actual source-replay full-sequence endpoint.

Reuses difficult development inputs; neither training nor qualification.
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
ROOT = Path("/home/chakwong/BayesFilter")
sys.path.insert(0, str(ROOT))

def main(output, reference):
    import tensorflow as tf
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
    memory = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    tf.config.experimental.enable_tensor_float_32_execution(False)
    tf.config.set_soft_device_placement(False)
    from bayesfilter.replication.kdm_neural_replay_tf import BearingsNeuralReplay

    class SourceLayout(BearingsNeuralReplay):
        def mlp(self, x, prefix):
            shape = tf.shape(x)
            y = super().mlp(tf.reshape(x, [-1, x.shape[-1]]), prefix)
            return tf.reshape(y, tf.concat([shape[:-1], [tf.shape(y)[-1]]], 0))

    def compare(a, b, dtype):
        b = tf.convert_to_tensor(b, dtype)
        assert a.shape == b.shape
        error = tf.abs(a-b)
        tol = 128*2**-23 if dtype == tf.float32 else 1e-9
        scale = error / (tf.constant(tol, dtype)*(1+tf.abs(b)))
        finite = bool(tf.reduce_all(tf.math.is_finite(a)).numpy())
        return dict(passed=finite and bool(tf.reduce_all(scale <= 1).numpy()),
                    max_absolute_error=float(tf.reduce_max(error).numpy()) if finite else None,
                    max_scaled_error=float(tf.reduce_max(scale).numpy()) if finite else None)

    ref = json.loads(reference.read_text())
    records = []
    with tf.device("/GPU:0"):
        for case in ref["cases"]:
            for layout, cls in (("runtime", BearingsNeuralReplay), ("source_flattened", SourceLayout)):
                tick = time.monotonic()
                model = cls(case, batch_size=case["batch_size"])
                dtype = model.dtype
                tensor = lambda x: tf.convert_to_tensor(x, dtype)
                def value_gradient(*inputs):
                    with tf.GradientTape() as tape:
                        loss, valid, outputs = model._sequence(*inputs)
                    gradients = tape.gradient(loss, model.trainable)
                    return loss, valid, outputs, gradients
                graph = tf.function(value_gradient, input_signature=model.evaluate.input_signature,
                                    autograph=False, jit_compile=True)
                for index, update in enumerate(case["updates"]):
                    parameters = dict(case["parameters"])
                    if index: parameters.update(case["updates"][index-1]["parameters_after"])
                    for name, p in model.parameters.items(): p.assign(tensor(parameters[name]))
                    captured, data = update["captured"], update["data"]
                    initial = captured["initial"]
                    loc, w, b = [initial[k] for k in ("particles_downscaled", "particle_weights", "bandwidths_downscaled")]
                    rw = initial.get("resampling_particle_weights", w)
                    rb = initial.get("resampling_bandwidths_downscaled", b)
                    inputs = [tensor(x) for x in (loc,w,b,rw,rb,data["observations"],data["states"])]
                    inputs += [tf.convert_to_tensor(data["ground_truth_mask"], tf.bool),
                               tensor(captured["samples"]), tensor(captured["noise"])]
                    loss, valid, outputs, gradients = graph(*inputs)
                    assert bool(valid.numpy()) and all(g is not None for g in gradients)
                    assert "GPU" in loss.device
                    checks = {name: compare(g, update["raw_gradients"][name], dtype)
                              for name, g in zip(model.names, gradients)}
                    row = dict(variant=case["variant"], dtype=case["dtype"], layout=layout,
                               update=index, loss=compare(loss,update["loss"],dtype), gradients=checks,
                               valid=True, failed_gradients=[k for k,v in checks.items() if not v["passed"]],
                               raw_proposal=compare(outputs[5],captured["proposal_before_normalization"],dtype))
                    records.append(row)
                assert graph.experimental_get_tracing_count() == 1
                # Replayed initial state bypasses frozen initialization networks.
                assert set(model.names).issubset(model.used_parameter_names)
                print(json.dumps(dict(variant=case["variant"], dtype=case["dtype"],layout=layout,
                    failed=sum(len(r["failed_gradients"]) for r in records[-4:]),
                    tracing_count=graph.experimental_get_tracing_count(), wall_seconds=time.monotonic()-tick)),flush=True)
                output.write_text(json.dumps(dict(schema="kdm_dense_layout_diagnostic_v1", records=records,
                    memory_policy=memory, tensorflow=tf.__version__, jit_compile=True, tf32=False,
                    reference=str(reference), common_source_preupdate_states=True,
                    interpretation="DEVELOPMENT_MECHANISM_DIAGNOSTIC_NOT_QUALIFICATION"), indent=2)+"\n")

if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--reference",type=Path,required=True)
    args=parser.parse_args();main(args.output,args.reference)
