"""Bounded source-stage GPU/XLA diagnostic; never full replication evidence."""
import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--reference", required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=["verify", "epoch"])
    parser.add_argument("--verification", type=Path)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    checks, timings = [], []
    result = dict(schema="kdm_source_stages_tf_diagnostic_v1", replication="INCOMPLETE",
        mode=args.mode, scope="SOURCE_STAGE_MECHANICS_AND_COST_NOT_TRAINING_QUALIFICATION",
        checks=checks, timings=timings, reference=str(args.reference),
        reference_sha256=hashlib.sha256(args.reference.read_bytes()).hexdigest())
    try:
        import tensorflow as tf
        from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
        memory = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
        tf.config.experimental.enable_tensor_float_32_execution(False)
        tf.config.set_soft_device_placement(False)
        from bayesfilter.replication.kdm_neural_replay_tf import BearingsNeuralReplay
        from bayesfilter.replication.kdm_source_stages_tf import SourceInitialCloud, SourceWeightPretraining, SourceFullTraining
        result.update(tensorflow_version=tf.__version__, jit_compile=True, tf32=False,
            memory_policy=memory, visible_device=os.environ.get("CUDA_VISIBLE_DEVICES"),
            initial_checkpoint_role="development fixture, not completed stage training",
            random_backend="TF stateless Philox, same law not Torch random bits",
            existing_float32_failures="preserved; this diagnostic cannot erase prior failures")
        reference = json.loads(args.reference.read_text())
        baseline_path = Path(reference["baseline"])
        if hashlib.sha256(baseline_path.read_bytes()).hexdigest() != reference["baseline_sha256"]:
            raise ValueError("Baseline hash mismatch")
        baseline = json.loads(baseline_path.read_text())
        for info in reference["datasets"].values():
            if hashlib.sha256(Path(info["path"]).read_bytes()).hexdigest() != info["sha256"]:
                raise ValueError("Dataset hash mismatch")
        result["datasets"] = reference["datasets"]
        result["checkpoint_sha256"] = reference["checkpoint_sha256"]
        def boolean(name, value):
            checks.append(dict(name=name, passed=bool(value)))
        def compare(name, actual, expected, dtype):
            actual, expected = tf.convert_to_tensor(actual, dtype), tf.convert_to_tensor(expected, dtype)
            if actual.shape != expected.shape:
                checks.append(dict(name=name, passed=False, reason="shape mismatch",
                    actual_shape=actual.shape.as_list(), expected_shape=expected.shape.as_list()))
                return
            bound = tf.constant(1e-9 if dtype == tf.float64 else 128*2**-23, dtype)*(1+tf.abs(expected))
            scaled = tf.reduce_max(tf.abs(actual-expected)/bound)
            finite = tf.reduce_all(tf.math.is_finite(actual)) & tf.reduce_all(tf.math.is_finite(expected))
            value = float(scaled.numpy()) if bool(tf.math.is_finite(scaled).numpy()) else None
            checks.append(dict(name=name, passed=bool(finite.numpy()) and value is not None and value <= 1,
                maximum_bound_ratio=value))
        def run(label, function, *inputs):
            write(out/"progress.json", dict(stage=label, completed_checks=len(checks),
                failed_checks=sum(not c["passed"] for c in checks)))
            tick = time.monotonic()
            value = function(*inputs)
            bool(value["valid"].numpy())
            timings.append(dict(stage=label, wall_seconds=time.monotonic()-tick))
            return value
        def sync_checks():
            write(out/"checks.partial.json", checks)
        def data_inputs(data, dtype):
            return [tf.constant(data["states"], dtype), tf.constant(data["observations"], dtype),
                    tf.constant(data["ground_truth_mask"], tf.bool)]
        def update_source(label, model, got, expected):
            d = model.dtype
            boolean(label+"/valid_commit", got["valid"].numpy() and got["committed"].numpy())
            for field in ("loss", "gradient_norm"):
                compare(label+"/"+field, got[field], expected[field], d)
            for i, name in enumerate(model.names):
                for field in ("raw_gradients", "parameters_after"):
                    compare(label+"/"+field+"/"+name, got[field][i], expected[field][name], d)
                for field in ("optimizer_gradients", "clipped_gradients"):
                    if field in expected:
                        compare(label+"/"+field+"/"+name, got[field][i], expected[field][name], d)
                for field, key in (("first_moments", "exp_avg"), ("second_moments", "exp_avg_sq")):
                    compare(label+"/"+field+"/"+name, got[field][i], expected["adam_after"][name][key], d)
        def update_pair(label, model, a, b):
            boolean(label+"/valid_commit", a["valid"].numpy() and b["valid"].numpy()
                and a["committed"].numpy() and b["committed"].numpy())
            for field in ("loss", "gradient_norm"):
                compare(label+"/"+field, a[field], b[field], model.dtype)
            for field in ("raw_gradients", "optimizer_gradients", "parameters_after", "first_moments", "second_moments"):
                for i, name in enumerate(model.names):
                    compare(label+"/"+field+"/"+name, a[field][i], b[field][i], model.dtype)
        if args.mode == "epoch":
            from diagnose_kdm_source_epoch_tf import run_epochs
            run_epochs(tf,reference,baseline,args,out,result,boolean,run,write)
        else:
            with tf.device("/GPU:0"):
                for case in reference["initial_cases"]:
                    label = "initial/"+case["variant"]+"/"+case["dtype"]
                    fixture = next(c for c in baseline["cases"]
                        if c["variant"] == case["variant"] and c["dtype"] == case["dtype"])
                    model = BearingsNeuralReplay(fixture)
                    initial = SourceInitialCloud(model)
                    truth = tf.constant(case["data"]["states"], model.dtype)[:, 0, :]
                    got = run(label+"/source_replay", initial.replay, truth, tf.constant(case["noise"], model.dtype))
                    boolean(label+"/valid", got["valid"].numpy())
                    source = case["output"]
                    for key in ("particles", "particle_weights", "bandwidths", "resampling_particle_weights", "resampling_bandwidths"):
                        fallback = {"resampling_particle_weights":"particle_weights", "resampling_bandwidths":"bandwidths"}.get(key, key)
                        compare(label+"/"+key, got[key], (source[key] if source.get(key) is not None else source[fallback]), model.dtype)
                    native = run(label+"/native", initial.sample, truth, tf.constant([811, 1], tf.int64))
                    replay = run(label+"/native_replay", initial.replay, truth, native["noise"])
                    boolean(label+"/native_valid", native["valid"].numpy())
                    for key in ("particles", "particle_weights", "bandwidths", "resampling_bandwidths"):
                        compare(label+"/native_replay/"+key, native[key], replay[key], model.dtype)
                    boolean(label+"/one_trace", initial.sample.experimental_get_tracing_count() == 1
                        and initial.replay.experimental_get_tracing_count() == 1)
                    sync_checks()
                for case in reference["pretraining_cases"]:
                    label = "pretrain/"+case["dtype"]
                    model = BearingsNeuralReplay(case)
                    stage = SourceWeightPretraining(model, len(case["updates"][0]["data"]["states"]))
                    d = model.dtype
                    for index, u in enumerate(case["updates"]):
                        inputs = data_inputs(u["data"], d)
                        got = run(label+f"/source_update{index}", stage.update, *inputs, tf.constant(u["captured"]["cloud"], d))
                        update_source(label+f"/update{index}", model, got, u)
                        for key in ("selected_states", "selected_observations"):
                            compare(label+f"/update{index}/"+key, got["outputs"][key], u[key], d)
                        boolean(label+f"/effective_batch{index}", stage.model.batch == u["effective_batch"])
                    before = model.state_dict()
                    inputs = data_inputs(case["updates"][0]["data"], d)
                    native = run(label+"/native_update", stage.native_update, *inputs, tf.constant([811, 2], tf.int64))
                    model.load_state_dict(before)
                    replay = run(label+"/native_replay", stage.update, *inputs, native["outputs"]["particles"])
                    update_pair(label+"/native_replay", model, native, replay)
                    before_eval = model.state_dict()
                    evaluation = run(label+"/validation", stage.native_evaluate, *inputs, tf.constant([811, 3], tf.int64))
                    boolean(label+"/validation_valid", evaluation["valid"].numpy())
                    boolean(label+"/validation_immutable", model.state_dict() == before_eval)
                    partial = SourceWeightPretraining(model, 8)
                    boolean(label+"/shared_partial_variables", partial.model.step is model.step
                        and all(partial.model.parameters[k] is v for k,v in model.parameters.items())
                        and all(a is b for a,b in zip(partial.model.moments1, model.moments1)))
                    boolean(label+"/one_trace", stage.update.experimental_get_tracing_count() == 1
                        and stage.native_update.experimental_get_tracing_count() == 1
                        and stage.native_evaluate.experimental_get_tracing_count() == 1)
                    sync_checks()
                for case in baseline["cases"]:
                    label = "full/"+case["variant"]+"/"+case["dtype"]
                    model = BearingsNeuralReplay(case)
                    d = model.dtype
                    u = case["updates"][0]
                    c = u["captured"]
                    initial = c["initial"]
                    tensor = lambda x: tf.constant(x, d)
                    inputs = [tensor(initial["particles_downscaled"]), tensor(initial["particle_weights"]),
                        tensor(initial["bandwidths_downscaled"]),
                        tensor(initial.get("resampling_particle_weights", initial["particle_weights"])),
                        tensor(initial.get("resampling_bandwidths_downscaled", initial["bandwidths_downscaled"])),
                        tensor(u["data"]["observations"]), tensor(u["data"]["states"]),
                        tf.constant(u["data"]["ground_truth_mask"], tf.bool)]
                    before = model.state_dict()
                    got = run(label+"/regression", model.update, *inputs, tensor(c["samples"]), tensor(c["noise"]))
                    update_source(label+"/source", model, got, u)
                    model.load_state_dict(before)
                    stage = SourceFullTraining(model)
                    dat = data_inputs(u["data"], d)
                    seed = tf.constant([731, 2], tf.int64)
                    native = run(label+"/native_initial_full", stage.update, *dat, seed)
                    model.load_state_dict(before)
                    # Capture the exact initial cloud consumed by the native sequence.
                    # fold_in returns int32; crossing the standalone int64 seed signature
                    # is unnecessary and would test an adapter rather than this call chain.
                    init = run(label+"/replay_initial", stage.initial.replay, dat[0][:,0,:],
                        tf.zeros_like(native["outputs"][6][0]))
                    init["particles"] = native["outputs"][6][0]
                    recurrent_inputs = [init[k] for k in ("particles", "particle_weights", "bandwidths",
                        "resampling_particle_weights", "resampling_bandwidths")]
                    replay = run(label+"/full_tape_replay", model.update, *recurrent_inputs, dat[1], dat[0], dat[2],
                        native["outputs"][6][1:], native["outputs"][7])
                    update_pair(label+"/initial_full_wiring", model, native, replay)
                    partial = model.for_batch(8)
                    boolean(label+"/shared_partial_variables", partial.step is model.step
                        and all(partial.parameters[k] is v for k,v in model.parameters.items())
                        and all(a is b for a,b in zip(partial.rates, model.rates)))
                    boolean(label+"/one_trace", stage.update.experimental_get_tracing_count() == 1
                        and model.update.experimental_get_tracing_count() == 1)
                    sync_checks()
                from bayesfilter.replication.kdm_stage_transfer_tf import transfer_stage_parameters
                for source in reference["stage_transfers"]:
                    label = "transfer/"+source["dtype"]
                    pre = next(c for c in reference["pretraining_cases"] if c["dtype"]==source["dtype"])
                    pre_model = BearingsNeuralReplay(pre)
                    state = pre_model.state_dict()
                    state["parameters"] = source["pretraining_parameters"]
                    mdpf = BearingsNeuralReplay(source["fixtures"]["mdpf"])
                    transfer_stage_parameters(mdpf,state,"weights_to_mdpf")
                    for key,value in source["mdpf_after"].items():
                        compare(label+"/weights_to_mdpf/"+key,mdpf.parameters[key],value,mdpf.dtype)
                    am = BearingsNeuralReplay(source["fixtures"]["am_dpf"])
                    am.step.assign(am.c(3))
                    for variable in am.moments1+am.moments2:
                        variable.assign(tf.ones_like(variable))
                    transfer_stage_parameters(am,mdpf.state_dict(),"mdpf_to_am_dpf")
                    for key,value in source["am_dpf_after"].items():
                        compare(label+"/mdpf_to_am_dpf/"+key,am.parameters[key],value,am.dtype)
                    boolean(label+"/optimizer_reset",float(am.step.numpy())==0 and
                        all(bool(tf.reduce_all(v==0).numpy()) for v in am.moments1+am.moments2))
                    bad = mdpf.state_dict()
                    bad["parameters"].pop(next(iter(bad["parameters"])))
                    before = am.state_dict()
                    rejected = False
                    try:
                        transfer_stage_parameters(am,bad,"mdpf_to_am_dpf")
                    except ValueError:
                        rejected = True
                    boolean(label+"/incomplete_transfer_atomic",rejected and am.state_dict()==before)
                    sync_checks()
                result["stage_transfers_checked"] = True
        result["allocator"] = tf.config.experimental.get_memory_info("GPU:0")
        result["status"] = "PASS" if all(c["passed"] for c in checks) else "FAIL"
    except Exception as exc:
        result.update(status="ERROR", exception=repr(exc), traceback=traceback.format_exc())
    result.update(wall_seconds=time.monotonic()-started, checks_total=len(checks),
        checks_failed=sum(not c["passed"] for c in checks),
        not_concluded=["full training", "no-worse trained performance", "all float32 compatibility"])
    write(out/"result.json", result)
    print(json.dumps({k:result[k] for k in ("status", "mode", "checks_total", "checks_failed", "wall_seconds")}))
    return 0 if result["status"] == "PASS" else 1

if __name__ == "__main__":
    raise SystemExit(main())
