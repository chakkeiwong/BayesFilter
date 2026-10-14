"""GPU/XLA diagnostic for source-native neural sampling, controller and resume."""
import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
REFERENCE = ROOT/"docs/plans/artifacts/controlled-smc-kdm-replication-20260928/kdm-neural-batch64-20261006-02/source/author-batch64.json"


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    out = args.output
    out.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    checks, cases = [], []
    result = dict(schema="kdm_training_bridge_gpu_diagnostic_v1",
                  replication="INCOMPLETE", scope="NATIVE_WIRING_AND_RESUME_NOT_FULL_TRAINING",
                  checks=checks, cases=cases, reference=str(REFERENCE),
                  reference_sha256=hashlib.sha256(REFERENCE.read_bytes()).hexdigest())
    try:
        import tensorflow as tf
        from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
        memory = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
        tf.config.experimental.enable_tensor_float_32_execution(False)
        tf.config.set_soft_device_placement(False)
        from bayesfilter.replication.kdm_neural_replay_tf import BearingsNeuralReplay
        from bayesfilter.replication.kdm_training_controller import SourceEpochController
        result.update(tensorflow_version=tf.__version__, jit_compile=True, tf32=False,
                      memory_policy=memory, initial_state="fixed captured source state",
                      random_backend="stateless Philox TF; independent of Torch random bits")
        reference = json.loads(REFERENCE.read_text())
        result["checkpoint_sha256"] = reference["checkpoint_sha256"]
        result["dataset_sha256"] = reference["dataset_sha256"]
        def boolean(name, value):
            checks.append(dict(name=name, passed=bool(value)))
        def compare(name, actual, expected, dtype):
            actual, expected = tf.convert_to_tensor(actual, dtype), tf.convert_to_tensor(expected, dtype)
            if actual.shape != expected.shape:
                checks.append(dict(name=name, passed=False, reason="shape mismatch"))
                return
            bound = tf.constant(1e-9 if dtype==tf.float64 else 128*2**-23,dtype)*(1+tf.abs(expected))
            scaled = tf.reduce_max(tf.abs(actual-expected)/bound)
            finite = tf.reduce_all(tf.math.is_finite(actual)) & tf.reduce_all(tf.math.is_finite(expected))
            ok = bool(finite.numpy()) and bool((scaled <= 1).numpy())
            checks.append(dict(name=name, passed=ok,
                               max_scaled_error=float(scaled.numpy()) if bool(tf.math.is_finite(scaled).numpy()) else None))
        with tf.device("/GPU:0"):
            for case in reference["cases"]:
                label = case["variant"]+"/"+case["dtype"]
                timings = {}
                def run(stage, function, *inputs):
                    write(out/"progress.json",dict(case=label, stage=stage, completed_cases=len(cases)))
                    tick=time.monotonic()
                    value=function(*inputs)
                    float(value["loss"].numpy())  # Synchronize timing.
                    timings[stage]=time.monotonic()-tick
                    return value
                model=BearingsNeuralReplay(case)
                dtype=model.dtype
                u=case["updates"][0]; initial=u["captured"]["initial"]; data=u["data"]
                tensor=lambda v:tf.convert_to_tensor(v,dtype)
                inputs=[tensor(initial["particles_downscaled"]), tensor(initial["particle_weights"]),
                        tensor(initial["bandwidths_downscaled"]),
                        tensor(initial.get("resampling_particle_weights",initial["particle_weights"])),
                        tensor(initial.get("resampling_bandwidths_downscaled",initial["bandwidths_downscaled"])),
                        tensor(data["observations"]),tensor(data["states"]),
                        tf.convert_to_tensor(data["ground_truth_mask"],tf.bool)]
                tape_inputs=[tensor(u["captured"]["samples"]),tensor(u["captured"]["noise"])]
                before=model.state_dict()
                # Cheap no-fire regression for the new reject-only optimizer guard.
                got=run("source_replay_compile",model.update,*inputs,*tape_inputs)
                boolean(label+"/source_valid_commit",got["valid"].numpy() and got["committed"].numpy())
                compare(label+"/source_loss",got["loss"],u["loss"],dtype)
                compare(label+"/source_gradnorm",got["gradient_norm"],u["gradient_norm"],dtype)
                source_keys=("particles","particle_weights","bandwidths",
                             "resampling_particle_weights","resampling_bandwidths")
                for idx,key in enumerate(source_keys):
                    fallback={"resampling_particle_weights":"particle_weights",
                              "resampling_bandwidths":"bandwidths"}.get(key,key)
                    compare(label+"/source/"+key,got["outputs"][idx],
                            [row.get(key,row.get(fallback)) for row in u["captured"]["outputs"]],dtype)
                for idx,name in enumerate(model.names):
                    for field in ("raw_gradients","clipped_gradients","parameters_after"):
                        compare(label+"/source/"+field+"/"+name,got[field][idx],u[field][name],dtype)
                    for field,key in (("first_moments","exp_avg"),("second_moments","exp_avg_sq")):
                        compare(label+"/source/"+field+"/"+name,got[field][idx],u["adam_after"][name][key],dtype)
                model.load_state_dict(before)
                native=run("native_compile",model.native_update,*inputs,tf.constant([731,1],tf.int64))
                boolean(label+"/native_valid_commit",native["valid"].numpy() and native["committed"].numpy())
                boolean(label+"/native_gpu","GPU:0" in native["loss"].device)
                model.load_state_dict(before)
                replay=run("native_tape_replay",model.update,*inputs,
                           native["outputs"][6][1:],native["outputs"][7])
                for field in ("loss","gradient_norm"):
                    compare(label+"/native_replay/"+field,native[field],replay[field],dtype)
                for idx in range(8):
                    compare(label+f"/native_replay/output{idx}",native["outputs"][idx],replay["outputs"][idx],dtype)
                for field in ("raw_gradients","clipped_gradients","parameters_after","first_moments","second_moments"):
                    for idx,name in enumerate(model.names):
                        compare(label+"/native_replay/"+field+"/"+name,native[field][idx],replay[field][idx],dtype)
                # Drive an actual scheduler reduction and assign all live TF rates.
                controller=SourceEpochController(case["learning_rates"],
                    dict(threshold=.001,factor=.1,patience=0,cooldown=0,min_lr=0.),
                    patience=3,start_offset=0)
                controller.observe(1.)
                event=controller.observe(1.)
                model.set_learning_rates(event["learning_rates"])
                checkpoint=dict(model=model.state_dict(),controller=controller.state_dict(),
                                next_seed=[731,2], data="development fixture rows0-63",
                                plan="docs/plans/controlled-smc-kdm-training-bridge-20261007.md")
                filename=out/(case["variant"]+"-"+case["dtype"]+"-checkpoint.json")
                write(filename,checkpoint)
                continued=run("native_steady_update",model.native_update,*inputs,tf.constant([731,2],tf.int64))
                continued_state=model.state_dict()
                # Restore from the on-disk artifact into the same compiled consumer:
                # resource variables change without retracing the computation.
                loaded=json.loads(filename.read_text())
                model.load_state_dict(loaded["model"])
                restored_controller=SourceEpochController.from_state_dict(loaded["controller"])
                boolean(label+"/checkpoint_roundtrip",
                        model.state_dict()==loaded["model"] and
                        restored_controller.state_dict()==loaded["controller"])
                resumed=run("native_resumed_update",model.native_update,*inputs,tf.constant(loaded["next_seed"],tf.int64))
                boolean(label+"/resume_exact_state",model.state_dict()==continued_state)
                compare(label+"/resume_loss",resumed["loss"],continued["loss"],dtype)
                # A change in the live learning-rate variables must affect updates
                # while preserving the same computed loss and gradients.
                model.load_state_dict(loaded["model"])
                model.set_learning_rates(case["learning_rates"])
                fullrate=run("native_rate_wiring",model.native_update,*inputs,tf.constant([731,2],tf.int64))
                for idx,name in enumerate(model.names):
                    old=tensor(loaded["model"]["parameters"][name])
                    compare(label+"/rate_effect/"+name,continued["parameters_after"][idx],
                            old+(fullrate["parameters_after"][idx]-old)*tf.constant(.1,dtype),dtype)
                boolean(label+"/controller_reduction_event",event["scheduler_event"])
                # Evaluation follows the same native sequence and does not mutate state.
                model.load_state_dict(loaded["model"])
                evaluation_before=model.state_dict()
                evaluated=run("native_eval_compile",model.native_evaluate,*inputs,tf.constant([731,2],tf.int64))
                compare(label+"/eval_training_loss",evaluated["loss"],continued["loss"],dtype)
                boolean(label+"/evaluation_no_state_change",model.state_dict()==evaluation_before)
                # Finite parameter giving an invalid bandwidth: reject the entire
                # update without poisoning Adam or advancing its step.
                bandname=("resampling_weighted_bandwidth_predictor.log_bandwidths"
                          if case["variant"]=="am_dpf" else "weighted_bandwidth_predictor.log_bandwidths")
                model.parameters[bandname].assign(tf.fill(model.parameters[bandname].shape,tf.constant(1000,dtype)))
                invalid_before=model.state_dict()
                invalid=run("invalid_sampler_guard",model.native_update,*inputs,tf.constant([731,3],tf.int64))
                boolean(label+"/invalid_rejected_without_update",
                        not bool(invalid["valid"].numpy()) and not bool(invalid["committed"].numpy())
                        and model.state_dict()==invalid_before)
                model.load_state_dict(loaded["model"])
                for name in ("update","native_update","native_evaluate"):
                    boolean(label+"/"+name+"/one_trace",getattr(model,name).experimental_get_tracing_count()==1)
                boolean(label+"/trainable_call_chain",model.used_parameter_names==set(model.names))
                cases.append(dict(case=label, timings=timings,
                    native_loss=float(native["loss"].numpy()),
                    max_rejection_iterations=int(tf.reduce_max(native["outputs"][8]).numpy()),
                    batch_size=model.batch,checkpoint=str(filename),
                    checkpoint_sha256=hashlib.sha256(filename.read_bytes()).hexdigest(),
                    failed_checks=[c for c in checks if c["name"].startswith(label+"/") and not c["passed"]]))
                write(out/"partial-result.json",result)
            # Source validation batch shape only; repeated fixture rows are not an
            # independent validation dataset and cannot qualify learned quality.
            for case in (c for c in reference["cases"] if c["dtype"]=="float32"):
                label=case["variant"]+"/float32/B256"
                fixture=copy.deepcopy(case);fixture["batch_size"]=256
                model=BearingsNeuralReplay(fixture)
                u=case["updates"][0];initial=u["captured"]["initial"];data=u["data"]
                tensor=lambda v:tf.convert_to_tensor(v,tf.float32)
                b64=[tensor(initial["particles_downscaled"]),tensor(initial["particle_weights"]),
                     tensor(initial["bandwidths_downscaled"]),
                     tensor(initial.get("resampling_particle_weights",initial["particle_weights"])),
                     tensor(initial.get("resampling_bandwidths_downscaled",initial["bandwidths_downscaled"])),
                     tensor(data["observations"]),tensor(data["states"]),
                     tf.convert_to_tensor(data["ground_truth_mask"],tf.bool)]
                inputs=[tf.tile(x,[4]+[1]*(x.shape.rank-1)) for x in b64]
                timings={}
                first=run("B256_eval_compile",model.native_evaluate,*inputs,tf.constant([731,4],tf.int64))
                second=run("B256_eval_steady",model.native_evaluate,*inputs,tf.constant([731,4],tf.int64))
                boolean(label+"/valid",first["valid"].numpy() and second["valid"].numpy())
                compare(label+"/deterministic_repeat",first["loss"],second["loss"],tf.float32)
                boolean(label+"/one_trace",model.native_evaluate.experimental_get_tracing_count()==1)
                cases.append(dict(case=label,timings=timings,
                                  scope="duplicated fixture shape/cost diagnostic",
                                  loss=float(first["loss"].numpy())))
        result.update(status="PASS" if all(c["passed"] for c in checks) else "FAIL",
                      allocator=tf.config.experimental.get_memory_info("GPU:0"))
    except Exception:
        result.update(status="ERROR", error=traceback.format_exc())
        traceback.print_exc()
    finally:
        result["wall_seconds"]=time.monotonic()-started
        write(out/"result.json",result)
    print(json.dumps(dict(status=result["status"],checks=len(checks),
                         passed=sum(c["passed"] for c in checks),wall_seconds=result["wall_seconds"])))
    return 0 if result["status"]=="PASS" else 1


if __name__=="__main__":
    raise SystemExit(main())
