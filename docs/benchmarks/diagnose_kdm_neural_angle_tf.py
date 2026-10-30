"""Bounded GPU/XLA diagnostic of the pinned author neural KDM consumer."""
import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["BAYESFILTER_PRELOAD_CUSTOM_OP"] = "0"
import argparse
import datetime
import hashlib
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
import time
import traceback

ROOT = Path("/home/chakwong/BayesFilter")
sys.path.insert(0,str(ROOT))
PLAN = "docs/plans/controlled-smc-kdm-neural-angle-conditioning-20261007.md"
RESULT = "docs/plans/controlled-smc-kdm-neural-angle-conditioning-result-20261007.md"

def write(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--reference",type=Path,required=True)
    args=parser.parse_args()
    out=args.output.resolve()
    out.mkdir(exist_ok=False)
    sources=out/"sources";sources.mkdir()
    files=[Path(__file__),ROOT/"docs/benchmarks/reference_kdm_neural_conditioning.py",
           ROOT/"bayesfilter/replication/kdm_neural_replay_tf.py",
           ROOT/"bayesfilter/replication/kdm_normal_von_mises_tf.py",
           ROOT/"bayesfilter/runtime/gpu_memory_policy.py",ROOT/PLAN]
    for path in files:
        shutil.copyfile(path,sources/path.name)
    manifest=dict(schema="kdm_neural_phase_manifest_v1",command=[sys.executable]+sys.argv,
        git_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        branch=subprocess.check_output(["git","branch","--show-current"],cwd=ROOT,text=True).strip(),
        environment=sys.executable,plan_file=PLAN,result_file=RESULT,
        start_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        data_version="recorded diagnostic development rows; not qualification data",
        random_seeds=[],cpu_wall_seconds=0.,gpu_wall_seconds=0.,
        cpu_gpu_status="CPU independent author; GPU TF XLA; source draws replayed",
        gpu_trust_basis="escalated_local_gpu_access",replication="INCOMPLETE")
    write(out/"manifest.json",manifest)
    gpu_start=None
    started=time.monotonic()
    try:
        shutil.copyfile(args.reference,out/"author.json")
        manifest["reference_reused_from"]=str(args.reference.resolve())
        reference=json.loads((out/"author.json").read_text())
        manifest["random_seeds"]=reference["diagnostic_seeds"]
        manifest["reference_components"]=reference["reference_components"]
        manifest["reference_sha256"]=hashlib.sha256((out/"author.json").read_bytes()).hexdigest()
        manifest["checkpoint_sha256"]=reference["checkpoint_sha256"]
        manifest["data_sha256"]=reference["dataset_sha256"]
        write(out/"manifest.json",manifest)
        gpu_start=time.monotonic()
        import tensorflow as tf
        from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
        memory=configure_tensorflow_gpu_memory_growth(tf,require_gpu=True)
        tf.config.experimental.enable_tensor_float_32_execution(False)
        tf.config.set_soft_device_placement(False)
        from bayesfilter.replication.kdm_neural_replay_tf import BearingsNeuralReplay
        checks=[];case_results=[];conditioning_diagnostics=[];legacy_particle_checks=[];angle_checks=[]
        def boolean(name,value):
            checks.append(dict(name=name,pass_check=bool(value)))
        def compare(name,actual,expected,dtype):
            expected=tf.convert_to_tensor(expected,dtype)
            actual=tf.convert_to_tensor(actual,dtype)
            if actual.shape != expected.shape:
                checks.append(dict(name=name,pass_check=False,actual_shape=actual.shape.as_list(),
                                   expected_shape=expected.shape.as_list()))
                return
            tolerance=128*2**-23 if dtype==tf.float32 else 1e-9
            bound=tf.constant(tolerance,dtype)*(1+tf.abs(expected))
            difference=tf.abs(actual-expected)
            finite=tf.reduce_all(tf.math.is_finite(actual)) & tf.reduce_all(tf.math.is_finite(expected))
            checks.append(dict(name=name,pass_check=bool((finite & tf.reduce_all(difference<=bound)).numpy()),
                max_absolute_error=float(tf.reduce_max(difference).numpy()),
                max_scaled_error=float(tf.reduce_max(difference/bound).numpy()),tolerance=tolerance))
        with tf.device("/GPU:0"):
            for case in reference["cases"]:
                label=case["fixture_id"]+"/"+case["variant"]+"/"+case["dtype"]
                write(out/"progress.json",dict(stage="TF_NEURAL_REPLAY",case=label,completed=len(case_results)))
                tick=time.monotonic()
                model=BearingsNeuralReplay(case, batch_size=int(case.get("batch_size", 2)))
                dtype=model.dtype
                timings=[];losses=[];devices=[]
                for update_index,update in enumerate(case["updates"]):
                    capture=update["captured"];initial=capture["initial"];data=update["data"]
                    tensor=lambda value:tf.convert_to_tensor(value,dtype)
                    inputs=[tensor(initial["particles_downscaled"]),tensor(initial["particle_weights"]),
                        tensor(initial["bandwidths_downscaled"]),
                        tensor(initial.get("resampling_particle_weights",initial["particle_weights"])),
                        tensor(initial.get("resampling_bandwidths_downscaled",initial["bandwidths_downscaled"])),
                        tensor(data["observations"]),tensor(data["states"]),
                        tf.convert_to_tensor(data["ground_truth_mask"],tf.bool),
                        tensor(capture["samples"]),tensor(capture["noise"])]
                    begin=time.monotonic()
                    got=model.update(*inputs)
                    boolean(label+f"/update{update_index}/valid",got["valid"].numpy())
                    timings.append(time.monotonic()-begin)
                    devices.append(got["loss"].device);losses.append(float(got["loss"].numpy()))
                    prefix=label+f"/update{update_index}"
                    compare(prefix+"/loss",got["loss"],update["loss"],dtype)
                    compare(prefix+"/gradient_norm",got["gradient_norm"],update["gradient_norm"],dtype)
                    keys=["particles","particle_weights","bandwidths",
                          "resampling_particle_weights","resampling_bandwidths"]
                    for idx,key in enumerate(keys):
                        fallback={"resampling_particle_weights":"particle_weights",
                                  "resampling_bandwidths":"bandwidths"}.get(key,key)
                        expected=[row.get(key,row.get(fallback)) for row in capture["outputs"]]
                        if key == "particles":
                            marker=len(checks)
                            compare(prefix+"/legacy_particles",got["outputs"][idx],expected,dtype)
                            legacy_particle_checks.extend(checks[marker:])
                            del checks[marker:]
                            compare(prefix+"/outputs/positions",got["outputs"][idx][...,:2],tensor(expected)[...,:2],dtype)
                        else:
                            compare(prefix+"/outputs/"+key,got["outputs"][idx],expected,dtype)
                    source_raw=tensor(capture["proposal_before_normalization"])
                    target_raw=got["outputs"][5]
                    compare(prefix+"/raw_proposals",target_raw,source_raw,dtype)
                    # Post-run float64 geometric diagnostic only; no runtime change.
                    with tf.device("/CPU:0"):
                        a=tf.cast(source_raw[...,2:],tf.float64)
                        b=tf.cast(target_raw[...,2:],tf.float64)
                        tau=128*2**-23 if dtype==tf.float32 else 1e-9
                        coordinate_bound=tau*(1+tf.abs(a))
                        radius=tf.sqrt(tf.reduce_sum(tf.square(a),-1))
                        error_radius=tf.sqrt(tf.reduce_sum(tf.square(coordinate_bound),-1))
                        separated=error_radius < radius
                        safe_radius=tf.where(radius>0,radius,tf.ones_like(radius))
                        ratio=error_radius/safe_radius
                        bound=tf.asin(tf.minimum(ratio,tf.ones_like(ratio)))+2*tau*(1+math.pi)
                        source_angle=tf.cast(tensor([row["particles"] for row in capture["outputs"]])[...,2],tf.float64)
                        target_angle=tf.cast(got["outputs"][0][...,2],tf.float64)
                        source_direct=tf.atan2(a[...,0],a[...,1])
                        target_direct=tf.atan2(b[...,0],b[...,1])
                        circular=lambda delta:tf.abs(tf.atan2(tf.sin(delta),tf.cos(delta)))
                        difference=circular(target_angle-source_angle)
                        source_local=circular(source_angle-source_direct)
                        target_local=circular(target_angle-target_direct)
                        finite=tf.reduce_all(tf.math.is_finite(a)) & tf.reduce_all(tf.math.is_finite(b))
                        finite=finite & tf.reduce_all(tf.math.is_finite(difference))
                        boolean(prefix+"/angle/finite",finite.numpy())
                        boolean(prefix+"/angle/origin_separated",tf.reduce_all(separated).numpy())
                        boolean(prefix+"/angle/source_local_map",tf.reduce_all(source_local<=tau*(1+math.pi)).numpy())
                        boolean(prefix+"/angle/tf_local_map",tf.reduce_all(target_local<=tau*(1+math.pi)).numpy())
                        boolean(prefix+"/angle/geometric_bound",tf.reduce_all(difference<=bound).numpy())
                        # A positive finite angle error just beyond the bound must fail.
                        eligible=separated & (bound < math.pi/4)
                        injected=circular(2*bound)
                        boolean(prefix+"/angle/negative_control_present",tf.reduce_any(eligible).numpy())
                        boolean(prefix+"/angle/negative_control_rejected",tf.reduce_all(tf.boolean_mask(injected>bound,eligible)).numpy())
                        angle_checks.append(dict(case=prefix,
                            minimum_radius=float(tf.reduce_min(radius).numpy()),
                            maximum_error_radius_ratio=float(tf.reduce_max(ratio).numpy()),
                            maximum_angle_error=float(tf.reduce_max(difference).numpy()),
                            maximum_scaled_angle_error=float(tf.reduce_max(difference/bound).numpy()),
                            maximum_source_local_error=float(tf.reduce_max(source_local).numpy()),
                            maximum_tf_local_error=float(tf.reduce_max(target_local).numpy()),
                            maximum_bound=float(tf.reduce_max(bound).numpy()),
                            all_origins_separated=bool(tf.reduce_all(separated).numpy())))

                    # Pure observability of the source angle transformation.
                    expected_particles=tensor([row["particles"] for row in capture["outputs"]])
                    scaled=tf.abs(got["outputs"][0]-expected_particles)/(tf.cast(128*2**-23 if dtype==tf.float32 else 1e-9,dtype)*(1+tf.abs(expected_particles)))
                    flat=int(tf.argmax(tf.reshape(scaled,[-1])).numpy())
                    coordinates=[]
                    for extent in reversed(scaled.shape.as_list()):
                        flat,index=divmod(flat,extent)
                        coordinates.insert(0,index)
                    step_index,batch_index,particle_index,dimension=coordinates
                    source_raw=tensor(capture["proposal_before_normalization"])
                    target_raw=got["outputs"][5]
                    a=source_raw[step_index,batch_index,particle_index,2:]
                    delta=target_raw[step_index,batch_index,particle_index,2:]-a
                    squared_norm=tf.reduce_sum(tf.square(a))
                    first_order=(a[1]*delta[0]-a[0]*delta[1])/squared_norm
                    conditioning_diagnostics.append(dict(case=prefix,worst_scaled_coordinate=coordinates,
                        dimension=("x","y","angle")[dimension],
                        source_particle=expected_particles[step_index,batch_index,particle_index,:].numpy().tolist(),
                        tf_particle=got["outputs"][0][step_index,batch_index,particle_index,:].numpy().tolist(),
                        source_raw_angle_vector=a.numpy().tolist(),tf_raw_angle_vector=(a+delta).numpy().tolist(),
                        vector_norm=float(tf.sqrt(squared_norm).numpy()),
                        angle_condition=float((1/tf.sqrt(squared_norm)).numpy()),
                        first_order_angle_difference=float(first_order.numpy()),
                        observed_angle_difference=float((got["outputs"][0][step_index,batch_index,particle_index,2]-expected_particles[step_index,batch_index,particle_index,2]).numpy()),
                        max_raw_vector_error=float(tf.reduce_max(tf.abs(target_raw-source_raw)).numpy()),
                        minimum_source_vector_norm=float(tf.reduce_min(tf.sqrt(tf.reduce_sum(tf.square(source_raw[...,2:]),-1))).numpy()),
                        role="EXPLANATORY_ONLY_NOT_NEW_TOLERANCE"))
                    for idx,name in enumerate(model.names):
                        for field in ("raw_gradients","clipped_gradients","parameters_after"):
                            compare(prefix+"/"+field+"/"+name,got[field][idx],update[field][name],dtype)
                        for field,sourcekey in (("first_moments","exp_avg"),("second_moments","exp_avg_sq")):
                            compare(prefix+"/"+field+"/"+name,got[field][idx],update["adam_after"][name][sourcekey],dtype)
                        boolean(prefix+"/adam_step/"+name,
                                int(got["step"].numpy())==update["adam_after"][name]["step"])
                    boolean(prefix+"/gpu_placement","GPU:0" in got["loss"].device)
                used=set(model.used_parameter_names)
                boolean(label+"/trainable_call_chain",used==set(case["trainable_names"]))
                for name in case["frozen_names"]:
                    compare(label+"/frozen/"+name,model.parameters[name],case["parameters"][name],dtype)
                boolean(label+"/one_trace",model.update.experimental_get_tracing_count()==1)
                selected=[x for x in checks if x["name"].startswith(label+"/")]
                case_results.append(dict(case=label,checks=len(selected),
                    failed_checks=[x for x in selected if not x["pass_check"]],
                    losses=losses,trace_count=model.update.experimental_get_tracing_count(),
                    call_seconds=timings,wall_seconds=time.monotonic()-tick,
                    used_trainable_names=sorted(used),device=devices))
        failed=[x for x in checks if not x["pass_check"]]
        result=dict(schema="kdm_neural_tf_comparison_v1",checks=checks,cases=case_results,
            conditioning_diagnostics=conditioning_diagnostics,
            angle_checks=angle_checks,legacy_particle_checks=legacy_particle_checks,
            checks_total=len(checks),checks_passed=len(checks)-len(failed),failed_checks=failed,
            memory_policy=memory,allocator=tf.config.experimental.get_memory_info("GPU:0"),
            physical_devices=[dict(name=d.name,details=tf.config.experimental.get_device_details(d))
                              for d in tf.config.list_physical_devices("GPU")],
            tensorflow=tf.__version__,jit_compile=True,tf32=False,replication="INCOMPLETE",
            historical_cost=reference["historical_cost"])
        write(out/"comparison.json",result)
        decision=dict(status="PASS",scope="CONDITIONAL_NEURAL_ANGLE_AND_ACTUAL_OPTIMIZER_SCREEN",
            replication="INCOMPLETE",numerical_screen_passed=not failed,
            dtype_screen_passed={dtype:not any("/"+dtype+"/" in x["name"] for x in failed)
                                 for dtype in ("float32","float64")},
            scientific_verdict="CONDITIONAL_ANGLE_SCREEN_PASSED" if not failed else "CONDITIONING_OR_DOWNSTREAM_MISMATCH",
            comparison_verdict="ALL_REPLAY_CHECKS_PASS" if not failed else "FAILED_DIAGNOSTIC_CHECKS")
        write(out/"decision.json",decision)
        print(json.dumps(dict(output=str(out),checks=len(checks),failed=len(failed),decision=decision)))
        return 0
    except Exception:
        (out/"error.txt").write_text(traceback.format_exc())
        write(out/"decision.json",dict(status="ERROR",replication="INCOMPLETE",
                                      scientific_verdict="HARNESS_OR_EXECUTION_FAILURE"))
        print(traceback.format_exc())
        return 1
    finally:
        if gpu_start is not None:manifest["gpu_wall_seconds"]=time.monotonic()-gpu_start
        manifest["wall_seconds"]=time.monotonic()-started
        manifest["outputs"]={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                             for p in out.iterdir() if p.is_file() and p.name not in ("manifest.json","command.log")}
        write(out/"manifest.json",manifest)

if __name__=="__main__":
    raise SystemExit(main())
