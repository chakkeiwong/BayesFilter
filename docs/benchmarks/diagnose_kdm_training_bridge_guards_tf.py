"""Focused GPU diagnostic: rejected sampler exhaustion and fresh consumer restore."""
import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]="true"
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))


def main():
    p=argparse.ArgumentParser();p.add_argument("--output",required=True,type=Path)
    args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    import tensorflow as tf
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
    memory=configure_tensorflow_gpu_memory_growth(tf,require_gpu=True)
    tf.config.experimental.enable_tensor_float_32_execution(False)
    tf.config.set_soft_device_placement(False)
    from bayesfilter.replication.kdm_neural_replay_tf import BearingsNeuralReplay
    base=ROOT/"docs/plans/artifacts/controlled-smc-kdm-replication-20260928"
    reference=base/"kdm-neural-batch64-20261006-02/source/author-batch64.json"
    saved=base/"kdm-training-bridge-20261007-03/diagnostic/am_dpf-float32-checkpoint.json"
    case=next(c for c in json.loads(reference.read_text())["cases"]
              if c["variant"]=="am_dpf" and c["dtype"]=="float32")
    checkpoint=json.loads(saved.read_text())
    checks=[]
    def check(name,passed):
        checks.append(dict(name=name,passed=bool(passed)))
    with tf.device("/GPU:0"):
        u=case["updates"][0];initial=u["captured"]["initial"];data=u["data"]
        t=lambda x:tf.convert_to_tensor(x,tf.float32)
        inputs=[t(initial["particles_downscaled"]),t(initial["particle_weights"]),
                t(initial["bandwidths_downscaled"]),
                t(initial.get("resampling_particle_weights",initial["particle_weights"])),
                t(initial.get("resampling_bandwidths_downscaled",initial["bandwidths_downscaled"])),
                t(data["observations"]),t(data["states"]),tf.convert_to_tensor(data["ground_truth_mask"],tf.bool)]
        # Deliberately lower the reject-only limit before tracing. Healthy model
        # parameters and inputs stay identical; do not use this limit in training.
        capped=BearingsNeuralReplay(case);capped.sampler.limit=1
        before=capped.state_dict()
        result=capped.native_update(*inputs,tf.constant([731,5],tf.int64))
        check("cap_exhaustion_rejected",not result["valid"].numpy() and not result["committed"].numpy())
        check("cap_exhaustion_state_unchanged",before==capped.state_dict())
        check("guard_has_finite_loss",tf.math.is_finite(result["loss"]).numpy())
        check("guard_has_finite_gradients",all(tf.reduce_all(tf.math.is_finite(g)).numpy() for g in result["raw_gradients"]))
        check("configured_limit_was_exercised",tf.reduce_max(result["outputs"][8]).numpy()==1)
        first=BearingsNeuralReplay(case);second=BearingsNeuralReplay(case)
        for model in (first,second):
            model.load_state_dict(checkpoint["model"])
        check("fresh_instances_restore_complete",first.state_dict()==second.state_dict()==checkpoint["model"])
        a=first.native_update(*inputs,tf.constant(checkpoint["next_seed"],tf.int64))
        b=second.native_update(*inputs,tf.constant(checkpoint["next_seed"],tf.int64))
        check("fresh_instances_valid",a["valid"].numpy() and b["valid"].numpy())
        check("fresh_instances_resume_exact",first.state_dict()==second.state_dict())
        before=first.state_dict()
        broken=json.loads(json.dumps(checkpoint["model"]))
        del broken["second_moments"][first.names[0]]
        try:
            first.load_state_dict(broken)
        except ValueError:
            rejected=True
        else:
            rejected=False
        check("incomplete_checkpoint_rejected_atomically",rejected and first.state_dict()==before)
    output=dict(schema="kdm_training_bridge_guard_diagnostic_v1",
        status="PASS" if all(c["passed"] for c in checks) else "FAIL",
        checks=checks,scope="REJECTION_AND_FRESH_RESTORE_DIAGNOSTIC",replication="INCOMPLETE",
        sampler_limit_exception="1 only for deliberately failing diagnostic; runtime default remains128",
        checkpoint_sha256=hashlib.sha256(saved.read_bytes()).hexdigest(),
        reference_sha256=hashlib.sha256(reference.read_bytes()).hexdigest(),
        memory_policy=memory,allocator=tf.config.experimental.get_memory_info("GPU:0"),
        tf32=False,jit_compile=True,seeds=[[731,5],checkpoint["next_seed"]],
        wall_seconds=time.monotonic()-started)
    (args.output/"result.json").write_text(json.dumps(output,indent=2,allow_nan=False)+"\n")
    print(json.dumps(dict(status=output["status"],checks=len(checks),
                         passed=sum(c["passed"] for c in checks),wall_seconds=output["wall_seconds"])))
    return 0 if output["status"]=="PASS" else 1


if __name__=="__main__":
    raise SystemExit(main())
