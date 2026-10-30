"""GPU/XLA autonomous-update diagnostic using current runtime consumer endpoints."""
import argparse, hashlib, json, os, sys, time
from pathlib import Path
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]="true"
ROOT=Path("/home/chakwong/BayesFilter");sys.path.insert(0,str(ROOT))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,allow_nan=False)+"\n")
def main(out,reference):
    out.mkdir(exist_ok=False)
    source=json.loads((reference/"result.json").read_text())
    assert source["status"]=="PASS" and source["native_source_sampling"]
    assert sha(reference/"inputs.json")==source["inputs_sha256"]
    inputs=json.loads((reference/"inputs.json").read_text())
    import tensorflow as tf
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
    memory=configure_tensorflow_gpu_memory_growth(tf,require_gpu=True)
    tf.config.experimental.enable_tensor_float_32_execution(False);tf.config.set_soft_device_placement(False)
    from bayesfilter.replication.kdm_neural_replay_tf import BearingsNeuralReplay
    from bayesfilter.replication.kdm_source_stages_tf import SourceFullTraining
    records=[]
    with tf.device("/GPU:0"):
        data={split:{k:tf.constant(v,tf.bool if k=="ground_truth_mask" else tf.float32) for k,v in values.items()}
              for split,values in inputs["datasets"].items()}
        for vi,case in enumerate(inputs["cases"]):
            tick=time.monotonic();model=BearingsNeuralReplay(case,batch_size=64);view=SourceFullTraining(model)
            def batch(split,i):return [data[split][k][i*64:(i+1)*64] for k in ("states","observations","ground_truth_mask")]
            def evaluate():
                values=[]
                for i in range(4):
                    got=view.evaluate(*batch("validation",i),tf.constant([inputs["seed"]+300+vi,i],tf.int64))
                    assert bool(got["valid"].numpy()) and "GPU" in got["loss"].device
                    values.append(float(got["loss"].numpy()))
                return values
            before=evaluate();losses=[];norms=[]
            for i in range(64):
                got=view.update(*batch("training",i),tf.constant([inputs["seed"]+200+vi,i],tf.int64))
                assert bool(got["valid"].numpy()) and bool(got["committed"].numpy())
                assert int(got["step"].numpy())==i+1 and "GPU" in got["loss"].device
                losses.append(float(got["loss"].numpy()));norms.append(float(got["gradient_norm"].numpy()))
                if (i+1)%16==0:print(json.dumps(dict(variant=case["variant"],completed_updates=i+1,loss=losses[-1])),flush=True)
            after=evaluate()
            assert all(bool(tf.reduce_all(model.parameters[n]==tf.constant(case["parameters"][n],tf.float32)).numpy()) for n in case["frozen_names"])
            changed=sum(not bool(tf.reduce_all(model.parameters[n]==tf.constant(case["parameters"][n],tf.float32)).numpy()) for n in model.names)
            assert changed>0 and set(model.names).issubset(model.used_parameter_names)
            assert view.update.experimental_get_tracing_count()==view.evaluate.experimental_get_tracing_count()==1
            checkpoint=out/(case["variant"]+"-final.json");write(checkpoint,model.state_dict())
            records.append(dict(variant=case["variant"],updates=64,losses=losses,gradient_norms=norms,
                validation_before=before,validation_after=after,changed_parameters=changed,frozen_parameters_unchanged=True,
                update_trace_count=1,evaluate_trace_count=1,checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),
                wall_seconds=time.monotonic()-tick))
    comparisons=[]
    for actual,author in zip(records,source["cases"]):
        assert actual["variant"]==author["variant"]
        mean=lambda x:sum(x)/len(x)
        comparisons.append(dict(variant=actual["variant"],tf_validation_before=mean(actual["validation_before"]),
            tf_validation_after=mean(actual["validation_after"]),source_validation_before=mean(author["validation_before"]),
            source_validation_after=mean(author["validation_after"]),role="DESCRIPTIVE_ONLY_SINGLE_PAIRED_START_INDEPENDENT_RANDOM_STREAMS"))
    write(out/"result.json",dict(status="PASS",scope="NATIVE_TRAINING_SMOKE_NOT_QUALIFICATION",cases=records,comparisons=comparisons,
        source_result_sha256=sha(reference/"result.json"),inputs_sha256=source["inputs_sha256"],
        tensorflow_version=tf.__version__,memory_policy=memory,jit_compile=True,tf32=False,
        native_sampling=True,realized_source_particle_replay=False,call_chain="SourceFullTraining.update -> _loss -> _native_sequence -> SourceKDMResampler._sample -> _apply_update",
        random_backend="TF stateless Philox; independent source streams, paired data/parameters",
        acceptance="64 committed finite updates per variant; frozen parameters unchanged; one trace per fixed shape",
        replication="INCOMPLETE"))
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True);p.add_argument("--reference",type=Path,required=True)
    a=p.parse_args();main(a.output,a.reference)
