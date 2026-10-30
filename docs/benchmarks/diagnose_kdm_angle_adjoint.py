"""Saved-case source/TF angle-adjoint diagnostics; never a runtime implementation."""
import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"]="true"
os.environ["BAYESFILTER_PRELOAD_CUSTOM_OP"]="0"
os.environ["OMP_NUM_THREADS"]="1"
os.environ["TF_NUM_INTEROP_THREADS"]="1"
os.environ["TF_NUM_INTRAOP_THREADS"]="1"
import argparse, ast, copy, gzip, inspect, json, subprocess, sys, textwrap, time
from pathlib import Path
from types import MethodType
ROOT=Path("/home/chakwong/BayesFilter");sys.path.insert(0,str(ROOT))
BASE=ROOT/"docs/plans/artifacts/controlled-smc-kdm-replication-20260928/gradient-followup-20261013/trained02/diagnostic"
def read(p):
    with gzip.open(p,"rt") as f:return json.load(f)
def save(p,x):
    with gzip.open(p,"wt",compresslevel=1) as f:json.dump(x,f,allow_nan=False)
def write(p,x):p.write_text(json.dumps(x,allow_nan=False)+"\n")
def main(out):
    started=time.monotonic();out.mkdir(exist_ok=False)
    import tensorflow as tf
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
    memory=configure_tensorflow_gpu_memory_growth(tf,require_gpu=True)
    tf.config.experimental.enable_tensor_float_32_execution(False);tf.config.set_soft_device_placement(False)
    from bayesfilter.replication.kdm_neural_replay_tf import BearingsNeuralReplay
    from bayesfilter.replication.kdm_source_stages_tf import SourceFullTraining
    from docs.benchmarks.diagnose_kdm_crossed_starts_tf import make_observed_coupled_sampler
    def host(x):
        if tf.is_tensor(x):return x.numpy().tolist()
        if isinstance(x,dict):return {k:host(v) for k,v in x.items()}
        if isinstance(x,(tuple,list)):return [host(v) for v in x]
        return x
    # A transparent adjoint tap. Forward and derivative w.r.t. value are identity.
    # The dummy variable receives a duplicate adjoint for observation only; its
    # derivative is manufactured and is never an algorithmic score.
    @tf.custom_gradient
    def tap(value,dummy):
        def grad(upstream):return upstream,upstream
        return tf.identity(value),grad
    def instrument(model):
        tree=ast.parse(textwrap.dedent(inspect.getsource(BearingsNeuralReplay.transition)))
        node=tree.body[0];node.args.args.append(ast.arg(arg="diagnostic_index"))
        revised=[];counts=dict(raw=0,normal=0,angle=0)
        for statement in node.body:
            revised.append(statement)
            if not isinstance(statement,ast.Assign):continue
            names=[t.id for t in statement.targets if isinstance(t,ast.Name)]
            src=ast.unparse(statement)
            if names==["proposal_before_normalization"]:
                revised.extend(ast.parse("internal = self.tap(internal, self.raw_probe[diagnostic_index])").body);counts["raw"]+=1
            elif names==["internal"] and "magnitude" in src and "tf.concat" in src:
                revised.extend(ast.parse("internal = self.tap(internal, self.normal_probe[diagnostic_index])").body);counts["normal"]+=1
            elif names==["locations"]:
                revised.extend(ast.parse("locations = self.tap(locations, self.angle_probe[diagnostic_index])").body);counts["angle"]+=1
        assert counts==dict(raw=1,normal=1,angle=1),counts
        node.body=revised
        seq=ast.parse(textwrap.dedent(inspect.getsource(BearingsNeuralReplay._sequence)))
        calls=0
        for n in ast.walk(seq):
            if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=="transition":
                n.keywords.append(ast.keyword(arg="diagnostic_index",value=ast.parse("i-1",mode="eval").body));calls+=1
        assert calls==1
        namespace=dict(tf=tf);exec(compile(ast.fix_missing_locations(tree),__file__,"exec"),namespace)
        exec(compile(ast.fix_missing_locations(seq),__file__,"exec"),namespace)
        model.transition=MethodType(namespace["transition"],model);model._sequence=MethodType(namespace["_sequence"],model)
        model.tap=tap
        model.raw_probe=tf.Variable(tf.zeros([16,64,25,4],tf.float32))
        model.normal_probe=tf.Variable(tf.zeros([16,64,25,4],tf.float32))
        model.angle_probe=tf.Variable(tf.zeros([16,64,25,3],tf.float32))
        return counts
    records={};requests={};traces={}
    for role in ("starting","best"):
        path=BASE/(role+"-gradient1.json.gz");case=read(path);state=case["source"]["before"]
        fixture=dict(variant="mdpf",dtype="float32",batch_size=64,parameters=state["parameters"],
            trainable_names=state["trainable_names"],learning_rates=state["rates"],constants=state["constants"])
        model=BearingsNeuralReplay(fixture,capture_sampling_diagnostics=True);model.load_state_dict(state)
        model.sampler._sample=make_observed_coupled_sampler(tf,model.sampler)
        counts=instrument(model);stage=SourceFullTraining(model)
        def diagnostic(*args):
            with tf.GradientTape() as tape:
                loss,valid,outputs=stage._loss(*args)
            grads=tape.gradient(loss,model.trainable+[model.raw_probe,model.normal_probe,model.angle_probe])
            return loss,valid,outputs,grads[:len(model.names)],grads[-3:]
        fn=tf.function(diagnostic,input_signature=[
            tf.TensorSpec([64,17,3],tf.float32),tf.TensorSpec([64,17,1],tf.float32),
            tf.TensorSpec([64,17],tf.bool),tf.TensorSpec([2],tf.int64)],jit_compile=True,autograph=False)
        args=[tf.constant(case["source"]["data"][k],tf.bool if k=="ground_truth_mask" else tf.float32)
            for k in ("states","observations","ground_truth_mask")]
        loss,valid,o,grads,adj=fn(*args,tf.constant([202610122,1],tf.int64))
        assert bool(valid) and "GPU:0" in loss.device
        baseline=case["tf_observed_same_source_state"]
        controls=dict(loss=float(loss)==baseline["loss"],gradients=dict(zip(model.names,host(grads)))==baseline["gradients"],
            raw=host(o[5])==baseline["captured"]["raw"],samples=host(o[6][1:])==baseline["captured"]["samples"])
        assert all(controls.values()),(role,"adjoint tap changed native computation",controls)
        records[role]=dict(controls=controls,raw=host(o[5]),raw_adjoint=host(adj[0]),
            normal_adjoint=host(adj[1]),angle_adjoint=host(adj[2]),instrumentation_counts=counts)
        requests[role]=dict(case=str(path),noise=host(o[7]))
        traces[role]=fn.experimental_get_tracing_count()
        save(out/(role+"-tf-adjoints.json.gz"),records[role])
        print(json.dumps(dict(role=role,tf_controls=controls,trace_count=traces[role])),flush=True)
    write(out/"author-input.json",requests)
    with (out/"author.log").open("w") as log:
        subprocess.run(["/home/chakwong/anaconda3/envs/tf-gpu2/bin/python","-I","-B",
            str(ROOT/"docs/benchmarks/reference_kdm_angle_adjoint.py"),"--input",str(out/"author-input.json"),
            "--output",str(out/"author-adjoints.json.gz")],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
    source=read(out/"author-adjoints.json.gz")
    kernels={}
    for dtype in (tf.float32,tf.float64):
        def local(raw,normal_upstream,angle_upstream):
            with tf.GradientTape() as tape:
                tape.watch(raw)
                radius=tf.sqrt(tf.reduce_sum(raw*raw,axis=-1))
                normal=raw/radius[...,None]
                target=tf.reduce_sum(normal*normal_upstream)
            pulled=tape.gradient(target,raw)
            with tf.GradientTape() as tape:
                tape.watch(normal)
                theta=tf.atan2(normal[...,0],normal[...,1])
                angle_target=tf.reduce_sum(theta*angle_upstream)
            angle_pulled=tape.gradient(angle_target,normal)
            analytic=(normal_upstream-normal*tf.reduce_sum(normal*normal_upstream,axis=-1,keepdims=True))/radius[...,None]
            analytic_angle=tf.stack([normal[...,1],-normal[...,0]],axis=-1)*angle_upstream[...,None]/tf.reduce_sum(normal*normal,axis=-1,keepdims=True)
            return dict(radius=radius,normal=normal,theta=theta,pulled=pulled,angle_pulled=angle_pulled,
                analytic=analytic,analytic_angle=analytic_angle)
        kernels[dtype.name]=tf.function(local,input_signature=[
            tf.TensorSpec([16,64,25,2],dtype),tf.TensorSpec([16,64,25,2],dtype),
            tf.TensorSpec([16,64,25],dtype)],jit_compile=True,autograph=False)
    def diff(a,b):
        a=tf.cast(a,tf.float64);b=tf.cast(b,tf.float64);delta=tf.abs(a-b)
        return dict(max_abs=float(tf.reduce_max(delta)),rms=float(tf.sqrt(tf.reduce_mean(delta*delta))),
            max_scaled=float(tf.reduce_max(delta/(tf.constant(2e-5,tf.float64)*(1+tf.abs(b))))))
    summaries={}
    for role in ("starting","best"):
        s=source["results"][role];baseline=s["baseline"];actual=records[role]
        r=tf.constant(baseline["raw"],tf.float32)[...,2:];t=tf.constant(actual["raw"],tf.float32)[...,2:]
        ng=tf.constant(baseline["normal_adjoint"],tf.float32)[...,2:]
        tg=tf.constant(actual["normal_adjoint"],tf.float32)[...,2:]
        ag=tf.constant(baseline["angle_adjoint"],tf.float32)[...,2]
        source_raw_adjoint=tf.constant(baseline["raw_adjoint"],tf.float32)[...,2:]
        tf_raw_adjoint=tf.constant(actual["raw_adjoint"],tf.float32)[...,2:]
        ops={}
        for dtype in (tf.float32,tf.float64):
            fn=kernels[dtype.name]
            for name,x,g in (("source_inputs_source_adjoint",r,ng),("tf_inputs_source_adjoint",t,ng),("tf_inputs_tf_adjoint",t,tg)):
                ops[dtype.name+"/"+name]=fn(tf.cast(x,dtype),tf.cast(g,dtype),tf.cast(ag,dtype))
        base=ops["float32/source_inputs_source_adjoint"]
        own=ops["float32/tf_inputs_tf_adjoint"]
        checks=dict(source_capture=s["controls"],tf_capture=actual["controls"],
            source_local_pullback=diff(base["pulled"],source_raw_adjoint),
            tf_local_pullback=diff(own["pulled"],tf_raw_adjoint),
            analytic_fp64=diff(ops["float64/source_inputs_source_adjoint"]["pulled"],ops["float64/source_inputs_source_adjoint"]["analytic"]),
            source_tf_raw=diff(t,r),source_tf_raw_adjoint=diff(tf_raw_adjoint,source_raw_adjoint),
            upstream_normal_adjoint=diff(tg,ng),
            raw_input_only_effect=diff(ops["float64/tf_inputs_source_adjoint"]["pulled"],ops["float64/source_inputs_source_adjoint"]["pulled"]),
            source_fp32_vs_promoted_fp64=diff(base["pulled"],ops["float64/source_inputs_source_adjoint"]["pulled"]))
        per_time=[]
        for i in range(16):
            per_time.append(dict(time=i+1,radius_min=float(tf.reduce_min(base["radius"][i])),
                raw_difference=diff(t[i],r[i]),raw_adjoint_difference=diff(tf_raw_adjoint[i],source_raw_adjoint[i]),
                normal_upstream_difference=diff(tg[i],ng[i])))
        panel=[{k:v for k,v in p.items() if k!="gradients"} for p in s["perturbations"]]
        summaries[role]=dict(checks=checks,per_time=per_time,perturbations=panel,location=s["location"],minimum_radius=s["minimum_radius"])
        save(out/(role+"-local-derivatives.json.gz"),host(ops))
    environment=dict(tf=tf.__version__,gpu_devices=[x.name for x in tf.config.list_logical_devices("GPU")],
        memory_growth=memory,tf32=False,jit_compile=True,gpu_trust_basis="escalated_local_gpu_access",
        author=source["provenance"])
    write(out/"environment.json",environment)
    result=dict(status="COMPLETED",replication="INCOMPLETE",scope="saved-case exact native adjoints and conditional source perturbations",
        summaries=summaries,trace_counts=traces,local_kernel_trace_counts={k:v.experimental_get_tracing_count() for k,v in kernels.items()},
        allocator=tf.config.experimental.get_memory_info("GPU:0"),wall_seconds=time.monotonic()-started,
        not_concluded=["complete trained-quality equivalence","native RNG perturbation envelope","population ranking"])
    assert all(x==1 for x in result["trace_counts"].values()) and all(x==1 for x in result["local_kernel_trace_counts"].values())
    write(out/"result.json",result)
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True);a=p.parse_args();main(a.output)
