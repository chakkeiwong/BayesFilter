"""Diagnostic forward interventions on saved KDM angle inputs; never a runtime path."""
import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["BAYESFILTER_PRELOAD_CUSTOM_OP"] = "0"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["TF_NUM_INTEROP_THREADS"] = "1"
os.environ["TF_NUM_INTRAOP_THREADS"] = "1"
import argparse, ast, gzip, hashlib, inspect, json, subprocess, sys, textwrap, time
from pathlib import Path
from types import MethodType
ROOT = Path("/home/chakwong/BayesFilter")
sys.path.insert(0, str(ROOT))
PHASE = ROOT/"docs/plans/artifacts/controlled-smc-kdm-replication-20260928/gradient-followup-20261013"
def read(path):
    with gzip.open(path, "rt") as f:
        return json.load(f)
def save(path, value):
    with gzip.open(path, "wt", compresslevel=1) as f:
        json.dump(value, f, allow_nan=False)
def write(path, value):
    path.write_text(json.dumps(value, allow_nan=False)+"\n")
def main(out):
    started = time.monotonic()
    out.mkdir(exist_ok=False)
    import tensorflow as tf
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
    memory = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    tf.config.experimental.enable_tensor_float_32_execution(False)
    tf.config.set_soft_device_placement(False)
    from bayesfilter.replication.kdm_neural_replay_tf import BearingsNeuralReplay
    from bayesfilter.replication.kdm_source_stages_tf import SourceFullTraining
    from docs.benchmarks.diagnose_kdm_crossed_starts_tf import make_observed_coupled_sampler
    def host(x):
        if tf.is_tensor(x): return x.numpy().tolist()
        if isinstance(x, dict): return {k:host(v) for k,v in x.items()}
        if isinstance(x, (tuple,list)): return [host(v) for v in x]
        return x
    def vector(grads):
        return tf.concat([tf.reshape(tf.constant(x,tf.float64),[-1]) for x in grads.values()],0)
    def distance(actual, expected):
        assert set(actual)==set(expected)
        a,b=vector(actual),vector({k:expected[k] for k in actual})
        delta=a-b
        assert bool(tf.reduce_all(tf.math.is_finite(delta)))
        return dict(max_abs=float(tf.reduce_max(tf.abs(delta))),
            max_scaled=float(tf.reduce_max(tf.abs(delta)/(2e-5*(1+tf.abs(b))))),
            l2=float(tf.linalg.norm(delta)))
    @tf.custom_gradient
    def intervene(value, saved, mask):
        # Artificial hybrid: change the forward value, preserve the original
        # local derivative through the TF proposal. This is not a new algorithm.
        def grad(upstream):
            return upstream, tf.zeros_like(saved), None
        return tf.where(mask, saved, value), grad
    def instrument(model, source_raw):
        tree=ast.parse(textwrap.dedent(inspect.getsource(BearingsNeuralReplay.transition)))
        node=tree.body[0]
        node.args.args.append(ast.arg(arg="diagnostic_index"))
        body=[];count=0
        for statement in node.body:
            if isinstance(statement,ast.Assign) and any(isinstance(x,ast.Name) and x.id=="proposal_before_normalization" for x in statement.targets):
                body.extend(ast.parse("internal = self.intervene(internal, self.source_raw[diagnostic_index], self.intervention_mask[diagnostic_index])").body)
                count+=1
            body.append(statement)
        assert count==1
        node.body=body
        seq=ast.parse(textwrap.dedent(inspect.getsource(BearingsNeuralReplay._sequence)))
        calls=0
        for n in ast.walk(seq):
            if isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr=="transition":
                n.keywords.append(ast.keyword(arg="diagnostic_index",value=ast.parse("i-1",mode="eval").body))
                calls+=1
        assert calls==1
        ns=dict(tf=tf)
        exec(compile(ast.fix_missing_locations(tree),__file__,"exec"),ns)
        exec(compile(ast.fix_missing_locations(seq),__file__,"exec"),ns)
        model.transition=MethodType(ns["transition"],model)
        model._sequence=MethodType(ns["_sequence"],model)
        model.intervene=intervene
        model.source_raw=tf.constant(source_raw,tf.float32)
        model.intervention_mask=tf.Variable(tf.zeros([16,64,25,4],tf.bool),trainable=False)
    requests={};summary={};all_records={};hashes={};traces={}
    angle_path=PHASE/"angle01/diagnostic/author-adjoints.json.gz"
    hashes[str(angle_path.relative_to(ROOT))]=hashlib.sha256(angle_path.read_bytes()).hexdigest()
    angle=read(angle_path)
    for role in ("starting","best"):
        path=PHASE/"trained02/diagnostic"/(role+"-gradient1.json.gz")
        hashes[str(path.relative_to(ROOT))]=hashlib.sha256(path.read_bytes()).hexdigest()
        case=read(path);source=case["source"];state=source["before"]
        fixture=dict(variant="mdpf",dtype="float32",batch_size=64,parameters=state["parameters"],
            trainable_names=state["trainable_names"],learning_rates=state["rates"],constants=state["constants"])
        model=BearingsNeuralReplay(fixture,capture_sampling_diagnostics=True)
        model.load_state_dict(state)
        model.sampler._sample=make_observed_coupled_sampler(tf,model.sampler)
        instrument(model,source["captured"]["raw"])
        stage=SourceFullTraining(model)
        def diagnostic(*args):
            with tf.GradientTape() as tape:
                loss,valid,outputs=stage._loss(*args)
            return loss,valid,outputs,tape.gradient(loss,model.trainable)
        fn=tf.function(diagnostic,input_signature=[
            tf.TensorSpec([64,17,3],tf.float32),tf.TensorSpec([64,17,1],tf.float32),
            tf.TensorSpec([64,17],tf.bool),tf.TensorSpec([2],tf.int64)],jit_compile=True,autograph=False)
        args=[tf.constant(source["data"][k],tf.bool if k=="ground_truth_mask" else tf.float32)
            for k in ("states","observations","ground_truth_mask")]
        def run(mask):
            model.intervention_mask.assign(mask)
            loss,valid,o,grads=fn(*args,tf.constant([202610122,1],tf.int64))
            assert bool(valid) and "GPU:0" in loss.device
            return dict(loss=float(loss),gradients=dict(zip(model.names,host(grads))),raw=host(o[5]),
                samples=host(o[6][1:]),ancestors=host(tf.cast(o[9][1:],tf.int32))),o
        zero=tf.zeros([16,64,25,4],tf.bool)
        baseline,outputs=run(zero)
        saved=case["tf_observed_same_source_state"]
        controls=dict(loss=baseline["loss"]==saved["loss"],gradients=baseline["gradients"]==saved["gradients"],
            **{k:baseline[k]==saved["captured"][k] for k in ("raw","samples","ancestors")})
        assert all(controls.values()),(role,"intervention harness altered unchanged arm",controls)
        location=angle["results"][role]["location"]
        pair_mask=tf.tensor_scatter_nd_update(zero,[location+[2],location+[3]],[True,True])
        all_mask=tf.broadcast_to(tf.constant([False,False,True,True]),[16,64,25,4])
        records={"unchanged":baseline}
        native_distance=distance(baseline["gradients"],source["gradients"])
        comparisons={}
        for name,mask in (("selected_pair",pair_mask),("all_angle_pairs",all_mask)):
            record,o=run(mask);records[name]=record
            d=distance(record["gradients"],source["gradients"])
            d["residual_l2_fraction"]=d["l2"]/native_distance["l2"]
            changed=int(tf.reduce_sum(tf.cast(tf.not_equal(record["ancestors"],source["captured"]["ancestors"]),tf.int32)))
            # Exact saved-value replacement is required on every chosen element.
            assert bool(tf.reduce_all(tf.boolean_mask(tf.constant(record["raw"]),mask)==tf.boolean_mask(model.source_raw,mask)))
            comparisons[name]=dict(gradients=d,loss_difference=record["loss"]-source["loss"],
                ancestor_differences=changed,localization_flag_removed=d["max_scaled"]<=1 and changed==0)
        a=tf.constant(baseline["raw"]);b=model.source_raw
        epsilon=float(tf.gather_nd(a-b,[location+[2]])[0])
        assert float(tf.gather_nd(a-b,[location+[3]])[0])==0.
        requests[role]=dict(case=str(path),noise=host(outputs[7]),components=[2],epsilons=[0.,epsilon])
        summary[role]=dict(controls=controls,location=location,observed_raw_difference=epsilon,
            native_gradients=native_distance,interventions=comparisons)
        all_records[role]=records
        traces[role]=fn.experimental_get_tracing_count()
        save(out/(role+"-tf-interventions.json.gz"),records)
        print(json.dumps(dict(role=role,summary=summary[role],traces=traces[role])),flush=True)
    write(out/"author-input.json",requests)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES="-1",LD_LIBRARY_PATH="/home/chakwong/anaconda3/envs/tf-gpu2/lib")
    with (out/"author.log").open("w") as log:
        subprocess.run(["/home/chakwong/anaconda3/envs/tf-gpu2/bin/python","-I","-B",
            str(ROOT/"docs/benchmarks/reference_kdm_angle_adjoint.py"),"--input",str(out/"author-input.json"),
            "--output",str(out/"author-interventions.json.gz")],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
    author=read(out/"author-interventions.json.gz")
    for role,s in author["results"].items():
        assert all(s["controls"].values()) and len(s["perturbations"])==2
        zero,p=s["perturbations"]
        assert zero["gradient_max_difference"]==0.
        assert p["perturbation"]["realized"]==summary[role]["observed_raw_difference"]
        native=all_records[role]["unchanged"]["gradients"]
        residual=distance(native,p["gradients"])
        residual["residual_l2_fraction"]=residual["l2"]/summary[role]["native_gradients"]["l2"]
        summary[role]["reverse_source_intervention"]=dict(perturbation=p["perturbation"],
            gradient_change=p["gradient_max_difference"],residual_to_native_tf=residual,
            loss_difference=p["loss_difference"],source_controls=s["controls"],zero_control=True)
    assert all(x==1 for x in traces.values()),traces
    write(out/"environment.json",dict(tf=tf.__version__,gpu_devices=[d.name for d in tf.config.list_logical_devices("GPU")],
        memory_growth=memory,tf32=False,jit_compile=True,gpu_trust_basis="escalated_local_gpu_access",
        author=author["provenance"],saved_case_hashes=hashes))
    write(out/"result.json",dict(status="COMPLETED",replication="INCOMPLETE",
        scope="saved-case artificial forward interventions; explanatory only",summaries=summary,trace_counts=traces,
        allocator=tf.config.experimental.get_memory_info("GPU:0"),wall_seconds=time.monotonic()-started,
        not_concluded=["trained quality equivalence","population numerical sensitivity envelope","new algorithm or numerical default"]))
if __name__=="__main__":
    p=argparse.ArgumentParser();p.add_argument("--output",type=Path,required=True)
    main(p.parse_args().output)
