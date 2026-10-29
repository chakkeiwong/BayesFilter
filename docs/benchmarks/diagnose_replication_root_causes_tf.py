"""GPU/XLA root-cause diagnostics, never a training or qualification consumer."""
from __future__ import annotations
import argparse, ast, inspect, json, os, sys
from pathlib import Path
os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH","true")
ROOT=Path("/home/chakwong/BayesFilter")
BASE=ROOT/"docs/plans/artifacts/controlled-smc-kdm-replication-20260928"
PRIOR=BASE/"gap-closure-20261010/localization-01"
sys.path.insert(0,str(ROOT))
def write(path,value):
    path.write_text(json.dumps(value,indent=2,allow_nan=False)+"\n")
def main(out,reference):
    import tensorflow as tf
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
    memory=configure_tensorflow_gpu_memory_growth(tf,require_gpu=True)
    tf.config.experimental.enable_tensor_float_32_execution(False)
    from bayesfilter.replication.kdm_neural_replay_tf import BearingsNeuralReplay
    from bayesfilter.replication.multivariate_controlled_smc_tf import MultivariateGaussianControlledSMC,quadratic
    from scipy.io import loadmat  # diagnostic-only MATLAB reference loading
    out.mkdir(exist_ok=False)
    def compare(actual,expected,dtype):
        actual=tf.convert_to_tensor(actual,dtype);expected=tf.convert_to_tensor(expected,dtype)
        assert actual.shape==expected.shape,(actual.shape,expected.shape)
        tolerance=128*2**-23 if dtype==tf.float32 else 1e-9
        error=tf.abs(actual-expected);scaled=error/(tf.constant(tolerance,dtype)*(1+tf.abs(expected)))
        finite=bool(tf.reduce_all(tf.math.is_finite(actual)).numpy())
        return dict(finite=finite,max_absolute_error=float(tf.reduce_max(error).numpy()) if finite else None,
            max_scaled_error=float(tf.reduce_max(scaled).numpy()) if finite else None,
            passed=finite and bool(tf.reduce_all(scaled<=1).numpy()))
    from bayesfilter.replication.kdm_source_stages_tf import SourceFullTraining,SourceWeightPretraining
    import textwrap
    wiring=[]
    for owner,name in [(BearingsNeuralReplay,'_update'),(BearingsNeuralReplay,'_native_update'),
                       (SourceFullTraining,'_update'),(SourceWeightPretraining,'_update'),
                       (SourceWeightPretraining,'_native_update')]:
        function=getattr(owner,name)
        tree=ast.parse(textwrap.dedent(inspect.getsource(function)))
        calls=[node.func.attr for node in ast.walk(tree) if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute)]
        assert '_apply_update' in calls,(owner.__name__,name,calls)
        wiring.append(dict(consumer=owner.__name__+'.'+name,optimizer='_apply_update',direct_call_checked=True))
    traces={};devices=[];checkpoint_checks=[]
    with tf.device("/GPU:0"):
        covariance=lambda v:[[v if i==j else 0. for j in range(8)] for i in range(8)]
        engines={s:MultivariateGaussianControlledSMC(observations=101,particles=512,replicas=1,
            dimension=8,observation_dimension=6,initial_mean=[0.]*8,initial_covariance=covariance(.01),
            transition_covariance=covariance(.001),transition_mean=lambda x:x,
            observation_log_density=lambda y,x:tf.reduce_sum(x[...,:6]*0.,-1),
            solver=s,jit_compile=True,precision_policy="none",first_failure_guards=False)
            for s in ("source_normal_equations","centered_qr")}
        rows=loadmat(BASE/"gap-closure-20261010/c0-source-fit-20261010/fit-trace.mat",simplify_cells=True)["FIT_TRACE"]
        fits=[]
        for row in rows:
            x=tf.constant(row["design"][:,1:9][None],tf.float64)
            target=tf.constant(row["target"][None],tf.float64)
            expected=tf.constant((row["design"]@row["beta"])[None],tf.float64)
            for solver,engine in engines.items():
                fit=engine.regress(x,target);prediction=quadratic(x,fit["A"],fit["b"],fit["c"])
                devices.append(fit["A"].device)
                fits.append(dict(time=int(row["time"]),branch=row["branch"],solver=solver,
                    valid=bool(fit["valid"].numpy()),prediction=compare(prediction,expected,tf.float64),
                    residual_l2=float(tf.sqrt(tf.reduce_sum((prediction-target)**2)).numpy())))
        for name,engine in engines.items():traces["regression/"+name]=engine.regress.experimental_get_tracing_count()
        write(out/"csmc-exact-input.json",dict(rows=fits,endpoint="regress -> _regress"))
        ref=json.loads(reference.read_text())
        prior=json.loads((PRIOR/"kdm/precision_observations.json").read_text())
        result=[]
        for case in ref["cases"]:
            model=BearingsNeuralReplay(case,batch_size=case["batch_size"]);dtype=model.dtype
            state=model.state_dict()
            model.set_learning_rates({n:r/3 for n,r in case["learning_rates"].items()})
            model.load_state_dict(state)
            assert model.state_dict()==state
            assert state["rates"]==case["learning_rates"]
            for bad in ({**state,"schema":"kdm_neural_training_state_v1"},
                        {**state,"adam_scalar_policy":"wrong"}):
                try:model.load_state_dict(bad)
                except ValueError:pass
                else:raise AssertionError("Mismatched optimizer checkpoint accepted")
                assert model.state_dict()==state
            view=model.for_batch(2)
            assert all(a is b for a,b in zip(view.rates,model.rates))
            checkpoint_checks.append(dict(variant=case["variant"],dtype=case["dtype"],
                exact_rate_roundtrip=True,legacy_rejected_without_mutation=True,shared_batch_rates=True))
            tensor=lambda x:tf.convert_to_tensor(x,dtype)
            def reset(j):
                p=case["parameters"] if j==0 else {**case["parameters"],**case["updates"][j-1]["parameters_after"]}
                for name,v in model.parameters.items():v.assign(tensor(p[name]))
                previous=None if j==0 else case["updates"][j-1]["adam_after"]
                for name,m,v in zip(model.names,model.moments1,model.moments2):
                    m.assign(tf.zeros_like(m) if j==0 else tensor(previous[name]["exp_avg"]))
                    v.assign(tf.zeros_like(v) if j==0 else tensor(previous[name]["exp_avg_sq"]))
                model.step.assign(tensor(j))
            specs=tuple(tf.TensorSpec(p.shape,dtype) for p in model.trainable)
            def isolated(*grads):
                def synthetic(*inputs):
                    return tf.add_n([tf.reduce_sum(p*g) for p,g in zip(model.trainable,inputs)]),tf.constant(True),()
                return model._apply_update(synthetic,grads,gradient_clip=None)
            isolated=tf.function(isolated,input_signature=specs,jit_compile=True,autograph=False)
            def source_scalars(*grads):
                t=tf.cast(model.step,tf.float64)+tf.constant(1.,tf.float64)
                c1=1-tf.pow(tf.constant(.9,tf.float64),t)
                c2=tf.cast(tf.sqrt(1-tf.pow(tf.constant(.999,tf.float64),t)),dtype)
                values=[]
                for name,p,g,m,v in zip(model.names,model.trainable,grads,model.moments1,model.moments2):
                    nm=m+tf.constant(1-.9,dtype)*(g-m)
                    nv=tf.constant(.999,dtype)*v+tf.constant(1-.999,dtype)*g*g
                    denominator=tf.sqrt(nv)/c2+tf.constant(1e-8,dtype)
                    step_size=tf.cast(tf.constant(case["learning_rates"][name],tf.float64)/c1,dtype)
                    values.append(p-step_size*(nm/denominator))
                return values
            candidate=tf.function(source_scalars,input_signature=specs,jit_compile=True,autograph=False)
            for j,update in enumerate(case["updates"]):
                reset(j)
                capture=update["captured"];initial=capture["initial"];data=update["data"]
                inputs=[tensor(initial["particles_downscaled"]),tensor(initial["particle_weights"]),
                    tensor(initial["bandwidths_downscaled"]),tensor(initial.get("resampling_particle_weights",initial["particle_weights"])),
                    tensor(initial.get("resampling_bandwidths_downscaled",initial["bandwidths_downscaled"])),
                    tensor(data["observations"]),tensor(data["states"]),tf.constant(data["ground_truth_mask"],tf.bool),
                    tensor(capture["samples"]),tensor(capture["noise"])]
                got=model.update(*inputs);assert bool(got["valid"].numpy()) and bool(got["committed"].numpy())
                devices.append(got["loss"].device)
                coupled=next(x for x in prior if x["variant"]==case["variant"] and x["dtype"]==case["dtype"]
                             and x["case"].endswith("/update"+str(j)))
                row=dict(variant=case["variant"],dtype=case["dtype"],update=j,
                    loss=compare(got["loss"],update["loss"],dtype),reset_gradients={},coupled_gradients={},adam={})
                for name,g in zip(model.names,got["raw_gradients"]):
                    row["reset_gradients"][name]=compare(g,update["raw_gradients"][name],dtype)
                    row["coupled_gradients"][name]=compare(coupled["raw_gradients"][name],update["raw_gradients"][name],dtype)
                grads=[tensor(update["clipped_gradients"][name]) for name in model.names]
                reset(j);old=isolated(*grads);assert bool(old["committed"].numpy())
                reset(j);new=candidate(*grads)
                for i,name in enumerate(model.names):
                    row["adam"][name]=dict(runtime=compare(old["parameters_after"][i],update["parameters_after"][name],dtype),
                        source_scalar_candidate=compare(new[i],update["parameters_after"][name],dtype),
                        first_moment=compare(old["first_moments"][i],update["adam_after"][name]["exp_avg"],dtype),
                        second_moment=compare(old["second_moments"][i],update["adam_after"][name]["exp_avg_sq"],dtype))
                result.append(row);write(out/"kdm-isolation.json",dict(rows=result))
            for name,kernel in [("update",model.update),("isolated",isolated),("candidate",candidate)]:
                traces[case["variant"]+"/"+case["dtype"]+"/"+name]=kernel.experimental_get_tracing_count()
        angles=json.loads((out.parent/"cpu/angles.json").read_text());angle_rows=[]
        for dtype_name in ("float32","float64"):
            dtype=tf.as_dtype(dtype_name)
            def angle_kernel(x):
                with tf.GradientTape() as tape:
                    tape.watch(x);r=tf.sqrt(x[:,2]**2+x[:,3]**2)
                    theta=tf.atan2(x[:,2]/r,x[:,3]/r);objective=tf.reduce_sum(theta)
                return theta,tape.gradient(objective,x)[:,2:]
            kernel=tf.function(angle_kernel,input_signature=[tf.TensorSpec([64,4],dtype)],jit_compile=True,autograph=False)
            for case in angles["cases"]:
                if case["dtype"]!=dtype_name:continue
                value,gradient=kernel(tf.constant(case["inputs"],dtype))
                angle_rows.append(dict(variant=case["variant"],dtype=dtype_name,update=case["update"],
                    angle=compare(value,case["angles"],dtype),gradient_source=compare(gradient,case["gradients"],dtype),
                    gradient_analytic=compare(gradient,case["exact_gradients"],dtype),
                    source_gradient_max_relative_error=case["source_gradient_max_relative_error"],
                    minimum_radius=case["minimum_radius"],maximum_wrapped_angle_change=case["maximum_wrapped_angle_change"]))
            traces["angle/"+dtype_name]=kernel.experimental_get_tracing_count()
        write(out/"angles.json",dict(rows=angle_rows))
    assert all("GPU:0" in d for d in devices) and all(n==1 for n in traces.values()),traces
    write(out/"summary.json",dict(status="COMPLETED",replication="INCOMPLETE",memory_policy=memory,
        tensorflow=tf.__version__,tf32=False,jit_compile=True,traces=traces,devices=sorted(set(devices)),optimizer_consumer_wiring=wiring,
        checkpoint_checks=checkpoint_checks,scope="matched-input root causes, no qualification",numpy_role="MATLAB reference loading/prediction only"))
    print(json.dumps(dict(status="COMPLETED",csmc_rows=len(fits),kdm_rows=len(result),angle_rows=len(angle_rows))))
if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--reference",type=Path,required=True);args=parser.parse_args();main(args.output,args.reference)
