"""GPU/XLA validation and bounded timing of independent Lorenz repetitions."""
import os
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
import argparse
import gc
import hashlib
import inspect
import json
import math
from pathlib import Path
import sys
import time

sys.path.insert(0, "/home/chakwong/BayesFilter")
import tensorflow as tf
from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
memory = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
tf.config.experimental.enable_tensor_float_32_execution(False)
import lorenz_refinement_tf as controller
D=tf.float64


@tf.function(input_signature=[tf.TensorSpec([2],tf.int32)],jit_compile=True,autograph=False)
def noise(seed):
    return (tf.random.stateless_normal([5,101,4,512,8],seed,dtype=D),
            tf.random.stateless_uniform([5,101,4],seed+tf.constant([0,100000]),dtype=D))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    parser.add_argument("--reference",type=Path,required=True)
    parser.add_argument("--mode",choices=["validate","pilot"],required=True)
    parser.add_argument("--scalar-pilot",type=Path)
    args=parser.parse_args();started=time.monotonic()
    checks,cases,rows=[],[],[]
    summary=dict(status="FAIL",scope="LORENZ_BATCH_"+args.mode.upper(),replication="INCOMPLETE",
        device="GPU",dtype="float64",tf32=False,jit_compile=True,replicas=4,
        gpu_memory_policy=memory,tensorflow=tf.__version__,checks=checks,cases=cases,rows=rows,
        heuristic_dominance_verdict="NOT_ASSESSED_IMPLEMENTATION_AND_COST_ONLY")
    def flag(name,good,**detail):
        checks.append(dict(name=name,passed=bool(good),**detail))
        if not good:raise ValueError(name)
    def compare(name,a,e,exact=False):
        a,e=tf.cast(a,D),tf.cast(e,D)
        delta=tf.abs(a-e);limit=tf.zeros_like(e) if exact else 1e-7+1e-9*tf.abs(e)
        flag(name,tf.reduce_all(tf.math.is_finite(a)&tf.math.is_finite(e)&(delta<=limit)).numpy(),
             max_absolute_error=float(tf.reduce_max(delta).numpy()))
    def progress():
        (args.output/"progress.json").write_text(json.dumps(dict(
            completed_rows=len(rows),context=dict(completed_cases=len(cases)),rows=rows),indent=2)+"\n")
    def snapshot(symbol):
        actual=Path(inspect.getfile(symbol));saved=Path(__file__).parent/actual.name
        flag("snapshot_"+symbol.__name__,actual.read_bytes()==saved.read_bytes())
    def load(item):
        payload=(args.reference/item["file"]).read_bytes()
        flag("hash_"+item["file"],hashlib.sha256(payload).hexdigest()==item["sha256"])
        return tf.reshape(tf.io.decode_raw(payload,tf.int32 if item["dtype"]=="int32" else D),item["shape"])
    def traces(engine,label):
        counts={k:getattr(engine.engine,k).experimental_get_tracing_count() for k in ("forward","fit")}
        counts["controller"]=engine.run.experimental_get_tracing_count()
        flag(label,all(v==1 for v in counts.values()),counts=counts)
    def diagnose_failure(batch, scalar, out, y, by, normal, uniform, grid_index):
        # Preserve rejected inputs and computed diagnostics before raising.
        detail=dict(grid_index=grid_index,counts=out["iterations"].numpy().tolist(),
            gram=out["minimum_fit_gram_ratio"].numpy().tolist(),
            rss=out["maximum_fit_rss"].numpy().tolist(),scalar_rows=[],matched_fits=[])
        folder=args.output/"rejected_batch";folder.mkdir()
        for name,value in dict(y=by,normal=normal,uniform=uniform,**out["history"]).items():
            payload=tf.io.serialize_tensor(value)
            tf.io.write_file(str(folder/(name+".tensor")),payload)
        for row in range(4):
            separate=scalar.run(y,normal[:,:,row:row+1],uniform[:,:,row:row+1],tf.constant(4))
            detail["scalar_rows"].append(dict(row=row,valid=bool(separate["valid"].numpy()),
                iterations=int(separate["iterations"].numpy()),
                gram=separate["minimum_fit_gram_ratio"].numpy().tolist(),
                rss=separate["maximum_fit_rss"].numpy().tolist()))
            for stage in range(1,detail["counts"][row]+1):
                saved=out["history"]
                previous=[saved[k][stage-1,:,row:row+1] for k in ("A","b","c","x","log_incremental_weight")]
                fitted=scalar.engine.fit(*previous)
                detail["matched_fits"].append(dict(row=row,stage=stage,
                    valid=bool(fitted["valid"].numpy()),
                    min_precision_margin=float(tf.reduce_min(fitted["precision_margin"]).numpy()),
                    min_gram_ratio=float(tf.reduce_min(fitted["gram_eigenvalue_ratio"]).numpy()),
                    max_rss=float(tf.reduce_max(fitted["residual_sum_squares"]).numpy())))
        summary["failure_diagnostics"]=detail
        # Non-finite rejected diagnostics remain JSON null, never false PASS.
        def finite_json(value):
            if isinstance(value,float) and not math.isfinite(value):return None
            if isinstance(value,list):return [finite_json(v) for v in value]
            if isinstance(value,dict):return {k:finite_json(v) for k,v in value.items()}
            return value
        summary["failure_diagnostics"]=finite_json(detail)
        (args.output/"failure-diagnostics.json").write_text(json.dumps(summary["failure_diagnostics"],indent=2,allow_nan=False)+"\n")
    try:
        snapshot(controller.MultivariateGaussianControlledSMC)
        from bayesfilter.replication.controlled_smc_tf import systematic_ancestors
        snapshot(systematic_ancestors)
        index=json.loads((args.reference/"fixture.json").read_text())
        y=tf.constant(index["metadata"]["y"],D)[:,None,:]
        by=tf.repeat(y,4,axis=1)
        if args.mode=="validate":
            @tf.function(input_signature=[tf.TensorSpec([4],tf.int32),tf.TensorSpec([2,4],D)],
                         jit_compile=True,autograph=False)
            def boundary(counts,ess):
                return controller.active_refinements(counts,tf.constant(4),ess,512)
            compare("independent_stop_boundaries",
                    boundary(tf.constant([0,1,1,4]),tf.constant([[512.,460.8,460.80001,1.]]*2,D)),
                    tf.constant([True,True,False,False]),exact=True)
            flag("boundary_one_trace",boundary.experimental_get_tracing_count()==1)
        for case_index,grid_index in enumerate((0,15,39)):
            forcing=2.5+6*grid_index/39
            batch=controller.LorenzRefinement(forcing=forcing,replicas=4)
            if args.mode=="validate":
                scalar=controller.LorenzRefinement(forcing=forcing)
                normal,uniform=noise(tf.constant([202609380,grid_index],tf.int32))
                # First case also checks old scalar behavior on the fixed69 design.
                if case_index==0:
                    oldn,oldu=load(index["normal"]),load(index["uniform"])
                    old=scalar.run(y,oldn[:,:,None,:,:],oldu[:,:,None],tf.constant(4))
                    flag("old_scalar_valid",old["valid"].numpy())
                    compare("old_scalar_count",old["iterations"],tf.constant(4),exact=True)
                    for stage,source in enumerate(index["cases"][0]["stages"]):
                        for key in ("x","ancestors","log_incremental_weight","logZ","ess"):
                            compare("old_stage%d_%s"%(stage,key),old["history"][key][stage,:,0],
                                    load(source["output"][key]),exact=(key=="ancestors"))
                before=time.monotonic();out=batch.run(by,normal,uniform,tf.constant(4))
                if not bool(out["valid"].numpy()):
                    diagnose_failure(batch,scalar,out,y,by,normal,uniform,grid_index)
                flag("batch_valid_%d"%grid_index,out["valid"].numpy())
                elapsed=time.monotonic()-before
                counts=out["iterations"].numpy().tolist()
                for row in range(4):
                    separate=scalar.run(y,normal[:,:,row:row+1],uniform[:,:,row:row+1],tf.constant(4))
                    flag("scalar_valid_%d_%d"%(grid_index,row),separate["valid"].numpy())
                    count=int(separate["iterations"].numpy())
                    flag("independent_count_%d_%d"%(grid_index,row),counts[row]==count)
                    compare("executed_%d_%d"%(grid_index,row),out["executed"][:,row],
                            separate["executed"],exact=True)
                    for stage in range(count+1):
                        for key in ("x","ancestors","log_incremental_weight","logZ","ess"):
                            compare("grid%d_row%d_stage%d_%s"%(grid_index,row,stage,key),
                                out["history"][key][stage,:,row],
                                separate["history"][key][stage,:,0],exact=(key=="ancestors"))
                        if stage==0:continue
                        saved=out["history"]
                        previous=[saved[k][stage-1,:,row:row+1] for k in ("A","b","c","x","log_incremental_weight")]
                        matched=scalar.engine.fit(*previous)
                        flag("matched_valid_%d_%d_%d"%(grid_index,row,stage),matched["valid"].numpy())
                        for key in ("A","b","c"):
                            compare("matched_grid%d_row%d_stage%d_%s"%(grid_index,row,stage,key),
                                    saved[key][stage,:,row],matched[key][:,0])
                traces(scalar,"scalar_traces_%d"%grid_index)
                traces(batch,"batch_traces_%d"%grid_index)
                cases.append(dict(grid_index=grid_index,forcing=forcing,iterations=counts,seconds=elapsed))
                progress()
                del scalar,out
            else:
                for repeat in range(4):
                    seed=[202609390,grid_index*100+repeat]
                    before=time.monotonic();normal,uniform=noise(tf.constant(seed,tf.int32))
                    out=batch.run(by,normal,uniform,tf.constant(4))
                    flag("valid_%d_%d"%(grid_index,repeat),out["valid"].numpy())
                    elapsed=time.monotonic()-before
                    counts=out["iterations"].numpy().tolist()
                    flag("counts_%d_%d"%(grid_index,repeat),all(1<=c<=4 for c in counts))
                    row=dict(grid_index=grid_index,forcing=forcing,repeat=repeat,seed=seed,
                        iterations=counts,cold=repeat==0,batch_seconds=elapsed,per_system_seconds=elapsed/4)
                    rows.append(row);progress()
                traces(batch,"batch_traces_%d"%grid_index)
            del batch
            gc.collect()
        flag("noise_one_trace",noise.experimental_get_tracing_count()==1)
        if args.mode=="pilot":
            old=json.loads((args.scalar_pilot/"summary.json").read_text())
            timing=old["timing_components"]
            warm=max(row["per_system_seconds"] for row in rows if not row["cold"])
            cold=max(row["batch_seconds"] for row in rows if row["cold"])
            forecast=2*(4000*(warm+timing["max_warm_apf"])+40*cold+160*timing["max_cold_apf"])
            summary.update(forecast_gpu_seconds=forecast,max_warm_per_system_seconds=warm,
                           max_cold_batch_seconds=cold,apf_timing_source=str(args.scalar_pilot))
        summary["status"]="PASS"
    except Exception as error:
        summary["error"]=repr(error);progress()
    summary["wall_seconds"]=time.monotonic()-started
    summary["allocator"]=tf.config.experimental.get_memory_info("GPU:0")
    (args.output/"summary.json").write_text(json.dumps(summary,indent=2,allow_nan=False)+"\n")
    print(json.dumps(dict(status=summary["status"],checks=len(checks),cases=len(cases),
        error=summary.get("error", "")[:600],forecast_gpu_seconds=summary.get("forecast_gpu_seconds"),
        wall_seconds=summary["wall_seconds"])))
    if summary["status"]!="PASS":raise SystemExit(1)


if __name__=="__main__":main()
