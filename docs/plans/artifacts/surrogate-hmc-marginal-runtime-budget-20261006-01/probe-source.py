#!/usr/bin/env python3
"""Diagnostic equal-runtime calibration; reporting lies outside timed calls."""
from __future__ import annotations
import argparse,hashlib,json,math,os,pathlib,statistics,subprocess,sys,time
ROOT=pathlib.Path(__file__).resolve().parents[1]
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--output",type=pathlib.Path,required=True)
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    os.environ.update(CUDA_VISIBLE_DEVICES="GPU-68251639-fe82-8f81-3ccc-2953c32e805b",
        TF_FORCE_GPU_ALLOW_GROWTH="true",TF_CPP_MIN_LOG_LEVEL="2",
        TF_NUM_INTRAOP_THREADS="4",TF_NUM_INTEROP_THREADS="2")
    sys.path.insert(0,str(ROOT))
    import tensorflow as tf
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
    memory=configure_tensorflow_gpu_memory_growth(tf)
    if not tf.config.list_physical_devices("GPU"):raise RuntimeError("GPU required")
    tf.config.experimental.enable_tensor_float_32_execution(True)
    from bayesfilter.testing import marginal_weight_campaign as base,marginal_expanded as ex
    def sync(result):
        # Materialize every inferential output. Reporting/diagnostic reductions
        # are performed later and are not part of this latency measurement.
        for key in ("value","score","valid"):
            if key in result:result[key].numpy()
    def timed(fn,*inputs):
        tick=time.perf_counter();out=fn(*inputs);sync(out)
        seconds=time.perf_counter()-tick
        return out,seconds
    campaign=ROOT/"docs/plans/artifacts/surrogate-hmc-marginal-expanded-20261006-03"
    started=time.monotonic();rows=[]
    for name in ("ksc-t10","m13-t10"):
        spec=base.specification(name);n=1024
        controls=json.loads((campaign/"scopes"/(name+"-n1024")/"controls.json").read_text())["controls"]
        observations=tf.cast(base.dataset(spec,"validation",dataset_id=1),tf.float32)
        theta=tf.constant(spec["points"][0],tf.float32)
        ledh=ex.compile_batch(spec,n,controls,"marginal_mixture")
        pf=ex.Kernels(spec).get("bootstrap",n,dtype=tf.float32)
        design=base.designs(spec,n,"validation",0,dataset_id=1)
        timed(ledh,theta[None],*design,observations)
        timed(pf,theta,observations,tf.constant([20261006,9900000],tf.int32))
        lt=[timed(ledh,theta[None],*design,observations)[1] for _ in range(5)]
        bt=[timed(pf,theta,observations,tf.constant([20261006,9900010+j],tf.int32))[1]
            for j in range(9)]
        target=statistics.median(lt);count=min(256,max(1,round(target/statistics.median(bt))))
        def block(repeats,seed_offset):
            values=[];tick=time.perf_counter()
            for j in range(repeats):
                out=pf(theta,observations,tf.constant([20261006,seed_offset+j],tf.int32))
                sync(out);values.append(float(out["value"]))
            peak=max(values)
            value=peak+math.log(sum(math.exp(v-peak) for v in values)/len(values))
            return dict(value=value,seconds=time.perf_counter()-tick,repeats=repeats)
        calibration=[]
        for attempt in range(3):
            measured=[block(count,9910000+attempt*10000+j*1000)["seconds"] for j in range(3)]
            cost=statistics.median(measured)
            calibration.append(dict(count=count,seconds=measured,ratio=cost/target))
            if .9<=cost/target<=1.1:break
            count=min(256,max(1,round(count*target/cost)))
        pairs=[]
        for design_id in range(8):
            design=base.designs(spec,n,"validation",design_id,dataset_id=1)
            out,seconds=timed(ledh,theta[None],*design,observations)
            comparator=block(count,10000000+design_id*1000)
            row=base.serial(out)
            ratio=comparator["seconds"]/seconds
            pairs.append(dict(design=design_id,ledh_value=row["value"][0],ledh_valid=row["valid"][0],
                ledh_seconds=seconds,bootstrap=comparator,cost_ratio=ratio,
                matched=.8<=ratio<=1.2,valid=bool(row["valid"][0]) and math.isfinite(comparator["value"])))
        row=dict(scope=name,n=n,point=0,controls=controls,target_seconds=target,
            warm_ledh_seconds=lt,warm_bootstrap_seconds=bt,calibration=calibration,
            frozen_repeats=count,rows=pairs,matched=sum(p["matched"] for p in pairs),
            passed=all(p["matched"] and p["valid"] for p in pairs))
        rows.append(row);base.save(args.output/"progress.json",dict(rows=rows))
        print(name,"matched",row["matched"],"/",len(pairs),flush=True)
    paths=[pathlib.Path(__file__),ROOT/"scripts/run_surrogate_hmc_marginal_expanded.py",
           ROOT/"bayesfilter/testing/marginal_expanded.py"]
    base.save(args.output/"result.json",dict(status="COMPLETE",passed=all(r["passed"] for r in rows),rows=rows,
        command=sys.argv,wall_seconds=time.monotonic()-started,
        git_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
        environment=dict(python=sys.executable,tensorflow=tf.__version__,gpu=os.environ["CUDA_VISIBLE_DEVICES"],
            memory_policy=memory,jit_compile=True,tf32=True,cpu_only=False),
        data_version="validation dataset1, designs0..7, point0; controls frozen by active scope",
        seed_policy="timing calibration seeds9900000..9939999; evaluation10000000..10007999",
        plan="docs/plans/surrogate-hmc-marginal-expanded-execution-2026-10-06.md",
        timing="synchronized value/score/valid calls; serialization excluded; PF aggregation included",
        nonclaim="Timing repair only; no score/PF equivalence, ranking, accuracy promotion, or HMC claim",
        source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}))
if __name__=="__main__":main()
