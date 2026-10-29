#!/usr/bin/env python3
"""Diagnostic only: paired actual-consumer covariance repair, with archived source."""
from __future__ import annotations
import argparse,hashlib,importlib,importlib.util,json,os,pathlib,subprocess,sys,time
ROOT=pathlib.Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument("--output",type=pathlib.Path,required=True)
 args=p.parse_args();args.output.mkdir(parents=True,exist_ok=False)
 os.environ.update(CUDA_VISIBLE_DEVICES="GPU-68251639-fe82-8f81-3ccc-2953c32e805b",
  TF_FORCE_GPU_ALLOW_GROWTH="true",TF_CPP_MIN_LOG_LEVEL="2",
  TF_NUM_INTRAOP_THREADS="4",TF_NUM_INTEROP_THREADS="2")
 sys.path.insert(0,str(ROOT))
 import tensorflow as tf
 from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
 memory=configure_tensorflow_gpu_memory_growth(tf)
 if not tf.config.list_physical_devices("GPU"):raise RuntimeError("trusted GPU required")
 tf.config.experimental.enable_tensor_float_32_execution(True)
 from bayesfilter.testing import marginal_weight_campaign as base,marginal_expanded as ex
 archive=ROOT/"docs/plans/artifacts/surrogate-hmc-marginal-expanded-20261006-03/source-before-continuation"
 sources={}
 def install(directory,label):
  for name in ("ledh_marginal_weights_tf","ledh_canonical_score_tf","ledh_canonical_batch_fused_tf"):
   fullname="bayesfilter.highdim."+name;path=directory/"bayesfilter/highdim"/(name+".py")
   spec=importlib.util.spec_from_file_location(fullname,path)
   mod=importlib.util.module_from_spec(spec);sys.modules[fullname]=mod;spec.loader.exec_module(mod)
   setattr(importlib.import_module("bayesfilter.highdim"),name,mod)
   sources[label+"/"+name]=dict(path=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest())
  base.canonical_batch_fused_value_score=mod.canonical_batch_fused_value_score
 start=time.monotonic();rows=[]
 for scope in ("lg1-t10","lg3-t10","ksc-t10","m13-t10","lg10-t10","lg20-t10"):
  spec=base.specification(scope);fns={}
  for label,directory in (("original",archive),("integrated",ROOT)):
   install(directory,label)
   fn=ex.compile_batch(spec,1024,base.controls(0),"marginal_mixture")
   fn.get_concrete_function();fns[label]=fn
  for part in ("calibration","validation"):
   obs=tf.cast(base.dataset(spec,part,dataset_id=1),tf.float32)
   for design_id in (0,1):
    design=base.designs(spec,1024,part,design_id,dataset_id=1)
    for point,theta in enumerate(spec["points"]):
     inputs=(tf.constant([theta],tf.float32),*design,obs);outputs={}
     for label,fn in fns.items():
      tick=time.monotonic();raw=fn(*inputs)
      for key in ("value","score","valid"):raw[key].numpy()
      outputs[label]=ex.one(dict(base.serial(raw),seconds=time.monotonic()-tick))
     a,b=outputs["original"],outputs["integrated"];delta=None
     if a["valid"] and b["valid"]:
      delta=dict(value=abs(a["value"]-b["value"]),
       score=.1*(sum((x-y)**2 for x,y in zip(a["score"],b["score"]))/len(a["score"]))**.5)
     count=b["diagnostics"]["pushed_covariance_precision_repair_count"]["maximum"]
     healthy_preserved=(not a["valid"] or b["valid"] and delta["value"]<.005 and delta["score"]<.005)
     recovery=(a["valid"] or b["valid"] and count>0)
     rows.append(dict(scope=scope,partition=part,design=design_id,point=point,
      outputs=outputs,delta=delta,healthy_preserved=healthy_preserved,recovery=recovery,
      passed=healthy_preserved and recovery and b["valid"]))
     base.save(args.output/"progress.json",dict(rows=rows))
  print(scope,"complete",flush=True)
 result=dict(status="COMPLETE",passed=all(r["passed"] for r in rows),rows=rows,
  wall_seconds=time.monotonic()-start,source_overrides=sources,
  command=sys.argv,git_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True,cwd=ROOT).strip(),
  environment=dict(python=sys.executable,tensorflow=tf.__version__,gpu=os.environ["CUDA_VISIBLE_DEVICES"],
   memory_policy=memory,jit_compile=True,tf32=True,cpu_only=False),
  data_version="six T10 scopes, calibration/validation dataset1 designs0/1 all points",
  plan="docs/plans/surrogate-hmc-marginal-expanded-execution-2026-10-06.md",
  source_sha256={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
   [pathlib.Path(__file__),ROOT/"bayesfilter/testing/marginal_expanded.py",ROOT/"bayesfilter/testing/marginal_weight_campaign.py"]},
  nonclaim="Paired implementation non-harm only; cross-process repeatability, score accuracy and HMC unproved.")
 base.save(args.output/"result.json",result)
 print("COMPLETE",result["passed"],flush=True)
if __name__=="__main__":main()
