#!/usr/bin/env python3
"""Diagnostic only: compare unchanged independent XLA traces with repaired consumer."""
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
 spec=base.specification("m13-t10");fns={};start=time.monotonic()
 for label,directory in (("original_a",archive),("original_b",archive),("integrated",ROOT)):
  install(directory,label)
  fn=ex.compile_batch(spec,1024,base.controls(0),"marginal_mixture")
  fn.get_concrete_function();fns[label]=fn
 old=json.loads((ROOT/"docs/plans/artifacts/surrogate-hmc-marginal-covariance-selective-repair-20261006-01/result.json").read_text())
 rows=[]
 def difference(a,b):
  if not (a["valid"] and b["valid"]):return None
  v=abs(a["value"]-b["value"]);s=.1*(sum((x-y)**2 for x,y in zip(a["score"],b["score"]))/len(a["score"]))**.5
  return dict(value=v,score=s,within_screen=v<.005 and s<.005)
 for saved in (r for r in old["rows"] if r["scope"]=="m13-t10"):
  part,design_id,point=saved["partition"],saved["design"],saved["point"]
  obs=tf.cast(base.dataset(spec,part,dataset_id=1),tf.float32)
  design=base.designs(spec,1024,part,design_id,dataset_id=1)
  inputs=(tf.constant([spec["points"][point]],tf.float32),*design,obs)
  outputs={}
  for label,fn in fns.items():
   repetitions=[]
   for rep in range(2):
    tick=time.monotonic();raw=fn(*inputs)
    for key in ("value","score","valid"):raw[key].numpy()
    repetitions.append(ex.one(dict(base.serial(raw),seconds=time.monotonic()-tick)))
   outputs[label]=dict(first=repetitions[0],repeat_difference=difference(*repetitions))
  row=dict(partition=part,design=design_id,point=point,outputs=outputs,
   original_recompile=difference(outputs["original_a"]["first"],outputs["original_b"]["first"]),
   original_vs_saved=difference(outputs["original_a"]["first"],saved["marginal_mixture"]),
   integrated_vs_original=difference(outputs["original_a"]["first"],outputs["integrated"]["first"]))
  rows.append(row);base.save(args.output/"progress.json",dict(rows=rows))
 result=dict(status="COMPLETE",rows=rows,wall_seconds=time.monotonic()-start,source_overrides=sources,
  command=sys.argv,git_commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True,cwd=ROOT).strip(),
  environment=dict(python=sys.executable,tensorflow=tf.__version__,gpu=os.environ["CUDA_VISIBLE_DEVICES"],
   memory_policy=memory,jit_compile=True,tf32=True,cpu_only=False),
  data_version="M13T10 calibration/validation dataset1 designs0/1 all2points",
  plan="docs/plans/surrogate-hmc-marginal-expanded-execution-2026-10-06.md",
  nonclaim="Recompilation diagnosis only; no accuracy or HMC claim.")
 base.save(args.output/"result.json",result)
 print(json.dumps({key:sum(not r[key]["within_screen"] for r in rows) for key in
  ("original_recompile","original_vs_saved","integrated_vs_original")}),flush=True)
if __name__=="__main__":main()
