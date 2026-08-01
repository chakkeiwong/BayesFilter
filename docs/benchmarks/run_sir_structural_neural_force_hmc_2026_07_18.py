#!/usr/bin/env python3
"""Run one P5 SIR/structural corrected neural-force HMC cell."""
from __future__ import annotations
import argparse,hashlib,json,subprocess,sys,time
from datetime import datetime,timezone
from pathlib import Path
from typing import Any,Mapping
ROOT=Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path: sys.path.insert(0,str(ROOT))
PLAN="docs/plans/bayesfilter-hnn-surrogate-hmc-p5-sir-structural-subplan-2026-07-17.md"
def main()->int:
 p=argparse.ArgumentParser();p.add_argument("--cell",required=True,choices=("SIR-SGQF","STR-UKF"));p.add_argument("--output-root",type=Path,required=True);p.add_argument("--smoke",action="store_true");a=p.parse_args()
 if a.output_root.exists(): raise FileExistsError(a.output_root)
 a.output_root.mkdir(parents=True);started=datetime.now(timezone.utc);clock=time.monotonic()
 import tensorflow as tf, tensorflow_probability as tfp
 from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth
 memory=configure_tensorflow_gpu_memory_growth(tf,require_gpu=True);tf.config.set_soft_device_placement(False);tf.config.experimental.enable_tensor_float_32_execution(True)
 from bayesfilter.testing import sir_structural_neural_force_hmc_tf as p5
 context=p5.load_context(a.cell);result=p5.run_smoke(context) if a.smoke else p5.run_cell(context,a.output_root)
 manifest={"schema":"bayesfilter.sir_structural_neural_force_hmc_p5_manifest.v1","git_commit":subprocess.run(("git","rev-parse","HEAD"),check=True,capture_output=True,text=True).stdout.strip(),"command":" ".join(sys.argv),"environment":"tf-gpu","tensorflow_version":tf.__version__,"tfp_version":tfp.__version__,"device":"/GPU:0","gpu_memory_policy":memory,"tf32_enabled":bool(tf.config.experimental.tensor_float_32_execution_enabled()),"jit_compile":True,"started_at_utc":started.isoformat(),"wall_time_seconds":time.monotonic()-clock,"plan_file":PLAN,"result_file":str(a.output_root/"result.json"),"trust_basis":"owner_designated_managed_session_visible_gpu_trusted"}
 result={**result,"run_manifest":manifest};write(a.output_root/"result.json",result);write(a.output_root/"run_manifest.json",manifest);write(a.output_root/"artifact_hashes.json",{"result_sha256":sha(a.output_root/"result.json"),"run_manifest_sha256":sha(a.output_root/"run_manifest.json")});print(json.dumps({"cell":a.cell,"passed":result["passed"],"smoke":a.smoke}));return 0 if result["passed"] else 1
def ready(v:Any)->Any:
 if isinstance(v,Mapping):return {str(k):ready(x) for k,x in v.items()}
 if isinstance(v,(tuple,list)):return [ready(x) for x in v]
 if isinstance(v,Path):return str(v)
 if hasattr(v,"numpy"):return ready(v.numpy().tolist())
 return v
def write(path:Path,payload:Mapping[str,Any])->None:
 with path.open("x",encoding="utf-8") as f:json.dump(ready(payload),f,sort_keys=True,indent=2);f.write("\n")
def sha(path:Path)->str:return hashlib.sha256(path.read_bytes()).hexdigest()
if __name__=="__main__":raise SystemExit(main())
