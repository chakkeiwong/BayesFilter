from pathlib import Path
import os,subprocess,time,json,sys
root=Path(__file__).resolve().parent
command=[sys.executable,str(root/"gpu_parity_diagnostic.py")]
env=dict(os.environ,CUDA_VISIBLE_DEVICES="1",TF_FORCE_GPU_ALLOW_GROWTH="true",TF_NUM_INTEROP_THREADS="2",TF_NUM_INTRAOP_THREADS="2",OMP_NUM_THREADS="2",TF_CPP_MIN_LOG_LEVEL="2")
started=time.monotonic()
with (root/"gpu-check-01.log").open("w") as log:
    try:code=subprocess.run(command,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=180).returncode
    except subprocess.TimeoutExpired:code="timeout"
result=dict(returncode=code,elapsed_seconds=time.monotonic()-started,command=command)
(root/"gpu-check-01-supervisor.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(result))
if code!=0:print((root/"gpu-check-01.log").read_text()[-2500:])
