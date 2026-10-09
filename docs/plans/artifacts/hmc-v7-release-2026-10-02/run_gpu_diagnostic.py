"""Record one bounded saved-evidence GPU diagnostic with an enclosing receipt."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time


p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--output", type=Path, required=True)
p.add_argument("--gpu", required=True)
p.add_argument("--timeout", type=float, required=True)
p.add_argument("--cpus", type=int, nargs=4, default=[20, 21, 22, 23])
p.add_argument("--native-sampling", action="store_true")
p.add_argument("command", nargs=argparse.REMAINDER)
args = p.parse_args()
command = args.command[1:] if args.command[:1] == ["--"] else args.command
root = args.output.resolve()
root.mkdir(parents=True, exist_ok=False)
env = os.environ.copy()
env.update(CUDA_VISIBLE_DEVICES=args.gpu, TF_FORCE_GPU_ALLOW_GROWTH="true",
    TF_NUM_INTRAOP_THREADS="2", TF_NUM_INTEROP_THREADS="1", OMP_NUM_THREADS="2",
    OPENBLAS_NUM_THREADS="1", TF_CPP_MIN_LOG_LEVEL="2", BAYESFILTER_PRELOAD_CUSTOM_OP="0")
env.pop("PYTHONPATH", None)
if len(set(args.cpus)) != 4:
    raise ValueError("four distinct CPU indices required")
os.sched_setaffinity(0, set(args.cpus))
started = time.monotonic()
record = dict(command=command, cwd=str(Path.cwd()), gpu_uuid=args.gpu,
    started_utc=datetime.now(timezone.utc).isoformat(), resource="gpu",
    environment={k:env[k] for k in ("CUDA_VISIBLE_DEVICES", "TF_FORCE_GPU_ALLOW_GROWTH",
        "TF_NUM_INTRAOP_THREADS", "TF_NUM_INTEROP_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS")},
    cpu_affinity=sorted(os.sched_getaffinity(0)), numerical_sampling=args.native_sampling,
    plan="docs/plans/bayesfilter-hmc-v7-release-plan-2026-10-02.md")
(root/"manifest.json").write_text(json.dumps(record, indent=2)+"\n")
with (root/"run.log").open("x") as log:
    try:
        code = subprocess.run(command, env=env, stdout=log,
            stderr=subprocess.STDOUT, timeout=args.timeout).returncode
    except subprocess.TimeoutExpired:
        code = 124
record.update(exit_code=code, wall_seconds=time.monotonic()-started,
    log_sha256=hashlib.sha256((root/"run.log").read_bytes()).hexdigest())
(root/"receipt.json").write_text(json.dumps(record, indent=2)+"\n")
(root/"launcher.py").write_bytes(Path(__file__).read_bytes())
print(json.dumps(record))
print((root/"run.log").read_text()[-6000:])
raise SystemExit(code)
