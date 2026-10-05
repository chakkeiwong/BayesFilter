"""Record a bounded CPU reference check without modifying its source tree."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

p = argparse.ArgumentParser()
p.add_argument('--output', type=Path, required=True)
p.add_argument('--cwd', type=Path, required=True)
p.add_argument('--timeout', type=float, required=True)
p.add_argument('command', nargs=argparse.REMAINDER)
args = p.parse_args()
command = args.command[1:] if args.command[:1] == ['--'] else args.command
root = args.output.resolve()
root.mkdir(parents=True, exist_ok=False)
env = os.environ.copy()
env.update(CUDA_VISIBLE_DEVICES='-1', TF_FORCE_GPU_ALLOW_GROWTH='true',
           TF_NUM_INTRAOP_THREADS='2', TF_NUM_INTEROP_THREADS='1',
           OMP_NUM_THREADS='2', OPENBLAS_NUM_THREADS='1',
           TF_CPP_MIN_LOG_LEVEL='2', BAYESFILTER_PRELOAD_CUSTOM_OP='0')
env.pop('PYTHONPATH', None)
os.sched_setaffinity(0, {20, 21, 22, 23})
started = time.monotonic()
record = dict(command=command, cwd=str(args.cwd.resolve()),
              started_utc=datetime.now(timezone.utc).isoformat(),
              environment={k: env[k] for k in ('CUDA_VISIBLE_DEVICES',
                 'TF_FORCE_GPU_ALLOW_GROWTH', 'TF_NUM_INTRAOP_THREADS',
                 'TF_NUM_INTEROP_THREADS', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS')},
              cpu_affinity=sorted(os.sched_getaffinity(0)),
              gpu_intentionally_hidden=True, resource='cpu_reference',
              plan='docs/plans/bayesfilter-hmc-v7-release-plan-2026-10-02.md')
(root/'manifest.json').write_text(json.dumps(record, indent=2)+'\n')
with (root/'run.log').open('x') as log:
    try:
        code = subprocess.run(command, cwd=args.cwd, env=env, stdout=log,
                              stderr=subprocess.STDOUT, timeout=args.timeout).returncode
    except subprocess.TimeoutExpired:
        code = 124
record.update(exit_code=code, wall_seconds=time.monotonic()-started,
              log_sha256=hashlib.sha256((root/'run.log').read_bytes()).hexdigest())
(root/'receipt.json').write_text(json.dumps(record, indent=2)+'\n')
print(json.dumps(record))
print((root/'run.log').read_text()[-4500:])
raise SystemExit(code)
