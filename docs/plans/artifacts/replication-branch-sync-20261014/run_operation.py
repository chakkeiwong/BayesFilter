from pathlib import Path
import json
import os
import subprocess
import sys
import time

root = Path(sys.argv[1]).resolve()
label = sys.argv[2]
command = sys.argv[3:]
scratch = Path('/tmp/bayesfilter-replication-sync-20261014')
environment = os.environ.copy()
environment.update(CUDA_VISIBLE_DEVICES='-1', TF_FORCE_GPU_ALLOW_GROWTH='true', DISPLAY='', WAYLAND_DISPLAY='', PYTHONDONTWRITEBYTECODE='1')
environment['PYTHONPATH'] = str(root)
log = scratch / (label + '.log')
started = time.monotonic()
with log.open('w') as output:
    process = subprocess.run(command, cwd=root, env=environment, stdout=output, stderr=subprocess.STDOUT, timeout=600)
receipt = dict(command=command, cwd=str(root), cpu_only=True, CUDA_VISIBLE_DEVICES='-1', PYTHONPATH=str(root), DISPLAY='', WAYLAND_DISPLAY='', exit_code=process.returncode, wall_seconds=time.monotonic()-started, log=str(log))
(scratch / (label + '.json')).write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt))
print('\n'.join(log.read_text().splitlines()[-8:]))
sys.exit(process.returncode)
