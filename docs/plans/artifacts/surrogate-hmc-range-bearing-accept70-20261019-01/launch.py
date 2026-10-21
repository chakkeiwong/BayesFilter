"""Launch the owner-requested target70 rerun with its saved, bounded command."""
import datetime as dt
import json
import os
from pathlib import Path
import subprocess

output = Path(__file__).resolve().parent
root = output.parents[3]
manifest = json.loads((output / "manifest.json").read_text())
if (output / "launch.json").exists() or (output / "run").exists():
    raise RuntimeError("This output already has a launch; inspect it before retrying")
environment = os.environ.copy()
environment.update(manifest["environment"])
with (output / "master.log").open("x") as log:
    child = subprocess.Popen(
        manifest["command"], cwd=root, env=environment, stdin=subprocess.DEVNULL,
        stdout=log, stderr=subprocess.STDOUT, start_new_session=True,
    )
receipt = dict(pid=child.pid, started_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
               command=manifest["command"], status="LAUNCHED",
               plan=manifest["plan"], maximum_worker_seconds=28800)
(output / "launch.json").write_text(json.dumps(receipt, indent=2) + "\n")
print(json.dumps(dict(pid=child.pid, status="LAUNCHED", output=str(output))))
