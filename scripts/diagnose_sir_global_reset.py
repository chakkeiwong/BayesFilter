"""Exact-seed diagnostic; never changes or admits the failed proposal."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "docs/benchmarks"))
from run_ledh_covariance_proposal import clean, design, prepare, spec_for


parser = argparse.ArgumentParser()
parser.add_argument("--output", required=True, type=Path)
args = parser.parse_args()
args.device, args.dtype, args.no_tf32 = "gpu", "float64", True
started = time.monotonic()
output = args.output
output.mkdir(parents=True, exist_ok=False)
tensorflow, memory = prepare(args)
from bayesfilter.highdim.covariance_proposal_tf import ProposalControls, TRACE_FIELDS, make_filter

source = ROOT / "docs/plans/artifacts/ledh-sir-guidance-20261019/attempt-02"
manifest = json.loads((source / "manifest.json").read_text())
failed = next(json.loads(line) for line in (source / "rows.jsonl").read_text().splitlines()
              if not json.loads(line)["valid"])
data_path = ROOT / manifest["data"][str(failed["data_seed"])]["path"]
observed = tensorflow.constant(json.loads(data_path.read_text())["observations"], tensorflow.float64)
spec = spec_for("sir_d18")
theta = spec.default_theta(tensorflow.float64)
fixed = design(tensorflow, 256, 50, 18, failed["seed"], tensorflow.float64)
program = make_filter(spec, 256, 50, ProposalControls(reset_epsilon=1., reset_steps=5, flow_steps=16),
                      tensorflow.float64, jit_compile=True)
result = program(theta, tensorflow.one_hot(0, 3, dtype=tensorflow.float64), observed,
                 *fixed, tensorflow.constant(failed["beta"], tensorflow.float64))
steps = int(result["steps_completed"])
trace = clean(result["trace"][:steps])
prior_trace = failed["traces"][0][:steps]
max_difference = max(abs(actual - prior) for current, previous in zip(trace, prior_trace)
                     for actual, prior in zip(current, previous))
record = dict(command=sys.argv, source_manifest_sha256=hashlib.sha256((source / "manifest.json").read_bytes()).hexdigest(),
              source_hashes=manifest["source_sha256"], data_seed=failed["data_seed"], seed=failed["seed"],
              arm=failed["arm"], beta=failed["beta"], reset_epsilon=1., reset_steps=5,
              valid=bool(result["valid"]), steps_completed=steps, terminal_trace=dict(zip(TRACE_FIELDS, trace[-1])),
              maximum_trace_difference=max_difference, reproduced=not bool(result["valid"]) and steps == 4 and max_difference < 1e-9,
              device=result["value"].device, memory_policy=memory, tensorflow=tensorflow.__version__,
              dtype="float64", tf32=False, jit_compile=True, cpu_only=False,
              cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
              xla_flags=os.environ.get("XLA_FLAGS"), wall_seconds=time.monotonic() - started,
              conclusion="Failed comparator is quarantined; no score, default, or full-design promotion")
(output / "diagnostic.json").write_text(json.dumps(clean(record), indent=2, allow_nan=False) + "\n")
print(json.dumps({key: record[key] for key in ("valid", "steps_completed", "maximum_trace_difference", "reproduced", "wall_seconds")}))
