"""Inject one host interruption after a real filter row, then use CLI resume.

This is a CPU-only mechanics check. Numerical values come from the registered
providers; only the pre-call interruption is artificial.
"""
import json
import os
from pathlib import Path
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["BAYESFILTER_PRELOAD_CUSTOM_OP"] = "0"
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bayesfilter.score_study.coordinator import execute, load_endpoint
from bayesfilter.score_study.registry import default_registry

study = json.loads(Path(sys.argv[1]).read_text())
output = Path(sys.argv[2])
calls = 0
def interrupt_once(name):
    endpoint = load_endpoint(name)
    def invoke(row, context):
        global calls
        calls += 1
        if calls == 2:
            raise KeyboardInterrupt("intentional host interruption after one real completed filter row")
        return endpoint(row, context)
    return invoke
try:
    execute(study, default_registry(), output, endpoint_loader=interrupt_once)
except KeyboardInterrupt:
    state = json.loads((output / "state.json").read_text())
    assert sum(r["execution_status"] == "complete" for r in state["rows"].values()) == 1
    assert sum(r["execution_status"] == "interrupted" for r in state["rows"].values()) == 1
    print(json.dumps({"interruption_injected": True, "output": str(output)}))
else:
    raise AssertionError("interruption was not exercised")
