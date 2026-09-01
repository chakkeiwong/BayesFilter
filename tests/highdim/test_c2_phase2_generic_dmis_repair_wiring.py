"""Executable call-chain check for the C2 generic-DMIS diagnostic.

This test is diagnostic infrastructure only. It verifies that the C2 adapter's
claim-bearing endpoint resolves to the model-independent kernel, rather than
merely checking that the kernel exists somewhere in the repository.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "docs/benchmarks/run_c2_phase2_generic_dmis_repair_20260902.py"
EXPECTED = "bayesfilter.highdim.frozen_dmis_control_variate_tf"


def test_c2_adapter_resolves_all_generic_dmis_endpoints() -> None:
    probe = r'''
import importlib.util
import json
from pathlib import Path

script = Path("SCRIPT_PLACEHOLDER")
spec = importlib.util.spec_from_file_location("c2_phase2_repair_wiring", script)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
globals_ = module._one_bank_record.__globals__
expected = "EXPECTED_PLACEHOLDER"
payload = {
    "value": globals_["frozen_importance_estimate"].__module__,
    "directional": globals_["frozen_importance_directional_estimate"].__module__,
    "mixture": globals_["complete_mixture_log_density"].__module__,
}
print(json.dumps(payload, sort_keys=True))
assert all(value == expected for value in payload.values())
'''.replace("SCRIPT_PLACEHOLDER", str(SCRIPT)).replace("EXPECTED_PLACEHOLDER", EXPECTED)
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = "-1"
    environment["MPLCONFIGDIR"] = "/tmp/mpl-c2-phase2-wiring-test"
    completed = subprocess.run(
        [sys.executable, "-c", probe],
        cwd=ROOT,
        env=environment,
        check=True,
        capture_output=True,
        text=True,
    )
    payload = json.loads(completed.stdout.strip().splitlines()[-1])
    assert payload == {
        "directional": EXPECTED,
        "mixture": EXPECTED,
        "value": EXPECTED,
    }
