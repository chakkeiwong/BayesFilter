"""CPU-only standard-library checks for the bounded benchmark supervisor."""
import json
from pathlib import Path
import subprocess
import sys

from scripts import benchmark_sir_fp32_batch_capacity as bench


def test_nonfinite_diagnostics_remain_explicit_json_null(tmp_path):
    path = tmp_path / "record.json"
    bench.save(path, {"value": float("nan"), "rows": [float("inf"), 3.0], "valid": False})
    assert json.loads(path.read_text()) == {"value": None, "rows": [None, 3.0], "valid": False}
    assert not path.with_suffix(".json.tmp").exists()


def test_supervisor_import_does_not_initialize_tensorflow():
    result = subprocess.run([sys.executable, "-c",
        "import sys; from scripts import benchmark_sir_fp32_batch_capacity; "
        "assert 'tensorflow' not in sys.modules"], cwd=bench.ROOT, capture_output=True)
    assert result.returncode == 0, result.stderr


def test_termination_reaps_only_owned_process_group():
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"],
                               start_new_session=True)
    try:
        bench.terminate_owned(process)
        assert process.poll() is not None
        bench.terminate_owned(process)  # Already-exited process is harmless.
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()


def test_resource_observers_report_bytes_without_framework():
    assert bench.meminfo()["MemTotal"] > 1024 ** 3
    assert bench.process_memory(999999999) == {}
    assert bench.PLAN.is_file() and bench.SELECTION.is_file()
