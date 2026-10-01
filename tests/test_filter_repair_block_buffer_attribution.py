"""Compiler-buffer diagnosis for the recorded D3 GPU allocator investigation."""

import hashlib
import json
import os
from pathlib import Path

from bayesfilter.inference import block_controller_tf as runtime
from tests.test_filter_repair_block_public_memory import (
    fixture,
)
from tests.test_filter_repair_block_public_memory import (
    test_complete_public_block_costs as complete_cost_check,
)


def test_public_block_gpu_buffer_attribution(monkeypatch, request):
    complete_cost_check("xla", 3, monkeypatch, request)
    root = Path(request.config.getoption("xmlpath")).parent
    owner = runtime._LAST_CONTROLLER[3]
    _, _, _, operands = fixture(3)
    hlo = owner.compiled.experimental_get_compiler_ir(*operands)(stage="optimized_hlo")
    (root / "block-optimized-hlo.txt").write_text(hlo)
    dump = root / "xla"
    memories = sorted(dump.glob("*memory-usage-report.txt"))
    assignments = sorted(dump.glob("*buffer-assignment.txt"))
    files = sorted(path for path in dump.rglob("*") if path.is_file())
    report = {"schema": "filter_block_buffer_attribution.v1", "dimension": 3,
        "xla_flags": os.environ.get("XLA_FLAGS"), "timing_eligible": False,
        "hlo_header": hlo.splitlines()[0],
        "optimized_hlo_sha256": hashlib.sha256(hlo.encode()).hexdigest(),
        "memory_reports": [str(p.relative_to(root)) for p in memories],
        "buffer_assignments": [str(p.relative_to(root)) for p in assignments],
        "files": {str(p.relative_to(root)): {"bytes": p.stat().st_size,
            "sha256": hashlib.sha256(p.read_bytes()).hexdigest()} for p in files},
        "nonclaims": ["Compiler planned buffers differ from TensorFlow allocator observations.",
            "Dumping disqualifies timing; this is separate from the frozen cost cohort."]}
    with (root / "block-buffer-attribution.json").open("x") as output:
        json.dump(report, output, indent=2, allow_nan=False)
        output.write("\n")
    assert memories and assignments, report
