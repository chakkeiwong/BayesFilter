"""Terminal M16 source, provenance and evidence audit; no new sampling."""
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "source-r1"))
sys.path.insert(0, str(ROOT.parent))
from audit_phase import PhaseAudit, read, sha


def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
    started = time.monotonic()
    audit = PhaseAudit()
    identity = audit.snapshot(ROOT / "source-r1")
    for name in ("boundary-pilot-gpu-r1", "sequential-pilot-cpu-r1", "sequential-pilot-gpu-r1",
                 "independent-cpu-r1", "fresh-cpu-r3", "fresh-gpu-r3"):
        audit.suite(ROOT / name, identity)
    for path in sorted(ROOT.glob("*-run.json")):
        audit.diagnostic(path)
    result = audit.finish(ROOT.parent / "m15-r1/reconciliation-terminal.json",
                         {"cpu_reference": 22000., "gpu": 15000.},
                         {"cpu_reference": 900., "gpu": 0.})
    for name in ("boundary-pilot-review-r2.json", "boundary-fresh-cpu-review.json", "boundary-fresh-gpu-review.json"):
        if read(ROOT / name)["passed"] is not True:
            result["invalid_artifacts"].append(name + ": independent reference disagreement")
    result.update(schema="bayesfilter.hmc_m16_reconciliation.v1",
        summary_sha256=sha(ROOT / "terminal-summary.json"),
        preserved_diagnostic_failure={"path": str(ROOT / "boundary-pilot-review.json"),
            "classification": "partial diagnostic JSON after NumPy boolean serialization; excluded",
            "repair": "stdlib NormalDist and encode-before-open, successful review-r2; no numerical rerun"},
        manifest={"command": sys.argv, "environment": sys.executable, "gpu_intentionally_hidden": True,
            "script_sha256": sha(Path(__file__)), "elapsed_seconds": time.monotonic()-started},
        interpretation="Integrity and accounting only; limited SBC power and compatibility-screen behavior remain explicit results.")
    if not result["terminal"]:
        raise ValueError("unfinished numerical workers")
    with (ROOT / "reconciliation-terminal.json").open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({k: result[k] for k in ("budget_seconds", "invalid_artifacts", "numerical_receipts_checked", "tensor_checksums_checked")}, indent=2))
    return int(bool(result["invalid_artifacts"] or result["budget_seconds"]["exceeded"]))


if __name__ == "__main__":
    raise SystemExit(main())
