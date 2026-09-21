"""Terminal M17 evidence integrity, pinned inputs and once-only accounting."""
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT.parent / "m16-r1/source-r1"
sys.path.insert(0, str(SOURCE))
sys.path.insert(0, str(ROOT.parent))
from audit_phase import PhaseAudit, read, sha


def main():
    assert os.environ.get("CUDA_VISIBLE_DEVICES") == "-1"
    started = time.monotonic()
    audit = PhaseAudit()
    identity = audit.snapshot(SOURCE)
    for name in ("matrix-cpu-r1", "matrix-gpu-r1"):
        audit.suite(ROOT / name, identity)
    for path in sorted(ROOT.glob("*-run.json")):
        audit.diagnostic(path)
    special_paths = sorted(ROOT.glob("*/diagnostic-run.json"))
    if len(special_paths) != 16:
        audit.outstanding.append("expected 12 position-field attempts and four matched-reference fits")
    pinned_inputs = []
    for path in special_paths:
        audit.diagnostic(path, artifacts=True)
        record = read(path)
        script = Path(record["command"][1])
        if sha(script) != record["script_sha256"]:
            audit.invalid.append(str(script) + ": diagnostic script changed")
        manifest = read(path.parent / "manifest.json")
        if manifest["source"]["identity"] != identity:
            audit.invalid.append(str(path) + ": wrong source")
        for name, item in manifest.get("inputs", {}).items():
            if sha(Path(item["path"])) != item["sha256"]:
                audit.invalid.append(str(path) + ": pinned input " + name)
            pinned_inputs.append({"role": name, **item})
    result = audit.finish(ROOT.parent / "m16-r1/reconciliation-terminal.json",
                         {"cpu_reference": 12000., "gpu": 10000.},
                         {"cpu_reference": 900., "gpu": 0.})
    result.update(schema="bayesfilter.hmc_m17_reconciliation.v1",
        summary_sha256=sha(ROOT / "terminal-summary.json"), pinned_inputs=pinned_inputs,
        preserved_failures="Pilot-only field attempts, engineering harness failures and posterior caps remain recorded; no replacement seeds.",
        manifest={"command": sys.argv, "environment": sys.executable, "gpu_intentionally_hidden": True,
                  "script_sha256": sha(Path(__file__)), "elapsed_seconds": time.monotonic()-started},
        interpretation="Integrity and accounting; single-fit numerical findings do not establish rankings or default readiness.")
    if not result["terminal"]:
        raise ValueError("unfinished or missing numerical workers")
    with (ROOT / "reconciliation-terminal.json").open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({k: result[k] for k in ("budget_seconds", "invalid_artifacts", "numerical_receipts_checked", "tensor_checksums_checked")}, indent=2))
    return int(bool(result["invalid_artifacts"] or result["budget_seconds"]["exceeded"]))


if __name__ == "__main__":
    raise SystemExit(main())
