"""Read-only recheck of the exact consumer inputs already named by the campaign."""
import argparse
import datetime
import hashlib
import json
from pathlib import Path
import time

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
started = time.monotonic()
suffixes = ("docs/handoffs/one_country_affine_bootstrap_bayesfilter_repair_request_2026_09_22.md",
            "docs/plans/artifacts/one-country-affine-macro-option1-bootstrap-diagnostic-20260918-run01")
paths = [Path(base) / suffix for base in ("/home/ubuntu/python/MacroFinance",
         "/home/ubuntu/workspace/MacroFinance") for suffix in suffixes]
references = [Path("/home/ubuntu/python/MacroFinance/docs/plans") / name for name in (
    "daily_asset_midas_recovery_result_2026_09_18.md",
    "daily_asset_midas_phase14_experiment_design_correction_2026_09_18.md")]
result = {"read_only": True, "timestamp_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "required_bootstrap_paths": {str(p): p.exists() for p in paths},
    "midas_reference_documents": {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                                  if p.is_file() else None for p in references},
    "scope": "Exact named bundles, not an exhaustive search of MacroFinance",
    "elapsed_seconds": time.monotonic()-started}
args.output.parent.mkdir(parents=True, exist_ok=True)
with args.output.open("x") as out:
    json.dump(result, out, indent=2)
print(json.dumps(result))
