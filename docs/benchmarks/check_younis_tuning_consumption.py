"""CPU-only diagnostic against an actual saved tuning/claim plumbing study."""
import json
import os
from pathlib import Path
import sys
from copy import deepcopy

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["BAYESFILTER_PRELOAD_CUSTOM_OP"] = "0"
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from bayesfilter.score_study.tuning import consume_selection

study = json.loads(Path(sys.argv[1]).read_text())
output = Path(sys.argv[2])
output.mkdir(parents=True, exist_ok=False)
row = study["rows"][0]
original_selection = json.loads(Path(row["tuning_selection"]).read_text())
controls = consume_selection(row, {"study": study})
assert controls == original_selection["selected_controls"]
checks = {"actual_issued_selection_consumed": True}

def rejected(label, modified_row, modified_study):
    try:
        consume_selection(modified_row, {"study": modified_study})
    except (ValueError, KeyError, OSError) as error:
        checks[label] = {"rejected": True, "reason": str(error)}
    else:
        raise AssertionError(label + " was admitted")

for key, value in [("horizon", 4), ("particles", 16), ("dtype", "float32")]:
    modified = deepcopy(study)
    modified["settings"][key] = value
    rejected(key + "_scope_mismatch", row, modified)
rejected("claim_data_leak", {**row, "dataset": study["partitions"]["calibration"][0]}, study)
modified_selection = deepcopy(original_selection)
modified_selection["selected_controls"]["flow_substeps"] += 1
changed_path = output / "caller-changed-selection.json"
changed_path.write_text(json.dumps(modified_selection))
rejected("caller_changed_controls", {**row, "tuning_selection": str(changed_path)}, study)
modified = deepcopy(study)
modified["evidence_class"] = "research"
rejected("mechanics_to_scientific_promotion", row, modified)
(output / "checks.json").write_text(json.dumps(checks, indent=2) + "\n")
print(json.dumps({"passed": len(checks), "output": str(output)}))
