"""Prepare only rejected/unstarted rows; retain original completed evidence."""
import ast
import copy
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parent
DEST = ROOT / "control-safety-repair-config-01"
DEST.mkdir(exist_ok=False)
previous = ROOT / "control-safety-screen-01"
bundle = json.loads((ROOT / "control-safety-config-01/bundle.json").read_text())
state = json.loads((previous / "weak/state.json").read_text())
completed = {key for key, row in state["rows"].items() if row["execution_status"] == "complete"}
failed = {key for key, row in state["rows"].items() if row["execution_status"] == "failed"}
assert len(completed) == 32 and failed == {f"coordinate_cap_{cap}_{ds}" for cap in (4, 8) for ds in (700, 701)}

for item in bundle["screen"]:
    study = json.loads(Path(item["study"]).read_text())
    if item["name"] == "weak":
        study["rows"] = [row for row in study["rows"] if row["id"] in failed]
        study["required_proposals"] = ["ledh"]
        study["budget"]["max_attempts"] = len(study["rows"])
    path = DEST / (item["name"] + "-study.json")
    path.write_text(json.dumps(study, indent=2, sort_keys=True)+"\n")
    item["study"] = str(path)
bundle["carry_forward_manifest"] = str(previous / "run-manifest.json")
bundle["carry_forward_rows"] = sorted(completed)
bundle["smoke"] = []
(DEST / "bundle.json").write_text(json.dumps(bundle, indent=2, sort_keys=True)+"\n")

# The repair changes validation only; accepted old rows run identical arithmetic.
path = "bayesfilter/highdim/higher_moment_contract_e.py"
old = ast.parse(subprocess.check_output(["git", "show", "41446e1e:"+path], text=True))
new = ast.parse(Path(path).read_text())
for tree in (old, new):
    tree.body = [node for node in tree.body if not (isinstance(node, ast.Import)
                 and [name.name for name in node.names] == ["math"])]
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == "higher_moment_shape_jvp")
    assert isinstance(function.body[1], ast.If)
    assert isinstance(function.body[1].body[0], ast.Raise)
    function.body.pop(1)
assert ast.dump(old) == ast.dump(new), "repair changed more than pre-trace validation"
source_diff = subprocess.check_output(["git", "diff", "41446e1e", "--name-only", "bayesfilter"], text=True).splitlines()
assert source_diff == [path], source_diff
audit = {"validation_only_change": True, "unchanged_accepted_arithmetic": True,
    "preserved_rows": sorted(completed), "rejected_rows": sorted(failed), "new_attempts": 76,
    "cumulative_attempts_including_smoke": 120, "predecessor": "41446e1e", "source_path": path,
    "evidence_limit": "AST equality outside validation; successful old inputs keep the same TensorFlow operations"}
(DEST / "repair-audit.json").write_text(json.dumps(audit, indent=2, sort_keys=True)+"\n")
print(json.dumps({"new_attempts": 76, "preserved_rows": 32, "validation_only_change": True}))
