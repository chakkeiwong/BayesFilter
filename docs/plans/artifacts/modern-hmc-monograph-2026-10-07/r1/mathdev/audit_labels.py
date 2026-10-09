"""Run MathDevMCP source-bound audits without its failing report projection."""
import argparse
import hashlib
import json
import os
import re
import shutil
import time
from collections import Counter
from pathlib import Path

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from mathdevmcp.latex_index import build_index
from mathdevmcp.proof_audit_v2 import audit_derivation_v2_for_label
from mathdevmcp.equation_locator import locate_equations_in_file

ROOT = Path(__file__).resolve().parents[6]
SOURCE = ROOT / "docs/chapters/ch26g_modern_hmc_methods.tex"
HERE = Path(__file__).resolve().parent
parser = argparse.ArgumentParser()
parser.add_argument("--output-tag", required=True)
args = parser.parse_args()
assert re.fullmatch(r"[A-Za-z0-9_-]+", args.output_tag)
OUTPUT = HERE / args.output_tag
OUTPUT.mkdir(exist_ok=True)
assert not (OUTPUT / "label-audits.jsonl").exists(), "Use a fresh output tag"
SNAPSHOT_DIR = OUTPUT / "source"
SNAPSHOT_DIR.mkdir(exist_ok=True)
SNAPSHOT = SNAPSHOT_DIR / SOURCE.name
shutil.copyfile(SOURCE, SNAPSHOT)
digest = hashlib.sha256(SNAPSHOT.read_bytes()).hexdigest()
index = build_index(SNAPSHOT_DIR)
equations = locate_equations_in_file(SNAPSHOT, root=SNAPSHOT_DIR)
labels = list(dict.fromkeys(x["label"] for x in equations if x.get("label")))
literal_labels = re.findall(r"\\label\{(eq:bf-modern-[^}]+)\}", SOURCE.read_text())
assert set(labels) == set(literal_labels), (set(literal_labels)-set(labels))
results = []
start = time.monotonic()
for i, label in enumerate(labels):
    result = audit_derivation_v2_for_label(
        str(SNAPSHOT_DIR), label, before=4, after=1, paragraph_context=True,
        backend="sympy", summary_only=True, file=SOURCE.name,
        source_digest=digest, index=index, task_context="symbolic_exposition",
    )
    results.append({"label":label,"result":result})
    with (OUTPUT / "label-audits.jsonl").open("a") as stream:
        stream.write(json.dumps(results[-1])+"\n")
    print(i+1, "/", len(labels), label, result.get("status"), flush=True)
    if time.monotonic()-start > 900:
        raise TimeoutError("The predeclared 15-minute per-job limit was reached")
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest() == digest
output = {
    "source":str(SOURCE), "source_sha256":digest,
    "snapshot":str(SNAPSHOT),
    "method":"MathDevMCP audit_derivation_v2_for_label; SymPy; source-bound",
    "coverage":{"literal_equation_labels":len(literal_labels),
                "localized_unique_labels":len(labels), "audited_labels":len(results)},
    "status_counts":dict(Counter(x["result"].get("status","missing") for x in results)),
    "wall_seconds":time.monotonic()-start,
    "results":results,
    "limits":"Complete label coverage is not complete proof coverage. Unsupported, "
             "unverified and inconclusive obligations require manual mathematical review.",
}
(OUTPUT / "label-audits.json").write_text(json.dumps(output,indent=2)+"\n")
print(json.dumps({k:v for k,v in output.items() if k!="results"},indent=2))
