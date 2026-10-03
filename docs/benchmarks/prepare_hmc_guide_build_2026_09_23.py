"""Build-only snapshot with referenced committed figures and owned guide edits."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
repo = Path(__file__).resolve().parents[2]
snapshot = json.loads((args.source / "source_snapshot.json").read_text())
shutil.copytree(args.source / "docs", args.output)
assets, missing = {}, []
for tex in args.output.rglob("*.tex"):
    for name in re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", tex.read_text()):
        target = args.output / name
        if target.is_file():
            continue
        relative = "docs/" + name
        result = subprocess.run(["git", "show", snapshot["git_commit"]+":"+relative],
                                cwd=repo, capture_output=True)
        if result.returncode:
            missing.append(relative)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(result.stdout)
        assets[relative] = hashlib.sha256(result.stdout).hexdigest()
manifest = {"base_commit": snapshot["git_commit"], "guide_source": str(args.source),
            "assets_from_committed_base": assets, "unresolved_literal_paths": sorted(set(missing)),
            "excludes": "concurrent chapter26b/bibliography edits"}
(args.output / "guide-build-inputs.json").write_text(json.dumps(manifest, indent=2)+"\n")
print(json.dumps({"committed_assets": len(assets), "unresolved": manifest["unresolved_literal_paths"]}))
