"""Freeze committed HMC sources plus explicitly owned overlays for this roadmap."""
import argparse
import hashlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import tarfile

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--output", required=True, type=Path)
parser.add_argument("--overlay", action="append", default=[])
parser.add_argument("--include-docs", action="store_true")
args = parser.parse_args()
repo = Path.cwd()
out = args.output.resolve()
out.mkdir(parents=True, exist_ok=False)
commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
paths = ["bayesfilter", "tests", "pytest.ini",
    "scripts", "docs/benchmarks/audit_hmc_m27_2026_09_23.py",
    "docs/benchmarks/audit_hmc_m29_2026_09_23.py", "docs/benchmarks/audit_hmc_m30_2026_09_23.py"]
if args.include_docs:
    paths += ["AGENTS.md", "docs/main.tex", "docs/preamble.tex", "docs/references.bib",
              "docs/chapters", "docs/appendices", "docs/generated", "docs/examples",
              "docs/reference", "docs/benchmarks"]
raw = subprocess.check_output(["git", "archive", commit, *paths])
with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
    archive.extractall(out, filter="data")
for name in args.overlay:
    path = Path(name)
    if path.is_absolute() or ".." in path.parts or not (repo / path).is_file():
        raise ValueError("overlay must name an existing repository-relative file")
    dest = out / path
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(repo / path, dest)
hashes = {str(p.relative_to(out)): hashlib.sha256(p.read_bytes()).hexdigest()
          for p in sorted(out.rglob("*.py"))}
identity = hashlib.sha256(json.dumps(hashes, sort_keys=True).encode()).hexdigest()
(out / "source_snapshot.json").write_text(json.dumps({"git_commit": commit,
    "identity": identity, "owned_overlays": args.overlay, "files": hashes,
    "excludes": "unrelated dirty Q20, training and governance edits"}, indent=2) + "\n")
print(json.dumps({"source": str(out), "identity": identity}))
