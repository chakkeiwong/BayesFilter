"""Snapshot a committed base plus only M30-owned edits."""
import argparse, hashlib, io, json, shutil, subprocess, tarfile
from pathlib import Path
parser=argparse.ArgumentParser();parser.add_argument("--name",required=True);args=parser.parse_args()
repo=Path.cwd();root=Path(__file__).resolve().parent;out=root/args.name;out.mkdir(exist_ok=False)
commit=subprocess.check_output(["git","rev-parse","HEAD"],text=True).strip()
owned=json.loads((root/"owned-source-paths.json").read_text())
paths=["bayesfilter","tests","pytest.ini","docs/benchmarks/audit_hmc_m27_2026_09_23.py", "docs/reference/hmc-tuning-interface.md", "docs/plans/bayesfilter-hmc-post-m29-next-phase-2026-09-23.md"]
paths += ["docs/plans/artifacts/hmc-repair-master-2026-09-16/m27-r1/"+name for name in ("m27-gpu-gaussian-lugsail-2026092283.json","m27-gpu-beta-binomial-lugsail-2026092284.json")]
raw=subprocess.check_output(["git","archive",commit,*paths])
with tarfile.open(fileobj=io.BytesIO(raw)) as archive: archive.extractall(out,filter="data")
for name in owned:
    dest=out/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(repo/name,dest)
files={str(p.relative_to(out)):hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted((out/"bayesfilter").rglob("*.py"))}
identity=hashlib.sha256(json.dumps(files,sort_keys=True,separators=(",",":")).encode()).hexdigest()
manifest={"git_commit":commit,"identity":identity,"files":files,"owned_overlays":owned,"excludes":"Unrelated dirty Q20/NeuTra code and governance edits"}
(out/"source_snapshot.json").write_text(json.dumps(manifest,indent=2)+"\n")
(root/(args.name+"-manifest.json")).write_text(json.dumps(manifest,indent=2)+"\n")
print(json.dumps({"source":str(out),"identity":identity,"owned_paths":len(owned)}))
