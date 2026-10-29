"""Metadata-only KDM source audit and official config generation; no training."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import yaml

PIN = "b0e2fd54db7b6c36d70e8e701ddc6a3f3d5dee18"
BASE = "src/experiments/bearings_only_experiments_methodical_25_particles_multiple_runs"
TEMPLATES = ("our_experiments.yaml", "comparison_experiments.yaml",
             "our_experiments_exp3_init.yaml", "lstm_experiments.yaml",
             "comparison_experiments_always_on_override.yaml")

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",required=True,type=Path)
    ap.add_argument("--output",required=True,type=Path)
    a=ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"]="-1"
    source=a.root/".localresources/code/younis-mdpf-neurips-2023"
    head=subprocess.check_output(["git","rev-parse","HEAD"],cwd=source,text=True).strip()
    assert head==PIN
    tree=subprocess.check_output(["git","ls-tree","-r","-z",PIN],cwd=source)
    blobs={}
    for row in tree.split(b"\0"):
        if row:
            left,path=row.split(b"\t",1)
            blobs[path.decode()]=left.split()[2].decode()
    wanted=["README.md","src/datasets/toy_problem_dataset.py",
            "src/datasets/bearings_only_dataset.py","src/models/kde_particle_filter/kde_particle_filter.py",
            BASE+"/scripts/generate_multiple_runs_config_files.py",
            BASE+"/experiment0002_importance/config.yaml",BASE+"/experiment0003_importance_init/config.yaml"]
    wanted += [str(p.relative_to(source)) for p in (source/BASE/"configs").glob("*.yaml")]
    a.output.mkdir(parents=True,exist_ok=True)
    snapshot=a.output/"author-source"
    assert not snapshot.exists(),"fresh snapshot required"
    rows=[]
    for name in sorted(set(wanted)):
        p=source/name
        raw=p.read_bytes()
        blob=hashlib.sha1(b"blob "+str(len(raw)).encode()+b"\0"+raw).hexdigest()
        assert blob==blobs[name],"modified author file: "+name
        dest=snapshot/name
        dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes(raw)
        rows.append(dict(path=name,sha256=sha(dest),git_blob=blob))
    gen=snapshot/BASE/"scripts/generate_multiple_runs_config_files.py"
    funcs=runpy.run_path(str(gen)) # Inspected, pinned author generator; no main().
    dest=snapshot/BASE/"configs/generated"
    dest.mkdir()
    generated=[]
    for name in TEMPLATES:
        template=snapshot/BASE/"configs"/name
        funcs["process_config_file"](str(template),11,str(dest))
        data=yaml.safe_load((dest/name).read_text())
        original=yaml.safe_load(template.read_text())
        assert len(data["experiments"])==11*len(original["experiments"])
        ids=[next(iter(x)) for x in data["experiments"]]
        assert len(ids)==len(set(ids))
        assert set(int(x.rsplit("_",1)[1]) for x in ids)==set(range(11))
        generated.append(dict(path=str((dest/name).relative_to(a.output)),
            sha256=sha(dest/name),trials=11,tasks=len(ids)))
    summary=dict(status="PASS",scope="KDM_SOURCE_METADATA_AND_CONFIG_GENERATION",
        replication="INCOMPLETE",source_specification="PARTIAL",
        cpu_only=True,gpu_intentionally_hidden=True,training_executed=False,
        commit=PIN,verified_files=rows,generated_configs=generated,pyyaml=yaml.__version__,
        gaps=[
          "Appendix B driver and numerical initialization/settings not located",
          "Generated bearings tasks retain original do_run flags; training is not enabled by this metadata action",
          "Component pretraining, MDPF-before-A-MDPF checkpoints and independent trials remain to be executed",
          "NLL and MSE training stages have separate claim boundaries",
          "Full empirical training allocation and TensorFlow learning parity remain unresolved"],
        scientific_verdict="Configuration preparation only; no training or filtering replication",
        checks=[dict(name="pinned_source_files_and_eleven_trial_config_generation",passed=True)])
    (a.output/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
    print(json.dumps(dict(status=summary["status"],files=len(rows),generated=len(generated),
                         training_executed=False)))
if __name__=="__main__":main()
