#!/usr/bin/env python3
"""Install only the fixed surrogate-HMC recovery command prefixes."""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
PYTHON="/home/chakwong/anaconda3/envs/tftwogpu/bin/python"
RULES=Path("/home/chakwong/.codex/rules/bayesfilter-surrogate-hmc.rules")
OUT=ROOT/"docs/plans/artifacts/surrogate-hmc-sequential-development-01"
MASTER="scripts/run_surrogate_hmc_recovery.py"
INSTALLER="scripts/install_surrogate_hmc_recovery_rules.py"
ADDITIONS=[
    'prefix_rule(pattern='+json.dumps([PYTHON,[MASTER,str(ROOT/MASTER)],["check","start","execute","status"]])+', decision="allow")',
    'prefix_rule(pattern='+json.dumps([PYTHON,[INSTALLER,str(ROOT/INSTALLER)]])+', decision="allow")',
]

def validate(path):
    checked=[]
    commands=[[PYTHON,str(ROOT/MASTER),action,"--output",str(OUT/"example")] for action in ("check","start","execute","status")]
    commands.append([PYTHON,str(ROOT/INSTALLER),"verify"])
    for command in commands:
        raw=subprocess.check_output(["codex","execpolicy","check","--rules",str(path),"--",*command],text=True)
        value=json.loads(raw)
        if value.get("decision")!="allow":
            raise ValueError("command does not match allow list: "+str(command))
        checked.append(dict(command=command,result=value))
    return checked

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("preview","install","verify"))
    args=parser.parse_args()
    OUT.mkdir(parents=True,exist_ok=True)
    if args.action=="verify":
        checked=validate(RULES)
        (OUT/"allow-list-verification.json").write_text(json.dumps(checked,indent=2)+"\n")
        print(json.dumps(dict(rules=str(RULES),matched_commands=len(checked))))
        return
    old=RULES.read_text()
    new=old
    for rule in ADDITIONS:
        if rule not in new:
            new+="\n"+rule+"\n"
    proposed=OUT/"allow-list.proposed.rules"
    proposed.write_text(new)
    checked=validate(proposed)
    if args.action=="install" and new!=old:
        backup=RULES.with_name(RULES.name+".backup-"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
        shutil.copy2(RULES,backup)
        temporary=RULES.with_suffix(".tmp")
        temporary.write_text(new)
        temporary.replace(RULES)
        print(json.dumps(dict(installed=str(RULES),backup=str(backup),added_rules=len(ADDITIONS))))
    else:
        print(json.dumps(dict(proposed=str(proposed),added_rules=sum(x not in old for x in ADDITIONS))))
    (OUT/"allow-list-verification.json").write_text(json.dumps(checked,indent=2)+"\n")

if __name__=="__main__":
    main()
