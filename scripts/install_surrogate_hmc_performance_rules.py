#!/usr/bin/env python3
"""Install and test only fixed entry points for the authorized repair campaign."""
import argparse
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
import subprocess

ROOT=Path(__file__).resolve().parents[1]
PYTHON="/home/chakwong/anaconda3/envs/tftwogpu/bin/python"
RULES=Path("/home/chakwong/.codex/rules/bayesfilter-surrogate-hmc.rules")
OUT=ROOT/"docs/plans/artifacts/surrogate-hmc-performance-command-approvals-20261011-01"
ENTRY_POINTS={
    "scripts/run_surrogate_hmc_performance_repair.py":("run","worker","regression","precision","advance"),
    "scripts/run_surrogate_hmc_five_model_master.py":("audit","run","worker","campaign","stage-worker","status"),
    "scripts/install_surrogate_hmc_performance_rules.py":("preview","install","verify"),
}

def check(path):
    results=[]
    for script,actions in ENTRY_POINTS.items():
        for spelling in (script,str(ROOT/script)):
            for action in actions:
                command=[PYTHON,spelling,action]
                raw=subprocess.check_output(["codex","execpolicy","check","--rules",str(path),"--",*command],text=True)
                result=json.loads(raw)
                if result.get("decision")!="allow": raise RuntimeError("Unmatched command: "+repr(command))
                results.append(dict(command=command,decision=result["decision"]))
    return results

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action",choices=("preview","install","verify"))
    args=parser.parse_args(); OUT.mkdir(parents=True,exist_ok=True)
    old=RULES.read_text(); new=old
    for script,actions in ENTRY_POINTS.items():
        rule='prefix_rule(pattern='+json.dumps([PYTHON,[script,str(ROOT/script)],list(actions)])+', decision="allow")'
        if rule not in new: new+="\n"+rule+"\n"
    proposed=OUT/"allow-list.proposed.rules"; proposed.write_text(new)
    checked=check(RULES if args.action=="verify" else proposed)
    backup=None
    if args.action=="install" and new!=old:
        backup=RULES.with_name(RULES.name+".backup-"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ"))
        shutil.copy2(RULES,backup)
        temporary=RULES.with_suffix(".tmp"); temporary.write_text(new); temporary.replace(RULES)
        checked=check(RULES)
    report=dict(action=args.action,rules=str(RULES),backup=str(backup) if backup else None,matched_commands=checked,
        boundaries="No global sandbox or approval-policy change. Direct Python entry points only; no shell wrapper or arbitrary script prefix added.")
    (OUT/"verification.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(dict(action=args.action,matched_commands=len(checked),report=str(OUT/"verification.json"))))

if __name__=="__main__": main()
