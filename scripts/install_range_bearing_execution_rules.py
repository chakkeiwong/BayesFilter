#!/usr/bin/env python3
"""Install fixed local range-bearing commands without changing sandbox policy."""
import argparse,datetime,json,shutil,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PY="/home/chakwong/anaconda3/envs/tftwogpu/bin/python"
RULES=Path("/home/chakwong/.codex/rules/bayesfilter-range-bearing.rules")
OUT=ROOT/"docs/plans/artifacts/range-bearing-execution-rules-20261010-01"
COMMANDS={
"scripts/run_range_bearing_covariance_comparison.py":["check","worker","run","report"],
"scripts/finalize_range_bearing_comparison.py":["checks","finalize","status"],
"scripts/install_range_bearing_execution_rules.py":["preview","install","verify"]}
def proposed():
 lines=["# Owner-requested grouped approvals for bounded local range-bearing work.",
 "# Exact form: interpreter -B absolute-script subcommand.",
 "# Global approval/sandbox/network policies and scientific budgets stay unchanged."]
 for relative,actions in COMMANDS.items():
  lines.append("prefix_rule(pattern="+json.dumps([PY,"-B",[relative,str(ROOT/relative)],actions])+', decision="allow")')
 return "\n".join(lines)+"\n"
def verify(path):
 checked=[]
 for relative,actions in COMMANDS.items():
  for action in actions:
   for target in (relative,str(ROOT/relative)):
    command=[PY,"-B",target,action]
    result=json.loads(subprocess.check_output(["codex","execpolicy","check","--rules",str(path),"--",*command],text=True))
    if result.get("decision")!="allow": raise ValueError("Unmatched command: "+str(command))
    checked.append(dict(command=command,decision=result["decision"]))
 for command in ([PY,"-c","print(1)"],["bash","-c","echo test"],[PY,"-B","/tmp/arbitrary.py","run"]):
  result=json.loads(subprocess.check_output(["codex","execpolicy","check","--rules",str(path),"--",*command],text=True))
  if result.get("decision")=="allow": raise ValueError("Arbitrary command matched new rules")
 return checked
def main():
 parser=argparse.ArgumentParser(description=__doc__)
 parser.add_argument("action",choices=["preview","install","verify"])
 args=parser.parse_args();OUT.mkdir(parents=True,exist_ok=True)
 candidate=OUT/"proposed.rules"
 if args.action=="verify": checked=verify(RULES)
 else:
  candidate.write_text(proposed());checked=verify(candidate)
  if args.action=="install" and (not RULES.exists() or RULES.read_text()!=proposed()):
   if RULES.exists():
    shutil.copy2(RULES,RULES.with_name(RULES.name+".backup-"+datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")))
   temporary=RULES.with_suffix(".tmp");temporary.write_text(proposed());temporary.replace(RULES)
 (OUT/(args.action+"-verification.json")).write_text(json.dumps(checked,indent=2)+"\n")
 print(json.dumps(dict(action=args.action,rules=str(RULES if args.action!="preview" else candidate),matched_commands=len(checked),negative_checks=3)))
if __name__=="__main__":main()
