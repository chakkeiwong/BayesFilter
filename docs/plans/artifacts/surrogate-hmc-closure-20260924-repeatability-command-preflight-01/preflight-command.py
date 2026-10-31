"""Measured dry-run and allow-list validation only; no GPU numerical execution."""
from pathlib import Path
import json
import resource
import subprocess
import sys
import time
sys.path.insert(0, "/home/chakwong/BayesFilter-ledh-graceful-failure-execution-20260917")
from bayesfilter.inference.surrogate_hmc_closure_program import preceding_charges, ROOT
from bayesfilter.inference.surrogate_hmc_program import validated_budget, run_job, write_json
forms = json.loads("{\"master\":[\"/home/chakwong/anaconda3/envs/tftwogpu/bin/python\",\"scripts/run_surrogate_hmc_program.py\",\"--stage\",\"closure\",\"--closure-stages\",\"m13-repeatability\",\"--output\",\"docs/plans/artifacts/surrogate-hmc-closure-20260922-m13-repeatability-01\"],\"dry_run\":[\"/home/chakwong/anaconda3/envs/tftwogpu/bin/python\",\"scripts/run_surrogate_hmc_program.py\",\"--stage\",\"closure\",\"--closure-stages\",\"m13-repeatability\",\"--output\",\"docs/plans/artifacts/surrogate-hmc-closure-20260922-m13-repeatability-01\",\"--dry-run\"],\"review\":[\"/home/chakwong/anaconda3/envs/tftwogpu/bin/python\",\"-m\",\"pytest\",\"-q\",\"tests/test_surrogate_hmc_closure_program.py\",\"tests/test_surrogate_hmc_m13_repeatability.py\",\"tests/test_surrogate_hmc_m13_consumer.py\",\"tests/test_surrogate_hmc_program.py\",\"tests/test_surrogate_hmc_dependency_execution.py\"],\"worker\":[\"/home/chakwong/anaconda3/envs/tftwogpu/bin/python\",\"-m\",\"bayesfilter.testing.surrogate_hmc_closure_diagnostics\",\"--stage\",\"m13-repeatability-archived-1\",\"--output\",\"docs/plans/artifacts/surrogate-hmc-closure-20260922-m13-repeatability-01/m13-repeatability-archived-1/report.json\",\"--device\",\"gpu\"],\"summary\":[\"/home/chakwong/anaconda3/envs/tftwogpu/bin/python\",\"-m\",\"bayesfilter.testing.surrogate_hmc_closure_diagnostics\",\"--stage\",\"m13-repeatability-summary\",\"--output\",\"docs/plans/artifacts/surrogate-hmc-closure-20260922-m13-repeatability-01/m13-repeatability-summary/report.json\",\"--device\",\"cpu\"]}")
output = ROOT / "docs/plans/artifacts/surrogate-hmc-closure-20260924-repeatability-command-preflight-01"
previous, _ = preceding_charges(output)
if output.exists() or previous["cpu_worker_seconds"]+120 > 54000:
    raise SystemExit("fresh output and sufficient CPU preflight budget required")
output.mkdir()
ledger = validated_budget(1.,120.)
write_json(output/"accounting.json",dict(charged_this_run=ledger["charged"], jobs=[]))
(output/"preflight-command.py").write_bytes(Path(__file__).read_bytes())
usage = resource.getrusage(resource.RUSAGE_SELF)
cpu_start, started = usage.ru_utime+usage.ru_stime, time.monotonic()
manifest = dict(command=[sys.executable,str(Path(__file__).resolve())], environment=sys.executable,
    cwd=str(ROOT), git_commit=subprocess.check_output(["git","rev-parse","HEAD"],cwd=ROOT,text=True).strip(),
    plan="docs/plans/surrogate-hmc-m13-repeatability-2026-09-24.md",
    gpu_intentionally_hidden=True, seeds="N/A; no numerical computation",
    status="RUNNING")
checks={}
try:
    for name, command in forms.items():
        check=["codex","execpolicy","check","--rules","/home/chakwong/.codex/rules/bayesfilter-surrogate-hmc.rules",*command]
        record=run_job(check,output/name,ledger,device="cpu",timeout=15.,cpu_allowance=10.)
        if record["status"]!="COMPLETED":
            raise ValueError("command policy check failed: "+name)
        decision=json.loads((output/name/"command.log").read_text())
        if decision["decision"]!="allow":
            raise ValueError("command not allowed: "+name)
        checks[name]=dict(command=command,decision=decision)
    # Preserve incremental charges before a dry run re-reads campaign receipts.
    write_json(output/"accounting.json",dict(charged_this_run=ledger["charged"],jobs=ledger["jobs"]))
    record=run_job(forms["dry_run"],output/"master-dry-run",ledger,device="cpu",timeout=30.,cpu_allowance=20.)
    if record["status"]!="COMPLETED":
        raise ValueError("master dry run failed")
    dry_run=json.loads((output/"master-dry-run/command.log").read_text())
    manifest["status"]="COMPLETED"
    write_json(output/"report.json",dict(status="COMMAND_PREFLIGHT_PASSED",checks=checks,dry_run=dry_run,
        numerical_execution_authorized=False,inference_launched=False))
finally:
    usage=resource.getrusage(resource.RUSAGE_SELF)
    ledger["charged"]["cpu_worker_seconds"]+=usage.ru_utime+usage.ru_stime-cpu_start
    accounting=dict(charged_this_run=ledger["charged"],previous_charge=previous,
        cumulative={k:previous[k]+ledger["charged"][k] for k in previous},jobs=ledger["jobs"])
    write_json(output/"accounting.json",accounting)
    manifest["wall_seconds"]=time.monotonic()-started
    write_json(output/"run-manifest.json",manifest)
    (output/"result.md").write_text("# M13 repeatability command preflight\n\n"+manifest["status"]+
        "\n\nCPU-only command checks; GPU intentionally hidden. No GPU numerical comparison was run.\n")
    print(json.dumps(dict(status=manifest["status"],charged=ledger["charged"],cumulative=accounting["cumulative"],output=str(output))))
