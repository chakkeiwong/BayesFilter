"""Recover completed author training after interrupted source-variability supervision.

Diagnostic/reference recovery only. Does not rerun training or change numerical
audit checks, evaluation metrics, seeds or scientific acceptance criteria.
"""
import ast
from datetime import datetime, timezone
import difflib
import json
import os
from pathlib import Path
import sys
import time
import traceback

import run_kdm_source_variability as phase

ROOT = phase.ROOT
OUT = phase.BASE / "kdm-source-variability-20261008-01"
RECOVERY = OUT / "recovery-20261008-01"
PLAN = "docs/plans/controlled-smc-kdm-source-variability-recovery-20261008.md"
ABSENT_AT = datetime(2026, 10, 8, 13, 6, 51, tzinfo=timezone.utc).timestamp()

def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError("Unexpected recovery edit match count: " + old[:80])
    return text.replace(old, new, 1)

def recovered_sources():
    """Return declared infrastructure-only edits; original files stay unchanged."""
    audit_original = (ROOT / phase.AUDIT).read_text()
    eval_original = (ROOT / phase.EVAL).read_text()
    root_line = "ROOT = Path(__file__).resolve().parents[2]"
    fixed_root = "ROOT = Path(" + repr(str(ROOT)) + ")"
    audit = replace_once(audit_original, root_line, fixed_root)
    evaluator = eval_original  # Already uses the fixed repository root.
    recovery_check = '''
def validate_recovered_completion(manifest, diagnostic):
    proof = manifest["completion_recovery"]
    archived = Path(proof["archived_manifest"])
    assert manifest["returncode"] is None
    assert manifest["process_returncode_observed"] is False
    assert manifest["completion_basis"] == "recovered_complete_artifacts_exit_unobserved"
    assert sha(archived) == proof["archived_manifest_sha256"]
    old = read(archived)
    assert old["status"] == "RUNNING" and "returncode" not in old
    assert old["hashes"] == manifest["hashes"]
    assert old["child_pid"] == manifest["child_pid"]
    assert proof["supervisor_tool_exit_code"] == 143
    assert proof["processes_absent_at_recovery"] is True
    assert sha(diagnostic/"result.json") == proof["result_sha256"]
    assert sha(diagnostic/"progress.json") == proof["terminal_progress_sha256"]
    assert read(diagnostic/"progress.json") == {"terminal": True, "status": "PASS"}
    result = read(diagnostic/"result.json")
    assert result["status"] == "PASS" and result["source_training_protocol_complete"]
    return True

'''
    audit = replace_once(audit, "def main():", recovery_check + "def main():")
    audit = replace_once(audit,
        'check(manifest["status"] == "COMPLETED" and manifest["returncode"] == 0, "terminal_success")',
        'check(manifest["status"] == "COMPLETED" and (manifest["returncode"] == 0 or '
        'validate_recovered_completion(manifest, diagnostic)), "terminal_success_or_verified_artifact_recovery")')
    audit = replace_once(audit,
        '    args.output.write_text(json.dumps(payload,indent=2,allow_nan=False)+"\\n")',
        '    payload["completion_recovery"] = dict(validated=manifest["returncode"] is None,\n'
        '        process_returncode_observed=manifest.get("process_returncode_observed", True),\n'
        '        result_sha256=manifest["result_sha256"], basis=manifest.get("completion_basis", "observed_exit"))\n'
        '    args.output.write_text(json.dumps(payload,indent=2,allow_nan=False)+"\\n")')
    evaluator = replace_once(evaluator,
        'assert audited_manifest["status"] == "COMPLETED" and audited_manifest["returncode"] == 0',
        'assert audited_manifest["status"] == "COMPLETED"\n'
        '        assert (audited_manifest["returncode"] == 0 or (\n'
        '            audited_manifest["returncode"] is None\n'
        '            and audited_manifest.get("process_returncode_observed") is False\n'
        '            and audit.get("completion_recovery", {}).get("validated") is True\n'
        '            and audit["completion_recovery"]["result_sha256"] == audited_manifest["result_sha256"]))')
    ast.parse(audit)
    ast.parse(evaluator)
    return [(phase.AUDIT, audit_original, audit, "author_terminal_audit.py"),
            (phase.EVAL, eval_original, evaluator, "development_evaluation.py")]

def main():
    if sys.argv[1:] == ["--check"]:
        changes = recovered_sources()
        print(json.dumps(dict(status="PASS",helpers=len(changes),
            changed_source_lines=[len(list(difflib.unified_diff(a.splitlines(),b.splitlines())))
                                  for _,a,b,_ in changes])))
        return 0
    if sys.argv[1:]:
        raise ValueError("Use no arguments to execute the fixed recovery")
    if RECOVERY.exists():
        raise RuntimeError("Recovery already attempted; inspect its preserved records")
    manifest = phase.read(OUT/"manifest.json")
    full_path = OUT/"unit01-full/manifest.json"
    full = phase.read(full_path)
    result_path = full_path.parent/"diagnostic/result.json"
    progress_path = full_path.parent/"diagnostic/progress.json"
    result, progress = phase.read(result_path), phase.read(progress_path)
    if manifest["status"] != "RUNNING" or full["status"] != "RUNNING":
        raise ValueError("Unexpected campaign state")
    for pid in (manifest["supervisor_pid"], full["child_pid"]):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            pass
        else:
            raise RuntimeError("Original process still exists: " + str(pid))
    for record in (manifest, full):
        for path,digest in record["hashes"].items():
            if phase.sha(ROOT/path) != digest:
                raise ValueError("Frozen launch dependency changed: " + path)
    assert result["status"] == "PASS" and result["source_training_protocol_complete"]
    assert result["constructor_seed_base"] == phase.SEEDS[0]
    assert progress == {"terminal":True, "status":"PASS"}
    assert [s["stage"] for s in result["stages"]] == ["weights","mdpf","am_dpf"]
    assert all(s["full_protocol"] and s["termination"] == "SOURCE_EARLY_STOP"
               and 0 < s["completed_epochs"] <= s["source_cap"] for s in result["stages"])
    units = list(manifest["units"])
    assert len(units) == 1 and units[0]["unit"] == 2 and units[0]["status"] == "COMPLETED"
    assert phase.sha(units[0]["evaluation"]) == units[0]["evaluation_sha256"]
    assert phase.verify_evaluation(Path(units[0]["evaluation"])) == 16
    changes = recovered_sources()
    upper_wall = ABSENT_AT - full["started_unix_seconds"]
    assert upper_wall > sum(s["wall_seconds"] for s in result["stages"])
    assert result_path.stat().st_mtime < ABSENT_AT
    # Existing phase overhead includes 120 spare seconds; use 60 for recovery.
    phase.ROUTINE = 300.0
    prospective = phase.ROUTINE + upper_wall + sum(
        phase.read(p).get("wall_seconds", 0.0)
        for p in OUT.glob("unit*-*/manifest.json") if p != full_path)
    if prospective + 420 > phase.CAP:
        raise RuntimeError("Recovery cannot fit the original phase cap")
    RECOVERY.mkdir()
    archived = RECOVERY/"unit01-full-manifest-before-recovery.json"
    archived.write_bytes(full_path.read_bytes())
    (RECOVERY/"campaign-manifest-before-recovery.json").write_bytes((OUT/"manifest.json").read_bytes())
    (RECOVERY/"status-before-recovery.json").write_bytes((phase.BASE/"status.json").read_bytes())
    (RECOVERY/"checkpoint-before-recovery.md").write_bytes(
        (ROOT/"docs/plans/controlled-smc-kdm-checkpoint.md").read_bytes())
    helper_records=[]
    for original, before, after, filename in changes:
        dest=RECOVERY/filename
        dest.write_text(after)
        (RECOVERY/(filename+".diff")).write_text("".join(difflib.unified_diff(
            before.splitlines(True),after.splitlines(True),fromfile=original,tofile=str(dest))))
        helper_records.append(dict(original=original,original_sha256=phase.sha(ROOT/original),
                                   recovery=str(dest),recovery_sha256=phase.sha(dest)))
    proof=dict(archived_manifest=str(archived),archived_manifest_sha256=phase.sha(archived),
        result_sha256=phase.sha(result_path),terminal_progress_sha256=phase.sha(progress_path),
        supervisor_tool_exit_code=143,signal_sender="UNKNOWN",
        processes_absent_at_recovery=True,first_verified_absent_utc="2026-10-08T13:06:51+00:00",
        recovered_utc=phase.stamp(),plan=PLAN,helpers=helper_records)
    full.update(status="COMPLETED",returncode=None,process_returncode_observed=False,
        completion_basis="recovered_complete_artifacts_exit_unobserved",completion_recovery=proof,
        result_path=str(result_path),result_sha256=phase.sha(result_path),wall_seconds=upper_wall,
        wall_seconds_basis="Conservative upper bound: launch to first verified process absence",
        terminal_artifact_utc=datetime.fromtimestamp(result_path.stat().st_mtime,timezone.utc).isoformat())
    phase.save(full_path,full)
    manifest.update(supervisor_pid=os.getpid(),routine_cpu_seconds=phase.ROUTINE,
        completion_recovery=proof,last_update_utc=phase.stamp())
    phase.save(OUT/"manifest.json",manifest)
    phase.save(RECOVERY/"recovery.json",dict(status="RUNNING",plan=PLAN,proof=proof,
        command=[sys.executable]+sys.argv,git_commit=manifest["git_commit"],
        started_utc=phase.stamp(),supervisor_pid=os.getpid(),training_repeated=False,
        phase_cap_seconds=phase.CAP,recovery_max_child_seconds=420,
        additional_routine_seconds=60,script_sha256=phase.sha(__file__)))
    phase.PLAN=PLAN
    phase.AUDIT=str((RECOVERY/"author_terminal_audit.py").relative_to(ROOT))
    phase.EVAL=str((RECOVERY/"development_evaluation.py").relative_to(ROOT))
    phase.refresh()
    started=time.monotonic()
    error=None
    try:
        audit_path=phase.run_step(OUT,1,phase.SEEDS[0],"audit",120,Path(full["reference"]))
        print(json.dumps(dict(stage="RECOVERED_AUDIT_PASS",result=str(audit_path))),flush=True)
        evaluation=phase.run_step(OUT,1,phase.SEEDS[0],"evaluation",300,Path(full["reference"]))
        controls=phase.verify_evaluation(evaluation)
        units.append(dict(unit=1,seed=phase.SEEDS[0],status="COMPLETED",
            evaluation=str(evaluation),evaluation_sha256=phase.sha(evaluation),
            fixed_tf_replay_controls=controls,completion_basis=full["completion_basis"]))
    except Exception:
        error=traceback.format_exc()
        units.append(dict(unit=1,seed=phase.SEEDS[0],status="FAILED",error=error))
    units.sort(key=lambda row:row["unit"])
    rc=phase.finish(OUT,manifest,units,error=error)
    report=phase.read(RECOVERY/"recovery.json")
    report.update(status="COMPLETED" if rc==0 else "FAILED",ended_utc=phase.stamp(),
                  wall_seconds=time.monotonic()-started,returncode=rc,error=error)
    phase.save(RECOVERY/"recovery.json",report)
    note=ROOT/phase.RESULT
    text=note.read_text().replace("plus 240 routine seconds", "plus 300 routine seconds")
    text += ("\nSupervision recovery: the original supervisor exited 143. Unit 1's complete\n"
             "artifacts were independently audited; its process exit code remains unobserved.\n"
             "Its CPU charge conservatively includes launch to first verified process absence,\n"
             "so it overstates measured training time. Original manifests and helper edits are\n"
             "preserved in recovery-20261008-01. No training was repeated or seed replaced.\n"
             "Recovery plan: [bounded recovery](controlled-smc-kdm-source-variability-recovery-20261008.md).\n")
    note.write_text(text)
    print(json.dumps(dict(recovery_status=report["status"],phase_returncode=rc,
                          recovery=str(RECOVERY/"recovery.json"))),flush=True)
    return rc

if __name__=="__main__":
    raise SystemExit(main())
