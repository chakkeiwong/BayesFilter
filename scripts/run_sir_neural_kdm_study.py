"""Budgeted supervisor for the reviewed SIR neural moment-training diagnostic."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time


def save(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def recover_screen(study, allowance, now=None):
    previous = json.loads((study / "ledger.json").read_text())
    expected = {"evaluation-probe"} | {
        f"screen-k{components}-w{width}-beta{beta}"
        for components, width in ((2, 16), (4, 32)) for beta in (.5, .9)}
    if previous["allowance_seconds"] != allowance:
        raise ValueError("Recovery must preserve the original campaign allowance")
    if {row["name"] for row in previous["runs"]} != expected or len(previous["runs"]) != 5:
        raise ValueError("Recovery requires exactly the declared completed screening processes")
    charged = 0
    records = []
    for original in previous["runs"]:
        record = dict(original)
        result_path = study / record["name"] / "result.json"
        if not result_path.exists():
            raise ValueError("Screen process has not produced its completed result")
        result = json.loads(result_path.read_text())
        expected_status = "probe_complete" if record["name"] == "evaluation-probe" else "family_complete"
        if result.get("status") != expected_status:
            raise ValueError("Screen result is incomplete")
        if not result["source_check"]["unchanged"]:
            raise ValueError("Screen source changed")
        if record["status"] == "running":
            process_state = Path(f"/proc/{record['pid']}/stat")
            if process_state.exists() and process_state.read_text().rsplit(")", 1)[1].split()[0] != "Z":
                raise ValueError("Screen process is still alive; do not duplicate it")
            started = datetime.strptime(record["start_utc"], "%Y-%m-%dT%H:%M:%SZ").replace(
                tzinfo=timezone.utc).timestamp()
            elapsed = (time.time() if now is None else now) - started
            if elapsed < 0.:
                raise ValueError("Recovery clock precedes the interrupted process")
            record.update(status="complete", returncode=None, wall_seconds=elapsed,
                charged_seconds=math.ceil(elapsed),
                recovery="Completed artifact; original exit code unavailable. Conservative charge through recovery.")
        elif record["status"] != "complete" or record["returncode"] != 0:
            raise ValueError("Failed screening evidence cannot be silently recovered")
        charged += record["charged_seconds"]
        record["evidence_root"] = str(study)
        records.append(record)
    if charged >= allowance:
        raise TimeoutError("No allowance remains after charging interrupted screening")
    return records, charged


def recover_completed_study(study, allowance):
    previous = json.loads((study / "ledger.json").read_text())
    if previous["allowance_seconds"] != allowance:
        raise ValueError("Recovery must preserve the original campaign allowance")
    records = [dict(record) for record in previous["recovered_runs"]]
    records.extend(dict(record, evidence_root=str(study)) for record in previous["runs"])
    expected = {"evaluation-probe", "rung128-seed263044001", "rung128-seed263044002"} | {
        f"screen-k{components}-w{width}-beta{beta}"
        for components, width in ((2, 16), (4, 32)) for beta in (.5, .9)}
    completed = {}
    for record in records:
        root = Path(record["evidence_root"]) / record["name"]
        if record["status"] == "failed" and record["name"].startswith("rung512-"):
            failure = json.loads((root / "failure.json").read_text())
            if failure["error"] != "TimeoutError('Neural KDM campaign process budget exhausted')" \
                    or not failure["source_check"]["unchanged"]:
                raise ValueError("Only a source-unchanged optional-rung budget failure is recoverable")
            continue
        if record["status"] != "complete" or record["name"] not in expected:
            raise ValueError("Unexpected or incomplete process in recovery")
        result = json.loads((root / "result.json").read_text())
        status = "fit_complete" if record["name"].startswith("rung128-") else (
            "probe_complete" if record["name"] == "evaluation-probe" else "family_complete")
        if result["status"] != status or not result["source_check"]["unchanged"]:
            raise ValueError("Completed recovery evidence is invalid")
        if record["name"] in completed:
            raise ValueError("Duplicate completed process in recovery")
        completed[record["name"]] = root / "result.json"
    if set(completed) != expected:
        raise ValueError("Recovery needs the complete screen and both 128-update continuations")
    charged = sum(record["charged_seconds"] for record in records)
    if charged != previous["charged_seconds"] or charged >= allowance:
        raise ValueError("Recovery accounting is inconsistent or exhausted")
    return records, charged, completed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--budget-seconds", type=float, default=22051.197895057)
    parser.add_argument("--final-reserve", type=float, default=4000.)
    parser.add_argument("--repair-reserve", type=float, default=1000.)
    parser.add_argument("--screen-study", type=Path)
    parser.add_argument("--finalize-study", type=Path)
    arguments = parser.parse_args()
    if arguments.screen_study is not None and arguments.finalize_study is not None:
        parser.error("Choose one recovery stage")
    repository = Path(__file__).resolve().parents[1]
    output = arguments.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    ledger = dict(allowance_seconds=arguments.budget_seconds, charged_seconds=0., runs=[],
                  final_reserve=arguments.final_reserve, repair_reserve=arguments.repair_reserve,
                  command=sys.argv, python=sys.executable,
                  supervisor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  plan="docs/plans/ledh-sir-neural-kdm-adaptation-20261020.md")
    evidence_roots = [output]
    recovered = {}
    if arguments.screen_study is not None:
        previous = arguments.screen_study.resolve()
        records, charged = recover_screen(previous, arguments.budget_seconds)
        ledger.update(charged_seconds=charged, recovered_runs=records,
                      recovered_screen=str(previous))
        evidence_roots.insert(0, previous)
        recovered = {record["name"]: previous / record["name"] / "result.json" for record in records}
    if arguments.finalize_study is not None:
        previous = arguments.finalize_study.resolve()
        records, charged, recovered = recover_completed_study(previous, arguments.budget_seconds)
        ledger.update(charged_seconds=charged, recovered_runs=records,
                      recovered_study=str(previous), optional_rung_incomplete=True)
        evidence_roots = [path.parent for path in recovered.values()] + [output]
    save(output / "ledger.json", ledger)

    def remaining():
        return arguments.budget_seconds - ledger["charged_seconds"]

    def run(name, mode, configuration, cap, updates=32, resume=None):
        if name in recovered:
            return json.loads(recovered[name].read_text())
        if cap <= 0. or cap > remaining():
            raise ValueError("Launch exceeds remaining process budget")
        command = [sys.executable, "-m", "bayesfilter.score_study.neural_kdm_study",
                   "--mode", mode, "--output", str(output / name), "--updates", str(updates),
                   "--budget-seconds", str(max(1., cap - 15.))]
        for key, value in configuration.items():
            command.extend(["--" + key.replace("_", "-"), str(value)])
        if resume is not None:
            command.extend(["--resume", str(resume)])
        record = dict(name=name, command=command, cap_seconds=cap, status="running",
                      start_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
        ledger["runs"].append(record)
        save(output / "ledger.json", ledger)
        begin = time.monotonic()
        with (output / f"{name}.log").open("w") as log:
            process = subprocess.Popen(command, cwd=repository, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            record["pid"] = process.pid
            save(output / "ledger.json", ledger)
            try:
                returncode = process.wait(timeout=cap)
            except subprocess.TimeoutExpired:
                import signal
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=10.)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                returncode = 124
        elapsed = time.monotonic() - begin
        charged = math.ceil(elapsed)
        ledger["charged_seconds"] += charged
        record.update(wall_seconds=elapsed, charged_seconds=charged, returncode=returncode,
                      status="complete" if returncode == 0 else "failed")
        save(output / "ledger.json", ledger)
        print(json.dumps(dict(name=name, returncode=returncode, elapsed=elapsed, remaining=remaining())), flush=True)
        if returncode != 0:
            raise RuntimeError(f"{name} failed; preserve evidence and inspect {name}.log")
        result = json.loads((output / name / "result.json").read_text())
        if not result["source_check"]["unchanged"]:
            raise ValueError("Source changed during a study process")
        return result

    def checkpoints():
        candidates = []
        for path in (path for root in evidence_roots for path in root.glob("**/checkpoint-*.json")):
            candidate = json.loads(path.read_text())
            if candidate["updates"] > 0 and candidate["validation"]["valid"]:
                candidates.append(dict(path=str(path), configuration=candidate["configuration"],
                    updates=candidate["updates"], validation_loss=candidate["validation"]["loss"]))
        return candidates

    try:
        configuration = dict(components=2, width=16, beta=.5, learning_rate=.001, seed=263044001)
        probe = run("evaluation-probe", "probe", configuration, 600.)
        warm_forward = statistics.mean(row["wall_seconds"] for row in probe["rows"][1:])
        ledger["warm_forward_seconds"] = warm_forward
        for components, width in ((2, 16), (4, 32)):
            for beta in (.5, .9):
                available = remaining() - arguments.final_reserve - arguments.repair_reserve
                projected = 128 * 25. + 512 * warm_forward + 400.
                if available < projected:
                    raise RuntimeError("Insufficient budget to complete the declared capacity/rate screen")
                run(f"screen-k{components}-w{width}-beta{beta}", "family",
                    dict(configuration, components=components, width=width, beta=beta),
                    min(available, max(4500., projected)))
        candidates = checkpoints()
        nominees = []
        for seed in (263044001, 263044002):
            seeded = [candidate for candidate in candidates if candidate["configuration"]["seed"] == seed]
            if seeded:
                nominees.append(min(seeded, key=lambda candidate: candidate["validation_loss"]))
        save(output / "nominations-0032.json", nominees)
        for nominee in nominees:
            available = remaining() - arguments.final_reserve - arguments.repair_reserve
            projected = 96 * 25. + 64 * warm_forward + 400.
            if available < projected:
                raise RuntimeError("Insufficient budget for the declared 128-update rung")
            result = run(f"rung128-seed{nominee['configuration']['seed']}", "fit",
                nominee["configuration"], min(available, max(3200., projected)),
                updates=128, resume=nominee["path"])
            nominee["continuation_status"] = result["status"]
        candidates = checkpoints()
        rung128 = [candidate for candidate in candidates if candidate["updates"] == 128]
        warm_updates = []
        for path in output.glob("rung128-*/result.json"):
            warm_updates.extend(json.loads(path.read_text()).get("warm_update_seconds", []))
        warm_update = statistics.mean(warm_updates) if warm_updates else 25.
        projected512 = len(rung128) * (384 * warm_update * 1.2 + 64 * warm_forward + 400.)
        extend = arguments.finalize_study is None and bool(rung128) and projected512 <= remaining() - arguments.final_reserve - arguments.repair_reserve
        ledger["rung512"] = dict(executed=extend, projected_seconds=projected512,
                                reason="fits measured budget" if extend else "preserve final evaluation and repair reserve",
                                prior_optional_rung_incomplete=arguments.finalize_study is not None)
        save(output / "ledger.json", ledger)
        if extend:
            for nominee in rung128:
                cap = 384 * warm_update * 1.2 + 64 * warm_forward + 400.
                run(f"rung512-seed{nominee['configuration']['seed']}", "fit", nominee["configuration"],
                    cap, updates=512, resume=nominee["path"])
        candidates = checkpoints()
        if not candidates:
            raise RuntimeError("No valid trained candidate; no holdout will be opened")
        selected = min(candidates, key=lambda candidate: candidate["validation_loss"])
        save(output / "frozen-selection.json", dict(selected=selected, candidates=candidates,
            holdout_opened=False, selection="validation only", screen_complete=True,
            rung128_complete=len(rung128) == len(nominees), rung512_executed=extend))
        result = run("final-holdout", "holdout", selected["configuration"],
                     remaining() - arguments.repair_reserve, resume=selected["path"])
        save(output / "result.json", dict(status="complete", selected=selected,
            remaining_seconds=remaining(), charged_seconds=ledger["charged_seconds"],
            holdout=result, score_accuracy_evaluated=False))
    except Exception as error:
        save(output / "failure.json", dict(error=repr(error), remaining_seconds=remaining(),
                                          ledger=str(output / "ledger.json")))
        raise


if __name__ == "__main__":
    main()
