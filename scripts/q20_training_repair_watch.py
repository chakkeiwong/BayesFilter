#!/usr/bin/env python3
"""Keep the authorized frozen repair running and write its terminal summary.

No numerical imports, settings changes or budget expansion. The campaign lock
and existing attempt/cumulative limits remain owned by the master.
"""
import argparse
from datetime import datetime
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import time


def summary(root):
    result = json.loads((root/"result.json").read_text())
    rows = ["# q20 training repair: completed tranche", "",
        "This is training-development evidence. No posterior estimate, calibrated recipe or statistically supported ranking is established.", "",
        "| Arm | Status | Added updates | Clipped fraction | Paired loss change | Final-bank change |",
        "| --- | --- | ---: | ---: | ---: | ---: |"]
    def interval(value):
        return "not assessed" if not value else f"{value['mean']:.5g} ± {value['half_width']:.4g}"
    for arm in ("control", "clip", "depth", "batch128"):
        item = result["arms"].get("continue-"+arm)
        if item is None:
            rows.append(f"| {arm} | deferred after sanity screen | — | — | — | — |")
            continue
        assessment = item.get("assessment", {})
        clipping = item.get("clipped_fraction")
        clip_text = "not assessed" if clipping is None else f"{clipping:.2%}"
        rows.append(f"| {arm} | {item['status']} | {item.get('updates', '—')} | {clip_text} | "
                    f"{interval(assessment.get('increment'))} | {interval(assessment.get('untouched_final_increment'))} |")
    rows += ["", "Intervals and differences are descriptive. Training streams and reused banks do not support a superiority claim.", "",
        "| Arm | Points | Post-training numerical check | Repair signals | Next action |",
        "| --- | ---: | --- | --- | --- |"]
    for arm in ("control", "clip", "depth", "batch128"):
        item = result["arms"].get("continue-"+arm, {})
        report = item.get("assessment", {}).get("post_training", {})
        checked = report.get("numerical_check_passed")
        check_text = "not checked" if checked is None else "passed" if checked else "failed"
        points = report.get("geometry", {}).get("rows", "not recorded")
        if checked and points != 1000:
            check_text = "short diagnostic only"
        signals = ", ".join(report.get("repair_triggers", [])) or "none recorded"
        rows.append(f"| {arm} | {points} | {check_text} | {signals} | "
                    f"{report.get('next_action', 'complete post-training assessment')} |")
    rows += ["", "| Arm | Minimum norm | Median | Mean | Norm RMS | p95 | p99 | Maximum | Fraction >1 | Fraction >norm(z) | Range of r (log units) |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    def number(value):
        return "not available" if value is None else f"{value:.6g}"
    for arm in ("control", "clip", "depth", "batch128"):
        geometry = result["arms"].get("continue-"+arm, {}).get("assessment", {}).get("post_training", {}).get("geometry", {})
        stats = geometry.get("score_residual_norm", {})
        values = [stats.get(k) for k in ("min", "median", "mean", "rms", "p95", "p99", "max")]
        values += [stats.get("exceedance_fraction", {}).get("1.0"), stats.get("fraction_larger_than_gaussian_score"),
                   geometry.get("r_log_target_over_gaussian_up_to_constant", {}).get("range")]
        rows.append("| "+arm+" | "+" | ".join(number(v) for v in values)+" |")
    rows += ["", "Score residuals are explanatory checks on base draws; they do not certify posterior coverage or HMC convergence.", "",
        "| Decision | Primary criterion | Veto status | Uncertainty | Next action | Not concluded |",
        "| --- | --- | --- | --- | --- | --- |",
        "| Close bounded repair tranche | Recorded updates, preserved state and assessment | Inspect each arm's numerical and clipping screens | Sustained learning, seeds, residual geometry and coverage | Extend improving viable maps or freshly tune an eligible frozen map | Posterior validity or optimal hyperparameters |", "",
        "| Inference status | Finding |", "| --- | --- |",
        "| Hard veto screen | Per-arm status and frozen-map parity are in result.json; rejected/deferred arms remain in the denominator |",
        "| Statistically supported ranking | None |",
        "| Descriptive-only differences | Loss changes, clipping fractions, scale diagnostics and timings |",
        "| Default-readiness | Not established |",
        "| Next evidence needed | Sustained training, fresh fixed-transport HMC tuning, posterior precision/coverage/reference checks |", "",
        f"Aggregate worker time: {result['aggregate_worker_seconds']/3600:.4f} hours. "
        f"Remaining campaign: {result['remaining_campaign_seconds']/3600:.4f} hours; "
        f"diagnostics: {result['remaining_diagnostic_seconds']/60:.3f} minutes.", "",
        "The strongest alternative explanation for a lower training loss remains learning one local region without useful posterior exploration. "
        "Fresh downstream checks can overturn any suggestion of useful geometry. One parent map and one stream per arm are the weakest part of this evidence.", ""]
    (root/"tranche-summary.md").write_text("\n".join(rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--request", required=True, type=Path)
    args = parser.parse_args()
    request = json.loads(args.request.read_text())
    deadline = datetime.fromisoformat(request["deadline"]).timestamp()
    report = args.campaign/"watcher.json"
    record = {"script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "request_sha256": hashlib.sha256(args.request.read_bytes()).hexdigest(),
        "source": str(args.source), "campaign": str(args.campaign),
        "poll_seconds": 30, "maximum_recovery_launches": 1, "recovery_launches": 0,
        "role": "observe_owned_master_and_resume_once_under_existing_limits", "status": "watching"}
    def save():
        temporary = report.with_suffix(".tmp")
        temporary.write_text(json.dumps(record, indent=2)+"\n")
        temporary.replace(report)
    save()
    while time.time() < deadline:
        if (args.campaign/"result.json").exists():
            summary(args.campaign)
            record["status"] = "completed"
            save()
            return 0
        free = False
        with (args.campaign/"coordinator.lock").open("a+") as lock:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                free = True
            except BlockingIOError:
                pass
        if free:
            state = json.loads((args.campaign/"campaign.json").read_text())
            # An orphan timeout still owns its deadline. Campaign.recover will
            # settle it conservatively when its full cap has expired.
            pending = [a for a in state["attempts"] if a["status"] == "running"
                       and time.time() < a["started_epoch"]+a["cap_seconds"]]
            if not pending:
                if record["recovery_launches"] >= record["maximum_recovery_launches"]:
                    record["status"] = "recovery_failed_requires_inspection"
                    save()
                    return 1
                record["recovery_launches"] += 1
                save()
                command = ["/home/ubuntu/anaconda3/envs/tfgpu/bin/python",
                    "docs/benchmarks/run_ssl_lstm_q20_production_2026_09_15.py", "repair-training",
                    "--request", str(args.request), "--output-dir", str(args.campaign)]
                with (args.campaign/"watcher-recovery.log").open("ab") as output:
                    subprocess.run(command, cwd=args.source, stdout=output, stderr=subprocess.STDOUT,
                                   check=False, timeout=max(1., deadline-time.time()))
        time.sleep(30)
    record["status"] = "deadline_reached"
    save()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
