"""CPU-only integrity/accounting closeout; no statistical recomputation."""
import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time

started = time.monotonic()
out = Path(__file__).resolve().parent
campaign = out.parent
root = campaign / "continuation-04"
repo = Path.cwd()
plan = repo / "docs/plans/bayesfilter-modern-hmc-sir-closeout-2026-10-20.md"
result_note = repo / "docs/plans/bayesfilter-modern-hmc-sir-result-2026-10-20.md"
cap = 1661.4659486340872


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_new(name, data):
    with (out / name).open("x") as stream:
        json.dump(data, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def close(a, b):
    assert math.isclose(a, b, rel_tol=0, abs_tol=1e-7), (a, b)


manifest = {
    "started_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
    "command": [sys.executable, *sys.argv],
    "environment": sys.executable,
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    "gpu_devices_intentionally_hidden": True,
    "numerical_replay": False,
    "plan": str(plan),
    "result_file": str(result_note),
    "driver_sha256": digest(__file__),
    "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "evidence_numerical_commit": "e68cdf1c4c6b75c16f92b86a3363a1e24690d22c",
    "cpu_gpu_role": "CPU standard-library integrity audit; no GPU or numerical kernels",
    "audit_random_seeds": "N/A: deterministic read-only audit; numerical seeds below",
    "cap_seconds": cap,
}
try:
    assert manifest["cuda_visible_devices"] == "-1"
    status = read(root / "status.json")
    assert status["status"] == "completed_bounded_cohort"
    assert all(m["status"] == "settled" for m in status["methods"])
    source = read(campaign / "continuation-source-03/source.json")
    expected = dict(source["source_and_input_sha256"])
    summaries = []
    numerical_provenance = []
    evidence_counts = {}
    for method in status["methods"]:
        name = method["method"]
        if name == "fixed_hmc":
            recovered = read(root / "recovered/fixed_hmc-evidence.json")
            phase = Path(recovered["phase"])
            bindings = recovered["sha256"]
            settlement_path = root / "recovered/fixed_hmc-settlement.json"
        else:
            phase = root / name
            bindings = read(phase / "terminal-evidence-sha256.json")
            settlement_path = phase / "settlement.json"
        evidence_counts[name] = len(bindings)
        for path, checksum in bindings.items():
            assert path not in expected or expected[path] == checksum, path
            expected[path] = checksum
        settled = read(settlement_path)
        assert settled["evidence_audit_passed"] is True
        assert settled["disposition"] == method["disposition"]
        close(settled["numerical_charged_seconds"], method["charged_seconds"])
        run = phase / "run"
        numerical = read(run / "manifest.json")
        tuning = read(run / "search/tuning/tuning_checkpoint.json")["result"]
        posterior = read(run / "posterior/result.json")
        assert tuning["completion_status"] == "complete"
        ids = tuning["verified_candidate_ids"]
        assert len(ids) == len(set(ids))
        assert {r["candidate_id"] for r in posterior["rows"]} == set(ids)
        assert posterior["tuning_membership_unchanged"] is True
        rows = {r["candidate_id"]: r for r in posterior["rows"]}
        for row in settled["rows"]:
            saved = rows[row["candidate_id"]]
            assert row["status"] == saved["status"]
            assert row.get("delivered", False) == saved.get("delivered", False)
            ref = saved.get("reference")
            if ref is not None:
                screen = ref["normalized_error_screen"]
                threshold = ref["normalized_tolerance"]
                assert threshold == numerical["design"]["posterior"]["reference_normalized_tolerance"]
                passed = all(math.isfinite(v) and 0 <= v <= threshold for v in screen)
                assert ref["passed"] == passed == row["reference_passed"]
                assert row["delivered"] == (row["posterior_passed"] and passed and not row["hard_vetoes"])
        disp = method["disposition"]
        assert len(ids) == disp["verified_count"] == len(posterior["rows"])
        summary = {"method": name, "status": method["cohort_status"],
                   "requested_roots": disp["requested_root_count"],
                   "verified": len(ids), "assessed": disp["assessed_count"],
                   "posterior_passed": sum(r.get("posterior_passed") is True for r in settled["rows"]),
                   "delivered": disp["delivered_count"],
                   "failed_reference_ids": [r["candidate_id"] for r in settled["rows"] if r.get("reference_passed") is False],
                   "unfinished_verified_ids": disp["unfinished_verified_ids"],
                   "charged_seconds": method["charged_seconds"],
                   "settlement_file": str(settlement_path),
                   "settlement_sha256": digest(settlement_path)}
        summaries.append(summary)
        assert numerical["use_xla"] and numerical["tf32_enabled"] and not numerical["cpu_reference"]
        assert numerical["memory_policy"]["all_physical_devices_memory_growth"]
        assert numerical["memory_policy"]["configured_before_logical_device_initialization"]
        numerical_provenance.append({
            "method": name, "command": numerical["command"],
            "manifest": str(run / "manifest.json"),
            "target": numerical["target"], "cuda_visible_devices": numerical["cuda_visible_devices"],
            "memory_policy": numerical["memory_policy"], "use_xla": True, "tf32_enabled": True,
            "tuning_seed": numerical["design"]["execution"]["seed"],
            "posterior_seed": numerical["design"]["posterior"]["seed"],
        })
    for path, checksum in expected.items():
        assert digest(path) == checksum, path
        assert time.monotonic() - started < cap, "closeout cap exhausted"
    accounting = read(root / "accounting.json")
    engineering = read(root / "engineering.json")
    close(math.fsum(m["charged_seconds"] for m in summaries), accounting["numerical_inclusive_seconds"])
    close(math.fsum(e["seconds"] for e in engineering["entries"]), accounting["exclusive_engineering_seconds"])
    close(accounting["post_preparation_seconds"] - accounting["pricing_inclusive_seconds"]
          - accounting["numerical_inclusive_seconds"] - accounting["exclusive_engineering_seconds"],
          accounting["signed_balance_seconds"])
    close(status["remaining_numerical_seconds"] + accounting["unused_engineering_hold_released_seconds"],
          accounting["signed_balance_seconds"])
    submission_path = campaign / "controller-recovery-20261018-01/launch-submission-01.json"
    submission = read(submission_path)
    assert submission["status"] == "submitted" and submission["submission_charge_at_closeout"]
    assert submission["launch_manifest_sha256"] == digest(submission_path.with_name("launch-manifest-01.json"))
    assert not any(e.get("receipt") == str(submission_path) for e in engineering["entries"])
    assert sum(m["requested_roots"] for m in summaries) == 24
    assert sum(m["verified"] for m in summaries) == 24
    assert sum(m["assessed"] for m in summaries) == 23
    assert sum(m["posterior_passed"] for m in summaries) == 23
    assert sum(m["delivered"] for m in summaries) == 21
    report = {"status": "passed", "finished_utc": status["finished_utc"],
              "methods": summaries, "unique_hash_bindings_checked": len(expected),
              "per_method_hash_bindings": evidence_counts, "method_ranking": None,
              "release_qualified": False, "tuning_membership_unchanged": True,
              "independent_statistical_recomputation": False,
              "scope": "Existing strict-audit receipts, fresh file integrity, member decisions and accounting arithmetic"}
    write_new("audit.json", report)
    manifest["numerical_provenance"] = numerical_provenance
    write_new("run-manifest.json", manifest)
    elapsed = time.monotonic() - started
    assert elapsed < cap
    settlement = {
        "status": "settled", "controller_accounting_sha256": digest(root / "accounting.json"),
        "controller_balance_seconds": accounting["signed_balance_seconds"],
        "exclusive_submission_seconds": submission["elapsed_seconds"],
        "submission_receipt": str(submission_path), "submission_sha256": digest(submission_path),
        "closeout_seconds": elapsed,
        "available_sir_balance_seconds": accounting["signed_balance_seconds"] - submission["elapsed_seconds"] - elapsed,
        "existing_controller_charges_reused_not_added_again": True,
        "pricing_recharged": False, "separate_diagnostic_pool_untouched": True,
        "other_existing_reserves_excluded": True,
        "audit_sha256": digest(out / "audit.json"),
        "run_manifest_sha256": digest(out / "run-manifest.json"),
        "charge_scope": "Inclusive standard-library closeout process through final receipt preparation; final JSON write excluded",
    }
    write_new("settlement.json", settlement)
    print(json.dumps({"status": "passed", "counts": [24, 23, 21],
                      "hash_bindings": len(expected), "closeout_seconds": elapsed,
                      "available_sir_balance_seconds": settlement["available_sir_balance_seconds"]}))
except BaseException as error:
    write_new("failure.json", {"status": "failed", "error": repr(error),
                               "elapsed_seconds": time.monotonic() - started})
    raise
