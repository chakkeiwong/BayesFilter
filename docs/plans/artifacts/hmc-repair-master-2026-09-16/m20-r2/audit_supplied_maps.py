"""Read-only terminal audit of the fixed supplied-map development inventory."""
from collections import Counter
import datetime
import hashlib
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[4]
started = time.monotonic()


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def near(left, right):
    assert len(left) == len(right)
    for x, y in zip(left, right):
        if isinstance(x, list):
            near(x, y)
        else:
            assert math.isclose(x, y, rel_tol=2e-11, abs_tol=2e-11), (x, y)


rows, failed_attempts, all_seeds = [], [], set()
receipt_count = 0
model_bank = [[x, x, x] for x in (-1., -.3, .4, 1.)]
for path in sorted(ROOT.glob("fits-*/**/execution.json")):
    attempt = read(path)
    worker = path.parent / "worker"
    if attempt["exit_code"]:
        assert not (worker / "manifest.json").exists()
        failed_attempts.append({"path": str(path.relative_to(REPO)), **attempt})
        continue
    result, manifest = read(worker / "result.json"), read(worker / "manifest.json")
    fit = worker / "fit"
    tuning = read(fit / "tuning/candidate_set_result.json")
    checkpoint = read(fit / "tuning/tuning_checkpoint.json")
    execution = read(fit / "tuning/execution_spec.json")["execution"]
    selection = read(fit / "posterior_selection.json")
    starts = read(fit / "start_coordinates.json")
    spec = read(worker / "supplied_map.json")
    kind, seed = result["map"], result["seed"]
    search = result.get("search")
    if search is None:
        # The preserved baseline driver predates the optional follow-up flag.
        assert path.relative_to(ROOT).parts[0] == "fits-cpu-r1"
        search = "native"
    assert result["status"] == "complete" and not result["inventory"]["failures"]
    assert manifest["git_commit"].startswith("01d67ec41")
    assert manifest["gpu_intentionally_hidden"] and not manifest["jit_compile"]
    assert manifest["cpu_threads"] == 1 and not spec["construction"]["training_performed"]
    for file, digest in execution["source_closure"].items():
        assert sha(Path(file)) == digest, file
    for file in ("bayesfilter/testing/inference_validation/funnel_maps.py",
                 "bayesfilter/testing/inference_validation/procedures.py"):
        assert sha(REPO / file) == manifest["source_hashes"][file], file
    assert tuning["scope"]["coordinate_system"] == "fixed_transport"
    assert starts["supplied_to_tuner"] == "latent" and starts["forward_roundtrip_checked"]
    near(starts["model_starts"], model_bank)
    strength = spec["construction"]["strength"]
    mapped = []
    for z in starts["latent_starts"]:
        v = 3. * z[0]
        s = v / 2. if strength is None else 6. * math.tanh(strength * v / 12.)
        mapped.append([v, math.exp(s) * z[1], math.exp(s) * z[2]])
    near(mapped, model_bank)
    for field in ("scope", "config", "candidates", "candidate_states", "work_items",
                  "verification_receipts", "verified_candidate_ids", "completion_status"):
        assert tuning[field] == checkpoint["result"][field], field
    candidates = {c["candidate_id"]: c for c in tuning["candidates"]}
    assert len(candidates) == result["inventory"]["candidate_count"]
    verified = set(tuning["verified_candidate_ids"])
    assert verified == {cid for cid, state in tuning["candidate_states"].items() if state == "verified"}
    assert verified == set(result["verified_candidate_ids"]) == set(selection["verified_candidate_ids"])
    assert tuning["nominee_id"] is None
    assert selection["rule"] == "first_verified"
    assert selection["selected_candidate_ids"] == sorted(verified)[:1]
    passed = set()
    for receipt in tuning["verification_receipts"]:
        receipt_count += 1
        stream_seed = tuple(receipt["seed_lineage"])
        assert stream_seed not in all_seeds, (kind, seed, stream_seed)
        all_seeds.add(stream_seed)
        candidate = candidates[receipt["candidate_id"]]
        assert receipt["epsilon"] == candidate["epsilon"]
        assert receipt["exact_l"] == candidate["leapfrog_steps"]
        assert receipt["candidate_record_hash"] == candidate["candidate_record_hash"]
        evidence_path = fit / "tuning/numerical_evidence" / (receipt["numerical_evidence_hash"] + ".json")
        evidence = read(evidence_path)
        assert evidence["seed"] == receipt["seed_lineage"]
        assert evidence["work"]["stage"] == receipt["stage"]
        assert evidence["candidate"] == candidate
        if receipt["stage"] == "verification" and receipt["promotion_eligible"]:
            assert receipt["decision"] == "passed" and receipt["evidence_validity"] == "valid"
            assert not receipt["hard_vetoes"] and not receipt["promotion_vetoes"]
            passed.add(receipt["candidate_id"])
    assert passed == verified
    if kind == "exact":
        assert verified
    health_failures = set()
    for row in result["posterior"]:
        assert row["warmup_exclusion_matches"] and row["candidate_id"] in verified
        member = read(fit / "members" / row["candidate_id"] / "result.json")
        assert member["posterior"]["assessment_role"] == "posterior_only"
        for check in member["posterior"]["warmup_checks"] + member["posterior"]["retained_checks"]:
            health_failures.update(check["health"]["member_health_failures"])
    assert result["unassessed_members"] + len(result["posterior"]) == len(verified)
    hashes = {name: sha(worker / name) for name in
              ("manifest.json", "result.json", "design.json", "supplied_map.json",
               "fit/start_coordinates.json", "fit/tuning/candidate_set_result.json")}
    rows.append({"path": str(path.parent.relative_to(REPO)), "map": kind, "seed": seed,
                 "search": search, "candidate_count": len(candidates),
                 "verified_count": len(verified), "verified_candidate_ids": sorted(verified),
                 "posterior": result["posterior"], "posterior_health_failures": sorted(health_failures),
                 "unassessed_members": result["unassessed_members"],
                 "wall_seconds": attempt["wall_seconds"], "file_hashes": hashes})

assert Counter((r["map"], r["search"]) for r in rows) == {
    ("exact", "native"): 2, ("partial", "native"): 2, ("partial_half", "native"): 2,
    ("partial", "intermediate_grid"): 2, ("partial_half", "intermediate_grid"): 2}
assert len(failed_attempts) == 1
summary = {"schema": "bayesfilter.supplied_funnel_terminal_audit.v1",
           "created_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
           "command": [sys.executable, str(Path(__file__).resolve())],
           "plan_file": "docs/plans/bayesfilter-hmc-supplied-whitening-plan-2026-09-22.md",
           "status": "passed", "complete_fit_count": len(rows),
           "all_physical_start_banks_match": True,
           "candidate_count": sum(r["candidate_count"] for r in rows),
           "verified_member_count": sum(r["verified_count"] for r in rows),
           "independent_receipt_seed_count": receipt_count, "rows": rows,
           "failed_attempts": failed_attempts,
           "fit_worker_seconds": sum(r["wall_seconds"] for r in rows + failed_attempts),
           "prior_audit_attempts": [{"wall_seconds": 0.284774708,
               "failure": "KeyError: search; baseline driver predates optional grid flag",
               "repair": "require preserved baseline directory before inferring native search",
               "numerical_evidence_changed": False}],
           "role": "source, start, evidence, retention and outcome audit; no posterior promotion",
           "wall_seconds": time.monotonic() - started}
destination = ROOT / "terminal-audit.json"
assert not destination.exists()
destination.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
print(json.dumps({k: v for k, v in summary.items() if k not in {"rows", "failed_attempts"}}, indent=2))
