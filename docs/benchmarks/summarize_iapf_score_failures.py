"""Post-run diagnostic verification; descriptive comparisons, no ranking."""
import csv
import hashlib
import json
import math
from pathlib import Path

WORKDIR = Path(__file__).resolve().parents[2]
ROOT = WORKDIR / "docs/plans/artifacts/iapf-score-failure-remedies-20260923-01"


def read_csv(path):
    with path.open() as stream:
        return list(csv.DictReader(stream))


def sha(path):
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main():
    jobs = json.loads((ROOT / "completed.json").read_text())
    assert len(jobs) == 45
    records, mechanisms, differences, cap_checks, identities = [], [], [], [], []
    checked_files = 0
    for job in jobs.values():
        assert job["status"] == "complete" and job["exit_code"] == 0 and job["sources_unchanged"]
        directory = Path(job["output_directory"])
        for name, digest in job["output_hashes"].items():
            assert sha(directory / name) == digest, str(directory / name)
            checked_files += 1
        for name, digest in job["source_hashes"].items():
            assert sha(WORKDIR / name) == digest, name
        for filename, target in (("records.csv", records), ("mechanism.csv", mechanisms),
                                 ("finite-differences.csv", differences),
                                 ("healthy-identity.csv", identities)):
            if (directory / filename).exists():
                target.extend(read_csv(directory / filename))
        task = job["task"]
        if task["mode"] == "learner" and task["data_seed"] == 971080 and task["method"] == "score":
            check = read_csv(directory / "checks.csv")[0]
            assert check["old_history_exact"] == check["old_guides_exact"] == "TRUE"
            hist = read_csv(directory / "history.csv")
            cap_checks.append(dict(label=task["label"], original_history_and_guides_exact=True,
                last_old_cv=float(hist[11]["cv"]), stop_iteration=int(hist[-1]["iteration"]),
                stopping_cv=float(hist[-1]["cv"]), N=int(hist[-1]["N"])))
    assert len(records) == 64 and len(mechanisms) == 4 and len(identities) == 8
    assert all(r["bitwise_identical"] == "TRUE" for r in identities)
    assert all(r["status"] == "complete" for r in records)
    assert all(math.isfinite(float(r["log_likelihood"])) for r in records)
    fresh = [r for r in records if r["data_seed"] != "971080"]
    keys = [(int(r["data_seed"]), int(r["label"]), r["method"]) for r in fresh]
    methods = ("score", "score_qr_rescue", "qr", "bootstrap", "fully_adapted",
               "current_observation", "full_oracle")
    expected = {(d, label, method) for d in (981080, 991080)
                for label in range(1, 5) for method in methods}
    assert len(keys) == len(set(keys)) and set(keys) == expected
    conditional = []
    for data in (981080, 991080):
        for method in methods:
            rows = [r for r in fresh if int(r["data_seed"]) == data and r["method"] == method]
            losses = [(float(r["ratio"])-1)**2 for r in rows]
            conditional.append(dict(data_seed=data, method=method, n=len(rows), failures=0,
                relative_rmse=math.sqrt(sum(losses)/len(rows)), mean_ratio=sum(float(r["ratio"]) for r in rows)/len(rows),
                mean_particles=sum(int(r["particles"]) for r in rows)/len(rows),
                largest_loss_fraction=max(losses)/sum(losses) if sum(losses) else 0,
                evidence_class="descriptive_only"))
    lookup = {(int(r["data_seed"]), int(r["label"]), r["method"]): r for r in fresh}
    heuristic_rows = []
    # The exact oracle is a correctness control/distance reference, not a
    # stochastic heuristic whose zero error an approximate guide can beat.
    for data in (981080, 991080):
        for method in ("score", "score_qr_rescue", "qr"):
            for control in ("bootstrap", "fully_adapted", "current_observation"):
                delta = sum((float(lookup[data, i, method]["ratio"])-1)**2 -
                            (float(lookup[data, i, control]["ratio"])-1)**2 for i in range(1, 5))/4
                heuristic_rows.append(dict(data_seed=data, method=method, control=control,
                    paired_mean_squared_error_difference=delta, promotion_veto=delta > 0,
                    statistically_supported_ranking=False))
    oracle_error = max(abs(float(r["log_error"])) for r in fresh if r["method"] == "full_oracle")
    assert oracle_error < 1e-8
    budget = json.loads((ROOT / "budget.json").read_text())
    verification = dict(status="verified", completed_jobs=len(jobs), checked_output_hashes=checked_files,
        completed_likelihood_records=len(records), original_failures_reclassified=False,
        mechanisms=mechanisms, cap_extensions=cap_checks, healthy_identity=identities,
        finite_difference_max_score_error=max(float(r["score_direction_error"]) for r in differences),
        finite_difference_max_hessian_error=max(float(r["hessian_vector_error"]) for r in differences),
        fresh_conditional_summaries=conditional, heuristic_comparisons=heuristic_rows,
        heuristic_dominance_verdict="observed_loss_veto" if any(r["promotion_veto"] for r in heuristic_rows)
            else "passed_descriptive_screen_no_ranking",
        full_oracle_max_abs_log_error=oracle_error, budget=budget,
        statistical_ranking_supported=False, paper_replication=False, default_readiness=False)
    (ROOT / "verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    for name, rows in (("records.csv", records), ("conditional-summary.csv", conditional),
                       ("heuristic-comparisons.csv", heuristic_rows), ("cap-extensions.csv", cap_checks)):
        with (ROOT / name).open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(dict(jobs=len(jobs), records=len(records), checked_output_hashes=checked_files,
        healthy_identity_cases=len(identities), heuristic_verdict=verification["heuristic_dominance_verdict"],
        finite_difference_max_score_error=verification["finite_difference_max_score_error"],
        finite_difference_max_hessian_error=verification["finite_difference_max_hessian_error"],
        oracle_error=oracle_error, cpu_seconds=budget["cpu_charged_seconds"]), indent=2))


if __name__ == "__main__":
    main()
