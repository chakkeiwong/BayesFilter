"""Post-run diagnostic audit and descriptive precision feasibility; stdlib only."""
import hashlib
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path("/home/chakwong/BayesFilter")
BASE = ROOT / "docs/plans/artifacts/controlled-smc-kdm-replication-20260928"
PHASE = BASE / "native-selection-20261013"


def read(path):
    return json.loads(path.read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False)+"\n")


def main():
    checks = []
    def check(name, ok):
        if not ok:
            raise AssertionError(name)
        checks.append(name)
    out = PHASE / "attempt01"
    diagnostic = out / "diagnostic"
    manifest = read(out / "manifest.json")
    result = read(diagnostic / "result.json")
    check("completed", manifest["status"] == result["status"] == "COMPLETED")
    check("96 full validation evaluations", result["evaluations"] == 96)
    check("CPU-only reference", manifest["cpu_only"] and manifest["gpu_intentionally_hidden"] and manifest["resource_wall_seconds"]["gpu"] == 0)
    check("time cap", 0 < manifest["resource_wall_seconds"]["cpu"] < 1800)
    check("worker check count", len(read(diagnostic / "checks.json")) == result["checks"] == 2638)
    for name, expected in manifest["source_hashes"].items():
        check("launch source snapshot "+name, sha(out / "sources" / name) == expected)
        if name.endswith(".py"):
            check("unchanged executed code "+name, sha(ROOT / name) == expected)
    for name, expected in read(diagnostic / "inputs.json").items():
        check("unchanged input "+name, sha(ROOT / name) == expected)
    source_root = ROOT / ".localresources/code/younis-mdpf-neurips-2023"
    for name, expected in read(diagnostic / "author-source-hashes.json").items():
        check("unchanged author source "+name, sha(source_root / name) == expected)
    ledger = read(BASE / "reference-diagnostics.json")
    charged = [x for x in ledger if x["id"] == "kdm-native-selection-attempt01_cpu"]
    check("exactly one CPU charge", len(charged) == 1)
    check("ledger seconds", charged[0]["charged_seconds"] == manifest["resource_wall_seconds"]["cpu"])
    check("ledger summary identity", charged[0]["summary_sha256"] == sha(out / "decision.json"))
    check("decision equals result", read(out / "decision.json") == result)
    rows = [json.loads(line) for line in (diagnostic / "evaluations.jsonl").read_text().splitlines()]
    check("raw evaluation count", len(rows) == 96)
    groups = {}
    for row in rows:
        key = (row["unit"], row["mode"], row["checkpoint"])
        groups.setdefault(key, []).append(row)
        check("1000 native rows", [i for b in row["batches"] for i in b["rows"]] == list(range(1000)))
        check("native partial batch", [b["count"] for b in row["batches"]] == [256,256,256,232])
        check("exact native loss weighting", sum(b["loss"]*b["count"] for b in row["batches"])/1000 == row["loss"])
        check("finite NLL", math.isfinite(row["loss"]))
    for key, values in groups.items():
        check("eight independent repetitions "+str(key), [r["repetition"] for r in values] == list(range(8)) and len({r["seed"] for r in values}) == 8)
    rng = []
    for unit in (1,2):
        for repetition in range(8):
            subset = [r for r in rows if r["unit"] == unit and r["mode"] == "shared_initial_rng" and r["repetition"] == repetition]
            check("actual shared initial RNG", len({r["rng_before"] for r in subset}) == 1)
            rng.append(dict(unit=unit, repetition=repetition, distinct_final_rng_states=len({r["rng_after"] for r in subset})))
        control = read(diagnostic / ("unit%02d" % unit) / "validation-control.json")
        check("source endpoint control", control["native"] == control["manual"])
    zero = read(diagnostic / "unit01/window0.json")
    original = read(BASE / "paired-calibration-20261011/calibration-01/unit01/training/diagnostic/mdpf-best.json")["model"]
    check("exact window starting state", zero == original)
    for epoch in (1,2):
        state = read(diagnostic / ("unit01/window%d.json" % epoch))
        batches = read(diagnostic / ("unit01/training-batches%d.json" % epoch))
        check("native training coverage", sorted(i for b in batches for i in b["rows"]) == list(range(5000)))
        check("native training partial batch", [b["count"] for b in batches] == [64]*78+[8])
        check("actual Adam update count", state["step"] == zero["step"]+79*epoch)
        check("fixed saved rates", state["rates"] == zero["rates"])
        check("real changed intermediate checkpoint", state["parameters"] != zero["parameters"])
    write(PHASE / "terminal-audit.json", dict(status="PASS", checks=checks, count=len(checks),
        worker_checks=result["checks"], rng_comparison=rng, audit_source_sha256=sha(Path(__file__)),
        result_sha256=sha(diagnostic / "result.json"), manifest_sha256=sha(out / "manifest.json"),
        nonclaims="No population ranking, completed KDM replication, or historical checkpoint-selection reconstruction"))

    sys.path.insert(0, str(ROOT / "docs/benchmarks"))
    from run_replication_gap_localization import campaign
    balance = campaign()
    check("no campaign process running", not balance["running"])
    calibration = BASE / "paired-calibration-20261011/calibration-01"
    decision_path = calibration / "decision.json"
    decision = read(decision_path)
    inputs = {str(decision_path.relative_to(ROOT)): sha(decision_path)}
    z = statistics.NormalDist().inv_cdf(1-.05/4)
    power_z = statistics.NormalDist().inv_cdf(.8)
    costs = []
    for unit in ("unit01", "unit02"):
        sp = BASE / "kdm-source-variability-20261008-01" / (unit+"-full/manifest.json")
        tp = calibration / unit / "training/manifest.json"
        inputs.update({str(p.relative_to(ROOT)): sha(p) for p in (sp,tp)})
        source_seconds, tf_seconds = read(sp)["wall_seconds"], read(tp)["wall_seconds"]
        cpu_remaining = min(balance["remaining_seconds"]["cpu"], balance["tranche_remaining_seconds"]["cpu"])
        gpu_remaining = min(balance["remaining_seconds"]["gpu"], balance["tranche_remaining_seconds"]["gpu"])
        costs.append(dict(unit=unit, source_cpu_seconds=source_seconds, tf_cpu_and_gpu_seconds=tf_seconds,
            pair_cpu_seconds=source_seconds+tf_seconds, pair_gpu_seconds=tf_seconds,
            affordable_fresh_pairs=min(int(cpu_remaining/(source_seconds+tf_seconds)),int(gpu_remaining/tf_seconds))))
    affordable = max(c["affordable_fresh_pairs"] for c in costs)
    estimates = {}
    for variant in ("mdpf", "am_dpf"):
        values = [u["variants"][variant] for u in decision["units"]]
        estimates[variant] = {}
        for metric, field, margin in (("nll", "nll_difference", .01), ("rmse", "log_rmse_ratio", math.log(1.01))):
            data = [x[field] for x in values]
            sd = statistics.stdev(data)
            count = math.ceil(((z+power_z)*sd/margin)**2)
            check("reproduced previous paired precision", count == decision["descriptive_statistics"][variant]["illustrative_units_at_true_equality"][metric])
            estimates[variant][metric] = dict(n_training_pairs=len(data), observations=data,
                mean=statistics.mean(data), sd=sd, margin=margin, illustrative_pairs_at_true_equality=count,
                illustrative_upper_margin_at_true_equality_with_affordable_pairs=z*sd/math.sqrt(affordable),
                sd_needed_at_affordable_n=margin*math.sqrt(affordable)/(z+power_z))
    write(PHASE / "precision-feasibility.json", dict(status="NO_JUSTIFIED_AFFORDABLE_QUALIFICATION_DESIGN",
        interpretation="Planning illustration only: n=3 plug-in SD, normal paired means, true equality, 80% power, four-endpoint Bonferroni 95%; no validated sample-size claim",
        cpu_remaining_seconds=cpu_remaining, active_gpu_remaining_seconds=gpu_remaining,
        alpha_each=.05/4, z=z, power_z=power_z, observed_costs=costs,
        optimistic_affordable_fresh_pairs=affordable, estimates=estimates, inputs=inputs,
        implementation_sha256=sha(Path(__file__)), ledger_sha256=sha(BASE / "reference-diagnostics.json"),
        next_action="Preserve criteria and freeze current implementation; demonstrate lower paired variance before an affordable qualification design, or obtain an explicit change of qualification scope",
        nonclaims="Does not prove every possible design impossible; repeated evaluations are not independent training units"))
    print(json.dumps(dict(terminal_checks=read(PHASE / "terminal-audit.json")["count"], worker_checks=result["checks"],
        affordable_pairs=affordable, cpu_remaining=cpu_remaining, gpu_remaining=gpu_remaining,
        precision_status="NO_JUSTIFIED_AFFORDABLE_QUALIFICATION_DESIGN")))


if __name__ == "__main__":
    main()
