"""Recovery verification and conditional heuristic comparison; standard library only."""
from pathlib import Path
import hashlib
import json
import math
import statistics
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[6]
OUT = Path(__file__).resolve().parent
BASE = OUT.parent
summary = json.loads((BASE / "analysis-01/summary.json").read_text())
budget = json.loads((BASE.parent / "budget.json").read_text())
checks = []
def check(name, ok):
    checks.append({"check": name, "pass_check": bool(ok)})
def close(a, b):
    return math.isclose(a, b, rel_tol=1e-10, abs_tol=1e-11)

rows = []
for attempt in budget["attempts"]:
    result = json.loads((ROOT / attempt["output"] / "result.json").read_text())
    rows.extend(dict(unit=result["unit"], **r) for r in result["rows"] if "arm" in r)
check("525 saved evaluations", len(rows) == 525)
check("stored errors equal numerical output minus seven-mixture reference",
      all(close(r["value_error"], r["value"] - r["reference_value"])
          and close(r["absolute_value_error"], abs(r["value_error"]))
          and all(close(e, v - ref) for e, v, ref in zip(r["score_error"], r["score"], r["reference_score"]))
          and all(close(a, abs(e)) for a, e in zip(r["absolute_score_error"], r["score_error"]))
          and close(r["score_l2_error"], math.sqrt(sum(e*e for e in r["score_error"])))
          for r in rows))
heuristic_rows = []
for cell in summary["cells"]:
    route, horizon, seed = cell["route"], cell["horizon"], cell["data_seed"]
    for label, arm in (("baseline", "baseline"), ("candidate", cell["selected_arm"])):
        selected = [r for r in rows if r["unit"].split("__")[0] in ("evaluation", "extension")
                    and r["route"] == route and r["horizon"] == horizon
                    and r["data_seed"] == seed and r["arm"] == arm]
        check(f"{route}/{horizon}/{seed}/{label}: eight unique designs",
              len(selected) == len({r["design_seed"] for r in selected}) == 8)
        saved = cell[label]
        check(f"{route}/{horizon}/{seed}/{label}: summary means",
              close(saved["mean_replicate_score_l2_error"], statistics.mean(r["score_l2_error"] for r in selected))
              and close(saved["mean_absolute_value_error"], statistics.mean(r["absolute_value_error"] for r in selected))
              and all(close(saved["mean_score"][j], statistics.mean(r["score"][j] for r in selected))
                      for j in range(2)))
        for name, error in cell["heuristic_score_l2_errors"].items():
            loss = saved["mean_replicate_score_l2_error"] > error
            heuristic_rows.append(dict(route=route, horizon=horizon, data_seed=seed, arm=label,
                                       heuristic=name, mean_score_l2_error=saved["mean_replicate_score_l2_error"],
                                       heuristic_score_l2_error=error, loses=loss))
latest = json.loads((ROOT / budget["attempts"][-1]["output"] / "manifest.json").read_text())
source_changes = [p for p, expected in latest["source_sha256"].items()
                  if hashlib.sha256((ROOT / p).read_bytes()).hexdigest() != expected]
check("latest numerical/source closure unchanged", not source_changes)
result = dict(schema="sqmc_ksc_recovery_verification_v1", created_utc=datetime.now(timezone.utc).isoformat(),
              cpu_only=True, gpu_intentionally_unused=True, framework_imported=False,
              checks=checks, all_pass=all(c["pass_check"] for c in checks),
              heuristic_comparisons=heuristic_rows, source_changes=source_changes,
              candidate_losses=sum(r["loses"] for r in heuristic_rows if r["arm"]=="candidate"),
              baseline_losses=sum(r["loses"] for r in heuristic_rows if r["arm"]=="baseline"),
              limitation="Heuristic screens use conditional descriptive mean score error. Passing is not a superiority claim.")
with (OUT / "verified-results.json").open("x") as f:
    json.dump(result, f, indent=2); f.write("\n")
lines = ["# Conditional heuristic comparison", "",
         "Eight paired designs per fixed dataset. Every original and repaired route is compared with all three prespecified heuristics.",
         "The quantities below are descriptive mean score-vector L2 errors; no selection used these heuristic comparisons.", "",
         "| Route | T | Data | Arm | Mean score error | Heuristic | Heuristic error | Screen |",
         "|---|---:|---:|---|---:|---|---:|---|"]
for r in heuristic_rows:
    lines.append(f"| {r['route']} | {r['horizon']} | {r['data_seed']} | {r['arm']} | {r['mean_score_l2_error']:.6f} | {r['heuristic']} | {r['heuristic_score_l2_error']:.6f} | {'FAIL' if r['loses'] else 'Pass'} |")
(OUT / "heuristic-comparison.md").write_text("\n".join(lines)+"\n")
print(json.dumps({k: result[k] for k in ("all_pass", "candidate_losses", "baseline_losses", "source_changes")}))
print("checks", len(checks))
if not result["all_pass"]:
    print([c["check"] for c in checks if not c["pass_check"]])
    raise SystemExit(1)
