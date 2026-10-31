"""Post-run explanatory comparison with the earlier mean score-error changes.

The primary precision screen remains in report.json. This does not select controls.
"""
from pathlib import Path
import json
import math
import statistics

root = Path(__file__).resolve().parent
artifacts = root.parent
sources = {1008: artifacts/"ledh-range-bearing-covariance-20261009-07",
           2016: artifacts/"ledh-range-bearing-particle-ladder-20261010-02/N2016",
           4032: artifacts/"ledh-range-bearing-particle-ladder-20261010-02/N4032"}
read = lambda p: [json.loads(s) for s in p.read_text().splitlines()] if p.exists() else []
data = {(n, mode): read(root/f"N{n}-{mode}"/"measurements.jsonl")
        for n in sources for mode in ("fp64", "fp32")}
out = []
for ds in (1704, 18020, 18021):
    for point in (0, 1):
        ref = json.loads((sources[1008]/"references"/f"{ds}-{point}-65536.json").read_text())["score"]
        errors = {}
        for n in sources:
            for mode in ("fp64", "fp32"):
                rows = [r for r in data[n, mode] if (r["data_seed"], r["point"]) == (ds, point)]
                if len(rows) == 8 and all(r["valid"] for r in rows):
                    errors[n, mode] = statistics.mean(math.dist(r["score"],ref) for r in rows)
        if not all((n,mode) in errors for n in (1008,2016) for mode in ("fp64","fp32")):
            continue
        gain64 = errors[1008,"fp64"]-errors[2016,"fp64"]
        gain32 = errors[1008,"fp32"]-errors[2016,"fp32"]
        out.append(dict(data_seed=ds, point=point,
                        fp64_mean_error_reduction_1008_to_2016=gain64,
                        fp32_mean_error_reduction_1008_to_2016=gain32,
                        precision_changes_observed_reduction_by=gain32-gain64,
                        absolute_change_as_fraction_of_observed_gap=abs((gain32-gain64)/gain64) if gain64 else None,
                        per_n=[dict(particles=n, mean_score_error_fp64=errors[n,"fp64"],
                                    mean_score_error_fp32=errors[n,"fp32"],
                                    mean_score_error_shift=errors[n,"fp32"]-errors[n,"fp64"])
                               for n in sources if all((n,mode) in errors for mode in ("fp64","fp32"))]))
result = dict(role="explanatory_only_not_a_replacement_for_predeclared_materiality_screen", cells=out,
              interpretation="Changes use the saved finite PF reference; neither precision nor particle-count comparison proves improved true filtering accuracy.",
              bound="For any fixed reference r, abs(norm(s32-r)-norm(s64-r)) <= norm(s32-s64).")
(root/"materiality-detail.json").write_text(json.dumps(result,indent=2)+"\n")
print(json.dumps(out,indent=2))
