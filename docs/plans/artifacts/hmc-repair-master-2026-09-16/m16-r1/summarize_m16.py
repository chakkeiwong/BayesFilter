"""Summarize complete M16 experiments without resampling or changing decisions."""
from collections import Counter
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def main():
    boundary, sequential, synthetic, sbc, pilots = [], [], [], [], []
    for directory in ("boundary-pilot-gpu-r1", "sequential-pilot-cpu-r1",
                      "sequential-pilot-gpu-r1", "independent-cpu-r1",
                      "fresh-cpu-r3", "fresh-gpu-r3"):
        root = ROOT / directory
        index = read(root / "run_index.json")
        designs = {j["design"]["design_id"]: j["design"] for j in index["plan"]["jobs"]}
        for name, job in index["jobs"].items():
            if job["status"] != "complete":
                raise ValueError("unfinished worker: " + name)
            design = designs[name]
            assessment = read(Path(job["result"]))["assessment"]
            if assessment["finding"] not in {"acceptance_rates_estimated", "power_estimated"}:
                raise ValueError("incomplete experiment: " + name)
            row = {"design_id": name, "device": design["device"], "result": job["result"],
                   "elapsed_seconds": sum(a["elapsed_seconds"] for a in job["attempts"])}
            if design["engine"] == "acceptance":
                keys = ("completed", "planned", "verified", "terminal_states", "qualification_interval")
                row.update({key: assessment[key] for key in keys})
                if assessment["numerical_hmc"]:
                    row.update(stationary_mean=design["options"]["analytic_stationary_acceptance"],
                               reference=assessment["reference"])
                    destination = pilots if "pilot" in directory else boundary
                else:
                    row.update(true_mean=assessment["true_mean"], persistence=assessment["persistence"])
                    destination = synthetic
            else:
                row.update(rates=assessment["rates"], planned=design["replications"])
                if "calibration_design" in design["options"]:
                    histograms = {}
                    for arm in assessment["rates"]:
                        values = []
                        for trial in assessment["trials"]:
                            observation = read(Path(trial[arm]["observations"]) / "invariance.json")
                            values.append(len(observation["looks"]))
                        histograms[arm] = dict(Counter(values))
                    row["look_counts"] = histograms
                    row["baseline_size_screen_passed"] = row["rates"]["baseline"]["interval"][1] <= .10
                    row["noop_size_screen_passed"] = row["rates"]["noop"]["interval"][1] <= .10
                    row["defect_power_screen_passed"] = row["rates"]["wrong_energy"]["interval"][0] >= .80
                    destination = pilots if "pilot" in directory else sequential
                else:
                    row.update(datasets=design["draws"], rank_draws=design["rank_draws"],
                               location_shift_posterior_sd=design["options"]["location_severity"])
                    row["null_size_screen_passed"] = row["rates"]["correct"]["interval"][1] <= .10
                    row["location_power_screen_passed"] = row["rates"]["location_defect"]["interval"][0] >= .80
                    destination = sbc
            destination.append(row)
    result = {"boundary": boundary, "sequential": sequential, "synthetic": synthetic,
              "sbc_statistic_power": sbc, "excluded_pilots": pilots,
              "ranking_supported": False, "nominal_acceptance_sequential_coverage": False,
              "public_whole_fit_sbc_defect_power_established": False,
              "interpretation": "Whole-experiment frequencies; pointwise binomial intervals; source/device scopes remain separate."}
    with (ROOT / "terminal-summary.json").open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps({"boundary": boundary, "sequential": sequential, "sbc_statistic_power": sbc}, indent=2))


if __name__ == "__main__":
    main()
