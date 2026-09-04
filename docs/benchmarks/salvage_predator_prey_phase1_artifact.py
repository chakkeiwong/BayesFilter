#!/usr/bin/env python3
"""Salvage Predator-Prey Phase 1 trust-region tuning artifact.

The Phase 1 campaign wrote 116 per-configuration evaluation files but never
wrote the campaign summary result.json. This script reconstructs it.
"""

import json
from pathlib import Path
from collections import defaultdict

ARTIFACT_DIR = Path("docs/benchmarks/artifacts/ledh_trust_region_predator_prey_t20_20260903")

def main():
    # Load all evaluation files
    all_files = sorted(ARTIFACT_DIR.glob("*.json"))
    print(f"Found {len(all_files)} evaluation files")

    # Parse into arms and configs
    arm_configs = defaultdict(list)

    for fpath in all_files:
        with open(fpath) as f:
            data = json.load(f)

        # Parse arm from filename: arm1_baseline_obs0_seed98301.json
        fname = fpath.stem
        parts = fname.split("_")

        # Determine arm type
        if "baseline" in fname:
            config_key = "arm1_baseline"
        elif "dual_cap" in fname or "dual-cap" in fname:
            config_key = "arm2_dual_cap"
        elif "trust_region" in fname or "trust-region" in fname:
            # Extract damping, scale_floor, radius
            # Format: arm3_trust_region_d1e-01_f1e-04_r0.1_obs0_seed98301
            d_parts = [p for p in parts if p.startswith("d") and "e-" in p]
            f_parts = [p for p in parts if p.startswith("f") and "e-" in p]
            r_parts = [p for p in parts if p.startswith("r") and ("." in p or p[1:].replace(".", "").isdigit())]

            if d_parts and f_parts and r_parts:
                damping = d_parts[0]
                scale_floor = f_parts[0]
                radius = r_parts[0]
                config_key = f"arm3_trust_region_{damping}_{scale_floor}_{radius}"
            else:
                config_key = "arm3_trust_region_unknown"
        else:
            config_key = "unknown"

        arm_configs[config_key].append(data)

    print(f"\nFound {len(arm_configs)} unique configurations:")
    for key in sorted(arm_configs.keys()):
        print(f"  {key}: {len(arm_configs[key])} runs")

    # Find best trust-region config using minimal intervention criterion
    # (lowest cap fire rate, then lowest damping, then smallest radius)
    baseline_results = arm_configs.get("arm1_baseline", [])
    dual_cap_results = arm_configs.get("arm2_dual_cap", [])

    trust_region_configs = {}
    for key, results in arm_configs.items():
        if "trust_region" in key:
            # Calculate mean objective and cap metrics
            objectives = [r["value"] for r in results]
            cap_rates = [r.get("fraction_coordinatewise_cap_active", 0.0) for r in results]

            mean_obj = sum(objectives) / len(objectives)
            mean_cap_rate = sum(cap_rates) / len(cap_rates)

            # Extract config params from key: arm3_trust_region_d1e-01_f1e-04_r0.1
            parts = key.split("_")
            d_parts = [p for p in parts if p.startswith("d") and "e-" in p]
            f_parts = [p for p in parts if p.startswith("f") and "e-" in p]
            r_parts = [p for p in parts if p.startswith("r") and ("." in p or p[1:].replace(".", "").isdigit())]

            if not (d_parts and f_parts and r_parts):
                continue

            damping_str = d_parts[0][1:]  # Remove 'd' prefix
            scale_floor_str = f_parts[0][1:]  # Remove 'f' prefix
            radius_str = r_parts[0][1:]  # Remove 'r' prefix

            # Convert to float
            damping = float(damping_str.replace("e-0", "e-").replace("e-", "e-0"))
            scale_floor = float(scale_floor_str.replace("e-0", "e-").replace("e-", "e-0"))
            radius = float(radius_str)

            trust_region_configs[key] = {
                "config": key,
                "damping": damping,
                "scale_floor": scale_floor,
                "radius": radius,
                "mean_objective": mean_obj,
                "mean_cap_rate": mean_cap_rate,
                "num_runs": len(results)
            }

    # Select by minimal intervention (lowest cap rate first)
    selected = min(trust_region_configs.values(), key=lambda x: (x["mean_cap_rate"], x["damping"], x["radius"]))

    print(f"\nSelected configuration:")
    print(f"  Config: {selected['config']}")
    print(f"  Damping: {selected['damping']}")
    print(f"  Scale floor: {selected['scale_floor']}")
    print(f"  Radius: {selected['radius']}")
    print(f"  Mean objective: {selected['mean_objective']:.3f}")
    print(f"  Mean cap fire rate: {selected['mean_cap_rate']:.3f}")

    # Create result.json
    result = {
        "schema": "ledh_trust_region_tuning.phase3_result.v1",
        "model": "predator_prey_T20",
        "date": "2026-09-03",
        "salvaged": "2026-09-04",
        "note": "Reconstructed from 116 evaluation files after Phase 1 campaign exited without writing summary",
        "selected_config": {
            "damping": selected["damping"],
            "scale_floor": selected["scale_floor"],
            "radius": selected["radius"]
        },
        "selection_criterion": "minimal_intervention",
        "baseline_mean_objective": sum(r["value"] for r in baseline_results) / len(baseline_results) if baseline_results else None,
        "dual_cap_mean_objective": sum(r["value"] for r in dual_cap_results) / len(dual_cap_results) if dual_cap_results else None,
        "selected_mean_objective": selected["mean_objective"],
        "selected_mean_cap_rate": selected["mean_cap_rate"],
        "total_configs_evaluated": len(trust_region_configs),
        "total_evaluation_files": len(all_files)
    }

    output_path = ARTIFACT_DIR / "result.json"
    with open(output_path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\nWrote: {output_path}")
    print("Phase 1 artifact salvage complete!")

if __name__ == "__main__":
    main()
