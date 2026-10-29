"""Bounded diagnostic of the existing ordinary preparation startup option.

Run against a declared frozen numerical source. This helper never issues a
candidate-set artifact or makes a posterior claim.
"""
from pathlib import Path
import argparse
import json
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("design", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--rounds", type=int, required=True)
    parser.add_argument("--preparation-restarts", type=int, default=0)
    args = parser.parse_args()
    sys.path.insert(0, str(args.source.resolve()))
    from bayesfilter.testing.inference_validation.designs import ValidationDesign, seed_for
    from bayesfilter.testing.inference_validation.execution import configure_worker, source_state
    from bayesfilter.testing.inference_validation.storage import read_json, write_json
    design = ValidationDesign.from_payload(read_json(args.design))
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    runtime = configure_worker(design)
    from bayesfilter.testing.inference_validation.targets import ValidationTarget
    from bayesfilter.testing.inference_validation.procedures import initial_starts
    from bayesfilter.inference import HMCKernelTuningConfig, HMCAcceptancePolicy
    from bayesfilter.inference.hmc_preparation import (
        prepare_operational_windowed_mass_handoff, _progress_json_value,
    )
    from bayesfilter.inference.hmc_preparation_common import _json_ready
    target = ValidationTarget(design.scenario.target, design.scenario.parameters,
        design.options["data"], control=design.scenario.control, jit_compile=design.device == "gpu")
    seed = seed_for(design.seed, design.design_id, 0, 0, "tuning")
    policy = HMCAcceptancePolicy(**design.options.get("acceptance_policy", {}))
    config = HMCKernelTuningConfig(
        preset=design.options.get("preparation_preset", "standard"),
        use_xla=design.device == "gpu", target_scope="inference_validation", seed=seed,
        target_status_trace_policy="per_chain_step",
        candidate_search_bound_expansion_steps=design.options.get("preparation_bound_expansion_steps", 0),
        bootstrap_initialization_rounds=args.rounds,
        metric_evidence_policy=design.options.get("metric_evidence_policy", "temporal_information"),
        metric_probe_num_results=design.options.get("metric_probe_num_results", 1),
        preparation_max_restarts=args.preparation_restarts,
        target_accept_prob=policy.target, acceptance_band=policy.practical_region,
        repair_band=policy.repair_region)
    manifest = {"command": [sys.executable, *sys.argv], "source": source_state(),
        "original_design": design.payload(), "runtime": runtime, "seed": seed,
        "bootstrap_initialization_rounds": args.rounds,
        "preparation_max_restarts": args.preparation_restarts,
        "plan_file": "docs/plans/bayesfilter-hmc-ssm-pilot-repair-2026-09-29.md",
        "result_file": str(args.output / "result.json"), "posterior_claim": False}
    write_json(args.output / "manifest.json", manifest)
    events = []

    def progress(stage, payload):
        events.append({"stage": stage, "elapsed_seconds": time.monotonic() - started,
            "details": _progress_json_value(_json_ready(payload))})
        write_json(args.output / "progress.json", {"events": events})

    try:
        prepared = prepare_operational_windowed_mass_handoff(adapter=target,
            initial_position=initial_starts(target, design.scenario.start)[0],
            config=config, progress_callback=progress)
        result = {"status": "prepared", "handoff_keys": sorted(prepared),
                  "posterior_claim": False, "artifact_authority": False}
        code = 0
    except Exception as exc:
        result = {"status": "failed", "exception": type(exc).__name__, "reason": str(exc),
            "details": _progress_json_value(_json_ready(getattr(exc, "details", {}))),
            "posterior_claim": False, "artifact_authority": False}
        code = 1
    result["elapsed_seconds"] = time.monotonic() - started
    write_json(args.output / "result.json", result)
    print(json.dumps({key: result[key] for key in ("status", "elapsed_seconds")}))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
