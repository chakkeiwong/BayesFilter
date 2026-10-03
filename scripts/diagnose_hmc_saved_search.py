"""Development reconstruction of a saved search, never historical replay authority.

Preserves frozen numerical inputs but issues new current-source identities and
streams. The two domains use the same public candidate controller and evidence
policy. No historical receipt is loaded as qualification.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))


def reconstruct(saved, destination, *, seconds_per_arm, cpu_reference=False):
    from bayesfilter.inference import HMCControllerConfig, HMCCandidateExecutionConfig, tune_hmc_kernel
    from bayesfilter.inference.hmc_candidate_set_execution import _issue_binding, _tensor_from_payload
    from bayesfilter.inference.hmc_candidate_set_tuning import _sha256
    from bayesfilter.inference.hmc_preparation import expanded_preparation_bound
    from bayesfilter.testing.inference_validation.targets import ValidationTarget
    from bayesfilter.testing.inference_validation.storage import read_json, write_json, file_hash
    from bayesfilter.testing.inference_validation.execution import source_state

    saved, destination = Path(saved), Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    wrapper = read_json(saved / "execution_spec.json")
    spec = wrapper["execution"]
    if _sha256(spec) != wrapper["binding_hash"]:
        raise ValueError("saved geometry checksum mismatch")
    original = read_json(saved / "candidate_set_result.json")
    from bayesfilter.inference.hmc_candidate_set_artifacts import load_candidate_set_result_payload
    load_candidate_set_result_payload(saved / "candidate_set_result.json")
    if original["verified_candidate_ids"]:
        raise ValueError("this diagnostic requires a saved empty-member search")
    lineage = spec["target_lineage"]
    target = ValidationTarget(lineage["model"], lineage["prior"], lineage["data"],
                              control=lineage["control"], jit_compile=False)
    if target.adapter_signature() != spec["scope"]["target_signature"]:
        raise ValueError("saved target differs from reconstructed target")
    execution = replace(HMCCandidateExecutionConfig.from_payload(spec["config"]),
                        preparation_elapsed_seconds=0.)
    # A saved GPU/XLA binding cannot be replayed in a CPU reference process.
    # Keep its target, frozen geometry and starts, but issue a new explicitly
    # non-XLA execution identity; this is diagnostic evidence only.
    if execution.use_xla and not cpu_reference:
        raise ValueError("saved GPU/XLA reconstruction requires --cpu-reference for a CPU diagnostic")
    if execution.use_xla:
        execution = replace(execution, use_xla=False,
                            non_xla_reason="explicit CPU saved-search diagnostic")
    search = replace(HMCControllerConfig.from_payload(original["config"]), max_wall_time_seconds=seconds_per_arm)
    old_domain = tuple(spec["scope"]["epsilon_domain"])
    started = time.monotonic()
    rows = []
    for steps in (0, 1):
        domain = (old_domain[0], expanded_preparation_bound(old_domain[1],
            factor=spec["scope"]["repair_factor"], steps=steps))
        binding = _issue_binding(adapter=target, layers=spec["layers"],
            initial_active_state=_tensor_from_payload(spec["initial_active_state"]),
            target_scope=spec["target_scope"], target_lineage=lineage,
            preparation={"source":"saved_geometry_current_source_development_reconstruction",
                         "original_binding_hash":wrapper["binding_hash"], "domain_expansion_steps":steps},
            config=execution, source_paths=[__file__], scope_id="saved-search-development",
            search_id="domain-" + str(steps), epsilon_domain=domain,
            repair_factor=spec["scope"]["repair_factor"],
            max_repairs_per_family=spec["scope"]["max_repairs_per_family"])
        result = tune_hmc_kernel(adapter=target, initial_position=binding.initial_active_state,
            config=search, candidate_set_adapter=binding.typed_adapter,
            output_dir=destination / ("domain-" + str(steps))).result
        rows.append({"expansion_steps":steps, "domain":domain,
            "verified_candidate_ids":result.verified_candidate_ids, "completion":result.completion_status,
            "candidates":len(result.candidates), "mass_signature":binding.scope.mass_signature,
            "start_bank_signature":binding.scope.start_bank_signature,
            "search_id":binding.scope.search_id})
    payload = {"saved_root":str(saved), "rows":rows, "source":source_state(),
        "input_sha256":{name:file_hash(saved/name) for name in ("execution_spec.json","candidate_set_result.json")},
        "elapsed_seconds":time.monotonic()-started, "command":sys.argv,
        "plan_file":"docs/plans/bayesfilter-hmc-repair-master-program-2026-09-16.md",
        "device":"CPU reference; GPU deliberately hidden; non-XLA",
        "interpretation":"same frozen geometry and proposals, new identities/streams; development only",
        "default_readiness":False, "ranking_supported":False}
    write_json(destination / "comparison.json", payload)
    print(json.dumps({k:payload[k] for k in ("rows","elapsed_seconds","interpretation")}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("saved", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds-per-arm", type=float, default=400.)
    parser.add_argument("--cpu-reference", action="store_true",
                        help="explicitly convert a saved GPU/XLA binding to a non-XLA CPU diagnostic")
    args = parser.parse_args()
    reconstruct(args.saved, args.output, seconds_per_arm=args.seconds_per_arm,
                cpu_reference=args.cpu_reference)


if __name__ == "__main__":
    main()
