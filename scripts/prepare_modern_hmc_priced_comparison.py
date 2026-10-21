#!/usr/bin/env python3
"""Freeze a scoped comparison; use the existing numerical snapshot's types."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import sys
import time

PLAN = "docs/plans/bayesfilter-modern-hmc-priced-comparison-2026-10-12.md"
METHODS = ("fixed_hmc", "randomized_hmc", "snaper", "mams")
TARGETS = ("NL-PP-SGQF", "NL-PP-UKF", "NL-SIR", "NL-STR")


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, data):
    with Path(path).open("x") as out:
        out.write(json.dumps(data, indent=2, sort_keys=True, allow_nan=False)+"\n")


def group_pairs(pairs):
    pairs = sorted(set(pairs))
    return tuple((length, tuple(e for l, e in pairs if l == length))
                 for length in sorted({l for l, _ in pairs}))


def scoped_seed(*, target, method, seed, namespace, stage):
    """Deterministic streams independent of cell ordering and target subsets."""
    if (target not in TARGETS or method not in METHODS or stage not in {"tuning", "posterior"}
            or type(seed) is not int or not 0 <= seed < 2**31
            or type(namespace) is not int or not 0 <= namespace < 2**31):
        raise ValueError("supported scope and explicit integer seed namespace required")
    payload = json.dumps([namespace, target, method, seed, stage], separators=(",", ":"))
    raw = hashlib.sha256(payload.encode()).digest()
    # Leave room for the posterior driver's per-member offsets.
    return (int.from_bytes(raw[:4], "big") & 0x7fffffff,
            int.from_bytes(raw[4:8], "big") & 0x3fffffff)


def comparison_inputs(n5, *, target, snaper_path=None, mams_path=None):
    """Require the pilots' actual target/data/geometry, not a caller's label."""
    if target not in TARGETS:
        raise ValueError("unsupported full-model target")
    n5 = Path(n5)
    preconfirmation = n5 / "generic-adaptation/preconfirmation-01"
    parent = "full-model-geometry-02-sir" if target == "NL-SIR" else "full-model-geometry-01"
    geometry = n5 / parent / target.lower()
    reference = preconfirmation / "reference-consumer-05" / (target + ".json")
    snaper_path = Path(snaper_path) if snaper_path else (
        n5 / "method-proposals-02" / (target.lower() + "-snaper") / "discarded-pilot.json")
    if mams_path is None:
        if target == "NL-PP-SGQF":
            mams_path = preconfirmation / "window-256" / target / "pilot/pilot-result.json"
        elif target in {"NL-SIR", "NL-STR"}:
            mams_path = n5 / "generic-adaptation/full-model-pricing-02" / target / "pilot/pilot-result.json"
        else:
            raise ValueError("an actual target-specific MAMS pilot path is required")
    mams_path = Path(mams_path)
    geometry_design = read(geometry / "design.json")
    reference_payload = read(reference)
    if (geometry_design["target"] != target or reference_payload["target"] != target
            or reference_payload["status"] != "compatible"):
        raise ValueError("target geometry/reference scope mismatch")
    expected = dict(data_sha256=geometry_design["metadata"]["data_sha256"],
                    geometry_sha256=digest(geometry / "geometry.json"))
    pilot_design_paths = (snaper_path.parent / "design.json", mams_path.parent.parent / "design.json")
    for method, path in zip(("snaper", "mams"), pilot_design_paths):
        design = read(path)
        scope = design.get("metadata", design)
        if (design["target"] != target or any(scope.get(k) != v for k, v in expected.items())
                or design.get("method", method) != method
                or scope.get("horizon") != (100 if target == "NL-STR" else 20)
                or scope.get("parameter_dim") != (5 if target == "NL-STR" else 3 if target == "NL-SIR" else 6)):
            raise ValueError("discarded pilot target/data/geometry scope mismatch: " + str(path))
    pilots = {"snaper": read(snaper_path), "mams": read(mams_path)}
    discarded = pilots["snaper"].get("adaptation_draws_discarded")
    # SNAPER records the number discarded, whereas MAMS records a Boolean.
    if (type(discarded) is not int or discarded <= 0
            or pilots["mams"].get("adaptation_and_pilot_draws_discarded") is not True
            or any(p.get("tuning_artifact_authority") is not False
                   or p.get("fresh_measurement_and_verification_required") is not True
                   for p in pilots.values())):
        raise ValueError("only discarded, non-authoritative pilots can propose roots")
    if pilots["mams"]["binding_hash"] != read(pilot_design_paths[1])["binding_hash"]:
        raise ValueError("MAMS pilot differs from its recorded target binding")
    inputs = (geometry / "design.json", geometry / "geometry.json", reference,
              snaper_path, mams_path, *pilot_design_paths)
    return geometry, reference, pilots, {"snaper": snaper_path, "mams": mams_path}, {
        str(path): digest(path) for path in inputs}


def posterior_settings(method_index):
    return dict(sequential=dict(warmup_chunk_results=1000, warmup_min_results=2000,
        warmup_check_window_results=1000, warmup_max_results=10000, warmup_rhat_max=1.05,
        retained_chunk_results=1000, retained_min_results=1000, retained_max_results=10000,
        retained_rhat_max=1.01, minimum_chain_count=4, energy_error_log_accept_threshold=-1000.,
        max_results_per_chain=10000, count_budget_reason=None),
        precision=dict(method="lugsail", batch_size=None, min_batches=20, lugsail_r=3, lugsail_c=.5),
        assessment=dict(warmup_bulk_ess_min=0., warmup_tail_ess_min=0., warmup_consecutive_checks=1,
                        retained_bulk_ess_min=400., retained_tail_ess_min=400.),
        seed=(20261012, 900000+1000*method_index), mcse_sd_ratio_max=.02,
        reference_mcse_multiplier=3., reference_normalized_tolerance=.1)


def repair_settings(method):
    if method not in METHODS:
        raise ValueError("unsupported comparison method")
    # The current controller deliberately restricts physical-duration repair
    # to uniform trajectories. Fixed HMC retains its supported count repair.
    return (dict(repair_trajectory_policy="fixed_count") if method=="fixed_hmc" else
            dict(repair_trajectory_policy="preserve_mean_duration", repair_mean_duration_max_steps=128))


def build_designs(n5, source, *, target="NL-PP-SGQF", seed=71410, seed_namespace=None,
                  plan=PLAN, snaper_path=None, mams_path=None):
    """Must run with source first on sys.path, before any library import."""
    from bayesfilter.inference import HMCControllerConfig, HMCCandidateExecutionConfig
    from bayesfilter.inference.hmc_acceptance_protocol import HMCReplicatedAcceptancePolicy
    from bayesfilter.inference.hmc_transition_spec import FrozenTransitionSpec, HMCTransitionProfile
    from bayesfilter.inference.hmc_trajectory_duration import compile_mean_duration
    from bayesfilter.testing.inference_validation.modern_tuning_consumer import (
        frozen_search_evidence_preflight, require_attainable_qualification)
    from scripts.run_modern_hmc_frozen_full_model import posterior_factory

    if not Path(sys.modules[HMCControllerConfig.__module__].__file__).resolve().is_relative_to(source):
        raise ValueError("comparison must use the declared immutable numerical source")
    if (target != "NL-PP-SGQF" or seed != 71410) and seed_namespace is None:
        raise ValueError("a new target or start seed requires an explicit random namespace")
    geometry, reference, pilots, paths, input_hashes = comparison_inputs(
        n5, target=target, snaper_path=snaper_path, mams_path=mams_path)
    designs, preflights = {}, {}
    for index, method in enumerate(METHODS):
        profile = HMCTransitionProfile("mams" if method == "mams" else "hmc",
            "fixed" if method == "fixed_hmc" else "uniform_duration")
        proposal = dict(role="frozen_development_hypotheses", no_posterior_selection=True,
                        adaptation_replication=False)
        if method in {"fixed_hmc", "randomized_hmc"}:
            compiled = [compile_mean_duration(profile=profile, epsilon=epsilon,
                requested_mean_duration=duration, max_steps_cap=128)
                for duration in (1., 2., 4., 8.) for epsilon in (.5, 1.)]
            pairs = [(c.frozen.max_steps, c.frozen.epsilon) for c in compiled]
            proposal.update(source="explicit classical duration/epsilon coverage",
                requested_mean_durations=[1., 2., 4., 8.], epsilons=[.5, 1.],
                compilations=[c.payload() for c in compiled],
                source_role="convenience geometric search hypotheses, not globally optimal tuning")
        else:
            pilot, path = pilots[method], paths[method]
            specs = [FrozenTransitionSpec.from_payload(c) for c in pilot["candidates"]]
            if not specs or any(c.profile != profile for c in specs):
                raise ValueError("discarded pilot transition profile disagrees with method")
            pairs = [(c.max_steps, c.epsilon) for c in specs]
            proposal.update(source=str(path), source_sha256=digest(path),
                source_role="actual target-specific discarded pilot; no prior qualification reused",
                pilot_config=pilot["config"], omissions=pilot["omitted_candidates"])
        groups = group_pairs(pairs)
        roots = sum(len(values) for _, values in groups)
        if roots != len(pairs):
            raise ValueError("unexpected duplicate proposed policy")
        cap = roots*2
        policy = HMCReplicatedAcceptancePolicy(trial_num_results=64, discarded_prefix=0,
            base_repetitions=32, max_repetitions=128, max_candidates=cap,
            search_family_alpha=.05, verification_family_alpha=.05, diagnostic_family_alpha=.05,
            temporal_tolerance=.1, qualification_mode="lower_bound", qualification_region=(.5, 1.),
            min_normalized_return_displacement=0.)
        tuning_seed = ((20261012, 71410+1000*index) if seed_namespace is None else
            scoped_seed(target=target, method=method, seed=seed, namespace=seed_namespace, stage="tuning"))
        execution = HMCCandidateExecutionConfig(measurement_num_results=64, verification_num_results=64,
            num_warmup_steps=0, seed=tuning_seed, acceptance_policy=policy,
            target_status_trace_policy="per_chain_step", use_xla=True, chain_mode="batched",
            chunk_max_results=64, transition_profile=profile)
        # Six looks (3 per stage) each need one controller unit. The candidate
        # reserve covers all of them; a separate equal family reserve covers repairs.
        search = HMCControllerConfig(primary_l_grid=tuple(l for l, _ in groups), epsilon_by_l=groups,
            total_budget_units=cap*6, repair_reserve_units=roots*6, candidate_reserve_units=6,
            replicated_acceptance_policy=policy, evidence_rungs=(1,2,4), max_candidates=cap,
            **repair_settings(method),
            pilot_enabled=False, refinement_rounds=0, explore_failed_intervals=False)
        # Attainability is target-independent. The GPU worker separately checks
        # its actual compilation context; this CPU design check is not GPU evidence.
        from dataclasses import replace
        check = frozen_search_evidence_preflight(replace(execution, use_xla=False,
            non_xla_reason="CPU-only deterministic design preflight"), search)
        require_attainable_qualification(check)
        settings = posterior_settings(index)
        if seed_namespace is not None:
            settings["seed"] = scoped_seed(target=target, method=method, seed=seed,
                                           namespace=seed_namespace, stage="posterior")
        posterior_factory(settings, target=target, use_xla=True)
        designs[method] = dict(schema="bayesfilter.modern_frozen_full_model_design.v1",
            plan=plan, target=target, method=method, seed=seed,
            geometry_directory=str(geometry), execution=execution.payload(), search=search.payload(),
            epsilon_domain=[.001,4.], repair_factor=1.5, max_repairs_per_family=1,
            proposal_metadata=proposal, posterior=settings, reference_compatibility=str(reference))
        if seed_namespace is not None:
            designs[method].update(seed_namespace=seed_namespace, proposal_input_sha256=input_hashes)
        preflights[method] = check
    return designs, preflights


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n5", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target", choices=TARGETS, default="NL-PP-SGQF")
    parser.add_argument("--seed", type=int, default=71410)
    parser.add_argument("--seed-namespace", type=int)
    parser.add_argument("--plan", default=PLAN)
    parser.add_argument("--snaper-pilot", type=Path)
    parser.add_argument("--mams-pilot", type=Path)
    args=parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise ValueError("design preflight must intentionally hide GPUs")
    source = args.source_root.resolve()
    sys.path.insert(0, str(source))
    started=time.monotonic()
    designs, preflights=build_designs(args.n5.resolve(), source, target=args.target, seed=args.seed,
        seed_namespace=args.seed_namespace, plan=args.plan, snaper_path=args.snaper_pilot,
        mams_path=args.mams_pilot)
    args.output.mkdir(parents=True, exist_ok=False)
    for method, design in designs.items():
        save(args.output/(method+".json"), design)
    save(args.output/"preflight.json", preflights)
    save(args.output/"manifest.json", dict(plan=args.plan, command=[sys.executable,*sys.argv],
        source=read(source/".validation-source.json"), driver_sha256=digest(__file__),
        wall_seconds=time.monotonic()-started, gpu_intentionally_hidden=True,
        design_sha256={name:digest(args.output/(name+".json")) for name in designs},
        role="design_preflight_only", statistical_power_established=False))


if __name__ == "__main__":
    main()
