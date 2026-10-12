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


def build_designs(n5, source):
    """Must run with source first on sys.path, before any library import."""
    from bayesfilter.inference import HMCControllerConfig, HMCCandidateExecutionConfig
    from bayesfilter.inference.hmc_acceptance_protocol import HMCReplicatedAcceptancePolicy
    from bayesfilter.inference.hmc_transition_spec import HMCTransitionProfile
    from bayesfilter.inference.hmc_trajectory_duration import compile_mean_duration
    from bayesfilter.testing.inference_validation.modern_tuning_consumer import (
        frozen_search_evidence_preflight, require_attainable_qualification)
    from scripts.run_modern_hmc_frozen_full_model import posterior_factory

    if not Path(sys.modules[HMCControllerConfig.__module__].__file__).resolve().is_relative_to(source):
        raise ValueError("comparison must use the declared immutable numerical source")
    n5 = Path(n5)
    p = n5 / "generic-adaptation/preconfirmation-01"
    snaper_path = n5 / "method-proposals-02/nl-pp-sgqf-snaper/discarded-pilot.json"
    mams_path = p / "window-256/NL-PP-SGQF/pilot/pilot-result.json"
    snaper, mams = read(snaper_path), read(mams_path)
    geometry = n5 / "full-model-geometry-01/nl-pp-sgqf"
    reference = p / "reference-consumer-05/NL-PP-SGQF.json"
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
            pilot = mams if method == "mams" else snaper
            path = mams_path if method == "mams" else snaper_path
            pairs = [(c["max_steps"], c["epsilon"]) for c in pilot["candidates"]]
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
        execution = HMCCandidateExecutionConfig(measurement_num_results=64, verification_num_results=64,
            num_warmup_steps=0, seed=(20261012, 71410+1000*index), acceptance_policy=policy,
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
        posterior_factory(settings, target="NL-PP-SGQF", use_xla=True)
        designs[method] = dict(schema="bayesfilter.modern_frozen_full_model_design.v1",
            plan=PLAN, target="NL-PP-SGQF", method=method, seed=71410,
            geometry_directory=str(geometry), execution=execution.payload(), search=search.payload(),
            epsilon_domain=[.001,4.], repair_factor=1.5, max_repairs_per_family=1,
            proposal_metadata=proposal, posterior=settings, reference_compatibility=str(reference))
        preflights[method] = check
    return designs, preflights


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n5", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args=parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
        raise ValueError("design preflight must intentionally hide GPUs")
    source = args.source_root.resolve()
    sys.path.insert(0, str(source))
    started=time.monotonic()
    designs, preflights=build_designs(args.n5.resolve(), source)
    args.output.mkdir(parents=True, exist_ok=False)
    for method, design in designs.items():
        save(args.output/(method+".json"), design)
    save(args.output/"preflight.json", preflights)
    save(args.output/"manifest.json", dict(plan=PLAN, command=[sys.executable,*sys.argv],
        source=read(source/".validation-source.json"), driver_sha256=digest(__file__),
        wall_seconds=time.monotonic()-started, gpu_intentionally_hidden=True,
        design_sha256={name:digest(args.output/(name+".json")) for name in designs},
        role="design_preflight_only", statistical_power_established=False))


if __name__ == "__main__":
    main()
