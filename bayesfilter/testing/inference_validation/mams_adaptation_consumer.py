"""Generic research consumer: discarded MAMS pilot to existing public tuner.

No numerical search or acceptance rule lives here. Caller-owned preparation,
pilot controls and qualification budgets are explicit and remain separate.
Posterior delivery is assessed by the existing retained-member API.
"""
from dataclasses import replace
import json
from pathlib import Path


def run_adapted_mams_search(binding, pilot_config, search_template, *, output_dir,
                            resume_pilot_from=None):
    from bayesfilter.inference import tune_hmc_kernel, tune_fixed_transport_hmc_kernel
    from bayesfilter.inference.hmc_mams_trajectory_proposals import propose_mams_trajectory_candidates
    from bayesfilter.inference.hmc_candidate_set_retained import (
        export_hmc_candidate_retained_runners, load_hmc_candidate_retained_runners)
    from bayesfilter.inference.hmc_candidate_set_execution import HMCCandidateExecutionBinding
    from bayesfilter.inference.hmc_mams_trajectory_proposals import MAMSPilotConfig
    from bayesfilter.inference.hmc_candidate_set_tuning import HMCControllerConfig

    if not isinstance(binding, HMCCandidateExecutionBinding) or not isinstance(
            pilot_config, MAMSPilotConfig) or not isinstance(search_template, HMCControllerConfig):
        raise TypeError("typed binding, pilot configuration and controller configuration required")
    if search_template.repair_trajectory_policy != "preserve_mean_duration":
        raise ValueError("adaptive consumer requires explicit duration-preserving repair policy")
    if search_template.replicated_acceptance_policy != binding.config.acceptance_policy:
        raise ValueError("search and binding qualification policies differ")
    if search_template.max_candidates < pilot_config.max_candidates:
        raise ValueError("search candidate cap cannot cover the declared pilot cohort")
    if search_template.repair_mean_duration_max_steps != pilot_config.max_steps_cap:
        raise ValueError("pilot and repair count caps differ")
    if (search_template.max_candidates < pilot_config.max_candidates
            * (1 + binding.scope.max_repairs_per_family)):
        raise ValueError("search candidate cap must reserve every declared family repair opportunity")
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=False)
    pilot = propose_mams_trajectory_candidates(binding, pilot_config,
        output_dir=directory/"pilot", resume_from=resume_pilot_from)
    pairs = pilot["epsilon_by_l"]
    search = replace(search_template, primary_l_grid=tuple(k for k, _ in pairs),
                     epsilon_by_l=pairs)
    design = dict(pilot_purpose_digest=pilot["purpose_digest"],
        search=search.payload(), binding_hash=binding.binding_hash,
        proposal_count=len(pilot["candidates"]),
        pilot_reserved_gradient_work=pilot["reserved_gradient_work"],
        pilot_actual_gradient_work=pilot["actual_gradient_work"],
        pilot_draws_reused_for_qualification=False,
        posterior_convergence_authority=False, no_nominee=True)
    (directory/"search-design.json").write_text(json.dumps(design, indent=2, allow_nan=False)+"\n")
    common = dict(initial_position=binding.initial_active_state, config=search,
                  candidate_set_adapter=binding.typed_adapter, output_dir=directory/"tuning")
    if binding._spec["adapter_kind"] != "fixed_transport":
        run = tune_hmc_kernel(adapter=binding._base_adapter, **common)
    else:
        run = tune_fixed_transport_hmc_kernel(base_adapter=binding._base_adapter,
                                             fixed_transport=binding.fixed_transport, **common)
    paths = export_hmc_candidate_retained_runners(candidate_set_result=run.result,
        retained_binding=binding, output_dir=directory/"members")
    members = load_hmc_candidate_retained_runners(tuple(paths.values()), adapter=binding._base_adapter)
    if set(members) != set(run.result.verified_candidate_ids):
        raise ValueError("retained export/reload omitted verified candidates")
    summary = dict(completion_status=run.result.completion_status,
        proposal_count=len(pilot["candidates"]), candidate_count=len(run.result.candidates),
        verified_candidate_ids=list(run.result.verified_candidate_ids),
        candidate_states=dict(run.result.candidate_states),
        repair_count=len(run.result.repair_actions), no_nominee=True,
        pilot_reserved_gradient_work=pilot["reserved_gradient_work"],
        qualification_gradient_work=run.result.search_state["gradient_work"],
        posterior_assessment="not_run", release_qualified=False)
    (directory/"result.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    return pilot, run, members
