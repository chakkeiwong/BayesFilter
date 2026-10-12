"""Frozen-proposal research consumer of the two public HMC tuning authorities.

This module owns only preflight and complete reporting/export of a declared
cohort. It contains no acceptance, numerical repair or posterior selection rule.
"""
from __future__ import annotations

import json
from pathlib import Path


def validate_frozen_search(binding, search):
    """Validate an explicit cohort before creating output or running a target."""
    from bayesfilter.inference.hmc_candidate_set_execution import HMCCandidateExecutionBinding
    from bayesfilter.inference.hmc_candidate_set_tuning import HMCControllerConfig

    if not isinstance(binding, HMCCandidateExecutionBinding) or not isinstance(search, HMCControllerConfig):
        raise TypeError("typed execution binding and controller configuration required")
    if search.replicated_acceptance_policy != binding.config.acceptance_policy:
        raise ValueError("search and binding qualification policies differ")
    if (search.pilot_enabled or search.refinement_rounds or search.refinement_l_grid
            or search.expansion_l_grid or search.explore_failed_intervals):
        raise ValueError("frozen consumer requires explicit proposals and only declared repairs")
    pairs = tuple((length, epsilon) for length, values in search.epsilon_by_l for epsilon in values)
    lower, upper = binding.scope.epsilon_domain
    if any(not lower <= epsilon <= upper for _, epsilon in pairs):
        raise ValueError("proposal epsilon outside the binding's declared domain")
    required = len(pairs) * (1 + binding.scope.max_repairs_per_family)
    if search.max_candidates < required:
        raise ValueError("candidate cap must reserve every declared family repair opportunity")
    if (search.replicated_acceptance_policy is None
            or search.replicated_acceptance_policy.max_candidates < search.max_candidates):
        raise ValueError("replicated error allowance cannot cover the candidate cap")
    binding.validate()
    return pairs


def candidate_inventory(result, primary_pairs):
    """Keep actual identities and uncreated roots distinct after any outcome.

    A descendant is not matched by its numerical pair: another primary family
    may already contain that pair. Its parent chain defines its lineage.
    """
    planned = tuple(tuple(pair) for pair in primary_pairs)
    if not planned or len(set(planned)) != len(planned):
        raise ValueError("distinct nonempty primary pairs required")
    by_id = {candidate.candidate_id: candidate for candidate in result.candidates}
    if len(by_id) != len(result.candidates):
        raise ValueError("duplicate created candidate identity")
    verified = set(result.verified_candidate_ids)
    if len(verified) != len(result.verified_candidate_ids) or not verified <= set(by_id):
        raise ValueError("invalid verified candidate inventory")
    roots = {}
    for candidate in result.candidates:
        if candidate.parent_candidate_id is None:
            pair = (candidate.leapfrog_steps, candidate.epsilon)
            if pair not in planned or pair in roots:
                raise ValueError("unrequested or duplicate primary candidate")
            roots[pair] = candidate.candidate_id
    rows = []
    for candidate in result.candidates:
        ancestor = candidate
        visited = {candidate.candidate_id}
        while ancestor.parent_candidate_id is not None:
            parent = by_id.get(ancestor.parent_candidate_id)
            if (parent is None or parent.candidate_id in visited
                    or parent.candidate_family_id != candidate.candidate_family_id
                    or parent.creation_ordinal >= ancestor.creation_ordinal):
                raise ValueError("invalid repair ancestry")
            visited.add(parent.candidate_id)
            ancestor = parent
        state = result.candidate_states[candidate.candidate_id]
        if (state == "verified") != (candidate.candidate_id in verified):
            raise ValueError("verified state and membership disagree")
        rows.append(dict(candidate_id=candidate.candidate_id,
            parent_candidate_id=candidate.parent_candidate_id,
            candidate_family_id=candidate.candidate_family_id, primary_candidate_id=ancestor.candidate_id,
            leapfrog_steps=candidate.leapfrog_steps, epsilon=candidate.epsilon,
            transition_spec=(None if candidate.transition_spec is None else candidate.transition_spec.payload()),
            tuning_state=state,
            status="posterior_not_started" if candidate.candidate_id in verified else "not_verified"))
    return dict(rows=rows, requested_primary_pairs=planned,
        uncreated_primary_pairs=[pair for pair in planned if pair not in roots],
        proposal_count=len(planned), created_primary_count=len(roots),
        candidate_count=len(result.candidates), repair_descendant_count=len(result.candidates)-len(roots))


def run_frozen_trajectory_search(binding, search, *, output_dir, proposal_metadata=None):
    """Run fresh public qualification and reload every verified root or child.

    ``proposal_metadata`` is descriptive provenance only. The executed search
    and public result own parameters and membership. Posterior assessment is
    left to the caller and cannot alter these records.
    """
    from bayesfilter.inference import tune_hmc_kernel, tune_fixed_transport_hmc_kernel
    from bayesfilter.inference.hmc_candidate_set_retained import (
        export_hmc_candidate_retained_runners, load_hmc_candidate_retained_runners)

    pairs = validate_frozen_search(binding, search)
    design = dict(search=search.payload(), binding_hash=binding.binding_hash,
        requested_primary_pairs=pairs, proposal_count=len(pairs),
        proposal_metadata=proposal_metadata,
        pilot_draws_reused_for_qualification=False,
        posterior_convergence_authority=False, no_nominee=True)
    encoded = json.dumps(design, indent=2, allow_nan=False)+"\n"
    directory = Path(output_dir)
    # A caller may already have written a discarded pilot under this root.
    # Preserve its layout and all absolute checkpoint paths. Never overwrite
    # an earlier qualification, even if it was interrupted before tuning.
    for name in ("search-design.json", "tuning", "members", "result.json"):
        if (directory/name).exists():
            raise FileExistsError(directory/name)
    directory.mkdir(parents=True, exist_ok=True)
    (directory/"search-design.json").write_text(encoded)
    common = dict(initial_position=binding.initial_active_state, config=search,
                  candidate_set_adapter=binding.typed_adapter, output_dir=directory/"tuning")
    if binding._spec["adapter_kind"] != "fixed_transport":
        run = tune_hmc_kernel(adapter=binding._base_adapter, **common)
    else:
        run = tune_fixed_transport_hmc_kernel(base_adapter=binding._base_adapter,
                                             fixed_transport=binding.fixed_transport, **common)
    inventory = candidate_inventory(run.result, pairs)
    paths = export_hmc_candidate_retained_runners(candidate_set_result=run.result,
        retained_binding=binding, output_dir=directory/"members")
    if set(paths) != set(run.result.verified_candidate_ids):
        raise ValueError("retained export omitted verified candidates")
    members = load_hmc_candidate_retained_runners(tuple(paths.values()), adapter=binding._base_adapter)
    if set(members) != set(run.result.verified_candidate_ids):
        raise ValueError("retained export/reload omitted verified candidates")
    summary = dict(**inventory, completion_status=run.result.completion_status,
        verified_candidate_ids=list(run.result.verified_candidate_ids),
        candidate_states=dict(run.result.candidate_states),
        repair_count=len(run.result.repair_actions),
        repair_actions=[action.payload() for action in run.result.repair_actions],
        no_nominee=True, qualification_gradient_work=run.result.search_state["gradient_work"],
        posterior_assessment="not_run", release_qualified=False)
    (directory/"result.json").write_text(json.dumps(summary, indent=2, allow_nan=False)+"\n")
    return run, members
