"""Explicit full-search configurations for scoped v7 release validation.

This diagnostic harness owns no tuning authority. Numerical choices are
development hypotheses recorded in the release plan, not library defaults.
"""
from __future__ import annotations

import math

from bayesfilter.inference.hmc_acceptance_protocol import HMCReplicatedAcceptancePolicy
from bayesfilter.inference.hmc_candidate_set_tuning import HMCControllerConfig
from .acceptance_validation_inventory import model_configuration


PRIMARY_L = (3, 5, 9, 13, 18, 25)


def full_search_configuration(case, *, seed, wall_seconds, classification="development",
                              replicated_trial_batch_size=1):
    if (type(replicated_trial_batch_size) is not int
            or not 1 <= replicated_trial_batch_size <= 32):
        raise ValueError("replicated_trial_batch_size must be an integer in [1, 32]")
    policy = HMCReplicatedAcceptancePolicy(trial_num_results=65, discarded_prefix=3,
        base_repetitions=32, max_repetitions=256, max_candidates=100,
        search_family_alpha=.05, verification_family_alpha=.05,
        diagnostic_family_alpha=.05, temporal_tolerance=.1,
        # A moving path can end near its starting draw by chance. Across many
        # independent trials the inherited endpoint veto compounds that event.
        # Keep endpoint distance as a report; adjacent movement, repeated states
        # and repeated short cycles retain their original health thresholds.
        # This explicit release profile does not change the library default.
        min_normalized_return_displacement=0.)
    configuration = model_configuration(case,policy_payload=policy.payload(),
        evidence_rungs=(1,2,4,8),seed=seed,wall_seconds=wall_seconds,
        classification=classification,expected_outcome="positive_delivery")
    # These geometries reproduce earlier development controls. Their earlier
    # one-pair delivery does not establish full-search delivery below.
    epsilon = configuration['epsilon_by_l'][0][1][0]
    if case == "lgssm_qr":
        epsilon = .95
        configuration['geometry'] = {"kind":"explicit_diagonal",
            "center":[math.atanh(.6/.999),math.log(.2)],"scale":[1.,1.]}
        configuration['active_starts'] = [[.5,-1.5],[.55,-1.45],[.45,-1.55],[.6,-1.4]]
    elif case == "nonlinear":
        epsilon = .15
        configuration['geometry'] = {"kind":"explicit_diagonal","center":[0.]*3,"scale":[1.]*3}
        configuration['active_starts'] = [[.6,.4,.7],[.65,.45,.75],[.55,.35,.65],[.7,.5,.8]]
    configuration.update(schema="bayesfilter.acceptance_model_config.v2",
        epsilon_by_l=[[l,[epsilon]] for l in PRIMARY_L], max_repairs_per_family=4)
    search = HMCControllerConfig(primary_l_grid=PRIMARY_L,
        epsilon_by_l=tuple((l,(epsilon,)) for l in PRIMARY_L),max_candidates=100,
        total_budget_units=1200,repair_reserve_units=200,evidence_rungs=(1,2,4,8),
        replicated_acceptance_policy=policy,refinement_rounds=1,
        epsilon_refinement_factors=(.8,1.25),refinement_l_grid=(4,7),
        max_wall_time_seconds=float(wall_seconds))
    configuration['search'] = search.payload()
    if replicated_trial_batch_size > 1:
        configuration['schema'] = 'bayesfilter.acceptance_model_config.v3'
        configuration['replicated_trial_batch_size'] = replicated_trial_batch_size
    configuration['provenance'] += (
        "; full-search release development: inherited six-L primary grid; M100; "
        "one refinement round at factors .8/1.25 and L4/7; four candidate-specific "
        "repair attempts with factor1.3; 32--256 trials; these are hypotheses, "
        "not frozen confirmation or automatic-preparation evidence; endpoint "
        "distance reporting-only per the October 3 counterexample audit; "
        "adjacent movement and repeated-path vetoes unchanged")
    if replicated_trial_batch_size > 1:
        configuration['provenance'] += (
            "; explicit independent-trial batching preserves each original TFP stream; "
            "batch size is an execution hypothesis, not a statistical default")
    return configuration
