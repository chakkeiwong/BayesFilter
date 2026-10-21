"""Scoped mixing hypotheses change nomination only, never posterior criteria."""
from dataclasses import asdict
import pytest
from bayesfilter.testing.surrogate_hmc_m13_public_tuning import tuning_config
from bayesfilter.testing.surrogate_hmc_m13_mixing_config import mixing_repair_config


@pytest.mark.parametrize("attempt,target,band", [(1,.70,(.50,.85)), (2,.70,(.50,.85))])
def test_repair_is_scoped_to_nomination_and_disjoint_seeds(attempt,target,band):
    original = tuning_config("M13-scope-fixture")
    repaired, metadata = mixing_repair_config(original, attempt)
    changed = {key for key, value in asdict(original).items()
               if value != asdict(repaired)[key]}
    assert changed == {"nomination", "fresh_seed"}
    assert repaired.nomination.target_accept_prob == target
    assert repaired.nomination.acceptance_policy.overall_band == band
    assert repaired.nomination.acceptance_policy.per_chain_band == (.05,1.)
    assert repaired.nomination.verification_repair_rounds == 2
    assert repaired.nomination.seed != repaired.fresh_seed
    assert repaired.nomination.seed != original.nomination.seed
    assert repaired.nomination.target_scope == original.nomination.target_scope
    assert repaired.nomination.max_leapfrog_steps == original.nomination.max_leapfrog_steps
    assert repaired.nomination.mass_window_results == original.nomination.mass_window_results
    assert repaired.nomination.mass_shrinkage == original.nomination.mass_shrinkage
    assert repaired.warmup_results == 2000 and repaired.verification_results == 1000
    assert repaired.warmup_rhat_max == 1.05
    assert repaired.warmup_max_results == repaired.verification_max_results == 10000
    assert metadata["force_changed"] is False and metadata["endpoint_changed"] is False
    assert metadata["posterior_qualified"] is False
    assert metadata["step_multipliers"] == ([.125, 1.] if attempt == 1 else [.25, 1.])
    assert metadata["step_probabilities"] == [.5, .5]
    assert repaired.payload()["nomination"]["target_accept_prob"] == target


@pytest.mark.parametrize("attempt", [True, False, 0, 3, None])
def test_invalid_attempt_cannot_silently_choose_hypothesis(attempt):
    with pytest.raises(ValueError):
        mixing_repair_config(tuning_config("fixture"), attempt)
