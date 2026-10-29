"""M13-scoped fixed step mixtures; public tuning and convergence checks unchanged."""
from dataclasses import replace
from bayesfilter.inference.hmc_tensorflow_tuning import FourChainMeanBandAcceptancePolicy

PLAN = "docs/plans/surrogate-hmc-m13-48h-continuation-2026-09-29.md"


def multipliers_for_attempt(attempt):
    """Initial mixture, then a scoped energy-step repair with a fresh screen."""
    if isinstance(attempt, bool) or attempt not in (1, 2):
        raise ValueError("the additional48-hour allocation has exactly two planned hypotheses")
    return (.125, 1.)


def mixing_repair_config(config, attempt):
    """Tune the fixed mixture through the public authority; verify fresh chains."""
    multipliers = multipliers_for_attempt(attempt)
    target, overall = .70, (.50, .85)
    nomination = replace(config.nomination,
        target_accept_prob=target,
        acceptance_policy=FourChainMeanBandAcceptancePolicy(
            overall_band=overall, per_chain_band=(.05, 1.)),
        verification_repair_rounds=2,
        budget_provenance="additional48-hour allocation; master enforces32-hour aggregate tuning cap and reserves10 hours for posterior assessment",
        initial_step_size_provenance="normalized .2 base-step warm start; same-target dual averaging for the frozen mixture, followed by at most two factor2 repairs",
        seed=(20260929, 3001 + 100*attempt))
    if attempt == 2:
        from .surrogate_hmc_m13_energy_repair import STEP_CAP
        nomination = replace(nomination, max_step_size=STEP_CAP,
            energy_error_max=config.energy_error_max,
            initial_step_size_provenance="independent .2 warm start; frozen nomination step capped at half the failed attempt1 step after the dedicated energy-repair screen")
    repaired = replace(config, nomination=nomination, fresh_seed=(20260929, 3002 + 100*attempt))
    metadata = dict(plan=PLAN, attempt=attempt,
        step_multipliers=list(multipliers), step_probabilities=[.5, .5],
        adaptation_acceptance_target=target,
        nomination_overall_acceptance_band=list(overall), nomination_per_chain_band=[.05, 1.],
        verification_step_repair_rounds=2, step_repair_factor=2.,
        rationale="constant independent step mixture retains small escape moves alongside larger exploration moves; both laws passed two discarded M13 health probes",
        hypothesis_order="first uses the screened mixture; second keeps that mixture and caps the fixed base step after the archived energy veto",
        max_step_size=nomination.max_step_size, nomination_energy_error_max=nomination.energy_error_max,
        role="nomination hypothesis only; fresh readiness, verification, ESS and posterior criteria unchanged",
        second_attempt_trigger="reviewed energy-veto diagnosis and completed fresh-binding repair probe; one full attempt remains",
        force_changed=False, endpoint_changed=False, posterior_qualified=False)
    return repaired, metadata
