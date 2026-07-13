from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import pytest

from bayesfilter.inference.hmc_phase30_diagnostics import (
    Phase30PosteriorThresholds,
    classify_phase30_posterior_checkpoint,
    evaluate_phase30_posterior_checkpoint,
    evaluate_phase30_warmup_epoch,
)


def _inputs(draws: int = 512, seed: int = 20260713) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    samples = rng.normal(size=(5, draws, 4))
    pre_state = samples - 0.01
    energy = rng.normal(size=(5, draws))
    accepted = np.broadcast_to(
        (np.arange(draws) % 10 < 7)[None, :], (5, draws)
    ).copy()
    delta_h = rng.normal(scale=0.1, size=(5, draws))
    return {
        "coordinate_samples": {
            "final_latent": samples,
            "named_model": 2.0 * samples + 1.0,
        },
        "pre_state": pre_state,
        "accepted_target_log_prob": rng.normal(size=(5, draws)),
        "is_accepted": accepted,
        "log_accept_ratio": -delta_h,
        "initial_energy": energy,
        "delta_h": delta_h,
    }


def test_iid_warmup_epoch_passes_role_separated_screen() -> None:
    result = evaluate_phase30_warmup_epoch(**_inputs())

    assert result["passed"] is True
    assert result["engineering_continuation_vetoes"] == []
    assert result["candidate_promotion_vetoes"] == []
    assert result["energy_diagnostics"]["evidence_role"] == "explanatory_only"


def test_finite_extreme_energy_has_no_control_or_promotion_effect() -> None:
    baseline_inputs = _inputs(seed=30)
    baseline = evaluate_phase30_warmup_epoch(**baseline_inputs)
    extreme_inputs = _inputs(seed=30)
    extreme_inputs["delta_h"][0, 0] = 1.0e100
    extreme_inputs["log_accept_ratio"][0, 0] = -1.0e100
    extreme = evaluate_phase30_warmup_epoch(**extreme_inputs)

    assert extreme["engineering_continuation_vetoes"] == baseline[
        "engineering_continuation_vetoes"
    ]
    assert extreme["candidate_promotion_vetoes"] == baseline[
        "candidate_promotion_vetoes"
    ]
    assert extreme["passed"] == baseline["passed"]
    assert extreme["energy_diagnostics"]["maximum_abs_delta_h"] == 1.0e100
    assert extreme["energy_diagnostics"]["finite_energy_magnitude_can_veto"] is False


def test_wrong_identity_and_nonfinite_accepted_target_are_engineering_vetoes() -> None:
    wrong = _inputs(seed=31)
    wrong["delta_h"][0, 0] = 1.0
    wrong["log_accept_ratio"][0, 0] = 1.0
    wrong_result = evaluate_phase30_warmup_epoch(**wrong)
    assert "hamiltonian_log_accept_identity_failure" in wrong_result[
        "engineering_continuation_vetoes"
    ]

    nonfinite = _inputs(seed=32)
    nonfinite["accepted_target_log_prob"][1, 2] = np.nan
    nonfinite_result = evaluate_phase30_warmup_epoch(**nonfinite)
    assert "accepted_target_log_prob_nonfinite" in nonfinite_result[
        "engineering_continuation_vetoes"
    ]
    assert nonfinite_result["energy_diagnostics"]["evidence_role"] == (
        "explanatory_only"
    )


def test_posterior_precision_only_failure_extends_below_cap_and_stops_at_cap() -> None:
    inputs = _inputs(seed=33)
    thresholds = Phase30PosteriorThresholds(
        rhat_max_exclusive=1.10,
        bulk_ess_min_exclusive=1.0e9,
        tail_ess_min_exclusive=1.0e9,
        mcse_sd_ratio_max=1.0,
    )
    below_cap = evaluate_phase30_posterior_checkpoint(
        **inputs,
        maximum_draws=1024,
        thresholds=thresholds,
    )
    at_cap = evaluate_phase30_posterior_checkpoint(
        **inputs,
        maximum_draws=512,
        thresholds=thresholds,
    )

    assert below_cap["posterior_health_passed"] is True
    assert below_cap["posterior_precision_passed"] is False
    assert below_cap["decision"] == "extend_precision_only"
    assert below_cap["precision_extension_eligible"] is True
    assert at_cap["decision"] == "posterior_precision_nonpromotion_at_cap"
    assert at_cap["precision_extension_eligible"] is False


def test_health_failure_blocks_precision_extension() -> None:
    inputs = _inputs(seed=34)
    inputs["coordinate_samples"]["final_latent"][0] += 2.0
    inputs["coordinate_samples"]["named_model"][0] += 4.0
    inputs["pre_state"] = inputs["coordinate_samples"]["final_latent"] - 0.01
    result = evaluate_phase30_posterior_checkpoint(
        **inputs,
        maximum_draws=1024,
        thresholds=Phase30PosteriorThresholds(
            bulk_ess_min_exclusive=1.0e9,
            tail_ess_min_exclusive=1.0e9,
            mcse_sd_ratio_max=1.0,
        ),
    )

    assert result["posterior_health_passed"] is False
    assert result["decision"] == "posterior_health_nonpromotion"
    assert result["precision_extension_eligible"] is False


def test_checkpoint_state_machine_is_exact_and_fail_closed() -> None:
    extension = classify_phase30_posterior_checkpoint(
        engineering_continuation_vetoes=(),
        posterior_health_vetoes=(),
        posterior_precision_vetoes=("bulk_ess",),
        draws=2048,
        maximum_draws=4096,
    )
    success = classify_phase30_posterior_checkpoint(
        engineering_continuation_vetoes=(),
        posterior_health_vetoes=(),
        posterior_precision_vetoes=(),
        draws=3072,
        maximum_draws=4096,
    )
    health = classify_phase30_posterior_checkpoint(
        engineering_continuation_vetoes=(),
        posterior_health_vetoes=("rhat",),
        posterior_precision_vetoes=("ess",),
        draws=2048,
        maximum_draws=4096,
    )
    engineering = classify_phase30_posterior_checkpoint(
        engineering_continuation_vetoes=("artifact",),
        posterior_health_vetoes=(),
        posterior_precision_vetoes=(),
        draws=2048,
        maximum_draws=4096,
    )

    assert extension["decision"] == "extend_precision_only"
    assert success["decision"] == "posterior_checkpoint_pass"
    assert health["decision"] == "posterior_health_nonpromotion"
    assert engineering["decision"] == "engineering_continuation_veto"
    with pytest.raises(ValueError, match="draw count"):
        classify_phase30_posterior_checkpoint(
            engineering_continuation_vetoes=(),
            posterior_health_vetoes=(),
            posterior_precision_vetoes=(),
            draws=4097,
            maximum_draws=4096,
        )
