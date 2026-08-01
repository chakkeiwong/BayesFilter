from __future__ import annotations

import os
import math

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

from docs.benchmarks import run_structural_ukf_neutra_confirmation as campaign


def test_repair_profile_binds_failed_attempt_and_preserves_promotion_gates() -> None:
    profile = campaign.KERNEL_PROFILES[campaign.REPAIR_PROFILE_ID]

    campaign._verify_kernel_profile(profile)

    assert profile.step_sizes == (0.05,)
    assert profile.num_leapfrog_steps == 8
    assert profile.tuning_rhat_max == 1.05
    assert profile.parent_result_sha256 == (
        "622ab339ceffdeb4850b53d9930a372dd9ea9e4ab13a7c59b4401bde10c8ffbe"
    )
    assert len(
        {
            profile.probe_seed,
            profile.tuning_verification_seed,
            profile.warmup_seed,
            profile.retained_seed,
        }
    ) == 4


def test_final_repair_increases_trajectory_without_relaxing_gates() -> None:
    profile = campaign.KERNEL_PROFILES[campaign.FINAL_REPAIR_PROFILE_ID]

    campaign._verify_kernel_profile(profile)

    assert profile.step_sizes == (0.05,)
    assert profile.num_leapfrog_steps == 12
    assert math.isclose(
        profile.step_sizes[0] * profile.num_leapfrog_steps,
        0.6,
        rel_tol=0.0,
        abs_tol=1.0e-15,
    )
    assert profile.tuning_rhat_max == 1.05
    assert profile.parent_result_sha256 == (
        "b02d22b9cfa51995c02c0df3f686770660d99a1f461600d224134b32ee200eed"
    )


def test_terminal_classification_requires_retained_sampler_validity() -> None:
    assert campaign._classify_terminal(None, None) == (
        "SAMPLER_BLOCKED_NO_TUNING_ADMISSION"
    )
    assert campaign._classify_terminal(
        {
            "hard_vetoes": (),
            "warmup_passed": True,
            "retained_passed": False,
        },
        None,
    ) == "SAMPLER_BLOCKED_RETAINED_CONVERGENCE"


def test_truth_tail_classification_does_not_upgrade_marginal_result() -> None:
    sequential = {
        "hard_vetoes": (),
        "warmup_passed": True,
        "retained_passed": True,
    }

    assert campaign._classify_terminal(
        sequential,
        {
            "severe_parameters": (),
            "marginal_parameters": ("rho",),
            "passed": False,
        },
    ) == "MARGINAL_TRUTH_TAIL_REQUIRES_SECOND_SEED"
