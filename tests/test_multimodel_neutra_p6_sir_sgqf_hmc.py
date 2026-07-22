from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

from docs.benchmarks.run_multimodel_neutra_p6_sir_sgqf_hmc import (
    _ordered_probe_candidates,
    _tuning_verification_admitted,
)


def test_short_probe_modern_rhat_orders_candidates_but_does_not_admit() -> None:
    rows = (
        {
            "grid_index": 0,
            "eligible": True,
            "minimum_bulk_ess": 50.0,
            "maximum_modern_rhat": 1.02,
        },
        {
            "grid_index": 1,
            "eligible": True,
            "minimum_bulk_ess": 500.0,
            "maximum_modern_rhat": 2.73,
        },
        {
            "grid_index": 2,
            "eligible": False,
            "minimum_bulk_ess": 5000.0,
            "maximum_modern_rhat": 1.001,
        },
    )

    ordered = _ordered_probe_candidates(rows)

    assert tuple(row["grid_index"] for row in ordered) == (0, 1)


def test_high_acceptance_or_bulk_ess_cannot_bypass_modern_rhat() -> None:
    health = {"health_passed": True, "acceptance_rate": 1.0}
    failed_rhat = {
        "input_all_finite": True,
        "diagnostics_all_finite": True,
        "passed": False,
        "max_finite_rhat": 2.73,
    }
    passed_rhat = {
        **failed_rhat,
        "passed": True,
        "max_finite_rhat": 1.008,
    }

    assert _tuning_verification_admitted(
        health=health, modern_rhat=failed_rhat
    ) is False
    assert _tuning_verification_admitted(
        health=health, modern_rhat=passed_rhat
    ) is True
