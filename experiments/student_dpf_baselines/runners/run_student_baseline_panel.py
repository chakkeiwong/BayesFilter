"""Run the first student-baseline comparison panel."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from experiments.student_dpf_baselines.adapters.advanced_particle_filter_adapter import (
    run_smoke_fixture as run_advanced_particle_filter,
)
from experiments.student_dpf_baselines.adapters.common import (
    BaselineStatus,
    write_json,
)
from experiments.student_dpf_baselines.adapters.mlcoe_adapter import (
    run_smoke_fixture as run_mlcoe,
)
from experiments.student_dpf_baselines.fixtures.common_fixtures import make_fixture


OUTPUT_PATH = Path(
    "experiments/student_dpf_baselines/reports/outputs/student_baseline_panel_2026-05-10.json"
)
REFERENCE_DIR = Path(
    "experiments/student_dpf_baselines/reports/outputs/references"
)

FIXTURES = ["lgssm_1d_short", "lgssm_cv_2d_short"]
SEEDS = [0, 1, 2]
PARTICLE_COUNTS = [128, 512]


def main() -> None:
    records = []
    for fixture_name in FIXTURES:
        fixture = make_fixture(fixture_name)
        reference = _read_reference(fixture_name)
        for seed in SEEDS:
            for num_particles in PARTICLE_COUNTS:
                records.append(
                    _augment_with_reference_metrics(
                        run_advanced_particle_filter(
                            fixture,
                            seed=seed,
                            num_particles=num_particles,
                        ),
                        reference,
                    )
                )
                records.append(
                    _augment_with_reference_metrics(
                        run_mlcoe(
                            fixture,
                            seed=seed,
                            num_particles=num_particles,
                        ),
                        reference,
                    )
                )
    write_json(
        OUTPUT_PATH,
        {
            "date": "2026-05-10",
            "panel": {
                "fixtures": FIXTURES,
                "seeds": SEEDS,
                "particle_counts": PARTICLE_COUNTS,
            },
            "records": records,
        },
    )


def _read_reference(fixture_name: str) -> dict:
    return json.loads((REFERENCE_DIR / f"{fixture_name}.json").read_text())


def _augment_with_reference_metrics(result, reference: dict) -> dict:
    record = result.to_dict()
    diagnostics = record.setdefault("diagnostics", {})
    if result.status != BaselineStatus.OK:
        diagnostics["reference_status"] = "not_compared"
        return record

    reference_ll = reference["log_likelihood"]
    diagnostics["reference_log_likelihood"] = reference_ll
    if result.log_likelihood is not None:
        diagnostics["log_likelihood_abs_error"] = abs(
            float(result.log_likelihood) - float(reference_ll)
        )

    if result.filtered_means is not None:
        diagnostics["filtered_mean_rmse_vs_reference"] = float(
            np.sqrt(
                np.mean(
                    (
                        np.asarray(result.filtered_means, dtype=float)
                        - np.asarray(reference["filtered_means"], dtype=float)
                    )
                    ** 2
                )
            )
        )
    if result.filtered_covariances is not None:
        diagnostics["filtered_covariance_rmse_vs_reference"] = float(
            np.sqrt(
                np.mean(
                    (
                        np.asarray(result.filtered_covariances, dtype=float)
                        - np.asarray(reference["filtered_covariances"], dtype=float)
                    )
                    ** 2
                )
            )
        )
    return record


if __name__ == "__main__":
    main()
