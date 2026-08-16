from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

ROOT = Path(__file__).resolve().parents[1]
PYTHON = "/home/chakwong/anaconda3/envs/tf-gpu/bin/python"
SCRIPT = ROOT / "docs/benchmarks/benchmark_actual_sv_three_route_simulation.py"


def test_actual_sv_three_route_benchmark_smoke_emits_expected_schema(tmp_path: Path) -> None:
    output = tmp_path / "three-route.json"
    markdown = tmp_path / "three-route.md"
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "-1"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    command = [
        PYTHON,
        str(SCRIPT),
        "--dims",
        "1",
        "--horizon",
        "4",
        "--dense-order",
        "201",
        "--mixture-components",
        "7,14",
        "--skip-scores",
        "--output",
        str(output),
        "--markdown-output",
        str(markdown),
    ]
    subprocess.run(command, check=True, cwd=ROOT, env=env, capture_output=True, text=True)

    payload = json.loads(output.read_text(encoding="utf-8"))
    markdown_text = markdown.read_text(encoding="utf-8")

    assert payload["schema_version"] == "actual_sv_three_route_simulation.v1"
    assert payload["dims"] == [1]
    assert payload["horizon"] == 4
    assert payload["fixture_kind"] == "simulated_exact_actual_sv_paths_iid_coordinates"

    assert payload["factorization_check"]["status"] == "passed"
    assert payload["factorization_check"]["abs_gap"] <= 1e-8

    batch_tt = payload["batch_tt_row"]
    assert batch_tt["dim"] == 1
    assert batch_tt["fixed_sigma"] == 1.0
    assert batch_tt["same_target_value_gap"] >= 0.0
    assert batch_tt["core_provenance"] == "ukf_center_frozen_truth_theta_simulated_dataset"

    exact = payload["exact_transformed_rows"][0]
    assert exact["transform_offset"] == 0.0
    assert exact["same_target_value_gap"] >= 0.0
    # Exact-transformed TT must match its own dense same-target reference tightly.
    assert exact["same_target_value_gap"] < 1e-6

    ksc = payload["ksc_surrogate_rows"][0]
    assert ksc["transform_offset"] > 0.0
    assert "surrogate" in ksc["nonclaim"]
    assert ksc["same_target_value_gap_tt_vs_dense"] >= 0.0
    assert ksc["same_target_value_gap_kalman_vs_dense"] >= 0.0

    refinement = payload["refinement_rows"][0]
    counts = [step["component_count"] for step in refinement["ladder"]]
    assert counts == [7, 14]
    assert len(refinement["refinement_changes"]) == 1
    assert isinstance(refinement["stabilized_under_1pct_rule"], bool)

    fits = payload["mixture_fit_rows"]
    assert [row["component_count"] for row in fits] == [7, 14]
    # Refined fit must not be worse than the coarser fit on the shared grid.
    assert fits[1]["weighted_l1_density_error"] <= fits[0]["weighted_l1_density_error"]

    cross = payload["cross_family_rows"][0]
    assert "descriptive" in cross["nonclaim"]

    assert "Actual-SV Three-Route Simulation Benchmark" in markdown_text
    assert "Route 2: exact-transformed Zhao-Cui" in markdown_text
    assert "Route 3: KSC-surrogate Zhao-Cui" in markdown_text
    assert "Route 4: dense Kalman refinement ladder" in markdown_text
    assert "descriptive only" in markdown_text
