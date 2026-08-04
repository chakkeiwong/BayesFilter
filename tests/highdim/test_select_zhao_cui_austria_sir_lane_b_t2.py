from __future__ import annotations

import json

from scripts.select_zhao_cui_austria_sir_lane_b_t2 import (
    EXPECTED_ARMS,
    build_selection,
)


def test_t2_selection_uses_minimum_viable_validation_rms(tmp_path) -> None:
    for index, arm_id in enumerate(EXPECTED_ARMS):
        directory = tmp_path / arm_id
        directory.mkdir()
        viable = index not in {1, 4}
        payload = {
            "schema_version": "bayesfilter.zhao_cui_austria_sir_lane_b_t2_pilot.v1",
            "status": "VIABLE_T2_PILOT_ARM" if viable else "REJECTED_T2_PILOT_ARM",
            "baseline_id": "zhao_cui_austria_sir_fixed_variant_training_base_t2_v1",
            "arm": {"arm_id": arm_id},
            "parent_t1_identity": "parent",
            "gates": {"viable": viable},
            "validation_metrics": {"normalized_log_density_rms": 10.0 - index},
            "artifact_manifest": f"artifact-{index}" if viable else None,
            "artifact_identity": f"identity-{index}" if viable else None,
            "run_manifest": {
                "source_sha256": {"source": "hash"},
                "prepared_result_sha256": {"training": "a", "validation": "b", "calibration": "c"},
            },
        }
        (directory / "result.json").write_text(json.dumps(payload) + "\n")
    selection = build_selection(tmp_path)
    assert selection["status"] == "SELECTED_VIABLE_T2_PILOT_ARM"
    assert selection["selected_arm_id"] == EXPECTED_ARMS[-1]
    assert selection["untouched_seed_used"] is False
