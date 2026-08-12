from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "docs/benchmarks/run_ssl_lstm_q20_seed_b_predictive_equivalence_repair_2026_08_08.py"


def _module():
    spec = importlib.util.spec_from_file_location("q20_predictive_equivalence_repair", RUNNER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _counts(module, *, equivalence_pass: int, material_difference: int):
    rows = {}
    for family, (role, _expected) in module.FAMILY_ROLES.items():
        rows[family] = {
            "PASS": equivalence_pass if role == "equivalence" else 0,
            "MATERIAL_DIFFERENCE": material_difference if role == "material" else 0,
            "INCONCLUSIVE_UNDERPOWERED": 0,
            "INVALID_HARD_VETO": 0,
        }
    return rows


def test_frozen_repair_invariants_are_enforced_in_source() -> None:
    module = _module()
    source = RUNNER.read_text(encoding="utf-8")
    assert module.BLOCK_LENGTH == 1
    assert module.BANDWIDTH_SUBSET_DRAWS == 128
    assert module.CALIBRATION_DRAWS == 16384
    assert module.MMD_TOLERANCES[0] == 0.0005
    assert module.STAGE_CAPS["nominate"] == 3600.0
    assert module.STAGE_CAPS["validate"] == 10800.0
    assert module.DEFAULT_ROOT.name == "r3"
    assert "standardize_forecast_paths(" in source
    assert "allow_floor_use=False" in source
    assert "2.0 * left.influence_values" in source
    assert "-2.0 * right.influence_values" in source
    assert '"bandwidth_source": "standardized_null_true_control_scale_bank_only"' in source
    assert '"archive_or_transport_loaded": False' in source
    assert "build_seed_b_terminal" not in source.split("def _run_material_like", 1)[0]
    assert "tf.shape(flat)[0], tf.float64" in source
    assert 'message="scale-bank correlation must have unit diagonal"' in source
    assert "stage_root=900000" in source
    assert "shape_stage_root=1100000" in source
    assert "stage_root=1300000" in source
    assert "shape_stage_root=1500000" in source


def test_tensorflow_shapes_serialize_without_numpy_computation() -> None:
    module = _module()

    class Shape:
        def as_list(self):
            return [4, 32, 2, 10]

    assert module._safe(Shape()) == [4, 32, 2, 10]


def test_candidate_selection_uses_smallest_complete_decision_pass() -> None:
    module = _module()
    failed = _counts(module, equivalence_pass=15, material_difference=16)
    passed = _counts(module, equivalence_pass=16, material_difference=16)
    selected = module.select_smallest_tolerance(
        [
            {"tolerance": 0.006, "counts": passed},
            {"tolerance": 0.004, "counts": failed},
            {"tolerance": 0.005, "counts": passed},
        ]
    )
    assert selected == 0.005


def test_invalid_or_wrong_direction_counts_fail_closed() -> None:
    module = _module()
    counts = _counts(module, equivalence_pass=16, material_difference=16)
    counts["identical"]["INVALID_HARD_VETO"] = 1
    assert not module._candidate_passes(counts, validation=False)
    counts = _counts(module, equivalence_pass=16, material_difference=16)
    counts["material_mean"]["PASS"] = 2
    assert not module._candidate_passes(counts, validation=False)


def test_material_stage_requires_passed_validation_receipt() -> None:
    source = RUNNER.read_text(encoding="utf-8")
    material = source.split("def _run_material_like", 1)[1].split("def run(", 1)[0]
    assert 'required_status="VALIDATION_PASSED"' in material
    assert 'required_status="PASS"' in material
    assert "if mode == \"audit\"" in material
