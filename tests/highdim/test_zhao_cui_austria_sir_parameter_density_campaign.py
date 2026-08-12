from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import tensorflow as tf

from bayesfilter.highdim.zhao_cui_austria_sir_parameter_density_campaign_config import (
    ARM_TABLE,
    CORE_AFFINE_CG_TABLE,
    INITIALIZER_AUDIT_TABLE,
    PREFIX_TANGENT_TABLE,
    PAIR_TANGENT_TABLE,
    DIRECT_TT_TANGENT_TABLE,
    FULL_TT_MINIMAX_TABLE,
    RANK12_MINIMAX_TABLE,
    CORE_AFFINE_LBFGS_TABLE,
    CORE_AFFINE_MINIMAX_TABLE,
    ROTATING_PREFIX_TANGENT_TABLE,
    axis_theta_rows,
    rotating_prefix_checkpoint_key,
    validation_theta_rows,
)


ROOT = Path(__file__).resolve().parents[2]


def _module(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_theta_rows_and_arm_ladder_are_frozen() -> None:
    assert axis_theta_rows(0.01)[0] == (0.0, 0.0, 0.0)
    assert len(axis_theta_rows(0.01)) == 7
    assert len(validation_theta_rows()) == 13
    assert len(ARM_TABLE) == 7
    assert any(float(row["l1_weight"]) == 0.0 for row in ARM_TABLE.values())
    assert any(float(row["derivative_weight"]) == 0.0 for row in ARM_TABLE.values())
    assert {int(row["rank"]) for row in ARM_TABLE.values()} == {2, 4}
    assert tuple(INITIALIZER_AUDIT_TABLE) == (
        "i01_r2_amp1_pert05_lr3e4",
        "i02_r4_amp1_pert05_lr3e4",
        "i03_add_ridge1e4_global1",
        "i04_add_ridge1e4_global10",
    )
    assert {int(row["rank"]) for row in INITIALIZER_AUDIT_TABLE.values()} == {2, 4}
    assert [int(row["steps"]) for row in INITIALIZER_AUDIT_TABLE.values()] == [
        32,
        32,
        0,
        0,
    ]
    assert tuple(PREFIX_TANGENT_TABLE) == (
        "p01_prefixw001",
        "p02_prefixw01",
        "p03_prefixw1",
        "p04_prefix16_g100_w001",
        "p05_prefix16_g100_w01",
        "p06_prefix64_g100_w0025",
    )
    assert [float(row["prefix_weight"]) for row in PREFIX_TANGENT_TABLE.values()] == [
        0.01,
        0.1,
        1.0,
        0.001,
        0.01,
        0.0025,
    ]
    assert tuple(PAIR_TANGENT_TABLE) == (
        "r01_pair_ridge1e4",
        "r02_pair_ridge1e3",
    )
    assert [float(row["ridge_fraction"]) for row in PAIR_TANGENT_TABLE.values()] == [
        1e-4,
        1e-3,
    ]
    assert tuple(DIRECT_TT_TANGENT_TABLE) == (
        "d01_r4_lr1e5",
        "d02_r4_lr3e5",
        "d03_r7_lr1e5",
        "d04_r7_lr3e5",
        "d05_r7_lr1e5_p100_g100",
    )
    assert [int(row["rank"]) for row in DIRECT_TT_TANGENT_TABLE.values()] == [
        4,
        4,
        7,
        7,
        7,
    ]
    assert tuple(ROTATING_PREFIX_TANGENT_TABLE) == (
        "q01_r7_pool512_batch64_steps256",
        "q02_r7_pool512_batch64_steps1024",
        "q03_r8_core_tangent_warm_start_steps256",
        "c01_core_affine_zero_lr1e3_steps256",
        "c02_core_affine_zero_lr3e4_steps256",
    )
    rotating = ROTATING_PREFIX_TANGENT_TABLE["q01_r7_pool512_batch64_steps256"]
    assert int(rotating["fit_pool_size"]) == 512
    assert int(rotating["prefix_batch_size"]) == 64
    assert int(rotating["calibration_size"]) == 64
    assert int(rotating["checkpoint_interval"]) == 8
    extended = ROTATING_PREFIX_TANGENT_TABLE["q02_r7_pool512_batch64_steps1024"]
    assert int(extended["steps"]) == 4 * int(rotating["steps"])
    assert {
        key: value for key, value in extended.items() if key != "steps"
    } == {
        key: value for key, value in rotating.items() if key != "steps"
    }
    structured = ROTATING_PREFIX_TANGENT_TABLE[
        "q03_r8_core_tangent_warm_start_steps256"
    ]
    assert structured["initializer_id"] == (
        "hash_verified_ungauged_core_tangent_s05_rank8_v1"
    )
    assert int(structured["rank"]) == 8
    for key in (
        "learning_rate",
        "steps",
        "point_weight",
        "global_weight",
        "prefix_weight",
        "fit_pool_size",
        "prefix_batch_size",
        "calibration_size",
        "checkpoint_interval",
        "pool_partition_seed",
        "minibatch_seed",
    ):
        assert structured[key] == rotating[key]
    core_affine = ROTATING_PREFIX_TANGENT_TABLE[
        "c01_core_affine_zero_lr1e3_steps256"
    ]
    assert core_affine["initializer_id"] == "current_frozen_basis_core_affine_zero_v1"
    assert int(core_affine["rank"]) == 8
    assert float(core_affine["learning_rate"]) == 1e-3
    for key in (
        "steps",
        "point_weight",
        "global_weight",
        "prefix_weight",
        "fit_pool_size",
        "prefix_batch_size",
        "calibration_size",
        "checkpoint_interval",
        "pool_partition_seed",
        "minibatch_seed",
    ):
        assert core_affine[key] == rotating[key]
    lower_lr = ROTATING_PREFIX_TANGENT_TABLE[
        "c02_core_affine_zero_lr3e4_steps256"
    ]
    assert float(lower_lr["learning_rate"]) == 3e-4
    assert {
        key: value for key, value in lower_lr.items() if key != "learning_rate"
    } == {
        key: value for key, value in core_affine.items() if key != "learning_rate"
    }
    assert tuple(CORE_AFFINE_LBFGS_TABLE) == ("l01_core_affine_fullpool_lbfgs",)
    lbfgs = CORE_AFFINE_LBFGS_TABLE["l01_core_affine_fullpool_lbfgs"]
    assert int(lbfgs["fit_pool_size"]) == 512
    assert int(lbfgs["calibration_size"]) == 64
    assert int(lbfgs["max_iterations"]) == 128
    assert int(lbfgs["num_correction_pairs"]) == 20
    assert float(lbfgs["gradient_tolerance"]) == 1e-8
    assert tuple(CORE_AFFINE_CG_TABLE) == (
        "n01_core_affine_fullpool_cg_from_l01",
    )
    cg = CORE_AFFINE_CG_TABLE["n01_core_affine_fullpool_cg_from_l01"]
    assert int(cg["fit_pool_size"]) == int(lbfgs["fit_pool_size"])
    assert int(cg["calibration_size"]) == int(lbfgs["calibration_size"])
    assert float(cg["point_weight"]) == float(lbfgs["point_weight"])
    assert float(cg["global_weight"]) == float(lbfgs["global_weight"])
    assert float(cg["prefix_weight"]) == float(lbfgs["prefix_weight"])
    assert int(cg["max_iterations"]) == 512
    assert float(cg["residual_tolerance"]) == 1e-10
    assert tuple(CORE_AFFINE_MINIMAX_TABLE) == (
        "m01_core_affine_gate_max_from_n01",
    )
    minimax = CORE_AFFINE_MINIMAX_TABLE["m01_core_affine_gate_max_from_n01"]
    assert int(minimax["fit_pool_size"]) == int(lbfgs["fit_pool_size"])
    assert int(minimax["calibration_size"]) == int(lbfgs["calibration_size"])
    assert float(minimax["temperature"]) == 64.0
    assert int(minimax["max_iterations"]) == 256
    assert tuple(FULL_TT_MINIMAX_TABLE) == (
        "f01_full_r8_gate_max_from_n01",
    )
    full_tt = FULL_TT_MINIMAX_TABLE["f01_full_r8_gate_max_from_n01"]
    assert int(full_tt["rank"]) == 8
    assert int(full_tt["position_size"]) == 32880
    assert int(full_tt["fit_pool_size"]) == int(lbfgs["fit_pool_size"])
    assert int(full_tt["calibration_size"]) == int(lbfgs["calibration_size"])
    assert float(full_tt["temperature"]) == float(minimax["temperature"])
    assert tuple(RANK12_MINIMAX_TABLE) == (
        "r12_rank12_gate_max_from_n01",
    )
    rank12 = RANK12_MINIMAX_TABLE["r12_rank12_gate_max_from_n01"]
    assert int(rank12["rank"]) == 12
    assert int(rank12["position_size"]) == 73800
    assert float(rank12["temperature"]) == 64.0


def test_rotating_prefix_checkpoint_key_uses_maximum_then_mean_then_update() -> None:
    assert rotating_prefix_checkpoint_key(2.0, 4.0, 16) < (
        rotating_prefix_checkpoint_key(2.1, 1.0, 8)
    )
    assert rotating_prefix_checkpoint_key(2.0, 3.0, 16) < (
        rotating_prefix_checkpoint_key(2.0, 4.0, 8)
    )
    assert rotating_prefix_checkpoint_key(2.0, 3.0, 8) < (
        rotating_prefix_checkpoint_key(2.0, 3.0, 16)
    )


def test_selector_uses_only_viable_validation_rows(tmp_path: Path) -> None:
    selector = _module(
        "scripts/select_zhao_cui_austria_sir_parameter_density_t1.py",
        "parameter_density_selector",
    )
    root = tmp_path / "pilots"
    rows = [
        ("a", True, 0.8, 0.9, 0.7, 4),
        ("b", True, 0.8, 0.8, 0.9, 4),
        ("c", False, 0.1, 0.1, 0.1, 2),
    ]
    for arm, passed, score, shape, mass, rank in rows:
        directory = root / arm
        directory.mkdir(parents=True)
        payload = {
            "schema_version": selector.PILOT_SCHEMA,
            "status": "VIABLE_T1_PARAMETER_DENSITY_ARM" if passed else "REJECTED_T1_PARAMETER_DENSITY_ARM",
            "arm_id": arm,
            "arm": {"rank": rank},
            "child_identity": f"child-{arm}",
            "artifact_directory": f"artifact-{arm}",
            "gates": {"passed": passed},
            "validation": {
                "selector_metrics": {
                    "maximum_standardized_score_residual": score,
                    "mean_paired_shape_ratio": shape,
                    "maximum_mass_standardized_residual": mass,
                }
            },
        }
        (directory / "result.json").write_text(json.dumps(payload))
    selected = selector.select(root)
    assert selected["selected"]["arm_id"] == "b"
    assert len(selected["rows"]) == 3


def test_selector_preserves_training_failure_as_nonviable(tmp_path: Path) -> None:
    selector = _module(
        "scripts/select_zhao_cui_austria_sir_parameter_density_t1.py",
        "parameter_density_selector_training_failure",
    )
    directory = tmp_path / "failed"
    directory.mkdir()
    (directory / "result.json").write_text(
        json.dumps(
            {
                "schema_version": selector.PILOT_SCHEMA,
                "status": "REJECTED_T1_PARAMETER_DENSITY_ARM",
                "arm_id": "failed",
                "arm": {"rank": 2},
                "child_identity": None,
                "artifact_directory": None,
                "gates": {"passed": False},
                "validation": None,
            }
        )
    )
    result = selector.select(tmp_path)
    assert result["selected"] is None
    assert result["rows"][0]["viable"] is False


def test_runner_claim_mode_requires_selector_artifact() -> None:
    source = (
        ROOT / "scripts/run_zhao_cui_austria_sir_parameter_density_t1.py"
    ).read_text()
    assert "--claim-selection" in source
    assert "--claim-artifact" not in source
    assert "selector child identity does not match" in source
