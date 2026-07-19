from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType

import pytest


os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT / "docs/benchmarks/run_ssl_lstm_a4_hmc_repair_2026_07_14.py"


@pytest.fixture(scope="module")
def repair() -> ModuleType:
    spec = importlib.util.spec_from_file_location("ssl_lstm_a4_hmc_repair", HARNESS)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_frozen_repair_kernel_namespace_and_seeds(repair: ModuleType) -> None:
    assert repair.STEP_SIZE == 0.19625
    assert repair.LEAPFROG_STEPS == 8
    assert repair.STEP_SIZE * repair.LEAPFROG_STEPS == pytest.approx(1.57)
    assert repair.TUNING_SEED == (20260714, 1521)
    assert repair._segment_label(0) == "repair_01_segment_0"
    assert repair._segment_output(0) != repair.base.A4_ROOT / "segment-0.json"
    assert "repair-01" in repair._segment_output(0).as_posix()


def test_prior_receipts_are_hash_bound_and_budget_complete(repair: ModuleType) -> None:
    seconds = repair.validate_prior_receipts()
    assert seconds == pytest.approx(1333.7487312000012)
    assert repair.base.GPU_BUDGET_SECONDS - seconds == pytest.approx(
        27466.2512688
    )
    assert [digest for _path, digest in repair.PRIOR_RECEIPTS] == [
        "d5aa099cc4835d427b570a7a22430a7b79498760dc99d8ed280c9bf39692c048",
        "b30098f573fb2a7a22f8a1a71b910d2b931fac7c169f049ac9e9efe6af87ab2d",
        "9e70e8dbd04de09c0bc3946d100d24d67ce520c18f63e58c8b5d3502762fa76f",
        "d12e7aeb1c9760b9d4bba9f9827c027e371d227a3cf5b84d7775f3a922021892",
    ]


def test_prior_receipt_drift_fails_closed(
    repair: ModuleType, monkeypatch: pytest.MonkeyPatch
) -> None:
    path, digest = repair.PRIOR_RECEIPTS[0]
    monkeypatch.setattr(repair, "PRIOR_RECEIPTS", ((path, "0" * 64),))
    with pytest.raises(repair.RepairError, match="SHA-256 drift"):
        repair.validate_prior_receipts()
    monkeypatch.setattr(repair, "PRIOR_RECEIPTS", ((path, digest),))


def test_source_bindings_include_repair_and_base_without_result(
    repair: ModuleType,
) -> None:
    rows = repair._source_bindings()
    paths = {row["path"] for row in rows}
    assert repair.REPAIR_PLAN_PATH.as_posix() in paths
    assert repair.REPAIR_HARNESS_PATH.as_posix() in paths
    assert repair.REPAIR_TEST_PATH.as_posix() in paths
    assert repair.BASE_HARNESS_PATH.as_posix() in paths
    assert repair.REPAIR_RESULT_PATH.as_posix() not in paths


def test_trusted_repair_manifest_fails_closed(repair: ModuleType) -> None:
    valid = {
        "run_manifest": {
            "trust_basis": "owner_designated_managed_session_visible_gpu_trusted",
            "cpu_gpu_status": "trusted_gpu_xla",
            "data_version": repair.base.TARGET_SEMANTIC_SHA256,
            "plan_path": repair.REPAIR_PLAN_PATH.as_posix(),
            "result_path": repair.REPAIR_RESULT_PATH.as_posix(),
            "wall_time_seconds": 1.0,
        }
    }
    repair._assert_trusted_repair_manifest(valid, Path("valid.json"))
    invalid = {"run_manifest": dict(valid["run_manifest"])}
    invalid["run_manifest"]["trust_basis"] = "untrusted"
    with pytest.raises(repair.RepairError, match="invalid trusted repair"):
        repair._assert_trusted_repair_manifest(invalid, Path("invalid.json"))


def test_no_overwrite_guard_checks_public_and_private_members(
    repair: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(repair, "ROOT", tmp_path)
    monkeypatch.setattr(repair, "ARCHIVE_DIR", Path("private"))
    output = Path("result.json")
    repair._require_fresh(output, "fresh")
    (tmp_path / output).write_text("occupied", encoding="ascii")
    with pytest.raises(repair.RepairError, match="refusing overwrite"):
        repair._require_fresh(output, "fresh")


def test_first_segment_projection_fits_authorized_remaining_budget(
    repair: ModuleType,
) -> None:
    consumed = repair.validate_prior_receipts()
    assert repair._projected_segment_seconds(0, ()) == 7200.0
    assert consumed + 900.0 + 7200.0 < repair.base.GPU_BUDGET_SECONDS
