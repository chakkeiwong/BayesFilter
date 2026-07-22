from __future__ import annotations

import base64
import copy
import importlib.util
import json
from pathlib import Path
import struct

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_PATH = ROOT / (
    "docs/plans/artifacts/complete-highdim-leaderboard/"
    "phase1-canonical-targets-2026-07-11.json"
)
CHECKER_PATH = ROOT / "docs/benchmarks/check_complete_highdim_phase1_canonical_targets.py"


def _load_checker():
    spec = importlib.util.spec_from_file_location("phase1_canonical_checker", CHECKER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def checker():
    return _load_checker()


@pytest.fixture()
def payload() -> dict:
    return json.loads(ARTIFACT_PATH.read_text(encoding="utf-8"))


def test_canonical_target_artifact_passes_independent_check(checker) -> None:
    checked = checker.check_artifact(ARTIFACT_PATH)
    assert checked["summary"]["row_count"] == 6
    assert checked["summary"]["filter_executed"] is False
    assert checked["summary"]["leaderboard_cell_admitted"] is False


def test_checker_rejects_row_reordering(checker, payload, tmp_path: Path) -> None:
    payload["rows"][0], payload["rows"][1] = payload["rows"][1], payload["rows"][0]
    path = tmp_path / "reordered.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="row order or identity mismatch"):
        checker.check_artifact(path)


def test_checker_rejects_field_omission(checker, payload, tmp_path: Path) -> None:
    payload["rows"][0]["authoritative_ordered_field_ledger"].pop()
    path = tmp_path / "omitted.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="field ledger mismatch"):
        checker.check_artifact(path)


def test_checker_rejects_implicit_multibyte_endian(checker, payload, tmp_path: Path) -> None:
    payload["rows"][0]["fields"][0]["header"]["dtype_descriptor"] = "=f8"
    path = tmp_path / "implicit-endian.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="not explicit little endian"):
        checker.check_artifact(path)


def test_checker_rejects_nonfinite_payload(checker, payload, tmp_path: Path) -> None:
    record = payload["rows"][0]["fields"][0]
    raw = bytearray(base64.b64decode(record["payload_base64"]))
    raw[:8] = struct.pack("<d", np.nan)
    record["payload_base64"] = base64.b64encode(raw).decode("ascii")
    path = tmp_path / "nonfinite.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="nonfinite payload"):
        checker.check_artifact(path)


def test_checker_rejects_source_hash_tamper(checker, payload, tmp_path: Path) -> None:
    sources = payload["rows"][0]["semantics"]["generator_data_config_paths_sha256"]
    first_path = next(iter(sources))
    sources[first_path] = "0" * 64
    path = tmp_path / "source-hash.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="stale source hash"):
        checker.check_artifact(path)


def test_checker_rejects_algorithm_specific_field_contamination(
    checker, payload, tmp_path: Path
) -> None:
    row = payload["rows"][0]
    row["authoritative_ordered_field_ledger"][0] = "initial_particles"
    path = tmp_path / "algorithm-field.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="field ledger mismatch"):
        checker.check_artifact(path)


def test_sir_target_binds_seed_81103_and_denies_matlab_reproduction(
    checker, payload
) -> None:
    row = next(
        item
        for item in payload["rows"]
        if item["row_id"] == "zhao_cui_spatial_sir_austria_j9_T20"
    )
    semantics = row["semantics"]

    assert semantics["dataset_seed"] == 81103
    assert semantics["target_generation_identity"] == checker.SIR_TARGET_GENERATION_IDENTITY
    assert semantics["author_matlab_rng1_reproduction_claimed"] is False


def test_checker_rejects_sir_matlab_reproduction_claim(
    checker, payload, tmp_path: Path
) -> None:
    row = next(
        item
        for item in payload["rows"]
        if item["row_id"] == "zhao_cui_spatial_sir_austria_j9_T20"
    )
    row["semantics"]["author_matlab_rng1_reproduction_claimed"] = True
    path = tmp_path / "sir-matlab-claim.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        ValueError,
        match="required semantics author_matlab_rng1_reproduction_claimed mismatch",
    ):
        checker.check_artifact(path)
