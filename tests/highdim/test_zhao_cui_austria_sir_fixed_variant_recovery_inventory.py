from __future__ import annotations

import json
from pathlib import Path
import subprocess

from scripts.run_zhao_cui_austria_sir_fixed_variant_recovery_inventory import (
    P88_BRANCH_HASH,
    _candidate_could_bind_complete_identity,
    _git_blob_inventory,
    _match_payload,
)


def test_recovery_inventory_rejects_prose_listing_missing_fields() -> None:
    prose = (
        f"P88 branch {P88_BRANCH_HASH}; missing coordinate_frame_mu, "
        "coordinate_frame_matrix, cdf_config, frozen_reference_samples, "
        "retained_branch_identity, source_dependency_closure, observation_hash"
    ).encode()
    candidate = _match_payload(prose, "blocker-note.md")
    assert candidate is not None
    assert candidate["all_identity_group_terms_present"] is True
    assert candidate["structured_json_object"] is False
    assert _candidate_could_bind_complete_identity(candidate) is False


def test_recovery_inventory_nominates_only_structured_complete_lead() -> None:
    payload = {
        "branch_hash": P88_BRANCH_HASH,
        "coordinate_frame_mu": [0.0],
        "coordinate_frame_matrix": [[1.0]],
        "transport_cdf_config": {"grid_size": 8},
        "frozen_reference_samples": [[0.0]],
        "retained_branch_identity": "hash",
        "source_dependency_closure": {"file": "hash"},
        "observation_hash": "hash",
    }
    candidate = _match_payload(json.dumps(payload).encode(), "candidate.json")
    assert candidate is not None
    candidate["sha256"] = "not-the-artifact-hash"
    assert candidate["structured_json_object"] is True
    assert candidate["all_identity_group_terms_present"] is True
    assert _candidate_could_bind_complete_identity(candidate) is True


def test_recovery_inventory_detects_p88_filename_without_promoting_it(tmp_path: Path) -> None:
    path = tmp_path / "p88-unrelated.bin"
    path.write_bytes(b"unrelated")
    candidate = _match_payload(path.read_bytes(), path.name)
    assert candidate is not None
    candidate["sha256"] = "not-the-artifact-hash"
    assert candidate["anchors"] == ["p88_filename"]
    assert _candidate_could_bind_complete_identity(candidate) is False


def test_recovery_inventory_reads_git_batch_output_pipe(tmp_path: Path) -> None:
    subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True)
    candidate = tmp_path / "candidate.json"
    candidate.write_text(
        json.dumps(
            {
                "branch_hash": P88_BRANCH_HASH,
                "coordinate_frame_mu": [0.0],
                "coordinate_frame_matrix": [[1.0]],
                "transport_cdf_config": {"grid_size": 8},
                "frozen_reference_samples": [[0.0]],
                "retained_branch_identity": "hash",
                "source_dependency_closure": {"file": "hash"},
                "observation_hash": "hash",
            }
        ),
        encoding="utf-8",
    )
    subprocess.run(["git", "add", "candidate.json"], cwd=tmp_path, check=True)
    git_candidates, counts = _git_blob_inventory(tmp_path)
    assert counts["blobs_scanned"] >= 1
    assert any(_candidate_could_bind_complete_identity(item) for item in git_candidates)
