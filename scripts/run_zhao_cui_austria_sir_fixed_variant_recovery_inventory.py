#!/usr/bin/env python
"""Inventory exact P88 recovery candidates without numerical reconstruction."""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
from typing import BinaryIO, Iterable, Mapping
import zipfile


P88_ARTIFACT_SHA256 = "ea5fc7434f328b95e3c2c53bca3e1a7bee6b35a452a81acce8230407ea11ef8e"
P88_BRANCH_HASH = "265f9a06877e9babbba22dde187487fde4b50d08d8ecb98cd26b16467b6c1f10"
P88_TARGET_ID = "zhao_cui_sir_austria_d18"
P88_SQUARE_NORMALIZER = "4.544027196172014e-06"
P88_FULL_NORMALIZER = "4.554027196172014e-06"
P88_FRAME_LOG_DET = "49.70422923235673"
SCHEMA = "bayesfilter.zhao_cui.austria_sir.p88_recovery_inventory.v1"
STATUS_PASS = "CANDIDATE_EXACT_P88_RECOVERY_PAYLOAD_FOUND_REQUIRES_ADMISSION_REVIEW"
STATUS_BLOCK = "BLOCK_EXACT_P88_RECOVERY_EXHAUSTED"
MAX_SCAN_BYTES = 64 * 1024 * 1024

ANCHORS = {
    "artifact_sha256": P88_ARTIFACT_SHA256,
    "branch_hash": P88_BRANCH_HASH,
    "target_id": P88_TARGET_ID,
    "square_normalizer": P88_SQUARE_NORMALIZER,
    "full_normalizer": P88_FULL_NORMALIZER,
    "frame_log_abs_det": P88_FRAME_LOG_DET,
}

IDENTITY_GROUPS = {
    "coordinate_frame_mu": ("coordinate_frame_mu",),
    "coordinate_frame_matrix": ("coordinate_frame_matrix",),
    "transport_cdf_config": (
        "transport_cdf_config",
        "cdf_config",
        "krcdfconfig",
    ),
    "frozen_reference_samples": (
        "frozen_reference_samples",
        "frozen_reference_points",
    ),
    "retained_branch_identity": (
        "retained_branch_identity",
        "retained_object_branch_hash",
        "retained_object_identity",
    ),
    "source_dependency_closure": (
        "source_dependency_closure",
        "callable_dependency_closure",
        "source_file_hashes",
    ),
    "observation_or_target_input_identity": (
        "source_observation_sha256",
        "observation_hash",
        "target_input_hash",
    ),
}

EXCLUDED_DIRECTORY_NAMES = {
    ".git",
    "__pycache__",
    ".pytest_cache",
}


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _match_payload(data: bytes, name: str) -> dict[str, object] | None:
    lowered = data.lower()
    lowered_name = name.lower().encode("utf-8", errors="ignore")
    anchors = sorted(
        key
        for key, value in ANCHORS.items()
        if value.encode("ascii") in data
    )
    if "p88" in name.lower() and "p88_filename" not in anchors:
        anchors.append("p88_filename")
    if not anchors:
        return None
    identity_groups = {
        group: any(term.encode("ascii") in lowered for term in terms)
        for group, terms in IDENTITY_GROUPS.items()
    }
    zip_members: list[str] = []
    if data.startswith(b"PK\x03\x04"):
        try:
            from io import BytesIO

            with zipfile.ZipFile(BytesIO(data)) as archive:
                zip_members = sorted(archive.namelist())
            member_bytes = "\n".join(zip_members).lower().encode("utf-8")
            identity_groups = {
                group: present
                or any(term.encode("ascii") in member_bytes for term in terms)
                for group, terms in IDENTITY_GROUPS.items()
                for present in (identity_groups[group],)
            }
        except (OSError, zipfile.BadZipFile):
            zip_members = []
    json_paths: list[str] = []
    json_object = False
    try:
        value = json.loads(data.decode("utf-8"))
        json_object = isinstance(value, Mapping)
        if json_object:
            json_paths = _matching_json_paths(value)
    except (UnicodeDecodeError, json.JSONDecodeError):
        pass
    return {
        "anchors": anchors,
        "identity_groups": identity_groups,
        "all_identity_group_terms_present": all(identity_groups.values()),
        "structured_json_object": json_object,
        "matching_json_paths": json_paths,
        "zip_members": zip_members[:200],
        "name_mentions_p88": b"p88" in lowered_name,
    }


def _matching_json_paths(value: object, prefix: str = "$") -> list[str]:
    matches: list[str] = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            path = f"{prefix}.{key}"
            lowered = str(key).lower()
            if any(term in lowered for terms in IDENTITY_GROUPS.values() for term in terms):
                matches.append(path)
            matches.extend(_matching_json_paths(item, path))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            matches.extend(_matching_json_paths(item, f"{prefix}[{index}]"))
    return matches


def _workspace_files(root: Path, output: Path) -> Iterable[Path]:
    output_resolved = output.resolve()
    for current, directory_names, file_names in os.walk(root):
        directory_names[:] = sorted(
            name for name in directory_names if name not in EXCLUDED_DIRECTORY_NAMES
        )
        current_path = Path(current)
        for file_name in sorted(file_names):
            path = current_path / file_name
            try:
                if path.resolve() == output_resolved or not path.is_file():
                    continue
            except OSError:
                continue
            yield path


def _scan_workspace(root: Path, output: Path) -> tuple[list[dict[str, object]], dict[str, int]]:
    candidates: list[dict[str, object]] = []
    counts = Counter()
    for path in _workspace_files(root, output):
        counts["files_considered"] += 1
        try:
            size = path.stat().st_size
        except OSError:
            counts["files_stat_failed"] += 1
            continue
        if size > MAX_SCAN_BYTES:
            counts["files_skipped_over_size_cap"] += 1
            continue
        try:
            data = path.read_bytes()
        except OSError:
            counts["files_read_failed"] += 1
            continue
        counts["files_scanned"] += 1
        match = _match_payload(data, path.as_posix())
        if match is None:
            continue
        counts["files_matched"] += 1
        match.update(
            {
                "surface": "workspace_file",
                "path": path.relative_to(root).as_posix(),
                "size_bytes": size,
                "sha256": _sha256_bytes(data),
            }
        )
        candidates.append(match)
    return candidates, dict(counts)


def _read_batch_object(
    input_handle: BinaryIO,
    output_handle: BinaryIO,
    oid: str,
) -> tuple[str, int, bytes]:
    input_handle.write((oid + "\n").encode("ascii"))
    input_handle.flush()
    header = output_handle.readline().decode("ascii").strip().split()
    if len(header) != 3:
        raise RuntimeError(f"unexpected git cat-file header for {oid}: {header}")
    returned_oid, object_type, size_text = header
    if returned_oid != oid:
        raise RuntimeError(f"git cat-file returned {returned_oid} for {oid}")
    size = int(size_text)
    data = output_handle.read(size)
    terminator = output_handle.read(1)
    if len(data) != size or terminator != b"\n":
        raise RuntimeError(f"truncated git object {oid}")
    return object_type, size, data


def _git_blob_inventory(root: Path) -> tuple[list[dict[str, object]], dict[str, int]]:
    listing = subprocess.run(
        [
            "git",
            "cat-file",
            "--batch-all-objects",
            "--batch-check=%(objectname) %(objecttype) %(objectsize)",
        ],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    blob_records: list[tuple[str, int]] = []
    counts = Counter()
    for line in listing.stdout.splitlines():
        oid, object_type, size_text = line.split()
        counts[f"objects_{object_type}"] += 1
        if object_type == "blob":
            blob_records.append((oid, int(size_text)))
    candidates: list[dict[str, object]] = []
    process = subprocess.Popen(
        ["git", "cat-file", "--batch"],
        cwd=root,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
    )
    if process.stdin is None or process.stdout is None:
        raise RuntimeError("failed to open git cat-file batch pipes")
    try:
        for oid, listed_size in blob_records:
            counts["blobs_considered"] += 1
            if listed_size > MAX_SCAN_BYTES:
                counts["blobs_skipped_over_size_cap"] += 1
                continue
            object_type, size, data = _read_batch_object(
                process.stdin,
                process.stdout,
                oid,
            )
            if object_type != "blob" or size != listed_size:
                raise RuntimeError(f"git object metadata changed for {oid}")
            counts["blobs_scanned"] += 1
            match = _match_payload(data, oid)
            if match is None:
                continue
            counts["blobs_matched"] += 1
            match.update(
                {
                    "surface": "git_blob",
                    "git_oid": oid,
                    "size_bytes": size,
                    "sha256": _sha256_bytes(data),
                }
            )
            candidates.append(match)
    finally:
        process.stdin.close()
        process.stdout.close()
        return_code = process.wait()
        if return_code != 0:
            raise RuntimeError(f"git cat-file --batch failed with {return_code}")
    return candidates, dict(counts)


def _git_provenance(root: Path) -> dict[str, object]:
    head = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    refs = subprocess.run(
        ["git", "for-each-ref", "--format=%(refname) %(objectname)"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    worktrees = subprocess.run(
        ["git", "worktree", "list", "--porcelain"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    fsck = subprocess.run(
        ["git", "fsck", "--full", "--unreachable", "--no-reflogs"],
        cwd=root,
        check=False,
        capture_output=True,
        text=True,
    )
    fsck_lines = sorted(line for line in fsck.stdout.splitlines() if line.strip())
    return {
        "head": head,
        "refs": refs,
        "worktrees": worktrees,
        "unreachable_object_count": len(fsck_lines),
        "unreachable_object_type_counts": dict(
            Counter(line.split()[1] for line in fsck_lines if len(line.split()) >= 2)
        ),
        "git_fsck_return_code": fsck.returncode,
    }


def _candidate_could_bind_complete_identity(candidate: Mapping[str, object]) -> bool:
    return bool(
        candidate.get("structured_json_object")
        and candidate.get("all_identity_group_terms_present")
        and (
            "branch_hash" in candidate.get("anchors", ())
            or candidate.get("sha256") == P88_ARTIFACT_SHA256
        )
    )


def run(repository_root: Path, output: Path) -> dict[str, object]:
    started = time.monotonic()
    provenance = _git_provenance(repository_root)
    workspace_candidates, workspace_counts = _scan_workspace(repository_root, output)
    git_candidates, git_counts = _git_blob_inventory(repository_root)
    all_candidates = workspace_candidates + git_candidates
    complete_leads = [
        candidate
        for candidate in all_candidates
        if _candidate_could_bind_complete_identity(candidate)
    ]
    status = STATUS_PASS if complete_leads else STATUS_BLOCK
    p88_files = [
        candidate
        for candidate in workspace_candidates
        if candidate.get("sha256") == P88_ARTIFACT_SHA256
    ]
    return {
        "schema": SCHEMA,
        "status": status,
        "decision": (
            "manual_admission_review_required"
            if complete_leads
            else "stop_without_replay_or_replacement"
        ),
        "plan": (
            "docs/plans/bayesfilter-zhao-cui-austria-sir-fixed-variant-"
            "baseline-recovery-plan-2026-07-30.md"
        ),
        "output": output.relative_to(repository_root).as_posix(),
        "search_contract": {
            "max_scanned_file_or_blob_bytes": MAX_SCAN_BYTES,
            "anchors": ANCHORS,
            "required_identity_groups": IDENTITY_GROUPS,
            "workspace_scope": ". excluding .git and cache directories",
            "git_scope": "all loose/packed reachable and unreachable blobs",
            "numerical_reconstruction": False,
            "tensorflow_imported": False,
            "gpu_execution_attempted": False,
        },
        "git_provenance": provenance,
        "workspace_counts": workspace_counts,
        "git_counts": git_counts,
        "p88_exact_file_copies": p88_files,
        "complete_identity_leads": complete_leads,
        "complete_identity_lead_count": len(complete_leads),
        "matched_candidate_count": len(all_candidates),
        "workspace_candidates": workspace_candidates,
        "git_blob_candidates": git_candidates,
        "wall_time_seconds": time.monotonic() - started,
        "replay_executed": False,
        "replay_reason": (
            "candidate_requires_manual_admission_review"
            if complete_leads
            else "no_candidate_bound_every_required_identity_group"
        ),
        "lane_b_authorized": False,
        "phase1_authorized": False,
        "hmc_authorized": False,
        "nonclaims": [
            "filename or text matches are leads, not identity proof",
            "no exact retained filter was reconstructed unless a later admitted replay says so",
            "no value, score, T2 fit, parameter extension, GPU, or HMC claim",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.repository_root.resolve()
    output = args.output if args.output.is_absolute() else root / args.output
    if output.exists():
        raise FileExistsError(f"refusing to overwrite recovery inventory: {output}")
    output.parent.mkdir(parents=True, exist_ok=False)
    payload = run(root, output)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "output": str(output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
