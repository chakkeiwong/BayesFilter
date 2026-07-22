from __future__ import annotations

import copy
import hashlib
import json
import os
import stat
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from bayesfilter.inference.hmc_identity import canonical_artifact_payload_hash
from bayesfilter.inference.hmc_serious_authority import (
    ATTEMPT1_AUTHORITY_PATH,
    ATTEMPT1_CLAIM_PATH,
    ATTEMPT1_DOCUMENT_EXPECTATIONS,
    ATTEMPT1_LOG_PATH,
    ATTEMPT1_OUTPUT_MANIFEST_PATH,
    ATTEMPT1_PUBLIC_RESULT_PATH,
    ATTEMPT1_TERMINAL_EXPECTATIONS,
    AUTHORITY_PATH,
    HMC_PHASE7_SERIOUS_AUTHORITY_PROPOSAL_SCHEMA_V1,
    HMC_PHASE7_SERIOUS_FAILURE_SCHEMA_V1,
    HMC_PHASE7_SERIOUS_PROGRESS_SCHEMA_V1,
    HMC_PHASE7_SERIOUS_RESULT_SCHEMA_V1,
    PHASE5_PREFLIGHT_ARTIFACT_HASH,
    PREFLIGHT_PATH,
    PROPOSAL_MANIFEST_PATH,
    PROPOSAL_PATH,
    SERIOUS_AUTHORITY_DECISION,
    SERIOUS_BLOCK_DECISION,
    SERIOUS_CONFIG_HASH,
    SERIOUS_DIAGNOSTIC_DEFINITIONS,
    SERIOUS_DIAGNOSTIC_NONCLAIMS,
    SERIOUS_FAILURE_NONCLAIMS,
    SERIOUS_NONCLAIMS,
    SERIOUS_PARAMETER_NAMES,
    SERIOUS_PASS_DECISION,
    SeriousInheritedEvidenceDriftError,
    SeriousInheritedEvidenceSession,
    SecureSeriousOutputSession,
    TRANSITION_IDENTITY_HASH,
    _artifact_reference_from_snapshot,
    build_default_serious_authority_proposal,
    build_serious_authority,
    build_serious_authority_proposal_manifest,
    build_serious_launch_claim,
    default_paths,
    expected_launcher_command,
    expected_serious_approval_statement,
    parse_serious_authority,
    parse_serious_authority_proposal,
    parse_serious_authority_proposal_manifest,
    parse_serious_launch_claim,
    parse_serious_progress,
    parse_serious_terminal_result,
    verify_serious_authority_proposal_candidate,
    verify_serious_authority_proposal_manifest,
    write_historical_archive_bundle,
)
from bayesfilter.inference.hmc_smoke_authority import (
    PinnedSmokeOutputDirectories,
    SmokeOutputReservationError,
    implementation_source_bundle_hash,
)
from bayesfilter.testing import deterministic_lgssm_hmc_phase7_tf as controller


def _rehash(payload: dict) -> dict:
    restored = copy.deepcopy(payload)
    restored.pop("artifact_hash", None)
    restored["artifact_hash"] = canonical_artifact_payload_hash(restored)
    return restored


@pytest.fixture(autouse=True)
def _fixed_cpu_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "-1")
    for name, value in {
        "TF_NUM_INTRAOP_THREADS": "8",
        "TF_NUM_INTEROP_THREADS": "1",
        "OMP_NUM_THREADS": "8",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
    }.items():
        monkeypatch.setenv(name, value)


def _proposal_bundle() -> tuple[dict, dict, dict, dict]:
    with SeriousInheritedEvidenceSession.open() as evidence:
        proposal = dict(
            build_default_serious_authority_proposal(
                python_executable=sys.executable,
                evidence_session=evidence,
            )
        )
        verify_serious_authority_proposal_candidate(
            proposal,
            python_executable=sys.executable,
            evidence_session=evidence,
        )
    proposal_data = (json.dumps(proposal, sort_keys=True, indent=2) + "\n").encode()
    proposal_reference = _artifact_reference_from_snapshot(
        path=PROPOSAL_PATH,
        payload=proposal,
        data=proposal_data,
    )
    manifest = dict(
        build_serious_authority_proposal_manifest(
            proposal_reference=proposal_reference
        )
    )
    manifest_data = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode()
    manifest_reference = _artifact_reference_from_snapshot(
        path=PROPOSAL_MANIFEST_PATH,
        payload=manifest,
        data=manifest_data,
    )
    approval = expected_serious_approval_statement(manifest["artifact_hash"])
    authority = dict(
        build_serious_authority(
            approval_statement=approval,
            approval_date="2026-07-12",
            proposal_manifest=manifest,
            proposal_manifest_reference=manifest_reference,
        )
    )
    claim = dict(
        build_serious_launch_claim(
            authority=authority,
            proposal=proposal,
            manifest=manifest,
            command=proposal["command"],
            paths=proposal["paths"],
            pid=1234,
        )
    )
    return proposal, manifest, authority, claim


def test_serious_proposal_round_trip_uses_runtime_inert_v2_snapshot() -> None:
    proposal, manifest, authority, claim = _proposal_bundle()

    assert parse_serious_authority_proposal(proposal) == proposal
    assert proposal["schema"] == HMC_PHASE7_SERIOUS_AUTHORITY_PROPOSAL_SCHEMA_V1
    assert proposal["decision"] == SERIOUS_AUTHORITY_DECISION
    assert proposal["runtime"]["mode"] == "serious"
    assert proposal["runtime"]["jit_compile"] is True
    assert proposal["attempt2_output_policy"] == "exclusive_create_all_outputs"
    assert proposal["attempt1_mutation_permitted"] is False
    assert proposal["attempt1_terminal_evidence"]["runtime_reached"] is False
    assert "attempt2" in proposal["paths"]["public_result_path"]
    assert proposal["phase8_authority"] is False
    assert proposal["neutra_authority"] is False
    assert parse_serious_authority_proposal_manifest(manifest) == manifest
    assert parse_serious_authority(authority) == authority
    assert parse_serious_launch_claim(claim) == claim


@pytest.mark.parametrize(
    "mutate",
    [
        lambda item: item.update(decision="AUTHORIZE_SMOKE"),
        lambda item: item["runtime"].update(mode="smoke"),
        lambda item: item["runtime"].update(jit_compile=False),
        lambda item: item["runtime"].update(worker_count=1),
        lambda item: item.update(phase8_authority=True),
        lambda item: item.update(neutra_authority=True),
        lambda item: item.update(attempt1_mutation_permitted=True),
        lambda item: item.update(attempt2_output_policy="replace_attempt1"),
        lambda item: item["attempt1_terminal_evidence"].update(runtime_reached=True),
        lambda item: item["paths"].update(public_result_path="other.json"),
        lambda item: item.update(unexpected=True),
    ],
)
def test_serious_proposal_rejects_scope_boundary_and_path_tamper(mutate) -> None:
    proposal, _manifest, _authority, _claim = _proposal_bundle()
    mutate(proposal)
    with pytest.raises((TypeError, ValueError)):
        parse_serious_authority_proposal(_rehash(proposal))


def test_serious_authority_requires_exact_manifest_bound_statement_and_date() -> None:
    _proposal, manifest, _authority, _claim = _proposal_bundle()
    with pytest.raises(ValueError, match="approval statement"):
        build_serious_authority(
            approval_statement="approved",
            approval_date="2026-07-12",
            proposal_manifest=manifest,
            proposal_manifest_reference=manifest["proposal_reference"],
        )
    with pytest.raises(ValueError, match="YYYY-MM-DD"):
        build_serious_authority(
            approval_statement=expected_serious_approval_statement(
                manifest["artifact_hash"]
            ),
            approval_date="2026-7-12",
            proposal_manifest=manifest,
            proposal_manifest_reference=manifest["proposal_reference"],
        )


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("pid", True),
        ("pid", 0),
        ("started_at_utc", "2026-07-12T00:00:00"),
        ("started_at_utc", "2026-07-12T00:00:00Z"),
        ("file_mode", "0600"),
        ("permanent_authority_consumption", False),
        ("exclusive_attempt2_outputs_authorized", False),
        ("attempt1_mutation_authorized", True),
    ],
)
def test_serious_claim_rejects_noncanonical_or_weakened_fields(
    field: str, value
) -> None:
    _proposal, _manifest, _authority, claim = _proposal_bundle()
    claim[field] = value
    with pytest.raises(ValueError):
        parse_serious_launch_claim(_rehash(claim))


def test_serious_claim_rejects_smoke_command_and_redirected_paths() -> None:
    proposal, manifest, authority, _claim = _proposal_bundle()
    smoke_command = (
        str(Path(sys.executable).resolve()),
        "scripts/run_hmc_phase6_typed_identity_smoke.py",
        "--stage",
        "burnin_sampling",
        "--phase7-smoke",
        "--phase7-smoke-authority",
        "authority.json",
    )
    with pytest.raises(ValueError, match="command"):
        build_serious_launch_claim(
            authority=authority,
            proposal=proposal,
            manifest=manifest,
            command=smoke_command,
            paths=default_paths(),
            pid=1234,
        )
    redirected = dict(default_paths())
    redirected["private_samples_path"] = "docs/plans/private.npz"
    with pytest.raises(ValueError, match="paths|path"):
        build_serious_launch_claim(
            authority=authority,
            proposal=proposal,
            manifest=manifest,
            command=expected_launcher_command(sys.executable),
            paths=redirected,
            pid=1234,
        )


def test_attempt1_authority_and_claim_cannot_parse_as_attempt2() -> None:
    with pytest.raises(ValueError, match="identity|schema"):
        parse_serious_authority(
            json.loads(ATTEMPT1_AUTHORITY_PATH.read_text(encoding="utf-8"))
        )


def test_attempt2_paths_are_disjoint_from_every_attempt1_path() -> None:
    active = {Path(value).resolve() for value in default_paths().values()}
    inherited = set(ATTEMPT1_TERMINAL_EXPECTATIONS) | {
        ATTEMPT1_AUTHORITY_PATH,
        ATTEMPT1_CLAIM_PATH,
        ATTEMPT1_OUTPUT_MANIFEST_PATH,
        ATTEMPT1_PUBLIC_RESULT_PATH,
        ATTEMPT1_LOG_PATH,
    }
    assert active.isdisjoint(path.resolve() for path in inherited)
    assert AUTHORITY_PATH.resolve() not in inherited


def test_proposal_construction_preserves_all_attempt1_exact_bytes_and_metadata() -> None:
    exact_paths = set(ATTEMPT1_TERMINAL_EXPECTATIONS) | set(
        ATTEMPT1_DOCUMENT_EXPECTATIONS
    )
    before = {
        path: (path.read_bytes(), stat.S_IMODE(path.stat().st_mode), path.stat().st_nlink)
        for path in exact_paths
    }
    with SeriousInheritedEvidenceSession.open() as evidence:
        proposal = build_default_serious_authority_proposal(
            python_executable=sys.executable,
            evidence_session=evidence,
        )
        parse_serious_authority_proposal(proposal)
    after = {
        path: (path.read_bytes(), stat.S_IMODE(path.stat().st_mode), path.stat().st_nlink)
        for path in exact_paths
    }
    assert after == before


def test_historical_archive_compatibility_entrypoint_is_read_only() -> None:
    from bayesfilter.inference.hmc_serious_authority import (
        HISTORICAL_ARCHIVE_MANIFEST_PATH,
        HISTORICAL_ARCHIVE_PATH,
    )

    before = {
        path: (path.read_bytes(), stat.S_IMODE(path.stat().st_mode), path.stat().st_ino)
        for path in (HISTORICAL_ARCHIVE_PATH, HISTORICAL_ARCHIVE_MANIFEST_PATH)
    }
    write_historical_archive_bundle()
    after = {
        path: (path.read_bytes(), stat.S_IMODE(path.stat().st_mode), path.stat().st_ino)
        for path in (HISTORICAL_ARCHIVE_PATH, HISTORICAL_ARCHIVE_MANIFEST_PATH)
    }
    assert after == before
    with pytest.raises(ValueError, match="fields|schema"):
        parse_serious_launch_claim(
            json.loads(ATTEMPT1_CLAIM_PATH.read_text(encoding="utf-8"))
        )


def _diagnostic(*, draw_count: int = 4000) -> dict:
    rows = [
        {
            "parameter": name,
            "rank_normalized_split_rhat": 1.0,
            "folded_rank_normalized_split_rhat": 1.0,
            "rhat": 1.0,
            "bulk_ess": 1200.0,
            "tail_ess": 500.0,
            "lower_tail_ess": 500.0,
            "upper_tail_ess": 600.0,
            "passed": True,
        }
        for name in SERIOUS_PARAMETER_NAMES
    ]
    return {
        "schema": "bayesfilter.rank_normalized_hmc_diagnostics.v1",
        "passed": True,
        "input_all_finite": True,
        "diagnostics_all_finite": True,
        "draw_count_per_chain": draw_count,
        "chain_count": 4,
        "parameter_count": 18,
        "split_draw_count_per_chain": draw_count // 2,
        "split_chain_count": 8,
        "thresholds": {
            "rhat_max": 1.01,
            "bulk_ess_min": 1000.0,
            "tail_ess_min": 400.0,
        },
        "definitions": dict(SERIOUS_DIAGNOSTIC_DEFINITIONS),
        "max_rhat": 1.0,
        "min_bulk_ess": 1200.0,
        "min_tail_ess": 500.0,
        "parameter_diagnostics": rows,
        "hard_vetoes": [],
        "nonclaims": list(SERIOUS_DIAGNOSTIC_NONCLAIMS),
    }


def _serious_result() -> dict:
    preflight = json.loads(PREFLIGHT_PATH.read_text(encoding="utf-8"))
    bundle_hash = "sha256:" + "a" * 64
    workers = []
    for index, pid in enumerate((111, 222)):
        workers.append(
            {
                "worker_index": index,
                "pid": pid,
                "child_worker_cache_seal_hash": "sha256:" + str(index + 1) * 64,
                "jit_compile": True,
                "use_xla": True,
                "compile_trace_count": 1,
                "first_call_s": 1.0,
                "warm_call_s": 0.5,
                "tensorflow_version": "2.19.1",
                "tfp_version": "0.25.0",
                "python_version": "3.11.14",
                "cuda_visible_devices": "-1",
                "thread_environment": {
                    "TF_NUM_INTRAOP_THREADS": "8",
                    "TF_NUM_INTEROP_THREADS": "1",
                    "OMP_NUM_THREADS": "8",
                    "OPENBLAS_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                    "CUDA_VISIBLE_DEVICES": "-1",
                    "TF_CPP_MIN_LOG_LEVEL": "1",
                    "MPLCONFIGDIR": "/tmp/matplotlib-bayesfilter-phase7-worker",
                },
                "child_source_references_verified": True,
                "child_implementation_references_verified": True,
                "child_loaded_source_bytes_verified": True,
                "child_implementation_source_bundle_hash": bundle_hash,
                "child_transition_identity_verified": True,
                "child_transition_identity_hash": TRANSITION_IDENTITY_HASH,
            }
        )
    return _rehash(
        {
            "schema": HMC_PHASE7_SERIOUS_RESULT_SCHEMA_V1,
            "passed": True,
            "decision": SERIOUS_PASS_DECISION,
            "smoke": False,
            "serious_authority_artifact_hash": "sha256:" + "1" * 64,
            "serious_launch_claim_artifact_hash": "sha256:" + "2" * 64,
            "serious_proposal_manifest_artifact_hash": "sha256:" + "3" * 64,
            "preflight_before_runtime_artifact_hash": PHASE5_PREFLIGHT_ARTIFACT_HASH,
            "config_hash": SERIOUS_CONFIG_HASH,
            "preflight_before_runtime": preflight,
            "burnin_results_per_chain": 2000,
            "retained_results_per_chain": 4000,
            "final_diagnostics": _diagnostic(),
            "worker_count": 2,
            "chains_per_worker": 2,
            "chain_count": 4,
            "worker_pids": [111, 222],
            "worker_metadata": workers,
            "private_retained_sample_reference": {
                "file_sha256": "b" * 64,
                "byte_count": 123,
                "shape_verified": True,
                "finite_verified": True,
                "provenance_verified": True,
                "path_publicized": False,
                "raw_samples_publicized": False,
            },
            "jit_compile": True,
            "jit_compile_false_runtime_executed": False,
            "cuda_visible_devices": "-1",
            "elapsed_seconds": 1.0,
            "serious_runtime_executed": True,
            "neutra_executed": False,
            "phase8_executed": False,
            "nonclaims": list(SERIOUS_NONCLAIMS),
        }
    )


def _check(stage: str, count: int, *, passed: bool = True) -> dict:
    return {
        "stage": stage,
        "completed_results_per_chain": count,
        "passed": passed,
        "max_rhat": 1.0 if passed else 2.0,
        "min_bulk_ess": 1200.0 if passed else 100.0,
        "min_tail_ess": 500.0 if passed else 50.0,
        "input_all_finite": True,
        "diagnostics_all_finite": True,
        "hard_vetoes": [],
    }


def _serious_progress(result: dict) -> dict:
    return _rehash(
        {
            "schema": HMC_PHASE7_SERIOUS_PROGRESS_SCHEMA_V1,
            "status": "result_written",
            "config_hash": SERIOUS_CONFIG_HASH,
            "smoke": False,
            "serious_authority_artifact_hash": result[
                "serious_authority_artifact_hash"
            ],
            "serious_launch_claim_artifact_hash": result[
                "serious_launch_claim_artifact_hash"
            ],
            "serious_proposal_manifest_artifact_hash": result[
                "serious_proposal_manifest_artifact_hash"
            ],
            "preflight_before_runtime_artifact_hash": PHASE5_PREFLIGHT_ARTIFACT_HASH,
            "burnin_checks": [_check("burnin", 2000)],
            "retained_checks": [_check("retained", 4000)],
            "completed": True,
            "passed": True,
            "result_artifact_hash": result["artifact_hash"],
        }
    )


def test_serious_result_and_progress_strict_round_trip() -> None:
    result = _serious_result()
    progress = _serious_progress(result)
    assert parse_serious_terminal_result(result) == result
    assert parse_serious_progress(progress) == progress


@pytest.mark.parametrize(
    "mutate",
    [
        lambda item: item["final_diagnostics"].update(max_rhat=float("nan")),
        lambda item: item["final_diagnostics"].update(max_rhat=1.001),
        lambda item: item["final_diagnostics"].update(draw_count_per_chain=3999),
        lambda item: item["final_diagnostics"]["parameter_diagnostics"][0].update(
            parameter="wrong"
        ),
        lambda item: item["final_diagnostics"]["parameter_diagnostics"][0].update(
            rhat=1.001
        ),
        lambda item: item["worker_metadata"][0].update(worker_index=1),
        lambda item: item["worker_metadata"][0].update(pid=222),
        lambda item: item["worker_metadata"][0].update(compile_trace_count=0),
        lambda item: item["worker_metadata"][0].update(
            child_transition_identity_hash="sha256:" + "f" * 64
        ),
        lambda item: item.update(jit_compile=False),
        lambda item: item.update(phase8_executed=True),
        lambda item: item.update(neutra_executed=True),
        lambda item: item["private_retained_sample_reference"].update(
            byte_count=0
        ),
    ],
)
def test_serious_result_rejects_forged_terminal_pass(mutate) -> None:
    result = _serious_result()
    mutate(result)
    with pytest.raises((TypeError, ValueError)):
        parse_serious_terminal_result(_rehash(result))


def test_serious_progress_requires_burnin_pass_before_retained() -> None:
    result = _serious_result()
    progress = _serious_progress(result)
    progress["burnin_checks"][0] = _check("burnin", 2000, passed=False)
    with pytest.raises(ValueError, match="before burn-in passed"):
        parse_serious_progress(_rehash(progress))


def test_serious_failure_cannot_claim_phase8_or_neutra() -> None:
    preflight = json.loads(PREFLIGHT_PATH.read_text(encoding="utf-8"))
    failure = _rehash(
        {
            "schema": HMC_PHASE7_SERIOUS_FAILURE_SCHEMA_V1,
            "passed": False,
            "decision": SERIOUS_BLOCK_DECISION,
            "smoke": False,
            "serious_authority_artifact_hash": "sha256:" + "1" * 64,
            "serious_launch_claim_artifact_hash": "sha256:" + "2" * 64,
            "serious_proposal_manifest_artifact_hash": "sha256:" + "3" * 64,
            "preflight_before_runtime_artifact_hash": PHASE5_PREFLIGHT_ARTIFACT_HASH,
            "stage": "burnin",
            "reason": "burnin_diagnostics_failed_at_cap",
            "config_hash": SERIOUS_CONFIG_HASH,
            "preflight_before_runtime": preflight,
            "worker_pids": [111, 222],
            "final_diagnostics": None,
            "jit_compile_false_runtime_executed": False,
            "cuda_visible_devices": "-1",
            "elapsed_seconds": 1.0,
            "serious_runtime_executed": True,
            "neutra_executed": False,
            "phase8_executed": False,
            "nonclaims": list(SERIOUS_FAILURE_NONCLAIMS),
        }
    )
    assert parse_serious_terminal_result(failure) == failure
    for field in ("phase8_executed", "neutra_executed"):
        tampered = copy.deepcopy(failure)
        tampered[field] = True
        with pytest.raises(ValueError, match="execution boundary"):
            parse_serious_terminal_result(_rehash(tampered))


def test_inherited_evidence_session_enforces_reviewed_modes() -> None:
    with SeriousInheritedEvidenceSession.open() as evidence:
        report = evidence.verify()
        assert report["phase6_smoke_attempt2_launch_claim.json"][
            "file_mode"
        ] == "0400"
        assert report["typed_identity_baseline_preflight.json"][
            "file_mode"
        ] == "0600"
        assert report["burnin_sampling.json"]["file_mode"] == "0400"
        assert report["burnin_sampling.json"]["byte_count"] == 0


def test_pin_additional_rejects_symlink_before_normalization(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    target.write_text("{}\n", encoding="utf-8")
    alias = tmp_path / "alias.json"
    alias.symlink_to(target)
    with SeriousInheritedEvidenceSession.open() as evidence:
        with pytest.raises(ValueError, match="symlink"):
            evidence.pin_additional(alias)


def test_historical_live_path_can_be_retired_only_after_atomic_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    old = tmp_path / "historical.json"
    old.write_bytes(b"old\n")
    archive = tmp_path / "historical.archive.json"
    archive.write_bytes(b"old\n")
    archive_manifest = tmp_path / "historical.archive.manifest.json"
    archive_manifest.write_bytes(b"{}\n")
    replacement = tmp_path / "replacement.json"
    replacement.write_bytes(b"")
    evidence_fd = os.open(old, os.O_RDONLY)
    capability_fd = os.open(old, os.O_RDONLY)
    archive_fd = os.open(archive, os.O_RDONLY)
    archive_manifest_fd = os.open(archive_manifest, os.O_RDONLY)
    replacement_fd = os.open(replacement, os.O_RDWR)
    replacement.chmod(0o400)
    try:
        session = object.__new__(SeriousInheritedEvidenceSession)
        session.attempt1 = SimpleNamespace(verify=lambda: {}, close=lambda: None)
        session.parent_entries = {}
        session.entries = []
        session._closed = False
        session._retired_path_invariants = set()
        session._EXPECTED_MODES = {
            old: 0o644,
            archive: 0o644,
            archive_manifest: 0o644,
        }
        parent_fd = os.open(tmp_path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
        parent_stat = os.fstat(parent_fd)
        session.parent_entries[tmp_path] = (
            parent_fd,
            (parent_stat.st_dev, parent_stat.st_ino),
        )
        old_stat = os.fstat(evidence_fd)
        session.entries.append(
            (old, parent_fd, evidence_fd, session._signature(old_stat), b"old\n")
        )
        for path, fd, data in (
            (archive, archive_fd, b"old\n"),
            (archive_manifest, archive_manifest_fd, b"{}\n"),
        ):
            session.entries.append(
                (path, parent_fd, fd, session._signature(os.fstat(fd)), data)
            )
        from bayesfilter.inference import hmc_serious_authority as serious

        monkeypatch.setattr(serious, "HISTORICAL_RESULT_PATH", old)
        monkeypatch.setattr(serious, "HISTORICAL_ARCHIVE_PATH", archive)
        monkeypatch.setattr(
            serious, "HISTORICAL_ARCHIVE_MANIFEST_PATH", archive_manifest
        )
        try:
            os.replace(replacement, old)
            session.retire_replaced_path(
                old,
                original_fd=capability_fd,
                replacement_fd=replacement_fd,
            )
            assert old in session._retired_path_invariants
            assert os.pread(evidence_fd, 4, 0) == b"old\n"
            assert os.pread(capability_fd, 4, 0) == b"old\n"
            assert old.read_bytes() == b""
        finally:
            session.close()
    finally:
        for fd in (
            replacement_fd,
            archive_manifest_fd,
            archive_fd,
            capability_fd,
            evidence_fd,
        ):
            try:
                os.close(fd)
            except OSError:
                pass


def test_historical_retirement_rejects_different_inode_descriptor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    old = tmp_path / "historical.json"
    old.write_bytes(b"old\n")
    other = tmp_path / "other.json"
    other.write_bytes(b"old\n")
    replacement = tmp_path / "replacement.json"
    replacement.write_bytes(b"")
    evidence_fd = os.open(old, os.O_RDONLY)
    other_fd = os.open(other, os.O_RDONLY)
    replacement_fd = os.open(replacement, os.O_RDWR)
    replacement.chmod(0o400)
    parent_fd = os.open(tmp_path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        session = object.__new__(SeriousInheritedEvidenceSession)
        session.parent_entries = {
            tmp_path: (
                parent_fd,
                (os.fstat(parent_fd).st_dev, os.fstat(parent_fd).st_ino),
            )
        }
        original_stat = os.fstat(evidence_fd)
        session.entries = [
            (
                old,
                parent_fd,
                evidence_fd,
                session._signature(original_stat),
                b"old\n",
            )
        ]
        session._retired_path_invariants = set()
        from bayesfilter.inference import hmc_serious_authority as serious

        monkeypatch.setattr(serious, "HISTORICAL_RESULT_PATH", old)
        os.replace(replacement, old)
        with pytest.raises(RuntimeError, match="descriptor changed"):
            session.retire_replaced_path(
                old,
                original_fd=other_fd,
                replacement_fd=replacement_fd,
            )
    finally:
        for fd in (parent_fd, replacement_fd, other_fd, evidence_fd):
            try:
                os.close(fd)
            except OSError:
                pass


def test_attempt2_reservation_is_exclusive_and_preserves_attempt1(
    tmp_path: Path,
) -> None:
    paths = {name: tmp_path / f"{name}.out" for name in default_paths()}
    directories = PinnedSmokeOutputDirectories.open(paths, repo_root=tmp_path)
    claim_fd = directories.open_exclusive("claim_path")
    collision = paths["public_result_path"]
    collision.write_text("other lane\n", encoding="utf-8")
    attempt1_before = ATTEMPT1_PUBLIC_RESULT_PATH.read_bytes()
    context = SimpleNamespace(
        output_directories=directories,
        claim_fd=claim_fd,
        consumed_evidence_session=SimpleNamespace(verify=lambda: {}),
    )
    session = None
    try:
        with pytest.raises(SmokeOutputReservationError) as caught:
            SecureSeriousOutputSession.reserve_from_context(context)
        session = caught.value.session
        assert caught.value.role == "public_result_path"
        assert collision.read_text(encoding="utf-8") == "other lane\n"
        assert ATTEMPT1_PUBLIC_RESULT_PATH.read_bytes() == attempt1_before == b""
    finally:
        if session is not None:
            session.close()
        else:
            os.close(claim_fd)
            directories.close()


def test_serious_output_metadata_binds_source_bundle_and_worker_seal() -> None:
    proposal, manifest, authority, claim = _proposal_bundle()
    config = controller.DeterministicLGSSMPhase7Config.load()
    preflight = json.loads(PREFLIGHT_PATH.read_text(encoding="utf-8"))
    source_bundle = {"repository_file:test.py": b"test source\n"}
    context = SimpleNamespace(
        authority_kind="phase7_serious",
        config=config,
        preflight=preflight,
        proposal=proposal,
        proposal_manifest=manifest,
        authority=authority,
        claim=claim,
        implementation_source_bundle=source_bundle,
    )
    seal = controller._secure_worker_cache_seal(
        config,
        worker_index=0,
        smoke=False,
        target_scope=preflight["target_scope"],
        launch_context=context,
    )
    assert seal["implementation_source_bundle_hash"] == (
        implementation_source_bundle_hash(source_bundle)
    )
    copied_context = copy.copy(context)
    copied_context.claim = {"artifact_hash": "sha256:" + "f" * 64}
    copied_seal = controller._secure_worker_cache_seal(
        config,
        worker_index=0,
        smoke=False,
        target_scope=preflight["target_scope"],
        launch_context=copied_context,
    )
    assert copied_seal["artifact_hash"] != seal["artifact_hash"]
