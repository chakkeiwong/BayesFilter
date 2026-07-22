from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path

from scripts import build_complete_highdim_leaderboard_launch_manifest as manifest
from scripts import prepare_complete_highdim_leaderboard_launch as preparation


def test_launch_manifest_binds_all_required_files_and_remains_unauthorized() -> None:
    payload = manifest.build_manifest()

    assert payload["schema_version"] == (
        "bayesfilter.complete_highdim_leaderboard.launch_manifest.v7"
    )
    missing = [
        entry["path"]
        for entry in payload["repository_bound_files"]
        if not entry["exists"]
    ]
    assert missing == []
    assert all(entry["exists"] for entry in payload["external_bound_files"])
    assert all(entry["sha256"] for entry in payload["external_bound_files"])
    assert all(entry["exists"] for entry in payload["runtime_executables"])
    assert payload["preflight"]["phase0_status"] == "PASS_PHASE0_BOUNDARY_FREEZE"
    assert payload["preflight"]["phase1_subplan_review"]["valid"] is True
    expected_ready = all(
        (
            payload["preflight"]["phase0_status"]
            == "PASS_PHASE0_BOUNDARY_FREEZE",
            payload["preflight"]["phase1_subplan_review"]["valid"],
            payload["preflight"]["gpu"]["valid"],
            payload["preflight"]["codex"]["preliminary_health_valid"],
            payload["preflight"]["isolation"]["valid"],
        )
    )
    assert payload["preflight"]["static_evidence_ready_for_launch_review"] is expected_ready
    assert payload["preflight"]["ready_except_human_approval"] is False
    assert payload["preflight"]["ready_to_request_human_approval"] is False
    assert payload["preflight"]["launch_authorized"] is False
    assert payload["approval_contract"]["automatic_merge_back"] is False
    assert payload["approval_contract"]["commit_push_merge_allowed"] is False
    assert payload["concrete_run"]["run_id"] == manifest.RUN_ID
    assert payload["concrete_run"]["launch_root"] == str(manifest.LAUNCH_ROOT)
    assert "<" not in payload["exact_wrapper_shell"]
    assert payload["exact_wrapper_argv"][:2] == ["/usr/bin/bash", str(
        manifest.SOURCE_SNAPSHOT_ROOT
        / "scripts/complete_highdim_leaderboard_exact_wrapper.sh"
    )]
    assert "--run-id" in payload["exact_wrapper_argv"]
    assert payload["frozen_source_snapshot"]["inventory"]["exists"] is True
    assert payload["runtime_fingerprint"]["exists"] is True
    assert payload["authorization_state_machine"][
        "static_technical_evidence_ready_to_request_human_approval"
    ] is expected_ready
    deadlines = payload["runtime_deadlines_seconds"]
    assert deadlines == {
        "common_clock_origin": "exact_wrapper_LAUNCH_STARTED_MONOTONIC",
        "codex_soft_cutoff": 26400,
        "codex_termination_deadline": 26700,
        "primary_export_timeout": 600,
        "watchdog_primary_verification_deadline": 27360,
        "namespace_outer_timeout": 27600,
        "supervisor_and_watchdog_hard_deadline": 28200,
        "producer_close_deadline": 28230,
        "outer_finalizer_deadline": 28720,
        "outer_term_deadline": 28740,
        "outer_kill_deadline": 28800,
    }
    assert deadlines["codex_termination_deadline"] + deadlines[
        "primary_export_timeout"
    ] <= deadlines["watchdog_primary_verification_deadline"]
    chronological = [
        deadlines["codex_soft_cutoff"],
        deadlines["codex_termination_deadline"],
        deadlines["watchdog_primary_verification_deadline"],
        deadlines["namespace_outer_timeout"],
        deadlines["supervisor_and_watchdog_hard_deadline"],
        deadlines["producer_close_deadline"],
        deadlines["outer_finalizer_deadline"],
        deadlines["outer_term_deadline"],
        deadlines["outer_kill_deadline"],
    ]
    assert chronological == sorted(chronological)
    disclosure = payload["codex_readable_initial_disclosure"]
    assert disclosure["runtime_trees"]["read_only"] is True
    assert disclosure["credential_channels"]["secret_values_logged_or_hashed"] is False
    assert disclosure["dynamic_handoff_staging_aliases_hidden"] is True
    assert disclosure["claude_technical_access"][
        "inherits_isolated_codex_filesystem_and_runtime_surface"
    ] is True
    assert disclosure["claude_technical_access"][
        "single_path_is_filesystem_enforced"
    ] is False
    assert disclosure["claude_technical_access"][
        "copied_snapshot_os_writable"
    ] is True
    assert disclosure["claude_technical_access"][
        "read_only_role_is_filesystem_enforced"
    ] is False
    assert disclosure["claude_review_prompt_policy"][
        "not_a_filesystem_isolation_boundary"
    ] is True
    assert payload["approval_contract"]["primary_export_only"] is True
    assert payload["approval_contract"]["fallback_export_allowed"] is False
    limitations = payload["owner_accepted_run_scoped_limitations"]
    assert limitations["run_id"] == manifest.RUN_ID
    assert limitations["repository_default"] is False
    assert limitations["reusable_for_other_runs"] is False
    assert limitations["technical_guarantees_repaired"] is False
    assert limitations["accepted_limitation_ids"] == list(
        manifest.OWNER_ACCEPTED_RUN_SCOPED_LIMITATION_IDS
    )
    assert limitations["sixth_review_round"] == {
        "authorized": True,
        "iteration": 6,
        "scope": "bind_this_waiver_post_run_audit_and_exact_launch_package_only",
        "ordinary_phase_review_limit_unchanged": 5,
    }
    assert all(
        limitation["accepted_not_repaired"] is True
        for limitation in limitations["accepted_limitations"]
    )
    audit = payload["mandatory_post_run_integrity_audit"]
    assert audit["required_before_completion_or_release_claim"] is True
    assert audit["plan"]["exists"] is True
    assert audit["plan"]["sha256"]
    assert audit["auditor"]["exists"] is True
    assert audit["auditor"]["sha256"]
    assert audit["launch_itself_grants_completion_or_release_authority"] is False
    assert payload["authorization_state_machine"][
        "completion_or_release_authority_granted_by_manifest"
    ] is False
    assert payload["concrete_run"][
        "final_exact_launch_command_approved"
    ] is False
    assert disclosure["credential_channels"][
        "claude_can_read_ephemeral_private_codex_auth_copy"
    ] is True
    assert disclosure["claude_technical_access"][
        "edit_and_command_tools_technically_available"
    ] is True
    assert disclosure["claude_technical_access"][
        "read_only_role_enforced_by_bound_settings"
    ] is False
    support = {
        record["destination_name"]: record["source"]
        for record in disclosure["support_files"]
    }
    assert support["trusted-claude-worker.sh"]["path"] == (
        "scripts/complete_highdim_leaderboard_claude_audit_worker.sh"
    )
    assert payload["mandatory_post_run_integrity_audit"][
        "semantic_inspection_receipt_required"
    ] is True
    assert payload["mandatory_post_run_integrity_audit"]["post_lock_receipt"][
        "required_for_structural_pass"
    ] is True
    assert payload["mandatory_post_run_integrity_audit"][
        "phase8_phase9_validator_exit_zero_and_all_checks_required"
    ] is True


def test_outer_boundary_recursively_closes_live_source_before_launch() -> None:
    source = (
        manifest.ROOT
        / "scripts/complete_highdim_leaderboard_outer_launch_boundary.sh"
    ).read_text(encoding="utf-8")

    wrapper = (
        manifest.ROOT
        / "scripts/complete_highdim_leaderboard_exact_wrapper.sh"
    ).read_text(encoding="utf-8")
    recursive_remount = source.index('findmnt -Rrno TARGET "$LIVE_ROOT"')
    pre_handoff_verifier = source.index(
        '"$PYTHON" "$LAUNCHER" "${args[@]}" --verify-only-before-handoff'
    )
    handoff_creation = source.index('/usr/bin/mkdir -m 700 "$handoff"')
    launcher = source.rindex('"$PYTHON" "$LAUNCHER" "${args[@]}"')
    assert "LAUNCH_STARTED_MONOTONIC" in wrapper
    assert wrapper.index("LAUNCH_STARTED_MONOTONIC") < wrapper.index(
        "exec /usr/bin/timeout"
    )
    assert pre_handoff_verifier < handoff_creation < recursive_remount < launcher
    assert 'remount,bind,ro "$target"' in source
    assert "LAUNCH_STARTED_MONOTONIC" in source
    assert 'mount --bind "$handoff" "$handoff_staging"' in source
    assert 'mount --bind "$handoff_staging" "$snapshot_handoff"' in source
    assert 'remount,bind,ro "$handoff_staging"' in source
    post_lock = source.index('post_lock_writer="$source_snapshot_root/scripts/')
    assert source.index('remount,bind,ro "$handoff_staging"') < post_lock
    assert source.index("forbidden-staging-post-seal-write") < post_lock
    assert "write_complete_highdim_leaderboard_post_lock_receipt.py" in source
    assert "failed to enumerate live-source mounts" in source
    assert "findmnt repository inventory was partial or inconsistent" in source
    assert 'print("\\n".join(sorted(set(targets))))' in source
    assert '/usr/bin/sort -u' in source


def test_isolation_preflight_supplies_approval_expiry_to_namespace_receipt() -> None:
    source = (
        manifest.ROOT
        / "scripts/run_complete_highdim_leaderboard_isolation_preflight.py"
    ).read_text(encoding="utf-8")

    assert '"approval_not_after_epoch": started_epoch + 3600' in source


def test_snapshot_overlays_are_hash_bound_and_copied_once(tmp_path: Path) -> None:
    program_root = tmp_path / "program"
    launch_root = tmp_path / "launch"
    source = program_root / "docs/plans/ledger.md"
    source.parent.mkdir(parents=True)
    launch_root.mkdir()
    source.write_text("reviewed ledger\n", encoding="utf-8")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    payload = {
        "program_root": str(program_root),
        "snapshot_overlay_files": [
            {
                "path": "docs/plans/ledger.md",
                "exists": True,
                "resolved_path": str(source.resolve()),
                "sha256": digest,
            }
        ],
    }

    preparation._install_snapshot_overlays(payload, launch_root)  # noqa: SLF001

    copied = launch_root / "docs/plans/ledger.md"
    assert copied.read_bytes() == source.read_bytes()
    try:
        preparation._install_snapshot_overlays(payload, launch_root)  # noqa: SLF001
    except RuntimeError as error:
        assert "collided" in str(error)
    else:
        raise AssertionError("an existing overlay destination was overwritten")


def test_launch_manifest_check_detects_bound_file_drift_shape() -> None:
    payload = manifest.build_manifest()
    forged = copy.deepcopy(payload)
    forged["repository_bound_files"][0]["sha256"] = "0" * 64

    assert forged != manifest.build_manifest()


def test_launch_review_verifier_rejects_wrong_manifest_hash(tmp_path) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text("{}\n", encoding="utf-8")
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(
        json.dumps(
            {
                "verdict": "AGREE",
                "reviewed_manifest_path": str(manifest_path),
                "reviewed_manifest_sha256": "0" * 64,
            }
        ),
        encoding="utf-8",
    )

    try:
        manifest._verify_launch_review(receipt_path, manifest_path)  # noqa: SLF001
    except ValueError as error:
        assert "relative_to" in str(error) or "bind" in str(error)
    else:
        raise AssertionError("wrong launch manifest hash was accepted")


def _launch_review_receipt(payload: dict, manifest_path: Path) -> dict:
    limitations = payload["owner_accepted_run_scoped_limitations"]
    audit = payload["mandatory_post_run_integrity_audit"]
    concrete = payload["concrete_run"]
    return {
        "schema_version": (
            "bayesfilter.complete_highdim_leaderboard."
            "launch_readiness_review_receipt.v2"
        ),
        "verdict": "AGREE",
        "iteration": 6,
        "review_scope": (
            "run_scoped_waiver_post_run_audit_and_exact_launch_package"
        ),
        "reviewed_manifest_path": str(manifest_path),
        "reviewed_manifest_sha256": hashlib.sha256(
            manifest_path.read_bytes()
        ).hexdigest(),
        "run_id": concrete["run_id"],
        "launch_root": concrete["launch_root"],
        "nonce": concrete["copy_sentinel_nonce"],
        "source_snapshot_root": concrete["source_snapshot_root"],
        "source_inventory_sha256": payload["frozen_source_snapshot"][
            "inventory"
        ]["sha256"],
        "runtime_fingerprint_sha256": payload["runtime_fingerprint"]["sha256"],
        "approval_instance_id": concrete["approval_instance_id"],
        "approval_not_after_epoch": concrete["approval_not_after_epoch"],
        "exact_wrapper_argv": payload["exact_wrapper_argv"],
        "accepted_limitation_ids": list(
            manifest.OWNER_ACCEPTED_RUN_SCOPED_LIMITATION_IDS
        ),
        "risk_acceptance_amendment_sha256": limitations[
            "authorization_amendment"
        ]["sha256"],
        "post_run_integrity_audit_plan_sha256": audit["plan"]["sha256"],
        "post_run_integrity_auditor_sha256": audit["auditor"]["sha256"],
        "final_exact_launch_command_authorized": False,
        "completion_or_release_authority_granted": False,
    }


def test_launch_review_verifier_accepts_only_run_scoped_iteration_six(
    tmp_path: Path,
) -> None:
    payload = manifest.build_manifest()
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    receipt = _launch_review_receipt(payload, manifest_path)
    receipt_path = tmp_path / "receipt.json"
    receipt_path.write_text(json.dumps(receipt), encoding="utf-8")

    manifest._verify_launch_review(receipt_path, manifest_path)  # noqa: SLF001

    for field, value in (
        ("iteration", 5),
        ("accepted_limitation_ids", ["CLAUDE_TOOL_CAPABILITY"]),
        ("final_exact_launch_command_authorized", True),
        ("completion_or_release_authority_granted", True),
    ):
        forged = copy.deepcopy(receipt)
        forged[field] = value
        receipt_path.write_text(json.dumps(forged), encoding="utf-8")
        try:
            manifest._verify_launch_review(  # noqa: SLF001
                receipt_path, manifest_path
            )
        except ValueError as error:
            assert "run-scoped waiver" in str(error)
        else:
            raise AssertionError(f"forged launch review field was accepted: {field}")
