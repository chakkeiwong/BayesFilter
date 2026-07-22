from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import pytest

from scripts import audit_complete_highdim_leaderboard_post_run_integrity as audit


def test_post_run_archive_membership_accepts_regular_relative_files(
    tmp_path: Path,
) -> None:
    archive_path = tmp_path / "primary.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        for name, content in (
            ("docs/plans/final.json", b"{}\n"),
            ("docs/plans/final.md", b"result\n"),
        ):
            info = tarfile.TarInfo(name)
            info.size = len(content)
            archive.addfile(info, io.BytesIO(content))

    assert audit._safe_archive_names(archive_path) == [  # noqa: SLF001
        "docs/plans/final.json",
        "docs/plans/final.md",
    ]


@pytest.mark.parametrize("unsafe_name", ["../escape", "/absolute", "a/../../b"])
def test_post_run_archive_membership_rejects_traversal(
    tmp_path: Path, unsafe_name: str
) -> None:
    archive_path = tmp_path / "unsafe.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        content = b"unsafe\n"
        info = tarfile.TarInfo(unsafe_name)
        info.size = len(content)
        archive.addfile(info, io.BytesIO(content))

    with pytest.raises(ValueError, match="unsafe"):
        audit._safe_archive_names(archive_path)  # noqa: SLF001


def test_post_run_archive_membership_rejects_symlink(tmp_path: Path) -> None:
    archive_path = tmp_path / "symlink.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        info = tarfile.TarInfo("link")
        info.type = tarfile.SYMTYPE
        info.linkname = "target"
        archive.addfile(info)

    with pytest.raises(ValueError, match="unsafe"):
        audit._safe_archive_names(archive_path)  # noqa: SLF001


def test_post_run_change_manifest_rejects_duplicate_or_overlapping_paths() -> None:
    with pytest.raises(ValueError, match="duplicates"):
        audit._safe_change_paths(  # noqa: SLF001
            {"added": ["a", "a"], "modified": [], "deleted": []}
        )
    with pytest.raises(ValueError, match="overlap"):
        audit._safe_change_paths(  # noqa: SLF001
            {"added": ["a"], "modified": ["a"], "deleted": []}
        )


def test_post_run_regular_file_must_be_read_only(tmp_path: Path) -> None:
    path = tmp_path / "artifact.json"
    path.write_text("{}\n", encoding="utf-8")

    with pytest.raises(ValueError, match="writable"):
        audit._safe_regular(path, tmp_path)  # noqa: SLF001

    path.chmod(0o444)
    assert audit._safe_regular(path, tmp_path).st_nlink == 1  # noqa: SLF001


def test_post_run_mount_decoder_handles_kernel_escapes() -> None:
    assert audit._decode_mount_path("/tmp/a\\040b\\011c") == "/tmp/a b\tc"  # noqa: SLF001


def test_post_run_mount_check_includes_snapshot_alias(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    handoff = tmp_path / "handoff"
    mountinfo = tmp_path / "mountinfo"
    mountinfo.write_text(
        "36 25 0:32 / "
        + str(audit.SNAPSHOT_HANDOFF)
        + " ro,relatime - ext4 /dev/test ro\n",
        encoding="utf-8",
    )
    original_open = Path.open

    def redirected_open(path: Path, *args: object, **kwargs: object):
        if path == Path("/proc/self/mountinfo"):
            return original_open(mountinfo, *args, **kwargs)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", redirected_open)
    result = audit._mount_aliases_are_closed(handoff)  # noqa: SLF001

    assert str(audit.SNAPSHOT_HANDOFF) in result["checked_aliases"]
    assert result["observed_mounts"] == [
        {
            "target": str(audit.SNAPSHOT_HANDOFF),
            "read_only": True,
            "mount_id": 36,
        }
    ]


def test_post_run_mount_check_rejects_writable_snapshot_alias(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    handoff = tmp_path / "handoff"
    mountinfo = tmp_path / "mountinfo"
    mountinfo.write_text(
        "36 25 0:32 / "
        + str(audit.SNAPSHOT_HANDOFF)
        + " rw,relatime - ext4 /dev/test rw\n",
        encoding="utf-8",
    )
    original_open = Path.open

    def redirected_open(path: Path, *args: object, **kwargs: object):
        if path == Path("/proc/self/mountinfo"):
            return original_open(mountinfo, *args, **kwargs)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", redirected_open)
    with pytest.raises(ValueError, match="handoff alias is mounted writable"):
        audit._mount_aliases_are_closed(handoff)  # noqa: SLF001


def test_recorded_snapshot_alias_maps_to_canonical_handoff(tmp_path: Path) -> None:
    handoff = tmp_path / "handoff"
    handoff.mkdir()
    name = f"{audit.RUN_ID}-artifact.json"
    path = handoff / name
    path.write_text("{}\n", encoding="utf-8")
    path.chmod(0o444)

    canonical, info = audit._canonical_recorded_file(  # noqa: SLF001
        audit.SNAPSHOT_HANDOFF / name, handoff
    )

    assert canonical == path
    assert info.st_nlink == 1
    with pytest.raises(ValueError, match="wrong alias"):
        audit._canonical_recorded_file(handoff / name, handoff)  # noqa: SLF001


def test_credential_scan_fails_on_handoff_match_without_echoing_secret(
    tmp_path: Path,
) -> None:
    handoff = tmp_path / "handoff"
    handoff.mkdir()
    secret = b"test-current-credential-value"
    leaked = handoff / "artifact.log"
    leaked.write_bytes(b"prefix " + secret + b" suffix")
    archive_path = handoff / "archive.tar.gz"
    with tarfile.open(archive_path, "w:gz"):
        pass

    with pytest.raises(ValueError, match="credential value leaked") as caught:
        audit._scan_credentials(  # noqa: SLF001
            handoff=handoff,
            archive_path=archive_path,
            archive_names=[],
            secrets=(secret,),
        )
    assert secret.decode() not in str(caught.value)


def _claude_payloads(*, tool_name: str = "Read", malformed: bool = False) -> dict[str, bytes]:
    prefix = ".complete_highdim_claude_audit/20260712T000000-test"
    raw_name = prefix + "-stream.jsonl"
    stderr_name = prefix + "-stderr.log"
    metadata_name = prefix + "-metadata.json"
    metadata = {
        "schema_version": audit.CLAUDE_AUDIT_SCHEMA,
        "worker_name": "test",
        "permission_mode": "plan",
        "metadata_generated_after_worker_exit_or_signal": True,
        "parsed_event_count": 2,
        "stream_parse_complete": not malformed,
        "invalid_stream_line_count": int(malformed),
        "observed_tool_uses": [{"id": "1", "name": tool_name}],
        "disallowed_tool_uses": [] if tool_name == "Read" else [
            {"id": "1", "name": tool_name}
        ],
        "read_only_instruction_contract_satisfied_by_observed_tools": (
            tool_name == "Read" and not malformed
        ),
        "technical_tool_capability_absent": False,
        "prompt_or_credential_value_recorded_by_wrapper": False,
        "raw_stream_path": raw_name,
        "stderr_path": stderr_name,
    }
    return {
        metadata_name: (json.dumps(metadata) + "\n").encode(),
        raw_name: (
            json.dumps(
                {
                    "type": "assistant",
                    "message": {
                        "content": [
                            {"type": "tool_use", "id": "1", "name": tool_name}
                        ]
                    },
                }
            )
            + "\n"
            + json.dumps({"type": "result", "result": "VERDICT: AGREE"})
            + "\n"
        ).encode(),
        stderr_name: b"",
    }


def test_claude_tool_audit_accepts_observed_read_only_use() -> None:
    result = audit._validate_claude_audit_records(_claude_payloads())  # noqa: SLF001

    assert result["observed_non_read_only_tool_use"] is False
    assert result["audited_invocation_count"] == 1


def test_claude_tool_audit_requires_at_least_one_invocation() -> None:
    with pytest.raises(ValueError, match="no audited Claude invocation"):
        audit._validate_claude_audit_records({})  # noqa: SLF001


def test_claude_tool_audit_rejects_metadata_raw_tool_mismatch() -> None:
    payloads = _claude_payloads()
    raw_name = next(name for name in payloads if name.endswith("-stream.jsonl"))
    payloads[raw_name] = payloads[raw_name].replace(b'"Read"', b'"Bash"')

    with pytest.raises(ValueError, match="read-only tool audit failed"):
        audit._validate_claude_audit_records(payloads)  # noqa: SLF001


def test_claude_tool_audit_rejects_consistently_ignored_non_object_event() -> None:
    payloads = _claude_payloads()
    metadata_name = next(name for name in payloads if name.endswith("-metadata.json"))
    raw_name = next(name for name in payloads if name.endswith("-stream.jsonl"))
    metadata = json.loads(payloads[metadata_name])
    metadata["parsed_event_count"] = 1
    metadata["observed_tool_uses"] = []
    payloads[metadata_name] = (json.dumps(metadata) + "\n").encode()
    payloads[raw_name] = (
        json.dumps([{"type": "tool_use", "id": "hidden", "name": "Bash"}])
        + "\n"
        + json.dumps({"type": "result", "result": "VERDICT: AGREE"})
        + "\n"
    ).encode()

    with pytest.raises(ValueError, match="read-only tool audit failed"):
        audit._validate_claude_audit_records(payloads)  # noqa: SLF001


@pytest.mark.parametrize(
    ("tool_name", "malformed"),
    [("Bash", False), ("Edit", False), ("Read", True)],
)
def test_claude_tool_audit_rejects_state_change_or_malformed_stream(
    tool_name: str, malformed: bool
) -> None:
    with pytest.raises(ValueError, match="read-only tool audit failed"):
        audit._validate_claude_audit_records(  # noqa: SLF001
            _claude_payloads(tool_name=tool_name, malformed=malformed)
        )
