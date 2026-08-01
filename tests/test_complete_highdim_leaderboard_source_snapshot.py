from __future__ import annotations

import json
from pathlib import Path

from scripts import freeze_complete_highdim_leaderboard_source_snapshot as snapshot


def test_snapshot_excludes_model_state_and_reincludes_tracked_log(
    tmp_path: Path, monkeypatch
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / ".git").mkdir()
    (root / "code.py").write_text("value = 1\n", encoding="utf-8")
    (root / ".claude").mkdir()
    (root / ".claude/session.json").write_text("secret\n", encoding="utf-8")
    (root / ".localenv/bin").mkdir(parents=True)
    (root / ".localenv/bin/python").symlink_to("/external/python")
    log = root / "docs/plans/logs/old/evidence.txt"
    log.parent.mkdir(parents=True)
    log.write_text("evidence\n", encoding="utf-8")
    monkeypatch.setattr(snapshot, "_tracked_log_paths", lambda _root: {log.relative_to(root).as_posix()})

    payload = snapshot.inventory(root)
    paths = {record["path"] for record in payload["entries"]}

    assert "code.py" in paths
    assert "docs/plans/logs/old/evidence.txt" in paths
    assert ".claude/session.json" not in paths
    assert ".localenv/bin/python" not in paths


def test_frozen_snapshot_detects_file_drift(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "repo"
    frozen = tmp_path / "frozen"
    inventory_path = tmp_path / "inventory.json"
    root.mkdir()
    target = root / "code.py"
    target.write_text("before\n", encoding="utf-8")
    monkeypatch.setattr(snapshot, "_tracked_log_paths", lambda _root: set())

    assert snapshot.main(
        [
            "freeze",
            "--source",
            str(root),
            "--snapshot-root",
            str(frozen),
            "--inventory",
            str(inventory_path),
            "--required-empty-directory",
            "docs/plans/logs/test-run",
        ]
    ) == 0
    payload = json.loads(inventory_path.read_text(encoding="utf-8"))
    snapshot.verify(frozen, payload)
    assert (frozen / "docs/plans/logs/test-run").is_dir()
    (frozen / "code.py").write_text("after\n", encoding="utf-8")

    try:
        snapshot.verify(frozen, payload)
    except ValueError as error:
        assert "verification failed" in str(error)
    else:
        raise AssertionError("snapshot drift was accepted")

    (frozen / "code.py").write_text("before\n", encoding="utf-8")
    (frozen / "unexpected.txt").write_text("unexpected\n", encoding="utf-8")
    try:
        snapshot.verify(frozen, payload)
    except ValueError as error:
        assert "verification failed" in str(error)
    else:
        raise AssertionError("unexpected snapshot file was accepted")


def test_snapshot_rejects_escaping_symlink(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "escape").symlink_to("/etc/passwd")
    monkeypatch.setattr(snapshot, "_tracked_log_paths", lambda _root: set())

    try:
        snapshot.inventory(root)
    except ValueError as error:
        assert "escapes approved roots" in str(error)
    else:
        raise AssertionError("escaping symlink was accepted")
