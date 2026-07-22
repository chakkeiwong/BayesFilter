from __future__ import annotations

import json
import tarfile
import subprocess
from pathlib import Path

from scripts import export_complete_highdim_leaderboard_isolated_changes as exporter


def test_isolated_export_contains_only_changes_since_snapshot(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    output = tmp_path / "logs"
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    (root / "kept.txt").write_text("before\n", encoding="utf-8")
    (root / "changed.txt").write_text("before\n", encoding="utf-8")
    (root / "deleted.txt").write_text("before\n", encoding="utf-8")
    (root / "docs/plans/logs").mkdir(parents=True)
    (root / "docs/plans/logs/ignored.log").write_text("old\n", encoding="utf-8")

    baseline_path = output / "run-baseline.json"
    exporter._write_json(baseline_path, exporter.snapshot(root))  # noqa: SLF001
    (root / "changed.txt").write_text("after\n", encoding="utf-8")
    (root / "added.txt").write_text("new\n", encoding="utf-8")
    (root / "deleted.txt").unlink()
    (root / "docs/plans/logs/ignored.log").write_text("new\n", encoding="utf-8")

    manifest = exporter.export_changes(root, baseline_path, output, "run")

    assert manifest["added"] == ["added.txt"]
    assert manifest["modified"] == ["changed.txt"]
    assert manifest["deleted"] == ["deleted.txt"]
    assert manifest["automatic_merge_performed"] is False
    with tarfile.open(output / "run-isolated-changed-files.tar.gz", "r:gz") as archive:
        assert sorted(archive.getnames()) == ["added.txt", "changed.txt"]
    hashes = json.loads((output / "run-export-sha256.json").read_text(encoding="utf-8"))
    assert hashes["files"]


def test_isolated_export_rejects_baseline_from_another_root(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    other = tmp_path / "other"
    output = tmp_path / "logs"
    root.mkdir()
    other.mkdir()
    output.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    baseline_path = output / "run-baseline.json"
    exporter._write_json(  # noqa: SLF001
        baseline_path,
        exporter.snapshot(root, root_identity=other),
    )

    try:
        exporter.export_changes(root, baseline_path, output, "run")
    except ValueError as error:
        assert "different workspace" in str(error)
    else:
        raise AssertionError("cross-workspace baseline was accepted")


def test_isolated_export_rejects_existing_output(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    output = tmp_path / "logs"
    root.mkdir()
    output.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    baseline_path = output / "run-baseline.json"
    exporter._write_json(baseline_path, exporter.snapshot(root))  # noqa: SLF001
    (output / "run-isolated-change-manifest.json").write_text(
        "do not overwrite\n", encoding="utf-8"
    )

    try:
        exporter.export_changes(root, baseline_path, output, "run")
    except FileExistsError as error:
        assert "refusing to overwrite" in str(error)
    else:
        raise AssertionError("existing export output was overwritten")


def test_export_hashes_exclude_declared_live_files(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    output = tmp_path / "logs"
    root.mkdir()
    output.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    baseline_path = output / "run-baseline.json"
    exporter._write_json(baseline_path, exporter.snapshot(root))  # noqa: SLF001
    live = output / "run-watchdog.log"
    live.write_text("still open\n", encoding="utf-8")

    exporter.export_changes(
        root,
        baseline_path,
        output,
        "run",
        hash_excluded_names=[live.name],
    )

    hashes = json.loads((output / "run-export-sha256.json").read_text())
    assert live.name in hashes["hash_excluded_live_or_control_files"]
    assert live not in {Path(entry["path"]) for entry in hashes["files"]}
