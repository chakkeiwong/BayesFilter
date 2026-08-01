from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
WORKER = REPO / "scripts/complete_highdim_leaderboard_claude_audit_worker.sh"


def _workspace(tmp_path: Path, fake_claude: str) -> tuple[Path, dict[str, str]]:
    workspace = tmp_path / "workspace"
    bin_dir = tmp_path / "bin"
    workspace.mkdir()
    bin_dir.mkdir()
    settings = tmp_path / "settings.json"
    settings.write_text("{}\n", encoding="utf-8")
    claude = bin_dir / "claude"
    claude.write_text(fake_claude, encoding="utf-8")
    claude.chmod(0o755)
    environment = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "CLAUDE_WORKER_SETTINGS": str(settings),
    }
    return workspace, environment


def _metadata(workspace: Path) -> dict:
    paths = list(
        (workspace / ".complete_highdim_claude_audit").glob("*-metadata.json")
    )
    assert len(paths) == 1
    return json.loads(paths[0].read_text(encoding="utf-8"))


def test_audited_worker_preserves_read_only_tool_events(tmp_path: Path) -> None:
    workspace, environment = _workspace(
        tmp_path,
        """#!/usr/bin/env bash
set -eu
printf '%s\n' '{"type":"assistant","message":{"content":[{"type":"tool_use","id":"t1","name":"Read","input":{}}]}}'
printf '%s\n' '{"type":"result","result":"VERDICT: AGREE"}'
""",
    )

    result = subprocess.run(
        [str(WORKER), "--cwd", str(workspace), "review one path"],
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=10,
    )

    assert result.returncode == 0
    assert result.stdout.strip() == "VERDICT: AGREE"
    metadata = _metadata(workspace)
    assert metadata["stream_parse_complete"] is True
    assert metadata["disallowed_tool_uses"] == []


def test_audited_worker_rejects_non_object_json_event(tmp_path: Path) -> None:
    workspace, environment = _workspace(
        tmp_path,
        """#!/usr/bin/env bash
set -eu
printf '%s\n' '[{"type":"tool_use","id":"hidden","name":"Bash"}]'
printf '%s\n' '{"type":"result","result":"VERDICT: AGREE"}'
""",
    )

    result = subprocess.run(
        [str(WORKER), "--cwd", str(workspace), "review one path"],
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
        timeout=10,
    )

    assert result.returncode == 43
    metadata = _metadata(workspace)
    assert metadata["stream_parse_complete"] is False
    assert metadata["invalid_stream_line_count"] == 1


def test_audited_worker_writes_metadata_after_term_signal(tmp_path: Path) -> None:
    workspace, environment = _workspace(
        tmp_path,
        """#!/usr/bin/env bash
set -eu
trap 'exit 143' TERM
sleep 30
""",
    )
    process = subprocess.Popen(
        [str(WORKER), "--cwd", str(workspace), "review one path"],
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    deadline = time.monotonic() + 5
    while not (workspace / ".complete_highdim_claude_audit").is_dir():
        assert time.monotonic() < deadline
        time.sleep(0.02)
    os.kill(process.pid, signal.SIGTERM)
    process.communicate(timeout=5)

    assert process.returncode == 143
    metadata = _metadata(workspace)
    assert metadata["metadata_generated_after_worker_exit_or_signal"] is True
    assert metadata["claude_exit_code"] == 143
