from __future__ import annotations

import json
import os
import subprocess
from collections.abc import Iterator
from pathlib import Path

import pytest

from scripts import kalman_qr_benchmark_contract as contract


ROOT = Path(__file__).resolve().parents[1]
PYTHON = "/home/ubuntu/anaconda3/envs/tfgpu/bin/python"
SCRIPT = "scripts/benchmark_kalman_qr_parameter_count_scaling.py"
OUTPUT = Path("/tmp/kalman_qr_phase6_cpu_xla_gateb_r3/import_discovery.json")
EXACT_ENVIRONMENT = {
    "CUDA_VISIBLE_DEVICES": "-1",
    "OMP_NUM_THREADS": "1",
    "TF_NUM_INTRAOP_THREADS": "1",
    "TF_NUM_INTEROP_THREADS": "1",
}
EXACT_ARGUMENTS = [
    "--phase6-import-discovery",
    "--device",
    "cpu",
    "--cpu-threads",
    "1",
    "--output-json",
    str(OUTPUT),
]


@pytest.fixture
def preserved_output() -> Iterator[Path]:
    parent_existed = OUTPUT.parent.exists()
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    existed = OUTPUT.exists()
    previous = OUTPUT.read_bytes() if existed else None
    previous_mode = OUTPUT.stat().st_mode if existed else None
    try:
        yield OUTPUT
    finally:
        if existed:
            assert previous is not None and previous_mode is not None
            OUTPUT.write_bytes(previous)
            OUTPUT.chmod(previous_mode)
        else:
            OUTPUT.unlink(missing_ok=True)
        if not parent_existed:
            OUTPUT.parent.rmdir()


def _environment(**changes: str | None) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(EXACT_ENVIRONMENT)
    for name, value in changes.items():
        if value is None:
            environment.pop(name, None)
        else:
            environment[name] = value
    return environment


def _run(
    arguments: list[str],
    *,
    environment: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [PYTHON, SCRIPT, *arguments],
        cwd=ROOT,
        env=environment or _environment(),
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_phase6_import_discovery_exact_child_emits_closed_import_manifest(
    preserved_output: Path,
) -> None:
    completed = _run(EXACT_ARGUMENTS)

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(preserved_output.read_text(encoding="utf-8"))
    assert set(payload) == {
        "schema",
        "kind",
        "command_argv",
        "environment",
        "fixture_constructed",
        "trace_requested",
        "selected_method_constructed",
        "concrete_function_invocations",
        "manifest",
        "nonclaims",
    }
    assert payload["schema"] == (
        "bayesfilter.kalman_qr_batched_xla_repair.phase6.import_discovery.v1"
    )
    assert payload["kind"] == "import_only_no_fixture_trace_or_execution"
    assert payload["command_argv"] == [PYTHON, SCRIPT, *EXACT_ARGUMENTS]
    assert payload["environment"] == EXACT_ENVIRONMENT
    assert payload["fixture_constructed"] is False
    assert payload["trace_requested"] is False
    assert payload["selected_method_constructed"] is False
    assert payload["concrete_function_invocations"] == 0
    assert payload["nonclaims"] == list(contract.PHASE6_NONCLAIMS)

    manifest = payload["manifest"]
    assert set(manifest) == {
        "schema",
        "repository_root",
        "entries",
        "manifest_sha256",
    }
    assert manifest["schema"] == contract.PHASE6_DEPENDENCY_SCHEMA
    assert manifest["repository_root"] == str(ROOT.resolve())
    assert manifest["manifest_sha256"] == contract.canonical_sha256(
        manifest["entries"]
    )
    paths = {entry["path"] for entry in manifest["entries"]}
    assert set(contract.PHASE6_REQUIRED_SOURCE_PATHS).issubset(paths)
    for entry in manifest["entries"]:
        assert set(entry) == {"module", "path", "sha256"}
        assert entry["sha256"] == contract.file_sha256(ROOT / entry["path"])


@pytest.mark.parametrize(
    ("arguments", "environment_changes", "wrong_output"),
    [
        (
            [*EXACT_ARGUMENTS, "--dimensions", "11"],
            {},
            None,
        ),
        (
            [*EXACT_ARGUMENTS, "--phase6-trace-only"],
            {},
            None,
        ),
        (
            [
                "--phase6-import-discovery",
                "--device",
                "auto",
                *EXACT_ARGUMENTS[3:],
            ],
            {},
            None,
        ),
        (
            EXACT_ARGUMENTS,
            {},
            True,
        ),
        (
            EXACT_ARGUMENTS,
            {"OMP_NUM_THREADS": "2"},
            None,
        ),
        (
            EXACT_ARGUMENTS,
            {"TF_NUM_INTEROP_THREADS": None},
            None,
        ),
    ],
    ids=[
        "meaningful-nondefault",
        "extra-mode",
        "wrong-device",
        "wrong-output",
        "wrong-environment-value",
        "missing-environment-value",
    ],
)
def test_phase6_import_discovery_rejects_nonexact_authority_before_writing(
    preserved_output: Path,
    tmp_path: Path,
    arguments: list[str],
    environment_changes: dict[str, str | None],
    wrong_output: bool | None,
) -> None:
    sentinel = b"preexisting-phase6-import-discovery\n"
    preserved_output.write_bytes(sentinel)
    unexpected_output = tmp_path / "wrong_import_discovery.json"
    invocation = list(arguments)
    if wrong_output:
        invocation[-1] = str(unexpected_output)

    completed = _run(invocation, environment=_environment(**environment_changes))

    assert completed.returncode != 0
    assert "Phase 6 import discovery rejected before import" in completed.stderr
    assert preserved_output.read_bytes() == sentinel
    if wrong_output:
        assert not unexpected_output.exists()
