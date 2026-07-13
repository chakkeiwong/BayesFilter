#!/usr/bin/env python3
"""Independent A2 boundary, artifact, ledger, and closure verifier."""

from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
import math
import os
import re
import struct
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PHASE_DIR = Path(
    "docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a2"
)
PLAN_PATH = Path(
    "docs/plans/bayesfilter-ssl-lstm-completion-phase-a2-terminal-state-forecast-api-subplan-2026-07-11.md"
)
RESULT_PATH = Path(
    "docs/plans/bayesfilter-ssl-lstm-completion-phase-a2-terminal-state-forecast-api-result-2026-07-11.md"
)
IMPLEMENTATION_REVIEW_PATH = Path(
    "docs/reviews/bayesfilter-ssl-lstm-completion-phase-a2-implementation-codex-substitute-review-2026-07-12.md"
)
RESULT_REVIEW_PATH = Path(
    "docs/reviews/bayesfilter-ssl-lstm-completion-phase-a2-result-codex-substitute-review-2026-07-12.md"
)
A3_PLAN_PATH = Path(
    "docs/plans/bayesfilter-ssl-lstm-completion-phase-a3-forecast-oracle-statistics-subplan-2026-07-11.md"
)
A3_REVIEW_PATH = Path(
    "docs/reviews/bayesfilter-ssl-lstm-completion-phase-a3-subplan-codex-substitute-review-2026-07-12.md"
)

HEAD_SHA256 = "a644d29c5c2fd09a0deb3a7b5212799ff1fcb163"
A2_CONTRACT_SIGNATURE = "8719aa65943dcc9e4b0499debfff8ec13a96d4cec12dc48d70a8922920058804"
A2_CONFIG_SIGNATURE = "ecb5a2cedac5f059da3bd3feee51a1065eb66aeff5aeb8dc0dd3b4e3a6926150"
POINT_MATRIX_SHA256 = "d6ba48e5a64897f87caeece4de776c139d8fc62d00fc118d89b4d88da468829a"

NONCLAIMS = [
    "A2 terminal-state and forecast engineering evidence only",
    "predictive law is conditional on the approximate historical SVD-UKF",
    "not posterior correctness or exact nonlinear filtering evidence",
    "not HMC or NeuTra readiness evidence",
    "not predictive equivalence, calibration, or model adequacy evidence",
    "not performance, product, public API, default, or release evidence",
    "not a sampler ranking or scientific claim",
]

ACCEPTED_A1 = {
    "bayesfilter/nonlinear/ssl_lstm_posterior_tf.py": "6dfd00a55f072a5e8fd3b1690c92ca6572cd895525cc915deaebec09ef6f3667",
    "docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a1/cpu-reference.json": "b6dc26637d584dbf6d62575a999af5cf43bb7bab35a5cf9eb6984d1cfaf6a068",
    "docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a1/gpu-xla-canary.json": "1538032c6e0c9ea664ea92ce9ea334c92c916c13831fd08af69865435c822f6e",
    "docs/plans/bayesfilter-ssl-lstm-completion-phase-a1-masked-posterior-target-result-2026-07-11.md": "78f269a53fb0536017d32bd12c2b36967cd013a85dcb1102936ed79ae95e34b5",
    "docs/reviews/bayesfilter-ssl-lstm-completion-phase-a1-result-codex-substitute-review-2026-07-11.md": "a3a7ca4f6396f352fe29e7da24e164ae4cb0d1144ea492b4b7270fe4f3f0decf",
}

LAZY_EXPORT_A1_SHA256 = "9bfbe2a912b6465e8338d61c48c51b91b2b30d1f11912a543772e5901998de68"
SUBPLAN_SHA256 = "6b6b9799782be3304ecbd2dee465c52285688b5e2d1b3087d911ccad1279bbb0"
SUBPLAN_REVIEW_SHA256 = "846574f1d6140efd5ff8e10f772f0d886be916585f30ccdac6960bd1eacfeaa1"
EXECUTION_LEDGER_A1_CLOSE_SHA256 = "fe951c4af438cf9f013550806e062b7ad1d83351d25523e154c5776f1be60931"

SUBPLAN_REVIEWS = tuple(
    Path(
        f"docs/reviews/bayesfilter-ssl-lstm-completion-phase-a2-subplan-codex-substitute-review-round{round_number}-2026-07-12.md"
    )
    for round_number in range(1, 7)
)

SOURCE_PATHS = (
    Path("bayesfilter/nonlinear/ssl_lstm_predictive_tf.py"),
    Path("bayesfilter/nonlinear/__init__.py"),
    Path("tests/test_ssl_lstm_predictive_tf.py"),
    Path("docs/benchmarks/benchmark_ssl_lstm_completion_phase_a2_terminal_forecast_2026_07_12.py"),
    Path("docs/benchmarks/verify_ssl_lstm_completion_phase_a2_terminal_forecast_2026_07_12.py"),
)

STRUCTURED_PATHS = tuple(
    PHASE_DIR / name
    for name in (
        "pre-run-boundary.json",
        "boundary-generation-write-trace.log",
        "executor-write-ledger.json",
        "executor-ledger-generation-write-trace.log",
        "post-result-write-ledger.json",
        "post-result-ledger-generation-write-trace.log",
        "focused-tests-write-trace.log",
        "innovation-bank.json",
        "innovation-bank.log",
        "cpu-generation-write-trace.log",
        "cpu-reference.json",
        "cpu-reference.log",
        "cpu-verification-write-trace.log",
        "cpu-reference-verify.log",
        "gpu-generation-write-trace.log",
        "gpu-xla-canary.json",
        "gpu-xla-canary.log",
        "gpu-verification-write-trace.log",
        "gpu-xla-canary-verify.log",
        "final-checkpoint.json",
        "final-checkpoint-generation-write-trace.log",
        "closure-generation-write-trace.log",
        "post-result-closure.json",
        "closure-verification-write-trace.log",
        "post-result-closure-verify.log",
    )
)

GOVERNANCE_PATHS = (
    RESULT_PATH,
    IMPLEMENTATION_REVIEW_PATH,
    RESULT_REVIEW_PATH,
    A3_PLAN_PATH,
    A3_REVIEW_PATH,
    Path("docs/plans/bayesfilter-ssl-lstm-completion-visible-execution-ledger-2026-07-11.md"),
    Path("docs/plans/bayesfilter-ssl-lstm-completion-approval-boundary-ledger-2026-07-11.md"),
    Path("docs/plans/bayesfilter-ssl-lstm-completion-visible-gated-execution-runbook-2026-07-11.md"),
    Path("docs/plans/bayesfilter-ssl-lstm-completion-visible-stop-handoff-2026-07-11.md"),
)

LITERAL_A2_PATHS = tuple(
    sorted(
        set(SOURCE_PATHS)
        | set(STRUCTURED_PATHS)
        | set(GOVERNANCE_PATHS)
        | {PLAN_PATH}
        | set(SUBPLAN_REVIEWS),
        key=lambda path: path.as_posix(),
    )
)

EXECUTION_LEDGER_PATH = Path(
    "docs/plans/bayesfilter-ssl-lstm-completion-visible-execution-ledger-2026-07-11.md"
)
DIRECT_EDIT_BASELINES: dict[Path, str | None] = {
    Path("bayesfilter/nonlinear/ssl_lstm_predictive_tf.py"): None,
    Path("bayesfilter/nonlinear/__init__.py"): LAZY_EXPORT_A1_SHA256,
    Path("tests/test_ssl_lstm_predictive_tf.py"): None,
    Path("docs/benchmarks/benchmark_ssl_lstm_completion_phase_a2_terminal_forecast_2026_07_12.py"): None,
    Path("docs/benchmarks/verify_ssl_lstm_completion_phase_a2_terminal_forecast_2026_07_12.py"): None,
    EXECUTION_LEDGER_PATH: EXECUTION_LEDGER_A1_CLOSE_SHA256,
    RESULT_PATH: None,
}

POINT_NAMES = [
    "truth_free",
    "phase2s_center",
    "shell_0_minus",
    "shell_0_plus",
    "shell_1_minus",
    "shell_1_plus",
    "shell_2_minus",
    "shell_2_plus",
    "shell_3_minus",
    "shell_3_plus",
]
POINTS_HEX = [
    ["0x1.6666666666666p-2", "-0x1.47ae147ae147bp-4", "0x1.4cccccccccccdp-1", "0x1.999999999999ap-5"],
    ["0x1.2410a2e2543f1p-1", "-0x1.fcd3132f8ba11p-4", "0x1.52631979a086cp-1", "0x1.1557ab4d560a3p-3"],
    ["0x1.ee87ac2b0ee48p-2", "-0x1.fcd3132f8ba11p-4", "0x1.52631979a086cp-1", "0x1.1557ab4d560a3p-3"],
    ["0x1.50dd6faf210bep-1", "-0x1.fcd3132f8ba11p-4", "0x1.52631979a086cp-1", "0x1.1557ab4d560a3p-3"],
    ["0x1.2410a2e2543f1p-1", "-0x1.b19cbccaf903cp-3", "0x1.52631979a086cp-1", "0x1.1557ab4d560a3p-3"],
    ["0x1.2410a2e2543f1p-1", "-0x1.2cd959924a756p-5", "0x1.52631979a086cp-1", "0x1.1557ab4d560a3p-3"],
    ["0x1.2410a2e2543f1p-1", "-0x1.fcd3132f8ba11p-4", "0x1.25964cacd3b9fp-1", "0x1.1557ab4d560a3p-3"],
    ["0x1.2410a2e2543f1p-1", "-0x1.fcd3132f8ba11p-4", "0x1.7f2fe6466d539p-1", "0x1.1557ab4d560a3p-3"],
    ["0x1.2410a2e2543f1p-1", "-0x1.fcd3132f8ba11p-4", "0x1.52631979a086cp-1", "0x1.8891e0688b5c0p-5"],
    ["0x1.2410a2e2543f1p-1", "-0x1.fcd3132f8ba11p-4", "0x1.52631979a086cp-1", "0x1.c88ade80893d6p-3"],
]

CHECK_NAMES = (
    "a1_entry_hashes",
    "bank_hashes",
    "batch_parity",
    "compiler_hlo",
    "covariance_validity",
    "device_placement",
    "eager_xla_parity",
    "filter_parity",
    "forecast_replay",
    "no_cache_writes",
    "observation_timing",
    "process_noise_placement",
    "status_admission",
    "total_target_parity",
    "write_boundary",
)


class ContractError(RuntimeError):
    pass


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256(path: Path) -> str:
    return _sha256_bytes((ROOT / path).read_bytes())


def _canonical_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _signature(payload: dict[str, Any]) -> str:
    projected = copy.deepcopy(payload)
    projected.pop("evidence_signature", None)
    projected.pop("created_at_utc", None)
    manifest = projected.get("run_manifest")
    if isinstance(manifest, dict):
        manifest.pop("started_at_utc", None)
        manifest.pop("completed_at_utc", None)
        manifest.pop("wall_time_seconds", None)
    return _sha256_bytes(_canonical_bytes(projected))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    absolute = ROOT / path
    absolute.parent.mkdir(parents=True, exist_ok=True)
    absolute.write_bytes(_canonical_bytes(payload) + b"\n")


def _load_json(path: Path) -> dict[str, Any]:
    def pairs_hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ContractError(f"duplicate JSON key {key!r} in {path}")
            result[key] = value
        return result

    def reject_constant(value: str) -> None:
        raise ContractError(f"nonfinite JSON constant {value!r} in {path}")

    payload = json.loads(
        (ROOT / path).read_text(encoding="utf-8"),
        object_pairs_hook=pairs_hook,
        parse_constant=reject_constant,
    )
    if not isinstance(payload, dict):
        raise ContractError(f"{path} must contain a JSON object")
    return payload


def _git(*args: str) -> str:
    environment = dict(os.environ)
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    completed = subprocess.run(
        ["git", *args],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    )
    return completed.stdout


def _git_head() -> str:
    return _git("rev-parse", "HEAD").strip()


def _tracked(path: Path) -> bool:
    environment = dict(os.environ)
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    completed = subprocess.run(
        ["git", "ls-files", "--error-unmatch", path.as_posix()],
        cwd=ROOT,
        capture_output=True,
        text=True,
        env=environment,
    )
    return completed.returncode == 0


def _file_row(path: Path, role: str) -> dict[str, Any]:
    absolute = ROOT / path
    exists = absolute.is_file()
    return {
        "path": path.as_posix(),
        "exists": exists,
        "tracked": _tracked(path),
        "sha256": _sha256(path) if exists else None,
        "role": role,
    }


def _binding_row(path: Path, role: str) -> dict[str, Any]:
    if not (ROOT / path).is_file():
        raise ContractError(f"missing binding path {path}")
    evidence_signature = None
    if path.suffix == ".json":
        payload = _load_json(path)
        evidence_signature = payload.get("evidence_signature")
        if not isinstance(evidence_signature, str):
            raise ContractError(f"JSON binding lacks evidence_signature: {path}")
    return {
        "path": path.as_posix(),
        "file_sha256": _sha256(path),
        "evidence_signature": evidence_signature,
        "role": role,
    }


def _dirty_paths() -> list[Path]:
    paths: set[Path] = set()
    for line in _git("status", "--porcelain=v1", "-uall").splitlines():
        raw = line[3:]
        if " -> " in raw:
            raw = raw.split(" -> ", 1)[1]
        paths.add(Path(raw))
    return sorted(paths, key=lambda path: path.as_posix())


def _cache_paths() -> list[str]:
    result = []
    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT)
        if path.name == ".pytest_cache" or "__pycache__" in relative.parts or path.suffix == ".pyc":
            result.append(relative.as_posix())
    return sorted(result)


def _a2_named_cache_paths(paths: list[str]) -> list[str]:
    needles = (
        "ssl_lstm_predictive_tf",
        "test_ssl_lstm_predictive_tf",
        "benchmark_ssl_lstm_completion_phase_a2_terminal_forecast",
        "verify_ssl_lstm_completion_phase_a2_terminal_forecast",
    )
    return sorted(path for path in paths if any(needle in path for needle in needles))


def _suppression_environment_verified() -> bool:
    return (
        os.environ.get("PYTHONDONTWRITEBYTECODE") == "1"
        and os.environ.get("PYTHONPYCACHEPREFIX") == "/tmp/bayesfilter-a2-pycache"
        and os.environ.get("TMPDIR") == "/tmp/bayesfilter-a2-tmp"
        and os.environ.get("CUDA_CACHE_PATH") == "/tmp/bayesfilter-a2-tmp/cuda-cache"
        and os.environ.get("XLA_FLAGS")
        == "--xla_gpu_cuda_data_dir=/usr/local/cuda --xla_dump_to=/tmp/bayesfilter-a2-tmp/xla"
    )


def _accepted_a1_rows() -> list[dict[str, Any]]:
    rows = []
    for raw_path, expected in sorted(ACCEPTED_A1.items()):
        path = Path(raw_path)
        row = _file_row(path, "accepted_a1_protected")
        if row["sha256"] != expected:
            raise ContractError(f"accepted A1 hash mismatch: {path}")
        rows.append(row)
    lazy = Path("bayesfilter/nonlinear/__init__.py")
    _verify_lazy_export_additions(lazy)
    rows.append(_file_row(lazy, "accepted_a1_lazy_exports_plus_exact_a2_additions"))
    return rows


def _verify_lazy_export_additions(path: Path) -> None:
    text = (ROOT / path).read_text(encoding="utf-8")
    export_names = (
        "SSLLSTMForecastConfig",
        "SSLLSTMForecastPaths",
        "SSLLSTMForecastProvenance",
        "SSLLSTMInnovationBank",
        "SSLLSTMTerminalState",
    )
    stripped = text
    for name in export_names:
        declaration = f'    "{name}",\n'
        mapping = (
            f'    "{name}": "bayesfilter.nonlinear.ssl_lstm_predictive_tf",\n'
        )
        if stripped.count(declaration) != 1 or stripped.count(mapping) != 1:
            raise ContractError(f"lazy export insertion is not exact for {name}")
        stripped = stripped.replace(declaration, "", 1).replace(mapping, "", 1)
    if _sha256_bytes(stripped.encode("utf-8")) != LAZY_EXPORT_A1_SHA256:
        raise ContractError("lazy exports differ outside the five A2 insertions")


def _verify_subplan_gate() -> None:
    if _sha256(PLAN_PATH) != SUBPLAN_SHA256:
        raise ContractError("A2 subplan hash does not match agreed review")
    review = SUBPLAN_REVIEWS[-1]
    if _sha256(review) != SUBPLAN_REVIEW_SHA256:
        raise ContractError("A2 Round 6 review hash mismatch")
    text = (ROOT / review).read_text(encoding="utf-8")
    if "VERDICT: AGREE" not in text or SUBPLAN_SHA256 not in text:
        raise ContractError("A2 Round 6 review does not bind an agreeing verdict")


def freeze_boundary(output: Path) -> None:
    if _git_head() != HEAD_SHA256:
        raise ContractError("HEAD drifted from the accepted A1 anchor")
    _verify_subplan_gate()
    opening_cache = _cache_paths()
    named = _a2_named_cache_paths(opening_cache)
    if named:
        raise ContractError(f"A2-named repository caches exist: {named}")
    a2_set = set(LITERAL_A2_PATHS)
    outside_rows = [
        _file_row(path, "outside_a2_dirty_explanatory")
        for path in _dirty_paths()
        if path not in a2_set
    ]
    closing_cache = _cache_paths()
    cache_scan = {
        "opening_cache_paths": opening_cache,
        "closing_cache_paths": closing_cache,
        "a2_named_cache_paths": _a2_named_cache_paths(closing_cache),
        "suppression_environment_verified": _suppression_environment_verified(),
        "passed": False,
    }
    cache_scan["passed"] = (
        not cache_scan["a2_named_cache_paths"]
        and cache_scan["suppression_environment_verified"]
    )
    if not cache_scan["passed"]:
        raise ContractError("A2 cache-suppression boundary failed")
    excluded_while_open = {
        output,
        PHASE_DIR / "boundary-generation-write-trace.log",
    }
    payload = {
        "schema_version": "bayesfilter.ssl_lstm_completion.phase_a2_scoped_boundary.v1",
        "status": "A2_SCOPED_BOUNDARY_FROZEN",
        "created_at_utc": _utc_now(),
        "git_commit": _git_head(),
        "accepted_a1_bindings": _accepted_a1_rows(),
        "literal_a2_rows": [
            _file_row(path, "literal_a2_write_set")
            for path in LITERAL_A2_PATHS
            if path not in excluded_while_open
        ],
        "outside_dirty_rows": outside_rows,
        "cache_scan": cache_scan,
        "evidence_signature": "",
        "nonclaims": NONCLAIMS,
    }
    payload["evidence_signature"] = _signature(payload)
    _write_json(output, payload)


def _check_exact_keys(value: dict[str, Any], keys: set[str], context: str) -> None:
    if set(value) != keys:
        raise ContractError(
            f"{context} keys differ: missing={sorted(keys - set(value))}, extra={sorted(set(value) - keys)}"
        )


def _verify_evidence_signature(payload: dict[str, Any], path: Path) -> None:
    actual = payload.get("evidence_signature")
    expected = _signature(payload)
    if actual != expected:
        raise ContractError(f"evidence signature mismatch: {path}")


def _verify_tensor_row(row: dict[str, Any], expected_name: str) -> bytes:
    _check_exact_keys(
        row,
        {"name", "dtype", "shape", "values_hex", "raw_little_endian_sha256"},
        f"tensor row {expected_name}",
    )
    if row["name"] != expected_name or row["dtype"] != "float64":
        raise ContractError(f"invalid tensor identity for {expected_name}")
    if not isinstance(row["shape"], list) or not all(
        isinstance(value, int) and value >= 0 for value in row["shape"]
    ):
        raise ContractError(f"invalid tensor shape for {expected_name}")
    size = math.prod(row["shape"])
    if not isinstance(row["values_hex"], list) or len(row["values_hex"]) != size:
        raise ContractError(f"tensor size mismatch for {expected_name}")
    values = []
    for encoded in row["values_hex"]:
        if not isinstance(encoded, str):
            raise ContractError(f"non-string tensor value in {expected_name}")
        value = float.fromhex(encoded)
        if not math.isfinite(value):
            raise ContractError(f"nonfinite tensor value in {expected_name}")
        values.append(value)
    raw = b"".join(struct.pack("<d", value) for value in values)
    if _sha256_bytes(raw) != row["raw_little_endian_sha256"]:
        raise ContractError(f"raw tensor hash mismatch for {expected_name}")
    return raw


def verify_bank(path: Path) -> dict[str, Any]:
    payload = _load_json(path)
    expected_keys = {
        "schema_version",
        "status",
        "created_at_utc",
        "tensorflow_version",
        "algorithm",
        "root_seed",
        "role",
        "role_code",
        "arm_id",
        "family_codes",
        "derived_seeds",
        "draw_count",
        "replication_count",
        "horizon",
        "tensors",
        "evidence_signature",
        "nonclaims",
    }
    _check_exact_keys(payload, expected_keys, "innovation bank")
    if payload["schema_version"] != "bayesfilter.ssl_lstm_completion.phase_a2_innovation_bank.v1":
        raise ContractError("innovation bank schema mismatch")
    if payload["status"] != "A2_INNOVATION_BANK_FROZEN":
        raise ContractError("innovation bank status mismatch")
    expected_scalars = {
        "algorithm": "philox",
        "root_seed": [20260712, 1202],
        "role": "paired_diagnostic_shared",
        "role_code": 101,
        "arm_id": 0,
        "family_codes": {"terminal": 1001, "process": 1002, "observation": 1003},
        "draw_count": 2,
        "replication_count": 2,
        "horizon": 10,
        "nonclaims": NONCLAIMS,
    }
    for key, expected in expected_scalars.items():
        if payload[key] != expected:
            raise ContractError(f"innovation bank field mismatch: {key}")
    if set(payload["derived_seeds"]) != {
        "role",
        "arm",
        "terminal",
        "process",
        "observation",
    }:
        raise ContractError("innovation derived seed schema mismatch")
    for seed in payload["derived_seeds"].values():
        if not isinstance(seed, list) or len(seed) != 2 or not all(
            isinstance(value, int) and -(2**31) <= value <= 2**31 - 1
            for value in seed
        ):
            raise ContractError("invalid derived seed")
    expected_tensors = {
        "terminal": [2, 2, 3],
        "process": [2, 2, 10, 1],
        "observation": [2, 2, 10, 1],
    }
    if [row.get("name") for row in payload["tensors"]] != list(expected_tensors):
        raise ContractError("innovation tensor ordering mismatch")
    for row in payload["tensors"]:
        _verify_tensor_row(row, row["name"])
        if row["shape"] != expected_tensors[row["name"]]:
            raise ContractError(f"innovation tensor shape mismatch: {row['name']}")
    _verify_evidence_signature(payload, path)
    return payload


def _bank_to_tensors(payload: dict[str, Any]):
    import tensorflow as tf

    tensors = {}
    for row in payload["tensors"]:
        values = [float.fromhex(value) for value in row["values_hex"]]
        tensors[row["name"]] = tf.reshape(
            tf.constant(values, tf.float64), row["shape"]
        )
    return tensors


def _runtime_replay(artifact: dict[str, Any]) -> tuple[Any, Any, Any]:
    import tensorflow as tf
    import bayesfilter.nonlinear.ssl_lstm_predictive_tf as predictive

    bank_path = Path(artifact["bank_binding"]["path"])
    bank_payload = verify_bank(bank_path)
    tensors = _bank_to_tensors(bank_payload)
    config = predictive.SSLLSTMForecastConfig()
    config.assert_evidence_config()
    point_values = [[float.fromhex(value) for value in row] for row in POINTS_HEX]
    points = tf.constant(point_values, tf.float64)
    terminal = predictive.extract_ssl_lstm_terminal_states(points, config)
    provisional = predictive.SSLLSTMInnovationBank(
        terminal_standard_normal=tensors["terminal"],
        process_standard_normal=tensors["process"],
        observation_standard_normal=tensors["observation"],
        root_seed=tf.constant(bank_payload["root_seed"], tf.int32),
        algorithm=bank_payload["algorithm"],
        role=bank_payload["role"],
        role_code=bank_payload["role_code"],
        arm_id=bank_payload["arm_id"],
        derived_seeds=tuple(
            tf.constant(bank_payload["derived_seeds"][name], tf.int32)
            for name in ("role", "arm", "terminal", "process", "observation")
        ),
        content_signature="",
    )
    bank = dataclasses.replace(
        provisional,
        content_signature=predictive._innovation_bank_signature(provisional),
    )
    predictive._validate_innovation_bank(
        bank,
        draw_count=bank_payload["draw_count"],
        config=config,
    )
    role = artifact["provenance"]["runtime"]["execution_role"]
    trust = artifact["provenance"]["runtime"]["trust_basis"]
    paths = predictive.forecast_ssl_lstm_paths(
        points[:2],
        bank,
        config,
        runtime_execution_role=role,
        trust_basis=trust,
    )
    return points, terminal, paths


def _tensor_rows_from_paths(paths: Any) -> list[tuple[str, Any]]:
    return [
        ("terminal_states", paths.terminal_states),
        ("states", paths.states),
        ("deterministic_transition_means", paths.deterministic_transition_means),
        ("process_innovations", paths.process_innovations),
        ("observation_means", paths.observation_means),
        ("observation_innovations", paths.observation_innovations),
        ("observations", paths.observations),
    ]


def verify_runtime_artifact(path: Path, log_path: Path | None = None) -> dict[str, Any]:
    payload = _load_json(path)
    expected_keys = {
        "schema_version",
        "artifact_role",
        "status",
        "created_at_utc",
        "run_manifest",
        "entry_bindings",
        "source_files",
        "frozen_design",
        "bank_binding",
        "cpu_reference_binding",
        "terminal_results",
        "forecast_tensors",
        "compiler_evidence",
        "provenance",
        "contract_checks",
        "evidence_signature",
        "nonclaims",
    }
    _check_exact_keys(payload, expected_keys, "runtime artifact")
    is_gpu = payload["artifact_role"] == "phase_a2_trusted_gpu_xla_canary"
    expected_schema = (
        "bayesfilter.ssl_lstm_completion.phase_a2_gpu_xla_canary.v1"
        if is_gpu
        else "bayesfilter.ssl_lstm_completion.phase_a2_cpu_reference.v1"
    )
    expected_status = "GPU_XLA_CANARY_PASSED" if is_gpu else "CPU_REFERENCE_CONTRACT_PASSED"
    if payload["schema_version"] != expected_schema or payload["status"] != expected_status:
        raise ContractError("runtime artifact identity/status mismatch")
    if payload["nonclaims"] != NONCLAIMS:
        raise ContractError("runtime artifact nonclaims mismatch")
    _verify_evidence_signature(payload, path)
    if _sha256(Path(payload["bank_binding"]["path"])) != payload["bank_binding"]["file_sha256"]:
        raise ContractError("runtime bank file crosslink mismatch")
    checks = payload["contract_checks"]
    expected_checks = set(CHECK_NAMES) | (
        {"cpu_gpu_parity", "cpu_reference_crosslink"} if is_gpu else set()
    )
    if [row.get("name") for row in checks] != sorted(expected_checks):
        raise ContractError("contract check ordering/membership mismatch")
    if any(not row.get("passed") for row in checks):
        raise ContractError("runtime artifact contains a failed contract check")
    if len(payload["terminal_results"]) != 10 or any(
        row.get("status") != 0 or not row.get("passed")
        for row in payload["terminal_results"]
    ):
        raise ContractError("terminal result admission failed")
    points, terminal, paths = _runtime_replay(payload)
    if _sha256_bytes(
        b"".join(
            struct.pack("<d", float(value))
            for value in __import__("tensorflow").unstack(
                __import__("tensorflow").reshape(points, [-1])
            )
        )
    ) != POINT_MATRIX_SHA256:
        raise ContractError("frozen point matrix hash mismatch")
    if any(int(value) != 0 for value in __import__("tensorflow").unstack(terminal.status)):
        raise ContractError("fresh terminal replay returned nonzero status")
    stored_rows = payload["forecast_tensors"]
    expected_names = [name for name, _tensor in _tensor_rows_from_paths(paths)]
    if [row.get("name") for row in stored_rows] != expected_names:
        raise ContractError("forecast tensor ordering mismatch")
    for row, (name, tensor) in zip(stored_rows, _tensor_rows_from_paths(paths), strict=True):
        raw = _verify_tensor_row(row, name)
        actual = b"".join(
            struct.pack("<d", float(value))
            for value in __import__("tensorflow").unstack(
                __import__("tensorflow").reshape(tensor, [-1])
            )
        )
        if raw != actual:
            threshold = 4096 * 2.0**-52 * max(
                1.0,
                max(abs(float.fromhex(value)) for value in row["values_hex"]),
                float(__import__("tensorflow").reduce_max(__import__("tensorflow").abs(tensor))),
            )
            stored = __import__("tensorflow").reshape(
                __import__("tensorflow").constant(
                    [float.fromhex(value) for value in row["values_hex"]],
                    __import__("tensorflow").float64,
                ),
                row["shape"],
            )
            residual = float(
                __import__("tensorflow").reduce_max(
                    __import__("tensorflow").abs(stored - tensor)
                )
            )
            if residual > threshold:
                raise ContractError(f"forecast replay mismatch for {name}")
    for row in payload["compiler_evidence"]:
        hlo = row.get("hlo_text")
        if not isinstance(hlo, str) or "ENTRY" not in hlo:
            raise ContractError("stored HLO evidence is invalid")
        encoded = hlo.encode("utf-8")
        if len(encoded) != row.get("hlo_byte_count") or _sha256_bytes(encoded) != row.get("hlo_sha256"):
            raise ContractError("stored HLO hash/length mismatch")
        if row.get("concrete_trace_count") != 1:
            raise ContractError("compiled trace count is not one")
    if log_path is not None:
        absolute = ROOT / log_path
        absolute.parent.mkdir(parents=True, exist_ok=True)
        absolute.write_text(
            json.dumps(
                {"status": "A2_RUNTIME_ARTIFACT_VERIFIED", "artifact": path.as_posix()},
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
    return payload


_STRACE_CALL = re.compile(
    r"^\s*(?:(?P<pid>\d+)\s+)?(?P<name>[A-Za-z_][A-Za-z0-9_]*)\((?P<args>.*)\)\s+=\s+(?P<result>.*)$"
)
_STRACE_MUTATING_SYSCALLS = frozenset(
    {
        "chmod",
        "chown",
        "copy_file_range",
        "creat",
        "fallocate",
        "fchmod",
        "fchmodat",
        "fchmodat2",
        "fchown",
        "fchownat",
        "fremovexattr",
        "fsetxattr",
        "ftruncate",
        "futimesat",
        "lchown",
        "link",
        "linkat",
        "lremovexattr",
        "lsetxattr",
        "mkdir",
        "mkdirat",
        "mknod",
        "mknodat",
        "removexattr",
        "rename",
        "renameat",
        "renameat2",
        "rmdir",
        "sendfile",
        "sendfile64",
        "setxattr",
        "symlink",
        "symlinkat",
        "truncate",
        "unlink",
        "unlinkat",
        "utime",
        "utimensat",
        "utimes",
    }
)
_STRACE_WRITE_FLAGS = frozenset(
    {"O_WRONLY", "O_RDWR", "O_CREAT", "O_TRUNC", "O_APPEND", "O_TMPFILE"}
)


def _split_strace_arguments(arguments: str) -> list[str]:
    result: list[str] = []
    start = 0
    quote = False
    escaped = False
    depth = 0
    for index, character in enumerate(arguments):
        if quote:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                quote = False
            continue
        if character == '"':
            quote = True
        elif character in "[{(":
            depth += 1
        elif character in "]})":
            depth = max(0, depth - 1)
        elif character == "," and depth == 0:
            result.append(arguments[start:index].strip())
            start = index + 1
    result.append(arguments[start:].strip())
    return result


def _strace_call(line: str) -> tuple[str, str, list[str], str] | None:
    match = _STRACE_CALL.match(line)
    if match is None:
        return None
    return (
        match.group("pid") or "root",
        match.group("name"),
        _split_strace_arguments(match.group("args")),
        match.group("result"),
    )


def _is_strace_mutation(name: str, arguments: list[str]) -> bool:
    if name in _STRACE_MUTATING_SYSCALLS:
        return True
    if name == "open" and len(arguments) >= 2:
        flags = arguments[1]
    elif name in {"openat", "openat2"} and len(arguments) >= 3:
        flags = arguments[2]
    else:
        return False
    return any(flag in flags for flag in _STRACE_WRITE_FLAGS)


def _parse_strace_mutations(path: Path) -> list[str]:
    if not (ROOT / path).is_file():
        raise ContractError(f"missing strace file {path}")
    mutations: list[str] = []
    for line in (ROOT / path).read_text(encoding="utf-8", errors="replace").splitlines():
        call = _strace_call(line)
        if call is not None and _is_strace_mutation(call[1], call[2]):
            mutations.append(line)
    return mutations


def _component_contained(path: Path, roots: tuple[Path, ...]) -> bool:
    normalized = path.resolve(strict=False)
    return any(normalized == root or root in normalized.parents for root in roots)


def audit_terminal_trace(path: Path) -> None:
    lines = (ROOT / path).read_text(encoding="utf-8", errors="replace").splitlines()
    if not lines:
        raise ContractError("terminal trace is empty")
    if any(
        "<unfinished ...>" in line
        or ("<... " in line and " resumed>" in line)
        for line in lines
    ):
        raise ContractError("terminal trace contains unfinished/resumed syscall")
    calls = []
    for line in lines:
        call = _strace_call(line)
        if call is None:
            raise ContractError(f"unparsed terminal trace line: {line}")
        if re.search(r'"(?:[^"\\]|\\.)*"\.\.\.', line) or re.search(
            r"\d+</[^>]*\.\.\.[^>]*>", line
        ):
            raise ContractError(f"truncated terminal trace path: {line}")
        calls.append((line, call))
    pids = {call[0] for _, call in calls}
    if "root" in pids or len(pids) != 1:
        raise ContractError(f"terminal trace must contain exactly one explicit PID: {sorted(pids)}")
    allowed_roots = tuple(
        candidate.resolve(strict=False)
        for candidate in (
            ROOT / PHASE_DIR,
            Path("/tmp/bayesfilter-a2-pycache"),
            Path("/tmp/bayesfilter-a2-tmp"),
        )
    )
    mutation_count = 0
    for line, call in calls:
        _pid, name, arguments, result = call
        if not _is_strace_mutation(name, arguments):
            continue
        if result.startswith("-1 "):
            continue
        if name not in {"open", "openat", "openat2"}:
            raise ContractError(f"terminal trace contains forbidden mutation syscall: {line}")
        destination = re.fullmatch(r"\d+<(/[^>]*)>", result)
        if destination is None:
            raise ContractError(
                f"write open lacks -yy resolved destination annotation: {line}"
            )
        if not _component_contained(Path(destination.group(1)), allowed_roots):
            raise ContractError(f"terminal trace contains disallowed mutation: {line}")
        mutation_count += 1
    if mutation_count == 0:
        raise ContractError("terminal trace contains no verified mutation")
    result = {
        "status": "A2_TERMINAL_WRITE_TRACE_AUDIT_PASSED",
        "trace_sha256": _sha256(path),
    }
    sys.stdout.write(_canonical_bytes(result).decode("utf-8") + "\n")


def _direct_edit_rows(before_boundary: bool) -> list[dict[str, Any]]:
    candidates = (
        list(SOURCE_PATHS) + [EXECUTION_LEDGER_PATH, RESULT_PATH]
        if before_boundary
        else list(GOVERNANCE_PATHS)
    )
    rows = []
    existing = [path for path in candidates if (ROOT / path).is_file()]
    for sequence, path in enumerate(existing, start=1):
        rows.append(
            {
                "sequence": sequence,
                "path": path.as_posix(),
                "before_sha256": DIRECT_EDIT_BASELINES.get(path),
                "after_sha256": _sha256(path),
                "reason": (
                    "A2 reviewed implementation/result artifact"
                    if before_boundary
                    else "A2 post-result review and handoff closure"
                ),
            }
        )
    return rows


def write_executor_ledger(output: Path) -> None:
    traces = [
        PHASE_DIR / "boundary-generation-write-trace.log",
        PHASE_DIR / "focused-tests-write-trace.log",
        PHASE_DIR / "cpu-generation-write-trace.log",
        PHASE_DIR / "cpu-verification-write-trace.log",
        PHASE_DIR / "gpu-generation-write-trace.log",
        PHASE_DIR / "gpu-verification-write-trace.log",
    ]
    payload = {
        "schema_version": "bayesfilter.ssl_lstm_completion.phase_a2_executor_write_ledger.v1",
        "status": "A2_EXECUTOR_WRITE_LEDGER_VALID",
        "created_at_utc": _utc_now(),
        "rows": _direct_edit_rows(True),
        "strace_bindings": [_binding_row(path, "closed_a2_subprocess_trace") for path in traces],
        "evidence_signature": "",
        "nonclaims": NONCLAIMS,
    }
    payload["evidence_signature"] = _signature(payload)
    _write_json(output, payload)


def write_post_result_ledger(output: Path) -> None:
    payload = {
        "schema_version": "bayesfilter.ssl_lstm_completion.phase_a2_post_result_write_ledger.v1",
        "status": "A2_POST_RESULT_WRITE_LEDGER_VALID",
        "created_at_utc": _utc_now(),
        "rows": _direct_edit_rows(False),
        "evidence_signature": "",
        "nonclaims": NONCLAIMS,
    }
    payload["evidence_signature"] = _signature(payload)
    _write_json(output, payload)


def _checkpoint_members() -> list[Path]:
    return [
        PLAN_PATH,
        *SUBPLAN_REVIEWS,
        *SOURCE_PATHS,
        PHASE_DIR / "pre-run-boundary.json",
        PHASE_DIR / "boundary-generation-write-trace.log",
        PHASE_DIR / "executor-write-ledger.json",
        PHASE_DIR / "executor-ledger-generation-write-trace.log",
        PHASE_DIR / "focused-tests-write-trace.log",
        PHASE_DIR / "innovation-bank.json",
        PHASE_DIR / "innovation-bank.log",
        PHASE_DIR / "cpu-generation-write-trace.log",
        PHASE_DIR / "cpu-reference.json",
        PHASE_DIR / "cpu-reference.log",
        PHASE_DIR / "cpu-verification-write-trace.log",
        PHASE_DIR / "cpu-reference-verify.log",
        PHASE_DIR / "gpu-generation-write-trace.log",
        PHASE_DIR / "gpu-xla-canary.json",
        PHASE_DIR / "gpu-xla-canary.log",
        PHASE_DIR / "gpu-verification-write-trace.log",
        PHASE_DIR / "gpu-xla-canary-verify.log",
        IMPLEMENTATION_REVIEW_PATH,
        RESULT_PATH,
    ]


def write_final_checkpoint(output: Path) -> None:
    if _git_head() != HEAD_SHA256:
        raise ContractError("HEAD drift before final checkpoint")
    opening_cache = _cache_paths()
    named = _a2_named_cache_paths(opening_cache)
    if named:
        raise ContractError(f"A2-named cache exists: {named}")
    payload = {
        "schema_version": "bayesfilter.ssl_lstm_completion.phase_a2_final_checkpoint.v1",
        "status": "A2_PRE_RESULT_CHECKPOINT_PASSED",
        "created_at_utc": _utc_now(),
        "git_commit": _git_head(),
        "member_rows": [_binding_row(path, "a2_pre_result_member") for path in _checkpoint_members()],
        "accepted_a1_bindings": _accepted_a1_rows(),
        "outside_dirty_rows": [
            _file_row(path, "outside_a2_dirty_explanatory")
            for path in _dirty_paths()
            if path not in set(LITERAL_A2_PATHS)
        ],
        "cache_scan": {
            "opening_cache_paths": opening_cache,
            "closing_cache_paths": _cache_paths(),
            "a2_named_cache_paths": named,
            "suppression_environment_verified": _suppression_environment_verified(),
            "passed": not named and _suppression_environment_verified(),
        },
        "evidence_signature": "",
        "nonclaims": NONCLAIMS,
    }
    if not payload["cache_scan"]["passed"]:
        raise ContractError("checkpoint cache scan failed")
    payload["evidence_signature"] = _signature(payload)
    _write_json(output, payload)


def close_phase(output: Path) -> None:
    checkpoint = PHASE_DIR / "final-checkpoint.json"
    members = [
        checkpoint,
        PHASE_DIR / "final-checkpoint-generation-write-trace.log",
        RESULT_PATH,
        RESULT_REVIEW_PATH,
        A3_PLAN_PATH,
        A3_REVIEW_PATH,
        PHASE_DIR / "post-result-write-ledger.json",
        PHASE_DIR / "post-result-ledger-generation-write-trace.log",
        Path("docs/plans/bayesfilter-ssl-lstm-completion-visible-execution-ledger-2026-07-11.md"),
        Path("docs/plans/bayesfilter-ssl-lstm-completion-approval-boundary-ledger-2026-07-11.md"),
        Path("docs/plans/bayesfilter-ssl-lstm-completion-visible-gated-execution-runbook-2026-07-11.md"),
        Path("docs/plans/bayesfilter-ssl-lstm-completion-visible-stop-handoff-2026-07-11.md"),
    ]
    payload = {
        "schema_version": "bayesfilter.ssl_lstm_completion.phase_a2_post_result_closure.v1",
        "status": "A2_POST_RESULT_CLOSURE_PASSED",
        "created_at_utc": _utc_now(),
        "git_commit": _git_head(),
        "member_rows": [_binding_row(path, "a2_post_result_member") for path in members],
        "final_checkpoint_binding": _binding_row(checkpoint, "immutable_final_checkpoint"),
        "accepted_a1_bindings": _accepted_a1_rows(),
        "outside_dirty_rows": [
            _file_row(path, "outside_a2_dirty_explanatory")
            for path in _dirty_paths()
            if path not in set(LITERAL_A2_PATHS)
        ],
        "cache_scan": {
            "opening_cache_paths": _cache_paths(),
            "closing_cache_paths": _cache_paths(),
            "a2_named_cache_paths": _a2_named_cache_paths(_cache_paths()),
            "suppression_environment_verified": _suppression_environment_verified(),
            "passed": not _a2_named_cache_paths(_cache_paths()) and _suppression_environment_verified(),
        },
        "evidence_signature": "",
        "nonclaims": NONCLAIMS,
    }
    if not payload["cache_scan"]["passed"]:
        raise ContractError("closure cache scan failed")
    payload["evidence_signature"] = _signature(payload)
    _write_json(output, payload)


def verify_closure(path: Path, log_path: Path | None) -> None:
    payload = _load_json(path)
    if payload.get("schema_version") != "bayesfilter.ssl_lstm_completion.phase_a2_post_result_closure.v1" or payload.get("status") != "A2_POST_RESULT_CLOSURE_PASSED":
        raise ContractError("closure identity/status mismatch")
    _verify_evidence_signature(payload, path)
    for row in payload["member_rows"]:
        member = Path(row["path"])
        if _sha256(member) != row["file_sha256"]:
            raise ContractError(f"closure member hash mismatch: {member}")
    if log_path is not None:
        (ROOT / log_path).write_text(
            json.dumps({"status": "A2_POST_RESULT_CLOSURE_VERIFIED"}, sort_keys=True) + "\n",
            encoding="utf-8",
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--freeze-boundary", action="store_true")
    group.add_argument("--artifact", type=Path)
    group.add_argument("--write-executor-ledger", action="store_true")
    group.add_argument("--write-post-result-ledger", action="store_true")
    group.add_argument("--write-final-checkpoint", action="store_true")
    group.add_argument("--close-phase", action="store_true")
    group.add_argument("--verify-closure", type=Path)
    group.add_argument("--audit-terminal-trace", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--log-path", type=Path)
    args = parser.parse_args()

    os.chdir(ROOT)
    if args.freeze_boundary:
        if args.output is None:
            parser.error("--freeze-boundary requires --output")
        freeze_boundary(args.output)
    elif args.artifact is not None:
        verify_runtime_artifact(args.artifact, args.log_path)
    elif args.write_executor_ledger:
        if args.output is None:
            parser.error("--write-executor-ledger requires --output")
        write_executor_ledger(args.output)
    elif args.write_post_result_ledger:
        if args.output is None:
            parser.error("--write-post-result-ledger requires --output")
        write_post_result_ledger(args.output)
    elif args.write_final_checkpoint:
        if args.output is None:
            parser.error("--write-final-checkpoint requires --output")
        write_final_checkpoint(args.output)
    elif args.close_phase:
        if args.output is None:
            parser.error("--close-phase requires --output")
        close_phase(args.output)
    elif args.verify_closure is not None:
        verify_closure(args.verify_closure, args.log_path)
    elif args.audit_terminal_trace is not None:
        audit_terminal_trace(args.audit_terminal_trace)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ContractError as exc:
        print(f"A2_CONTRACT_ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
