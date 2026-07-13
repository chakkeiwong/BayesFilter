#!/usr/bin/env python3
"""Independent verifier and closure helper for Phase A3 oracle artifacts."""

from __future__ import annotations

import argparse
import ast
import copy
import dataclasses
import hashlib
import importlib
import importlib.metadata
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

PHASE_DIR = Path("docs/plans/artifacts/ssl-lstm-completion-2026-07-11/phase-a3")
PLAN_PATH = Path(
    "docs/plans/bayesfilter-ssl-lstm-completion-phase-a3-forecast-oracle-statistics-subplan-2026-07-11.md"
)
RESULT_PATH = Path(
    "docs/plans/bayesfilter-ssl-lstm-completion-phase-a3-forecast-oracle-statistics-result-2026-07-11.md"
)
BOUNDARY_PATH = PHASE_DIR / "pre-run-boundary.json"
FIXTURE_PATH = PHASE_DIR / "fixture-contract.json"
HARNESS_ANCHOR_PATH = PHASE_DIR / "harness-review-anchor.json"
HARNESS_REVIEW_PATH = Path(
    "docs/reviews/bayesfilter-ssl-lstm-completion-phase-a3-harness-codex-substitute-review-2026-07-13.md"
)
CPU_REFERENCE_PATH = PHASE_DIR / "oracle-cpu-reference.json"
CPU_VERIFY_RECEIPT_PATH = PHASE_DIR / "oracle-cpu-reference-verify.log"
CPU_GENERATION_TRACE_PATH = PHASE_DIR / "oracle-cpu-generation-write-trace.log"
CPU_VERIFICATION_TRACE_PATH = PHASE_DIR / "oracle-cpu-verification-write-trace.log"
FOCUSED_TESTS_TRACE_PATH = PHASE_DIR / "focused-tests-write-trace.log"
GPU_GENERATION_TRACE_PATH = PHASE_DIR / "oracle-gpu-generation-write-trace.log"
GPU_VERIFICATION_TRACE_PATH = PHASE_DIR / "oracle-gpu-verification-write-trace.log"
EXECUTOR_LEDGER_TRACE_PATH = PHASE_DIR / "executor-ledger-generation-write-trace.log"
FINAL_CHECKPOINT_TRACE_PATH = PHASE_DIR / "final-checkpoint-generation-write-trace.log"
POST_RESULT_LEDGER_TRACE_PATH = PHASE_DIR / "post-result-ledger-generation-write-trace.log"
CLOSURE_GENERATION_TRACE_PATH = PHASE_DIR / "closure-generation-write-trace.log"
CLOSURE_VERIFICATION_TRACE_PATH = PHASE_DIR / "closure-verification-write-trace.log"
POST_RESULT_CLOSURE_PATH = PHASE_DIR / "post-result-closure.json"
POST_RESULT_CLOSURE_RECEIPT_PATH = PHASE_DIR / "post-result-closure-verify.log"
PREDICTIVE_SOURCE = Path("bayesfilter/inference/predictive_equivalence.py")
ORACLE_SOURCE = Path("bayesfilter/testing/scalar_lgssm_forecast_oracle.py")
PREDICTIVE_TEST = Path("tests/test_predictive_equivalence.py")
ORACLE_TEST = Path("tests/test_scalar_lgssm_forecast_oracle.py")
GENERATOR_PATH = Path(
    "docs/benchmarks/benchmark_ssl_lstm_completion_phase_a3_forecast_oracle_2026_07_13.py"
)
VERIFIER_PATH = Path(
    "docs/benchmarks/verify_ssl_lstm_completion_phase_a3_forecast_oracle_2026_07_13.py"
)
IMPLEMENTATION_REVIEW_PATH = Path(
    "docs/reviews/bayesfilter-ssl-lstm-completion-phase-a3-implementation-codex-substitute-review-2026-07-13.md"
)
RESULT_REVIEW_PATH = Path(
    "docs/reviews/bayesfilter-ssl-lstm-completion-phase-a3-result-codex-substitute-review-2026-07-13.md"
)
A4_PLAN_PATH = Path(
    "docs/plans/bayesfilter-ssl-lstm-completion-phase-a4-calibration-design-freeze-subplan-2026-07-11.md"
)
A4_REVIEW_PATH = Path(
    "docs/reviews/bayesfilter-ssl-lstm-completion-phase-a4-subplan-codex-substitute-review-2026-07-13.md"
)

HEAD_SHA256 = "a644d29c5c2fd09a0deb3a7b5212799ff1fcb163"
PLAN_SHA256 = "67ee503a15f5e7a81ca2a37e52cc6b60264c1cff89ff5cff1a9fddd3187161c4"
EXPECTED_SOURCE_HASHES = {
    PREDICTIVE_SOURCE: "99ddaa1dcb15e9f3ec7a5a18f96ebd0f656848c40ea76c896b387cace294bc16",
    ORACLE_SOURCE: "74889d699e3575ee163c64d9a67325f0376e161106e9b36fb6b61453c3a5eb43",
    PREDICTIVE_TEST: "5e6a137c12b3131c8ff7471d74abd4a877777ef6432a2c51f5c62cceedf9290d",
    ORACLE_TEST: "977134cbc92b63ca6d8dab7a1e6ca25eb58137cb27430518a1aacc120cecfab8",
}

CPU_SCHEMA = "bayesfilter.ssl_lstm_completion.phase_a3_cpu_oracle.v1"
GPU_SCHEMA = "bayesfilter.ssl_lstm_completion.phase_a3_gpu_xla_oracle.v1"
CPU_STATUS = "A3_CPU_ORACLE_PASSED"
GPU_STATUS = "A3_GPU_XLA_ORACLE_PASSED"
CPU_GPU_TOLERANCE_MULTIPLIER = 8192
FAMILY_CODES = {
    "terminal_standard_normal": 3101,
    "process_standard_normal": 3102,
    "observation_standard_normal": 3103,
}

NONCLAIMS = [
    "A3 scalar-LGSSM oracle and predictive-statistics engineering evidence only",
    "not SSL-LSTM predictive equivalence or calibration evidence",
    "not posterior correctness or parameter agreement evidence",
    "not HMC or NeuTra validity, readiness, training, or comparison evidence",
    "not calibrated A4 margins, bandwidths, blocks, bootstrap counts, or seeds",
    "not performance, product, public API, default, or release evidence",
    "not a sampler ranking, model-adequacy result, or scientific claim",
]

CHECK_NAMES = (
    "analytic_formula_exact",
    "analytic_covariance_valid",
    "direct_simulation_replay",
    "monte_carlo_oracle_agreement",
    "summary_statistics",
    "standardization",
    "quadratic_mmd_roles",
    "signed_u_form_preserved",
    "common_random_numbers_excluded",
    "cross_chain_schedule",
    "cross_chain_inference_admission",
    "cross_chain_null_coverage",
    "hierarchical_indices",
    "joint_alpha_allocation",
    "simultaneous_intervals",
    "controlled_alternatives",
    "decision_fail_closed",
    "fixture_binding",
    "source_binding",
    "compiler_hlo",
    "device_placement",
)

GOVERNANCE_PATHS = (
    Path("docs/plans/bayesfilter-ssl-lstm-completion-visible-execution-ledger-2026-07-11.md"),
    Path("docs/plans/bayesfilter-ssl-lstm-completion-approval-boundary-ledger-2026-07-11.md"),
    Path("docs/plans/bayesfilter-ssl-lstm-completion-visible-gated-execution-runbook-2026-07-11.md"),
    Path("docs/plans/bayesfilter-ssl-lstm-completion-visible-stop-handoff-2026-07-11.md"),
)


class ContractError(RuntimeError):
    """Raised on any fail-closed A3 verification error."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256(path: Path) -> str:
    return _sha256_bytes((ROOT / path).read_bytes())


def _signature(payload: dict[str, Any]) -> str:
    projection = copy.deepcopy(payload)
    projection.pop("evidence_signature", None)
    projection.pop("created_at_utc", None)
    manifest = projection.get("run_manifest")
    if isinstance(manifest, dict):
        for field in ("started_at_utc", "completed_at_utc", "wall_time_seconds"):
            manifest.pop(field, None)
    return _sha256_bytes(_canonical_bytes(projection))


def _load_json(path: Path) -> dict[str, Any]:
    def pairs_hook(rows: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in rows:
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
        raise ContractError(f"{path} must contain an object")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    absolute = ROOT / path
    absolute.parent.mkdir(parents=True, exist_ok=True)
    absolute.write_bytes(_canonical_bytes(payload) + b"\n")


def _git(*arguments: str) -> str:
    environment = dict(os.environ)
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    return subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        env=environment,
    ).stdout


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not_installed"


def _hex_float(value: str) -> float:
    result = float.fromhex(value)
    if not math.isfinite(result):
        raise ContractError(f"nonfinite fixture constant {value!r}")
    return result


def _tensor_row(name: str, value: Any) -> dict[str, Any]:
    import tensorflow as tf

    tensor = tf.convert_to_tensor(value, dtype=tf.float64)
    values = [float(item) for item in tf.reshape(tensor, [-1])]
    if not all(math.isfinite(item) for item in values):
        raise ContractError(f"nonfinite tensor {name}")
    raw = b"".join(struct.pack("<d", item) for item in values)
    return {
        "name": name,
        "dtype": "float64",
        "shape": [int(size) for size in tensor.shape],
        "values_hex": [item.hex() for item in values],
        "raw_little_endian_sha256": _sha256_bytes(raw),
    }


def _int_tensor_row(name: str, value: Any) -> dict[str, Any]:
    import tensorflow as tf

    tensor = tf.convert_to_tensor(value, dtype=tf.int32)
    values = [int(item) for item in tf.reshape(tensor, [-1])]
    raw = b"".join(struct.pack("<i", item) for item in values)
    return {
        "name": name,
        "dtype": "int32",
        "shape": [int(size) for size in tensor.shape],
        "values": values,
        "raw_little_endian_sha256": _sha256_bytes(raw),
    }


def _bool_tensor_row(name: str, value: Any) -> dict[str, Any]:
    import tensorflow as tf

    tensor = tf.convert_to_tensor(value, dtype=tf.bool)
    values = [bool(item) for item in tf.reshape(tensor, [-1])]
    raw = bytes(int(item) for item in values)
    return {
        "name": name,
        "dtype": "bool",
        "shape": [int(size) for size in tensor.shape],
        "values": values,
        "raw_little_endian_sha256": _sha256_bytes(raw),
    }


def _verify_tensor_row(row: dict[str, Any]) -> None:
    if row.get("dtype") == "float64":
        values = [float.fromhex(item) for item in row.get("values_hex", [])]
        if not all(math.isfinite(item) for item in values):
            raise ContractError(f"nonfinite encoded tensor {row.get('name')}")
        raw = b"".join(struct.pack("<d", item) for item in values)
    elif row.get("dtype") == "int32":
        values = row.get("values", [])
        if not all(isinstance(item, int) and -(2**31) <= item < 2**31 for item in values):
            raise ContractError(f"invalid int32 tensor {row.get('name')}")
        raw = b"".join(struct.pack("<i", item) for item in values)
    elif row.get("dtype") == "bool":
        values = row.get("values", [])
        if not all(type(item) is bool for item in values):
            raise ContractError(f"invalid bool tensor {row.get('name')}")
        raw = bytes(int(item) for item in values)
    else:
        raise ContractError(f"unknown tensor dtype {row.get('dtype')}")
    count = math.prod(row.get("shape", []))
    if count != len(values):
        raise ContractError(f"tensor shape/value mismatch: {row.get('name')}")
    if _sha256_bytes(raw) != row.get("raw_little_endian_sha256"):
        raise ContractError(f"tensor content hash mismatch: {row.get('name')}")


def _decode_tensor_row(row: dict[str, Any], tf: Any) -> Any:
    _verify_tensor_row(row)
    shape = row["shape"]
    if row["dtype"] == "float64":
        values = [float.fromhex(item) for item in row["values_hex"]]
        dtype = tf.float64
    elif row["dtype"] == "int32":
        values = row["values"]
        dtype = tf.int32
    elif row["dtype"] == "bool":
        values = row["values"]
        dtype = tf.bool
    else:
        raise ContractError("unsupported tensor dtype")
    return tf.reshape(tf.constant(values, dtype=dtype), shape)


def _section(payload: dict[str, Any], name: str) -> dict[str, dict[str, Any]]:
    rows = payload.get("tensor_sections", {}).get(name)
    if not isinstance(rows, list):
        raise ContractError(f"missing tensor section {name!r}")
    result: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("name"), str):
            raise ContractError(f"malformed tensor row in {name!r}")
        _verify_tensor_row(row)
        if row["name"] in result:
            raise ContractError(f"duplicate tensor row {name}/{row['name']}")
        result[row["name"]] = row
    return result


def _tensor(payload: dict[str, Any], section: str, name: str, tf: Any) -> Any:
    rows = _section(payload, section)
    if name not in rows:
        raise ContractError(f"missing tensor row {section}/{name}")
    return _decode_tensor_row(rows[name], tf)


def _dataclass_dict(value: Any) -> dict[str, Any]:
    if dataclasses.is_dataclass(value):
        return {field.name: getattr(value, field.name) for field in dataclasses.fields(value)}
    if isinstance(value, dict):
        return value
    raise ContractError(f"expected dataclass or dict, got {type(value).__name__}")


def _status_text(value: Any) -> str:
    raw = value.numpy() if hasattr(value, "numpy") else value
    return raw.decode("ascii") if isinstance(raw, bytes) else str(raw)


def _parameters(fixture: dict[str, Any], oracle: Any, tf: Any) -> Any:
    values = fixture["lgssm"]
    return oracle.ScalarLGSSMParameters(
        transition_coefficient=tf.constant(_hex_float(values["a_hex"]), tf.float64),
        transition_offset=tf.constant(_hex_float(values["b_hex"]), tf.float64),
        observation_coefficient=tf.constant(_hex_float(values["c_hex"]), tf.float64),
        observation_offset=tf.constant(_hex_float(values["d_hex"]), tf.float64),
        terminal_mean=tf.constant(_hex_float(values["terminal_mean_hex"]), tf.float64),
        terminal_variance=tf.constant(_hex_float(values["p_terminal_hex"]), tf.float64),
        process_variance=tf.constant(_hex_float(values["process_variance_q_hex"]), tf.float64),
        observation_variance=tf.constant(_hex_float(values["observation_variance_r_hex"]), tf.float64),
    )


def _source_rows() -> list[dict[str, Any]]:
    paths = (PREDICTIVE_SOURCE, ORACLE_SOURCE, PREDICTIVE_TEST, ORACLE_TEST, GENERATOR_PATH, VERIFIER_PATH)
    for path, expected in EXPECTED_SOURCE_HASHES.items():
        if _sha256(path) != expected:
            raise ContractError(f"frozen source hash drift: {path}")
    return [
        {"path": path.as_posix(), "sha256": _sha256(path), "role": "a3_source_or_test", "exists": True}
        for path in paths
    ]


def _semantic_contract_sha256(path: Path, excluded_fields: tuple[str, ...]) -> str:
    projection = copy.deepcopy(_load_json(path))
    for field in excluded_fields:
        projection.pop(field, None)
    return _sha256_bytes(_canonical_bytes(projection))


def _verify_harness_anchor() -> str:
    anchor = _load_json(HARNESS_ANCHOR_PATH)
    if (ROOT / HARNESS_ANCHOR_PATH).read_bytes() != _canonical_bytes(anchor) + b"\n":
        raise ContractError("harness review anchor is not canonical JSON")
    expected_keys = {
        "schema_version", "status", "created_at_utc", "review_class", "verdict",
        "reviewed_files", "boundary_semantic_sha256", "fixture_semantic_sha256",
        "review_binding", "evidence_signature", "nonclaims",
    }
    if (
        set(anchor) != expected_keys
        or anchor.get("schema_version")
        != "bayesfilter.ssl_lstm_completion.phase_a3_harness_review_anchor.v1"
        or anchor.get("status") != "A3_HARNESS_REVIEW_ANCHOR_FROZEN"
        or anchor.get("evidence_signature") != _signature(anchor)
    ):
        raise ContractError("harness review anchor identity/signature differs")
    expected_files = [
        {"path": GENERATOR_PATH.as_posix(), "sha256": _sha256(GENERATOR_PATH)},
        {"path": VERIFIER_PATH.as_posix(), "sha256": _sha256(VERIFIER_PATH)},
    ]
    if (
        anchor["review_class"] != "CODEX_SUBSTITUTE_REVIEW_WEAKER_THAN_CLAUDE"
        or anchor["verdict"] != "AGREE"
        or anchor["reviewed_files"] != expected_files
        or anchor["nonclaims"] != NONCLAIMS
    ):
        raise ContractError("harness review anchor verdict or file hashes differ")
    expected_review_binding = {
        "path": HARNESS_REVIEW_PATH.as_posix(),
        "sha256": _sha256(HARNESS_REVIEW_PATH),
    }
    if anchor["review_binding"] != expected_review_binding:
        raise ContractError("harness review-record binding differs")
    boundary_semantic_sha256 = _semantic_contract_sha256(
        BOUNDARY_PATH,
        ("created_at_utc", "evidence_signature", "harness_review_anchor_sha256"),
    )
    fixture_semantic_sha256 = _semantic_contract_sha256(
        FIXTURE_PATH,
        ("created_at_utc", "evidence_signature", "boundary_sha256"),
    )
    if (
        anchor["boundary_semantic_sha256"] != boundary_semantic_sha256
        or anchor["fixture_semantic_sha256"] != fixture_semantic_sha256
    ):
        raise ContractError("harness review anchor contract semantics differ")
    review_text = (ROOT / HARNESS_REVIEW_PATH).read_text(encoding="utf-8")
    required_review_text = (
        "CODEX_SUBSTITUTE_REVIEW",
        "explicitly weaker than Claude",
        expected_files[0]["sha256"],
        expected_files[1]["sha256"],
        boundary_semantic_sha256,
        fixture_semantic_sha256,
        "VERDICT: AGREE",
    )
    if not all(fragment in review_text for fragment in required_review_text):
        raise ContractError("harness review record does not bind the agreed exact hashes")
    return _sha256(HARNESS_ANCHOR_PATH)


def _verify_fixed_inputs() -> tuple[dict[str, Any], dict[str, Any]]:
    if _git("rev-parse", "HEAD").strip() != HEAD_SHA256:
        raise ContractError("HEAD drift")
    if _sha256(PLAN_PATH) != PLAN_SHA256:
        raise ContractError("A3 plan drift")
    boundary = _load_json(BOUNDARY_PATH)
    fixture = _load_json(FIXTURE_PATH)
    if boundary.get("evidence_signature") != _signature(boundary):
        raise ContractError("boundary signature mismatch")
    if fixture.get("evidence_signature") != _signature(fixture):
        raise ContractError("fixture signature mismatch")
    if boundary.get("harness_review_anchor_sha256") != _verify_harness_anchor():
        raise ContractError("boundary does not bind the reviewed harness anchor")
    if fixture.get("boundary_sha256") != _sha256(BOUNDARY_PATH):
        raise ContractError("fixture does not bind the current reviewed boundary")
    for row in boundary.get("a2_entry_bindings", []):
        if _sha256(Path(row["path"])) != row["sha256"]:
            raise ContractError(f"A1/A2 entry drift: {row['path']}")
    _source_rows()
    return boundary, fixture


def _bank(
    payload: dict[str, Any],
    section: str,
    arm_id: int,
    fixture_constants: dict[str, Any],
    oracle: Any,
    tf: Any,
) -> Any:
    rows = _section(payload, section)
    expected = {"terminal_standard_normal", "process_standard_normal", "observation_standard_normal"}
    if set(rows) != expected:
        raise ContractError(f"bank section {section!r} differs")
    leading_shape = [
        int(fixture_constants["chain_count_per_arm"]),
        int(fixture_constants["draw_count_per_chain"]),
        int(fixture_constants["forecast_replication_count"]),
    ]
    expected_shapes = {
        "terminal_standard_normal": leading_shape,
        "process_standard_normal": [*leading_shape, int(fixture_constants["horizon"])],
        "observation_standard_normal": [*leading_shape, int(fixture_constants["horizon"])],
    }
    for name, expected_shape in expected_shapes.items():
        if rows[name].get("dtype") != "float64" or rows[name].get("shape") != expected_shape:
            raise ContractError(f"primary bank geometry differs: {section}/{name}")
    return oracle.ScalarLGSSMInnovationBank(
        terminal_standard_normal=_decode_tensor_row(rows["terminal_standard_normal"], tf),
        process_standard_normal=_decode_tensor_row(rows["process_standard_normal"], tf),
        observation_standard_normal=_decode_tensor_row(rows["observation_standard_normal"], tf),
        root_seed=tf.constant([0, 0], tf.int32),
        arm_id=arm_id,
    )


def _bank_hashes(payload: dict[str, Any], section: str) -> dict[str, str]:
    rows = _section(payload, section)
    expected = {
        "terminal_standard_normal",
        "process_standard_normal",
        "observation_standard_normal",
    }
    if set(rows) != expected:
        raise ContractError(f"bank section {section!r} differs")
    return {
        name: rows[name]["raw_little_endian_sha256"] for name in sorted(expected)
    }


def _pairwise_domain_nonreuse(left: dict[str, str], right: dict[str, str]) -> bool:
    return set(left) == set(right) and bool(left) and all(
        left[name] != right[name] for name in left
    )


def _bank_domain_ledger(
    primary: tuple[dict[str, str], dict[str, str]],
    coverage_rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    ledger = []
    for arm_id, hashes in enumerate(primary, start=1):
        for family in sorted(hashes):
            ledger.append(
                {
                    "purpose": "primary",
                    "replication": None,
                    "arm_id": arm_id,
                    "family": family,
                    "raw_little_endian_sha256": hashes[family],
                }
            )
    for row in coverage_rows:
        for arm_id, key in ((1, "left_tensor_hashes"), (2, "right_tensor_hashes")):
            for family in sorted(row[key]):
                ledger.append(
                    {
                        "purpose": "coverage",
                        "replication": row["replication"],
                        "arm_id": arm_id,
                        "family": family,
                        "raw_little_endian_sha256": row[key][family],
                    }
                )
    return ledger


def _seed_pair(seed: Any) -> list[int]:
    return [int(item) for item in seed]


def _seed_domain_ledger(
    tf: Any, root_seed: list[int], domain_ledger: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    root = tf.constant(root_seed, tf.int32)
    rows = []
    for domain in domain_ledger:
        parent = root
        if domain["purpose"] == "coverage":
            parent = tf.random.experimental.stateless_fold_in(
                root,
                tf.constant(10000 + int(domain["replication"]), tf.int32),
                alg="philox",
            )
        arm_seed = tf.random.experimental.stateless_fold_in(
            parent, tf.constant(domain["arm_id"], tf.int32), alg="philox"
        )
        family_code = FAMILY_CODES[domain["family"]]
        family_seed = tf.random.experimental.stateless_fold_in(
            arm_seed, tf.constant(family_code, tf.int32), alg="philox"
        )
        rows.append(
            {
                **domain,
                "parent_seed": _seed_pair(parent),
                "arm_seed": _seed_pair(arm_seed),
                "family_code": family_code,
                "family_seed": _seed_pair(family_seed),
            }
        )
    return rows


def _cpu_seed_domain_attestation(
    payload: dict[str, Any],
    fixture: dict[str, Any],
    domain_ledger: list[dict[str, Any]],
    seed_ledger: list[dict[str, Any]],
    tf: Any,
) -> None:
    if len(domain_ledger) != len(seed_ledger):
        raise ContractError("seed-domain ledger length differs")
    coverage = _stacked_bank(payload, "coverage_innovation_banks", tf)
    primary = {
        1: {
            name: _tensor(payload, "innovation_bank_left", name, tf)
            for name in FAMILY_CODES
        },
        2: {
            name: _tensor(payload, "innovation_bank_right", name, tf)
            for name in FAMILY_CODES
        },
    }
    for domain, seed_row in zip(domain_ledger, seed_ledger):
        if {key: seed_row[key] for key in domain} != domain:
            raise ContractError("seed-domain row does not bind raw-hash domain")
        if domain["purpose"] == "primary":
            observed = primary[domain["arm_id"]][domain["family"]]
        else:
            observed = coverage[domain["family"]][
                int(domain["replication"]), int(domain["arm_id"]) - 1
            ]
        regenerated = tf.random.stateless_normal(
            observed.shape,
            tf.constant(seed_row["family_seed"], tf.int32),
            dtype=tf.float64,
            alg="philox",
        )
        if _tensor_row(domain["family"], regenerated) != _tensor_row(
            domain["family"], observed
        ):
            raise ContractError("CPU seed-domain regeneration differs from tensor authority")
        if (
            _tensor_row(domain["family"], observed)["raw_little_endian_sha256"]
            != domain["raw_little_endian_sha256"]
        ):
            raise ContractError("CPU seed-domain row hash differs from tensor authority")


def _stacked_bank(
    payload: dict[str, Any], section: str, tf: Any
) -> dict[str, Any]:
    rows = _section(payload, section)
    expected = {
        "terminal_standard_normal",
        "process_standard_normal",
        "observation_standard_normal",
    }
    if set(rows) != expected:
        raise ContractError(f"stacked bank section {section!r} differs")
    return {name: _decode_tensor_row(rows[name], tf) for name in sorted(expected)}


def _tensor_hashes(values: dict[str, Any]) -> dict[str, str]:
    return {
        name: _tensor_row(name, values[name])["raw_little_endian_sha256"]
        for name in sorted(values)
    }


def _indices(payload: dict[str, Any], section: str, constants: dict[str, Any], stats: Any, tf: Any) -> Any:
    rows = _section(payload, section)
    expected = {"chain_indices", "draw_indices", "forecast_replication_indices"}
    if set(rows) != expected:
        raise ContractError(f"index section {section!r} differs")
    return stats.HierarchicalBootstrapIndices(
        chain_indices=_decode_tensor_row(rows["chain_indices"], tf),
        draw_indices=_decode_tensor_row(rows["draw_indices"], tf),
        forecast_replication_indices=_decode_tensor_row(rows["forecast_replication_indices"], tf),
        block_length=int(constants["block_length"]),
        block_mode="moving",
        chain_mode="stratified_fixed_chains",
        seed=tf.constant([0, 0], tf.int32),
        status=tf.constant("VALID"),
    )


def _verify_index_geometry(indices: Any, constants: dict[str, Any], tf: Any) -> None:
    bootstrap_count = int(constants["bootstrap_count"])
    chain_count = int(constants["chain_count_per_arm"])
    draw_count = int(constants["draw_count_per_chain"])
    replication_count = int(constants["forecast_replication_count"])
    block_length = int(constants["block_length"])
    if draw_count % block_length:
        raise ContractError("moving-block geometry does not divide draw count")
    expected_shapes = {
        "chain_indices": (bootstrap_count, chain_count),
        "draw_indices": (bootstrap_count, chain_count, draw_count),
        "forecast_replication_indices": (
            bootstrap_count,
            chain_count,
            draw_count,
            replication_count,
        ),
    }
    rows = _dataclass_dict(indices)
    if (
        _status_text(indices.status) != "VALID"
        or indices.block_length != block_length
        or indices.block_mode != "moving"
        or indices.chain_mode != "stratified_fixed_chains"
    ):
        raise ContractError("hierarchical index metadata differs")
    for name, shape in expected_shapes.items():
        if tuple(rows[name].shape) != shape:
            raise ContractError(f"hierarchical index shape differs: {name}")
    expected_chains = tf.broadcast_to(
        tf.range(chain_count, dtype=tf.int32)[None, :],
        [bootstrap_count, chain_count],
    )
    if not bool(tf.reduce_all(rows["chain_indices"] == expected_chains)):
        raise ContractError("hierarchical index chain strata differ")
    block_count = draw_count // block_length
    blocks = tf.reshape(
        rows["draw_indices"],
        [bootstrap_count, chain_count, block_count, block_length],
    )
    starts = blocks[..., 0]
    offsets = tf.range(block_length, dtype=tf.int32)
    if (
        not bool(tf.reduce_all(blocks - starts[..., None] == offsets))
        or not bool(tf.reduce_all(starts >= 0))
        or not bool(tf.reduce_all(starts <= draw_count - block_length))
        or not bool(tf.reduce_all(rows["draw_indices"] >= 0))
        or not bool(tf.reduce_all(rows["draw_indices"] < draw_count))
    ):
        raise ContractError("draw indices are not non-circular contiguous moving blocks")
    replication_indices = rows["forecast_replication_indices"]
    if (
        not bool(tf.reduce_all(replication_indices >= 0))
        or not bool(tf.reduce_all(replication_indices < replication_count))
    ):
        raise ContractError("forecast-replication indices are out of bounds")


def _verify_index_seed_diagnostic(
    left: Any, right: Any, constants: dict[str, Any], stats: Any, tf: Any
) -> None:
    seeds = tf.random.experimental.stateless_split(
        tf.constant(constants["root_seed"], tf.int32), 2, alg="philox"
    )
    for arm, observed in enumerate((left, right)):
        replay = stats.hierarchical_resample_indices(
            chain_count=int(constants["chain_count_per_arm"]),
            draw_count=int(constants["draw_count_per_chain"]),
            forecast_replication_count=int(constants["forecast_replication_count"]),
            block_length=int(constants["block_length"]),
            bootstrap_count=int(constants["bootstrap_count"]),
            seed=seeds[arm],
            chain_mode="stratified_fixed_chains",
            block_mode="moving",
            jit_compile=True,
        )
        for name in (
            "chain_indices",
            "draw_indices",
            "forecast_replication_indices",
        ):
            if not bool(
                tf.reduce_all(
                    _dataclass_dict(observed)[name] == _dataclass_dict(replay)[name]
                )
            ):
                raise ContractError(f"diagnostic integer replay differs: {name}")


def _resample(tf: Any, paths: Any, indices: Any) -> Any:
    chain_indices = indices.chain_indices
    draw_indices = indices.draw_indices
    replication_indices = indices.forecast_replication_indices
    rows = []
    for bootstrap in range(int(draw_indices.shape[0])):
        chains = []
        for chain in range(int(paths.shape[0])):
            selected = tf.gather(paths[int(chain_indices[bootstrap, chain])], draw_indices[bootstrap, chain])
            draws = [
                tf.gather(selected[draw], replication_indices[bootstrap, chain, draw])
                for draw in range(int(paths.shape[1]))
            ]
            chains.append(tf.stack(draws))
        rows.append(tf.stack(chains))
    return tf.stack(rows)


def _bootstrap_features(tf: Any, left: Any, right: Any) -> Any:
    count = math.prod(int(item) for item in left.shape[1:4])
    def arm(paths: Any) -> tuple[Any, Any]:
        mean = tf.reduce_mean(paths, axis=[1, 2, 3])
        centered = paths - mean[:, None, None, None, :]
        variance = tf.reduce_sum(tf.square(centered), axis=[1, 2, 3]) / tf.constant(float(count - 1), tf.float64)
        return mean, tf.math.log(variance)
    left_mean, left_log = arm(left)
    right_mean, right_log = arm(right)
    return tf.concat([left_mean - right_mean, left_log - right_log], axis=1)


def _scale_tolerance(tf: Any, left: Any, right: Any, multiplier: int) -> float:
    scale = tf.maximum(
        tf.constant(1.0, tf.float64),
        tf.maximum(
            tf.reduce_max(tf.abs(tf.cast(left, tf.float64))),
            tf.reduce_max(tf.abs(tf.cast(right, tf.float64))),
        ),
    )
    return float(tf.constant(multiplier * 2.0**-52, tf.float64) * scale)


def _manual_summary(tf: Any, paths: Any, probabilities: Any) -> dict[str, Any]:
    horizon = int(paths.shape[-1])
    flat = tf.reshape(paths, [-1, horizon])
    count = int(flat.shape[0])
    means = tf.reduce_mean(flat, axis=0)
    centered = flat - means
    variances = tf.reduce_sum(tf.square(centered), axis=0) / tf.constant(
        float(count - 1), tf.float64
    )
    covariance = tf.matmul(centered, centered, transpose_a=True) / tf.constant(
        float(count - 1), tf.float64
    )
    moments = tf.stack(
        [tf.reduce_mean(tf.pow(centered, order), axis=0) for order in (3, 4)]
    )
    sorted_values = tf.sort(flat, axis=0)
    positions = probabilities * tf.constant(float(count - 1), tf.float64)
    lower = tf.cast(tf.floor(positions), tf.int32)
    upper = tf.cast(tf.math.ceil(positions), tf.int32)
    fraction = positions - tf.floor(positions)
    quantiles = tf.gather(sorted_values, lower) + fraction[:, None] * (
        tf.gather(sorted_values, upper) - tf.gather(sorted_values, lower)
    )
    return {
        "means": means,
        "variances": variances,
        "log_variances": tf.math.log(variances),
        "central_moments": moments,
        "quantiles": quantiles,
        "cross_horizon_covariance": covariance,
    }


def _manual_analytic(fixture: dict[str, Any], probabilities: Any, tf: Any) -> dict[str, Any]:
    values = fixture["lgssm"]
    a, b, c, d = (_hex_float(values[name]) for name in ("a_hex", "b_hex", "c_hex", "d_hex"))
    mt = _hex_float(values["terminal_mean_hex"])
    pt = _hex_float(values["p_terminal_hex"])
    q = _hex_float(values["process_variance_q_hex"])
    r = _hex_float(values["observation_variance_r_hex"])
    horizon = int(fixture["fixture_constants"]["horizon"])
    state_mean = [a**h * mt + b * sum(a**j for j in range(h)) for h in range(1, horizon + 1)]
    state_covariance = [[a ** (h + k) * pt + q * sum(a ** (h-j) * a ** (k-j) for j in range(1, min(h, k)+1)) for k in range(1, horizon+1)] for h in range(1, horizon+1)]
    state_mean = tf.constant(state_mean, tf.float64)
    state_covariance = tf.constant(state_covariance, tf.float64)
    observation_mean = c * state_mean + d
    observation_covariance = c*c*state_covariance + r*tf.eye(horizon, dtype=tf.float64)
    variance = tf.linalg.diag_part(observation_covariance)
    tfp = importlib.import_module("tensorflow_probability")
    quantiles = tfp.distributions.Normal(observation_mean[:, None], tf.sqrt(variance)[:, None]).quantile(probabilities[None, :])
    return {"state_mean": state_mean, "state_covariance": state_covariance, "observation_mean": observation_mean, "observation_covariance": observation_covariance, "observation_variance": variance, "observation_log_variance": tf.math.log(variance), "observation_third_central_moment": tf.zeros([horizon], tf.float64), "observation_fourth_central_moment": 3.0*tf.square(variance), "observation_quantiles": quantiles}


def _recompute_core(payload: dict[str, Any], fixture: dict[str, Any]) -> dict[str, Any]:
    import tensorflow as tf
    oracle = importlib.import_module("bayesfilter.testing.scalar_lgssm_forecast_oracle")
    stats = importlib.import_module("bayesfilter.inference.predictive_equivalence")
    constants = fixture["fixture_constants"]
    horizon = int(constants["horizon"])
    probabilities = tf.constant([_hex_float(item) for item in fixture["quantile_contract"]["probabilities_hex"]], tf.float64)
    parameters = _parameters(fixture, oracle, tf)
    analytic = oracle.analytic_scalar_lgssm_forecast(parameters, horizon=horizon, quantile_probabilities=probabilities, jit_compile=True)
    manual = _manual_analytic(fixture, probabilities, tf)
    banks = [
        _bank(payload, "innovation_bank_left", 1, constants, oracle, tf),
        _bank(payload, "innovation_bank_right", 2, constants, oracle, tf),
    ]
    bank_hashes = [
        _bank_hashes(payload, "innovation_bank_left"),
        _bank_hashes(payload, "innovation_bank_right"),
    ]
    primary_domain_nonreuse = _pairwise_domain_nonreuse(*bank_hashes)
    simulations = [oracle.simulate_scalar_lgssm_forecast(parameters, bank, horizon=horizon, jit_compile=True) for bank in banks]
    config = stats.PredictiveStatisticsConfig(horizon=horizon, quantile_probabilities=tuple(float(item) for item in probabilities), jit_compile=True)
    summaries = [stats.summarize_forecast_paths(item.observations, config) for item in simulations]
    bandwidths = tf.constant([_hex_float(item) for item in constants["bandwidths_hex"]], tf.float64)
    weights = tf.constant([_hex_float(item) for item in constants["mixture_weights_hex"]], tf.float64)
    standardized = [stats.standardize_forecast_paths(item.observations, analytic.observation_mean, tf.sqrt(analytic.observation_variance), scale_floor=tf.constant(2.0**-40, tf.float64), jit_compile=True, allow_floor_use=False) for item in simulations]
    mmd = stats.fixed_rbf_mmd(tf.reshape(standardized[0], [-1, horizon]), tf.reshape(standardized[1], [-1, horizon]), bandwidths=bandwidths, mixture_weights=weights, sampling_contract="iid_oracle_fixture", iid_samples_verified=True, independent_arm_banks_verified=primary_domain_nonreuse, jit_compile=True)
    schedule = tf.constant(constants["chain_pair_schedule"], tf.int32)
    cross = stats.cross_chain_linear_mmd(standardized[0], standardized[1], bandwidths=bandwidths, mixture_weights=weights, chain_pair_schedule=schedule, independent_arm_banks_verified=primary_domain_nonreuse, stationarity_verified=True, mixing_verified=True, jit_compile=True)
    mmd_alpha = tf.constant(_hex_float(constants["mmd_alpha_hex"]), tf.float64)
    mmd_interval = stats.cross_chain_mmd_upper_interval(cross, mmd_alpha=mmd_alpha, block_length=int(constants["block_length"]), jit_compile=True)
    indices_left = _indices(payload, "resampling_indices_left", constants, stats, tf)
    indices_right = _indices(payload, "resampling_indices_right", constants, stats, tf)
    _verify_index_geometry(indices_left, constants, tf)
    _verify_index_geometry(indices_right, constants, tf)
    _verify_index_seed_diagnostic(indices_left, indices_right, constants, stats, tf)
    bootstrap_left = _resample(tf, simulations[0].observations, indices_left)
    bootstrap_right = _resample(tf, simulations[1].observations, indices_right)
    bootstrap = _bootstrap_features(tf, bootstrap_left, bootstrap_right)
    base = tf.concat([summaries[0].means-summaries[1].means, summaries[0].log_variances-summaries[1].log_variances], axis=0)
    feature_alpha = tf.constant(_hex_float(constants["feature_alpha_hex"]), tf.float64)
    feature_interval = stats.simultaneous_feature_intervals(base, feature_alpha=feature_alpha, method="bootstrap_max_statistic", bootstrap_estimates=bootstrap, minimum_bootstrap_count=20, jit_compile=True)
    margins = tf.concat(
        [
            tf.fill([horizon], tf.constant(_hex_float(constants["mean_margin_hex"]), tf.float64)),
            tf.fill([horizon], tf.constant(_hex_float(constants["log_variance_margin_hex"]), tf.float64)),
        ],
        axis=0,
    )
    mmd_tolerance = tf.constant(
        _hex_float(constants["mmd_tolerance_hex"]), tf.float64
    )
    sections = {
        "analytic": [(_tensor_row(name, value) if getattr(value.dtype, "is_floating", False) else _bool_tensor_row(name, value)) for name, value in _dataclass_dict(analytic).items() if hasattr(value, "dtype") and (getattr(value.dtype, "is_floating", False) or value.dtype.name == "bool")],
        "manual_analytic": [_tensor_row(name, value) for name, value in manual.items()],
        "simulation_left": [_tensor_row("observations", simulations[0].observations)],
        "simulation_right": [_tensor_row("observations", simulations[1].observations)],
        "summary": [_tensor_row(name, value) for name, value in _dataclass_dict(summaries[0]).items() if hasattr(value, "dtype") and getattr(value.dtype, "is_floating", False)],
        "quadratic_mmd": [_tensor_row(name, value) for name, value in _dataclass_dict(mmd).items() if hasattr(value, "dtype") and getattr(value.dtype, "is_floating", False)],
        "cross_chain_mmd": [_tensor_row(name, value) for name, value in _dataclass_dict(cross).items() if hasattr(value, "dtype") and getattr(value.dtype, "is_floating", False)],
        "feature_inputs": [
            _tensor_row("base_features", base),
            _tensor_row("bootstrap_features", bootstrap),
            _tensor_row("margins", margins),
            _tensor_row("mmd_tolerance", mmd_tolerance),
        ],
        "feature_interval": [_tensor_row(name, value) for name, value in _dataclass_dict(feature_interval).items() if hasattr(value, "dtype") and getattr(value.dtype, "is_floating", False)],
        "mmd_interval": [(_int_tensor_row(name, value) if getattr(value.dtype, "is_integer", False) else _tensor_row(name, value)) for name, value in _dataclass_dict(mmd_interval).items() if hasattr(value, "dtype") and (getattr(value.dtype, "is_floating", False) or getattr(value.dtype, "is_integer", False))],
    }
    return {"tf": tf, "oracle": oracle, "stats": stats, "parameters": parameters, "analytic": analytic, "manual": manual, "banks": banks, "bank_hashes": bank_hashes, "primary_domain_nonreuse": primary_domain_nonreuse, "simulations": simulations, "summaries": summaries, "standardized": standardized, "mmd": mmd, "cross": cross, "mmd_interval": mmd_interval, "indices_left": indices_left, "indices_right": indices_right, "bootstrap_features": bootstrap, "base_features": base, "feature_interval": feature_interval, "margins": margins, "mmd_tolerance": mmd_tolerance, "sections": sections}


def _compare_rows(observed: list[dict[str, Any]], expected: list[dict[str, Any]], label: str) -> None:
    if [row["name"] for row in observed] != [row["name"] for row in expected]:
        raise ContractError(f"tensor row names differ: {label}")
    for left, right in zip(observed, expected):
        _verify_tensor_row(left)
        if left != right:
            raise ContractError(f"fresh recomputation differs: {label}/{left['name']}")


def _near_rows(observed: list[dict[str, Any]], expected: list[dict[str, Any]], label: str) -> None:
    if [row["name"] for row in observed] != [row["name"] for row in expected]:
        raise ContractError(f"tensor row names differ: {label}")
    for left, right in zip(observed, expected):
        _verify_tensor_row(left)
        _verify_tensor_row(right)
        if left["dtype"] != right["dtype"] or left["shape"] != right["shape"]:
            raise ContractError(f"tensor metadata differs: {label}/{left['name']}")
        if left["dtype"] != "float64":
            if left != right:
                raise ContractError(f"exact tensor mismatch: {label}/{left['name']}")
            continue
        observed_values = [float.fromhex(item) for item in left["values_hex"]]
        expected_values = [float.fromhex(item) for item in right["values_hex"]]
        scale = max(1.0, *(abs(item) for item in observed_values + expected_values))
        tolerance = CPU_GPU_TOLERANCE_MULTIPLIER * 2.0**-52 * scale
        if max((abs(a-b) for a, b in zip(observed_values, expected_values)), default=0.0) > tolerance:
            raise ContractError(f"scale-aware tensor mismatch: {label}/{left['name']}")


def _decision_row(label: str, decision: Any) -> dict[str, Any]:
    return {"label": label, "status": decision.status, "primary_interval_status": decision.primary_interval_status, "mmd_upper_bound_status": decision.mmd_upper_bound_status, "hard_veto_codes": list(decision.hard_veto_codes)}


def _verify_decisions(payload: dict[str, Any], core: dict[str, Any], fixture: dict[str, Any]) -> None:
    tf, stats = core["tf"], core["stats"]
    constants = fixture["fixture_constants"]
    horizon = int(constants["horizon"])
    feature_alpha = tf.constant(_hex_float(constants["feature_alpha_hex"]), tf.float64)
    mmd_alpha = tf.constant(_hex_float(constants["mmd_alpha_hex"]), tf.float64)
    total_alpha = tf.constant(_hex_float(constants["total_alpha_hex"]), tf.float64)
    margins = core["margins"]
    tolerance = core["mmd_tolerance"]
    feature_rows = _section(payload, "feature_inputs")
    for row in (
        _tensor_row("margins", margins),
        _tensor_row("mmd_tolerance", tolerance),
    ):
        if feature_rows.get(row["name"]) != row:
            raise ContractError(f"persisted decision input is not derived: {row['name']}")
    feature_interval, mmd_interval = core["feature_interval"], core["mmd_interval"]
    observed = stats.classify_predictive_evidence(feature_interval, mmd_interval, margins=margins, mmd_tolerance=tolerance, total_alpha=total_alpha, feature_alpha=feature_alpha, mmd_alpha=mmd_alpha)
    mechanics = stats.classify_predictive_evidence(feature_interval, mmd_interval, margins=margins, mmd_tolerance=tolerance, total_alpha=total_alpha, feature_alpha=feature_alpha, mmd_alpha=mmd_alpha, mechanics_only=True)
    invalid_total_alpha = total_alpha / tf.constant(2.0, tf.float64)
    invalid_alpha = stats.classify_predictive_evidence(feature_interval, mmd_interval, margins=margins, mmd_tolerance=tolerance, total_alpha=invalid_total_alpha, feature_alpha=feature_alpha, mmd_alpha=mmd_alpha)
    zero = tf.zeros([2 * horizon], tf.float64)
    narrow = margins / tf.cast(horizon**2, tf.float64)
    wide = margins
    branch_tolerance = tolerance + tf.abs(mmd_interval.upper)
    expected_branch_rows = [
        _tensor_row("zero_estimate", zero),
        _tensor_row("narrow_standard_error", narrow),
        _tensor_row("inconclusive_standard_error", wide),
        _tensor_row("branch_mmd_tolerance", branch_tolerance),
    ]
    if list(_section(payload, "decision_branch_inputs").values()) != expected_branch_rows:
        raise ContractError("persisted branch inputs differ from reviewed derivations")
    expected_derivations = {
        "margins": "concat(fill(horizon,mean_margin_hex),fill(horizon,log_variance_margin_hex))",
        "mmd_tolerance": "float.fromhex(fixture_constants.mmd_tolerance_hex)",
        "zero_estimate": "zeros([2*horizon],float64)",
        "narrow_standard_error": "margins/horizon**2",
        "inconclusive_standard_error": "margins",
        "branch_mmd_tolerance": "mmd_tolerance+abs(main_mmd_interval.upper)",
        "invalid_total_alpha": "total_alpha/2",
    }
    if payload["statistical_metadata"].get("decision_input_derivations") != expected_derivations:
        raise ContractError("decision-input derivation metadata differs")
    pass_interval = stats.simultaneous_feature_intervals(zero, feature_alpha=feature_alpha, method="bonferroni_studentized", standard_error=narrow, jit_compile=True)
    material_interval = stats.simultaneous_feature_intervals(2.0*margins, feature_alpha=feature_alpha, method="bonferroni_studentized", standard_error=narrow, jit_compile=True)
    inconclusive_interval = stats.simultaneous_feature_intervals(zero, feature_alpha=feature_alpha, method="bonferroni_studentized", standard_error=wide, jit_compile=True)
    branch = [stats.classify_predictive_evidence(item, mmd_interval, margins=margins, mmd_tolerance=branch_tolerance, total_alpha=total_alpha, feature_alpha=feature_alpha, mmd_alpha=mmd_alpha) for item in (pass_interval, material_interval, inconclusive_interval)]
    expected = [_decision_row("independent_identical_law_fixture", observed), _decision_row("mechanics_only_hard_veto", mechanics), _decision_row("invalid_joint_alpha_hard_veto", invalid_alpha), _decision_row("synthetic_pass_branch", branch[0]), _decision_row("synthetic_material_difference_branch", branch[1]), _decision_row("synthetic_inconclusive_branch", branch[2])]
    if payload["decision_rows"] != expected:
        raise ContractError("decision rows differ from authenticated recomputation")
    if [item.status for item in branch] != ["PASS", "MATERIAL_DIFFERENCE", "INCONCLUSIVE_UNDERPOWERED"] or mechanics.status != "INVALID_HARD_VETO" or invalid_alpha.status != "INVALID_HARD_VETO":
        raise ContractError("decision branch fixture failed")


def _verify_coverage(payload: dict[str, Any], core: dict[str, Any], fixture: dict[str, Any]) -> None:
    tf, stats, oracle = core["tf"], core["stats"], core["oracle"]
    constants = fixture["fixture_constants"]
    count = int(constants["coverage_replication_count"])
    chain_count = int(constants["chain_count_per_arm"])
    draw_count = int(constants["draw_count_per_chain"])
    forecast_count = int(constants["forecast_replication_count"])
    horizon = int(constants["horizon"])
    banks = _stacked_bank(payload, "coverage_innovation_banks", tf)
    expected_terminal_shape = (count, 2, chain_count, draw_count, forecast_count)
    expected_extended_shape = (*expected_terminal_shape, horizon)
    if (
        tuple(banks["terminal_standard_normal"].shape) != expected_terminal_shape
        or tuple(banks["process_standard_normal"].shape) != expected_extended_shape
        or tuple(banks["observation_standard_normal"].shape) != expected_extended_shape
    ):
        raise ContractError("coverage innovation-bank hierarchy differs")
    persisted_observations = [
        _tensor(payload, "coverage_observations", name, tf)
        for name in ("left", "right")
    ]
    expected_observation_shape = (count, chain_count, draw_count, forecast_count, horizon)
    if any(tuple(item.shape) != expected_observation_shape for item in persisted_observations):
        raise ContractError("coverage-observation hierarchy differs")
    analytic = core["analytic"]
    bandwidths = tf.constant([_hex_float(item) for item in constants["bandwidths_hex"]], tf.float64)
    weights = tf.constant([_hex_float(item) for item in constants["mixture_weights_hex"]], tf.float64)
    schedule = tf.constant(constants["chain_pair_schedule"], tf.int32)
    alpha = tf.constant(_hex_float(constants["mmd_alpha_hex"]), tf.float64)
    successes = 0
    replay_rows = [[], []]
    provenance_rows = []
    for index in range(count):
        bank_objects = [
            oracle.ScalarLGSSMInnovationBank(
                terminal_standard_normal=banks["terminal_standard_normal"][index, arm],
                process_standard_normal=banks["process_standard_normal"][index, arm],
                observation_standard_normal=banks["observation_standard_normal"][index, arm],
                root_seed=tf.constant([0, 0], tf.int32),
                arm_id=arm + 1,
            )
            for arm in range(2)
        ]
        hashes = [
            {
                name: _tensor_row(name, getattr(bank, name))[
                    "raw_little_endian_sha256"
                ]
                for name in (
                    "terminal_standard_normal",
                    "process_standard_normal",
                    "observation_standard_normal",
                )
            }
            for bank in bank_objects
        ]
        domain_separated = _pairwise_domain_nonreuse(*hashes)
        provenance_rows.append(
            {
                "replication": index,
                "left_tensor_hashes": hashes[0],
                "right_tensor_hashes": hashes[1],
                "domain_separation_nonreuse_verified": domain_separated,
            }
        )
        replayed = [
            oracle.simulate_scalar_lgssm_forecast(
                core["parameters"], bank, horizon=horizon, jit_compile=True
            ).observations
            for bank in bank_objects
        ]
        for arm in range(2):
            replay_rows[arm].append(replayed[arm])
        arms = [stats.standardize_forecast_paths(item, analytic.observation_mean, tf.sqrt(analytic.observation_variance), scale_floor=tf.constant(2.0**-40, tf.float64), jit_compile=True, allow_floor_use=False) for item in replayed]
        statistic = stats.cross_chain_linear_mmd(arms[0], arms[1], bandwidths=bandwidths, mixture_weights=weights, chain_pair_schedule=schedule, independent_arm_banks_verified=domain_separated, stationarity_verified=True, mixing_verified=True, jit_compile=True)
        interval = stats.cross_chain_mmd_upper_interval(statistic, mmd_alpha=alpha, block_length=int(constants["block_length"]), jit_compile=True)
        if not interval.inference_admissible or _status_text(interval.status) != "VALID":
            raise ContractError("coverage replicate is inadmissible")
        successes += int(float(interval.lower) <= 0.0 <= float(interval.upper))
    domain_ledger = _bank_domain_ledger(tuple(core["bank_hashes"]), provenance_rows)
    domain_hashes = [row["raw_little_endian_sha256"] for row in domain_ledger]
    global_domain_nonreuse = len(domain_hashes) == len(set(domain_hashes))
    if not global_domain_nonreuse:
        raise ContractError("innovation tensor reused across purpose/domain/family")
    seed_domain_ledger = _seed_domain_ledger(tf, constants["root_seed"], domain_ledger)
    family_seeds = [tuple(row["family_seed"]) for row in seed_domain_ledger]
    global_seed_domain_nonreuse = len(family_seeds) == len(set(family_seeds))
    if not global_seed_domain_nonreuse:
        raise ContractError("Philox family seed reused across purpose/domain/family")
    replayed_observations = [tf.stack(rows) for rows in replay_rows]
    _near_rows(
        list(_section(payload, "coverage_observations").values()),
        [
            _tensor_row("left", replayed_observations[0]),
            _tensor_row("right", replayed_observations[1]),
        ],
        "coverage observation replay",
    )
    tfp = importlib.import_module("tensorflow_probability")
    lower = 0.0 if successes == 0 else float(tfp.distributions.Beta(tf.constant(float(successes), tf.float64), tf.constant(float(count-successes+1), tf.float64)).quantile(alpha))
    nominal = 1.0 - float(alpha)
    required_lower = nominal - _hex_float(constants["coverage_slack_hex"])
    expected_uncertainty = {
        "coverage_replication_count": count,
        "coverage_interval": "exact_binomial_clopper_pearson",
        "coverage_successes": successes,
        "coverage_lower_bound": lower,
        "coverage_required_lower_bound": required_lower,
        "coverage_confidence_alpha": float(alpha),
        "coverage_slack_hex": constants["coverage_slack_hex"],
        "bootstrap_count": int(constants["bootstrap_count"]),
        "block_length": int(constants["block_length"]),
        "classification": "A3_TEST_FIXTURE_ONLY_NOT_A4_FROZEN",
    }
    uncertainty = payload["uncertainty"]
    if uncertainty != expected_uncertainty:
        raise ContractError("coverage uncertainty contract differs")
    if lower < required_lower:
        raise ContractError("coverage lower bound fails the reviewed requirement")
    provenance = payload["bank_provenance"]
    if (
        provenance.get("coverage_section") != "coverage_innovation_banks"
        or provenance.get("coverage_arm_axis") != {"axis": 1, "arm_ids": [1, 2]}
        or provenance.get("coverage_stacked_tensor_hashes") != _tensor_hashes(banks)
        or provenance.get("coverage_replication_rows") != provenance_rows
        or provenance.get("domain_separation_ledger") != domain_ledger
        or provenance.get("global_raw_tensor_hash_nonreuse") is not True
        or provenance.get("seed_domain_ledger") != seed_domain_ledger
        or provenance.get("global_family_seed_nonreuse") is not True
        or provenance.get("seed_domain_attestation_role")
        != "cpu_only_seed_domain_generation_attestation_not_cross_backend_replay_authority"
        or provenance.get("domain_separation_claim")
        != "raw_tensor_hash_global_nonreuse_only_not_probabilistic_independence"
    ):
        raise ContractError("coverage bank provenance differs from tensor authority")
    if payload["schema_version"] == CPU_SCHEMA:
        if payload.get("artifact_role") != "phase_a3_cpu_hidden_oracle_reference":
            raise ContractError("CPU seed attestation requires CPU-hidden artifact role")
        _cpu_seed_domain_attestation(
            payload, fixture, domain_ledger, seed_domain_ledger, tf
        )
    core["global_domain_nonreuse"] = global_domain_nonreuse


def _verify_compiler(payload: dict[str, Any], core: dict[str, Any], schema: str) -> None:
    oracle, parameters, tf = core["oracle"], core["parameters"], core["tf"]
    probabilities = core["analytic"].quantile_probabilities
    bank = core["banks"][0]
    programs = [("scalar_lgssm_analytic", oracle.scalar_lgssm_analytic_compiled_program(int(probabilities.shape[0])), (parameters.as_tensor(), probabilities)), ("scalar_lgssm_simulation", oracle.scalar_lgssm_simulation_compiled_program(*bank.terminal_standard_normal.shape), (parameters.as_tensor(), bank.terminal_standard_normal, bank.process_standard_normal, bank.observation_standard_normal))]
    expected_fragment = "CPU:" if schema == CPU_SCHEMA else "GPU:"
    rows = payload["compiler_evidence"]
    if len(rows) != 2:
        raise ContractError("compiler evidence row count differs")
    for row, (name, program, inputs) in zip(rows, programs):
        hlo = str(program.experimental_get_compiler_ir(*inputs)(stage="hlo"))
        outputs = program(*inputs)
        devices = sorted({str(item.device) for item in tf.nest.flatten(outputs) if hasattr(item, "device") and item.device})
        trace_count = len(program._list_all_concrete_functions_for_serialization())
        encoded = hlo.encode("utf-8")
        expected = {
            "callable_name": name,
            "hlo_text": hlo,
            "hlo_sha256": _sha256_bytes(encoded),
            "hlo_byte_count": len(encoded),
            "hlo_entry_present": bool(hlo and "ENTRY" in hlo),
            "concrete_trace_count": trace_count,
            "output_devices": devices,
        }
        if row != expected or trace_count != 1 or not all(expected_fragment in device for device in devices):
            raise ContractError(f"compiler/HLO/device evidence differs for {name}")


def _verify_diagnostics(payload: dict[str, Any], core: dict[str, Any], fixture: dict[str, Any]) -> None:
    tf = core["tf"]
    analytic, manual = core["analytic"], core["manual"]
    expected_residual_rows = {}
    for name, expected in manual.items():
        residual = float(tf.reduce_max(tf.abs(tf.cast(getattr(analytic, name), tf.float64)-expected)))
        threshold = _scale_tolerance(tf, getattr(analytic, name), expected, 512)
        expected_residual_rows[name] = {"residual": residual, "threshold": threshold}
        if residual > threshold:
            raise ContractError(f"analytic residual differs or fails: {name}")
    if payload["deterministic_residuals"] != expected_residual_rows:
        raise ContractError("analytic residual rows differ from independent recomputation")
    analytic_check = next(
        row for row in payload["contract_checks"] if row["name"] == "analytic_formula_exact"
    )
    expected_analytic_check = {
        "name": "analytic_formula_exact",
        "role": "promotion_criterion_and_hard_veto",
        "passed": all(
            row["residual"] <= row["threshold"]
            for row in expected_residual_rows.values()
        ),
        "residual": max(row["residual"] for row in expected_residual_rows.values()),
        "threshold": max(row["threshold"] for row in expected_residual_rows.values()),
    }
    if analytic_check != expected_analytic_check:
        raise ContractError("analytic aggregate gate differs from recomputation")
    if (_status_text(analytic.status) != "VALID" or not bool(analytic.log_variance_valid) or bool(tf.reduce_any(analytic.degenerate_variance_mask)) or float(analytic.state_symmetry_residual) > float(analytic.state_psd_tolerance) or float(analytic.observation_symmetry_residual) > float(analytic.observation_psd_tolerance) or float(analytic.minimum_state_covariance_eigenvalue) < -float(analytic.state_psd_tolerance) or float(analytic.minimum_observation_covariance_eigenvalue) < -float(analytic.observation_psd_tolerance)):
        raise ContractError("analytic covariance/log-variance diagnostics fail")
    left = core["simulations"][0].observations
    summary = core["summaries"][0]
    probabilities = core["analytic"].quantile_probabilities
    manual_summary = _manual_summary(tf, left, probabilities)
    summary_residuals = {
        name: float(
            tf.reduce_max(
                tf.abs(tf.cast(getattr(summary, name), tf.float64) - expected)
            )
        )
        for name, expected in manual_summary.items()
    }
    summary_thresholds = {
        name: _scale_tolerance(tf, getattr(summary, name), expected, 1024)
        for name, expected in manual_summary.items()
    }
    summary_check = next(
        row for row in payload["contract_checks"] if row["name"] == "summary_statistics"
    )
    expected_summary_check = {
        "name": "summary_statistics",
        "role": "promotion_criterion",
        "passed": (
            _status_text(summary.status) == "VALID"
            and int(summary.path_count) == int(math.prod(left.shape[:-1]))
            and all(
                summary_residuals[name] <= summary_thresholds[name]
                for name in manual_summary
            )
        ),
        "residual": max(summary_residuals.values()),
        "threshold": max(summary_thresholds.values()),
    }
    if summary_check != expected_summary_check:
        raise ContractError("summary gate differs from independent recomputation")
    count = math.prod(int(item) for item in left.shape[:-1])
    constants = fixture["fixture_constants"]
    tfp = importlib.import_module("tensorflow_probability")
    alpha = tf.constant(_hex_float(constants["feature_alpha_hex"]), tf.float64)
    tail = alpha / tf.cast(4*int(constants["horizon"]), tf.float64)
    critical = tfp.distributions.Normal(tf.constant(0.0, tf.float64), tf.constant(1.0, tf.float64)).quantile(1.0-tail)
    z = tf.abs(summary.means-analytic.observation_mean) / tf.sqrt(analytic.observation_variance/tf.cast(count, tf.float64))
    df = tf.constant(float(count-1), tf.float64)
    ratio = df*summary.variances/analytic.observation_variance
    chi = tfp.distributions.Chi2(df)
    if not bool(tf.reduce_max(z) <= critical) or not bool(tf.reduce_all((ratio >= chi.quantile(tail)) & (ratio <= chi.quantile(1.0-tail)))):
        raise ContractError("independent Monte Carlo Normal/chi-square screen fails")
    metadata = payload["statistical_metadata"]
    if metadata["quadratic_mmd"] != {"status": "VALID", "sampling_contract": "iid_oracle_fixture", "iid_samples_verified": True, "independent_arm_banks_verified": core["primary_domain_nonreuse"], "inference_admissible": False}:
        raise ContractError("quadratic MMD role metadata differs")
    if core["mmd"].inference_admissible or _status_text(core["mmd"].status) != "VALID":
        raise ContractError("quadratic MMD is not valid descriptive-only output")
    signed = core["stats"].fixed_rbf_mmd(tf.constant([[0.0],[2.0]], tf.float64), tf.constant([[0.0],[2.0]], tf.float64), bandwidths=tf.constant([0.5,1.0], tf.float64), mixture_weights=tf.constant([0.5,0.5], tf.float64), sampling_contract="paired_diagnostic_shared", jit_compile=True)
    if not float(signed.squared_mmd_u) < 0.0 or signed.inference_admissible:
        raise ContractError("signed-U preservation fixture fails")
    provenance = payload["bank_provenance"]
    expected_arm_rows = [
        {
            "arm_id": arm_id,
            "section": section,
            "role": "domain_separated_oracle_arm_not_probabilistic_independence",
            "tensor_hashes": hashes,
        }
        for arm_id, section, hashes in (
            (1, "innovation_bank_left", core["bank_hashes"][0]),
            (2, "innovation_bank_right", core["bank_hashes"][1]),
        )
    ]
    if provenance.get("arm_rows") != expected_arm_rows:
        raise ContractError("primary bank provenance differs from tensor authority")
    if not core["global_domain_nonreuse"]:
        raise ContractError("primary arm innovation families are reused")


def _verify_alternatives(payload: dict[str, Any], core: dict[str, Any], fixture: dict[str, Any]) -> None:
    tf, stats = core["tf"], core["stats"]
    constants = fixture["fixture_constants"]
    feature_alpha = tf.constant(_hex_float(constants["feature_alpha_hex"]), tf.float64)
    mmd_alpha = tf.constant(_hex_float(constants["mmd_alpha_hex"]), tf.float64)
    total_alpha = tf.constant(_hex_float(constants["total_alpha_hex"]), tf.float64)
    margins = core["margins"]
    tolerance = core["mmd_tolerance"]
    bandwidths = tf.constant([_hex_float(item) for item in constants["bandwidths_hex"]], tf.float64)
    weights = tf.constant([_hex_float(item) for item in constants["mixture_weights_hex"]], tf.float64)
    schedule = tf.constant(constants["chain_pair_schedule"], tf.int32)
    records = []
    bootstrap_left = _resample(
        tf, core["simulations"][0].observations, core["indices_left"]
    )
    analytic = core["analytic"]
    right = core["simulations"][1].observations
    alternatives = fixture["controlled_alternatives"]
    mean_shift = tf.constant(_hex_float(alternatives["mean_shift_hex"]), tf.float64)
    variance_increment = tf.constant(
        _hex_float(alternatives["variance_increment_hex"]), tf.float64
    )
    skew_coefficient = tf.constant(
        _hex_float(alternatives["skew_coefficient_hex"]), tf.float64
    )
    dependence_correlation = tf.constant(
        _hex_float(alternatives["dependence_correlation_hex"]), tf.float64
    )
    centered = right - analytic.observation_mean
    standardized_base = centered / tf.sqrt(analytic.observation_variance)
    common = standardized_base[..., :1]
    correlations = analytic.observation_covariance[:, 0] / tf.sqrt(
        analytic.observation_variance * analytic.observation_variance[0]
    )
    independent_weight = tf.sqrt(1.0 - tf.square(dependence_correlation))
    normalization = tf.sqrt(
        1.0
        + 2.0
        * independent_weight
        * dependence_correlation
        * correlations
    )
    reconstructed_paths = {
        "mean": right + mean_shift,
        "variance": analytic.observation_mean
        + centered
        * tf.sqrt(
            (analytic.observation_variance + variance_increment)
            / analytic.observation_variance
        ),
        "skew": right
        + skew_coefficient
        * (tf.square(centered) - analytic.observation_variance),
        "dependence": analytic.observation_mean
        + tf.sqrt(analytic.observation_variance)
        * (
            independent_weight * standardized_base
            + dependence_correlation * common
        )
        / normalization,
    }
    config = stats.PredictiveStatisticsConfig(
        horizon=int(constants["horizon"]),
        quantile_probabilities=tuple(
            _hex_float(item)
            for item in fixture["quantile_contract"]["probabilities_hex"]
        ),
        jit_compile=True,
    )
    right_summary = core["summaries"][1]
    mechanics = {
        "mean_shift_mean_residual": float(
            tf.reduce_max(
                tf.abs(
                    tf.reduce_mean(
                        reconstructed_paths["mean"] - right, axis=[0, 1, 2]
                    )
                    - tf.fill([int(constants["horizon"])], mean_shift)
                )
            )
        ),
        "variance_log_variance_direction": float(
            tf.reduce_max(
                stats.summarize_forecast_paths(
                    reconstructed_paths["variance"], config
                ).log_variances
                - right_summary.log_variances
            )
        ),
        "skew_third_moment_change": float(
            tf.reduce_max(
                tf.abs(
                    stats.summarize_forecast_paths(
                        reconstructed_paths["skew"], config
                    ).central_moments[0]
                    - right_summary.central_moments[0]
                )
            )
        ),
        "dependence_covariance_change": float(
            tf.reduce_max(
                tf.abs(
                    stats.summarize_forecast_paths(
                        reconstructed_paths["dependence"], config
                    ).cross_horizon_covariance
                    - right_summary.cross_horizon_covariance
                )
            )
        ),
    }
    for name, paths in reconstructed_paths.items():
        if _section(payload, f"alternative_{name}_inputs").get("paths") != _tensor_row(
            "paths", paths
        ):
            raise ContractError(f"alternative {name} path differs from frozen formula")
        summary = stats.summarize_forecast_paths(paths, config)
        estimate = tf.concat(
            [
                core["summaries"][0].means - summary.means,
                core["summaries"][0].log_variances - summary.log_variances,
            ],
            axis=0,
        )
        bootstrap_right = _resample(tf, paths, core["indices_right"])
        bootstrap = _bootstrap_features(tf, bootstrap_left, bootstrap_right)
        if _tensor_row("feature_estimate", estimate) != _section(
            payload, f"alternative_{name}_inputs"
        )["feature_estimate"] or _tensor_row(
            "bootstrap_feature_estimates", bootstrap
        ) != _section(payload, f"alternative_{name}_inputs")[
            "bootstrap_feature_estimates"
        ]:
            raise ContractError(f"alternative {name} persisted features differ")
        feature = stats.simultaneous_feature_intervals(estimate, feature_alpha=feature_alpha, method="bootstrap_max_statistic", bootstrap_estimates=bootstrap, minimum_bootstrap_count=20, jit_compile=True)
        standardized = stats.standardize_forecast_paths(paths, core["analytic"].observation_mean, tf.sqrt(core["analytic"].observation_variance), scale_floor=tf.constant(2.0**-40, tf.float64), jit_compile=True, allow_floor_use=False)
        cross = stats.cross_chain_linear_mmd(core["standardized"][0], standardized, bandwidths=bandwidths, mixture_weights=weights, chain_pair_schedule=schedule, independent_arm_banks_verified=core["global_domain_nonreuse"], stationarity_verified=True, mixing_verified=True, jit_compile=True)
        interval = stats.cross_chain_mmd_upper_interval(cross, mmd_alpha=mmd_alpha, block_length=int(constants["block_length"]), jit_compile=True)
        decision = stats.classify_predictive_evidence(feature, interval, margins=margins, mmd_tolerance=tolerance, total_alpha=total_alpha, feature_alpha=feature_alpha, mmd_alpha=mmd_alpha)
        valid = feature.inference_admissible and interval.inference_admissible and decision.status != "INVALID_HARD_VETO"
        records.append({"name": name, "valid": bool(valid), "decision": _decision_row(name, decision), "repair_trigger": bool(valid and decision.status != "MATERIAL_DIFFERENCE"), "feature_max_abs": float(tf.reduce_max(tf.abs(estimate))), "mmd_estimate": float(cross.squared_mmd_linear), "mmd_lower": float(interval.lower), "mmd_upper": float(interval.upper)})
        _near_rows(list(_section(payload, f"alternative_{name}_feature_interval").values()), [_tensor_row(field, value) for field, value in _dataclass_dict(feature).items() if hasattr(value, "dtype") and getattr(value.dtype, "is_floating", False)], f"alternative {name} feature interval")
        _near_rows(list(_section(payload, f"alternative_{name}_cross_chain").values()), [_tensor_row("squared_mmd_linear", cross.squared_mmd_linear), _tensor_row("kernel_contrast_sequence", cross.kernel_contrast_sequence), _int_tensor_row("chain_pair_schedule", cross.chain_pair_schedule)], f"alternative {name} cross-chain")
        _near_rows(list(_section(payload, f"alternative_{name}_mmd_interval").values()), [(_int_tensor_row(field, value) if getattr(value.dtype, "is_integer", False) else _tensor_row(field, value)) for field, value in _dataclass_dict(interval).items() if hasattr(value, "dtype") and (getattr(value.dtype, "is_floating", False) or getattr(value.dtype, "is_integer", False))], f"alternative {name} MMD interval")
    expected_alternative_diagnostics = {
        "mechanics": mechanics,
        "records": records,
        "policy": "valid_underpowered_is_repair_trigger_not_hard_veto",
    }
    if payload["alternative_diagnostics"] != expected_alternative_diagnostics:
        raise ContractError("controlled-alternative diagnostics differ")
    by_name = {row["name"]: row for row in records}
    if (
        mechanics["mean_shift_mean_residual"] > 512.0 * 2.0**-52
        or mechanics["variance_log_variance_direction"] <= 0.0
        or mechanics["skew_third_moment_change"] <= 0.0
        or mechanics["dependence_covariance_change"] <= 0.0
        or by_name["mean"]["decision"]["status"] != "MATERIAL_DIFFERENCE"
        or by_name["variance"]["decision"]["status"] != "MATERIAL_DIFFERENCE"
        or not all(row["valid"] for row in records)
    ):
        raise ContractError("controlled-alternative minimum mechanics gate fails")


def _trace_for_artifact(path: Path, schema: str) -> Path:
    if schema == CPU_SCHEMA:
        return PHASE_DIR / "oracle-cpu-generation-write-trace.log"
    return PHASE_DIR / "oracle-gpu-generation-write-trace.log"


_TRACE_INTERPRETER = "/home/ubuntu/anaconda3/envs/tfgpu/bin/python"
_TRACE_CALL = re.compile(
    r"^(?P<pid>\d+)\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\((?P<args>.*)\)\s+=\s+(?P<result>.*)$"
)
_TRACE_UNFINISHED = re.compile(
    r"^(?P<pid>\d+)\s+(?P<name>[A-Za-z_][A-Za-z0-9_]*)\((?P<prefix>.*)<unfinished \.\.\.>$"
)
_TRACE_RESUMED = re.compile(
    r"^(?P<pid>\d+)\s+<\.\.\. (?P<name>[A-Za-z_][A-Za-z0-9_]*) resumed>(?P<suffix>.*)$"
)
_TRACE_SIGNAL = re.compile(r"^\d+\s+--- SIG[A-Z0-9]+ \{.*\} ---$")
_TRACE_LIFECYCLE = re.compile(
    r"^\d+\s+\+\+\+ (?:exited with \d+|killed by SIG[A-Z0-9]+(?: \(core dumped\))?) \+\+\+$"
)
_WRITE_FLAGS = frozenset(
    {"O_WRONLY", "O_RDWR", "O_CREAT", "O_TRUNC", "O_APPEND", "O_TMPFILE"}
)
_READ_ONLY_FILE_SYSCALLS = frozenset(
    {
        "access", "chdir", "faccessat", "faccessat2", "fchdir", "getcwd",
        "getxattr", "lgetxattr", "listxattr", "llistxattr", "lstat",
        "name_to_handle_at", "newfstatat", "readlink", "readlinkat", "stat",
        "statfs", "statfs64", "statx",
    }
)
_PATH_MUTATION_ARGUMENTS = {
    "chmod": (0,), "chown": (0,), "fchmodat": (1,), "fchmodat2": (1,),
    "fchownat": (1,), "futimesat": (1,), "lchown": (0,), "link": (0, 1),
    "linkat": (1, 3), "lremovexattr": (0,), "lsetxattr": (0,),
    "mkdir": (0,), "mkdirat": (1,), "mknod": (0,), "mknodat": (1,),
    "removexattr": (0,), "rename": (0, 1), "renameat": (1, 3),
    "renameat2": (1, 3), "rmdir": (0,), "setxattr": (0,),
    "symlink": (1,), "symlinkat": (2,), "truncate": (0,), "unlink": (0,),
    "unlinkat": (1,), "utime": (0,), "utimensat": (1,), "utimes": (0,),
}
_DESCRIPTOR_MUTATION_DESTINATION = {
    "copy_file_range": 2,
    "fallocate": 0,
    "fchmod": 0,
    "fchown": 0,
    "fremovexattr": 0,
    "fsetxattr": 0,
    "ftruncate": 0,
    "pwrite64": 0,
    "pwritev": 0,
    "pwritev2": 0,
    "sendfile": 0,
    "sendfile64": 0,
    "splice": 2,
    "tee": 1,
    "vmsplice": 0,
    "write": 0,
    "writev": 0,
}
_WRITABLE_MMAP_SYSCALLS = frozenset({"mmap", "mmap2"})
_MAPPED_WRITE_FLUSH_SYSCALLS = frozenset({"msync"})
_PROCESS_TERMINATION_SYSCALLS = frozenset({"exit", "exit_group"})
_FORBIDDEN_NAMESPACE_MUTATIONS = frozenset(
    {
        "chroot", "fspick", "fsopen", "fsmount", "move_mount", "mount",
        "open_tree", "pivot_root", "swapoff", "swapon", "umount", "umount2",
    }
)


def _trace_contract(path: Path) -> dict[str, Any]:
    cpu_artifact = CPU_REFERENCE_PATH
    cpu_log = PHASE_DIR / "oracle-cpu-reference.log"
    gpu_artifact = PHASE_DIR / "oracle-gpu-xla-canary.json"
    gpu_log = PHASE_DIR / "oracle-gpu-xla-canary.log"
    closure = PHASE_DIR / "post-result-closure.json"
    contracts = {
        "focused-tests-write-trace.log": {
            "argv": (
                _TRACE_INTERPRETER, "-m", "pytest", "-p", "no:cacheprovider", "-q",
                ORACLE_TEST.as_posix(), PREDICTIVE_TEST.as_posix(),
            ),
            "writes": (),
        },
        CPU_GENERATION_TRACE_PATH.name: {
            "argv": (
                _TRACE_INTERPRETER, GENERATOR_PATH.as_posix(),
                "--mode", "cpu-reference", "--fixture", FIXTURE_PATH.as_posix(),
                "--output", cpu_artifact.as_posix(), "--log-path", cpu_log.as_posix(),
            ),
            "writes": (cpu_artifact, cpu_log),
        },
        CPU_VERIFICATION_TRACE_PATH.name: {
            "argv": (
                _TRACE_INTERPRETER, VERIFIER_PATH.as_posix(),
                "--artifact", cpu_artifact.as_posix(),
                "--log-path", CPU_VERIFY_RECEIPT_PATH.as_posix(),
            ),
            "writes": (CPU_VERIFY_RECEIPT_PATH,),
        },
        "oracle-gpu-generation-write-trace.log": {
            "argv": (
                _TRACE_INTERPRETER, GENERATOR_PATH.as_posix(),
                "--mode", "gpu-xla-canary", "--fixture", FIXTURE_PATH.as_posix(),
                "--cpu-reference", cpu_artifact.as_posix(),
                "--output", gpu_artifact.as_posix(), "--log-path", gpu_log.as_posix(),
            ),
            "writes": (gpu_artifact, gpu_log),
        },
        "oracle-gpu-verification-write-trace.log": {
            "argv": (
                _TRACE_INTERPRETER, VERIFIER_PATH.as_posix(),
                "--artifact", gpu_artifact.as_posix(),
                "--log-path", (PHASE_DIR / "oracle-gpu-xla-canary-verify.log").as_posix(),
            ),
            "writes": (PHASE_DIR / "oracle-gpu-xla-canary-verify.log",),
        },
        "executor-ledger-generation-write-trace.log": {
            "argv": (
                _TRACE_INTERPRETER, VERIFIER_PATH.as_posix(), "--write-executor-ledger",
                "--output", (PHASE_DIR / "executor-write-ledger.json").as_posix(),
            ),
            "writes": (PHASE_DIR / "executor-write-ledger.json",),
        },
        "final-checkpoint-generation-write-trace.log": {
            "argv": (
                _TRACE_INTERPRETER, VERIFIER_PATH.as_posix(), "--write-final-checkpoint",
                "--output", (PHASE_DIR / "final-checkpoint.json").as_posix(),
            ),
            "writes": (PHASE_DIR / "final-checkpoint.json",),
        },
        "post-result-ledger-generation-write-trace.log": {
            "argv": (
                _TRACE_INTERPRETER, VERIFIER_PATH.as_posix(), "--write-post-result-ledger",
                "--output", (PHASE_DIR / "post-result-write-ledger.json").as_posix(),
            ),
            "writes": (PHASE_DIR / "post-result-write-ledger.json",),
        },
        "closure-generation-write-trace.log": {
            "argv": (
                _TRACE_INTERPRETER, VERIFIER_PATH.as_posix(), "--close-phase",
                "--output", closure.as_posix(),
            ),
            "writes": (closure,),
        },
        "closure-verification-write-trace.log": {
            "argv": (
                _TRACE_INTERPRETER, VERIFIER_PATH.as_posix(),
                "--verify-closure", closure.as_posix(),
                "--log-path", (PHASE_DIR / "post-result-closure-verify.log").as_posix(),
            ),
            "writes": (PHASE_DIR / "post-result-closure-verify.log",),
        },
    }
    try:
        return contracts[path.name]
    except KeyError as exc:
        raise ContractError(f"unmapped A3 trace role: {path}") from exc


def _split_arguments(arguments: str) -> list[str]:
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
        elif character in "([{":
            depth += 1
        elif character in ")]}":
            depth = max(0, depth - 1)
        elif character == "," and depth == 0:
            result.append(arguments[start:index].strip())
            start = index + 1
    result.append(arguments[start:].strip())
    return result


def _trace_records(lines: list[str]) -> list[dict[str, str]]:
    pending: dict[str, tuple[str, str]] = {}
    records: list[dict[str, str]] = []
    for raw_line in lines:
        if _TRACE_SIGNAL.fullmatch(raw_line) or _TRACE_LIFECYCLE.fullmatch(raw_line):
            continue
        unfinished = _TRACE_UNFINISHED.fullmatch(raw_line)
        if unfinished is not None:
            pid = unfinished.group("pid")
            if pid in pending:
                raise ContractError(f"nested unfinished trace record for PID {pid}")
            pending[pid] = (unfinished.group("name"), unfinished.group("prefix"))
            continue
        resumed = _TRACE_RESUMED.fullmatch(raw_line)
        if resumed is not None:
            pid = resumed.group("pid")
            if pid not in pending:
                raise ContractError(f"resumed trace record lacks pending call: {raw_line}")
            name, prefix = pending.pop(pid)
            if resumed.group("name") != name:
                raise ContractError(f"resumed trace syscall mismatch for PID {pid}")
            raw_line = f"{pid} {name}({prefix}{resumed.group('suffix')}"
        elif "<unfinished ...>" in raw_line or " resumed>" in raw_line:
            raise ContractError(f"malformed split trace record: {raw_line}")
        match = _TRACE_CALL.fullmatch(raw_line)
        if match is None:
            raise ContractError(f"unparsed trace record: {raw_line}")
        if (
            match.group("name") not in _DESCRIPTOR_MUTATION_DESTINATION
            and re.search(r'"(?:[^"\\]|\\.)*"\.\.\.', raw_line)
        ) or re.search(r"\d+</[^>]*\.\.\.[^>]*>", raw_line):
            raise ContractError(f"truncated trace value: {raw_line}")
        records.append(
            {
                "pid": match.group("pid"),
                "name": match.group("name"),
                "arguments": match.group("args"),
                "result": match.group("result").strip(),
                "line": raw_line,
            }
        )
    if pending:
        raise ContractError(f"trace ended with pending calls for PIDs {sorted(pending)}")
    return records


def _successful_trace_call(result: str) -> bool:
    if result.startswith("-1 "):
        return False
    if result == "?" or result.startswith("? "):
        raise ContractError(f"trace call has indeterminate result {result!r}")
    return True


def _trace_string(token: str) -> str:
    try:
        value = ast.literal_eval(token.strip())
    except (SyntaxError, ValueError) as exc:
        raise ContractError(f"invalid strace string token {token!r}") from exc
    if not isinstance(value, str):
        raise ContractError(f"strace token is not a string: {token!r}")
    return value


def _trace_argv(token: str) -> tuple[str, ...]:
    token = token.strip()
    if not token.startswith("[") or not token.endswith("]"):
        raise ContractError(f"strace argv is not a complete array: {token!r}")
    body = token[1:-1].strip()
    if not body:
        return ()
    return tuple(_trace_string(item) for item in _split_arguments(body))


def _fd_annotation(token: str) -> tuple[int, str] | None:
    match = re.match(r"^(?P<fd>-?\d+)<(?P<target>[^>]*)>", token.strip())
    if match is None:
        return None
    return int(match.group("fd")), match.group("target")


def _resolved_exec(record: dict[str, str]) -> tuple[str, tuple[str, ...]]:
    arguments = _split_arguments(record["arguments"])
    if record["name"] == "execve" and len(arguments) >= 2:
        executable = _trace_string(arguments[0])
        argv_token = arguments[1]
    elif record["name"] == "execveat" and len(arguments) >= 3:
        executable = _trace_string(arguments[1])
        if not executable:
            annotation = _fd_annotation(arguments[0])
            if annotation is None or not annotation[1].startswith("/"):
                raise ContractError("execveat AT_EMPTY_PATH lacks a resolved executable")
            executable = annotation[1]
        elif not executable.startswith("/"):
            annotation = _fd_annotation(arguments[0])
            if annotation is None or not annotation[1].startswith("/"):
                raise ContractError("relative execveat executable lacks resolved dirfd")
            executable = str(Path(annotation[1]) / executable)
        argv_token = arguments[2]
    else:
        raise ContractError(f"malformed execution trace record: {record['line']}")
    return str(Path(executable).resolve(strict=False)), _trace_argv(argv_token)


def _authenticate_trace_execution(
    records: list[dict[str, str]], expected_argv: tuple[str, ...]
) -> tuple[str, int]:
    if not records:
        raise ContractError("trace contains no syscall records")
    first = records[0]
    root_pid = first["pid"]
    if (
        first["name"] not in {"execve", "execveat"}
        or not _successful_trace_call(first["result"])
    ):
        raise ContractError("trace does not begin with the successful root execution")
    root_executions = [
        row
        for row in records
        if row["pid"] == root_pid
        and row["name"] in {"execve", "execveat"}
        and _successful_trace_call(row["result"])
    ]
    if len(root_executions) != 1:
        raise ContractError("trace must contain exactly one successful root execution")
    executable, argv = _resolved_exec(root_executions[0])
    if (
        executable != str(Path(expected_argv[0]).resolve(strict=False))
        or argv != expected_argv
    ):
        raise ContractError("trace root execution does not match the reviewed role argv")
    child_executions = sum(
        row["pid"] != root_pid
        and row["name"] in {"execve", "execveat"}
        and _successful_trace_call(row["result"])
        for row in records
    )
    root_terminations = [
        row
        for row in records
        if row["pid"] == root_pid and row["name"] in _PROCESS_TERMINATION_SYSCALLS
    ]
    if len(root_terminations) != 1 or _trace_exit_code(root_terminations[0]) != 0:
        raise ContractError("trace root process did not terminate exactly once with exit code zero")
    return root_pid, child_executions


def _inside(path: Path, roots: tuple[Path, ...]) -> bool:
    resolved = path.resolve(strict=False)
    return any(resolved == root or root in resolved.parents for root in roots)


def _resolved_path_argument(arguments: list[str], index: int) -> Path:
    if index >= len(arguments):
        raise ContractError("mutation trace lacks a required path argument")
    value = _trace_string(arguments[index])
    candidate = Path(value)
    if candidate.is_absolute():
        return candidate.resolve(strict=False)
    if index > 0:
        previous = arguments[index - 1].strip()
        match = re.match(r"^(?:AT_FDCWD|-?\d+)<(?P<target>[^>]*)>", previous)
        if match is not None:
            target = match.group("target")
            if not target.startswith("/"):
                raise ContractError("relative mutation dirfd lacks an absolute -yy path")
            return (Path(target) / candidate).resolve(strict=False)
        if previous == "AT_FDCWD" or re.fullmatch(r"-?\d+", previous):
            raise ContractError("relative mutation dirfd lacks a -yy annotation")
    raise ContractError(f"relative mutation path lacks a resolved dirfd: {value!r}")


def _write_open_destination(
    name: str, arguments: list[str], result: str
) -> tuple[Path, str] | None:
    if name == "creat":
        flags = "O_WRONLY|O_CREAT|O_TRUNC"
    elif name == "open" and len(arguments) >= 2:
        flags = arguments[1]
    elif name in {"openat", "openat2"} and len(arguments) >= 3:
        flags = arguments[2]
    else:
        return None
    if name != "creat" and not any(flag in flags for flag in _WRITE_FLAGS):
        return None
    annotation = _fd_annotation(result)
    if annotation is None or not annotation[1].startswith("/"):
        raise ContractError("write open lacks a resolved -yy destination")
    return Path(annotation[1]).resolve(strict=False), flags


def _is_null_device_result(result: str) -> bool:
    return re.fullmatch(r"-?\d+</dev/null<char 1:3>>", result.strip()) is not None


def _is_authenticated_thread_name_path(path: Path, root_pid: str) -> bool:
    parts = path.resolve(strict=False).parts
    return (
        len(parts) == 6
        and parts[:2] == ("/", "proc")
        and parts[2] == root_pid
        and parts[3] == "task"
        and parts[4].isdigit()
        and parts[5] == "comm"
    )


def _descriptor_destination(arguments: list[str], index: int) -> tuple[int, str]:
    if index >= len(arguments):
        raise ContractError("descriptor mutation lacks its destination argument")
    annotation = _fd_annotation(arguments[index])
    if annotation is None:
        raise ContractError("descriptor mutation destination lacks -yy annotation")
    return annotation


def _trace_exit_code(record: dict[str, str]) -> int:
    arguments = _split_arguments(record["arguments"])
    if (
        record["name"] not in _PROCESS_TERMINATION_SYSCALLS
        or len(arguments) != 1
        or not re.fullmatch(r"-?\d+", arguments[0])
        or record["result"] != "?"
    ):
        raise ContractError(f"malformed process-termination trace: {record['line']}")
    return int(arguments[0])


def _audit_trace(
    path: Path,
    *,
    terminal: bool = False,
) -> str:
    lines = (ROOT / path).read_text(encoding="utf-8", errors="strict").splitlines()
    if not lines:
        raise ContractError(f"empty trace: {path}")
    contract = _trace_contract(path)
    records = _trace_records(lines)
    root_pid, child_execution_count = _authenticate_trace_execution(
        records, contract["argv"]
    )
    if terminal and child_execution_count != 0:
        raise ContractError("terminal trace contains successful child executions")
    temporary_roots = tuple(
        item.resolve(strict=False)
        for item in (
            Path("/tmp/bayesfilter-a3-pycache"),
            Path("/tmp/bayesfilter-a3-tmp"),
        )
    )
    expected = {
        (ROOT / item).resolve(strict=False) for item in contract["writes"]
    }
    observed_repository_writes: set[Path] = set()
    for record in records:
        name = record["name"]
        arguments = _split_arguments(record["arguments"])
        result = record["result"]
        if name in _PROCESS_TERMINATION_SYSCALLS:
            _trace_exit_code(record)
            continue
        if not _successful_trace_call(result):
            continue
        if name in {"execve", "execveat"} or name in _READ_ONLY_FILE_SYSCALLS:
            continue
        if name in {"open", "openat", "openat2", "creat"}:
            destination = _write_open_destination(name, arguments, result)
            if destination is None:
                continue
            target, flags = destination
            if _is_null_device_result(result) or target == Path("/dev/null") or (
                target.parts[:2] == ("/", "dev")
                and all(
                    flag not in flags
                    for flag in ("O_CREAT", "O_TRUNC", "O_APPEND", "O_TMPFILE")
                )
            ):
                continue
            if _is_authenticated_thread_name_path(target, root_pid):
                continue
            if _inside(target, temporary_roots):
                continue
            if target not in expected:
                raise ContractError(f"unexpected A3 repository write: {record['line']}")
            observed_repository_writes.add(target)
            continue
        if name in _PATH_MUTATION_ARGUMENTS:
            targets = tuple(
                _resolved_path_argument(arguments, index)
                for index in _PATH_MUTATION_ARGUMENTS[name]
            )
            if not targets or not all(_inside(target, temporary_roots) for target in targets):
                raise ContractError(f"path mutation escaped reviewed temp roots: {record['line']}")
            continue
        if name in _DESCRIPTOR_MUTATION_DESTINATION:
            fd, annotation = _descriptor_destination(
                arguments, _DESCRIPTOR_MUTATION_DESTINATION[name]
            )
            if annotation.startswith("/"):
                target = Path(annotation).resolve(strict=False)
                if _is_authenticated_thread_name_path(target, root_pid):
                    continue
                if _inside(target, temporary_roots):
                    continue
                if target in expected:
                    observed_repository_writes.add(target)
                    continue
                if target.parts[:2] == ("/", "dev"):
                    continue
            elif annotation.startswith(("pipe:[", "socket:[", "anon_inode:")):
                continue
            raise ContractError(f"descriptor mutation escaped reviewed outputs: {record['line']}")
        if name in _WRITABLE_MMAP_SYSCALLS:
            if len(arguments) < 5:
                raise ContractError(f"malformed mmap trace record: {record['line']}")
            writable_shared = (
                "PROT_WRITE" in arguments[2] and "MAP_SHARED" in arguments[3]
            )
            anonymous = "MAP_ANONYMOUS" in arguments[3] or "MAP_ANON" in arguments[3]
            if not writable_shared or anonymous:
                continue
            fd, annotation = _descriptor_destination(arguments, 4)
            if annotation.startswith("/"):
                target = Path(annotation).resolve(strict=False)
                if _inside(target, temporary_roots) or target in expected:
                    if target in expected:
                        observed_repository_writes.add(target)
                    continue
                if target.parts[:2] == ("/", "dev"):
                    continue
            raise ContractError(f"writable shared mmap escaped reviewed outputs: {record['line']}")
        if name in _MAPPED_WRITE_FLUSH_SYSCALLS:
            continue
        if name in _FORBIDDEN_NAMESPACE_MUTATIONS:
            raise ContractError(f"forbidden namespace mutation: {record['line']}")
        raise ContractError(f"unclassified successful file syscall: {record['line']}")
    missing = sorted(str(item) for item in expected - observed_repository_writes)
    if missing:
        raise ContractError(f"trace lacks expected successful write opens: {missing}")
    return _sha256(path)


def _verify_cpu_replay_authority_chain(
    *, expected_file_sha256: str, expected_evidence_signature: str
) -> dict[str, Any]:
    for required in (
        CPU_REFERENCE_PATH,
        CPU_VERIFY_RECEIPT_PATH,
        CPU_GENERATION_TRACE_PATH,
        CPU_VERIFICATION_TRACE_PATH,
    ):
        if not (ROOT / required).is_file():
            raise ContractError(f"required CPU replay-authority chain member missing: {required}")
    cpu_payload = _load_json(CPU_REFERENCE_PATH)
    if (ROOT / CPU_REFERENCE_PATH).read_bytes() != _canonical_bytes(cpu_payload) + b"\n":
        raise ContractError("CPU replay-authority artifact is not canonical JSON")
    current_artifact_sha256 = _sha256(CPU_REFERENCE_PATH)
    if (
        cpu_payload.get("schema_version") != CPU_SCHEMA
        or cpu_payload.get("status") != CPU_STATUS
        or cpu_payload.get("evidence_signature") != _signature(cpu_payload)
        or current_artifact_sha256 != expected_file_sha256
        or cpu_payload.get("evidence_signature") != expected_evidence_signature
    ):
        raise ContractError("GPU CPU-crosslink mismatch")
    generation_trace_sha256 = _audit_trace(CPU_GENERATION_TRACE_PATH)
    receipt = _load_json(CPU_VERIFY_RECEIPT_PATH)
    if (ROOT / CPU_VERIFY_RECEIPT_PATH).read_bytes() != _canonical_bytes(receipt) + b"\n":
        raise ContractError("CPU verification receipt is not canonical JSON")
    expected_receipt = {
        "status": "A3_RUNTIME_ARTIFACT_VERIFIED",
        "artifact_sha256": current_artifact_sha256,
        "evidence_signature": cpu_payload["evidence_signature"],
        "generation_trace_sha256": generation_trace_sha256,
    }
    if receipt != expected_receipt:
        raise ContractError("CPU verification receipt does not bind the current replay authority")
    _audit_trace(CPU_VERIFICATION_TRACE_PATH)
    return cpu_payload


def _verify_manifest(
    payload: dict[str, Any],
    fixture: dict[str, Any],
    boundary: dict[str, Any],
    path: Path,
    schema: str,
    tf: Any,
) -> None:
    cpu = schema == CPU_SCHEMA
    canonical_artifact = PHASE_DIR / (
        "oracle-cpu-reference.json" if cpu else "oracle-gpu-xla-canary.json"
    )
    canonical_log = PHASE_DIR / (
        "oracle-cpu-reference.log" if cpu else "oracle-gpu-xla-canary.log"
    )
    if path != canonical_artifact:
        raise ContractError("runtime artifact path is not canonical for its schema")
    argv = [
        GENERATOR_PATH.as_posix(),
        "--mode",
        "cpu-reference" if cpu else "gpu-xla-canary",
        "--fixture",
        FIXTURE_PATH.as_posix(),
    ]
    if not cpu:
        argv.extend(
            ["--cpu-reference", (PHASE_DIR / "oracle-cpu-reference.json").as_posix()]
        )
    argv.extend(
        ["--output", canonical_artifact.as_posix(), "--log-path", canonical_log.as_posix()]
    )
    environment_names = (
        "CUDA_VISIBLE_DEVICES",
        "PYTHONDONTWRITEBYTECODE",
        "PYTHONPYCACHEPREFIX",
        "TMPDIR",
        "CUDA_CACHE_PATH",
        "XLA_FLAGS",
    )
    expected_environment = {
        "CUDA_VISIBLE_DEVICES": "-1" if cpu else None,
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPYCACHEPREFIX": "/tmp/bayesfilter-a3-pycache",
        "TMPDIR": "/tmp/bayesfilter-a3-tmp",
        "CUDA_CACHE_PATH": "/tmp/bayesfilter-a3-tmp/cuda-cache",
        "XLA_FLAGS": (
            "--xla_gpu_cuda_data_dir=/usr/local/cuda "
            "--xla_dump_to=/tmp/bayesfilter-a3-tmp/xla"
        ),
    }
    if {name: os.environ.get(name) for name in environment_names} != expected_environment:
        raise ContractError("fresh verifier selected environment differs from frozen command")
    command_key = "cpu_artifact" if cpu else "gpu_artifact"
    expected_stable = {
        "git_commit": _git("rev-parse", "HEAD").strip(),
        "command": " ".join(argv),
        "reviewed_command_key": command_key,
        "reviewed_command": boundary["exact_commands"][command_key],
        "cwd": str(ROOT),
        "interpreter": sys.executable,
        "conda_env": os.environ.get("CONDA_DEFAULT_ENV", "unknown"),
        "python_version": sys.version.split()[0],
        "packages": {
            "tensorflow": str(tf.__version__),
            "tensorflow_probability": _package_version("tensorflow-probability"),
        },
        "environment": expected_environment,
        "physical_devices": [
            {"name": str(item.name), "device_type": str(item.device_type)}
            for item in tf.config.list_physical_devices()
        ],
        "logical_devices": [
            {"name": str(item.name), "device_type": str(item.device_type)}
            for item in tf.config.list_logical_devices()
        ],
        "tf32_enabled": bool(
            tf.config.experimental.tensor_float_32_execution_enabled()
        ),
        "jit_compile": True,
        "dtype": "float64",
        "random_seeds": fixture["fixture_constants"]["root_seed"],
        "output_paths": [canonical_artifact.as_posix()],
        "plan_path": PLAN_PATH.as_posix(),
        "result_path": RESULT_PATH.as_posix(),
        "fixture_path": FIXTURE_PATH.as_posix(),
        "fixture_evidence_signature": fixture["evidence_signature"],
        "execution_role": (
            "cpu_hidden_xla_reference" if cpu else "trusted_gpu_xla_oracle"
        ),
        "trust_basis": (
            "cpu_hidden_reference_exception_not_gpu_evidence"
            if cpu
            else "owner_designated_managed_session_visible_gpu_trusted"
        ),
    }
    manifest = payload["run_manifest"]
    generation_recorded = {
        "git_dirty",
        "started_at_utc",
        "completed_at_utc",
        "wall_time_seconds",
    }
    if set(manifest) != set(expected_stable) | generation_recorded:
        raise ContractError("runtime manifest fields differ")
    if {
        key: value for key, value in manifest.items() if key not in generation_recorded
    } != expected_stable:
        raise ContractError("runtime manifest stable values differ")
    snapshot = boundary.get("outside_worktree_snapshot")
    porcelain_count = snapshot.get("porcelain_row_count") if isinstance(snapshot, dict) else None
    if (
        type(manifest["git_dirty"]) is not bool
        or manifest["git_dirty"] is not True
        or type(porcelain_count) is not int
        or porcelain_count <= 0
    ):
        raise ContractError("generation-time dirty-worktree evidence differs")
    try:
        started = datetime.fromisoformat(manifest["started_at_utc"])
        completed = datetime.fromisoformat(manifest["completed_at_utc"])
    except (TypeError, ValueError) as exc:
        raise ContractError("runtime manifest timestamps are invalid") from exc
    if (
        started.tzinfo is None
        or started.utcoffset() is None
        or completed.tzinfo is None
        or completed.utcoffset() is None
        or completed < started
        or payload["created_at_utc"] != manifest["completed_at_utc"]
    ):
        raise ContractError("runtime manifest timestamp ordering differs")
    wall_time = manifest["wall_time_seconds"]
    if (
        not isinstance(wall_time, (int, float))
        or isinstance(wall_time, bool)
        or not math.isfinite(wall_time)
        or wall_time < 0.0
    ):
        raise ContractError("runtime manifest wall time is invalid")


def verify_runtime_artifact(path: Path, log_path: Path | None) -> None:
    boundary, fixture = _verify_fixed_inputs()
    payload = _load_json(path)
    schema = payload.get("schema_version")
    expected_status = CPU_STATUS if schema == CPU_SCHEMA else GPU_STATUS if schema == GPU_SCHEMA else None
    if expected_status is None or payload.get("status") != expected_status:
        raise ContractError("runtime artifact identity/status mismatch")
    expected_keys = {
        "schema_version", "artifact_role", "status", "created_at_utc", "run_manifest",
        "boundary_binding", "fixture_binding", "source_files", "cpu_reference_binding",
        "cpu_gpu_parity", "tensor_sections", "compiler_evidence",
        "deterministic_residuals", "monte_carlo_diagnostics", "alternative_diagnostics",
        "role_ledger", "decision_rows", "uncertainty", "contract_checks",
        "bank_provenance", "resampling_provenance", "statistical_metadata",
        "evidence_signature", "nonclaims",
    }
    if set(payload) != expected_keys:
        raise ContractError(f"runtime artifact keys differ: {sorted(set(payload) ^ expected_keys)}")
    if payload["evidence_signature"] != _signature(payload):
        raise ContractError("runtime evidence signature mismatch")
    if payload["nonclaims"] != NONCLAIMS:
        raise ContractError("runtime nonclaims differ")
    if payload["boundary_binding"]["file_sha256"] != _sha256(BOUNDARY_PATH):
        raise ContractError("boundary crosslink mismatch")
    if payload["fixture_binding"]["file_sha256"] != _sha256(FIXTURE_PATH):
        raise ContractError("fixture crosslink mismatch")
    if payload["source_files"] != _source_rows():
        raise ContractError("runtime source bindings differ")
    check_rows = payload["contract_checks"]
    if [row.get("name") for row in check_rows] != list(CHECK_NAMES):
        raise ContractError("runtime check names/order differ")
    if not all(row.get("passed") is True for row in check_rows):
        raise ContractError("runtime artifact contains a failed hard check")
    for rows in payload["tensor_sections"].values():
        for row in rows:
            _verify_tensor_row(row)
    if payload["role_ledger"].get("quadratic_mmd_u") != "descriptive_only_even_iid_fixture":
        raise ContractError("quadratic MMD role was promoted")
    bank_provenance = payload["bank_provenance"]
    resampling_provenance = payload["resampling_provenance"]
    if not isinstance(bank_provenance, dict) or bank_provenance.get("authority") != "materialized_float64_tensor_rows_and_raw_hashes" or bank_provenance.get("seed_metadata_is_replay_authority") is not False:
        raise ContractError("bank replay authority differs")
    if not isinstance(resampling_provenance, dict) or resampling_provenance.get("authority") != "materialized_indices_replay_authority" or resampling_provenance.get("seed_metadata_is_replay_authority") is not False:
        raise ContractError("resampling replay authority differs")
    expected_resampling_hashes = {
        arm: {
            name: row["raw_little_endian_sha256"]
            for name, row in _section(payload, section).items()
        }
        for arm, section in (
            ("left", "resampling_indices_left"),
            ("right", "resampling_indices_right"),
        )
    }
    if (
        resampling_provenance.get("sections")
        != ["resampling_indices_left", "resampling_indices_right"]
        or resampling_provenance.get("arm_tensor_hashes")
        != expected_resampling_hashes
    ):
        raise ContractError("resampling provenance differs from index authority")
    recomputed = _recompute_core(payload, fixture)
    expected_sections = recomputed["sections"]
    observed_sections = payload["tensor_sections"]
    for section, expected_rows in expected_sections.items():
        observed = _section(payload, section)
        expected_names = [row["name"] for row in expected_rows]
        selected = [observed[name] for name in expected_names]
        _near_rows(selected, expected_rows, section)
    _verify_coverage(payload, recomputed, fixture)
    _verify_diagnostics(payload, recomputed, fixture)
    _verify_decisions(payload, recomputed, fixture)
    _verify_alternatives(payload, recomputed, fixture)
    _verify_compiler(payload, recomputed, schema)
    if schema == GPU_SCHEMA:
        binding = payload["cpu_reference_binding"]
        if not isinstance(binding, dict):
            raise ContractError("GPU artifact lacks CPU crosslink")
        canonical_cpu_path = CPU_REFERENCE_PATH
        if set(binding) != {"path", "file_sha256", "evidence_signature"}:
            raise ContractError("GPU CPU-crosslink fields differ")
        if binding["path"] != canonical_cpu_path.as_posix():
            raise ContractError("GPU CPU-crosslink path is not canonical")
        cpu_payload = _verify_cpu_replay_authority_chain(
            expected_file_sha256=binding["file_sha256"],
            expected_evidence_signature=binding["evidence_signature"],
        )
        parity = payload["cpu_gpu_parity"]
        if not isinstance(parity, dict) or parity.get("passed") is not True:
            raise ContractError("GPU parity did not pass")
        if bank_provenance.get("generation_mode") != "gpu_reconstruction_from_cpu_artifact_values":
            raise ContractError("GPU bank provenance is not CPU-artifact reconstruction")
        for section in (
            "innovation_bank_left", "innovation_bank_right",
            "resampling_indices_left", "resampling_indices_right",
            "coverage_innovation_banks",
        ):
            if payload["tensor_sections"][section] != cpu_payload["tensor_sections"][section]:
                raise ContractError(f"GPU materialized authority differs from CPU: {section}")
        maximum_residual = 0.0
        maximum_threshold = 0.0
        if set(payload["tensor_sections"]) != set(cpu_payload["tensor_sections"]):
            raise ContractError("GPU/CPU tensor section sets differ")
        for section in payload["tensor_sections"]:
            gpu_rows = payload["tensor_sections"][section]
            cpu_rows = cpu_payload["tensor_sections"][section]
            if [row["name"] for row in gpu_rows] != [row["name"] for row in cpu_rows]:
                raise ContractError(f"GPU/CPU tensor row names differ: {section}")
            for gpu_row, cpu_row in zip(gpu_rows, cpu_rows):
                if gpu_row["dtype"] != cpu_row["dtype"] or gpu_row["shape"] != cpu_row["shape"]:
                    raise ContractError(f"GPU/CPU tensor metadata differs: {section}")
                if gpu_row["dtype"] != "float64":
                    if gpu_row != cpu_row:
                        raise ContractError(f"GPU/CPU exact tensor mismatch: {section}")
                    continue
                gpu_values = [float.fromhex(item) for item in gpu_row["values_hex"]]
                cpu_values = [float.fromhex(item) for item in cpu_row["values_hex"]]
                scale = max(1.0, *(abs(item) for item in gpu_values + cpu_values))
                threshold = CPU_GPU_TOLERANCE_MULTIPLIER * 2.0**-52 * scale
                residual = max((abs(a-b) for a, b in zip(gpu_values, cpu_values)), default=0.0)
                maximum_residual = max(maximum_residual, residual)
                maximum_threshold = max(maximum_threshold, threshold)
                if residual > threshold:
                    raise ContractError(f"GPU/CPU tensor parity fails: {section}/{gpu_row['name']}")
        if parity.get("maximum_absolute_residual") != maximum_residual or parity.get("maximum_scale_aware_threshold") != maximum_threshold or parity.get("tolerance_multiplier") != CPU_GPU_TOLERANCE_MULTIPLIER:
            raise ContractError("GPU parity summary differs from independent recomputation")
    elif payload["cpu_reference_binding"] is not None or payload["cpu_gpu_parity"] is not None:
        raise ContractError("CPU artifact must not self-crosslink")
    _verify_manifest(payload, fixture, boundary, path, schema, recomputed["tf"])
    generation_trace = _trace_for_artifact(path, schema)
    trace_hash = _audit_trace(generation_trace)
    status = {
        "status": "A3_RUNTIME_ARTIFACT_VERIFIED",
        "artifact_sha256": _sha256(path),
        "evidence_signature": payload["evidence_signature"],
        "generation_trace_sha256": trace_hash,
    }
    if log_path is not None:
        (ROOT / log_path).write_bytes(_canonical_bytes(status) + b"\n")
    print(_canonical_bytes(status).decode("utf-8"))


def _binding(path: Path, role: str) -> dict[str, Any]:
    if not (ROOT / path).is_file():
        raise ContractError(f"required closure member missing: {path}")
    payload_signature = None
    if path.suffix == ".json":
        value = _load_json(path)
        payload_signature = value.get("evidence_signature")
    return {
        "path": path.as_posix(),
        "file_sha256": _sha256(path),
        "evidence_signature": payload_signature,
        "role": role,
    }


def _verify_binding(row: dict[str, Any], *, role: str) -> None:
    if set(row) != {"path", "file_sha256", "evidence_signature", "role"}:
        raise ContractError("closure binding fields differ")
    if row.get("role") != role:
        raise ContractError(f"closure binding role differs: {row.get('path')}")
    path = Path(row["path"])
    if _binding(path, role) != row:
        raise ContractError(f"closure binding drift: {path}")


def _verify_binding_rows(
    rows: Any,
    expected_paths: tuple[Path, ...],
    *,
    role: str,
) -> None:
    if not isinstance(rows, list) or len(rows) != len(expected_paths):
        raise ContractError(f"{role} binding-row length differs")
    expected_rows = [_binding(path, role) for path in expected_paths]
    if rows != expected_rows:
        raise ContractError(f"{role} binding rows differ")


def _trace_binding(path: Path) -> dict[str, Any]:
    return {
        "path": path.as_posix(),
        "file_sha256": _sha256(path),
        "evidence_signature": None,
        "role": "role_authenticated_a3_trace",
    }


def _verify_trace_chain(
    rows: Any,
    expected_paths: tuple[Path, ...],
) -> None:
    if not isinstance(rows, list) or len(rows) != len(expected_paths):
        raise ContractError("authenticated trace-chain length differs")
    expected_rows = [_trace_binding(path) for path in expected_paths]
    if rows != expected_rows:
        raise ContractError("authenticated trace-chain bindings differ")


def _verify_signed_artifact(
    path: Path,
    *,
    schema: str,
    status: str,
    expected_keys: set[str],
) -> dict[str, Any]:
    payload = _load_json(path)
    if (ROOT / path).read_bytes() != _canonical_bytes(payload) + b"\n":
        raise ContractError(f"noncanonical signed artifact: {path}")
    if set(payload) != expected_keys:
        raise ContractError(f"signed artifact fields differ: {path}")
    if payload.get("schema_version") != schema or payload.get("status") != status:
        raise ContractError(f"signed artifact identity/status differs: {path}")
    if payload.get("evidence_signature") != _signature(payload):
        raise ContractError(f"signed artifact signature differs: {path}")
    if payload.get("nonclaims") != NONCLAIMS:
        raise ContractError(f"signed artifact nonclaims differ: {path}")
    return payload


def _current_head_without_subprocess() -> str:
    git_dir = ROOT / ".git"
    head = (git_dir / "HEAD").read_text(encoding="ascii").strip()
    if not head.startswith("ref: "):
        if re.fullmatch(r"[0-9a-f]{40}", head):
            return head
        raise ContractError("Git HEAD is neither a valid symbolic ref nor detached commit")
    reference = head.removeprefix("ref: ")
    loose = git_dir / reference
    if loose.is_file():
        value = loose.read_text(encoding="ascii").strip()
        if re.fullmatch(r"[0-9a-f]{40}", value):
            return value
        raise ContractError("Git loose HEAD reference is malformed")
    packed = git_dir / "packed-refs"
    if packed.is_file():
        matches = [
            line.split(" ", 1)[0]
            for line in packed.read_text(encoding="ascii").splitlines()
            if not line.startswith(("#", "^")) and line.endswith(f" {reference}")
        ]
        if len(matches) == 1 and re.fullmatch(r"[0-9a-f]{40}", matches[0]):
            return matches[0]
    raise ContractError("Git symbolic HEAD reference cannot be resolved")


def _require_frozen_head(recorded: str | None = None) -> str:
    current = _current_head_without_subprocess()
    if current != HEAD_SHA256 or (recorded is not None and recorded != HEAD_SHA256):
        raise ContractError("governance artifact HEAD differs from the frozen A3 commit")
    _verify_harness_anchor()
    return current


def _base_trace_paths() -> tuple[Path, ...]:
    return (
        FOCUSED_TESTS_TRACE_PATH,
        CPU_GENERATION_TRACE_PATH,
        CPU_VERIFICATION_TRACE_PATH,
        GPU_GENERATION_TRACE_PATH,
        GPU_VERIFICATION_TRACE_PATH,
    )


def _executor_direct_paths() -> tuple[Path, ...]:
    return (
        PREDICTIVE_SOURCE,
        ORACLE_SOURCE,
        PREDICTIVE_TEST,
        ORACLE_TEST,
        GENERATOR_PATH,
        VERIFIER_PATH,
        RESULT_PATH,
    )


def _checkpoint_member_paths() -> tuple[Path, ...]:
    return (
        PLAN_PATH,
        PREDICTIVE_SOURCE,
        ORACLE_SOURCE,
        PREDICTIVE_TEST,
        ORACLE_TEST,
        GENERATOR_PATH,
        VERIFIER_PATH,
        *_artifact_paths(),
        PHASE_DIR / "executor-write-ledger.json",
        EXECUTOR_LEDGER_TRACE_PATH,
        IMPLEMENTATION_REVIEW_PATH,
        RESULT_PATH,
    )


def _post_result_direct_paths() -> tuple[Path, ...]:
    return (RESULT_PATH, RESULT_REVIEW_PATH, A4_PLAN_PATH, A4_REVIEW_PATH, *GOVERNANCE_PATHS)


def _closure_member_paths() -> tuple[Path, ...]:
    return (
        PHASE_DIR / "final-checkpoint.json",
        FINAL_CHECKPOINT_TRACE_PATH,
        RESULT_PATH,
        RESULT_REVIEW_PATH,
        A4_PLAN_PATH,
        A4_REVIEW_PATH,
        PHASE_DIR / "post-result-write-ledger.json",
        POST_RESULT_LEDGER_TRACE_PATH,
        *GOVERNANCE_PATHS,
    )


def _artifact_paths() -> tuple[Path, ...]:
    return (
        HARNESS_ANCHOR_PATH,
        PHASE_DIR / "pre-run-boundary.json",
        PHASE_DIR / "fixture-contract.json",
        FOCUSED_TESTS_TRACE_PATH,
        CPU_GENERATION_TRACE_PATH,
        PHASE_DIR / "oracle-cpu-reference.json",
        PHASE_DIR / "oracle-cpu-reference.log",
        CPU_VERIFICATION_TRACE_PATH,
        PHASE_DIR / "oracle-cpu-reference-verify.log",
        GPU_GENERATION_TRACE_PATH,
        PHASE_DIR / "oracle-gpu-xla-canary.json",
        PHASE_DIR / "oracle-gpu-xla-canary.log",
        GPU_VERIFICATION_TRACE_PATH,
        PHASE_DIR / "oracle-gpu-xla-canary-verify.log",
    )


def _verified_executor_ledger(path: Path) -> dict[str, Any]:
    canonical = PHASE_DIR / "executor-write-ledger.json"
    if path != canonical:
        raise ContractError("executor-ledger validator requires the canonical path")
    payload = _verify_signed_artifact(
        path,
        schema="bayesfilter.ssl_lstm_completion.phase_a3_executor_write_ledger.v1",
        status="A3_EXECUTOR_WRITE_LEDGER_VALID",
        expected_keys={
            "schema_version", "status", "created_at_utc", "git_commit", "rows",
            "trace_bindings", "evidence_signature", "nonclaims",
        },
    )
    _require_frozen_head(payload["git_commit"])
    expected_paths = _executor_direct_paths()
    expected_rows = [
        {
            "sequence": index,
            "path": member.as_posix(),
            "after_sha256": _sha256(member),
            "reason": "A3 reviewed oracle/statistics implementation or result",
        }
        for index, member in enumerate(expected_paths, start=1)
    ]
    if payload["rows"] != expected_rows:
        raise ContractError("executor-ledger source/result rows differ")
    _verify_trace_chain(payload["trace_bindings"], _base_trace_paths())
    return payload


def _verified_final_checkpoint(path: Path) -> dict[str, Any]:
    canonical = PHASE_DIR / "final-checkpoint.json"
    if path != canonical:
        raise ContractError("checkpoint validator requires the canonical path")
    payload = _verify_signed_artifact(
        path,
        schema="bayesfilter.ssl_lstm_completion.phase_a3_final_checkpoint.v1",
        status="A3_PRE_RESULT_CHECKPOINT_PASSED",
        expected_keys={
            "schema_version", "status", "created_at_utc", "git_commit",
            "member_rows", "trace_bindings", "evidence_signature", "nonclaims",
        },
    )
    _require_frozen_head(payload["git_commit"])
    _verify_binding_rows(
        payload["member_rows"],
        _checkpoint_member_paths(),
        role="a3_pre_result_member",
    )
    _verified_executor_ledger(PHASE_DIR / "executor-write-ledger.json")
    _verify_trace_chain(
        payload["trace_bindings"],
        (*_base_trace_paths(), EXECUTOR_LEDGER_TRACE_PATH),
    )
    return payload


def _verified_post_result_ledger(path: Path) -> dict[str, Any]:
    canonical = PHASE_DIR / "post-result-write-ledger.json"
    if path != canonical:
        raise ContractError("post-result-ledger validator requires the canonical path")
    payload = _verify_signed_artifact(
        path,
        schema="bayesfilter.ssl_lstm_completion.phase_a3_post_result_write_ledger.v1",
        status="A3_POST_RESULT_WRITE_LEDGER_VALID",
        expected_keys={
            "schema_version", "status", "created_at_utc", "git_commit", "rows",
            "final_checkpoint_binding", "trace_bindings", "evidence_signature",
            "nonclaims",
        },
    )
    _require_frozen_head(payload["git_commit"])
    _verify_binding_rows(
        payload["rows"],
        _post_result_direct_paths(),
        role="a3_post_result_direct_edit",
    )
    _verify_binding(
        payload["final_checkpoint_binding"],
        role="immutable_a3_final_checkpoint",
    )
    _verified_final_checkpoint(PHASE_DIR / "final-checkpoint.json")
    _verify_trace_chain(
        payload["trace_bindings"],
        (*_base_trace_paths(), EXECUTOR_LEDGER_TRACE_PATH, FINAL_CHECKPOINT_TRACE_PATH),
    )
    return payload


def write_executor_ledger(output: Path) -> None:
    if output != PHASE_DIR / "executor-write-ledger.json":
        raise ContractError("executor ledger requires the canonical output path")
    current_head = _require_frozen_head()
    direct_paths = _executor_direct_paths()
    rows = [
        {
            "sequence": index,
            "path": path.as_posix(),
            "after_sha256": _sha256(path),
            "reason": "A3 reviewed oracle/statistics implementation or result",
        }
        for index, path in enumerate(direct_paths, start=1)
    ]
    traces = _base_trace_paths()
    for trace in traces:
        _audit_trace(trace)
    payload = {
        "schema_version": "bayesfilter.ssl_lstm_completion.phase_a3_executor_write_ledger.v1",
        "status": "A3_EXECUTOR_WRITE_LEDGER_VALID",
        "created_at_utc": _utc_now(),
        "git_commit": current_head,
        "rows": rows,
        "trace_bindings": [_trace_binding(path) for path in traces],
        "evidence_signature": "",
        "nonclaims": NONCLAIMS,
    }
    payload["evidence_signature"] = _signature(payload)
    _write_json(output, payload)


def write_post_result_ledger(output: Path) -> None:
    if output != PHASE_DIR / "post-result-write-ledger.json":
        raise ContractError("post-result ledger requires the canonical output path")
    current_head = _require_frozen_head()
    checkpoint_path = PHASE_DIR / "final-checkpoint.json"
    _verified_final_checkpoint(checkpoint_path)
    predecessor_traces = (*_base_trace_paths(), EXECUTOR_LEDGER_TRACE_PATH)
    _audit_trace(FINAL_CHECKPOINT_TRACE_PATH)
    trace_paths = (*predecessor_traces, FINAL_CHECKPOINT_TRACE_PATH)
    members = _post_result_direct_paths()
    payload = {
        "schema_version": "bayesfilter.ssl_lstm_completion.phase_a3_post_result_write_ledger.v1",
        "status": "A3_POST_RESULT_WRITE_LEDGER_VALID",
        "created_at_utc": _utc_now(),
        "git_commit": current_head,
        "rows": [_binding(path, "a3_post_result_direct_edit") for path in members],
        "final_checkpoint_binding": _binding(
            checkpoint_path, "immutable_a3_final_checkpoint"
        ),
        "trace_bindings": [_trace_binding(path) for path in trace_paths],
        "evidence_signature": "",
        "nonclaims": NONCLAIMS,
    }
    payload["evidence_signature"] = _signature(payload)
    _write_json(output, payload)


def write_final_checkpoint(output: Path) -> None:
    if output != PHASE_DIR / "final-checkpoint.json":
        raise ContractError("final checkpoint requires the canonical output path")
    current_head = _require_frozen_head()
    executor_path = PHASE_DIR / "executor-write-ledger.json"
    _verified_executor_ledger(executor_path)
    predecessor_traces = _base_trace_paths()
    _audit_trace(EXECUTOR_LEDGER_TRACE_PATH)
    trace_paths = (*predecessor_traces, EXECUTOR_LEDGER_TRACE_PATH)
    members = _checkpoint_member_paths()
    payload = {
        "schema_version": "bayesfilter.ssl_lstm_completion.phase_a3_final_checkpoint.v1",
        "status": "A3_PRE_RESULT_CHECKPOINT_PASSED",
        "created_at_utc": _utc_now(),
        "git_commit": current_head,
        "member_rows": [_binding(path, "a3_pre_result_member") for path in members],
        "trace_bindings": [_trace_binding(path) for path in trace_paths],
        "evidence_signature": "",
        "nonclaims": NONCLAIMS,
    }
    payload["evidence_signature"] = _signature(payload)
    _write_json(output, payload)


def close_phase(output: Path) -> None:
    if output != POST_RESULT_CLOSURE_PATH:
        raise ContractError("phase closure requires the canonical output path")
    current_head = _require_frozen_head()
    checkpoint = PHASE_DIR / "final-checkpoint.json"
    post_result_path = PHASE_DIR / "post-result-write-ledger.json"
    _verified_post_result_ledger(post_result_path)
    predecessor_traces = (
        *_base_trace_paths(), EXECUTOR_LEDGER_TRACE_PATH, FINAL_CHECKPOINT_TRACE_PATH
    )
    _audit_trace(POST_RESULT_LEDGER_TRACE_PATH)
    trace_paths = (*predecessor_traces, POST_RESULT_LEDGER_TRACE_PATH)
    members = _closure_member_paths()
    payload = {
        "schema_version": "bayesfilter.ssl_lstm_completion.phase_a3_post_result_closure.v1",
        "status": "A3_POST_RESULT_CLOSURE_PASSED",
        "created_at_utc": _utc_now(),
        "git_commit": current_head,
        "member_rows": [_binding(path, "a3_post_result_member") for path in members],
        "final_checkpoint_binding": _binding(checkpoint, "immutable_a3_final_checkpoint"),
        "trace_bindings": [_trace_binding(path) for path in trace_paths],
        "evidence_signature": "",
        "nonclaims": NONCLAIMS,
    }
    payload["evidence_signature"] = _signature(payload)
    _write_json(output, payload)


def _verified_closure_status(path: Path) -> dict[str, Any]:
    if path != POST_RESULT_CLOSURE_PATH:
        raise ContractError("closure verification requires the canonical closure path")
    payload = _verify_signed_artifact(
        path,
        schema="bayesfilter.ssl_lstm_completion.phase_a3_post_result_closure.v1",
        status="A3_POST_RESULT_CLOSURE_PASSED",
        expected_keys={
            "schema_version", "status", "created_at_utc", "git_commit",
            "member_rows", "final_checkpoint_binding", "trace_bindings",
            "evidence_signature", "nonclaims",
        },
    )
    _require_frozen_head(payload["git_commit"])
    _verify_binding_rows(
        payload["member_rows"],
        _closure_member_paths(),
        role="a3_post_result_member",
    )
    _verify_binding(
        payload["final_checkpoint_binding"],
        role="immutable_a3_final_checkpoint",
    )
    _verified_post_result_ledger(PHASE_DIR / "post-result-write-ledger.json")
    predecessor_traces = (
        *_base_trace_paths(),
        EXECUTOR_LEDGER_TRACE_PATH,
        FINAL_CHECKPOINT_TRACE_PATH,
        POST_RESULT_LEDGER_TRACE_PATH,
    )
    _verify_trace_chain(payload["trace_bindings"], predecessor_traces)
    closure_trace_sha256 = _audit_trace(CLOSURE_GENERATION_TRACE_PATH)
    return {
        "status": "A3_POST_RESULT_CLOSURE_VERIFIED",
        "closure_sha256": _sha256(path),
        "closure_generation_trace_sha256": closure_trace_sha256,
    }


def verify_closure(path: Path, log_path: Path | None) -> None:
    status = _verified_closure_status(path)
    if log_path is not None:
        if log_path != POST_RESULT_CLOSURE_RECEIPT_PATH:
            raise ContractError("closure verification requires the canonical receipt path")
        (ROOT / log_path).write_bytes(_canonical_bytes(status) + b"\n")
    print(_canonical_bytes(status).decode("utf-8"))


def audit_terminal_trace(path: Path) -> None:
    if path != CLOSURE_VERIFICATION_TRACE_PATH:
        raise ContractError("terminal audit requires the canonical closure-verification trace")
    expected_receipt = _verified_closure_status(POST_RESULT_CLOSURE_PATH)
    receipt = _load_json(POST_RESULT_CLOSURE_RECEIPT_PATH)
    if (
        (ROOT / POST_RESULT_CLOSURE_RECEIPT_PATH).read_bytes()
        != _canonical_bytes(receipt) + b"\n"
        or receipt != expected_receipt
    ):
        raise ContractError("terminal closure-verification receipt differs")
    trace_sha256 = _audit_trace(path, terminal=True)
    print(
        _canonical_bytes(
            {"status": "A3_TERMINAL_WRITE_TRACE_AUDIT_PASSED", "trace_sha256": trace_sha256}
        ).decode("utf-8")
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
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
    if args.artifact is not None:
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
        print(f"A3_CONTRACT_ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
