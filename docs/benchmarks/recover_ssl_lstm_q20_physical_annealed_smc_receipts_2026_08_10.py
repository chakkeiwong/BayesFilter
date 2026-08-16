#!/usr/bin/env python3
"""Recover and validate historical SMC tensor receipts without target calls."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping


os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

ROOT = Path(__file__).resolve().parents[2]
ARTIFACT_ROOT = Path(
    "docs/plans/artifacts/"
    "ssl-lstm-q20-physical-annealed-smc-repair-2026-08-10/r2"
)
MATERIAL = ARTIFACT_ROOT / "material.json"
RECOVERY = ARTIFACT_ROOT / "receipt-recovery-v1.json"
PARTICLES = 100

INITIAL_NAMES = {
    "component_labels",
    "is_accepted",
    "log_accept_ratio",
    "proposal_log_prob",
    "proposed_finite",
    "root_signs",
    "roots",
    "sign",
    "status_code",
    "target_log_prob",
    "theta",
    "valid",
    "z",
}
PRE_NAMES = {
    "log_weights",
    "normalized_weights",
    "proposal_log_prob",
    "root_signs",
    "roots",
    "sign",
    "target_log_prob",
    "theta",
    "z",
}
POST_NAMES = {
    "is_accepted",
    "log_accept_ratio",
    "parents",
    "proposal_log_prob",
    "proposed_finite",
    "resampled_signs",
    "roots",
    "sign",
    "status_code",
    "target_log_prob",
    "theta",
    "valid",
    "z",
}

FLOAT_NAMES = {
    "log_accept_ratio",
    "log_weights",
    "normalized_weights",
    "proposal_log_prob",
    "target_log_prob",
    "theta",
    "z",
}
INT_NAMES = {"component_labels", "parents", "roots", "status_code"}
BOOL_NAMES = {
    "is_accepted",
    "proposed_finite",
    "resampled_signs",
    "root_signs",
    "sign",
    "valid",
}


class ReceiptRecoveryError(RuntimeError):
    """Raised when immutable SMC evidence cannot be recovered exactly."""


def _abs(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _sha(path: Path) -> str:
    return hashlib.sha256(_abs(path).read_bytes()).hexdigest()


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    absolute = _abs(path)
    absolute.parent.mkdir(parents=True, exist_ok=True)
    if absolute.exists():
        raise ReceiptRecoveryError(f"refusing to overwrite recovery artifact: {path}")
    encoded = json.dumps(payload, sort_keys=True, indent=2, allow_nan=False) + "\n"
    temporary = absolute.with_suffix(absolute.suffix + ".tmp")
    temporary.write_text(encoded, encoding="ascii")
    temporary.replace(absolute)


def _logical_name(path: Path) -> str:
    stem = path.name.removesuffix(".tftensor")
    for marker in ("-pre-", "-post-"):
        if marker in stem:
            return stem.split(marker, maxsplit=1)[1]
    if stem.startswith("initial-"):
        return stem.removeprefix("initial-")
    if stem == "central-estimates" or stem == "sensitivity-estimates":
        return stem
    raise ReceiptRecoveryError(f"unrecognized tensor filename: {path}")


def _expected_dtype(name: str, tf: Any) -> Any:
    if name in FLOAT_NAMES or name in {"central-estimates", "sensitivity-estimates"}:
        return tf.float64
    if name in INT_NAMES:
        return tf.int32
    if name in BOOL_NAMES:
        return tf.bool
    raise ReceiptRecoveryError(f"no expected dtype for tensor: {name}")


def _expected_shape(name: str) -> list[int]:
    if name in {"theta", "z"}:
        return [PARTICLES, 4]
    if name == "central-estimates":
        return [8]
    if name == "sensitivity-estimates":
        return [2]
    return [PARTICLES]


def _load_tensor(path: Path, tf: Any) -> Any:
    name = _logical_name(path)
    tensor = tf.io.parse_tensor(
        tf.io.read_file(str(_abs(path))), out_type=_expected_dtype(name, tf)
    )
    return tf.ensure_shape(tensor, _expected_shape(name))


def _record(path: Path, tensor: Any) -> Mapping[str, Any]:
    absolute = _abs(path)
    return {
        "path": path.as_posix(),
        "sha256": _sha(path),
        "bytes": absolute.stat().st_size,
        "dtype": tensor.dtype.name,
        "shape": list(tensor.shape),
    }


def _flatten_receipts(receipts: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    if "pre" in receipts or "post" in receipts:
        return [
            receipt
            for group in (receipts.get("pre", {}), receipts.get("post", {}))
            for receipt in group.values()
        ]
    return list(receipts.values())


def _assert_equal(left: Any, right: Any, message: str, tf: Any) -> None:
    try:
        tf.debugging.assert_equal(left, right, message=message)
    except BaseException as error:
        raise ReceiptRecoveryError(message) from error


def _assert_near(left: Any, right: Any, message: str, tf: Any) -> None:
    try:
        tf.debugging.assert_near(left, right, atol=1.0e-10, rtol=1.0e-10, message=message)
    except BaseException as error:
        raise ReceiptRecoveryError(message) from error


def _child_tensor(root: Path, prefix: str, name: str, tf: Any) -> Any:
    return _load_tensor(root / f"{prefix}-{name}.tftensor", tf)


def _validate_child(root: Path, tf: Any) -> Mapping[str, Any]:
    terminal = json.loads(_abs(root / "canary.json").read_text(encoding="utf-8"))
    stage_count = int(terminal["stage_count"])
    initial = {
        name: _child_tensor(root, "initial", name, tf) for name in INITIAL_NAMES
    }
    _assert_equal(initial["roots"], tf.range(PARTICLES, dtype=tf.int32), "initial roots", tf)
    _assert_equal(initial["root_signs"], initial["sign"], "initial root signs", tf)
    _assert_equal(initial["valid"], tf.ones(PARTICLES, tf.bool), "initial validity", tf)
    if not bool(
        tf.reduce_all(
            tf.math.is_finite(initial["target_log_prob"])
            & tf.math.is_finite(initial["proposal_log_prob"])
        ).numpy()
    ):
        raise ReceiptRecoveryError(f"non-finite initial target in {root}")

    prior = initial
    total_sign_changes = 0
    stage_summaries = []
    for stage_index in range(stage_count):
        prefix = f"stage-{stage_index:02d}"
        stage = json.loads(_abs(root / f"{prefix}.json").read_text(encoding="utf-8"))
        pre = {name: _child_tensor(root, f"{prefix}-pre", name, tf) for name in PRE_NAMES}
        for name in ("z", "theta", "proposal_log_prob", "target_log_prob", "sign", "roots"):
            _assert_equal(pre[name], prior[name], f"{root.name} {prefix} pre continuity {name}", tf)
        _assert_equal(
            pre["root_signs"],
            tf.gather(initial["root_signs"], pre["roots"]),
            f"{root.name} {prefix} root-sign ancestry",
            tf,
        )
        _assert_near(
            pre["normalized_weights"],
            tf.nn.softmax(pre["log_weights"]),
            f"{root.name} {prefix} normalized weights",
            tf,
        )
        _assert_near(
            tf.reduce_sum(pre["normalized_weights"]),
            tf.constant(1.0, tf.float64),
            f"{root.name} {prefix} weight sum",
            tf,
        )
        json_ess = float(stage["pre_resampling_ess_fraction"])
        tensor_ess = tf.math.reciprocal(
            tf.reduce_sum(tf.square(pre["normalized_weights"]))
        ) / tf.constant(float(PARTICLES), tf.float64)
        _assert_near(tensor_ess, tf.constant(json_ess, tf.float64), f"{root.name} {prefix} ESS", tf)
        _assert_near(
            tf.reduce_max(pre["normalized_weights"]),
            tf.constant(float(stage["pre_resampling_maximum_weight"]), tf.float64),
            f"{root.name} {prefix} maximum weight",
            tf,
        )

        if bool(stage["resampled"]):
            post = {
                name: _child_tensor(root, f"{prefix}-post", name, tf)
                for name in POST_NAMES
            }
            parents = post["parents"]
            if not bool(
                tf.reduce_all((parents >= 0) & (parents < PARTICLES)).numpy()
            ):
                raise ReceiptRecoveryError(f"{root.name} {prefix} parent bounds")
            _assert_equal(post["roots"], tf.gather(pre["roots"], parents), f"{root.name} {prefix} roots", tf)
            _assert_equal(post["resampled_signs"], tf.gather(pre["sign"], parents), f"{root.name} {prefix} resampled signs", tf)
            _assert_equal(post["valid"], tf.ones(PARTICLES, tf.bool), f"{root.name} {prefix} validity", tf)
            if not bool(
                tf.reduce_all(
                    tf.math.is_finite(post["target_log_prob"])
                    & tf.math.is_finite(post["proposal_log_prob"])
                    & tf.math.is_finite(post["log_accept_ratio"])
                ).numpy()
            ):
                raise ReceiptRecoveryError(f"non-finite retained post tensor in {root.name} {prefix}")
            sign_changes = int(
                tf.reduce_sum(tf.cast(post["sign"] != post["resampled_signs"], tf.int32)).numpy()
            )
            if sign_changes != int(stage["hmc_sign_changes"]):
                raise ReceiptRecoveryError(f"{root.name} {prefix} HMC sign-change mismatch")
            total_sign_changes += sign_changes
            prior = post
        else:
            if stage_index != stage_count - 1 or not bool(stage["terminal_pre_resampling"]):
                raise ReceiptRecoveryError(f"{root.name} nonterminal stage lacks resampling")
            prior = pre
        stage_summaries.append(
            {
                "stage_index": stage_index,
                "resampled": bool(stage["resampled"]),
                "pre_receipt_count": len(PRE_NAMES),
                "post_receipt_count": len(POST_NAMES) if bool(stage["resampled"]) else 0,
                "ess_fraction": float(tensor_ess.numpy()),
            }
        )

    if total_sign_changes != int(terminal["total_hmc_sign_changes"]):
        raise ReceiptRecoveryError(f"{root.name} total HMC sign-change mismatch")
    terminal_weights = pre["normalized_weights"]
    terminal_mass = tf.reduce_sum(terminal_weights * tf.cast(pre["sign"], tf.float64))
    return {
        "name": root.name,
        "stage_count": stage_count,
        "nonterminal_stage_count": stage_count - 1,
        "terminal_negative_region_probability": float(terminal_mass.numpy()),
        "total_hmc_sign_changes": total_sign_changes,
        "stages": stage_summaries,
    }


def recover() -> Mapping[str, Any]:
    import tensorflow as tf

    tf.config.set_visible_devices([], "GPU")
    if tf.config.list_physical_devices("GPU"):
        raise ReceiptRecoveryError("receipt recovery must remain CPU-only")
    material = json.loads(_abs(MATERIAL).read_text(encoding="utf-8"))
    child_roots = sorted(path for path in _abs(ARTIFACT_ROOT).iterdir() if path.is_dir())
    if len(child_roots) != 10:
        raise ReceiptRecoveryError("expected exactly ten material child directories")

    all_paths = sorted(_abs(ARTIFACT_ROOT).rglob("*.tftensor"))
    records = []
    for absolute in all_paths:
        relative = absolute.relative_to(ROOT)
        tensor = _load_tensor(relative, tf)
        records.append(_record(relative, tensor))

    linked: dict[str, Mapping[str, Any]] = {}
    for child in child_roots:
        terminal = json.loads((child / "canary.json").read_text(encoding="utf-8"))
        for receipt in terminal["initial_receipts"].values():
            linked[str(receipt["path"])] = receipt
        for stage_index in range(int(terminal["stage_count"])):
            stage = json.loads((child / f"stage-{stage_index:02d}.json").read_text(encoding="utf-8"))
            for receipt in _flatten_receipts(stage["receipts"]):
                linked[str(receipt["path"])] = receipt
    for receipt in material["aggregate_receipts"].values():
        linked[str(receipt["path"])] = receipt
    for path, receipt in linked.items():
        if _sha(Path(path)) != str(receipt["sha256"]):
            raise ReceiptRecoveryError(f"original manifest receipt mismatch: {path}")

    children = [_validate_child(child.relative_to(ROOT), tf) for child in child_roots]
    central = tf.constant(
        [row["terminal_negative_region_probability"] for row in children[:8]], tf.float64
    )
    sensitivity = tf.constant(
        [row["terminal_negative_region_probability"] for row in children[8:]], tf.float64
    )
    _assert_near(
        central,
        tf.constant(material["central"]["batch_estimates"], tf.float64),
        "recovered central mass estimates",
        tf,
    )
    _assert_near(
        sensitivity,
        tf.constant(material["sensitivity"]["batch_estimates"], tf.float64),
        "recovered sensitivity mass estimates",
        tf,
    )
    _assert_equal(
        _load_tensor(ARTIFACT_ROOT / "central-estimates.tftensor", tf),
        central,
        "central aggregate receipt",
        tf,
    )
    _assert_equal(
        _load_tensor(ARTIFACT_ROOT / "sensitivity-estimates.tftensor", tf),
        sensitivity,
        "sensitivity aggregate receipt",
        tf,
    )

    all_relative = {path.relative_to(ROOT).as_posix() for path in all_paths}
    linked_paths = set(linked)
    recovered_paths = sorted(all_relative - linked_paths)
    expected_recovered = sorted(
        path.relative_to(ROOT).as_posix()
        for path in all_paths
        if "-pre-" in path.name
        and _logical_name(path) in (PRE_NAMES & POST_NAMES)
        and path.relative_to(ROOT).as_posix() not in linked_paths
    )
    counts = {
        "original_manifest_linked_child_receipts": len(linked_paths) - 2,
        "original_manifest_linked_aggregate_receipts": 2,
        "recovered_unlinked_child_receipts": len(recovered_paths),
        "verified_child_tensor_files": len(all_paths) - 2,
        "verified_aggregate_tensor_files": 2,
        "verified_total_tensor_files": len(all_paths),
    }
    gates = {
        "all_original_manifest_hashes_match": True,
        "all_tensor_files_parse_with_expected_dtype_and_shape": True,
        "all_stage_weight_and_ancestry_invariants_hold": True,
        "all_stage_continuity_invariants_hold": True,
        "terminal_estimates_reproduce_material_artifact": True,
        "exactly_210_overwritten_pre_receipts_recovered": (
            len(recovered_paths) == 210 and recovered_paths == expected_recovered
        ),
        "expected_990_child_and_2_aggregate_tensors_verified": len(all_paths) == 992,
    }
    payload = {
        "schema": "bayesfilter.ssl_lstm.q20_physical_annealed_smc.receipt_recovery.v1",
        "status": "SMC_RECEIPT_RECOVERY_PASSED" if all(gates.values()) else "SMC_RECEIPT_RECOVERY_FAILED",
        "role": "post_run_provenance_recovery_without_target_reevaluation",
        "artifact_root": ARTIFACT_ROOT.as_posix(),
        "material_sha256": _sha(MATERIAL),
        "counts": counts,
        "gates": gates,
        "children": children,
        "recovered_unlinked_paths": recovered_paths,
        "tensor_inventory": records,
        "nonclaims": [
            "receipt recovery does not add samples or reevaluate the target",
            "receipt recovery does not expand two-known-region SMC authority",
            "receipt recovery does not establish global HMC travel or full-posterior coverage",
        ],
    }
    if not all(gates.values()):
        raise ReceiptRecoveryError(f"receipt recovery gates failed: {gates}")
    _write_json(RECOVERY, payload)
    return payload


if __name__ == "__main__":
    result = recover()
    print(json.dumps({"status": result["status"], "counts": result["counts"]}, sort_keys=True))
