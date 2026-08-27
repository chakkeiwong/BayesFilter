"""Audit one q=20 pilot receipt against its declared finite measure ledger.

This is a CPU-hidden, TensorFlow-only diagnostic. It does not rerun the target
or prove an SMC-U theorem; it checks that the stored protocol, stage ledger,
terminal weights, and mode diagnostic describe the same finite computation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping


if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
    raise RuntimeError("measure audit requires CUDA_VISIBLE_DEVICES=-1")
if os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH") != "true":
    raise RuntimeError("measure audit requires TF_FORCE_GPU_ALLOW_GROWTH=true")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensorflow as tf

tf.config.set_visible_devices([], "GPU")
if tf.config.list_physical_devices("GPU"):
    raise RuntimeError("measure audit found a visible GPU")

from bayesfilter.testing.particle_authority_contracts_tf import canonical_protocol_hash


RUNNER = Path(__file__).resolve()
MODE_AXIS = 2
TOLERANCE = 1.0e-10


class AuditError(RuntimeError):
    """Raised when the stored measure contract cannot be checked."""


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if hasattr(value, "numpy"):
        return _jsonable(value.numpy())
    if hasattr(value, "tolist"):
        return _jsonable(value.tolist())
    if hasattr(value, "item"):
        return _jsonable(value.item())
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise AuditError(f"refusing to overwrite artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(_jsonable(payload), sort_keys=True, indent=2, allow_nan=False) + "\n").encode("ascii")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(encoded)
    temporary.replace(path)


def _load_tensor(path: Path, dtype: Any) -> Any:
    if not path.is_file():
        raise AuditError(f"missing tensor receipt: {path}")
    return tf.io.parse_tensor(tf.convert_to_tensor(path.read_bytes()), out_type=dtype)


def _close(value: Any, tolerance: float = TOLERANCE) -> bool:
    return bool(tf.reduce_all(tf.math.is_finite(tf.convert_to_tensor(value))).numpy()) and abs(float(value)) <= tolerance


def audit(pilot_root: Path) -> Mapping[str, Any]:
    started = time.perf_counter()
    pilot_path = pilot_root / "pilot.json"
    if not pilot_path.is_file():
        raise AuditError(f"missing pilot receipt: {pilot_path}")
    pilot = json.loads(pilot_path.read_text(encoding="utf-8"))
    arm = pilot.get("arms", {}).get("M0")
    if not isinstance(arm, Mapping) or arm.get("status") != "PASS_GATE":
        raise AuditError("M0 arm is not PASS_GATE")
    protocol = arm.get("protocol")
    if not isinstance(protocol, Mapping) or int(protocol.get("mode_axis", -1)) != MODE_AXIS:
        raise AuditError("protocol lacks the explicit mode_axis=2 binding")
    recomputed_hash = canonical_protocol_hash(protocol)
    protocol_hash_ok = recomputed_hash == arm["configuration"].get("protocol_hash")

    receipts = arm["receipts"]
    paths = {}
    for name in ("final_theta", "final_target_log_prob", "final_proposal_log_prob", "final_normalized_weights"):
        stored = Path(str(receipts[name]["path"]))
        candidates = (stored, ROOT / stored, pilot_root / stored.name)
        path = next((candidate for candidate in candidates if candidate.is_file()), None)
        if path is None:
            raise AuditError(f"missing receipt file for {name}: {stored}")
        paths[name] = path
    receipt_hashes = {}
    receipt_hash_ok = True
    for name, path in paths.items():
        digest = _sha(path)
        receipt_hashes[name] = digest
        receipt_hash_ok = receipt_hash_ok and digest == receipts[name]["sha256"]
    theta = _load_tensor(paths["final_theta"], tf.float64)
    target_log = _load_tensor(paths["final_target_log_prob"], tf.float64)
    proposal_log = _load_tensor(paths["final_proposal_log_prob"], tf.float64)
    weights = _load_tensor(paths["final_normalized_weights"], tf.float64)
    n = int(theta.shape[0])
    shape_ok = theta.shape == (n, 4) and target_log.shape == (n,) and proposal_log.shape == (n,) and weights.shape == (n,)
    finite_ok = all(
        bool(tf.reduce_all(tf.math.is_finite(value)).numpy())
        for value in (theta, target_log, proposal_log, weights)
    )
    weight_sum_residual = tf.reduce_sum(weights) - 1.0
    weights_nonnegative = bool(tf.reduce_all(weights >= 0.0).numpy())

    stage_paths = sorted(pilot_root.glob("m0-stage-*.json"))
    stages = [json.loads(path.read_text(encoding="utf-8")) for path in stage_paths]
    stage_indices = [int(stage.get("stage_index", -1)) for stage in stages]
    stage_order_ok = stage_indices == list(range(len(stages)))
    beta_values = [float(stage.get("beta", float("nan"))) for stage in stages]
    beta_ok = bool(beta_values) and all(left < right for left, right in zip([0.0] + beta_values[:-1], beta_values)) and abs(beta_values[-1] - 1.0) <= 1.0e-12
    log_normalizers = [float(stage["log_normalizer"]) for stage in stages]
    cumulative_residual = sum(log_normalizers[i] - (log_normalizers[i - 1] if i else 0.0) for i in range(len(log_normalizers))) - log_normalizers[-1]
    ledger_ok = bool(stages) and all(map(lambda value: value == value and abs(value) < float("inf"), log_normalizers)) and abs(cumulative_residual) <= TOLERANCE
    terminal_delta = float(stages[-1]["delta_beta"])
    expected_weights = tf.nn.softmax(terminal_delta * (target_log - proposal_log))
    terminal_weight_residual = tf.reduce_max(tf.abs(expected_weights - weights))
    mode_fraction = tf.reduce_sum(weights * tf.cast(theta[:, MODE_AXIS] < 0.0, tf.float64))
    declared_mode_fraction = float(arm["diagnostics"]["terminal_weighted_negative_mode_fraction"])
    mode_residual = mode_fraction - declared_mode_fraction
    mutation_receipts = arm["diagnostics"].get("mutation_receipts", [])
    acceptance_ok = True
    for receipt in mutation_receipts:
        proposal_count = float(receipt.get("proposal_count", float("nan")))
        accepted_count = float(receipt.get("accepted_count", float("nan")))
        rate = float(receipt.get("acceptance_rate", float("nan")))
        acceptance_ok = acceptance_ok and proposal_count == n and 0.0 <= accepted_count <= proposal_count and abs(rate - accepted_count / proposal_count) <= TOLERANCE

    gates = {
        "protocol_hash_ok": protocol_hash_ok,
        "mode_axis_bound": int(protocol["mode_axis"]) == MODE_AXIS,
        "receipt_hashes_ok": receipt_hash_ok,
        "shape_ok": shape_ok,
        "finite_ok": finite_ok,
        "weights_sum_ok": abs(float(weight_sum_residual.numpy())) <= TOLERANCE,
        "weights_nonnegative": weights_nonnegative,
        "stage_order_ok": stage_order_ok,
        "beta_schedule_ok": beta_ok,
        "mass_ledger_self_consistent": ledger_ok,
        "terminal_weights_match_last_increment": _close(terminal_weight_residual),
        "mode_diagnostic_matches_axis": _close(mode_residual),
        "acceptance_receipts_particle_level": acceptance_ok,
    }
    status = "PASS_MEASURE_AUDIT" if all(gates.values()) else "MEASURE_AUDIT_FAIL"
    return {
        "schema": "bayesfilter.ssl_lstm.q20.particle_authority.measure_audit.v1",
        "status": status,
        "pilot_root": pilot_root.as_posix(),
        "mode_axis": MODE_AXIS,
        "stage_count": len(stages),
        "particle_count": n,
        "gates": gates,
        "diagnostics": {
            "protocol_hash_recomputed": recomputed_hash,
            "weight_sum_residual": weight_sum_residual,
            "terminal_weight_residual": terminal_weight_residual,
            "mass_ledger_cumulative_residual": cumulative_residual,
            "mode_fraction_residual": mode_residual,
            "final_log_normalizer": log_normalizers[-1] if log_normalizers else None,
            "receipt_hashes": receipt_hashes,
        },
        "run_manifest": {
            "command": " ".join(sys.argv),
            "python": sys.executable,
            "python_version": platform.python_version(),
            "tensorflow": tf.__version__,
            "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
            "tf_force_gpu_allow_growth": os.environ["TF_FORCE_GPU_ALLOW_GROWTH"],
            "git_commit": subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=ROOT, text=True).strip(),
            "git_dirty": bool(subprocess.check_output(("git", "status", "--porcelain"), cwd=ROOT, text=True).strip()),
            "wall_seconds": time.perf_counter() - started,
        },
        "nonclaims": [
            "This checks the stored finite computation only; it does not prove SMC-U unbiasedness or q=20 mode discovery.",
            "A passing ledger audit does not establish IID Gaussian whitening, posterior correctness, HMC readiness, or default status.",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    if args.pilot_root.is_absolute() or args.output_root.is_absolute() or ".." in args.pilot_root.parts or ".." in args.output_root.parts:
        raise AuditError("paths must be repository-relative")
    pilot_root = ROOT / args.pilot_root
    output_root = ROOT / args.output_root
    if output_root.exists():
        raise AuditError(f"refusing to overwrite output root: {output_root}")
    output_root.mkdir(parents=True)
    try:
        result = audit(pilot_root)
    except Exception as exc:
        failure = {
            "schema": "bayesfilter.ssl_lstm.q20.particle_authority.measure_audit.failure.v1",
            "status": "MEASURE_AUDIT_FAIL",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "pilot_root": pilot_root.as_posix(),
            "run_manifest": {
                "command": " ".join(sys.argv),
                "python": sys.executable,
                "python_version": platform.python_version(),
                "tensorflow": tf.__version__,
                "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
                "tf_force_gpu_allow_growth": os.environ["TF_FORCE_GPU_ALLOW_GROWTH"],
            },
        }
        _write_json(output_root / "failure.json", failure)
        (output_root / "result.md").write_text(
            "# q=20 Measure Audit Failure\n\n"
            + f"Status: `{failure['status']}`\n\n"
            + f"Reason: `{failure['error']}`\n",
            encoding="ascii",
        )
        print(json.dumps({"status": failure["status"], "output_root": args.output_root.as_posix()}, sort_keys=True))
        return 2
    _write_json(output_root / "result.json", result)
    (output_root / "result.md").write_text(
        "# q=20 Measure Audit\n\n" + f"Status: `{result['status']}`\n\n" + "The audit checks finite receipt identities only; it is not an SMC-U or posterior proof.\n",
        encoding="ascii",
    )
    print(json.dumps({"status": result["status"], "output_root": args.output_root.as_posix()}, sort_keys=True))
    return 0 if result["status"] == "PASS_MEASURE_AUDIT" else 2


if __name__ == "__main__":
    raise SystemExit(main())
