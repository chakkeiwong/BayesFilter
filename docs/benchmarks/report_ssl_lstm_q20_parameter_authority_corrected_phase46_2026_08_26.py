"""Report a fixed-law finite support envelope for seven theta pilot banks.

This is a CPU-only, read-only diagnostic.  It recomputes the M0 proposal log
density from the retained geometry, checks the stored value, and reports raw
pairwise finite-bank support statistics.  It does not train a transport or
assign a density to an empirical cloud.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping


if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
    raise RuntimeError("Phase 46 report requires CUDA_VISIBLE_DEVICES=-1")
if os.environ.get("TF_FORCE_GPU_ALLOW_GROWTH") != "true":
    raise RuntimeError("Phase 46 report requires TF_FORCE_GPU_ALLOW_GROWTH=true")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PHASE28_PATH = ROOT / "docs/benchmarks/run_ssl_lstm_q20_parameter_authority_corrected_phase28_2026_08_25.py"
SPEC = importlib.util.spec_from_file_location("phase28_helpers_phase46", PHASE28_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError("cannot load the audited Phase 28 proposal helpers")
PHASE28 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PHASE28)
tf = PHASE28.tf

RUNNER = Path(__file__).resolve()
PLAN = ROOT / "docs/plans/bayesfilter-ssl-lstm-q20-parameter-authority-corrected-continuation-2026-08-25.md"
GEOMETRY = PHASE28.GEOMETRY
EXPECTED_TARGET = "9a86e60081f1b9cd288dbdb1dcbe1e9a5b5e23d9b5ef97afdb72ee95c23d7278"
EXPECTED_M0 = "a1f0f0493bb8bd594923b61ee9a92f3c8dcb72a612b64ad675b9ab7ff4723631"
EXPECTED_C0 = "270fc99b81d08e23670c62fcd02e69e7452f26b5e5641187c3083faecbac7067"
EXPECTED_VERSION = "v2.8-support-envelope-diagnostic"
EXPECTED_MEASURE = "theta_R4"
EXPECTED_SCHEMA = "bayesfilter.ssl_lstm.q20.corrected_theta_support_envelope_report.v1"
BANK_LABELS = ("authority", "bank_a", "bank_b", "bank_c", "bank_n512_a", "bank_n512_b", "bank_n512_c")
M0_EPSILON = 0.20
PROPOSAL_RECOMPUTE_TOLERANCE = 1.0e-10


class Phase46Error(RuntimeError):
    """Raised when a support-envelope receipt is not auditable."""


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_safe(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, tf.TensorShape):
        return [_safe(item) for item in value.as_list()]
    if isinstance(value, tf.dtypes.DType):
        return value.name
    if hasattr(value, "numpy"):
        return _safe(value.numpy())
    if hasattr(value, "tolist"):
        return _safe(value.tolist())
    if hasattr(value, "item"):
        return _safe(value.item())
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    if path.exists():
        raise Phase46Error(f"refusing to overwrite artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (json.dumps(_safe(payload), sort_keys=True, indent=2, allow_nan=False) + "\n").encode("ascii")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(encoded)
    temporary.replace(path)


def _load(path: Path) -> Mapping[str, Any]:
    if not path.is_file():
        raise Phase46Error(f"missing artifact: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_tensor(receipt: Mapping[str, Any]) -> tf.Tensor:
    path = Path(str(receipt["path"]))
    if not path.is_absolute():
        path = ROOT / path
    encoded = path.read_bytes()
    if hashlib.sha256(encoded).hexdigest() != str(receipt["sha256"]):
        raise Phase46Error(f"tensor hash mismatch: {path}")
    value = tf.io.parse_tensor(encoded, out_type=getattr(tf, str(receipt["dtype"])))
    value = tf.ensure_shape(value, receipt["shape"])
    if value.dtype.is_floating or value.dtype.is_complex:
        tf.debugging.assert_all_finite(value, f"non-finite tensor {path}")
    return value


def _root(path: Path) -> Mapping[str, Any]:
    if path.is_absolute() or ".." in path.parts:
        raise Phase46Error(f"all input paths must be repository-relative: {path}")
    return _load(ROOT / path / "pilot.json")


def _arm(pilot: Mapping[str, Any], arm_name: str, particles: int, calibration_particles: int, protocol: str) -> Mapping[str, Any]:
    if pilot.get("status") != "PASS_THETA_MEASURE_PILOT":
        raise Phase46Error("pilot status is not PASS_THETA_MEASURE_PILOT")
    if pilot.get("measure") != EXPECTED_MEASURE:
        raise Phase46Error("pilot measure mismatch")
    if int(pilot.get("calibration", {}).get("particle_count", -1)) != calibration_particles:
        raise Phase46Error("calibration particle count mismatch")
    arm = pilot.get("arms", {}).get(arm_name)
    if not isinstance(arm, Mapping) or arm.get("status") != "PASS_THETA_MEASURE_PILOT":
        raise Phase46Error(f"{arm_name} arm is not passing")
    if arm.get("target_signature") != EXPECTED_TARGET:
        raise Phase46Error(f"{arm_name} target signature mismatch")
    if arm.get("protocol", {}).get("measure") != EXPECTED_MEASURE:
        raise Phase46Error(f"{arm_name} protocol measure mismatch")
    if arm.get("configuration", {}).get("protocol_hash") != protocol:
        raise Phase46Error(f"{arm_name} protocol hash mismatch")
    if int(arm.get("configuration", {}).get("particles", -1)) != particles:
        raise Phase46Error(f"{arm_name} particle count mismatch")
    gates = arm.get("gates", {})
    for key in ("theta_shape_N_by_4", "target_status_valid", "density_terms_finite", "finite_log_mass"):
        if gates.get(key) is not True:
            raise Phase46Error(f"{arm_name} gate {key} is not true")
    return arm


def _cloud(arm: Mapping[str, Any]) -> Mapping[str, tf.Tensor]:
    receipts = arm["receipts"]
    values = {key: _load_tensor(receipts[key]) for key in ("final_theta", "final_normalized_weights", "final_roots", "final_target_log_theta", "final_proposal_log_theta")}
    n = int(values["final_theta"].shape[0])
    expected = {
        "final_theta": (n, 4),
        "final_normalized_weights": (n,),
        "final_roots": (n,),
        "final_target_log_theta": (n,),
        "final_proposal_log_theta": (n,),
    }
    for key, shape in expected.items():
        if values[key].shape != shape:
            raise Phase46Error(f"{key} shape mismatch: {values[key].shape} != {shape}")
    weights = tf.cast(values["final_normalized_weights"], tf.float64)
    weights = tf.maximum(weights, tf.constant(1.0e-300, tf.float64))
    weights = weights / tf.reduce_sum(weights)
    return {"theta": tf.cast(values["final_theta"], tf.float64), "weights": weights, "roots": values["final_roots"], "target": tf.cast(values["final_target_log_theta"], tf.float64), "proposal": tf.cast(values["final_proposal_log_theta"], tf.float64)}


def _summary(cloud: Mapping[str, tf.Tensor], proposal: tf.Tensor) -> Mapping[str, Any]:
    theta = cloud["theta"]
    weights = cloud["weights"]
    target = cloud["target"]
    stored = cloud["proposal"]
    ratio = target - stored
    mean = tf.reduce_sum(weights[:, None] * theta, axis=0)
    coordinate_min = tf.reduce_min(theta, axis=0)
    coordinate_max = tf.reduce_max(theta, axis=0)
    roots = tf.unique(cloud["roots"]).y
    sign = theta[:, PHASE28.MODE_AXIS] < 0.0
    recomputed_residual = tf.reduce_max(tf.abs(proposal - stored))
    fields = {
        "ess_fraction": tf.constant(1.0, tf.float64) / tf.reduce_sum(tf.square(weights)) / tf.cast(tf.shape(theta)[0], tf.float64),
        "negative_mode_fraction": tf.reduce_sum(weights * tf.cast(sign, tf.float64)),
        "theta_mean_0": mean[0],
        "root_count": tf.cast(tf.size(roots), tf.float64),
        "target_log_min": tf.reduce_min(target),
        "target_log_max": tf.reduce_max(target),
        "proposal_log_min": tf.reduce_min(stored),
        "proposal_log_max": tf.reduce_max(stored),
        "log_ratio_min": tf.reduce_min(ratio),
        "log_ratio_max": tf.reduce_max(ratio),
    }
    return {
        "rows": tf.shape(theta)[0],
        "roots": tf.size(roots),
        "ess_fraction": fields["ess_fraction"],
        "negative_mode_fraction": fields["negative_mode_fraction"],
        "theta_mean": mean,
        "coordinate_min": coordinate_min,
        "coordinate_max": coordinate_max,
        "target_log_range": (tf.reduce_min(target), tf.reduce_max(target)),
        "proposal_log_range": (tf.reduce_min(stored), tf.reduce_max(stored)),
        "log_ratio_range": (tf.reduce_min(ratio), tf.reduce_max(ratio)),
        "recomputed_proposal_max_abs_residual": recomputed_residual,
        "scalar_fields": fields,
    }


def _pairwise(left: Mapping[str, tf.Tensor], right: Mapping[str, tf.Tensor]) -> Mapping[str, Any]:
    x = left["theta"]
    y = right["theta"]
    lower_x = tf.reduce_min(x, axis=0)
    upper_x = tf.reduce_max(x, axis=0)
    lower_y = tf.reduce_min(y, axis=0)
    upper_y = tf.reduce_max(y, axis=0)
    inside = tf.reduce_all((x >= lower_y[None, :]) & (x <= upper_y[None, :]), axis=1)
    weighted_inside = tf.reduce_sum(left["weights"] * tf.cast(inside, tf.float64))
    intersection_width = tf.maximum(tf.minimum(upper_x, upper_y) - tf.maximum(lower_x, lower_y), 0.0)
    union_width = tf.maximum(tf.maximum(upper_x, upper_y) - tf.minimum(lower_x, lower_y), 0.0)
    intersection_volume = tf.reduce_prod(intersection_width)
    union_volume = tf.reduce_prod(union_width)
    volume_ratio = tf.where(union_volume > 0.0, intersection_volume / union_volume, tf.constant(0.0, tf.float64))
    distances = tf.sqrt(tf.reduce_sum(tf.square(x[:, None, :] - y[None, :, :]), axis=2))
    nearest = tf.reduce_min(distances, axis=1)
    return {
        "weighted_fraction_left_inside_right_box": weighted_inside,
        "unweighted_count_left_inside_right_box": tf.reduce_sum(tf.cast(inside, tf.int32)),
        "coordinate_box_intersection_volume": intersection_volume,
        "coordinate_box_union_volume": union_volume,
        "coordinate_box_intersection_over_union": volume_ratio,
        "weighted_nearest_neighbor_mean": tf.reduce_sum(left["weights"] * nearest),
        "nearest_neighbor_min": tf.reduce_min(nearest),
        "nearest_neighbor_max": tf.reduce_max(nearest),
        "left_box_min": lower_x,
        "left_box_max": upper_x,
        "right_box_min": lower_y,
        "right_box_max": upper_y,
    }


def _scalar_envelope(summaries: Mapping[str, Mapping[str, Any]]) -> Mapping[str, Any]:
    fields = tuple(summaries["bank_n512_a"]["scalar_fields"].keys())
    checks: dict[str, Any] = {}
    outside: list[str] = []
    for field in fields:
        a = float(summaries["bank_n512_a"]["scalar_fields"][field].numpy())
        b = float(summaries["bank_n512_b"]["scalar_fields"][field].numpy())
        c = float(summaries["bank_n512_c"]["scalar_fields"][field].numpy())
        low, high = min(a, b), max(a, b)
        inside = low <= c <= high
        checks[field] = {"n512_a": a, "n512_b": b, "n512_c": c, "low": low, "high": high, "inside": inside}
        if not inside:
            outside.append(field)
    return {"fields": checks, "outside_fields": outside, "n512_c_inside_all_scalar_fields": not outside}


def _markdown(result: Mapping[str, Any]) -> str:
    lines = ["# v2.8 Fixed-Law Theta Support Envelope", "", f"Status: `{result['status']}`", f"Branch: `{result['branch']}` (descriptive only)", "", "| Source | rows | roots | ESS | neg-mode | theta mean[0] | proposal residual |", "|---|---:|---:|---:|---:|---:|---:|"]
    for row in result["support_rows"]:
        lines.append(f"| {row['source']} | {row['rows']} | {row['roots']} | {float(row['ess_fraction']):.6f} | {float(row['negative_mode_fraction']):.6f} | {float(row['theta_mean_0']):.6f} | {float(row['recomputed_proposal_max_abs_residual']):.3e} |")
    lines.extend(["", "The envelope and pairwise metrics are finite empirical diagnostics. They do not establish common support, IID whitening, posterior correctness, or mode discovery.", ""])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (*BANK_LABELS, "output-root"):
        parser.add_argument(f"--{name.replace('_', '-')}", required=True, type=Path)
    args = parser.parse_args()
    paths = [getattr(args, name.replace("-", "_")) for name in (*BANK_LABELS, "output-root")]
    if any(path.is_absolute() or ".." in path.parts for path in paths):
        raise Phase46Error("all paths must be repository-relative")
    output = ROOT / args.output_root
    if output.exists():
        raise Phase46Error(f"refusing to overwrite output root: {output}")
    started = time.perf_counter()
    root_map = {label: getattr(args, label.replace("-", "_")) for label in BANK_LABELS}
    pilots = {label: _root(path) for label, path in root_map.items()}
    pilot_paths = {label: ROOT / path / "pilot.json" for label, path in root_map.items()}
    if len({_sha(path) for path in pilot_paths.values()}) != len(BANK_LABELS):
        raise Phase46Error("pilot receipts are not distinct")
    counts = {label: (512, 128) if label.startswith("bank_n512") else (256, 64) for label in BANK_LABELS}
    arms = {label: _arm(pilots[label], "M0", counts[label][0], counts[label][1], EXPECTED_M0) for label in BANK_LABELS}
    for label in BANK_LABELS:
        _arm(pilots[label], "C0", counts[label][0], counts[label][1], EXPECTED_C0)
    geometry = PHASE28._load_geometry()
    clouds: dict[str, Mapping[str, tf.Tensor]] = {}
    summaries: dict[str, Mapping[str, Any]] = {}
    for label in BANK_LABELS:
        cloud = _cloud(arms[label])
        recomputed = PHASE28._proposal_log_theta(cloud["theta"], geometry, M0_EPSILON)
        clouds[label] = cloud
        summaries[label] = _summary(cloud, recomputed)
        if float(summaries[label]["recomputed_proposal_max_abs_residual"].numpy()) > PROPOSAL_RECOMPUTE_TOLERANCE:
            raise Phase46Error(f"stored proposal mismatch for {label}")
    support_rows = []
    for label in BANK_LABELS:
        item = summaries[label]
        support_rows.append({"source": label, "rows": item["rows"], "roots": item["roots"], "ess_fraction": item["ess_fraction"], "negative_mode_fraction": item["negative_mode_fraction"], "theta_mean": item["theta_mean"], "theta_mean_0": item["theta_mean"][0], "coordinate_min": item["coordinate_min"], "coordinate_max": item["coordinate_max"], "target_log_range": item["target_log_range"], "proposal_log_range": item["proposal_log_range"], "log_ratio_range": item["log_ratio_range"], "recomputed_proposal_max_abs_residual": item["recomputed_proposal_max_abs_residual"], "finite": True})
    pairwise = {}
    for left in BANK_LABELS:
        for right in BANK_LABELS:
            if left == right:
                continue
            pairwise[f"{left}__to__{right}"] = _pairwise(clouds[left], clouds[right])
    envelope = _scalar_envelope(summaries)
    branch = "n512_c_inside_two_bank_scalar_envelope" if envelope["n512_c_inside_all_scalar_fields"] else "n512_c_outside_two_bank_scalar_envelope"
    result = {
        "schema": EXPECTED_SCHEMA,
        "status": "PASS_V2_8_SUPPORT_ENVELOPE_REPORT",
        "plan_version": EXPECTED_VERSION,
        "role": "read_only_fixed_theta_proposal_support_envelope",
        "measure": EXPECTED_MEASURE,
        "target_signature": EXPECTED_TARGET,
        "branch": branch,
        "branch_is_statistical_ranking": False,
        "support_rows": support_rows,
        "scalar_envelope": envelope,
        "pairwise": pairwise,
        "proposal_recompute_tolerance": PROPOSAL_RECOMPUTE_TOLERANCE,
        "decision_table": [
            {"decision": "retain_theta_target", "status": "pass", "primary_criterion": "pilot/hash/measure/finite gates", "veto": "none", "next_action": "retain theta authority", "not_concluded": "posterior correctness"},
            {"decision": "promote_IID_whitening", "status": "veto", "primary_criterion": "finite support envelope", "veto": "finite empirical envelope is not a population law", "next_action": "keep whitening closed", "not_concluded": "IID Gaussian law"},
            {"decision": "change_objective", "status": "defer", "primary_criterion": "support diagnostic", "veto": "no uncertainty-supported downstream comparison", "next_action": "write a new objective plan only after independent validation", "not_concluded": "objective superiority"},
        ],
        "inference_status": {"hard_veto_screen": "passed", "statistically_supported_ranking": "none", "descriptive_differences": "raw finite-bank envelope and pairwise metrics", "default_readiness": "not_ready", "next_evidence": "proposal/support repair or a separately governed objective experiment"},
        "red_team": {"strongest_alternative": "coordinate boxes overstate support and both N512 banks share proposal bias", "overturning_evidence": "a separately generated proposal/support route with downstream validation", "weakest_evidence": "finite boxes and nearest-neighbor summaries in only four dimensions"},
        "nonclaims": ["No common-support or population theorem.", "No IID Gaussian whitening, posterior correctness, exhaustive mode discovery, normalizer, HMC, canonical LEDH, superiority, or default-readiness claim.", "Pairwise metrics are descriptive and no bank is pooled, selected, or used for training."],
        "sources": {"geometry": GEOMETRY, "geometry_sha256": _sha(GEOMETRY), **{f"{label}_root": root_map[label] for label in BANK_LABELS}, **{f"{label}_pilot_sha256": _sha(pilot_paths[label]) for label in BANK_LABELS}},
        "run_manifest": {"program": PLAN, "runner": RUNNER, "command": " ".join(sys.argv), "git_commit": subprocess.check_output(("git", "rev-parse", "HEAD"), cwd=ROOT, text=True).strip(), "git_dirty": bool(subprocess.check_output(("git", "status", "--porcelain"), cwd=ROOT, text=True).strip()), "python": sys.executable, "python_version": platform.python_version(), "tensorflow": tf.__version__, "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"], "tf_force_gpu_allow_growth": os.environ["TF_FORCE_GPU_ALLOW_GROWTH"], "gpu_hidden_intentionally": True, "jit_compile": False, "wall_seconds": time.perf_counter() - started, "source_sha256": {"plan": _sha(PLAN), "runner": _sha(RUNNER), "phase28_helpers": _sha(PHASE28_PATH)}}
    }
    _write_json(output / "result.json", result)
    (output / "result.md").write_text(_markdown(result), encoding="ascii")
    print(json.dumps({"status": result["status"], "branch": branch, "output_root": args.output_root.as_posix()}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
