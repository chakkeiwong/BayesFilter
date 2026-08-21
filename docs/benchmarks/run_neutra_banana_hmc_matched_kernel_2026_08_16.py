#!/usr/bin/env python3
"""Run the no-retuning banana learned-transport bank-by-kernel cross-over."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PLAN = ROOT / "docs/plans/bayesfilter-neutra-banana-hmc-repair-plan-2026-08-16.md"
DISCOVERY = ROOT / "docs/plans/artifacts/neutra-banana-hmc-repair-2026-08-16-r3"
REPLICATION_RUNNER = ROOT / "docs/benchmarks/run_neutra_replication_hmc_campaign_2026_08_16.py"
REPAIR_RUNNER = ROOT / "docs/benchmarks/run_neutra_banana_repair_2026_08_16.py"
DEFAULT_OUTPUT = ROOT / "docs/plans/artifacts/neutra-banana-hmc-matched-kernel-2026-08-16-r1"


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="0")
    parser.add_argument("--time-cap", type=float, default=1800.0)
    return parser.parse_args()


def _load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_safe(v) for v in value]
    if hasattr(value, "numpy"):
        return _safe(value.numpy().tolist())
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return value.as_posix()
    return value


def _write(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_safe(payload), sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _read(path: Path) -> Mapping[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError(f"JSON object required: {path}")
    return value


def _kernel(arm: str) -> Mapping[str, Any]:
    tuning = _read(DISCOVERY / arm / "tuning/tuning_result.json")
    kernel = tuning.get("final_kernel_payload")
    if tuning.get("passed") is not True or not isinstance(kernel, Mapping):
        raise RuntimeError(f"discovery kernel is not admissible: {arm}")
    return kernel


def _central_bank(tf_module: Any) -> Any:
    rows = tf_module.zeros((4, 16), tf_module.float64)
    offsets = tf_module.constant((0.0, 0.25, -0.25, 0.25), tf_module.float64)
    axes = tf_module.constant((0, 0, 0, 1), tf_module.int32)
    return tf_module.tensor_scatter_nd_update(rows, tf_module.stack((tf_module.range(4), axes), axis=1), offsets)


def main() -> int:
    args = _args()
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(f"output root must be fresh: {output}")
    if not PLAN.is_file() or not DISCOVERY.is_dir():
        raise FileNotFoundError("reviewed plan or discovery root is missing")
    output.mkdir(parents=True)
    os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device)
    started = time.perf_counter()
    import tensorflow as tf
    from bayesfilter.runtime.gpu_memory_policy import configure_tensorflow_gpu_memory_growth

    memory_policy = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)
    tf.config.set_soft_device_placement(False)
    tf.config.experimental.enable_tensor_float_32_execution(False)
    logical = tuple(tf.config.list_logical_devices("GPU"))
    if len(logical) != 1:
        raise RuntimeError(f"expected one visible GPU, found {logical}")
    replication = _load(REPLICATION_RUNNER, "banana_matched_replication")
    repair = _load(REPAIR_RUNNER, "banana_matched_training")
    base = replication._load_base()
    model = base._model(tf, "banana")
    base_adapter = replication.AnalyticControlAdapter(tf, model, "analytic_control:banana:base_v1")
    learned, training = repair._train(tf, model, seed=15, updates=6000)
    state_hash = replication._state_hash(tf, learned)
    replication._bind_frozen(learned, tf, model, state_hash)
    discovery_training = _read(DISCOVERY / "learned_training.json")
    if state_hash != str(discovery_training["state_hash"]):
        raise RuntimeError("learned transport replay hash does not match discovery")
    audit = replication._audit(tf, learned, model, count=131072, seed=(20260816, 59015))
    if not bool(tf.convert_to_tensor(audit["passed"]).numpy()):
        raise RuntimeError("learned transport replay failed exact discovery audit")

    original = replication._initial_bank(tf, 16)
    central = _central_bank(tf)
    kernel_a = _kernel("learned_original_bank")
    kernel_b = _kernel("learned_central_bank")
    cells = (
        ("original_bank_central_kernel", original, kernel_b),
        ("central_bank_original_kernel", central, kernel_a),
    )
    rows: dict[str, Any] = {}
    progress = {"schema": "bayesfilter.neutra.banana_hmc_matched_kernel_progress.v1", "phase": "started", "completed_cells": 0}
    _write(output / "progress.json", progress)
    for name, bank, kernel in cells:
        if time.perf_counter() - started > float(args.time_cap):
            raise TimeoutError("matched-kernel campaign time cap exhausted")
        cell_root = output / name
        hmc = replication._run_hmc(tf, model, learned, base_adapter, bank, {"passed": True, "final_kernel_payload": kernel}, cell_root / "hmc", name, max(1.0, float(args.time_cap) - (time.perf_counter() - started)))
        row = {"cell": name, "initial_bank": bank, "frozen_kernel": kernel, "hmc": hmc, "hmc_passed": bool(hmc.get("passed", False) and hmc.get("post_hmc_exact_law", {}).get("passed", False))}
        rows[name] = row
        _write(cell_root / "cell_result.json", row)
        progress.update({"phase": name, "completed_cells": len(rows), "latest_hmc_passed": row["hmc_passed"]})
        _write(output / "progress.json", progress)

    manifest = {
        "schema": "bayesfilter.neutra.banana_hmc_matched_kernel_manifest.v1",
        "plan": PLAN.as_posix(), "discovery_root": DISCOVERY.as_posix(),
        "git_commit": subprocess.run(("git", "rev-parse", "HEAD"), cwd=ROOT, check=True, text=True, capture_output=True).stdout.strip(),
        "device": str(logical[0]), "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"], "memory_policy": memory_policy,
        "dtype": "float64", "jit_compile": True, "tf32_enabled": False, "training_seed": 15, "training_updates": 6000,
        "transport_state_hash": state_hash, "no_retuning": True, "cells": [name for name, *_ in cells], "wall_seconds": time.perf_counter() - started,
        "trust_basis": "owner_designated_managed_session_visible_gpu_trusted",
    }
    result = {"schema": "bayesfilter.neutra.banana_hmc_matched_kernel_result.v1", "manifest": manifest, "training": training, "proposal_audit": audit, "cells": rows, "decision": {"promotion": False, "status": "matched_kernel_diagnosis_complete", "nonclaims": ["no default kernel", "no default start bank", "no SSL-LSTM transfer"]}, "wall_seconds": time.perf_counter() - started}
    progress.update({"phase": "complete", "completed_cells": len(rows)})
    _write(output / "progress.json", progress); _write(output / "run_manifest.json", manifest); _write(output / "result.json", result)
    _write(output / "artifact_hashes.json", {"schema": "bayesfilter.neutra.banana_hmc_matched_kernel_hashes.v1", "artifacts": {p.relative_to(output).as_posix(): _sha256(p) for p in sorted(output.rglob("*")) if p.is_file() and p.name != "artifact_hashes.json"}})
    print(json.dumps({"output_root": output.as_posix(), "wall_seconds": result["wall_seconds"], "cells": {name: row["hmc_passed"] for name, row in rows.items()}}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
