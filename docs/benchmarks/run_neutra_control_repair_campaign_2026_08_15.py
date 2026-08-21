#!/usr/bin/env python3
"""Run the reviewed Gaussian LR and banana target-specific repair campaign."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Mapping

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PLAN = ROOT / "docs/plans/bayesfilter-neutra-control-repair-plan-2026-08-15.md"
BASE_RUNNER = ROOT / "docs/benchmarks/run_neutra_generic_five_stage_model_2026_08_15.py"
DEFAULT_OUTPUT = ROOT / "docs/plans/artifacts/neutra-control-repair-2026-08-15"
RATES = (2.0e-4, 5.0e-4, 1.0e-3)
TOTAL_UPDATES = 3000
BATCH_SIZE = 4096
SELECTION_COUNT = 65536
AUDIT_COUNT = 131072
CRITICAL_VALUE = 3.2905267314919255


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="0")
    parser.add_argument("--time-cap", type=float, default=3600.0)
    return parser.parse_args()


def _base_runner() -> Any:
    spec = importlib.util.spec_from_file_location("neutra_control_base", BASE_RUNNER)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load known-law target runner")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _ready(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_ready(v) for v in value]
    if hasattr(value, "numpy"):
        return _ready(value.numpy().tolist())
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "item"):
        return value.item()
    if isinstance(value, Path):
        return value.as_posix()
    return value


def _write(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_ready(value), sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _state_hash(tf: Any, transport: Any) -> str:
    digest = hashlib.sha256()
    for variable in transport.trainable_variables:
        digest.update(bytes(tf.io.serialize_tensor(variable.read_value()).numpy()))
    return digest.hexdigest()


def _train(tf: Any, *, base: Any, model: Mapping[str, Any], config: Any, seed: int, rate: float) -> tuple[Any, Mapping[str, Any]]:
    from bayesfilter.inference.neutra_weighted_training import MatchedReverseKLNeuTraTrainer

    trainer = MatchedReverseKLNeuTraTrainer(config, model["target_log_prob"])
    selection_latent = tf.random.stateless_normal(
        (SELECTION_COUNT, int(model["dimension"])),
        seed=(20260815 + int(seed), 71001), dtype=tf.float64,
    )
    best_loss = float(base._selection_loss(tf, trainer.transport, model["target_log_prob"], selection_latent).numpy())
    best_update = 0
    best_state = tuple(tf.identity(v) for v in trainer.variables)
    clipped = 0
    terminal = best_loss
    for update in range(1, TOTAL_UPDATES + 1):
        fraction = float(update) / float(TOTAL_UPDATES)
        multiplier = 1.0 if fraction < 0.60 else 0.1 if fraction < 0.85 else 0.01
        trainer.optimizer.learning_rate.assign(float(rate) * multiplier)
        latent = tf.random.stateless_normal(
            (BATCH_SIZE, int(model["dimension"])),
            seed=(20260815 + int(seed), 72000 + update), dtype=tf.float64,
        )
        step = trainer.train_step(latent)
        clipped += int(bool(step.clipping_applied.numpy()))
        if update % 250 == 0 or update == TOTAL_UPDATES:
            terminal = float(base._selection_loss(tf, trainer.transport, model["target_log_prob"], selection_latent).numpy())
            if terminal < best_loss:
                best_loss = terminal
                best_update = update
                best_state = tuple(tf.identity(v) for v in trainer.variables)
    for variable, value in zip(trainer.variables, best_state, strict=True):
        variable.assign(value)
    if not math.isfinite(best_loss) or best_update < 0:
        raise RuntimeError("nonfinite or invalid training result")
    return trainer.transport, {
        "selected_loss": best_loss,
        "terminal_loss": terminal,
        "selected_update": best_update,
        "clipped_updates": clipped,
        "executed_updates": TOTAL_UPDATES,
        "state_hash": _state_hash(tf, trainer.transport),
    }


def _config(tf: Any, *, model: Mapping[str, Any], seed: int, hidden: int, permutation: str, initialization_scale: float) -> Any:
    from bayesfilter.inference.neutra_weighted_training import WeightedNeuTraConfig

    stages = int(model["stages"])
    return WeightedNeuTraConfig(
        dimension=int(model["dimension"]),
        hidden_layers=(int(hidden), int(hidden)),
        stages=stages,
        activation="elu",
        s_max=1.0,
        stage_s_max=tuple(model["stage_caps"]),
        stage_unbounded_scale_linear=(True,) + (False,) * (stages - 1),
        permutation_policy=str(permutation),
        initialization_scale=float(initialization_scale),
        initialization_seed=(20260815, 40001 + 100 * int(seed)),
        learning_rate=1.0e-3,
        gradient_clip_norm=10.0,
        jit_compile=True,
    )


def _max_standardized(tf: Any, validation: Mapping[str, Any]) -> float:
    values = []
    for screen in validation["screens"].values():
        estimate = tf.convert_to_tensor(screen["estimate"], tf.float64)
        exact = tf.convert_to_tensor(screen["exact"], tf.float64)
        error = tf.math.divide_no_nan(tf.abs(estimate - exact), tf.convert_to_tensor(screen["standard_error"], tf.float64))
        values.append(float(tf.reduce_max(error).numpy()))
    return max(values) if values else float("inf")


def _run_cell(tf: Any, *, base: Any, model: Mapping[str, Any], seed: int, rate: float, arm: Mapping[str, Any], count: int) -> Mapping[str, Any]:
    config = _config(tf, model=model, seed=seed, hidden=int(arm["hidden"]), permutation=str(arm["permutation"]), initialization_scale=float(arm["initialization_scale"]))
    transport, training = _train(tf, base=base, model=model, config=config, seed=seed, rate=rate)
    audit = base._proposal_audit(tf, transport, model, sample_count=int(count), seed=(20260815, 73000 + int(seed)))
    passed = bool(tf.convert_to_tensor(audit["passed"]).numpy())
    return {
        "target": model["name"],
        "arm": dict(arm),
        "seed": int(seed),
        "learning_rate": float(rate),
        "training": training,
        "validation": audit,
        "known_law_gate_passed": passed,
        "max_standardized_discrepancy": _max_standardized(tf, audit),
        "config": config.manifest_payload(),
    }


def _select(cells: list[Mapping[str, Any]], *, group_key: str) -> tuple[Mapping[str, Any], list[Mapping[str, Any]]]:
    grouped: dict[tuple[str, float], list[Mapping[str, Any]]] = {}
    for row in cells:
        key = (str(row["arm"][group_key]), float(row["learning_rate"]))
        grouped.setdefault(key, []).append(row)
    summaries = []
    for (arm_id, rate), rows in grouped.items():
        if len(rows) != 2:
            raise RuntimeError(f"selection requires two seeds for {arm_id} {rate}")
        summaries.append({
            "arm": arm_id,
            "learning_rate": rate,
            "selection_passed_both_seeds": all(bool(row["known_law_gate_passed"]) for row in rows),
            "mean_max_standardized_discrepancy": statistics.mean(float(row["max_standardized_discrepancy"]) for row in rows),
            "mean_selection_loss": statistics.mean(float(row["training"]["selected_loss"]) for row in rows),
        })
    passing = [row for row in summaries if row["selection_passed_both_seeds"]]
    pool = passing if passing else summaries
    chosen = min(pool, key=lambda row: (row["mean_max_standardized_discrepancy"], row["mean_selection_loss"], row["arm"], row["learning_rate"]))
    return chosen, summaries


def main() -> int:
    args = _args()
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(f"output root must be fresh: {output}")
    if float(args.time_cap) <= 0.0:
        raise ValueError("time cap must be positive")
    if not PLAN.is_file() or not BASE_RUNNER.is_file():
        raise FileNotFoundError("reviewed plan or base runner is missing")
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
    base = _base_runner()
    progress = {"schema": "bayesfilter.neutra.control_repair_progress.v1", "phase": "started", "completed_cells": 0, "time_cap_seconds": float(args.time_cap)}
    _write(output / "progress.json", progress)

    all_cells: list[Mapping[str, Any]] = []
    gaussian = base._model(tf, "gaussian")
    gaussian_arm = {"id": "baseline", "hidden": 32, "permutation": "full_reverse", "initialization_scale": 0.02}
    gaussian_selection = []
    for seed in (2, 3):
        for rate in RATES:
            if time.perf_counter() - started > float(args.time_cap):
                raise TimeoutError("repair campaign time cap exhausted during Gaussian selection")
            row = _run_cell(tf, base=base, model=gaussian, seed=seed, rate=rate, arm=gaussian_arm, count=SELECTION_COUNT)
            gaussian_selection.append(row)
            progress.update({"phase": "gaussian_selection", "completed_cells": len(all_cells) + len(gaussian_selection)})
            _write(output / "progress.json", progress)
    chosen_gaussian, gaussian_summary = _select(gaussian_selection, group_key="id")
    gaussian_final = []
    for seed in (4, 5):
        row = _run_cell(tf, base=base, model=gaussian, seed=seed, rate=float(chosen_gaussian["learning_rate"]), arm=gaussian_arm, count=AUDIT_COUNT)
        gaussian_final.append(row)
        progress.update({"phase": "gaussian_confirmation", "completed_cells": len(gaussian_selection) + len(gaussian_final)})
        _write(output / "progress.json", progress)

    banana = base._model(tf, "banana")
    banana_arms = (
        {"id": "baseline", "hidden": 32, "permutation": "full_reverse", "initialization_scale": 0.02},
        {"id": "identity_biased", "hidden": 32, "permutation": "full_reverse", "initialization_scale": 0.005},
        {"id": "root_preserving", "hidden": 32, "permutation": "root_preserving_reverse", "initialization_scale": 0.02},
        {"id": "width64", "hidden": 64, "permutation": "full_reverse", "initialization_scale": 0.02},
    )
    banana_selection = []
    for arm in banana_arms:
        for seed in (0, 1):
            for rate in RATES:
                if time.perf_counter() - started > float(args.time_cap):
                    raise TimeoutError("repair campaign time cap exhausted during banana selection")
                row = _run_cell(tf, base=base, model=banana, seed=seed, rate=rate, arm=arm, count=SELECTION_COUNT)
                banana_selection.append(row)
                progress.update({"phase": "banana_selection", "completed_cells": len(gaussian_selection) + len(gaussian_final) + len(banana_selection)})
                _write(output / "progress.json", progress)
    banana_choices = {}
    banana_summaries = {}
    for arm in banana_arms:
        rows = [row for row in banana_selection if row["arm"]["id"] == arm["id"]]
        choice, summary = _select(rows, group_key="id")
        banana_choices[arm["id"]] = choice
        banana_summaries[arm["id"]] = summary
    banana_screen = []
    for arm in banana_arms:
        choice = banana_choices[arm["id"]]
        for seed in (2, 3):
            row = _run_cell(tf, base=base, model=banana, seed=seed, rate=float(choice["learning_rate"]), arm=arm, count=SELECTION_COUNT)
            banana_screen.append(row)
            progress.update({"phase": "banana_screen", "completed_cells": len(gaussian_selection) + len(gaussian_final) + len(banana_selection) + len(banana_screen)})
            _write(output / "progress.json", progress)
    nominated_arm = min(
        banana_arms,
        key=lambda arm: (
            not all(row["known_law_gate_passed"] for row in banana_screen if row["arm"]["id"] == arm["id"]),
            statistics.mean(row["max_standardized_discrepancy"] for row in banana_screen if row["arm"]["id"] == arm["id"]),
            arm["id"],
        ),
    )
    nominated_choice = banana_choices[nominated_arm["id"]]
    banana_confirmation = []
    for seed in (4, 5):
        row = _run_cell(tf, base=base, model=banana, seed=seed, rate=float(nominated_choice["learning_rate"]), arm=nominated_arm, count=AUDIT_COUNT)
        banana_confirmation.append(row)
        progress.update({"phase": "banana_confirmation", "completed_cells": len(gaussian_selection) + len(gaussian_final) + len(banana_selection) + len(banana_screen) + len(banana_confirmation)})
        _write(output / "progress.json", progress)

    manifest = {
        "schema": "bayesfilter.neutra.control_repair_manifest.v1",
        "plan": PLAN.as_posix(),
        "git_commit": subprocess.run(("git", "rev-parse", "HEAD"), cwd=ROOT, check=True, text=True, capture_output=True).stdout.strip(),
        "device": str(logical[0]),
        "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "dtype": "float64", "jit_compile": True, "tf32_enabled": False,
        "batch_size": BATCH_SIZE, "selection_count": SELECTION_COUNT, "audit_count": AUDIT_COUNT,
        "total_updates": TOTAL_UPDATES, "learning_rates": RATES,
        "memory_policy": memory_policy, "tensorflow_version": tf.__version__,
        "trust_basis": "owner_designated_managed_session_visible_gpu_trusted",
        "time_cap_seconds": float(args.time_cap), "wall_seconds": time.perf_counter() - started,
    }
    result = {
        "schema": "bayesfilter.neutra.control_repair_result.v1",
        "manifest": manifest,
        "gaussian": {"selection": gaussian_summary, "chosen": chosen_gaussian, "confirmation": gaussian_final, "passed_both_confirmation_seeds": all(row["known_law_gate_passed"] for row in gaussian_final)},
        "banana": {"arm_selection": banana_summaries, "choices": banana_choices, "screen": banana_screen, "nominated_arm": nominated_arm, "nominated_choice": nominated_choice, "confirmation": banana_confirmation, "passed_both_confirmation_seeds": all(row["known_law_gate_passed"] for row in banana_confirmation)},
        "decision": {"promotion": False, "no_hmc": True, "status": "control_repair_campaign_complete", "nonclaims": ["no SSL-LSTM transfer claim", "no HMC or posterior-correctness claim", "no universal architecture or learning-rate claim"]},
        "wall_seconds": time.perf_counter() - started,
    }
    progress.update({"phase": "complete", "completed_cells": len(gaussian_selection) + len(gaussian_final) + len(banana_selection) + len(banana_screen) + len(banana_confirmation)})
    _write(output / "progress.json", progress)
    _write(output / "run_manifest.json", manifest)
    _write(output / "result.json", result)
    _write(output / "artifact_hashes.json", {"schema": "bayesfilter.neutra.control_repair_hashes.v1", "artifacts": {p.relative_to(output).as_posix(): _sha256(p) for p in sorted(output.rglob("*")) if p.is_file() and p.name != "artifact_hashes.json"}})
    print(json.dumps({"output_root": output.as_posix(), "wall_seconds": result["wall_seconds"], "gaussian_passed": result["gaussian"]["passed_both_confirmation_seeds"], "banana_passed": result["banana"]["passed_both_confirmation_seeds"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
