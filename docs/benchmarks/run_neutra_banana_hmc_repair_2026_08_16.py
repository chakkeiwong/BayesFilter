#!/usr/bin/env python3
"""Diagnose banana NeuTra HMC failure with start and geometry controls."""

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
REPLICATION_RUNNER = ROOT / "docs/benchmarks/run_neutra_replication_hmc_campaign_2026_08_16.py"
REPAIR_RUNNER = ROOT / "docs/benchmarks/run_neutra_banana_repair_2026_08_16.py"
DEFAULT_OUTPUT = ROOT / "docs/plans/artifacts/neutra-banana-hmc-repair-2026-08-16-r1"
DIMENSION = 16
CURVATURE = 0.35
SEED = 15
HMC_CHAINS = 4
AUDIT_COUNT = 131072


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="0")
    parser.add_argument("--time-cap", type=float, default=3600.0)
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


def _hash_payload(payload: Any) -> str:
    return hashlib.sha256(json.dumps(_safe(payload), sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


class AnalyticBananaTransport:
    """Exact triangular banana transport used only as a mechanics control."""

    def __init__(self, tf_module: Any) -> None:
        self.tf = tf_module
        self.curvature = tf_module.constant(35, tf_module.float64) / tf_module.constant(100, tf_module.float64)
        self.parameter_dim = DIMENSION
        self.config = type("AnalyticConfig", (), {"dimension": DIMENSION})()
        self.trainable_variables: tuple[Any, ...] = ()

    def manifest_payload(self) -> Mapping[str, Any]:
        return {
            "schema": "bayesfilter.neutra.analytic_banana_transport.v1",
            "dimension": DIMENSION,
            "curvature": CURVATURE,
            "unit_jacobian": True,
            "mechanics_control_only": True,
        }

    def bind_frozen_identity(self, identity: Mapping[str, Any]) -> None:
        required = {"checkpoint_sha256", "training_state_hash", "transport_tensor_hash"}
        if not required.issubset(identity):
            raise ValueError("analytic transport frozen identity is incomplete")
        self._frozen_identity = dict(identity)

    def _rank2(self, value: Any) -> tuple[Any, bool]:
        rows = self.tf.cast(self.tf.convert_to_tensor(value), self.tf.float64)
        if rows.shape.rank == 1:
            return rows[self.tf.newaxis, :], True
        if rows.shape.rank != 2 or rows.shape[-1] != DIMENSION:
            raise ValueError("analytic banana transport expects [batch, 16]")
        return rows, False

    def forward_and_logdet(self, latent: Any) -> tuple[Any, Any]:
        rows, scalar = self._rank2(latent)
        tfm = self.tf
        physical = tfm.concat((rows[:, :1], rows[:, 1:2] + self.curvature * (tfm.square(rows[:, :1]) - 1.0), rows[:, 2:]), axis=1)
        logdet = tfm.zeros((tfm.shape(rows)[0],), tfm.float64)
        return (physical[0], logdet[0]) if scalar else (physical, logdet)

    def forward(self, latent: Any) -> Any:
        return self.forward_and_logdet(latent)[0]

    def forward_batch(self, latent: Any) -> Any:
        return self.forward(latent)

    def log_abs_det_jacobian(self, latent: Any) -> Any:
        return self.forward_and_logdet(latent)[1]

    def log_abs_det_jacobian_batch(self, latent: Any) -> Any:
        return self.log_abs_det_jacobian(latent)

    def pullback_score_batch(self, latent: Any, output_score: Any) -> Any:
        rows, _ = self._rank2(latent)
        score, _ = self._rank2(output_score)
        result = self.tf.identity(score)
        result = self.tf.concat((score[:, :1] + 2.0 * self.curvature * rows[:, :1] * score[:, 1:2], score[:, 1:]), axis=1)
        return result

    def pullback_score(self, latent: Any, output_score: Any) -> Any:
        result = self.pullback_score_batch(latent if self.tf.convert_to_tensor(latent).shape.rank == 2 else self.tf.convert_to_tensor(latent)[self.tf.newaxis, :], output_score if self.tf.convert_to_tensor(output_score).shape.rank == 2 else self.tf.convert_to_tensor(output_score)[self.tf.newaxis, :])
        return result[0] if self.tf.convert_to_tensor(latent).shape.rank == 1 else result

    def log_abs_det_jacobian_score_batch(self, latent: Any) -> Any:
        rows, _ = self._rank2(latent)
        return self.tf.zeros_like(rows)

    def log_abs_det_jacobian_score(self, latent: Any) -> Any:
        return self.log_abs_det_jacobian_score_batch(latent)


def _central_bank(tf_module: Any) -> Any:
    rows = tf_module.zeros((HMC_CHAINS, DIMENSION), tf_module.float64)
    offsets = tf_module.constant((0.0, 0.25, -0.25, 0.25), tf_module.float64)
    axes = tf_module.constant((0, 0, 0, 1), tf_module.int32)
    return tf_module.tensor_scatter_nd_update(rows, tf_module.stack((tf_module.range(HMC_CHAINS), axes), axis=1), offsets)


def _tune(replication: Any, base_adapter: Any, transport: Any, initial: Any, out: Path, scope: str, seed_offset: int) -> Mapping[str, Any]:
    from bayesfilter.inference.fixed_transport_hmc_tuning_tf import FixedTransportHMCKernelTuningConfig, tune_fixed_transport_hmc_kernel

    cfg = FixedTransportHMCKernelTuningConfig(
        initial_step_size=0.05,
        leapfrog_grid=(3, 5, 10, 15, 20, 25),
        chain_count=HMC_CHAINS,
        initial_state_bank=tuple(tuple(float(x) for x in row) for row in initial.numpy().tolist()),
        target_accept_prob=0.70,
        acceptance_band=(0.55, 0.90),
        repair_band=(0.40, 0.95),
        fixed_grid_fallback_acceptance_max=0.95,
        budget_schedule=(32, 64, 128),
        tune_num_results=16,
        screen_num_results=64,
        screen_num_burnin_steps=16,
        verification_num_results=2000,
        verification_num_burnin_steps=64,
        require_modern_rank_normalized_verification=True,
        verification_coordinate_system="hmc_coordinates",
        verification_min_retained_results_per_chain=2000,
        tune_seed_base=(20260816, 55001 + seed_offset),
        screen_seed_base=(20260816, 56001 + seed_offset),
        verification_seed_base=(20260816, 57001 + seed_offset),
        chain_execution_mode="tf_function",
        use_xla=True,
        target_scope=scope,
        output_filename="tuning_result.json",
    )
    return tune_fixed_transport_hmc_kernel(base_adapter=base_adapter, fixed_transport=transport, initial_position=initial[0], config=cfg, output_dir=out).payload()


def _run_hmc(replication: Any, tf_module: Any, model: Mapping[str, Any], transport: Any, base_adapter: Any, initial: Any, tuning: Mapping[str, Any], out: Path, scope: str) -> Mapping[str, Any]:
    return replication._run_hmc(tf_module, model, transport, base_adapter, initial, tuning, out, scope, 3600.0)


def _proposal_audit(replication: Any, tf_module: Any, transport: Any, model: Mapping[str, Any], seed: tuple[int, int]) -> Mapping[str, Any]:
    audit = replication._audit(tf_module, transport, model, count=AUDIT_COUNT, seed=seed)
    audit["passed"] = bool(tf_module.convert_to_tensor(audit["passed"]).numpy())
    return audit


def main() -> int:
    args = _args()
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(f"output root must be fresh: {output}")
    if not PLAN.is_file() or not REPLICATION_RUNNER.is_file() or not REPAIR_RUNNER.is_file():
        raise FileNotFoundError("reviewed plan or runner is missing")
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
    replication = _load(REPLICATION_RUNNER, "banana_hmc_repair_replication")
    repair = _load(REPAIR_RUNNER, "banana_hmc_repair_training")
    base = replication._load_base()
    model = base._model(tf, "banana")
    base_adapter = replication.AnalyticControlAdapter(tf, model, "analytic_control:banana:base_v1")
    progress = {"schema": "bayesfilter.neutra.banana_hmc_repair_progress.v1", "phase": "training", "completed_arms": 0}
    _write(output / "progress.json", progress)

    learned, training = repair._train(tf, model, seed=SEED, updates=6000)
    learned_hash = replication._state_hash(tf, learned)
    replication._bind_frozen(learned, tf, model, learned_hash)
    # Reuse the terminal r3 audit partition for deterministic replay; changing
    # this seed would turn a reproducibility check into a new stochastic veto.
    learned_audit = _proposal_audit(replication, tf, learned, model, (20260816, 59015))
    if not learned_audit["passed"]:
        raise RuntimeError("frozen learned transport failed proposal audit")
    _write(output / "learned_training.json", {"seed": SEED, "training": training, "state_hash": learned_hash, "audit": learned_audit, "config": _safe(learned.config.manifest_payload())})

    original = replication._initial_bank(tf, DIMENSION)
    central = _central_bank(tf)
    analytic = AnalyticBananaTransport(tf)
    analytic_hash = _hash_payload(analytic.manifest_payload())
    analytic.bind_frozen_identity({"checkpoint_sha256": analytic_hash, "training_state_hash": analytic_hash, "transport_tensor_hash": analytic_hash})
    analytic_audit = _proposal_audit(replication, tf, analytic, model, (20260816, 69016))
    if not analytic_audit["passed"]:
        raise RuntimeError("analytic banana mechanics control failed exact-law audit")

    arms = (("learned_original_bank", learned, original, 101), ("learned_central_bank", learned, central, 202), ("analytic_original_bank", analytic, original, 303))
    rows: dict[str, Any] = {}
    for name, transport, initial, offset in arms:
        arm_root = output / name
        arm_root.mkdir(parents=True)
        transport_hash = learned_hash if transport is learned else analytic_hash
        adapter_scope = f"analytic_control:banana:hmc_repair:{name}:{transport_hash}"
        tuning = _tune(replication, base_adapter, transport, initial, arm_root / "tuning", adapter_scope, offset)
        row = {"arm": name, "transport_hash": transport_hash, "initial_bank": initial, "proposal_audit_passed": True, "tuning": tuning}
        if tuning.get("passed") is True:
            row["hmc"] = _run_hmc(replication, tf, model, transport, base_adapter, initial, tuning, arm_root / "hmc", name)
        else:
            row["hmc"] = {"status": "blocked_by_tuning_veto", "passed": False}
        row["hmc_passed"] = bool(row["hmc"].get("passed", False) and row["hmc"].get("post_hmc_exact_law", {}).get("passed", False))
        rows[name] = row
        _write(arm_root / "arm_result.json", row)
        progress.update({"phase": name, "completed_arms": len(rows), "latest_hmc_passed": row["hmc_passed"]})
        _write(output / "progress.json", progress)

    manifest = {
        "schema": "bayesfilter.neutra.banana_hmc_repair_manifest.v1",
        "plan": PLAN.as_posix(),
        "git_commit": subprocess.run(("git", "rev-parse", "HEAD"), cwd=ROOT, check=True, text=True, capture_output=True).stdout.strip(),
        "target": model["manifest"], "device": str(logical[0]), "cuda_visible_devices": os.environ["CUDA_VISIBLE_DEVICES"],
        "memory_policy": memory_policy, "dtype": "float64", "jit_compile": True, "tf32_enabled": False,
        "training_seed": SEED, "training_updates": 6000, "batch_size": 4096, "audit_count": AUDIT_COUNT,
        "arms": [name for name, *_ in arms], "hmc_chains": HMC_CHAINS,
        "trust_basis": "owner_designated_managed_session_visible_gpu_trusted", "wall_seconds": time.perf_counter() - started,
    }
    result = {"schema": "bayesfilter.neutra.banana_hmc_repair_result.v1", "manifest": manifest, "learned_training": {"state_hash": learned_hash, "audit_passed": learned_audit["passed"]}, "analytic_control": {"transport_hash": analytic_hash, "audit_passed": analytic_audit["passed"]}, "arms": rows, "decision": {"promotion": False, "status": "banana_hmc_repair_complete", "nonclaims": ["no universal HMC kernel", "no statistical superiority", "no SSL-LSTM transfer", "no production/default readiness"]}, "wall_seconds": time.perf_counter() - started}
    progress.update({"phase": "complete", "completed_arms": len(rows)})
    _write(output / "progress.json", progress); _write(output / "run_manifest.json", manifest); _write(output / "result.json", result)
    _write(output / "artifact_hashes.json", {"schema": "bayesfilter.neutra.banana_hmc_repair_hashes.v1", "artifacts": {p.relative_to(output).as_posix(): _sha256(p) for p in sorted(output.rglob("*")) if p.is_file() and p.name != "artifact_hashes.json"}})
    print(json.dumps({"output_root": output.as_posix(), "wall_seconds": result["wall_seconds"], "arms": {name: row["hmc_passed"] for name, row in rows.items()}}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
