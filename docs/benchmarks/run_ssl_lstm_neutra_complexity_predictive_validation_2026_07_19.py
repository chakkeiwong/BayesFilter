#!/usr/bin/env python3
"""Q-general recovery and predictive validation after retained-HMC admission."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import resource
import subprocess
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont


def _configure_visibility_before_tensorflow_import() -> str:
    mode = None
    if "--mode" in sys.argv:
        index = sys.argv.index("--mode")
        if index + 1 < len(sys.argv):
            mode = sys.argv[index + 1]
    if mode in {"contract-smoke", "calibrate"}:
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
        return f"cpu-hidden-{mode}"
    if mode not in {"calibrate", "validate"} and os.environ.get("CUDA_VISIBLE_DEVICES") is not None:
        return str(os.environ["CUDA_VISIBLE_DEVICES"])
    probe = subprocess.run(
        ("nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"),
        check=True,
        capture_output=True,
        text=True,
        timeout=5.0,
    )
    available = {
        int(line.strip())
        for line in probe.stdout.splitlines()
        if line.strip().isdigit()
    }
    selected = "1" if 1 in available else ("0" if 0 in available else "")
    if not selected:
        raise RuntimeError("no physical GPU 1 or GPU 0 is available")
    os.environ["CUDA_VISIBLE_DEVICES"] = selected
    return selected


SELECTED_GPU = _configure_visibility_before_tensorflow_import()

import numpy as np
import tensorflow as tf


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bayesfilter.inference.batched_value_score import FixedTransportValueScoreAdapter  # noqa: E402
from bayesfilter.inference.cpu_forecast_pool import (  # noqa: E402
    CPUForecastPool,
    CPUForecastPoolConfig,
)
from bayesfilter.inference.hmc import (  # noqa: E402
    RetainedSampleHMCArchiveConfig,
    build_retained_sample_hmc_archive_runner,
)
from bayesfilter.inference.hmc_posterior_diagnostics import (  # noqa: E402
    compute_coordinate_diagnostics,
)
from bayesfilter.inference.neutra_artifacts import load_frozen_neutra_artifact  # noqa: E402
from bayesfilter.inference.posterior_adapter import ValueScoreCapability  # noqa: E402
from bayesfilter.inference.predictive_equivalence import (  # noqa: E402
    PredictiveContractError,
    chain_bartlett_long_run_covariance,
    classify_split_proper_score_equivalence,
    conditional_mean_log_variance_influence,
    proper_score_loss,
    split_quadratic_loss_confidence_bounds,
)
from bayesfilter.nonlinear.ssl_lstm_complexity_predictive_tf import (  # noqa: E402
    CALIBRATION_DRAWS_PER_CHAIN,
    FORECAST_HORIZON,
    FORECAST_REPLICATION_COUNT,
    calibration_from_observation_banks,
    calibration_seed_roots,
    complexity_calibration_signature,
)
from bayesfilter.nonlinear.ssl_lstm_complexity_target_tf import (  # noqa: E402
    PRIOR_CENTER,
    complexity_posterior_target,
)


SCHEMA = "bayesfilter.ssl_lstm.neutra_complexity_predictive_validation.v1"
PHASE5_SCHEMA = "bayesfilter.ssl_lstm.neutra_complexity_retained_hmc.v1"
PLAN = Path(
    "docs/plans/"
    "bayesfilter-ssl-lstm-neutra-hmc-state-complexity-ladder-plan-2026-07-19.md"
)
SCRIPT = Path(__file__).resolve().relative_to(ROOT)
Q_VALUES = (1, 2, 5, 10, 20)
CHARTS = ("chart-a", "chart-b")
ROOT_SEED = 20260719
PHASE5_MAX_DRAWS = 4096
PREDICTIVE_DRAWS_PER_CHAIN = 12288
SEGMENT_RESULTS = 256
INITIAL_Z = (
    (0.0, 0.0, 0.0, 0.0),
    (0.5, -0.5, 0.5, -0.5),
    (-0.5, 0.5, -0.5, 0.5),
    (0.5, 0.5, -0.5, -0.5),
)
FORECAST_BLOCK_DRAWS = 256
THETA_MAP_CHUNK_ROWS = 4096
FORECAST_WORKERS_BY_Q = {1: 32, 2: 32, 5: 32, 10: 32, 20: 16}
HAC_MULTIPLIER = 3.0
RIDGE_LADDER = (0.0,)
CONDITION_NUMBER_MAX = 1.0e8
AVERAGE_ALPHA = 0.025
HORIZON_ALPHA = 0.0025
FAMILYWISE_ALPHA = 0.05
NEGLIGIBLE_ANCHOR_LOSS = max(0.5 * 0.05**2, 0.25 * math.log(1.05) ** 2)
MATERIAL_ANCHOR_LOSS = min(
    0.5 * 0.20**2,
    0.25 * math.log(1.25) ** 2,
    0.25 * math.log(0.80) ** 2,
)
ACCEPTABLE_LOSS = 0.5 * (NEGLIGIBLE_ANCHOR_LOSS + MATERIAL_ANCHOR_LOSS)
HOST_RAM_CAP_BYTES = 64 * 1024**3
COST_MARGIN = 1.50
FIRST_HMC_SEGMENT_RESERVE_SECONDS = 900.0
FIRST_FORECAST_BLOCK_RESERVE_SECONDS = 1800.0
PLOT_COLORS = ("#0b6e4f", "#d95d39", "#2d5f9a", "#c28b18")


class PredictiveValidationError(RuntimeError):
    pass


class ResourceStop(PredictiveValidationError):
    pass


class HostMemoryVeto(PredictiveValidationError):
    pass


class Budget:
    def __init__(self, seconds: float, *, prior_seconds: float = 0.0) -> None:
        self.seconds = float(seconds)
        self.prior_seconds = float(prior_seconds)
        self.started = time.perf_counter()
        self.hmc_seconds_per_transition_leapfrog: list[float] = []
        self.forecast_seconds_per_draw: list[float] = []

    @property
    def elapsed(self) -> float:
        return self.prior_seconds + time.perf_counter() - self.started

    def require(self, reserve_seconds: float) -> None:
        if self.elapsed + float(reserve_seconds) >= self.seconds:
            raise ResourceStop("declared Phase 6 cumulative cap exhausted")

    def observe_hmc(self, seconds_per_transition_leapfrog: float) -> None:
        value = float(seconds_per_transition_leapfrog)
        if math.isfinite(value) and value > 0.0:
            self.hmc_seconds_per_transition_leapfrog.append(value)

    def observe_forecast(self, seconds_per_draw: float) -> None:
        value = float(seconds_per_draw)
        if math.isfinite(value) and value > 0.0:
            self.forecast_seconds_per_draw.append(value)

    def hmc_reserve(self, transition_leapfrogs: int, *, cold: bool = False) -> float:
        execution = 0.0
        if self.hmc_seconds_per_transition_leapfrog:
            execution = (
                max(self.hmc_seconds_per_transition_leapfrog)
                * int(transition_leapfrogs)
                * COST_MARGIN
            )
        return max(
            60.0,
            execution + (FIRST_HMC_SEGMENT_RESERVE_SECONDS if cold else 0.0),
        )

    def forecast_reserve(self, draw_count: int, *, cold: bool = False) -> float:
        execution = 0.0
        if self.forecast_seconds_per_draw:
            execution = max(self.forecast_seconds_per_draw) * int(draw_count) * COST_MARGIN
        return max(
            60.0,
            execution + (FIRST_FORECAST_BLOCK_RESERVE_SECONDS if cold else 0.0),
        )


def json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if hasattr(value, "numpy"):
        return json_safe(value.numpy())
    if isinstance(value, float) and not math.isfinite(value):
        return "NaN" if math.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
    return value


def canonical(payload: Any) -> bytes:
    return (
        json.dumps(
            json_safe(payload),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def payload_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical(payload)).hexdigest()


def strict_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PredictiveValidationError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, payload: Any, *, replace: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        raise PredictiveValidationError(f"output already exists: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(canonical(payload))
    temporary.replace(path)


def write_tensor(path: Path, tensor: tf.Tensor) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise PredictiveValidationError(f"tensor output already exists: {path}")
    path.write_bytes(bytes(tf.io.serialize_tensor(tensor).numpy()))
    return sha256(path)


def parse_tensor(path: Path, dtype: tf.DType = tf.float64) -> tf.Tensor:
    return tf.io.parse_tensor(path.read_bytes(), out_type=dtype)


def repo_path(path: Path, *, label: str) -> Path:
    resolved = (ROOT / path).resolve()
    if not resolved.is_relative_to(ROOT):
        raise PredictiveValidationError(f"{label} must remain inside the repository")
    return resolved


def git(*args: str) -> str:
    return subprocess.check_output(("git", *args), cwd=ROOT, text=True).strip()


def execution_source_signature() -> str:
    paths = (
        SCRIPT,
        Path("bayesfilter/nonlinear/ssl_lstm_complexity_predictive_tf.py"),
        Path("bayesfilter/nonlinear/ssl_lstm_complexity_target_tf.py"),
        Path("bayesfilter/inference/cpu_forecast_pool.py"),
        Path("bayesfilter/inference/batched_value_score.py"),
        Path("bayesfilter/inference/neutra_artifacts.py"),
        Path("bayesfilter/inference/hmc.py"),
        Path("bayesfilter/inference/hmc_posterior_diagnostics.py"),
        Path("bayesfilter/inference/predictive_equivalence.py"),
    )
    return payload_sha256({path.as_posix(): sha256(ROOT / path) for path in paths})


def enforce_parent_memory() -> int:
    parent = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024)
    if parent > HOST_RAM_CAP_BYTES:
        raise HostMemoryVeto("Phase 6 parent RSS exceeded 64 GiB")
    return parent


def enforce_forecast_memory(metadata: Mapping[str, Any]) -> int:
    aggregate = int(metadata["aggregate_parent_worker_ru_maxrss_bytes"])
    if aggregate > HOST_RAM_CAP_BYTES:
        raise HostMemoryVeto("Phase 6 parent plus forecast-worker RSS exceeded 64 GiB")
    return aggregate


class TargetBridge:
    def __init__(self, target: Any, *, evidence_path: str) -> None:
        self.target = target
        self.parameter_dim = int(target.parameter_dim)
        self.parameter_names = tuple(target.parameter_names)
        self.target_scope = f"{target.target_scope}:phase6_fixed_kernel_extension"
        self.evidence_path = evidence_path

    def adapter_signature(self) -> str:
        return hashlib.sha256(
            (self.target.adapter_signature() + ":phase6-fixed-kernel-extension").encode("ascii")
        ).hexdigest()

    def target_signature(self) -> str:
        return self.target.target_signature()

    def value_score_capability(self) -> ValueScoreCapability:
        return ValueScoreCapability(
            value_score_authority="graph_native",
            xla_hmc_ready=True,
            full_chain_xla_diagnostic_ready=True,
            runtime_backend="ssl_lstm_complexity_phase6_target_bridge",
            evidence_path=self.evidence_path,
            target_scope=self.target_scope,
            nonclaims=(
                "fixed admitted Phase 5 kernel extension only",
                "predictive sample-size completion",
                "cannot rescue Phase 5 sampler failure",
            ),
        )

    def log_prob_and_grad(self, values: Any) -> tuple[tf.Tensor, tf.Tensor]:
        tensor = tf.convert_to_tensor(values, tf.float64)
        if tensor.shape.rank == 1:
            return self.target.value_and_score(tensor)
        if tensor.shape.rank == 2:
            return self.target.batch_value_and_score(tensor)
        raise ValueError("Phase 6 target bridge requires rank one or two")


def configure_gpu() -> None:
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        raise PredictiveValidationError("Phase 6 validation requires a visible GPU")
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
        except RuntimeError as exc:
            if "cannot be modified after being initialized" not in str(exc):
                raise
    tf.config.experimental.enable_tensor_float_32_execution(True)
    tf.config.set_soft_device_placement(False)


def calibration_receipt(
    calibration: Any,
    q: int,
    *,
    forecast_signatures: tuple[str, ...],
    source_signature: str,
    observation_bank_binding: Mapping[str, Any],
    seed_domain_contract: Mapping[str, Any],
) -> dict[str, Any]:
    target = complexity_posterior_target(q, jit_compile=True)
    return {
        "schema": SCHEMA,
        "mode": "calibrate",
        "status": "CALIBRATION_FROZEN",
        "q": q,
        "center": json_safe(calibration.center),
        "scale": json_safe(calibration.scale),
        "chain_count": calibration.chain_count,
        "draw_count_per_chain": calibration.draw_count_per_chain,
        "replication_count": calibration.replication_count,
        "seed_roots": json_safe(calibration.seed_roots),
        "forecast_signatures": list(forecast_signatures),
        "target_signature": calibration.target_signature,
        "calibration_signature": calibration.calibration_signature,
        "source_target_signature": target.target_signature(),
        "execution_source_signature": source_signature,
        "observation_bank": dict(observation_bank_binding),
        "retained_input_used": False,
        "seed_domain": "q_specific_calibration_truth_fixture",
        "seed_domain_contract": dict(seed_domain_contract),
        "nonclaims": [
            "calibration-only synthetic-truth forecast bank",
            "no retained A/B values accessed",
            "no predictive equivalence or HMC claim",
        ],
    }


def calibration_material_contract(
    args: argparse.Namespace,
    *,
    source_signature: str,
    seed_contract: Mapping[str, Any],
    target_signature: str,
) -> dict[str, Any]:
    return {
        "q": int(args.q),
        "execution_source_signature": source_signature,
        "target_signature": target_signature,
        "seed_domain_contract": dict(seed_contract),
        "worker_count": FORECAST_WORKERS_BY_Q[args.q],
        "chain_count": len(calibration_seed_roots(args.q)),
        "draw_count_per_chain": CALIBRATION_DRAWS_PER_CHAIN,
        "forecast_replication_count": FORECAST_REPLICATION_COUNT,
        "forecast_horizon": FORECAST_HORIZON,
        "retained_input_used": False,
        "host_ram_cap_bytes": HOST_RAM_CAP_BYTES,
    }


def load_calibration_chain(
    *,
    output: Path,
    q: int,
    chain_index: int,
    root: tuple[int, int],
    source_signature: str,
) -> tuple[tf.Tensor, str, dict[str, Any]] | None:
    tensor_path = output / f"calibration-chain-{chain_index:02d}.tftensor"
    receipt_path = output / f"calibration-chain-{chain_index:02d}.json"
    if not tensor_path.exists() and not receipt_path.exists():
        return None
    if not tensor_path.is_file() or not receipt_path.is_file():
        raise PredictiveValidationError("incomplete calibration chain checkpoint")
    receipt = strict_json(receipt_path)
    seeds = forecast_seeds_from_root(root, CALIBRATION_DRAWS_PER_CHAIN)
    expected_seed_hash = hashlib.sha256(seeds.tobytes()).hexdigest()
    if receipt.get("schema") != SCHEMA or receipt.get("mode") != "calibrate-chain":
        raise PredictiveValidationError("calibration chain receipt schema mismatch")
    if receipt.get("q") != q or receipt.get("chain_index") != chain_index:
        raise PredictiveValidationError("calibration chain receipt identity mismatch")
    if tuple(receipt.get("root_seed", ())) != root:
        raise PredictiveValidationError("calibration chain root mismatch")
    if receipt.get("seed_hash") != expected_seed_hash:
        raise PredictiveValidationError("calibration chain seed hash mismatch")
    if receipt.get("execution_source_signature") != source_signature:
        raise PredictiveValidationError("calibration chain source binding mismatch")
    if sha256(tensor_path) != receipt.get("observation_tensor_sha256"):
        raise PredictiveValidationError("calibration chain tensor hash mismatch")
    observations = parse_tensor(tensor_path)
    if tuple(observations.shape) != (
        CALIBRATION_DRAWS_PER_CHAIN,
        FORECAST_REPLICATION_COUNT,
        FORECAST_HORIZON,
    ):
        raise PredictiveValidationError("calibration chain tensor shape mismatch")
    observation_hash = hashlib.sha256(
        np.ascontiguousarray(observations.numpy()).tobytes()
    ).hexdigest()
    if receipt.get("observation_hash") != observation_hash:
        raise PredictiveValidationError("calibration chain content hash mismatch")
    forecast_signature = payload_sha256(
        {
            "root": list(root),
            "seed_hash": expected_seed_hash,
            "observation_hash": observation_hash,
        }
    )
    if receipt.get("forecast_signature") != forecast_signature:
        raise PredictiveValidationError("calibration chain signature mismatch")
    if int(receipt.get("aggregate_parent_worker_ru_maxrss_bytes", -1)) > HOST_RAM_CAP_BYTES:
        raise PredictiveValidationError("calibration chain exceeded host RAM cap")
    return observations, forecast_signature, receipt


def run_calibration(args: argparse.Namespace) -> dict[str, Any]:
    output = repo_path(args.output_root, label="output root")
    output.mkdir(parents=True, exist_ok=args.resume)
    summary_path = output / "calibration.json"
    started = time.perf_counter()
    source_signature = execution_source_signature()
    seed_contract = seed_domain_contract(args.q)
    target = complexity_posterior_target(args.q, jit_compile=True)
    material_contract = calibration_material_contract(
        args,
        source_signature=source_signature,
        seed_contract=seed_contract,
        target_signature=target.target_signature(),
    )
    prior_seconds = 0.0
    if args.resume and not summary_path.is_file():
        raise PredictiveValidationError("calibration resume requires calibration.json")
    if args.resume:
        previous = strict_json(summary_path)
        if previous.get("material_contract") != material_contract:
            raise PredictiveValidationError("calibration resume material contract mismatch")
        if previous.get("status") == "CALIBRATION_FROZEN":
            validate_calibration(args.q, args.output_root / "calibration.json")
            return previous
        prior_seconds = float(previous["run_manifest"]["charged_seconds"])
    budget = Budget(args.cap_seconds, prior_seconds=prior_seconds)
    roots = calibration_seed_roots(args.q)
    worker_config = CPUForecastPoolConfig(
        worker_factory_path=(
            "bayesfilter.nonlinear.ssl_lstm_complexity_predictive_tf:"
            "complexity_forecast_worker_factory"
        ),
        worker_config={"q": args.q},
        worker_count=FORECAST_WORKERS_BY_Q[args.q],
        cores_per_worker=1,
    )
    banks = []
    signatures = []
    worker_receipts = []
    truth = np.repeat(
        np.asarray(PRIOR_CENTER.numpy(), dtype=np.float64)[None, :],
        CALIBRATION_DRAWS_PER_CHAIN,
        axis=0,
    )
    missing = []
    for chain_index, root in enumerate(roots):
        loaded = load_calibration_chain(
            output=output,
            q=args.q,
            chain_index=chain_index,
            root=root,
            source_signature=source_signature,
        )
        if loaded is None:
            missing.append((chain_index, root))
            continue
        observations, signature, receipt = loaded
        banks.append(observations)
        signatures.append(signature)
        worker_receipts.append(receipt)
        elapsed = float(receipt.get("elapsed_seconds", 0.0))
        budget.observe_forecast(elapsed / CALIBRATION_DRAWS_PER_CHAIN)
    resource_stop = None
    hard_veto = None
    if missing:
        ordered = {
            int(receipt["chain_index"]): (bank, signature, receipt)
            for bank, signature, receipt in zip(
                banks, signatures, worker_receipts, strict=True
            )
        }
        launched_new_chain = False
        try:
            with CPUForecastPool(worker_config) as pool:
                for chain_index, root in missing:
                    if execution_source_signature() != source_signature:
                        raise PredictiveValidationError(
                            "execution source drift during calibration"
                        )
                    seeds = forecast_seeds_from_root(root, CALIBRATION_DRAWS_PER_CHAIN)
                    reserve = budget.forecast_reserve(
                        CALIBRATION_DRAWS_PER_CHAIN,
                        cold=not launched_new_chain,
                    )
                    budget.require(reserve)
                    chain_started = time.perf_counter()
                    _means, _variances, observations_np, metadata = pool.evaluate(
                        truth,
                        seeds,
                        request_id=f"q{args.q}-calibration-chain-{chain_index}",
                    )
                    elapsed = time.perf_counter() - chain_started
                    budget.observe_forecast(elapsed / CALIBRATION_DRAWS_PER_CHAIN)
                    launched_new_chain = True
                    aggregate_rss = enforce_forecast_memory(metadata)
                    observations = tf.constant(observations_np, tf.float64)
                    tensor_path = output / f"calibration-chain-{chain_index:02d}.tftensor"
                    tensor_hash = write_tensor(tensor_path, observations)
                    seed_hash = hashlib.sha256(seeds.tobytes()).hexdigest()
                    observation_hash = hashlib.sha256(
                        np.ascontiguousarray(observations_np).tobytes()
                    ).hexdigest()
                    signature = payload_sha256(
                        {
                            "root": list(root),
                            "seed_hash": seed_hash,
                            "observation_hash": observation_hash,
                        }
                    )
                    receipt = {
                        "schema": SCHEMA,
                        "mode": "calibrate-chain",
                        "q": args.q,
                        "chain_index": chain_index,
                        "root_seed": list(root),
                        "seed_hash": seed_hash,
                        "observation_hash": observation_hash,
                        "observation_tensor_sha256": tensor_hash,
                        "forecast_signature": signature,
                        "elapsed_seconds": elapsed,
                        "resource_reserve_seconds_before_launch": reserve,
                        "aggregate_parent_worker_ru_maxrss_bytes": aggregate_rss,
                        "execution_source_signature": source_signature,
                        "resumed": False,
                        "worker_metadata": json_safe(metadata),
                    }
                    write_json(
                        output / f"calibration-chain-{chain_index:02d}.json",
                        receipt,
                    )
                    ordered[chain_index] = (observations, signature, receipt)
                    partial = {
                        "schema": SCHEMA,
                        "mode": "calibrate",
                        "status": "RUNNING",
                        "q": args.q,
                        "material_contract": material_contract,
                        "completed_chain_indices": sorted(ordered),
                        "run_manifest": run_manifest(args, budget),
                    }
                    write_json(summary_path, partial, replace=True)
        except ResourceStop as exc:
            resource_stop = str(exc)
        except (HostMemoryVeto, PredictiveValidationError) as exc:
            hard_veto = str(exc)
        banks = [ordered[index][0] for index in sorted(ordered)]
        signatures = [ordered[index][1] for index in sorted(ordered)]
        worker_receipts = [ordered[index][2] for index in sorted(ordered)]
    if resource_stop is not None:
        payload = {
            "schema": SCHEMA,
            "mode": "calibrate",
            "status": "RESOURCE_STOP",
            "q": args.q,
            "material_contract": material_contract,
            "completed_chain_indices": [
                int(receipt["chain_index"]) for receipt in worker_receipts
            ],
            "resource_stop": resource_stop,
            "retained_input_used": False,
            "run_manifest": run_manifest(args, budget),
            "nonclaims": [
                "partial calibration checkpoints only",
                "no frozen calibration scale",
                "no retained, HMC, or predictive claim",
            ],
        }
        write_json(summary_path, payload, replace=True)
        return payload
    if hard_veto is not None:
        payload = {
            "schema": SCHEMA,
            "mode": "calibrate",
            "status": "HARD_VETO",
            "q": args.q,
            "material_contract": material_contract,
            "completed_chain_indices": [
                int(receipt["chain_index"]) for receipt in worker_receipts
            ],
            "hard_veto": hard_veto,
            "retained_input_used": False,
            "run_manifest": run_manifest(args, budget),
            "nonclaims": [
                "partial calibration checkpoints only",
                "no frozen calibration scale",
                "no retained, HMC, or predictive claim",
            ],
        }
        write_json(summary_path, payload, replace=True)
        return payload
    if len(banks) != len(roots):
        raise PredictiveValidationError("calibration completed without every chain")
    calibration = calibration_from_observation_banks(
        tuple(banks),
        q=args.q,
        seed_roots=roots,
        target_signature=target.target_signature(),
        forecast_signatures=tuple(signatures),
    )
    bank_path = output / "calibration-observation-banks.tftensor"
    stacked_banks = tf.stack(banks, axis=0)
    if bank_path.is_file():
        try:
            tf.debugging.assert_equal(parse_tensor(bank_path), stacked_banks)
        except tf.errors.InvalidArgumentError as exc:
            raise PredictiveValidationError(
                "existing calibration observation bank content mismatch"
            ) from exc
        bank_hash = sha256(bank_path)
    else:
        bank_hash = write_tensor(bank_path, stacked_banks)
    wall = time.perf_counter() - started
    payload = calibration_receipt(
        calibration,
        args.q,
        forecast_signatures=tuple(signatures),
        source_signature=source_signature,
        observation_bank_binding={
            "path": bank_path.relative_to(ROOT).as_posix(),
            "sha256": bank_hash,
            "shape": [
                len(banks),
                CALIBRATION_DRAWS_PER_CHAIN,
                FORECAST_REPLICATION_COUNT,
                FORECAST_HORIZON,
            ],
        },
        seed_domain_contract=seed_contract,
    )
    payload["worker_receipts"] = worker_receipts
    payload["material_contract"] = material_contract
    payload["run_manifest"] = run_manifest(args, budget)
    payload["wall_seconds"] = wall
    write_json(summary_path, payload, replace=args.resume or summary_path.exists())
    return payload


def forecast_seeds_from_root(root: tuple[int, int], count: int) -> np.ndarray:
    root_tensor = tf.constant(root, tf.int32)
    seeds = [
        tf.random.experimental.stateless_fold_in(
            root_tensor, tf.constant(index, tf.int32), alg="philox"
        ).numpy()
        for index in range(int(count))
    ]
    return np.asarray(seeds, dtype=np.int32)


def validate_calibration(q: int, path: Path) -> dict[str, Any]:
    resolved = repo_path(path, label="calibration receipt")
    payload = strict_json(resolved)
    if payload.get("schema") != SCHEMA or payload.get("mode") != "calibrate":
        raise PredictiveValidationError("calibration receipt schema/mode mismatch")
    if payload.get("status") != "CALIBRATION_FROZEN" or int(payload.get("q", -1)) != q:
        raise PredictiveValidationError("calibration receipt status/q mismatch")
    if payload.get("retained_input_used") is not False:
        raise PredictiveValidationError("calibration receipt used retained input")
    if payload.get("seed_domain") != "q_specific_calibration_truth_fixture":
        raise PredictiveValidationError("calibration seed-domain mismatch")
    if payload.get("seed_domain_contract") != seed_domain_contract(q):
        raise PredictiveValidationError("calibration seed-domain contract mismatch")
    if payload.get("execution_source_signature") != execution_source_signature():
        raise PredictiveValidationError("calibration execution-source drift")
    roots = tuple(tuple(int(value) for value in row) for row in payload["seed_roots"])
    signatures = tuple(str(value) for value in payload["forecast_signatures"])
    if roots != calibration_seed_roots(q):
        raise PredictiveValidationError("calibration seed roots mismatch")
    if int(payload.get("chain_count", -1)) != len(roots):
        raise PredictiveValidationError("calibration chain count mismatch")
    if int(payload.get("draw_count_per_chain", -1)) != CALIBRATION_DRAWS_PER_CHAIN:
        raise PredictiveValidationError("calibration draw count mismatch")
    if int(payload.get("replication_count", -1)) != FORECAST_REPLICATION_COUNT:
        raise PredictiveValidationError("calibration replication count mismatch")
    receipts = payload.get("worker_receipts")
    if not isinstance(receipts, list) or len(receipts) != len(roots):
        raise PredictiveValidationError("calibration worker receipts mismatch")
    for index, (root, receipt, signature) in enumerate(
        zip(roots, receipts, signatures, strict=True)
    ):
        seeds = forecast_seeds_from_root(root, CALIBRATION_DRAWS_PER_CHAIN)
        expected_seed_hash = hashlib.sha256(seeds.tobytes()).hexdigest()
        if int(receipt.get("chain_index", -1)) != index:
            raise PredictiveValidationError("calibration worker chain index mismatch")
        if tuple(receipt.get("root_seed", ())) != root:
            raise PredictiveValidationError("calibration worker root mismatch")
        if receipt.get("seed_hash") != expected_seed_hash:
            raise PredictiveValidationError("calibration worker seed hash mismatch")
        expected_forecast_signature = payload_sha256(
            {
                "root": list(root),
                "seed_hash": expected_seed_hash,
                "observation_hash": receipt.get("observation_hash"),
            }
        )
        if receipt.get("forecast_signature") != expected_forecast_signature:
            raise PredictiveValidationError("calibration worker signature mismatch")
        if signature != expected_forecast_signature:
            raise PredictiveValidationError("calibration forecast signature mismatch")
        if int(receipt.get("aggregate_parent_worker_ru_maxrss_bytes", -1)) > HOST_RAM_CAP_BYTES:
            raise PredictiveValidationError("calibration receipt exceeded host RAM cap")
    center = tf.constant(payload["center"], tf.float64)
    scale = tf.constant(payload["scale"], tf.float64)
    if center.shape != (10,) or scale.shape != (10,) or not bool(
        tf.reduce_all(tf.math.is_finite(center)).numpy()
        and tf.reduce_all(tf.math.is_finite(scale)).numpy()
        and tf.reduce_all(scale > 0.0).numpy()
    ):
        raise PredictiveValidationError("calibration center/scale invalid")
    target = complexity_posterior_target(q, jit_compile=True)
    if payload.get("target_signature") != target.target_signature():
        raise PredictiveValidationError("calibration target signature mismatch")
    bank = payload.get("observation_bank")
    if not isinstance(bank, Mapping):
        raise PredictiveValidationError("calibration observation bank missing")
    bank_path = repo_path(Path(str(bank["path"])), label="calibration observation bank")
    if sha256(bank_path) != bank.get("sha256"):
        raise PredictiveValidationError("calibration observation bank hash mismatch")
    bank_tensor = parse_tensor(bank_path)
    expected_bank_shape = (
        len(roots),
        CALIBRATION_DRAWS_PER_CHAIN,
        FORECAST_REPLICATION_COUNT,
        FORECAST_HORIZON,
    )
    if tuple(bank_tensor.shape) != expected_bank_shape:
        raise PredictiveValidationError("calibration observation bank shape mismatch")
    for index, receipt in enumerate(receipts):
        observation_hash = hashlib.sha256(
            np.ascontiguousarray(bank_tensor[index].numpy()).tobytes()
        ).hexdigest()
        if receipt.get("observation_hash") != observation_hash:
            raise PredictiveValidationError("calibration bank/receipt hash mismatch")
    replay = calibration_from_observation_banks(
        tuple(tf.unstack(bank_tensor, axis=0)),
        q=q,
        seed_roots=roots,
        target_signature=target.target_signature(),
        forecast_signatures=signatures,
    )
    try:
        tf.debugging.assert_equal(center, replay.center)
        tf.debugging.assert_equal(scale, replay.scale)
    except tf.errors.InvalidArgumentError as exc:
        raise PredictiveValidationError("calibration center/scale replay mismatch") from exc
    expected_signature = complexity_calibration_signature(
        q=q,
        chain_count=len(roots),
        draw_count_per_chain=CALIBRATION_DRAWS_PER_CHAIN,
        replication_count=FORECAST_REPLICATION_COUNT,
        seed_roots=roots,
        target_signature=target.target_signature(),
        forecast_signatures=signatures,
    )
    if payload.get("calibration_signature") != expected_signature:
        raise PredictiveValidationError("calibration signature replay mismatch")
    return {
        "path": resolved.relative_to(ROOT).as_posix(),
        "sha256": sha256(resolved),
        "center": center,
        "scale": scale,
        "calibration_signature": payload["calibration_signature"],
        "execution_source_signature": payload["execution_source_signature"],
    }


def load_phase5(q: int, path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    resolved = repo_path(path, label="Phase 5 summary")
    payload = strict_json(resolved)
    if payload.get("schema") != PHASE5_SCHEMA or payload.get("status") != "ADMITTED":
        raise PredictiveValidationError("Phase 5 summary is not ADMITTED")
    if int(payload.get("q", -1)) != q or payload.get("both_charts_admitted") is not True:
        raise PredictiveValidationError("Phase 5 q/chart admission mismatch")
    charts = payload.get("charts")
    if not isinstance(charts, Mapping) or set(charts) != set(CHARTS):
        raise PredictiveValidationError("Phase 5 chart set mismatch")
    recorded_sources = payload.get("source_bindings", {}).get("source_sha256", {})
    current_sources = {
        "target": sha256(ROOT / "bayesfilter/nonlinear/ssl_lstm_complexity_target_tf.py"),
        "adapter": sha256(ROOT / "bayesfilter/inference/batched_value_score.py"),
        "artifact_loader": sha256(ROOT / "bayesfilter/inference/neutra_artifacts.py"),
        "hmc": sha256(ROOT / "bayesfilter/inference/hmc.py"),
        "diagnostics": sha256(
            ROOT / "bayesfilter/inference/hmc_posterior_diagnostics.py"
        ),
    }
    if not isinstance(recorded_sources, Mapping) or any(
        recorded_sources.get(key) != value for key, value in current_sources.items()
    ):
        raise PredictiveValidationError("shared runtime source drift since Phase 5")
    for chart in CHARTS:
        if charts[chart]["final_admission"].get("admitted") is not True:
            raise PredictiveValidationError(f"{chart} Phase 5 admission missing")
    return payload, {
        "path": resolved.relative_to(ROOT).as_posix(),
        "sha256": sha256(resolved),
    }


def load_adapter(q: int, phase5: Mapping[str, Any], chart: str) -> tuple[Any, dict[str, Any]]:
    row = phase5["charts"][chart]
    binding = row["binding"]
    kernel = row["kernel"]
    payload_path = repo_path(Path(binding["payload_path"]), label=f"{chart} payload")
    if sha256(payload_path) != binding["payload_sha256"]:
        raise PredictiveValidationError(f"{chart} payload hash mismatch")
    target = complexity_posterior_target(q, jit_compile=True)
    artifact = load_frozen_neutra_artifact(
        strict_json(payload_path), expected_target_signature=target.target_signature()
    )
    if artifact.artifact_signature != binding["artifact_signature"]:
        raise PredictiveValidationError(f"{chart} artifact signature mismatch")
    bridge = TargetBridge(target, evidence_path="phase5_retained_hmc_admission")
    adapter = FixedTransportValueScoreAdapter(
        base_adapter=bridge,
        transport=artifact.transport,
        target_scope=f"{bridge.target_scope}:{chart}",
        runtime_backend="ssl_lstm_complexity_phase6_fixed_transport_hmc",
        evidence_path=bridge.evidence_path,
        xla_hmc_ready=True,
        full_chain_xla_diagnostic_ready=True,
        require_batch_native=True,
    )
    return adapter, kernel


def read_phase5_archives(
    phase5: Mapping[str, Any],
    chart: str,
) -> tuple[list[dict[str, Any]], tf.Tensor]:
    summary_output = repo_path(
        Path(phase5["run_manifest"]["output_root"]), label="Phase 5 output root"
    )
    archives = []
    phase5_contract = phase5.get("material_contract", {})
    expected_phase4 = phase5_contract.get("phase4_summary", {}).get("sha256")
    expected_source = phase5_contract.get("execution_source_signature")
    if not expected_phase4 or not expected_source:
        raise PredictiveValidationError("Phase 5 material contract is incomplete")
    for segment in phase5["charts"][chart]["segments"]:
        label = str(segment["label"])
        archive_dir = summary_output / "retained-private" / chart
        archive = read_archive(archive_dir, label)
        hashes = segment.get("archive_hashes", {})
        expected_hashes = {
            "private_manifest_sha256": archive["manifest_sha256"],
            "sample_sha256": archive["sample_sha256"],
            "final_state_sha256": archive["final_state_sha256"],
            "final_target_log_prob_sha256": archive[
                "final_target_log_prob_sha256"
            ],
        }
        if any(hashes.get(key) != value for key, value in expected_hashes.items()):
            raise PredictiveValidationError("Phase 5 public/archive hash mismatch")
        caller = archive["manifest"]["metadata"]["caller_metadata"]
        if caller.get("phase4_summary_sha256") != expected_phase4:
            raise PredictiveValidationError("Phase 5 archive Phase 4 binding mismatch")
        if caller.get("execution_source_signature") != expected_source:
            raise PredictiveValidationError("Phase 5 archive source binding mismatch")
        if archives:
            if caller.get("previous_manifest_sha256") != archives[-1]["manifest_sha256"]:
                raise PredictiveValidationError("Phase 5 archive lineage mismatch")
            if caller.get("previous_final_state_sha256") != archives[-1]["final_state_sha256"]:
                raise PredictiveValidationError("Phase 5 archive state lineage mismatch")
        else:
            if caller.get("previous_manifest_sha256") is not None:
                raise PredictiveValidationError("Phase 5 initial archive lineage mismatch")
            if caller.get("previous_final_state_sha256") is not None:
                raise PredictiveValidationError("Phase 5 initial state lineage mismatch")
        archives.append(archive)
    admitted_draws = int(phase5["charts"][chart]["final_admission"]["draw_count_per_chain"])
    if len(archives) * SEGMENT_RESULTS != admitted_draws:
        raise PredictiveValidationError("Phase 5 archive count disagrees with admission")
    if admitted_draws > PHASE5_MAX_DRAWS or admitted_draws <= 0:
        raise PredictiveValidationError("Phase 5 admission draw count is invalid")
    return archives, archives[-1]["final_state"]


def phase6_extension_segment_count(admitted_draws: int) -> int:
    draws = int(admitted_draws)
    if draws <= 0 or draws > PHASE5_MAX_DRAWS or draws % SEGMENT_RESULTS:
        raise PredictiveValidationError("Phase 5 admission draw count is invalid")
    remaining = PREDICTIVE_DRAWS_PER_CHAIN - draws
    if remaining < 0 or remaining % SEGMENT_RESULTS:
        raise PredictiveValidationError("Phase 6 extension length is invalid")
    return remaining // SEGMENT_RESULTS


def read_archive(archive_dir: Path, label: str) -> dict[str, Any]:
    manifest_path = archive_dir / f"{label}_private_manifest.json"
    manifest = strict_json(manifest_path)
    if manifest.get("artifact_type") != "bayesfilter_private_retained_sample_hmc_archive":
        raise PredictiveValidationError("unexpected retained archive type")
    shards = manifest.get("sample_shards")
    sidecars = manifest.get("sidecars")
    if not isinstance(shards, list) or len(shards) != 1 or not isinstance(sidecars, Mapping):
        raise PredictiveValidationError("retained archive descriptor mismatch")
    shard = shards[0]
    state = sidecars["final_state"]
    target = sidecars["final_target_log_prob"]
    for row in (shard, state, target):
        row_path = repo_path(Path(row["path"]), label="archive tensor")
        if sha256(row_path) != row["sha256"]:
            raise PredictiveValidationError("archive tensor hash mismatch")
    samples = parse_tensor(repo_path(Path(shard["path"]), label="samples"))
    final_state = parse_tensor(repo_path(Path(state["path"]), label="state"))
    final_target = parse_tensor(repo_path(Path(target["path"]), label="target"))
    if tuple(samples.shape) != (SEGMENT_RESULTS, 4, 4):
        raise PredictiveValidationError("retained sample shape mismatch")
    if tuple(final_state.shape) != (4, 4) or tuple(final_target.shape) != (4,):
        raise PredictiveValidationError("retained sidecar shape mismatch")
    return {
        "samples": samples,
        "final_state": final_state,
        "final_target_log_prob": final_target,
        "manifest": manifest,
        "manifest_sha256": sha256(manifest_path),
        "sample_sha256": shard["sha256"],
        "final_state_sha256": state["sha256"],
        "final_target_log_prob_sha256": target["sha256"],
    }


def extension_seed(q: int, chart: str, index: int) -> tuple[int, int]:
    chart_offset = 0 if chart == "chart-a" else 1000
    return ROOT_SEED, 50000 + 100 * q + chart_offset + index


def extend_chart(
    *,
    q: int,
    chart: str,
    adapter: Any,
    kernel: Mapping[str, Any],
    phase5_binding: Mapping[str, Any],
    output: Path,
    budget: Budget,
    source_signature: str,
) -> list[dict[str, Any]]:
    prior, current = read_phase5_archives(phase5_binding, chart)
    prior_draws = len(prior) * SEGMENT_RESULTS
    extension_segments = phase6_extension_segment_count(prior_draws)
    runner = build_retained_sample_hmc_archive_runner(
        adapter,
        current,
        RetainedSampleHMCArchiveConfig(
            num_results=SEGMENT_RESULTS,
            num_burnin_steps=0,
            step_size=float(kernel["step_size"]),
            num_leapfrog_steps=int(kernel["num_leapfrog_steps"]),
            seed=extension_seed(q, chart, 0),
            use_xla=True,
            target_scope=adapter.target_scope,
        ),
    )
    extension_dir = output / "extension-private" / chart
    extension = []
    previous = prior[-1]
    kernel_signature = payload_sha256(kernel)
    for index in range(extension_segments):
        if execution_source_signature() != source_signature:
            raise PredictiveValidationError("execution source drift during HMC extension")
        transition_leapfrogs = SEGMENT_RESULTS * int(kernel["num_leapfrog_steps"])
        label = f"{chart}-phase6-extension-{index:03d}"
        manifest_path = extension_dir / f"{label}_private_manifest.json"
        if manifest_path.is_file():
            archive = read_archive(extension_dir, label)
        else:
            reserve = budget.hmc_reserve(
                transition_leapfrogs,
                cold=int(getattr(runner, "_call_count", 0)) == 0,
            )
            budget.require(reserve)
            started = time.perf_counter()
            result = runner.run(
                archive_dir=extension_dir,
                archive_label=label,
                current_state=current,
                seed=extension_seed(q, chart, index),
                step_size=float(kernel["step_size"]),
                metadata={
                    "schema": SCHEMA,
                    "chart": chart,
                    "segment_index": index,
                    "segment_seed": list(extension_seed(q, chart, index)),
                    "phase5_summary_sha256": phase5_binding["_summary_sha256"],
                    "execution_source_signature": source_signature,
                    "kernel_signature": kernel_signature,
                    "target_signature": adapter.base_adapter.target_signature(),
                    "adapter_signature": adapter.adapter_signature(),
                    "previous_manifest_sha256": previous["manifest_sha256"],
                    "previous_final_state_sha256": previous["final_state_sha256"],
                },
                overwrite=False,
            )
            elapsed = time.perf_counter() - started
            budget.observe_hmc(elapsed / transition_leapfrogs)
            diagnostics = json_safe(result.diagnostics)
            if not diagnostics.get("retained_samples_all_finite"):
                raise PredictiveValidationError("nonfinite Phase 6 extension samples")
            if diagnostics.get("divergence_count") not in (None, 0):
                raise PredictiveValidationError("positive divergence in Phase 6 extension")
            archive = read_archive(extension_dir, label)
        caller = archive["manifest"]["metadata"]["caller_metadata"]
        if caller.get("phase5_summary_sha256") != phase5_binding["_summary_sha256"]:
            raise PredictiveValidationError("Phase 6 extension Phase 5 binding mismatch")
        if caller.get("execution_source_signature") != source_signature:
            raise PredictiveValidationError("Phase 6 extension source binding mismatch")
        if caller.get("kernel_signature") != kernel_signature:
            raise PredictiveValidationError("Phase 6 extension kernel binding mismatch")
        if caller.get("target_signature") != adapter.base_adapter.target_signature():
            raise PredictiveValidationError("Phase 6 extension target binding mismatch")
        if caller.get("adapter_signature") != adapter.adapter_signature():
            raise PredictiveValidationError("Phase 6 extension adapter binding mismatch")
        if caller.get("segment_seed") != list(extension_seed(q, chart, index)):
            raise PredictiveValidationError("Phase 6 extension seed binding mismatch")
        if caller.get("previous_manifest_sha256") != previous["manifest_sha256"]:
            raise PredictiveValidationError("Phase 6 extension lineage mismatch")
        if caller.get("previous_final_state_sha256") != previous["final_state_sha256"]:
            raise PredictiveValidationError("Phase 6 extension state lineage mismatch")
        extension.append(archive)
        previous = archive
        current = archive["final_state"]
        enforce_parent_memory()
    return [*prior, *extension]


def forecast_seeds(q: int, chart: str, start: int, count: int) -> np.ndarray:
    chart_offset = 0 if chart == "chart-a" else 1_000_000
    indices = np.arange(start, start + count, dtype=np.int64)
    return np.stack(
        (
            np.full(count, ROOT_SEED, dtype=np.int32),
            (60000 + 100 * q + chart_offset + indices).astype(np.int32),
        ),
        axis=1,
    )


def seed_domain_contract(q: int) -> dict[str, Any]:
    calibration = np.concatenate(
        [
            forecast_seeds_from_root(root, CALIBRATION_DRAWS_PER_CHAIN)
            for root in calibration_seed_roots(q)
        ],
        axis=0,
    )
    maximum_extension_segments = (PREDICTIVE_DRAWS_PER_CHAIN - 512) // SEGMENT_RESULTS
    hmc = np.asarray(
        [
            extension_seed(q, chart, index)
            for chart in CHARTS
            for index in range(maximum_extension_segments)
        ],
        dtype=np.int32,
    )
    forecast = np.concatenate(
        [
            forecast_seeds(q, chart, 0, 4 * PREDICTIVE_DRAWS_PER_CHAIN)
            for chart in CHARTS
        ],
        axis=0,
    )
    sets = {
        "calibration": {tuple(row) for row in calibration.tolist()},
        "hmc_extension": {tuple(row) for row in hmc.tolist()},
        "predictive_forecast": {tuple(row) for row in forecast.tolist()},
    }
    disjoint = (
        sets["calibration"].isdisjoint(sets["hmc_extension"])
        and sets["calibration"].isdisjoint(sets["predictive_forecast"])
        and sets["hmc_extension"].isdisjoint(sets["predictive_forecast"])
    )
    if not disjoint:
        raise PredictiveValidationError("Phase 6 seed domains overlap")
    return {
        "q": q,
        "pairwise_disjoint": True,
        "counts": {key: len(value) for key, value in sets.items()},
        "sha256": {
            "calibration": hashlib.sha256(calibration.tobytes()).hexdigest(),
            "hmc_extension": hashlib.sha256(hmc.tobytes()).hexdigest(),
            "predictive_forecast": hashlib.sha256(forecast.tobytes()).hexdigest(),
        },
    }


def forecast_chart(
    *,
    q: int,
    chart: str,
    theta_chain_major: tf.Tensor,
    calibration: Mapping[str, Any],
    output: Path,
    budget: Budget,
    source_signature: str,
) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor, list[dict[str, Any]]]:
    workers = FORECAST_WORKERS_BY_Q[q]
    config = CPUForecastPoolConfig(
        worker_factory_path=(
            "bayesfilter.nonlinear.ssl_lstm_complexity_predictive_tf:"
            "complexity_forecast_worker_factory"
        ),
        worker_config={"q": q},
        worker_count=workers,
        cores_per_worker=1,
    )
    flat = tf.reshape(theta_chain_major, (-1, 4)).numpy()
    theta_hash = hashlib.sha256(np.ascontiguousarray(flat).tobytes()).hexdigest()
    means_blocks = []
    variance_blocks = []
    observation_blocks = []
    receipts = []
    block_root = output / "forecast-blocks" / chart
    launched_new_block = False
    with CPUForecastPool(config) as pool:
        for start in range(0, flat.shape[0], FORECAST_BLOCK_DRAWS):
            if execution_source_signature() != source_signature:
                raise PredictiveValidationError("execution source drift during forecast")
            stop = min(start + FORECAST_BLOCK_DRAWS, flat.shape[0])
            block_path = block_root / f"block-{start:06d}.json"
            mean_path = block_root / f"block-{start:06d}-means.tftensor"
            variance_path = block_root / f"block-{start:06d}-variances.tftensor"
            observation_path = block_root / f"block-{start:06d}-observations.tftensor"
            if block_path.is_file():
                receipt = strict_json(block_path)
                expected_input_hash = hashlib.sha256(
                    np.ascontiguousarray(flat[start:stop]).tobytes()
                ).hexdigest()
                if receipt.get("theta_block_sha256") != expected_input_hash:
                    raise PredictiveValidationError("forecast theta block hash mismatch")
                if receipt.get("schema") != SCHEMA or receipt.get("q") != q:
                    raise PredictiveValidationError("forecast receipt schema/q mismatch")
                if receipt.get("chart") != chart:
                    raise PredictiveValidationError("forecast receipt chart mismatch")
                if receipt.get("start") != start or receipt.get("stop") != stop:
                    raise PredictiveValidationError("forecast receipt range mismatch")
                if receipt.get("theta_chain_major_sha256") != theta_hash:
                    raise PredictiveValidationError("forecast full-theta binding mismatch")
                if receipt.get("calibration_sha256") != calibration["sha256"]:
                    raise PredictiveValidationError("forecast calibration binding mismatch")
                if receipt.get("calibration_signature") != calibration["calibration_signature"]:
                    raise PredictiveValidationError("forecast calibration signature mismatch")
                if receipt.get("execution_source_signature") != source_signature:
                    raise PredictiveValidationError("forecast source binding mismatch")
                seeds = forecast_seeds(q, chart, start, stop - start)
                if receipt.get("seed_hash") != hashlib.sha256(seeds.tobytes()).hexdigest():
                    raise PredictiveValidationError("forecast seed binding mismatch")
                if sha256(mean_path) != receipt["mean_sha256"]:
                    raise PredictiveValidationError("forecast mean block hash mismatch")
                if sha256(variance_path) != receipt["variance_sha256"]:
                    raise PredictiveValidationError("forecast variance block hash mismatch")
                if sha256(observation_path) != receipt["observation_sha256"]:
                    raise PredictiveValidationError("forecast observation block hash mismatch")
                means = parse_tensor(mean_path)
                variances = parse_tensor(variance_path)
                observations = parse_tensor(observation_path)
                if int(receipt.get("aggregate_parent_worker_ru_maxrss_bytes", -1)) > HOST_RAM_CAP_BYTES:
                    raise PredictiveValidationError("forecast receipt exceeded host RAM cap")
            else:
                item_count = stop - start
                reserve = budget.forecast_reserve(
                    item_count, cold=not launched_new_block
                )
                budget.require(reserve)
                seeds = forecast_seeds(q, chart, start, item_count)
                started = time.perf_counter()
                means_np, variance_np, observations_np, metadata = pool.evaluate(
                    flat[start:stop],
                    seeds,
                    request_id=f"q{q}-{chart}-forecast-{start}-{stop}",
                )
                elapsed = time.perf_counter() - started
                budget.observe_forecast(elapsed / item_count)
                launched_new_block = True
                aggregate_rss = enforce_forecast_memory(metadata)
                means = tf.constant(means_np, tf.float64)
                variances = tf.constant(variance_np, tf.float64)
                observations = tf.constant(observations_np, tf.float64)
                mean_hash = write_tensor(mean_path, means)
                variance_hash = write_tensor(variance_path, variances)
                observation_hash = write_tensor(observation_path, observations)
                receipt = {
                    "schema": SCHEMA,
                    "q": q,
                    "chart": chart,
                    "start": start,
                    "stop": stop,
                    "seed_hash": hashlib.sha256(seeds.tobytes()).hexdigest(),
                    "theta_chain_major_sha256": theta_hash,
                    "theta_block_sha256": hashlib.sha256(
                        np.ascontiguousarray(flat[start:stop]).tobytes()
                    ).hexdigest(),
                    "mean_sha256": mean_hash,
                    "variance_sha256": variance_hash,
                    "observation_sha256": observation_hash,
                    "worker_metadata": json_safe(metadata),
                    "elapsed_seconds": elapsed,
                    "resource_reserve_seconds_before_launch": reserve,
                    "execution_source_signature": source_signature,
                    "aggregate_parent_worker_ru_maxrss_bytes": aggregate_rss,
                    "calibration_signature": calibration["calibration_signature"],
                    "calibration_sha256": calibration["sha256"],
                }
                write_json(block_path, receipt)
            means_blocks.append(means)
            variance_blocks.append(variances)
            observation_blocks.append(observations)
            receipts.append(receipt)
    shape = (4, PREDICTIVE_DRAWS_PER_CHAIN, FORECAST_REPLICATION_COUNT, FORECAST_HORIZON)
    means = tf.reshape(tf.concat(means_blocks, axis=0), shape)
    variances = tf.reshape(tf.concat(variance_blocks, axis=0), shape)
    observations = tf.reshape(tf.concat(observation_blocks, axis=0), shape)
    if not bool(
        tf.reduce_all(tf.math.is_finite(means)).numpy()
        and tf.reduce_all(tf.math.is_finite(variances)).numpy()
        and tf.reduce_all(tf.math.is_finite(observations)).numpy()
        and tf.reduce_all(variances > 0.0).numpy()
    ):
        raise PredictiveValidationError("forecast blocks are nonfinite or nonpositive")
    center = calibration["center"][tf.newaxis, tf.newaxis, tf.newaxis, :]
    scale = calibration["scale"][tf.newaxis, tf.newaxis, tf.newaxis, :]
    return (
        (means - center) / scale,
        variances / tf.square(scale),
        (observations - center) / scale,
        receipts,
    )


def load_theta_draws(adapter: Any, archives: list[dict[str, Any]]) -> tf.Tensor:
    z = tf.concat([row["samples"] for row in archives], axis=0)
    flat = tf.reshape(z, (-1, 4))
    chunks = []
    for start in range(0, int(flat.shape[0]), THETA_MAP_CHUNK_ROWS):
        chunks.append(
            adapter.latent_to_position(flat[start : start + THETA_MAP_CHUNK_ROWS])
        )
    theta = tf.reshape(tf.concat(chunks, axis=0), (PREDICTIVE_DRAWS_PER_CHAIN, 4, 4))
    return tf.transpose(theta, (1, 0, 2))


def sampler_extension_screen(
    z_chain_major: tf.Tensor,
    theta_chain_major: tf.Tensor,
    archives: list[dict[str, Any]],
) -> dict[str, Any]:
    if z_chain_major.shape != theta_chain_major.shape or z_chain_major.shape != (
        4,
        PREDICTIVE_DRAWS_PER_CHAIN,
        4,
    ):
        raise PredictiveValidationError("Phase 6 sampler screen shape mismatch")
    hard_vetoes = []
    for name, values in (("z", z_chain_major), ("theta", theta_chain_major)):
        if not bool(tf.reduce_all(tf.math.is_finite(values)).numpy()):
            hard_vetoes.append(f"nonfinite_{name}")
    divergence_statuses = []
    draw_major_z = tf.transpose(z_chain_major, (1, 0, 2))
    initial_state = tf.constant(INITIAL_Z, tf.float64)
    previous = tf.concat((initial_state[tf.newaxis, ...], draw_major_z[:-1]), axis=0)
    moved = [
        bool(value)
        for value in tf.reduce_any(tf.not_equal(draw_major_z, previous), axis=(0, 2))
        .numpy()
        .tolist()
    ]
    if not all(moved):
        hard_vetoes.append("unmoved_chain")
    for archive in archives:
        diagnostics = archive["manifest"]["diagnostics_private_metadata"]
        health = diagnostics["sampler_health_diagnostics"]
        if int(health["log_accept_ratio"]["nonfinite_count"]) > 0:
            hard_vetoes.append("nonfinite_log_accept_ratio")
        if int(health["target_log_prob"]["nonfinite_count"]) > 0:
            hard_vetoes.append("nonfinite_target_log_prob")
        divergence_statuses.append(str(diagnostics["native_divergence_status"]))
        if diagnostics.get("divergence_count") not in (None, 0):
            hard_vetoes.append("positive_native_divergence")
    diagnostics = {}
    for name, values in (("z", z_chain_major), ("theta", theta_chain_major)):
        row = json_safe(compute_coordinate_diagnostics(values))
        failures = []
        rhat = row["rank_normalized_split_rhat"]["maximum"]
        bulk = row["rank_normalized_ess"]["bulk"]
        tail = row["rank_normalized_ess"]["tail"]
        ratio = row["mean"]["mcse_sd_ratio"]
        if max(float(value) for value in rhat) > 1.01:
            failures.append("rank_normalized_split_rhat_above_threshold")
        if min(float(value) for value in bulk) < 400.0:
            failures.append("bulk_ess_below_threshold")
        if min(float(value) for value in tail) < 400.0:
            failures.append("tail_ess_below_threshold")
        if max(float(value) for value in ratio) > 0.05:
            failures.append("mcse_sd_ratio_above_threshold")
        diagnostics[name] = {"values": row, "failures": failures}
        hard_vetoes.extend(f"{name}:{failure}" for failure in failures)
    return {
        "passed": not hard_vetoes,
        "hard_vetoes": list(dict.fromkeys(hard_vetoes)),
        "native_divergence_statuses": divergence_statuses,
        "chain_moved": moved,
        "coordinate_diagnostics": diagnostics,
        "role": "renewed_sampler_validity_after_predictive_sample_size_extension",
    }


def predictive_decision(
    means: Mapping[str, tf.Tensor],
    variances: Mapping[str, tf.Tensor],
) -> dict[str, Any]:
    feature = {
        chart: conditional_mean_log_variance_influence(means[chart], variances[chart])
        for chart in CHARTS
    }
    estimate = feature["chart-a"].feature_estimate - feature["chart-b"].feature_estimate
    influence = tf.concat(
        (2.0 * feature["chart-a"].influence_values, -2.0 * feature["chart-b"].influence_values),
        axis=0,
    )
    hac = chain_bartlett_long_run_covariance(
        influence,
        bandwidth_multiplier=HAC_MULTIPLIER,
        ridge_ladder=RIDGE_LADDER,
        condition_number_max=CONDITION_NUMBER_MAX,
    )
    if not hac.inference_admissible:
        return {
            "status": "INVALID_HARD_VETO",
            "hard_veto_codes": ["BARTLETT_HAC_NOT_ADMISSIBLE_WITH_ZERO_RIDGE"],
            "feature_estimate_a_minus_b": json_safe(estimate),
            "hac": {
                "bandwidth": hac.bandwidth,
                "bandwidth_multiplier": hac.bandwidth_multiplier,
                "condition_number": json_safe(hac.condition_number),
                "eigenvalues": json_safe(hac.eigenvalues),
                "ridge_ladder": list(RIDGE_LADDER),
                "inference_admissible": False,
                "status": json_safe(hac.status),
            },
            "bounds": None,
            "acceptable_loss": ACCEPTABLE_LOSS,
            "nonclaims": [
                "invalid predictive inference is not a material-difference result",
                "not an oracle-posterior comparison",
                "not a sampler or transport ranking",
            ],
        }
    average_loss = proper_score_loss(tf.fill((10,), tf.constant(0.1, tf.float64)))
    bounds = split_quadratic_loss_confidence_bounds(
        estimate,
        hac.regularized_covariance,
        average_loss,
        average_alpha=AVERAGE_ALPHA,
        horizon_alpha=HORIZON_ALPHA,
        familywise_alpha=FAMILYWISE_ALPHA,
    )
    decision = classify_split_proper_score_equivalence(
        bounds,
        acceptable_average_loss=tf.constant(ACCEPTABLE_LOSS, tf.float64),
        acceptable_horizon_loss=tf.constant(ACCEPTABLE_LOSS, tf.float64),
    )
    return {
        "status": decision.status,
        "hard_veto_codes": list(decision.hard_veto_codes),
        "feature_estimate_a_minus_b": json_safe(estimate),
        "hac": {
            "bandwidth": hac.bandwidth,
            "bandwidth_multiplier": hac.bandwidth_multiplier,
            "condition_number": json_safe(hac.condition_number),
            "eigenvalues": json_safe(hac.eigenvalues),
            "ridge_ladder": list(RIDGE_LADDER),
            "inference_admissible": hac.inference_admissible,
            "status": json_safe(hac.status),
        },
        "bounds": {
            "average_point_loss": json_safe(bounds.average_point_loss),
            "average_lower_bound": json_safe(bounds.average_lower_bound),
            "average_upper_bound": json_safe(bounds.average_upper_bound),
            "horizon_point_losses": json_safe(bounds.horizon_point_losses),
            "horizon_lower_bounds": json_safe(bounds.horizon_lower_bounds),
            "horizon_upper_bounds": json_safe(bounds.horizon_upper_bounds),
            "allocated_familywise_alpha": json_safe(bounds.allocated_familywise_alpha),
            "inference_admissible": bounds.inference_admissible,
            "status": json_safe(bounds.status),
        },
        "acceptable_loss": ACCEPTABLE_LOSS,
        "nonclaims": [
            "replication-stability decision between two admitted charts",
            "not an oracle-posterior comparison",
            "not a sampler or transport ranking",
        ],
    }


def recovery_diagnostics(theta: Mapping[str, tf.Tensor]) -> dict[str, Any]:
    rows = {}
    truth = PRIOR_CENTER
    for chart, samples in theta.items():
        flat = tf.reshape(samples, (-1, 4))
        mean = tf.reduce_mean(flat, axis=0)
        centered = flat - mean
        sd = tf.sqrt(
            tf.reduce_sum(tf.square(centered), axis=0)
            / tf.cast(tf.shape(flat)[0] - 1, tf.float64)
        )
        lower = tf.sort(flat, axis=0)[int(flat.shape[0] * 0.025)]
        upper = tf.sort(flat, axis=0)[int(flat.shape[0] * 0.975)]
        rows[chart] = {
            "posterior_mean": json_safe(mean),
            "posterior_sd": json_safe(sd),
            "standardized_mean_error": json_safe(tf.abs(mean - truth) / sd),
            "marginal_95pct_truth_coverage": json_safe((lower <= truth) & (truth <= upper)),
            "role": "descriptive_single_synthetic_dataset",
        }
    return rows


def _plot_series(
    draw: ImageDraw.ImageDraw,
    values: np.ndarray,
    box: tuple[int, int, int, int],
    color: str,
    *,
    lower: float,
    upper: float,
) -> None:
    x0, y0, x1, y1 = box
    scale = max(upper - lower, 1.0e-12)
    points = []
    for index, value in enumerate(values):
        x = x0 + (x1 - x0) * index / max(len(values) - 1, 1)
        y = y1 - (y1 - y0) * (float(value) - lower) / scale
        points.append((int(x), int(np.clip(y, y0, y1))))
    if len(points) > 1:
        draw.line(points, fill=color, width=2)


def write_path_plot_artifacts(
    paths_by_chart: Mapping[str, tf.Tensor], output: Path
) -> dict[str, Any]:
    paths = {}
    draw_indices = (0, 3071, 6143, 9215, 12287)
    for chart in CHARTS:
        rows = []
        for chain in range(4):
            for draw in draw_indices:
                rows.append(
                    {
                        "chain": chain,
                        "draw": draw,
                        "simulated_observation_path": json_safe(
                            paths_by_chart[chart][chain, draw, 0]
                        ),
                    }
                )
        paths[chart] = rows
    path = output / "path-plot-data.json"
    write_json(path, {"schema": SCHEMA, "paths": paths})
    image = Image.new("RGB", (1200, 700), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text(
        (18, 12),
        "Dispersed posterior simulated forecast paths; descriptive only",
        fill="#111111",
        font=font,
    )
    arrays = {
        chart: np.asarray(paths_by_chart[chart].numpy(), dtype=np.float64)
        for chart in CHARTS
    }
    lower = min(float(np.min(value)) for value in arrays.values())
    upper = max(float(np.max(value)) for value in arrays.values())
    for panel, chart in enumerate(CHARTS):
        x0 = 25 + 590 * panel
        box = (x0 + 38, 62, x0 + 555, 640)
        draw.rectangle((x0, 45, x0 + 570, 660), outline="#555555", width=1)
        draw.text((x0 + 8, 50), chart, fill="#111111", font=font)
        for chain in range(4):
            for draw_index in draw_indices:
                _plot_series(
                    draw,
                    arrays[chart][chain, draw_index, 0],
                    box,
                    PLOT_COLORS[chain],
                    lower=lower,
                    upper=upper,
                )
    draw.text(
        (18, 675),
        "colors identify four HMC chains; five dispersed draws per chain",
        fill="#333333",
        font=font,
    )
    png = output / "path-plots.png"
    pdf = output / "path-plots.pdf"
    image.save(png, format="PNG")
    image.convert("RGB").save(pdf, format="PDF", resolution=120.0)
    return {
        "data": {"path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path)},
        "png": {
            "path": png.relative_to(ROOT).as_posix(),
            "sha256": sha256(png),
            "bytes": png.stat().st_size,
        },
        "pdf": {
            "path": pdf.relative_to(ROOT).as_posix(),
            "sha256": sha256(pdf),
            "bytes": pdf.stat().st_size,
        },
        "role": "descriptive_explanatory_only",
        "path_type": "simulated_standardized_observation_path",
    }


def run_validation(args: argparse.Namespace) -> dict[str, Any]:
    output = repo_path(args.output_root, label="output root")
    output.mkdir(parents=True, exist_ok=args.resume)
    summary_path = output / "summary.json"
    calibration = validate_calibration(args.q, args.calibration_receipt)
    phase5, phase5_binding = load_phase5(args.q, args.phase5_summary)
    phase5["_summary_sha256"] = phase5_binding["sha256"]
    source_signature = execution_source_signature()
    phase5_admission_draws = {
        chart: int(phase5["charts"][chart]["final_admission"]["draw_count_per_chain"])
        for chart in CHARTS
    }
    material_contract = {
        "q": args.q,
        "phase5": phase5_binding,
        "calibration": {
            "path": calibration["path"],
            "sha256": calibration["sha256"],
            "calibration_signature": calibration["calibration_signature"],
        },
        "predictive_draws_per_chain": PREDICTIVE_DRAWS_PER_CHAIN,
        "phase5_admission_draws_per_chain": phase5_admission_draws,
        "execution_source_signature": source_signature,
        "forecast_workers": FORECAST_WORKERS_BY_Q[args.q],
        "forecast_replications": FORECAST_REPLICATION_COUNT,
        "forecast_horizon": FORECAST_HORIZON,
        "forecast_block_draws": FORECAST_BLOCK_DRAWS,
        "hac_multiplier": HAC_MULTIPLIER,
        "ridge_ladder": list(RIDGE_LADDER),
        "condition_number_max": CONDITION_NUMBER_MAX,
        "average_alpha": AVERAGE_ALPHA,
        "horizon_alpha": HORIZON_ALPHA,
        "familywise_alpha": FAMILYWISE_ALPHA,
        "acceptable_loss": ACCEPTABLE_LOSS,
        "host_ram_cap_bytes": HOST_RAM_CAP_BYTES,
    }
    prior_seconds = 0.0
    if args.resume and not summary_path.is_file():
        raise PredictiveValidationError("resume requires summary.json")
    if args.resume:
        previous = strict_json(summary_path)
        if previous.get("material_contract") != material_contract:
            raise PredictiveValidationError("resume material contract mismatch")
        prior_seconds = float(previous["run_manifest"]["charged_seconds"])
    budget = Budget(args.cap_seconds, prior_seconds=prior_seconds)
    for chart in CHARTS:
        for segment in phase5["charts"][chart]["segments"]:
            value = segment.get("seconds_per_transition_leapfrog")
            if value is not None:
                budget.observe_hmc(float(value))
    theta = {}
    means = {}
    variances = {}
    observation_paths = {}
    chart_receipts = {}
    resource_stop = None
    hard_veto = None
    try:
        for chart in CHARTS:
            if execution_source_signature() != source_signature:
                raise PredictiveValidationError("execution source drift during validation")
            adapter, kernel = load_adapter(args.q, phase5, chart)
            archives = extend_chart(
                q=args.q,
                chart=chart,
                adapter=adapter,
                kernel=kernel,
                phase5_binding=phase5,
                output=output,
                budget=budget,
                source_signature=source_signature,
            )
            theta[chart] = load_theta_draws(adapter, archives)
            z_chain_major = tf.transpose(
                tf.concat([row["samples"] for row in archives], axis=0),
                (1, 0, 2),
            )
            sampler_screen = sampler_extension_screen(
                z_chain_major, theta[chart], archives
            )
            if not sampler_screen["passed"]:
                raise PredictiveValidationError(
                    "Phase 6 renewed sampler screen failed: "
                    + ",".join(sampler_screen["hard_vetoes"])
                )
            (
                means[chart],
                variances[chart],
                observation_paths[chart],
                forecast_receipts,
            ) = forecast_chart(
                q=args.q,
                chart=chart,
                theta_chain_major=theta[chart],
                calibration=calibration,
                output=output,
                budget=budget,
                source_signature=source_signature,
            )
            chart_receipts[chart] = {
                "kernel": kernel,
                "sampler_extension_screen": sampler_screen,
                "retained_draws_per_chain": PREDICTIVE_DRAWS_PER_CHAIN,
                "phase5_admission_draws_per_chain": phase5_admission_draws[chart],
                "extension_segment_count": len(archives) - len(phase5["charts"][chart]["segments"]),
                "forecast_block_count": len(forecast_receipts),
                "forecast_receipt_sha256": payload_sha256(forecast_receipts),
            }
            partial = {
                "schema": SCHEMA,
                "mode": "validate",
                "status": "RUNNING",
                "q": args.q,
                "material_contract": material_contract,
                "calibration_binding": {
                    key: json_safe(value) for key, value in calibration.items()
                    if key not in {"center", "scale"}
                },
                "phase5_binding": phase5_binding,
                "charts": chart_receipts,
                "run_manifest": run_manifest(args, budget),
            }
            write_json(summary_path, partial, replace=True)
    except ResourceStop as exc:
        resource_stop = str(exc)
    except (HostMemoryVeto, PredictiveValidationError) as exc:
        hard_veto = str(exc)
    decision = None
    recovery = None
    plot = None
    if not resource_stop and not hard_veto and set(means) == set(CHARTS):
        try:
            decision = predictive_decision(means, variances)
            if decision["status"] == "INVALID_HARD_VETO":
                hard_veto = "predictive statistical hard veto: " + ",".join(
                    decision.get("hard_veto_codes", [])
                )
        except (PredictiveContractError, tf.errors.InvalidArgumentError) as exc:
            hard_veto = f"predictive statistical contract failed: {exc}"
            decision = {
                "status": "INVALID_HARD_VETO",
                "hard_veto_codes": ["PREDICTIVE_STATISTICAL_CONTRACT_FAILED"],
            }
        recovery = recovery_diagnostics(theta)
        plot = write_path_plot_artifacts(observation_paths, output)
    status = (
        "HARD_VETO"
        if hard_veto
        else "RESOURCE_STOP"
        if resource_stop
        else "PREDICTIVE_REPLICATION_PASSED"
        if decision and decision["status"] == "PASS"
        else "PREDICTIVE_MATERIAL_DIFFERENCE"
        if decision and decision["status"] == "MATERIAL_DIFFERENCE"
        else "HARD_VETO"
        if decision and decision["status"] == "INVALID_HARD_VETO"
        else "PREDICTIVE_INCONCLUSIVE"
    )
    payload = {
        "schema": SCHEMA,
        "mode": "validate",
        "status": status,
        "q": args.q,
        "material_contract": material_contract,
        "calibration_binding": {
            key: json_safe(value) for key, value in calibration.items()
            if key not in {"center", "scale"}
        },
        "phase5_binding": phase5_binding,
        "charts": chart_receipts,
        "predictive_decision": decision,
        "synthetic_recovery": recovery,
        "path_plot_artifacts": plot,
        "resource_stop": resource_stop,
        "hard_veto": hard_veto,
        "candidate_rejection_is_not_research_direction_rejection": True,
        "run_manifest": run_manifest(args, budget),
        "inference_status": {
            "hard_veto_screen": "failed" if hard_veto else "passed",
            "statistically_supported_ranking": "none",
            "descriptive_only_differences": [
                "truth recovery on one synthetic dataset",
                "continuous path and feature differences",
                "runtime and memory below cap",
            ],
            "default_readiness": "not_assessed",
            "next_evidence_needed": (
                "ordered next q rung"
                if status == "PREDICTIVE_REPLICATION_PASSED"
                else "repair named by Phase 6 result"
            ),
        },
        "nonclaims": [
            "no oracle posterior distribution",
            "no frequentist calibration from one synthetic dataset",
            "no NeuTra or sampler superiority",
            "no model adequacy or production-readiness claim",
        ],
    }
    write_json(summary_path, payload, replace=True)
    return payload


def run_manifest(args: argparse.Namespace, budget: Budget) -> dict[str, Any]:
    try:
        gpu_memory = json_safe(tf.config.experimental.get_memory_info("GPU:0"))
    except (ValueError, RuntimeError):
        gpu_memory = {"status": "unavailable"}
    return {
        "git_commit": git("rev-parse", "HEAD"),
        "command": " ".join(sys.argv),
        "environment": os.environ.get("CONDA_DEFAULT_ENV", "N/A"),
        "python": sys.version.split()[0],
        "tensorflow": tf.__version__,
        "selected_physical_gpu": SELECTED_GPU,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "logical_gpus": [device.name for device in tf.config.list_logical_devices("GPU")],
        "jit_compile": True,
        "tf32": bool(tf.config.experimental.tensor_float_32_execution_enabled()),
        "charged_seconds": budget.elapsed,
        "cap_seconds": budget.seconds,
        "host_ru_maxrss_bytes": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024),
        "host_ram_cap_bytes": HOST_RAM_CAP_BYTES,
        "gpu_allocator_memory": gpu_memory,
        "random_seed_domains": {
            "calibration": "q-specific 40000 family",
            "hmc_extension": "q/chart-specific 50000 family",
            "predictive_forecast": "q/chart-specific 60000 family",
            "pairwise_disjoint": True,
        },
        "trust_basis": (
            "cpu_hidden_calibration_truth_fixture"
            if args.mode == "calibrate"
            else "owner_designated_managed_session_visible_gpu_trusted"
        ),
        "plan": PLAN.as_posix(),
        "output_root": args.output_root.as_posix(),
    }


def contract_payload(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "mode": "contract-smoke",
        "status": "PASSED",
        "q": args.q,
        "phase5_admission_draws_per_chain": {
            "minimum": 512,
            "maximum": PHASE5_MAX_DRAWS,
            "derived_from_each_admitted_phase5_receipt": True,
        },
        "predictive_draws_per_chain": PREDICTIVE_DRAWS_PER_CHAIN,
        "phase6_extension_segment_range": {
            "if_phase5_admits_at_4096": phase6_extension_segment_count(
                PHASE5_MAX_DRAWS
            ),
            "if_phase5_admits_at_512": phase6_extension_segment_count(512),
        },
        "forecast_worker_count": FORECAST_WORKERS_BY_Q[args.q],
        "forecast_replication_count": FORECAST_REPLICATION_COUNT,
        "forecast_horizon": FORECAST_HORIZON,
        "predictive_contract": {
            "estimator": "rao_blackwell_conditional_mean_log_variance",
            "hac_multiplier": HAC_MULTIPLIER,
            "ridge_ladder": list(RIDGE_LADDER),
            "average_alpha": AVERAGE_ALPHA,
            "horizon_alpha": HORIZON_ALPHA,
            "familywise_alpha": FAMILYWISE_ALPHA,
            "acceptable_loss": ACCEPTABLE_LOSS,
        },
        "calibration_requires_no_retained_input": True,
        "calibration_execution_target": "persistent_multicore_cpu_with_cuda_hidden",
        "validation_execution_target": "gpu_xla_hmc_plus_persistent_cpu_forecast_pool",
        "path_plot_artifacts": ["json", "png", "pdf"],
        "host_ram_cap_bytes": HOST_RAM_CAP_BYTES,
        "material_execution_authorized": False,
        "nonclaims": [
            "contract/import smoke only",
            "no calibration, HMC extension, retained read, or forecast execution",
        ],
    }


def validate_args(args: argparse.Namespace) -> None:
    if args.mode == "contract-smoke":
        return
    if not args.authorize_material_run:
        raise PredictiveValidationError("material modes require --authorize-material-run")
    if args.cap_seconds is None or args.cap_seconds <= 0.0:
        raise PredictiveValidationError("material modes require a positive cap")
    if args.output_root is None:
        raise PredictiveValidationError("material modes require an explicit output root")
    repo_path(args.output_root, label="output root")
    if args.mode == "calibrate":
        if args.phase5_summary is not None or args.calibration_receipt is not None:
            raise PredictiveValidationError("calibration mode forbids retained/validation inputs")
        return
    if args.phase5_summary is None or args.calibration_receipt is None:
        raise PredictiveValidationError("validate requires Phase 5 and calibration receipts")
    repo_path(args.phase5_summary, label="Phase 5 summary")
    repo_path(args.calibration_receipt, label="calibration receipt")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("contract-smoke", "calibrate", "validate"), required=True)
    parser.add_argument("--q", type=int, choices=Q_VALUES, required=True)
    parser.add_argument("--phase5-summary", type=Path)
    parser.add_argument("--calibration-receipt", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--cap-seconds", type=float)
    parser.add_argument("--authorize-material-run", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.cap_seconds is not None and (
        not math.isfinite(args.cap_seconds) or args.cap_seconds <= 0.0
    ):
        parser.error("--cap-seconds must be finite and positive")
    if args.mode == "contract-smoke" and args.resume:
        parser.error("contract-smoke cannot resume")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    validate_args(args)
    if args.mode == "contract-smoke":
        payload = contract_payload(args)
    else:
        if args.mode == "validate":
            configure_gpu()
            payload = run_validation(args)
        else:
            payload = run_calibration(args)
    print(
        "JSON_SUMMARY "
        + json.dumps(
            {"mode": payload["mode"], "status": payload["status"], "q": payload["q"]},
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
