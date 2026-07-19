#!/usr/bin/env python3
"""Run bounded Optuna tuning for the `(32,32)` SSL-LSTM NeuTra family."""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

import tensorflow as tf


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bayesfilter.inference.neutra_artifacts import load_frozen_neutra_artifact  # noqa: E402
from bayesfilter.inference.neutra_training import (  # noqa: E402
    DSGE_PAPER_TRAINING_BATCH_SIZE,
    NeuTraReverseKLTrainer,
    ssl_lstm_tuned_capacity_neutra_config,
)
from bayesfilter.nonlinear.ssl_lstm_posterior_tf import (  # noqa: E402
    FREE_PARAMETER_NAMES,
    PRIOR_CENTER_VALUES,
    locked_ssl_lstm_posterior_target,
)


HELPER_PATH = ROOT / "docs/benchmarks/run_ssl_lstm_neutra_phase4_bounded_training_2026_07_14.py"
HELPER_SPEC = importlib.util.spec_from_file_location("optuna_phase4_helpers", HELPER_PATH)
if HELPER_SPEC is None or HELPER_SPEC.loader is None:
    raise RuntimeError("unable to load SSL-LSTM diagnostic helpers")
HELPERS = importlib.util.module_from_spec(HELPER_SPEC)
sys.modules[HELPER_SPEC.name] = HELPERS
HELPER_SPEC.loader.exec_module(HELPERS)

SCHEMA = "bayesfilter.ssl_lstm_neutra.optuna_tuning.v1"
PLAN_PATH = Path(
    "docs/plans/bayesfilter-ssl-lstm-neutra-optuna-plateau-training-repair-plan-2026-07-15.md"
)
DEFAULT_OUTPUT_ROOT = Path(
    "docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/optuna-plateau-repair"
)
BASELINE_RESULT = Path(
    "docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/"
    "capacity-32x32-diagnostic/diagnostic-result.json"
)
BASELINE_SHA256 = "5ae83bc90faf7463a5b74437cdaf904aa54112a8a9945fc2b8ebddc994b47a00"
VALIDATION_BATCH_SIZE = 64
SATURATION_MAX = 0.05
INVERSE_RADIUS_MAX = 4.30
T_CRITICAL_ONE_SIDED_95_DF63 = 1.6694022215079607


class TuningError(RuntimeError):
    """Raised when the tuning harness or evidence contract is invalid."""


class ResourceStop(TuningError):
    """Raised when the shared process budget is exhausted."""


@dataclass(frozen=True)
class Stream:
    label: str
    initialization_seed: tuple[int, int]
    training_seed: tuple[int, int]
    validation_seed: tuple[int, int]


@dataclass(frozen=True)
class TrialParameters:
    learning_rate: float
    initialization_scale: float
    gradient_clip_norm: float

    def __post_init__(self) -> None:
        if not 1.0e-4 <= float(self.learning_rate) <= 2.0e-3:
            raise ValueError("learning_rate outside study search contract")
        if float(self.initialization_scale) not in {0.005, 0.01, 0.02}:
            raise ValueError("initialization_scale outside study search contract")
        if float(self.gradient_clip_norm) not in {5.0, 10.0}:
            raise ValueError("gradient_clip_norm outside study search contract")


STREAMS = (
    Stream("seed-a", (20260715, 4101), (20260715, 5101), (20260715, 5201)),
    Stream("seed-b", (20260715, 4102), (20260715, 5102), (20260715, 5202)),
)


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical(payload: Any) -> bytes:
    return (
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def write_json(path: Path, payload: Any, *, replace: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not replace:
        raise TuningError(f"output already exists: {path}")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(canonical(payload))
    temporary.replace(path)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args: str) -> str:
    return subprocess.check_output(("git", *args), cwd=ROOT, text=True).strip()


def parse_rungs(value: str) -> tuple[int, ...]:
    try:
        rungs = tuple(int(item.strip()) for item in value.split(","))
    except ValueError as exc:
        raise argparse.ArgumentTypeError("rungs must be comma-separated integers") from exc
    if not rungs or rungs[0] <= 0 or any(right <= left for left, right in zip(rungs, rungs[1:])):
        raise argparse.ArgumentTypeError("rungs must be strictly increasing and positive")
    return rungs


def trial_parameters(trial: Any) -> TrialParameters:
    return TrialParameters(
        learning_rate=trial.suggest_float("learning_rate", 1.0e-4, 2.0e-3, log=True),
        initialization_scale=trial.suggest_categorical(
            "initialization_scale", [0.005, 0.01, 0.02]
        ),
        gradient_clip_norm=trial.suggest_categorical(
            "gradient_clip_norm", [5.0, 10.0]
        ),
    )


def fixed_timing_parameters() -> TrialParameters:
    return TrialParameters(1.0e-3, 0.01, 5.0)


def source_bindings() -> dict[str, Any]:
    baseline = ROOT / BASELINE_RESULT
    if sha256(baseline) != BASELINE_SHA256:
        raise TuningError("capacity diagnostic baseline hash drift")
    paths = {
        "runner": Path(__file__).resolve().relative_to(ROOT),
        "trainer": Path("bayesfilter/inference/neutra_training.py"),
        "controller": Path("bayesfilter/inference/neutra_training_control.py"),
        "artifact_loader": Path("bayesfilter/inference/neutra_artifacts.py"),
        "target": Path("bayesfilter/nonlinear/ssl_lstm_posterior_tf.py"),
        "plan": PLAN_PATH,
        "baseline": BASELINE_RESULT,
    }
    return {
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "source_paths": {key: value.as_posix() for key, value in paths.items()},
        "source_sha256": {
            key: sha256(ROOT / value) for key, value in paths.items()
        },
    }


def step_batch(stream: Stream, step: int) -> tf.Tensor:
    seed = tf.random.experimental.stateless_fold_in(
        tf.constant(stream.training_seed, tf.int32), int(step)
    )
    return tf.random.stateless_normal(
        (DSGE_PAPER_TRAINING_BATCH_SIZE, 4), seed=seed, dtype=tf.float64
    )


def validation_batch(stream: Stream) -> tf.Tensor:
    return tf.random.stateless_normal(
        (VALIDATION_BATCH_SIZE, 4),
        seed=tf.constant(stream.validation_seed, tf.int32),
        dtype=tf.float64,
    )


def stream_config(target: Any, stream: Stream, params: TrialParameters) -> Any:
    return ssl_lstm_tuned_capacity_neutra_config(
        dimension=4,
        fixed_translation=PRIOR_CENTER_VALUES,
        target_parameter_names=FREE_PARAMETER_NAMES,
        target_signature=target.target_signature(),
        target_adapter_signature=target.adapter_signature(),
        learning_rate=params.learning_rate,
        initialization_scale=params.initialization_scale,
        gradient_clip_norm=params.gradient_clip_norm,
        initialization_seed=stream.initialization_seed,
        jit_compile=True,
    )


def paired_interval(initial: list[float], final: list[float]) -> dict[str, float]:
    if len(initial) != VALIDATION_BATCH_SIZE or len(final) != VALIDATION_BATCH_SIZE:
        raise TuningError("paired validation batches must contain 64 rows")
    differences = [after - before for before, after in zip(initial, final)]
    mean = math.fsum(differences) / len(differences)
    variance = math.fsum((value - mean) ** 2 for value in differences) / (
        len(differences) - 1
    )
    standard_error = math.sqrt(max(variance, 0.0) / len(differences))
    return {
        "mean_difference": mean,
        "standard_error": standard_error,
        "one_sided_95_upper": mean
        + T_CRITICAL_ONE_SIDED_95_DF63 * standard_error,
    }


def validation_row(trainer: Any, validation_z: tf.Tensor, step: int) -> dict[str, Any]:
    row = HELPERS._host_validation(  # noqa: SLF001
        trainer.validation_batch(validation_z), family="dense_iaf", s_max=1.0
    )
    return {
        "step": int(step),
        "learning_rate": float(trainer.learning_rate_at(step).numpy()),
        **row,
    }


def _rung_vetoes(row: dict[str, Any]) -> list[str]:
    vetoes = []
    if not math.isfinite(float(row["mean_loss"])):
        vetoes.append("nonfinite_validation_loss")
    if float(row["saturation_fraction"]) > SATURATION_MAX:
        vetoes.append("dense_scale_saturation_above_cap")
    return vetoes


def _terminal_vetoes(
    *,
    initial: dict[str, Any],
    final: dict[str, Any],
    probes: dict[str, Any],
) -> tuple[list[str], dict[str, float]]:
    interval = paired_interval(initial["per_sample_loss"], final["per_sample_loss"])
    vetoes = _rung_vetoes(final)
    if not probes["all_finite"]:
        vetoes.append("probe_nonfinite")
    if probes["roundtrip_max_abs"] > 1.0e-9:
        vetoes.append("roundtrip_residual_above_threshold")
    if probes["moderate_shell_max_inverse_radius"] > INVERSE_RADIUS_MAX:
        vetoes.append("moderate_shell_missing_support")
    if interval["one_sided_95_upper"] >= 0.0:
        vetoes.append("heldout_loss_improvement_not_established")
    return sorted(set(vetoes)), interval


class Budget:
    def __init__(self, seconds: float) -> None:
        self.seconds = float(seconds)
        self.started = time.perf_counter()

    @property
    def elapsed(self) -> float:
        return time.perf_counter() - self.started

    def require(self, reserve_seconds: float = 0.0) -> None:
        if self.elapsed + float(reserve_seconds) >= self.seconds:
            raise ResourceStop("shared tuning GPU-time cap exhausted")


def run_stream(
    *,
    stream: Stream,
    params: TrialParameters,
    rungs: tuple[int, ...],
    output_dir: Path,
    budget: Budget,
    report: Callable[[int, dict[str, Any]], bool] | None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    target = locked_ssl_lstm_posterior_target()
    config = stream_config(target, stream, params)
    trainer = NeuTraReverseKLTrainer(target, config)
    validation_z = validation_batch(stream)
    initial = validation_row(trainer, validation_z, 0)
    history = [initial]
    checkpoints = []
    warmup = step_batch(stream, 1)
    hlo = trainer._compiled_train_step.experimental_get_compiler_ir(warmup)(  # noqa: SLF001
        stage="hlo"
    )
    if not isinstance(hlo, str) or "HloModule" not in hlo:
        raise TuningError("tuning stream did not expose XLA HLO")

    rung_set = set(rungs)
    for step in range(1, rungs[-1] + 1):
        budget.require(reserve_seconds=30.0)
        trainer.train_step(warmup if step == 1 else step_batch(stream, step))
        if step not in rung_set:
            continue
        row = validation_row(trainer, validation_z, step)
        history.append(row)
        state = trainer.state_payload()
        checkpoint_path = output_dir / f"checkpoint-{step:04d}.json"
        write_json(checkpoint_path, state)
        checkpoints.append(
            {
                "step": step,
                "path": checkpoint_path.relative_to(ROOT).as_posix(),
                "sha256": sha256(checkpoint_path),
                "state_hash": state["state_hash"],
            }
        )
        vetoes = _rung_vetoes(row)
        prune = bool(vetoes)
        if report is not None and not prune:
            prune = bool(report(rungs.index(step), row))
            if prune:
                vetoes.append("optuna_pruner")
        if prune:
            result = {
                "status": "PRUNED",
                "stream": asdict(stream),
                "config": config.manifest_payload(),
                "rungs": list(rungs),
                "history": history,
                "checkpoints": checkpoints,
                "vetoes": vetoes,
                "runtime_seconds": time.perf_counter() - started,
                "hlo_sha256": hashlib.sha256(hlo.encode()).hexdigest(),
            }
            write_json(output_dir / "result.json", result)
            return result

    frozen = trainer.frozen_transport_payload(
        transport_id=f"ssl-lstm-optuna-{output_dir.parent.name}-{stream.label}",
        target_signature=target.target_signature(),
    )
    frozen_path = output_dir / "frozen-payload.json"
    write_json(frozen_path, frozen)
    loaded = load_frozen_neutra_artifact(
        frozen, expected_target_signature=target.target_signature()
    )
    probes = HELPERS._probe_diagnostics(target, loaded.transport)  # noqa: SLF001
    vetoes, interval = _terminal_vetoes(
        initial=initial,
        final=history[-1],
        probes=probes,
    )
    result = {
        "status": "VETOED" if vetoes else "SURVIVED",
        "stream": asdict(stream),
        "config": config.manifest_payload(),
        "rungs": list(rungs),
        "history": history,
        "checkpoints": checkpoints,
        "vetoes": vetoes,
        "paired_final_minus_initial": interval,
        "terminal_probes": probes,
        "objective": float(history[-1]["mean_loss"]),
        "frozen_path": frozen_path.relative_to(ROOT).as_posix(),
        "frozen_sha256": sha256(frozen_path),
        "runtime_seconds": time.perf_counter() - started,
        "hlo_sha256": hashlib.sha256(hlo.encode()).hexdigest(),
        "device": sorted({variable.device for variable in trainer.variables}),
    }
    write_json(output_dir / "result.json", result)
    return result


def run_trial(
    *,
    trial: Any,
    params: TrialParameters,
    rungs: tuple[int, ...],
    output_root: Path,
    budget: Budget,
) -> dict[str, Any]:
    trial_dir = output_root / "trials" / f"trial-{trial.number:04d}"
    trial_dir.mkdir(parents=True, exist_ok=False)
    results = []
    a_history: list[dict[str, Any]] = []

    def report_a(index: int, row: dict[str, Any]) -> bool:
        a_history.append(row)
        trial.report(float(row["mean_loss"]), step=index + 1)
        return trial.should_prune()

    try:
        a = run_stream(
            stream=STREAMS[0],
            params=params,
            rungs=rungs,
            output_dir=trial_dir / STREAMS[0].label,
            budget=budget,
            report=report_a,
        )
    except ResourceStop as exc:
        write_resource_stop_receipt(trial_dir, STREAMS[0], params, budget, exc)
        raise
    results.append(a)
    if a["status"] != "SURVIVED":
        return _trial_result(trial.number, params, results, budget.elapsed)

    def report_b(index: int, row: dict[str, Any]) -> bool:
        common_worst = max(float(a_history[index]["mean_loss"]), float(row["mean_loss"]))
        trial.report(common_worst, step=len(rungs) + index + 1)
        return trial.should_prune()

    try:
        b = run_stream(
            stream=STREAMS[1],
            params=params,
            rungs=rungs,
            output_dir=trial_dir / STREAMS[1].label,
            budget=budget,
            report=report_b,
        )
    except ResourceStop as exc:
        write_resource_stop_receipt(trial_dir, STREAMS[1], params, budget, exc)
        raise
    results.append(b)
    return _trial_result(trial.number, params, results, budget.elapsed)


def write_resource_stop_receipt(
    trial_dir: Path,
    stream: Stream,
    params: TrialParameters,
    budget: Budget,
    error: ResourceStop,
) -> None:
    """Preserve interruption provenance without classifying a candidate veto."""

    write_json(
        trial_dir / "resource-stop.json",
        {
            "schema": SCHEMA,
            "status": "RESOURCE_STOP",
            "stream": asdict(stream),
            "params": asdict(params),
            "charged_gpu_seconds": budget.elapsed,
            "gpu_cap_seconds": budget.seconds,
            "error": str(error),
            "candidate_veto": False,
            "scientific_interpretation": "none",
        },
    )


def _trial_result(
    trial_number: int,
    params: TrialParameters,
    streams: list[dict[str, Any]],
    charged_seconds: float,
) -> dict[str, Any]:
    survived = len(streams) == 2 and all(row["status"] == "SURVIVED" for row in streams)
    vetoes = sorted(
        {
            f"{row['stream']['label']}:{reason}"
            for row in streams
            for reason in row.get("vetoes", [])
        }
    )
    objective = (
        max(float(row["objective"]) for row in streams) if survived else None
    )
    return {
        "schema": SCHEMA,
        "trial_number": int(trial_number),
        "params": asdict(params),
        "status": "SURVIVED" if survived else "PRUNED_OR_VETOED",
        "vetoes": vetoes,
        "objective": objective,
        "objective_kind": "worst_historical_stream_terminal_heldout_mean_rkl",
        "streams": streams,
        "charged_gpu_seconds_at_completion": float(charged_seconds),
    }


def configure_gpu() -> list[Any]:
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        raise TuningError("trusted tuning requires a visible GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.experimental.enable_tensor_float_32_execution(True)
    tf.config.set_soft_device_placement(False)
    return gpus


def run_timing_smoke(args: Any, bindings: dict[str, Any], gpus: list[Any]) -> dict[str, Any]:
    output_root = ROOT / args.output_root
    output_root.mkdir(parents=True, exist_ok=False)
    budget = Budget(args.gpu_cap_seconds)
    try:
        result = run_stream(
            stream=STREAMS[0],
            params=fixed_timing_parameters(),
            rungs=args.rungs,
            output_dir=output_root / "timing-smoke" / STREAMS[0].label,
            budget=budget,
            report=None,
        )
    except ResourceStop as exc:
        payload = {
            "schema": SCHEMA,
            "mode": "timing-smoke",
            "status": "RESOURCE_STOP",
            "error": str(exc),
            "candidate_veto": False,
            "scientific_interpretation": "none",
            "source_bindings": bindings,
            "run_manifest": run_manifest(args, gpus, budget.elapsed),
            "nonclaims": ["incomplete timing smoke only"],
        }
        write_json(output_root / "timing-smoke-resource-stop.json", payload)
        return payload
    payload = {
        "schema": SCHEMA,
        "mode": "timing-smoke",
        "status": "COMPLETED",
        "result": result,
        "source_bindings": bindings,
        "run_manifest": run_manifest(args, gpus, budget.elapsed),
        "nonclaims": [
            "timing and integration smoke only",
            "no hyperparameter nomination",
            "no training-quality, HMC, posterior, or scientific claim",
        ],
    }
    write_json(output_root / "timing-smoke-result.json", payload)
    return payload


def run_study(args: Any, bindings: dict[str, Any], gpus: list[Any]) -> dict[str, Any]:
    import optuna

    output_root = ROOT / args.output_root
    if output_root.exists() and not args.resume:
        raise TuningError("study output root exists; pass --resume only for the same study")
    output_root.mkdir(parents=True, exist_ok=True)
    storage_path = output_root / "study.sqlite3"
    storage = f"sqlite:///{storage_path}"
    sampler = optuna.samplers.TPESampler(
        seed=args.sampler_seed,
        n_startup_trials=min(2, args.n_trials),
        multivariate=False,
    )
    pruner = optuna.pruners.SuccessiveHalvingPruner(
        min_resource=1,
        reduction_factor=2,
        min_early_stopping_rate=0,
    )
    study = optuna.create_study(
        study_name=args.study_name,
        storage=storage,
        load_if_exists=args.resume,
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
    )
    budget = Budget(args.gpu_cap_seconds)
    records: list[dict[str, Any]] = []

    def objective(trial: Any) -> float:
        budget.require(reserve_seconds=30.0)
        params = trial_parameters(trial)
        try:
            record = run_trial(
                trial=trial,
                params=params,
                rungs=args.rungs,
                output_root=output_root,
                budget=budget,
            )
        except ResourceStop:
            trial.set_user_attr("resource_stop", True)
            trial.set_user_attr("candidate_veto", False)
            raise
        finally:
            tf.keras.backend.clear_session()
            gc.collect()
        records.append(record)
        trial_path = output_root / "trials" / f"trial-{trial.number:04d}" / "result.json"
        write_json(trial_path, record)
        trial.set_user_attr("record_path", trial_path.relative_to(ROOT).as_posix())
        trial.set_user_attr("record_sha256", sha256(trial_path))
        trial.set_user_attr("vetoes", record["vetoes"])
        if record["status"] != "SURVIVED":
            raise optuna.TrialPruned(";".join(record["vetoes"]) or "pruned")
        return float(record["objective"])

    resource_stop = None
    try:
        study.optimize(
            objective,
            n_trials=args.n_trials,
            timeout=args.timeout_seconds,
            gc_after_trial=True,
        )
    except ResourceStop as exc:
        resource_stop = str(exc)

    trials = []
    for trial in study.trials:
        trials.append(
            {
                "number": trial.number,
                "state": trial.state.name,
                "params": trial.params,
                "value": trial.value,
                "intermediate_values": {
                    str(key): value for key, value in trial.intermediate_values.items()
                },
                "user_attrs": trial.user_attrs,
            }
        )
    complete = [trial for trial in study.trials if trial.state.name == "COMPLETE"]
    best = None
    if complete:
        selected = min(complete, key=lambda row: float(row.value))
        best = {
            "trial_number": selected.number,
            "params": selected.params,
            "objective": selected.value,
            "record_path": selected.user_attrs.get("record_path"),
            "record_sha256": selected.user_attrs.get("record_sha256"),
        }
    payload = {
        "schema": SCHEMA,
        "mode": "study",
        "status": "RESOURCE_STOP" if resource_stop else "COMPLETED",
        "decision": "JOINT_SURVIVOR_AVAILABLE" if best else "NO_JOINT_SURVIVOR",
        "study_name": study.study_name,
        "storage_path": storage_path.relative_to(ROOT).as_posix(),
        "sampler": "TPESampler",
        "pruner": "SuccessiveHalvingPruner",
        "rungs": list(args.rungs),
        "stream_order": [stream.label for stream in STREAMS],
        "trials": trials,
        "best_nomination_proxy": best,
        "ranking_note": (
            "The scalar objective nominates a representative joint survivor only. "
            "It does not statistically rank viable trials or establish correctness."
        ),
        "resource_stop": resource_stop,
        "source_bindings": bindings,
        "run_manifest": run_manifest(args, gpus, budget.elapsed),
        "nonclaims": [
            "historical-stream hyperparameter nomination only",
            "no fresh-seed confirmation",
            "no HMC or posterior-correctness claim",
            "no superiority or default-readiness claim",
        ],
    }
    write_json(output_root / "study-summary.json", payload, replace=args.resume)
    return payload


def run_manifest(args: Any, gpus: list[Any], charged_seconds: float) -> dict[str, Any]:
    return {
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "command": " ".join(sys.argv),
        "environment": os.environ.get("CONDA_DEFAULT_ENV", "N/A"),
        "python": sys.version.split()[0],
        "tensorflow": tf.__version__,
        "optuna": _optuna_version(),
        "physical_gpus": [gpu.name for gpu in gpus],
        "dtype": "float64",
        "jit_compile": True,
        "tf32": bool(tf.config.experimental.tensor_float_32_execution_enabled()),
        "soft_device_placement": False,
        "trust_basis": "owner_designated_managed_session_visible_gpu_trusted",
        "sampler_seed": args.sampler_seed,
        "charged_gpu_seconds": float(charged_seconds),
        "gpu_cap_seconds": float(args.gpu_cap_seconds),
        "completed_at_utc": now(),
        "output_root": args.output_root.as_posix(),
        "plan": PLAN_PATH.as_posix(),
    }


def _optuna_version() -> str:
    try:
        import optuna
    except ModuleNotFoundError:
        return "unavailable"
    return str(optuna.__version__)


def parse_args(argv: list[str] | None = None) -> Any:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("timing-smoke", "study"), required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--rungs", type=parse_rungs, default=(50, 100, 200, 400))
    parser.add_argument("--n-trials", type=int, default=6)
    parser.add_argument("--gpu-cap-seconds", type=float, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=None)
    parser.add_argument("--sampler-seed", type=int, default=20260715)
    parser.add_argument("--study-name", default="ssl_lstm_neutra_32x32_tuning_v1")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.n_trials <= 0:
        parser.error("--n-trials must be positive")
    if not math.isfinite(args.gpu_cap_seconds) or args.gpu_cap_seconds <= 0.0:
        parser.error("--gpu-cap-seconds must be finite and positive")
    if args.mode == "timing-smoke" and args.resume:
        parser.error("timing smoke cannot resume")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    bindings = source_bindings()
    gpus = configure_gpu()
    payload = (
        run_timing_smoke(args, bindings, gpus)
        if args.mode == "timing-smoke"
        else run_study(args, bindings, gpus)
    )
    print(
        "JSON_SUMMARY "
        + json.dumps(
            {
                "mode": payload["mode"],
                "status": payload["status"],
                "decision": payload.get("decision"),
                "charged_gpu_seconds": payload["run_manifest"]["charged_gpu_seconds"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
