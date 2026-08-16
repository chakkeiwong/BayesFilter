#!/usr/bin/env python3
"""Whole-path energy diagnostics for an authorized q=20 posterior mixture."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "8")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


PLAN = Path(
    "docs/plans/"
    "bayesfilter-posterior-predictive-diagnostic-and-multimodal-hmc-survey-plan-2026-08-09.md"
)
RUNNER = Path(
    "docs/benchmarks/"
    "run_ssl_lstm_q20_posterior_predictive_energy_diagnostic_2026_08_09.py"
)
DEFAULT_OUTPUT_ROOT = Path(
    "docs/plans/artifacts/"
    "ssl-lstm-q20-posterior-predictive-energy-diagnostic-2026-08-09/r1"
)
POSTERIOR_SCHEMA = "bayesfilter.posterior_draw_archive.v1"
EXPECTED_STATUS = "POSTERIOR_AUTHORITY_PASSED"
EXPECTED_WEIGHT_SEMANTICS = "equal_weight_empirical_posterior"
ELIGIBLE_MODE_WEIGHT_STATUS = {"RESOLVED", "NOT_APPLICABLE_UNIMODAL"}
PARAMETER_NAMES = (
    "latent_mean_weight.0.0",
    "latent_mean_bias.0",
    "observation_weight.0.0",
    "observation_bias.0",
)
TARGET_SIGNATURE = "9a86e60081f1b9cd288dbdb1dcbe1e9a5b5e23d9b5ef97afdb72ee95c23d7278"
HORIZONS = (10, 20, 30, 50, 100)
PATH_COUNT = 1000
PERMUTATION_COUNT = 9999
PERMUTATION_BATCH_SIZE = 250
ALPHA = 0.01
Q = 20
PARAMETER_DIM = 4
THREADS = 8
CAMPAIGN_CAP_SECONDS = 1200.0
SEED_WORD = 20260809


class SSLLSTMPosteriorPredictiveError(RuntimeError):
    """Raised when the q=20 posterior-predictive contract fails."""


def _abs(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _sha(path: Path) -> str:
    return hashlib.sha256(_abs(path).read_bytes()).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(_abs(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SSLLSTMPosteriorPredictiveError(f"expected JSON object: {path}")
    return payload


def _safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_safe(item) for item in value]
    if isinstance(value, bytes):
        return value.decode("ascii")
    if hasattr(value, "as_list"):
        return _safe(value.as_list())
    if hasattr(value, "numpy"):
        return _safe(value.numpy())
    if hasattr(value, "tolist"):
        return _safe(value.tolist())
    if hasattr(value, "item"):
        return _safe(value.item())
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    absolute = _abs(path)
    absolute.parent.mkdir(parents=True, exist_ok=True)
    if absolute.exists():
        raise SSLLSTMPosteriorPredictiveError(
            f"refusing to overwrite artifact: {path}"
        )
    absolute.write_text(
        json.dumps(_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="ascii",
    )


def _write_tensor(path: Path, tensor: Any, tf: Any) -> dict[str, Any]:
    absolute = _abs(path)
    absolute.parent.mkdir(parents=True, exist_ok=True)
    if absolute.exists():
        raise SSLLSTMPosteriorPredictiveError(
            f"refusing to overwrite artifact: {path}"
        )
    serialized = bytes(tf.io.serialize_tensor(tensor).numpy())
    absolute.write_bytes(serialized)
    return {
        "path": path.as_posix(),
        "sha256": hashlib.sha256(serialized).hexdigest(),
        "bytes": len(serialized),
        "dtype": tensor.dtype.name,
        "shape": tensor.shape,
    }


def classify_p_value(value: float) -> str:
    if not math.isfinite(value) or not 0.0 < value <= 1.0:
        raise SSLLSTMPosteriorPredictiveError("p-value must be finite and in (0,1]")
    return (
        "DISTINGUISHED_AT_1_PERCENT"
        if value < ALPHA
        else "NOT_DISTINGUISHED_AT_1_PERCENT"
    )


def _require_receipt(receipt: Any, label: str) -> tuple[Path, Mapping[str, Any]]:
    if not isinstance(receipt, Mapping):
        raise SSLLSTMPosteriorPredictiveError(f"{label} receipt is missing")
    path = Path(str(receipt.get("path", "")))
    if not _abs(path).is_file():
        raise SSLLSTMPosteriorPredictiveError(f"{label} file is missing")
    if _sha(path) != receipt.get("sha256"):
        raise SSLLSTMPosteriorPredictiveError(f"{label} SHA-256 mismatch")
    return path, receipt


def load_authorized_posterior_draws(
    artifact_path: Path, tf: Any
) -> tuple[Any, dict[str, Any]]:
    """Load equal-weight physical draws from a passed posterior authority."""

    artifact = _read_json(artifact_path)
    if artifact.get("schema") != POSTERIOR_SCHEMA:
        raise SSLLSTMPosteriorPredictiveError("posterior artifact schema mismatch")
    if artifact.get("status") != EXPECTED_STATUS:
        raise SSLLSTMPosteriorPredictiveError("posterior authority is not passed")
    if artifact.get("target_signature") != TARGET_SIGNATURE:
        raise SSLLSTMPosteriorPredictiveError("posterior target signature mismatch")
    if tuple(artifact.get("parameter_names", ())) != PARAMETER_NAMES:
        raise SSLLSTMPosteriorPredictiveError("posterior parameter names mismatch")
    if artifact.get("warmup_excluded") is not True:
        raise SSLLSTMPosteriorPredictiveError("posterior warm-up is not excluded")
    if artifact.get("draw_weight_semantics") != EXPECTED_WEIGHT_SEMANTICS:
        raise SSLLSTMPosteriorPredictiveError(
            "posterior draws are not an equal-weight empirical posterior"
        )
    mode_status = artifact.get("relative_mode_weights_status")
    if mode_status not in ELIGIBLE_MODE_WEIGHT_STATUS:
        raise SSLLSTMPosteriorPredictiveError(
            "posterior relative mode weights are unresolved"
        )
    diagnostics = artifact.get("sampler_diagnostics")
    if not isinstance(diagnostics, Mapping) or diagnostics.get("passed") is not True:
        raise SSLLSTMPosteriorPredictiveError("sampler diagnostics are not passed")
    diagnostic_path, diagnostic_receipt = _require_receipt(
        diagnostics.get("result"), "sampler diagnostic result"
    )
    draw_path, draw_receipt = _require_receipt(
        artifact.get("physical_draws"), "physical posterior draws"
    )
    if draw_receipt.get("dtype") != "float64":
        raise SSLLSTMPosteriorPredictiveError("posterior draw dtype must be float64")
    draws = tf.io.parse_tensor(_abs(draw_path).read_bytes(), out_type=tf.float64)
    if (
        draws.shape.rank != 2
        or not draws.shape.is_fully_defined()
        or int(draws.shape[0]) < 2
        or tuple(draws.shape[1:]) != (PARAMETER_DIM,)
        or list(draws.shape) != list(draw_receipt.get("shape", ()))
    ):
        raise SSLLSTMPosteriorPredictiveError(
            "posterior draw tensor must match static shape [draw,4]"
        )
    if not bool(tf.reduce_all(tf.math.is_finite(draws))):
        raise SSLLSTMPosteriorPredictiveError("posterior draws are nonfinite")
    return draws, {
        "artifact_path": artifact_path.as_posix(),
        "artifact_sha256": _sha(artifact_path),
        "schema": POSTERIOR_SCHEMA,
        "status": EXPECTED_STATUS,
        "target_signature": TARGET_SIGNATURE,
        "parameter_names": PARAMETER_NAMES,
        "warmup_excluded": True,
        "draw_weight_semantics": EXPECTED_WEIGHT_SEMANTICS,
        "relative_mode_weights_status": mode_status,
        "mode_weight_authority": artifact.get("mode_weight_authority"),
        "draw_count": int(draws.shape[0]),
        "physical_draws": dict(draw_receipt),
        "sampler_diagnostics": {
            "result": dict(diagnostic_receipt),
            "resolved_path": diagnostic_path.as_posix(),
        },
    }


def ssl_lstm_batch_conditional_simulator(
    *, horizon: int, forecast: Callable[..., Any]
) -> Callable[[Any, Any], Any]:
    """Return the one-parameter-row/one-path q=20 simulator adapter."""

    if horizon not in HORIZONS:
        raise SSLLSTMPosteriorPredictiveError("horizon is outside the frozen grid")

    def simulator(parameter_rows: Any, seed: Any) -> Any:
        import tensorflow as tf

        parameters = tf.convert_to_tensor(parameter_rows, tf.float64)
        if (
            parameters.shape.rank != 2
            or not parameters.shape.is_fully_defined()
            or tuple(parameters.shape[1:]) != (PARAMETER_DIM,)
        ):
            raise SSLLSTMPosteriorPredictiveError(
                "simulator parameters require static shape [path,4]"
            )
        count = int(parameters.shape[0])
        result = forecast(
            parameters,
            q=Q,
            seed=tf.convert_to_tensor(seed, tf.int32),
            replication_count=1,
            horizon=horizon,
        )
        observations = tf.convert_to_tensor(result.observations, tf.float64)
        if tuple(observations.shape) != (count, 1, horizon):
            raise SSLLSTMPosteriorPredictiveError(
                "forecast did not preserve one path per posterior row"
            )
        if not bool(tf.reduce_all(result.status)) or not bool(
            tf.reduce_all(tf.math.is_finite(observations))
        ):
            raise SSLLSTMPosteriorPredictiveError("forecast validity failed")
        return tf.reshape(observations, [count, horizon])

    return simulator


def _seeds(horizon: int) -> dict[str, tuple[int, int]]:
    if horizon not in HORIZONS:
        raise SSLLSTMPosteriorPredictiveError("invalid horizon seed request")
    base = 700000 + 10 * horizon
    return {
        "posterior": (SEED_WORD, base + 1),
        "posterior_simulator": (SEED_WORD, base + 2),
        "truth_simulator": (SEED_WORD, base + 3),
        "permutation": (SEED_WORD, base + 4),
    }


def run(
    posterior_artifact: Path,
    output_root: Path,
    *,
    cap_seconds: float = CAMPAIGN_CAP_SECONDS,
) -> dict[str, Any]:
    started = time.perf_counter()
    import tensorflow as tf
    from bayesfilter.nonlinear.ssl_lstm_complexity_predictive_tf import (
        forecast_complexity_conditional_moments,
    )
    from bayesfilter.nonlinear.ssl_lstm_complexity_target_tf import PRIOR_CENTER
    from bayesfilter.testing.posterior_predictive_tf import (
        posterior_predictive_energy_test,
    )

    tf.config.threading.set_intra_op_parallelism_threads(THREADS)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.set_visible_devices([], "GPU")
    if tf.config.list_physical_devices("GPU"):
        raise SSLLSTMPosteriorPredictiveError("CPU diagnostic found a visible GPU")
    draws, posterior_receipt = load_authorized_posterior_draws(
        posterior_artifact, tf
    )
    rows = []
    output_root = Path(output_root)
    for horizon in HORIZONS:
        seeds = _seeds(horizon)
        result = posterior_predictive_energy_test(
            draws,
            tf.convert_to_tensor(PRIOR_CENTER, tf.float64),
            path_count=PATH_COUNT,
            posterior_seed=seeds["posterior"],
            posterior_simulator_seed=seeds["posterior_simulator"],
            truth_simulator_seed=seeds["truth_simulator"],
            permutation_seed=seeds["permutation"],
            conditional_simulator=ssl_lstm_batch_conditional_simulator(
                horizon=horizon, forecast=forecast_complexity_conditional_moments
            ),
            permutation_count=PERMUTATION_COUNT,
            permutation_batch_size=PERMUTATION_BATCH_SIZE,
            jit_compile=True,
        )
        label = f"t{horizon:03d}"
        index_receipt = _write_tensor(
            output_root / f"{label}-posterior-indices.tftensor",
            result.posterior_predictive.posterior_indices,
            tf,
        )
        permutation_receipt = _write_tensor(
            output_root / f"{label}-permutation-statistics.tftensor",
            result.energy.permutation_statistics,
            tf,
        )
        p_value = float(result.energy.p_value)
        row = {
            "schema": "bayesfilter.ssl_lstm.q20_posterior_predictive_energy.v1",
            "status": classify_p_value(p_value),
            "horizon": horizon,
            "path_count_per_arm": PATH_COUNT,
            "posterior_draw_count": int(draws.shape[0]),
            "posterior_rows_sampled_with_replacement": True,
            "one_posterior_row_per_path": True,
            "replication_count_per_selected_parameter": 1,
            "posterior_indices": index_receipt,
            "selected_unique_posterior_rows": tf.size(
                tf.unique(result.posterior_predictive.posterior_indices).y
            ),
            "energy_statistic": result.energy.statistic,
            "p_value": result.energy.p_value,
            "exceedance_count": result.energy.exceedance_count,
            "alpha": ALPHA,
            "permutation_count": PERMUTATION_COUNT,
            "permutation_statistics": permutation_receipt,
            "seeds": seeds,
            "jit_compile": True,
            "posterior_receipt": posterior_receipt,
            "nonclaims": (
                "not proof of equal distributions after non-rejection",
                "not an independent posterior authority",
                "not proof of HMC correctness",
                "not a joint or multiplicity-adjusted decision",
            ),
        }
        row_path = output_root / f"{label}.json"
        _write_json(row_path, row)
        rows.append(
            {
                "horizon": horizon,
                "status": row["status"],
                "energy_statistic": row["energy_statistic"],
                "p_value": row["p_value"],
                "exceedance_count": row["exceedance_count"],
                "selected_unique_posterior_rows": row[
                    "selected_unique_posterior_rows"
                ],
                "receipt": row_path.as_posix(),
                "receipt_sha256": _sha(row_path),
            }
        )
        if time.perf_counter() - started > cap_seconds:
            raise SSLLSTMPosteriorPredictiveError("campaign wall cap exceeded")
    payload = {
        "schema": "bayesfilter.ssl_lstm.q20_posterior_predictive_energy_campaign.v1",
        "status": "FIVE_POSTERIOR_PREDICTIVE_DIAGNOSTICS_COMPLETED",
        "rows": rows,
        "posterior_receipt": posterior_receipt,
        "horizons": HORIZONS,
        "path_count_per_arm": PATH_COUNT,
        "permutation_count_per_horizon": PERMUTATION_COUNT,
        "alpha_per_horizon": ALPHA,
        "joint_test_computed": False,
        "combined_p_value_computed": False,
        "multiplicity_adjustment_applied": False,
        "posterior_mean_used": False,
        "posterior_median_used": False,
        "posterior_map_used": False,
        "run_manifest": {
            "git_commit": subprocess.run(
                ("git", "rev-parse", "HEAD"), cwd=ROOT, check=True,
                capture_output=True, text=True
            ).stdout.strip(),
            "git_dirty": bool(
                subprocess.run(
                    ("git", "status", "--porcelain"), cwd=ROOT, check=True,
                    capture_output=True, text=True
                ).stdout.strip()
            ),
            "command": " ".join(sys.argv),
            "environment": os.environ.get("CONDA_DEFAULT_ENV", "unknown"),
            "python": platform.python_version(),
            "tensorflow": tf.__version__,
            "cpu_gpu_status": "CPU_ONLY_GPU_HIDDEN",
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cpu_threads": THREADS,
            "jit_compile": True,
            "wall_time_seconds": time.perf_counter() - started,
            "cap_seconds": cap_seconds,
            "plan": PLAN.as_posix(),
            "plan_sha256": _sha(PLAN),
            "runner": RUNNER.as_posix(),
            "runner_sha256": _sha(RUNNER),
            "output_root": output_root.as_posix(),
        },
        "nonclaims": (
            "non-rejection is not equivalence",
            "predictive agreement does not prove posterior or sampler correctness",
            "no result is valid unless the upstream posterior authority is valid",
            "no joint or familywise decision",
        ),
    }
    _write_json(output_root / "summary.json", payload)
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--posterior-artifact", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--cap-seconds", type=float, default=CAMPAIGN_CAP_SECONDS)
    args = parser.parse_args(argv)
    payload = run(
        args.posterior_artifact,
        args.output_root,
        cap_seconds=args.cap_seconds,
    )
    print(
        json.dumps(
            _safe(
                {
                    "status": payload["status"],
                    "rows": payload["rows"],
                    "wall_time_seconds": payload["run_manifest"][
                        "wall_time_seconds"
                    ],
                }
            ),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
