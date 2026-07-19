#!/usr/bin/env python3
"""Render bounded scalar SSL-LSTM NeuTra validation figures."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shlex
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from statistics import NormalDist
from typing import Any

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib-bayesfilter")

import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bayesfilter.inference.neutra_artifacts import load_frozen_neutra_artifact  # noqa: E402
from bayesfilter.nonlinear.ssl_lstm_posterior_tf import FREE_PARAMETER_NAMES  # noqa: E402
from bayesfilter.nonlinear.ssl_lstm_predictive_tf import (  # noqa: E402
    SSLLSTMForecastConfig,
    forecast_ssl_lstm_paths,
    make_ssl_lstm_innovation_bank,
)


PLAN_PATH = Path("docs/plans/bayesfilter-ssl-lstm-direct-visual-validation-plan-2026-07-18.md")
RESULT_PATH = Path("docs/plans/bayesfilter-ssl-lstm-direct-visual-validation-result-2026-07-18.md")
SCRIPT_PATH = Path(__file__).resolve().relative_to(ROOT)
PHASE7_ROOT = Path(
    "docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/phase-7-retained-admission"
)
PHASE7_RECEIPT = PHASE7_ROOT / "retained-acquisition.json"
PHASE7_SHA256 = "b79e5f6041e284de40bbd3834cc909fd12f45d012f172e570acccaa62dbe31a5"
PILOT_RECEIPT = Path(
    "docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/phase-8-predictive-design/"
    "target-pilot-repair-03.json"
)
TARGET_RECEIPT = Path(
    "docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/phase-8-predictive-design/"
    "target-integration/target-integration-preflight-repair-04.json"
)
CONTROLLED_AUDIT = Path(
    "docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/phase-8-predictive-design/"
    "directional-region-remedy/audit.json"
)
TRANSPORT_ROOT = Path(
    "docs/plans/artifacts/ssl-lstm-neutra-2026-07-14/"
    "trial0-alternative-confirmation-2026-07-16"
)
PAYLOADS = {
    "fresh-g": (
        TRANSPORT_ROOT / "fresh-g/best-frozen-payload.json",
        "6e147d5b33d003e0c895f294fc6b33523dcf97dc24af794d26a677886dedc354",
        "5e485163a01f7f2a02d511fd40fa8d16f8249d528940a453df6386e1d68505aa",
    ),
    "fresh-h": (
        TRANSPORT_ROOT / "fresh-h/best-frozen-payload.json",
        "ed0e42602aa39788ca1ea8d3c881d8bf85e15b91a687ef9adbe00a7b2c9120fb",
        "afa52cc59fba6e566649b085ae0367e3d91eb5a1cfd30fd9b7a5a15fcf4fd44a",
    ),
}
TARGET_SIGNATURE = "549efdf2aa5d9534226cb29c3678489d92766f92e6140901355eac33618f719e"
PILOT_DRAWS = 64
SEGMENT_DRAWS = 256
CHAIN_COUNT = 4
PARAMETER_DIM = 4
BLOCK_LENGTH = 16
FORECAST_CHUNK = 16
SEEDS = {"fresh-g": (20260718, 5101), "fresh-h": (20260718, 5201)}
ARM_IDS = {"fresh-g": 11, "fresh-h": 12}
COLORS = ("#0b6e4f", "#d95d39", "#2d5f9a", "#c28b18")
PLOT_FONT = ImageFont.load_default()


class VisualValidationError(RuntimeError):
    """Raised when the direct-visualization evidence contract fails."""


def _absolute(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def _sha256(path: Path) -> str:
    return hashlib.sha256(_absolute(path).read_bytes()).hexdigest()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(_absolute(path).read_text(encoding="utf-8"))


def _git(*args: str) -> str:
    return subprocess.run(
        ("git", *args), cwd=ROOT, check=True, capture_output=True, text=True
    ).stdout


def orient_pilot_prefix(tensor: tf.Tensor) -> tf.Tensor:
    """Validate persisted orientation and return chain-major pilot values."""

    values = tf.convert_to_tensor(tensor, dtype=tf.float64)
    if tuple(values.shape) != (SEGMENT_DRAWS, CHAIN_COUNT, PARAMETER_DIM):
        raise VisualValidationError("retained segment shape/orientation mismatch")
    prefix = tf.transpose(values[:PILOT_DRAWS], [1, 0, 2])
    if tuple(prefix.shape) != (CHAIN_COUNT, PILOT_DRAWS, PARAMETER_DIM):
        raise VisualValidationError("pilot prefix shape mismatch")
    if not bool(tf.reduce_all(tf.math.is_finite(prefix)).numpy()):
        raise VisualValidationError("pilot prefix contains non-finite values")
    return prefix


def _inside_phase7(path: Path) -> bool:
    try:
        _absolute(path).resolve().relative_to(_absolute(PHASE7_ROOT).resolve())
    except ValueError:
        return False
    return True


def load_pilot_prefixes() -> tuple[dict[str, tf.Tensor], dict[str, Any]]:
    if _sha256(PHASE7_RECEIPT) != PHASE7_SHA256:
        raise VisualValidationError("Phase-7 receipt hash drift")
    receipt = _json(PHASE7_RECEIPT)
    if receipt.get("decision") != "PHASE7_RETAINED_ADMISSION_PASSED_PHASE8_HANDOFF":
        raise VisualValidationError("Phase-7 admission is not authoritative")
    pilot_receipt = _json(PILOT_RECEIPT)
    split = pilot_receipt.get("split_contract", {})
    if not (
        split.get("pilot_permanently_excluded_from_phase9") is True
        and split.get("pilot_draw_count_per_chain") == PILOT_DRAWS
        and split.get("confirmation_forecast_bank_opened") is False
    ):
        raise VisualValidationError("pilot exclusion boundary is not authenticated")

    prefixes: dict[str, tf.Tensor] = {}
    audit: dict[str, Any] = {}
    for chart in PAYLOADS:
        segments = receipt["charts"][chart]["segments"]
        if len(segments) != 2:
            raise VisualValidationError(f"unexpected segment count for {chart}")
        chart_rows = []
        for index, public in enumerate(segments):
            label = f"{chart}-retained-segment-{index:03d}"
            if public.get("label") != label or public.get("passed") is not True:
                raise VisualValidationError(f"segment identity mismatch: {label}")
            manifest_path = PHASE7_ROOT / "retained-private" / chart / f"{label}_private_manifest.json"
            if _sha256(manifest_path) != public["archive_hashes"]["private_manifest_sha256"]:
                raise VisualValidationError(f"manifest hash mismatch: {label}")
            manifest = _json(manifest_path)
            shard = manifest["sample_shards"][0]
            sample_path = Path(shard["path"])
            if not _inside_phase7(sample_path):
                raise VisualValidationError("retained shard escapes Phase-7 root")
            sample_hash = _sha256(sample_path)
            if sample_hash != shard["sha256"] or sample_hash != public["archive_hashes"]["sample_sha256"]:
                raise VisualValidationError(f"sample hash mismatch: {label}")
            parsed = index == 0
            if parsed:
                raw = tf.io.parse_tensor(_absolute(sample_path).read_bytes(), out_type=tf.float64)
                prefixes[chart] = orient_pilot_prefix(raw)
            chart_rows.append(
                {
                    "segment": index,
                    "manifest_sha256": _sha256(manifest_path),
                    "sample_sha256": sample_hash,
                    "tensor_values_deserialized": parsed,
                    "selected_indices": [0, PILOT_DRAWS - 1] if parsed else None,
                }
            )
        audit[chart] = chart_rows
    return prefixes, audit


def map_to_theta(chart: str, z: tf.Tensor) -> tuple[tf.Tensor, dict[str, Any]]:
    payload_path, payload_hash, transport_hash = PAYLOADS[chart]
    if _sha256(payload_path) != payload_hash:
        raise VisualValidationError(f"transport payload hash drift: {chart}")
    artifact = load_frozen_neutra_artifact(
        _json(payload_path), expected_target_signature=TARGET_SIGNATURE
    )
    if artifact.manifest.transport_hash != transport_hash:
        raise VisualValidationError(f"transport identity drift: {chart}")

    @tf.function(jit_compile=True, reduce_retracing=True)
    def mapper(values: tf.Tensor) -> tf.Tensor:
        return artifact.transport.forward_z_to_theta_batch(values)

    flat = tf.reshape(z, [CHAIN_COUNT * PILOT_DRAWS, PARAMETER_DIM])
    theta = tf.reshape(mapper(flat), [CHAIN_COUNT, PILOT_DRAWS, PARAMETER_DIM])
    if not bool(tf.reduce_all(tf.math.is_finite(theta)).numpy()):
        raise VisualValidationError(f"mapped theta non-finite: {chart}")
    return theta, {
        "payload_sha256": payload_hash,
        "transport_hash": transport_hash,
        "output_device": str(theta.device),
        "mapped_theta_sha256": hashlib.sha256(bytes(tf.io.serialize_tensor(theta).numpy())).hexdigest(),
    }


def summarize_paths(paths: np.ndarray) -> dict[str, Any]:
    values = np.asarray(paths, dtype=np.float64)
    if values.shape != (CHAIN_COUNT, PILOT_DRAWS, 2, 10) or not np.isfinite(values).all():
        raise VisualValidationError("forecast path summary shape/non-finite failure")
    flat = values.reshape(-1, 10)
    return {
        "mean": flat.mean(axis=0),
        "variance": flat.var(axis=0, ddof=1),
        "q05": np.quantile(flat, 0.05, axis=0),
        "q50": np.quantile(flat, 0.50, axis=0),
        "q95": np.quantile(flat, 0.95, axis=0),
        "chain_mean": values.mean(axis=(1, 2)),
        "path_count": int(flat.shape[0]),
    }


def moment_difference_diagnostic(
    left: np.ndarray, right: np.ndarray
) -> dict[str, np.ndarray | float | int]:
    """Return estimates and approximate block-normal simultaneous bands."""

    arrays = []
    estimates = []
    for raw in (left, right):
        values = np.asarray(raw, dtype=np.float64)
        if values.shape != (CHAIN_COUNT, PILOT_DRAWS, 2, 10):
            raise VisualValidationError("moment path shape mismatch")
        flat = values.reshape(-1, 10)
        mean = flat.mean(axis=0)
        variance = flat.var(axis=0, ddof=1)
        if np.any(variance <= 0.0) or not np.isfinite(variance).all():
            raise VisualValidationError("moment variance is non-positive")
        draw_mean_influence = values.mean(axis=2) - mean
        draw_logvar_influence = (
            np.mean(np.square(values - mean[None, None, None, :]), axis=2) - variance
        ) / variance
        influence = np.concatenate((draw_mean_influence, draw_logvar_influence), axis=2)
        blocks = influence.reshape(CHAIN_COUNT, PILOT_DRAWS // BLOCK_LENGTH, BLOCK_LENGTH, 20).mean(axis=2)
        arrays.append(blocks.reshape(-1, 20))
        estimates.append(np.concatenate((mean, np.log(variance))))
    block_count = arrays[0].shape[0]
    se = np.sqrt(arrays[0].var(axis=0, ddof=1) / block_count + arrays[1].var(axis=0, ddof=1) / block_count)
    multiplier = NormalDist().inv_cdf(1.0 - 0.05 / (2.0 * 20.0))
    difference = estimates[0] - estimates[1]
    return {
        "difference": difference,
        "lower": difference - multiplier * se,
        "upper": difference + multiplier * se,
        "standard_error": se,
        "multiplier": multiplier,
        "block_count_per_arm": block_count,
    }


def _font() -> ImageFont.ImageFont:
    return PLOT_FONT


def _save_figure(image: Image.Image, output: Path) -> None:
    image.save(output.with_suffix(".png"), format="PNG")
    image.convert("RGB").save(output.with_suffix(".pdf"), format="PDF", resolution=120.0)


def _panel_box(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int], title: str) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = box
    draw.rectangle(box, outline="#555555", width=1)
    draw.text((x0 + 5, y0 + 4), title, fill="#111111", font=_font())
    return x0 + 34, y0 + 22, x1 - 8, y1 - 20


def _series_line(draw: ImageDraw.ImageDraw, series: np.ndarray, box: tuple[int, int, int, int], color: str, *, lo: float, hi: float) -> None:
    x0, y0, x1, y1 = box
    denom = max(hi - lo, 1.0e-12)
    points = []
    for index, value in enumerate(np.asarray(series, dtype=float)):
        x = x0 + (x1 - x0) * index / max(len(series) - 1, 1)
        y = y1 - (y1 - y0) * (float(value) - lo) / denom
        points.append((int(x), int(np.clip(y, y0, y1))))
    if len(points) > 1:
        draw.line(points, fill=color, width=2)


def _plot_traces(values: dict[str, np.ndarray], output: Path, *, space: str) -> None:
    image = Image.new("RGB", (1500, 760), "white")
    draw = ImageDraw.Draw(image)
    draw.text((20, 12), f"Four-launch retained trajectories ({space}-space; first 64 excluded pilot draws)", fill="#111111", font=_font())
    all_values = np.concatenate(tuple(np.asarray(values[key]) for key in ("fresh-g", "fresh-h")), axis=None)
    lo = float(np.min(all_values)); hi = float(np.max(all_values))
    for row, chart in enumerate(("fresh-g", "fresh-h")):
        array = np.asarray(values[chart])
        for coordinate in range(4):
            x = 20 + coordinate * 365
            y = 48 + row * 350
            box = _panel_box(draw, (x, y, x + 345, y + 320), f"{chart} {('z' + str(coordinate + 1)) if space == 'z' else FREE_PARAMETER_NAMES[coordinate]}")
            for chain in range(CHAIN_COUNT):
                _series_line(draw, array[chain, :, coordinate], box, COLORS[chain], lo=lo, hi=hi)
    draw.text((20, 730), "colors: launch 1 green, launch 2 orange, launch 3 blue, launch 4 gold", fill="#333333", font=_font())
    _save_figure(image, output)


def _plot_fans(summaries: dict[str, dict[str, Any]], output: Path) -> None:
    image = Image.new("RGB", (1250, 540), "white")
    draw = ImageDraw.Draw(image)
    draw.text((20, 12), "Posterior-predictive fan charts and launch-specific means", fill="#111111", font=_font())
    global_lo = min(float(np.min(summaries[c]["q05"])) for c in summaries)
    global_hi = max(float(np.max(summaries[c]["q95"])) for c in summaries)
    for panel, chart, color in ((0, "fresh-g", "#0b6e4f"), (1, "fresh-h", "#d95d39")):
        x = 20 + panel * 610
        box = _panel_box(draw, (x, 48, x + 580, 500), f"{chart} ({summaries[chart]['path_count']} paths)")
        row = summaries[chart]
        x0, y0, x1, y1 = box
        q05, q50, q95 = (np.asarray(row[key]) for key in ("q05", "q50", "q95"))
        for index in range(9):
            xa = x0 + (x1 - x0) * index / 9; xb = x0 + (x1 - x0) * (index + 1) / 9
            ya0 = y1 - (y1 - y0) * (q05[index] - global_lo) / max(global_hi-global_lo, 1e-12)
            yb0 = y1 - (y1 - y0) * (q05[index+1] - global_lo) / max(global_hi-global_lo, 1e-12)
            ya1 = y1 - (y1 - y0) * (q95[index] - global_lo) / max(global_hi-global_lo, 1e-12)
            yb1 = y1 - (y1 - y0) * (q95[index+1] - global_lo) / max(global_hi-global_lo, 1e-12)
            draw.polygon([(xa, ya0), (xb, yb0), (xb, yb1), (xa, ya1)], fill="#d9e5ed")
        _series_line(draw, q50, box, color, lo=global_lo, hi=global_hi)
        for chain in range(CHAIN_COUNT):
            _series_line(draw, np.asarray(row["chain_mean"])[chain], box, COLORS[chain], lo=global_lo, hi=global_hi)
    _save_figure(image, output)


def _plot_moments(row: dict[str, Any], output: Path) -> None:
    image = Image.new("RGB", (1250, 540), "white")
    draw = ImageDraw.Draw(image)
    draw.text((20, 12), "Explanatory 95% Bonferroni block-normal bands; no equivalence decision", fill="#111111", font=_font())
    difference = np.asarray(row["difference"]); lower = np.asarray(row["lower"]); upper = np.asarray(row["upper"])
    for panel, slc, title in ((0, slice(0, 10), "Predictive mean: G - H"), (1, slice(10, 20), "Log predictive variance: G - H")):
        x = 20 + panel * 610
        box = _panel_box(draw, (x, 48, x + 580, 500), title)
        lo = float(min(np.min(lower[slc]), np.min(difference[slc]))); hi = float(max(np.max(upper[slc]), np.max(difference[slc])))
        _series_line(draw, lower[slc], box, "#b7c9d6", lo=lo, hi=hi)
        _series_line(draw, upper[slc], box, "#b7c9d6", lo=lo, hi=hi)
        _series_line(draw, difference[slc], box, "#2d5f9a", lo=lo, hi=hi)
    _save_figure(image, output)


def _controlled_summary() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    audit = _json(CONTROLLED_AUDIT)
    if audit.get("decision") != "LOCKED_CONTROLLED_AUDIT_PASSED_TARGET_CONFIRMATION_STILL_CLOSED":
        raise VisualValidationError("controlled audit decision drift")
    candidate = audit["candidates"][0]
    rows = []
    for family in candidate["families"]:
        if family["family"]["role"] == "explanatory":
            continue
        rows.append(
            {
                "name": family["family"]["name"],
                "role": family["family"]["role"],
                "coverage": family["coverage"]["estimate"],
                "required_decision": family["required_decision"]["estimate"],
                "false_or_boundary": family["false_or_boundary_decision"]["estimate"],
                "invalid": family["invalid_procedure"]["estimate"],
                "gate_passed": family["gate"]["passed"],
            }
        )
    return rows, {"decision": audit["decision"], "sha256": _sha256(CONTROLLED_AUDIT)}


def _plot_controlled(rows: list[dict[str, Any]], output: Path) -> None:
    image = Image.new("RGB", (1500, 720), "white")
    draw = ImageDraw.Draw(image)
    draw.text((20, 12), "Locked synthetic-law calibration (1,536 replications per family)", fill="#111111", font=_font())
    x0, y0, width, height = 45, 60, 1400, 280
    x1 = x0 + width
    for index, row in enumerate(rows):
        x = x0 + (index + 0.5) * width / len(rows)
        bar = width / len(rows) * 0.3
        for offset, value, color in ((-bar/2, row["coverage"], "#2d5f9a"), (bar/2, row["required_decision"], "#0b6e4f")):
            draw.rectangle((x + offset - bar/2, y0 + height * (1 - value), x + offset + bar/2, y0 + height), fill=color)
        draw.text((x - 24, y0 + height + 8), str(index + 1), fill="#222222", font=_font())
    draw.text((x0, y0 - 15), "coverage / required decision (thresholds 0.90 / 0.80)", fill="#222222", font=_font())
    x0b, y0b, heightb = 45, 410, 220
    for index, row in enumerate(rows):
        x = x0b + (index + 0.5) * width / len(rows)
        bar = width / len(rows) * 0.3
        for offset, value, color in ((-bar/2, row["false_or_boundary"], "#d95d39"), (bar/2, row["invalid"], "#c28b18")):
            draw.rectangle((x + offset - bar/2, y0b + heightb * (1 - value / 0.06), x + offset + bar/2, y0b + heightb), fill=color)
        draw.text((x - 24, y0b + heightb + 8), str(index + 1), fill="#222222", font=_font())
    draw.text((x0b, y0b - 15), "false/boundary and invalid rates (0.05 threshold)", fill="#222222", font=_font())
    draw.text((20, 690), "family numbers follow the locked audit receipt; explanatory view only", fill="#333333", font=_font())
    _save_figure(image, output)


def _plain(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def run(output_dir: Path, *, wall_cap_seconds: float) -> dict[str, Any]:
    started_at = datetime.now(UTC).isoformat()
    started = time.perf_counter()
    prefixes, archive_audit = load_pilot_prefixes()
    theta: dict[str, tf.Tensor] = {}
    mappings: dict[str, Any] = {}
    paths: dict[str, np.ndarray] = {}
    forecast_provenance: dict[str, Any] = {}
    bank_hashes: list[str] = []
    config = SSLLSTMForecastConfig()
    config.assert_evidence_config()
    for chart in PAYLOADS:
        theta[chart], mappings[chart] = map_to_theta(chart, prefixes[chart])
        flat = tf.reshape(theta[chart], [CHAIN_COUNT * PILOT_DRAWS, PARAMETER_DIM])
        bank = make_ssl_lstm_innovation_bank(
            config, int(flat.shape[0]), tf.constant(SEEDS[chart], tf.int32), "independent_arm", ARM_IDS[chart]
        )
        bank_hashes.extend(bank.tensor_hashes().values())
        forecast = forecast_ssl_lstm_paths(
            flat,
            bank,
            config,
            draw_chunk_size=FORECAST_CHUNK,
            runtime_execution_role="trusted_gpu_xla_canary",
            trust_basis="owner_designated_managed_session_visible_gpu_trusted",
        )
        if any(status != 0 for status in forecast.provenance.terminal_covariance_statuses):
            raise VisualValidationError(f"terminal covariance failure: {chart}")
        paths[chart] = np.asarray(
            tf.reshape(tf.squeeze(forecast.observations, -1), [CHAIN_COUNT, PILOT_DRAWS, 2, 10]).numpy()
        )
        forecast_provenance[chart] = {
            "innovation_root_seed": list(SEEDS[chart]),
            "innovation_bank_signature": bank.content_signature,
            "innovation_tensor_hashes": bank.tensor_hashes(),
            "output_devices": list(forecast.provenance.output_devices),
            "terminal_covariance_status_count": len(forecast.provenance.terminal_covariance_statuses),
            "terminal_covariance_failure_count": sum(status != 0 for status in forecast.provenance.terminal_covariance_statuses),
            "path_sha256": hashlib.sha256(paths[chart].tobytes(order="C")).hexdigest(),
        }
        if time.perf_counter() - started > wall_cap_seconds:
            raise VisualValidationError("visual-validation wall cap exceeded")
    if len(bank_hashes) != len(set(bank_hashes)):
        raise VisualValidationError("innovation tensors overlap across arms")

    summaries = {chart: summarize_paths(paths[chart]) for chart in PAYLOADS}
    moment = moment_difference_diagnostic(paths["fresh-g"], paths["fresh-h"])
    controlled_rows, controlled_binding = _controlled_summary()
    if not all(row["gate_passed"] for row in controlled_rows):
        raise VisualValidationError("locked controlled audit family gate drift")

    output_dir = _absolute(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    bases = {
        "z_traces": output_dir / "ssl-lstm-launch-traces-z",
        "theta_traces": output_dir / "ssl-lstm-launch-traces-theta",
        "predictive_fans": output_dir / "ssl-lstm-predictive-fans",
        "moment_differences": output_dir / "ssl-lstm-moment-differences",
        "controlled_calibration": output_dir / "ssl-lstm-controlled-calibration",
    }
    _plot_traces({key: np.asarray(value.numpy()) for key, value in prefixes.items()}, bases["z_traces"], space="z")
    _plot_traces({key: np.asarray(value.numpy()) for key, value in theta.items()}, bases["theta_traces"], space="theta")
    _plot_fans(summaries, bases["predictive_fans"])
    _plot_moments(moment, bases["moment_differences"])
    _plot_controlled(controlled_rows, bases["controlled_calibration"])
    figures = {}
    for name, base in bases.items():
        figures[name] = {
            suffix[1:]: {
                "path": base.with_suffix(suffix).relative_to(ROOT).as_posix(),
                "sha256": _sha256(base.with_suffix(suffix)),
                "bytes": base.with_suffix(suffix).stat().st_size,
            }
            for suffix in (".png", ".pdf")
        }

    wall = time.perf_counter() - started
    payload = {
        "schema": "bayesfilter.ssl_lstm.direct_visual_validation.v1",
        "status": "PASSED",
        "decision": "PASSED_VISUAL_PACKAGE_CONFIRMATION_CLOSED",
        "pilot_boundary": {
            "draws_per_chain": PILOT_DRAWS,
            "selected_indices": [0, PILOT_DRAWS - 1],
            "segment0_deserialized": True,
            "segment1_deserialized": False,
            "confirmation_suffix_inspected": False,
        },
        "archive_audit": archive_audit,
        "mapping": mappings,
        "forecast_provenance": forecast_provenance,
        "predictive_summaries": summaries,
        "moment_diagnostic": moment,
        "controlled_calibration": {"binding": controlled_binding, "families": controlled_rows},
        "figures": figures,
        "source_bindings": {
            "plan": {"path": PLAN_PATH.as_posix(), "sha256": _sha256(PLAN_PATH)},
            "runner": {"path": SCRIPT_PATH.as_posix(), "sha256": _sha256(SCRIPT_PATH)},
            "phase7_receipt": {"path": PHASE7_RECEIPT.as_posix(), "sha256": _sha256(PHASE7_RECEIPT)},
            "pilot_receipt": {"path": PILOT_RECEIPT.as_posix(), "sha256": _sha256(PILOT_RECEIPT)},
            "target_adapter_receipt": {"path": TARGET_RECEIPT.as_posix(), "sha256": _sha256(TARGET_RECEIPT)},
        },
        "run_manifest": {
            "command": " ".join(shlex.quote(item) for item in (sys.executable, *sys.argv)),
            "cwd": str(ROOT),
            "git_commit": _git("rev-parse", "HEAD").strip(),
            "git_dirty": bool(_git("status", "--porcelain").strip()),
            "python": platform.python_version(),
            "tensorflow": tf.__version__,
            "tensorflow_probability": __import__("tensorflow_probability").__version__,
            "renderer": "Pillow raster/vector PDF renderer",
            "conda_env": os.environ.get("CONDA_DEFAULT_ENV"),
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "physical_gpus": [device.name for device in tf.config.list_physical_devices("GPU")],
            "logical_gpus": [device.name for device in tf.config.list_logical_devices("GPU")],
            "jit_compile": True,
            "tf32_enabled": bool(tf.config.experimental.tensor_float_32_execution_enabled()),
            "trust_basis": "owner_designated_managed_session_visible_gpu_trusted",
            "random_seeds": {key: list(value) for key, value in SEEDS.items()},
            "started_at_utc": started_at,
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "wall_time_seconds": wall,
            "wall_cap_seconds": wall_cap_seconds,
        },
        "inference_status": {
            "hard_veto_screen": "passed for artifact integrity and forecast generation",
            "statistically_supported_ranking": "none",
            "descriptive_only_differences": "all retained-arm visual and moment differences",
            "default_readiness": "not established",
            "next_evidence_needed": "separately authorized retained-chain formal confirmation and independent synthetic-data calibration for model adequacy",
        },
        "nonclaims": [
            "plots do not prove posterior correctness or equivalence",
            "G and H are peer replications and neither is an oracle",
            "approximate moment bands are explanatory and emit no decision",
            "controlled synthetic calibration is not retained-chain posterior evidence",
            "confirmation suffix remains unread and confirmation remains closed",
        ],
    }
    result_path = output_dir / "visual-validation-result.json"
    result_path.write_text(json.dumps(_plain(payload), sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    return payload


def _require_gpu() -> None:
    physical = tf.config.list_physical_devices("GPU")
    if not physical:
        raise VisualValidationError("trusted GPU is required")
    for gpu in physical:
        tf.config.experimental.set_memory_growth(gpu, True)
    tf.config.experimental.enable_tensor_float_32_execution(True)
    if not tf.config.list_logical_devices("GPU"):
        raise VisualValidationError("logical GPU is unavailable")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--wall-cap-seconds", type=float, default=900.0)
    args = parser.parse_args(argv)
    if not math.isfinite(args.wall_cap_seconds) or args.wall_cap_seconds <= 0:
        raise VisualValidationError("wall cap must be positive and finite")
    _require_gpu()
    with tf.device("/GPU:0"):
        payload = run(args.output_dir, wall_cap_seconds=float(args.wall_cap_seconds))
    print(json.dumps({"decision": payload["decision"], "wall_time_seconds": payload["run_manifest"]["wall_time_seconds"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
