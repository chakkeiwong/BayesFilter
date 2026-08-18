#!/usr/bin/env python3
"""Austria-SIR SQMC particle-count and GenUT trust-region comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import tensorflow as tf

from bayesfilter.runtime.gpu_memory_policy import (
    configure_tensorflow_gpu_memory_growth,
)

MEMORY_POLICY = configure_tensorflow_gpu_memory_growth(tf, require_gpu=True)

from bayesfilter.highdim.genut_shape_lm_tf import GENUT_SHAPE_SOLVER_ID
from bayesfilter.highdim.transport_chunk_policy import select_transport_chunks
from bayesfilter.highdim.ledh_pfpf_genut_initial_rqmc_tf import (
    finite_value_standard_score_initial_rqmc,
)
from docs.benchmarks.run_ledh_pfpf_genut_full_sqmc_full_horizons import (
    build_full_horizon_models,
    campaign_inputs,
)
from docs.benchmarks.run_ledh_pfpf_genut_initial_rqmc_all_models import _design


PLAN = Path(
    "docs/plans/bayesfilter-genut-sqmc-streaming-n16128-plan-2026-08-18.md"
)
ARTIFACT_ROOT = Path(
    "docs/benchmarks/artifacts/genut-sqmc-streaming-n16128-20260818"
)
SCHEMA = "bayesfilter.genut_sqmc_particle_trust_austria.v1"
PARTICLE_COUNTS = (1008, 2016, 4032, 16128)
CLAIM_SEEDS = tuple(range(97701, 97717))
SMOKE_SEEDS = (97701,)
ROUTES = (
    "iid_dual_cap",
    "previous_inverse_cdf",
    "repaired_fixed_previous_controls",
    "repaired_permutation",
)
RESET_VARIANTS = ("legacy", "trust_region")
DEFAULT_RESET_VARIANTS = ("trust_region",)
TV_GATE = 1.0e-4
EQUAL_WEIGHT_TOLERANCE = 2.0e-6
TRUST_CONTROLS = {
    "lm_damping": 1.0e-2,
    "lm_scale_floor": 1.0e-4,
    "radius": 0.5,
}
ROUTE_CONTROLS = {
    "iid_dual_cap": {
        "candidate_id": "previous_exact",
        "epsilon": 8.0,
        "sinkhorn_steps": 8,
        "balance_steps": 8,
        "ridge": 1.0e-5,
        "map_multiplier": 3.0,
        "hilbert_bits": 12,
        "diagonal_steps": 4,
        "diagonal_strength": 0.2,
        "pairwise_steps": 4,
        "pairwise_strength": 0.02,
        "radial_cap": 2.0,
        "coordinate_cap": 0.98,
        "coordinate_cap_power": 8,
    },
    "previous_inverse_cdf": {
        "candidate_id": "previous_exact",
        "epsilon": 8.0,
        "sinkhorn_steps": 8,
        "balance_steps": 8,
        "ridge": 1.0e-5,
        "map_multiplier": 3.0,
        "hilbert_bits": 12,
        "diagonal_steps": 4,
        "diagonal_strength": 0.2,
        "pairwise_steps": 4,
        "pairwise_strength": 0.02,
        "radial_cap": 2.0,
        "coordinate_cap": 0.98,
        "coordinate_cap_power": 8,
    },
    "repaired_fixed_previous_controls": {
        "candidate_id": "previous_exact",
        "epsilon": 8.0,
        "sinkhorn_steps": 8,
        "balance_steps": 8,
        "ridge": 1.0e-5,
        "map_multiplier": 3.0,
        "hilbert_bits": 12,
        "diagonal_steps": 4,
        "diagonal_strength": 0.2,
        "pairwise_steps": 4,
        "pairwise_strength": 0.02,
        "radial_cap": 2.0,
        "coordinate_cap": 0.98,
        "coordinate_cap_power": 8,
    },
    "repaired_permutation": {
        "candidate_id": "conservative",
        "epsilon": 8.0,
        "sinkhorn_steps": 8,
        "balance_steps": 8,
        "ridge": 1.0e-5,
        "map_multiplier": 2.0,
        "hilbert_bits": 12,
        "diagonal_steps": 3,
        "diagonal_strength": 0.15,
        "pairwise_steps": 3,
        "pairwise_strength": 0.01,
        "radial_cap": 1.5,
        "coordinate_cap": 0.97,
        "coordinate_cap_power": 6,
    },
}
MAP_LOCATION = (
    154.7870330810547, 63.45612716674805, 153.8306427001953,
    63.28734588623047, 160.0264434814453, 63.21662521362305,
    158.84326171875, 63.38702392578125, 163.36050415039062,
    63.44179153442383, 167.7389678955078, 63.12157440185547,
    172.77383422851562, 63.165809631347656, 178.56570434570312,
    62.24188995361328, 192.7811279296875, 61.54795455932617,
)
MAP_BASE_SCALE = (
    145.95436096191406, 44.90191650390625, 146.35519409179688,
    46.02606201171875, 149.20570373535156, 44.7772216796875,
    150.05792236328125, 45.64059829711914, 151.98240661621094,
    45.37775421142578, 154.77890014648438, 44.939109802246094,
    157.52894592285156, 45.161991119384766, 160.46359252929688,
    44.672645568847656, 166.47596740722656, 43.7184944152832,
)


def _safe(value: Any) -> Any:
    if hasattr(value, "numpy"):
        value = value.numpy()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, dict):
        return {str(key): _safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_safe(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _output_directory(stage: str) -> Path:
    root = ROOT / ARTIFACT_ROOT
    root.mkdir(parents=True, exist_ok=True)
    for index in range(1, 100):
        path = root / f"{stage}_attempt{index:02d}"
        try:
            path.mkdir()
        except FileExistsError:
            continue
        return path
    raise RuntimeError("no unused attempt directory remains")


def _write_json(path: Path, payload: Any) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(_safe(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _mean_variance(values: list[float]) -> dict[str, float]:
    variance = statistics.variance(values) if len(values) > 1 else 0.0
    return {
        "mean": statistics.fmean(values),
        "sample_variance": variance,
        "sample_sd": math.sqrt(variance),
        "mcse": math.sqrt(variance / len(values)),
    }


def _make_evaluator(
    model: Any,
    particle_count: int,
    route: str,
    reset: str,
    transport_plan: str,
):
    controls = ROUTE_CONTROLS[route]
    horizon = int(model.observations.shape[0])
    dimension = model.callbacks.state_dimension
    process_steps = horizon
    ancestry_policy = {
        "iid_dual_cap": "existing_one_to_one",
        "previous_inverse_cdf": "hilbert_inverse_cdf",
        "repaired_fixed_previous_controls": "hilbert_permutation_one_to_one",
        "repaired_permutation": "hilbert_permutation_one_to_one",
    }[route]
    full_sqmc = route != "iid_dual_cap"
    chunks = (
        select_transport_chunks(particle_count)
        if transport_plan == "streaming"
        else None
    )

    @tf.function(
        input_signature=(
            tf.TensorSpec([3], tf.float32),
            tf.TensorSpec([horizon, 9], tf.float32),
            tf.TensorSpec([particle_count, dimension], tf.float32),
            tf.TensorSpec([process_steps, particle_count, dimension], tf.float32),
            tf.TensorSpec([process_steps, particle_count], tf.float32),
            tf.TensorSpec([dimension], tf.float32),
            tf.TensorSpec([dimension], tf.float32),
            tf.TensorSpec([particle_count, dimension], tf.float32),
        ),
        jit_compile=True,
        autograph=False,
    )
    def evaluate(
        theta,
        observations,
        initial_noise,
        process_noise,
        ancestor_uniforms,
        location,
        scale,
        design,
    ):
        with tf.device("/GPU:0"):
            return finite_value_standard_score_initial_rqmc(
                model.callbacks,
                theta,
                observations,
                initial_noise,
                process_noise,
                design,
                ancestry_policy=ancestry_policy,
                process_ancestor_uniforms=ancestor_uniforms,
                state_map_location=location,
                state_map_scale=scale,
                hilbert_bits=int(controls["hilbert_bits"]),
                state_map_policy="fixed_supplied" if full_sqmc else "adaptive_empirical",
                reset_policy="contract_e",
                dual_cap_enabled=True,
                dual_cap_diagonal_steps=int(controls["diagonal_steps"]),
                dual_cap_diagonal_strength=float(controls["diagonal_strength"]),
                dual_cap_pairwise_steps=int(controls["pairwise_steps"]),
                dual_cap_pairwise_strength=float(controls["pairwise_strength"]),
                dual_cap_pairwise_particle_rms_cap=float(controls["radial_cap"]),
                dual_cap_coordinate_cap=float(controls["coordinate_cap"]),
                dual_cap_coordinate_cap_power=int(controls["coordinate_cap_power"]),
                trust_region_enabled=reset == "trust_region",
                trust_region_lm_damping=TRUST_CONTROLS["lm_damping"],
                trust_region_lm_scale_floor=TRUST_CONTROLS["lm_scale_floor"],
                trust_region_radius=TRUST_CONTROLS["radius"],
                score_child_block_size=126,
                transport_plan_mode=transport_plan,
                transport_row_chunk_size=(
                    chunks.row_chunk_size if chunks is not None else None
                ),
                transport_col_chunk_size=(
                    chunks.col_chunk_size if chunks is not None else None
                ),
                functional_time_loop=True,
                epsilon=float(controls["epsilon"]),
                sinkhorn_steps=int(controls["sinkhorn_steps"]),
                balance_steps=int(controls["balance_steps"]),
                ridge=float(controls["ridge"]),
                marginal_tolerance=TV_GATE,
            )

    return evaluate


def _row(
    model: Any,
    evaluator: Any,
    particle_count: int,
    route: str,
    reset: str,
    transport_plan: str,
    seed: int,
) -> dict[str, Any]:
    controls = ROUTE_CONTROLS[route]
    # Point-set generation is part of the frozen stochastic scope. TensorFlow's
    # stateless-normal implementation is device-specific, so construct all
    # inputs on CPU and transfer the resulting tensors into the GPU evaluator.
    with tf.device("/CPU:0"):
        inputs = campaign_inputs(model, seed, particle_count)
        design = _design(model.callbacks.state_dimension, particle_count)
    inputs = inputs["iid_existing" if route == "iid_dual_cap" else "full_sqmc_halton"]
    location = tf.constant(MAP_LOCATION, tf.float32)
    scale = tf.constant(MAP_BASE_SCALE, tf.float32) * float(
        controls["map_multiplier"]
    )
    if route == "iid_dual_cap":
        location = tf.zeros_like(location)
        scale = tf.ones_like(scale)
    started = time.perf_counter()
    value, score, diagnostics = evaluator(
        model.theta,
        model.observations,
        inputs["initial"],
        inputs["process"],
        inputs["ancestors"],
        location,
        scale,
        design,
    )
    finite = bool(tf.math.is_finite(value).numpy()) and bool(
        tf.reduce_all(tf.math.is_finite(score)).numpy()
    )
    valid = bool(diagnostics["program_valid"].numpy())
    maximum_tv = float(
        tf.reduce_max(diagnostics["post_quotient_column_tv_error"]).numpy()
    )
    maximum_saturation = float(
        tf.reduce_max(diagnostics["state_map_saturation_rate"]).numpy()
    )
    unique = int(tf.reduce_min(diagnostics["ancestry_unique_count"]).numpy())
    permutation_valid = bool(
        tf.reduce_all(diagnostics["ancestry_permutation_valid"]).numpy()
    )
    equal_weight_error = float(
        tf.reduce_max(diagnostics["ancestry_equal_weight_error"]).numpy()
    )
    expected_chunks = (
        select_transport_chunks(particle_count)
        if transport_plan == "streaming"
        else None
    )
    transport_plan_id = int(diagnostics["transport_plan_id"].numpy())
    transport_row_chunk_size = int(
        diagnostics["transport_row_chunk_size"].numpy()
    )
    transport_col_chunk_size = int(
        diagnostics["transport_col_chunk_size"].numpy()
    )
    row_valid = (
        finite
        and valid
        and maximum_tv <= TV_GATE
        and maximum_saturation == 0.0
        and int(diagnostics["trust_region_solver_id"].numpy())
        == (1 if reset == "trust_region" else 0)
        and transport_plan_id == (1 if transport_plan == "streaming" else 0)
        and transport_row_chunk_size
        == (
            expected_chunks.row_chunk_size
            if expected_chunks is not None
            else particle_count
        )
        and transport_col_chunk_size
        == (
            expected_chunks.col_chunk_size
            if expected_chunks is not None
            else particle_count
        )
    )
    if route in ("repaired_fixed_previous_controls", "repaired_permutation"):
        row_valid = (
            row_valid
            and unique == particle_count
            and permutation_valid
            and equal_weight_error <= EQUAL_WEIGHT_TOLERANCE
        )
    return {
        "particle_count": particle_count,
        "route": route,
        "reset_variant": reset,
        "transport_plan": transport_plan,
        "reset_route_id": (
            GENUT_SHAPE_SOLVER_ID
            if reset == "trust_region"
            else "dual_cap_genut_primal_b098_p8_radial2_v1"
        ),
        "controls": controls,
        "seed": seed,
        "value": float(value.numpy()) if finite else None,
        "score": [float(item) for item in score.numpy()] if finite else None,
        "finite": finite,
        "program_valid": valid,
        "row_valid": row_valid,
        "maximum_tv_error": maximum_tv,
        "maximum_saturation": maximum_saturation,
        "minimum_unique_ancestors": unique,
        "permutation_valid": permutation_valid,
        "maximum_equal_weight_error": equal_weight_error,
        "minimum_ess": float(tf.reduce_min(diagnostics["ess"]).numpy()),
        "score_child_block_size": int(
            diagnostics["score_child_block_size"].numpy()
        ),
        "transport_plan_id": transport_plan_id,
        "transport_row_chunk_size": transport_row_chunk_size,
        "transport_col_chunk_size": transport_col_chunk_size,
        "maximum_weight": float(
            tf.reduce_max(diagnostics["maximum_normalized_weight"]).numpy()
        ),
        "maximum_hilbert_ties": int(
            tf.reduce_max(diagnostics["hilbert_tie_count"]).numpy()
        ),
        "maximum_dual_cap_active_fraction": float(
            tf.reduce_max(diagnostics["dual_cap_active_fraction"]).numpy()
        ),
        "minimum_dual_cap_radial_scale": float(
            tf.reduce_min(diagnostics["dual_cap_minimum_radial_scale"]).numpy()
        ),
        "initial_sha256": inputs["initial_sha256"],
        "process_sha256": inputs["process_sha256"],
        "ancestor_sha256": inputs["ancestor_sha256"],
        "elapsed_seconds": time.perf_counter() - started,
        "value_device": value.device,
    }


def _aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for count in PARTICLE_COUNTS:
        for route in ROUTES:
            for reset in RESET_VARIANTS:
                cell = [
                    row
                    for row in rows
                    if row["particle_count"] == count
                    and row["route"] == route
                    and row["reset_variant"] == reset
                ]
                if not cell:
                    continue
                summary: dict[str, Any] = {
                    "particle_count": count,
                    "route": route,
                    "reset_variant": reset,
                    "count": len(cell),
                    "all_valid": all(row["row_valid"] for row in cell),
                    "score_components": [],
                    "maximum_tv_error": max(row["maximum_tv_error"] for row in cell),
                    "minimum_unique_ancestors": min(
                        row["minimum_unique_ancestors"] for row in cell
                    ),
                }
                if not summary["all_valid"]:
                    summary["invalid_rows"] = [
                        {
                            "seed": row["seed"],
                            "finite": row["finite"],
                            "program_valid": row["program_valid"],
                            "maximum_tv_error": row["maximum_tv_error"],
                            "minimum_ess": row["minimum_ess"],
                        }
                        for row in cell
                        if not row["row_valid"]
                    ]
                    output.append(summary)
                    continue
                summary["value"] = _mean_variance(
                    [row["value"] for row in cell]
                )
                for index in range(3):
                    component = _mean_variance(
                        [row["score"][index] for row in cell]
                    )
                    component["n_times_variance"] = (
                        count * component["sample_variance"]
                    )
                    summary["score_components"].append(component)
                summary["total_score_variance"] = sum(
                    component["sample_variance"]
                    for component in summary["score_components"]
                )
                output.append(summary)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", choices=("smoke", "claim"), required=True)
    parser.add_argument(
        "--particle-counts",
        type=int,
        nargs="+",
        default=list(PARTICLE_COUNTS),
    )
    parser.add_argument(
        "--routes", nargs="+", choices=ROUTES, default=list(ROUTES)
    )
    parser.add_argument(
        "--resets",
        nargs="+",
        choices=RESET_VARIANTS,
        default=list(DEFAULT_RESET_VARIANTS),
        help=(
            "reset variants to execute; trust_region is the SQMC test default, "
            "and legacy remains an explicit historical comparator"
        ),
    )
    parser.add_argument(
        "--transport-plan",
        choices=("dense", "streaming"),
        default="dense",
    )
    args = parser.parse_args()
    if any(count not in PARTICLE_COUNTS for count in args.particle_counts):
        raise ValueError("particle count outside reviewed ladder")
    memory_policy = MEMORY_POLICY
    tf.config.set_soft_device_placement(False)
    tf.config.experimental.enable_tensor_float_32_execution(True)
    logical = tf.config.list_logical_devices("GPU")
    if len(logical) != 1:
        raise RuntimeError("campaign requires exactly one visible logical GPU")
    with tf.device("/CPU:0"):
        models = build_full_horizon_models(include_references=False)
    model = next(item for item in models if item.row_id == "austria_sir_T20")
    seeds = SMOKE_SEEDS if args.stage == "smoke" else CLAIM_SEEDS
    output = _output_directory(args.stage)
    rows: list[dict[str, Any]] = []
    checkpoint = output / "checkpoint.json"
    started = time.perf_counter()
    for count in args.particle_counts:
        for route in args.routes:
            for reset in args.resets:
                evaluator = _make_evaluator(
                    model, count, route, reset, args.transport_plan
                )
                for seed in seeds:
                    row = _row(
                        model,
                        evaluator,
                        count,
                        route,
                        reset,
                        args.transport_plan,
                        seed,
                    )
                    rows.append(row)
                    _write_json(checkpoint, {"schema": SCHEMA, "rows": rows})
                    print(
                        json.dumps(
                            {
                                "count": count,
                                "route": route,
                                "reset": reset,
                                "seed": seed,
                                "valid": row["row_valid"],
                            }
                        ),
                        flush=True,
                    )
    result_path = output / "result.json"
    source_paths = (
        Path(__file__),
        ROOT / PLAN,
        ROOT / "bayesfilter/highdim/genut_guided_proposal_tf.py",
        ROOT / "bayesfilter/highdim/ledh_pfpf_genut_initial_rqmc_tf.py",
        ROOT / "bayesfilter/highdim/higher_moment_contract_e.py",
        ROOT / "bayesfilter/highdim/genut_shape_lm_tf.py",
        ROOT / "bayesfilter/highdim/sqmc_tf.py",
    )
    payload = {
        "schema": SCHEMA,
        "status": (
            f"{args.stage}_complete"
            if all(row["row_valid"] for row in rows)
            else f"{args.stage}_complete_with_invalid_rows"
        ),
        "plan": str(PLAN),
        "model_id": model.row_id,
        "horizon": int(model.observations.shape[0]),
        "particle_counts": args.particle_counts,
        "routes": args.routes,
        "reset_variants": args.resets,
        "default_reset_variants": list(DEFAULT_RESET_VARIANTS),
        "transport_plan": args.transport_plan,
        "seeds": list(seeds),
        "rows": rows,
        "summaries": _aggregate(rows),
        "trust_region": {
            "route_id": GENUT_SHAPE_SOLVER_ID,
            "controls": TRUST_CONTROLS,
        },
        "device": {
            "logical_devices": [item.name for item in logical],
            "tf32": True,
            "jit_compile": True,
            "trust_basis": "owner_designated_managed_session_visible_gpu_trusted",
        },
        "memory_policy": dict(memory_policy),
        "gpu_allocator": {
            key: int(value)
            for key, value in tf.config.experimental.get_memory_info("GPU:0").items()
        },
        "git_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "wall_time_seconds": time.perf_counter() - started,
        "run_manifest": {
            "command": [sys.executable, *sys.argv],
            "environment": sys.prefix,
            "host": platform.node(),
            "python": platform.python_version(),
            "tensorflow": tf.__version__,
            "output_json": str(result_path.relative_to(ROOT)),
            "source_sha256": {
                str(path.relative_to(ROOT)): _sha256(path) for path in source_paths
            },
        },
        "nonclaims": [
            "no exact observed-data SIR score",
            "no statistically supported route ranking",
            "no formal SQMC theorem transfer through Contract-E",
            "no HMC, NeuTra, or default readiness",
        ],
    }
    _write_json(result_path, payload)
    print(json.dumps({"status": payload["status"], "output": str(output)}))


if __name__ == "__main__":
    main()
