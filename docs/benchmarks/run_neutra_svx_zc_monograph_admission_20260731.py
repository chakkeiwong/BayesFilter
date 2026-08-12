#!/usr/bin/env python3
"""Bounded CPU admission screen for the monograph fixed-branch SVX-ZC route."""

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
from typing import Any, Mapping

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PLAN = ROOT / "docs/plans/bayesfilter-svx-zc-ukf-initializer-default-admission-plan-2026-08-01.md"
ROW = "actual_sv"
HORIZON = 10
DEGREE = 8
ORDER = 17
RANKS = (1, 2, 4, 6)
TAU = 1.0e-8
COORDINATE_HALF_WIDTH = 8.0
FIT_RESIDUAL_VETO = 1.0e-8
CONDITION_VETO = 1.0e10
MASS_TOLERANCE = 1.0e-10
FD_STEPS = (1.0e-2, 3.0e-3, 1.0e-3, 3.0e-4)


def _json_value(value: Any) -> Any:
    import tensorflow as tf

    if isinstance(value, tf.Tensor):
        array = value.numpy()
        return array.item() if array.ndim == 0 else array.tolist()
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    return value


def _sha256_tensor(value: Any) -> str:
    import tensorflow as tf

    return hashlib.sha256(tf.io.serialize_tensor(value).numpy()).hexdigest()


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()


def _rank_record(rank: int) -> dict[str, Any]:
    import tensorflow as tf

    from bayesfilter.highdim import (
        AffineCoordinateMap,
        ProductBasis,
        LegendreBasis1D,
        MeasureConvention,
        DensityMeasure,
        MassMeasure,
        ScalarAdjacentTTConfig,
    )
    import docs.benchmarks.run_contract_e_tp_phase6_zhao_cui_comparator as comparator
    from bayesfilter.highdim.sv_mixture_cut4 import (
        exact_transformed_sv_scalar_dense_reference,
    )

    model, theta, observations = comparator._row_inputs(ROW, HORIZON)
    raw_observations = comparator._sv_dataset(81101)["observations"][:HORIZON]
    initial_cores, adjacent_initial_cores, ukf_manifest = comparator._ukf_initial_cores(
        model=model,
        theta=theta,
        raw_observations=raw_observations,
        degree=DEGREE,
        order=ORDER,
        rank=rank,
        coordinate_half_width=COORDINATE_HALF_WIDTH,
    )
    config = comparator._comparator_config(
        degree=DEGREE,
        order=ORDER,
        rank=rank,
        seed=f"svx-zc-monograph-admission-20260731-rank{rank}",
        transition_before_first_observation=False,
        coordinate_half_width=COORDINATE_HALF_WIDTH,
        density_tau=0.0,
        initial_cores=initial_cores,
        adjacent_initial_cores=adjacent_initial_cores,
        initialization_rule=ukf_manifest["initializer_rule"],
    )
    args = argparse.Namespace(
        row=ROW,
        horizon=HORIZON,
        degree=DEGREE,
        order=ORDER,
        rank=rank,
        coordinate_half_width=COORDINATE_HALF_WIDTH,
        density_tau=TAU,
        seed=f"svx-zc-monograph-admission-20260731-rank{rank}",
        target_preparation=None,
    )
    started = time.perf_counter()
    candidate = comparator._run(args)
    wall_time = time.perf_counter() - started
    result = None
    # Recompute the in-memory result so the structural checks inspect the
    # actual saved density objects, not only serialized summaries.
    value_result = __import__("bayesfilter.highdim", fromlist=["scalar_adjacent_state_fixed_tt_value"]).scalar_adjacent_state_fixed_tt_value(
        model,
        theta,
        observations,
        config,
        fixture_id=f"svx-zc-monograph-admission.rank{rank}.value",
        branch_seed_prefix=f"svx-zc-monograph-admission-20260731-rank{rank}",
    )
    result = __import__("bayesfilter.highdim", fromlist=["scalar_adjacent_state_fixed_tt_score"]).scalar_adjacent_state_fixed_tt_score(
        model,
        theta,
        observations,
        config,
        finite_difference_h=FD_STEPS,
        fixture_id=f"svx-zc-monograph-admission.rank{rank}",
        branch_seed_prefix=f"svx-zc-monograph-admission-20260731-rank{rank}",
    )

    coordinate_map = config.scalar_coordinate_map
    reference = tf.constant([[-0.75], [0.0], [0.75]], tf.float64)
    physical, forward_log_det = coordinate_map.forward(reference)
    recovered, inverse_log_det = coordinate_map.inverse(physical)
    coordinate_pass = bool(
        tf.reduce_all(tf.abs(recovered - reference) <= 1.0e-12).numpy()
        and tf.reduce_all(tf.abs(forward_log_det + inverse_log_det) <= 1.0e-12).numpy()
    )

    positivity_rows = []
    closure_rows = []
    max_condition = 0.0
    max_residual = 0.0
    all_finite = True
    positivity_pass = True
    closure_pass = True
    for index, step in enumerate(value_result.steps):
        basis = step.density.sqrt_tt.product_basis
        if basis.dimension == 1:
            points = reference
            values = step.density.normalized_marginal_density_values((0,), points)
        else:
            nodes, _weights = comparator.highdim.legendre_gauss_nodes_weights(ORDER)
            grid = tf.stack(tf.meshgrid(nodes, nodes, indexing="ij"), axis=-1)
            points = tf.reshape(grid, (-1, 2))
            values = step.density.normalized_marginal_density_values((0, 1), points)
        finite = bool(tf.reduce_all(tf.math.is_finite(values)).numpy())
        nonnegative = bool(tf.reduce_all(values >= -1.0e-14).numpy())
        positivity_rows.append(
            {"time_index": index, "finite": finite, "nonnegative": nonnegative,
             "min_value": float(tf.reduce_min(values).numpy())}
        )
        positivity_pass = positivity_pass and finite and nonnegative
        all_finite = all_finite and finite
        mass_error = abs(float(step.marginal_mass.numpy()) - 1.0)
        closure_rows.append({"time_index": index, "mass_error": mass_error,
                             "query_count": int(points.shape[0])})
        closure_pass = closure_pass and mass_error <= MASS_TOLERANCE
        for update in step.fit_result.core_update_statuses:
            condition = update.get("condition_number")
            if condition is not None:
                max_condition = max(max_condition, float(condition))
        max_residual = max(max_residual, float(step.fit_result.fit_residual.numpy()))

    rank_saturation_pass = max_residual <= FIT_RESIDUAL_VETO
    condition_pass = max_condition <= CONDITION_VETO
    fd_pass = candidate["own_scalar_fd"]["status"] == "pass"
    finite_value_score = bool(
        math.isfinite(float(result.log_likelihood.numpy()))
        and bool(tf.reduce_all(tf.math.is_finite(result.score)).numpy())
    )
    # The candidate consumes transformed observations; the independent dense
    # helper consumes raw y and applies log(y**2) internally.
    reference_result = exact_transformed_sv_scalar_dense_reference(
        model, theta, raw_observations, order=321, radius=8.0
    )
    value_gap = float(
        abs(value_result.log_likelihood.numpy() - reference_result.log_likelihood.numpy())
    )
    hard_pass = all(
        (
            finite_value_score,
            coordinate_pass,
            positivity_pass,
            closure_pass,
            condition_pass,
            rank_saturation_pass,
            fd_pass,
            candidate["hard_vetoes"]["forbidden_retained_grid_route_used"] is False,
        )
    )
    return {
        "rank": rank,
        "status": "PASS_HARD_VETOES" if hard_pass else "BLOCKED_HARD_VETO",
        "candidate": candidate,
        "value": float(value_result.log_likelihood.numpy()),
        "score": _json_value(result.score),
        "reference": {
            "value": float(reference_result.log_likelihood.numpy()),
            "order": 321,
            "radius": 8.0,
            "value_gap": value_gap,
            "value_gap_per_observation": value_gap / HORIZON,
        },
        "hard_vetoes": {
            "finite_value_and_score": finite_value_score,
            "coordinate_jacobian_consistency": coordinate_pass,
            "positivity": positivity_pass,
            "retained_marginal_closure": closure_pass,
            "condition_number": condition_pass,
            "rank_saturation_residual": rank_saturation_pass,
            "same_scalar_fd": fd_pass,
            "forbidden_retained_grid_route_used": False,
        },
        "diagnostics": {
            "max_condition_number": max_condition,
            "max_fit_residual": max_residual,
            "fit_residual_veto": FIT_RESIDUAL_VETO,
            "condition_veto": CONDITION_VETO,
            "tau": TAU,
            "positivity_rows": positivity_rows,
            "closure_rows": closure_rows,
            "all_finite": all_finite,
            "wall_time_seconds": wall_time,
        },
        "route_identity": {
            "route_id": "zhao_cui_fixed_adjacent_state_squared_tt_v1",
            "classification": "extension_or_invention",
            "authority": "docs/main.tex -> ch36b/ch37/ch38",
            "source_route_veto": "not_applicable_under_monograph_authority",
        },
        "nonclaims": [
            "not author-source-faithful",
            "not exact filtering",
            "not posterior correctness",
            "not NeuTra batch-native training readiness",
            "not HMC convergence or default readiness",
        ],
    }


def run(output_root: Path) -> Mapping[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"refusing to overwrite existing root: {output_root}")
    output_root.mkdir(parents=True)
    started = time.perf_counter()
    records = []
    for rank in RANKS:
        record = _rank_record(rank)
        records.append(record)
        (output_root / f"rank{rank}.json").write_text(
            json.dumps(_json_value(record), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    passed = [record["rank"] for record in records if record["status"] == "PASS_HARD_VETOES"]
    return {
        "schema_version": "bayesfilter.svx_zc.monograph_admission.v1",
        "status": "ADMITTED_FIXED_BRANCH_CANDIDATE" if passed else "NO_RANK_PASSED_HARD_VETOES",
        "cell_id": "SVX-ZC",
        "route_id": "zhao_cui_fixed_adjacent_state_squared_tt_v1",
        "route_classification": "extension_or_invention",
        "authority": "docs/main.tex and included ch36b/ch37/ch38",
        "rank_ladder": RANKS,
        "passed_ranks": passed,
        "records": records,
        "run_manifest": {
            "git_commit": _git_commit(),
            "command": " ".join(sys.argv),
            "environment": os.environ.get("CONDA_DEFAULT_ENV", "not_detected"),
            "python": sys.executable,
            "python_version": platform.python_version(),
            "device": "CPU-only; CUDA_VISIBLE_DEVICES=-1",
            "dtype": "float64",
            "jit_compile": False,
            "seed_policy": "rank-bound deterministic branch prefixes",
            "data_seed": 81101,
            "horizon": HORIZON,
            "observation_hash": _sha256_tensor(
                __import__(
                    "docs.benchmarks.run_contract_e_tp_phase6_zhao_cui_comparator",
                    fromlist=["_row_inputs"],
                )._row_inputs(ROW, HORIZON)[2]
            ),
            "plan": str(PLAN.relative_to(ROOT)),
            "wall_time_seconds": time.perf_counter() - started,
            "output_root": str(output_root.relative_to(ROOT)),
        },
        "decision": (
            "A hard-veto pass removes only the obsolete source-route blocker; "
            "a separate batch-native target-adapter plan remains required."
            if passed
            else "Preserve SVX-ZC blocked on the observed numerical veto and repair the smallest failed gate."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output_root = args.output_root if args.output_root.is_absolute() else ROOT / args.output_root
    result = run(output_root)
    (output_root / "result.json").write_text(
        json.dumps(_json_value(result), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_root / "run_manifest.json").write_text(
        json.dumps(_json_value(result["run_manifest"]), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": result["status"], "output_root": str(output_root)}))


if __name__ == "__main__":
    main()
