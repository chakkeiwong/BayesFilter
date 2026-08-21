"""P2A cost prototype: batched tangent-replay cost vs value cost (one step).

Plan obligations (rev 3, P2A):
1. forward+dot_A vs chunked-forward modes at p in {3, 30, 300};
2. solver-reuse checks: scaled-primal vs normal-equation agreement,
   derivative consistency vs the actual scaled primal solver, cost with
   and without factorization reuse;
3. same-scalar FD spot check at p=3;
4. peak memory tracked (tracemalloc; eager CPU float64 — recorded as
   ENGINEERING measurements only, no feasibility language per V12).

Adjoint mode is represented by its lower-bound proxy (value pass + one
transposed solve set); a true adjoint implementation is a P2 decision
input, not built here. The full-horizon T=120 stress is a separate
obligation and runs after the mode choice (this script's scope is the
mode-selection measurement the plan names).

Scope guard: one fitted ALS step on a synthetic smooth target over the
branch-axis block at n=2 (the fitted-step shape the engine uses); model
tangents are synthetic smooth functions with a leading p-axis.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import platform
import subprocess
import sys
import time
import tracemalloc
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "1")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import tensorflow as tf

import bayesfilter.highdim as highdim
from bayesfilter.highdim.derivatives import (
    differentiate_design_matrix,
    fixed_design_lsq_derivative,
)
from bayesfilter.highdim.fitting import FixedTTFitConfig, FixedTTFitter, _solve_scaled_augmented_ridge
from bayesfilter.highdim.squared_tt_engine_v0_tf import DiscreteIndicatorBasis1D
from bayesfilter.highdim.tt import TTCore

DTYPE = tf.float64
PLAN = "docs/plans/bayesfilter-zhao-cui-generic-highdim-analytic-score-program-plan-2026-08-15.md"


def _basis(n: int, degree: int, branch: int) -> highdim.ProductBasis:
    convention = highdim.MeasureConvention(
        density_measure=highdim.DensityMeasure.REFERENCE_MEASURE,
        mass_measure=highdim.MassMeasure.REFERENCE_MEASURE,
        reference_weight_name="omega",
    )
    bases = (
        [highdim.LegendreBasis1D(highdim.BoundedInterval(-1.0, 1.0), degree) for _ in range(n)]
        + [DiscreteIndicatorBasis1D(branch)]
        + [highdim.LegendreBasis1D(highdim.BoundedInterval(-1.0, 1.0), degree) for _ in range(n)]
    )
    return highdim.ProductBasis(bases, convention)


def _rows(count: int, n: int, branch: int, seed: int) -> tf.Tensor:
    z = tf.random.stateless_uniform([count, 2 * n], tf.constant((seed, 3), tf.int32), -1.0, 1.0, DTYPE)
    g = tf.cast(
        tf.random.stateless_uniform([count, 1], tf.constant((seed, 4), tf.int32), 0, branch, tf.int32),
        DTYPE,
    )
    return tf.concat([z[:, :n], g, z[:, n:]], axis=1)


def _target_and_tangents(rows: tf.Tensor, p: int, seed: int) -> tuple[tf.Tensor, tf.Tensor]:
    z = tf.concat([rows[:, :2], rows[:, 3:]], axis=1)
    base = tf.exp(-0.5 * tf.reduce_sum(tf.square(z), axis=1)) * (
        1.0 + 0.3 * tf.sin(2.0 * z[:, 0]) * tf.cos(z[:, 1])
    )
    rng = np.random.default_rng(seed)
    freq = tf.constant(rng.uniform(0.5, 2.0, size=(p, 4)), DTYPE)
    dots = base[None, :] * tf.sin(
        tf.einsum("pk,nk->pn", freq, z) + tf.constant(rng.uniform(0, 3, size=(p, 1)), DTYPE)
    )
    return base, dots  # [N], [p, N]


def _init_cores(dims: list[int], rank: int, seed: int) -> tuple[TTCore, ...]:
    cores = []
    for axis, dim in enumerate(dims):
        left = 1 if axis == 0 else rank
        right = 1 if axis == len(dims) - 1 else rank
        cores.append(
            TTCore(
                0.3
                * tf.random.stateless_normal(
                    [left, dim, right], tf.constant((seed, 50 + axis), tf.int32), dtype=DTYPE
                )
            )
        )
    return tuple(cores)


def _value_sweep(
    basis, rows, target, weights, cores, config: FixedTTFitConfig, sweeps: int
) -> tuple[tuple[TTCore, ...], list[Any], float]:
    """Value ALS; returns cores, per-update (design, normal factor) cache, wall."""

    fitter = FixedTTFitter()
    cache = []
    start = time.perf_counter()
    current = cores
    for _s in range(sweeps):
        for idx in range(len(current)):
            system = fitter.build_core_update_system(
                basis, rows, target, weights, current, idx, config
            )
            solve = _solve_scaled_augmented_ridge(
                design=system.design_matrix, target_values=target, weights=weights, ridge=config.ridge
            )
            updated = list(current)
            updated[idx] = TTCore(tf.reshape(solve.solution, current[idx].values.shape))
            current = tuple(updated)
            cache.append((idx, system.design_matrix, tuple(c for c in current)))
    return current, cache, time.perf_counter() - start


def _forward_tangent_replay(
    basis, rows, target, dot_targets, weights, cores0, config, sweeps: int, chunk: int | None
) -> tuple[float, int]:
    """Ordered replay with dot_A for all p tangents; returns (wall, peak_bytes).

    Factorization sharing: the primal normal matrix is formed once per
    update and `tf.linalg.lu` factors it once; all tangent RHS columns
    reuse the factors via lu_solve (multi-RHS).
    """

    fitter = FixedTTFitter()
    p = int(dot_targets.shape[0])
    tracemalloc.start()
    start = time.perf_counter()
    current = cores0
    dot_current = tuple(TTCore(tf.zeros_like(c.values)) for c in cores0)  # per-parameter stacks
    dot_stacks = [tuple(TTCore(tf.zeros_like(c.values)) for c in cores0) for _ in range(p)]
    for _s in range(sweeps):
        for idx in range(len(current)):
            system = fitter.build_core_update_system(
                basis, rows, target, weights, current, idx, config
            )
            design = system.design_matrix
            normal = tf.matmul(design, design * weights[:, None], transpose_a=True) + config.ridge * tf.eye(
                int(design.shape[1]), dtype=DTYPE
            )
            lu, perm = tf.linalg.lu(normal)
            rhs = tf.linalg.matvec(design, weights * target, transpose_a=True)
            solution = tf.linalg.lu_solve(lu, perm, rhs[:, None])[:, 0]
            new_core = TTCore(tf.reshape(solution, current[idx].values.shape))

            parameter_range = range(p)
            chunks = (
                [list(parameter_range)] if chunk is None else
                [list(parameter_range)[i : i + chunk] for i in range(0, p, chunk)]
            )
            for chunk_ids in chunks:
                dot_rhs_columns = []
                for j in chunk_ids:
                    dot_design = differentiate_design_matrix(
                        basis, rows, current, dot_stacks[j], idx
                    )
                    dot_normal_c = (
                        tf.linalg.matvec(
                            tf.matmul(dot_design, design * weights[:, None], transpose_a=True)
                            + tf.matmul(design, dot_design * weights[:, None], transpose_a=True),
                            solution,
                        )
                    )
                    dot_rhs = (
                        tf.linalg.matvec(dot_design, weights * target, transpose_a=True)
                        + tf.linalg.matvec(design, weights * dot_targets[j], transpose_a=True)
                    )
                    dot_rhs_columns.append(dot_rhs - dot_normal_c)
                stacked = tf.stack(dot_rhs_columns, axis=1)
                dot_solutions = tf.linalg.lu_solve(lu, perm, stacked)
                for column, j in enumerate(chunk_ids):
                    updated = list(dot_stacks[j])
                    updated[idx] = TTCore(
                        tf.reshape(dot_solutions[:, column], current[idx].values.shape)
                    )
                    dot_stacks[j] = tuple(updated)
            updated_cores = list(current)
            updated_cores[idx] = new_core
            current = tuple(updated_cores)
    wall = time.perf_counter() - start
    _current_bytes, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return wall, peak


def _solver_reuse_checks(basis, rows, target, dot_targets, weights, cores0, config) -> dict[str, Any]:
    """Plan-bound checks: scaled-primal vs normal-equation agreement and
    derivative consistency vs the actual scaled solver."""

    fitter = FixedTTFitter()
    system = fitter.build_core_update_system(basis, rows, target, weights, cores0, 0, config)
    design = system.design_matrix
    scaled = _solve_scaled_augmented_ridge(
        design=design, target_values=target, weights=weights, ridge=config.ridge
    )
    normal = tf.matmul(design, design * weights[:, None], transpose_a=True) + config.ridge * tf.eye(
        int(design.shape[1]), dtype=DTYPE
    )
    rhs = tf.linalg.matvec(design, weights * target, transpose_a=True)
    direct = tf.linalg.solve(normal, rhs[:, None])[:, 0]
    agreement = float(
        (tf.linalg.norm(scaled.solution - direct) / tf.maximum(tf.linalg.norm(direct), 1.0)).numpy()
    )
    dot_design = differentiate_design_matrix(
        basis, rows, cores0, tuple(TTCore(0.1 * tf.ones_like(c.values)) for c in cores0), 0
    )
    derivative = fixed_design_lsq_derivative(
        design_matrix=design,
        target_values=target,
        weights=weights,
        coefficients=scaled.solution,
        dot_target_values=dot_targets[0],
        ridge=config.ridge,
        dot_design_matrix=dot_design,
    )
    step = 1e-6
    # FD of the SCALED solver along the same perturbation direction
    perturbed_cores = tuple(
        TTCore(c.values + step * 0.1 * tf.ones_like(c.values)) for c in cores0
    )
    system_plus = fitter.build_core_update_system(
        basis, rows, target + step * dot_targets[0], weights, perturbed_cores, 0, config
    )
    plus = _solve_scaled_augmented_ridge(
        design=system_plus.design_matrix,
        target_values=target + step * dot_targets[0],
        weights=weights,
        ridge=config.ridge,
    )
    system_minus = fitter.build_core_update_system(
        basis, rows, target - step * dot_targets[0], weights,
        tuple(TTCore(c.values - step * 0.1 * tf.ones_like(c.values)) for c in cores0), 0, config,
    )
    minus = _solve_scaled_augmented_ridge(
        design=system_minus.design_matrix,
        target_values=target - step * dot_targets[0],
        weights=weights,
        ridge=config.ridge,
    )
    fd = (plus.solution - minus.solution) / (2.0 * step)
    consistency = float(
        (tf.linalg.norm(derivative.dot_coefficients - fd) / tf.maximum(tf.linalg.norm(fd), 1.0)).numpy()
    )
    return {
        "scaled_vs_normal_relative": agreement,
        "derivative_vs_scaled_solver_fd_relative": consistency,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    n, degree, branch, rank, sweeps, n_rows = 2, 10, 5, 4, 2, 2048
    basis = _basis(n, degree, branch)
    dims = [int(b.basis_dim) for b in basis.bases]
    rows = _rows(n_rows, n, branch, 71)
    weights = tf.fill([n_rows], tf.constant(1.0 / n_rows, DTYPE))
    config = FixedTTFitConfig(
        ranks=tuple([1] + [rank] * (len(dims) - 1) + [1]),
        ridge=1e-10, max_sweeps=sweeps, sweep_order=tuple(range(len(dims))),
        row_budget=n_rows, column_budget=4096,
        dense_matrix_byte_budget=1 << 30, normal_matrix_byte_budget=1 << 30,
        condition_number_warning=1e12, condition_number_veto=1e14,
        holdout_tolerance=1e30,
    )
    cores0 = _init_cores(dims, rank, 72)

    results: dict[str, Any] = {"cells": []}
    target, _ = _target_and_tangents(rows, 1, 73)
    _cores, _cache, value_wall = _value_sweep(basis, rows, target, weights, cores0, config, sweeps)
    results["value_wall_seconds"] = value_wall

    for p in (3, 30, 300):
        _t, dots = _target_and_tangents(rows, p, 73)
        for mode, chunk in (("forward_full", None), ("forward_chunk32", 32)):
            if mode == "forward_chunk32" and p <= 32:
                continue
            wall, peak = _forward_tangent_replay(
                basis, rows, target, dots, weights, cores0, config, sweeps, chunk
            )
            results["cells"].append(
                {
                    "p": p, "mode": mode, "wall_seconds": wall,
                    "gradient_to_value_ratio": wall / value_wall,
                    "tracemalloc_peak_mb": peak / 1e6,
                }
            )
            print(json.dumps(results["cells"][-1]))

    results["solver_reuse_checks"] = _solver_reuse_checks(
        basis, rows, target, _target_and_tangents(rows, 3, 73)[1], weights, cores0, config
    )
    results.update(
        {
            "schema_version": "p2a_cost_prototype.v1",
            "plan": PLAN,
            "timestamp_utc": _dt.datetime.now(tz=_dt.timezone.utc).isoformat(),
            "host": platform.node(),
            "git_commit": subprocess.run(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True
            ).stdout.strip(),
            "config": {
                "n": n, "degree": degree, "branch": branch, "rank": rank,
                "sweeps": sweeps, "rows": n_rows,
            },
            "nonclaims": [
                "one-step eager CPU measurement; NOT full-horizon feasibility (plan P2A binds a separate T=120 stress)",
                "adjoint mode not implemented here; decision inputs only",
                "no wall-clock feasibility language (V12): engineering measurement",
            ],
        }
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"value_wall": value_wall, "cells": len(results["cells"])}))


if __name__ == "__main__":
    main()
