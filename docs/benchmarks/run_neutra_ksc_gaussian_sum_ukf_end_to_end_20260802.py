#!/usr/bin/env python3
"""Run the scoped KSC Gaussian-sum NeuTra campaign.

This runner deliberately uses a private CellSpec rather than the historical
KSC registry entry, whose target is a different filter and horizon.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bayesfilter.testing.ksc_gaussian_sum_ukf_scope import (
    KSC_GAUSSIAN_SUM_UKF_PARAMETER_NAMES,
    KSC_GAUSSIAN_SUM_UKF_TARGET_SIGNATURE,
)
from bayesfilter.testing.neutra_model_registry_tf import CellSpec, RecipeSpec


def _adapter():
    from bayesfilter.testing.ksc_gaussian_sum_ukf_neutra_target_tf import (
        make_ksc_gaussian_sum_ukf_neutra_adapter,
    )

    return make_ksc_gaussian_sum_ukf_neutra_adapter()


def _identity_geometry(tf):
    return tf.zeros((2,), tf.float64), tf.eye(2, dtype=tf.float64), {
        "center_role": "source_prior_origin",
        "factor_role": "identity",
        "path": None,
    }


def _physical(tf, values):
    from bayesfilter.testing.exact_sv_sgqf_neutra_target_tf import (
        source_chart_physical_parameters,
    )

    shape = tf.shape(values)
    flat = tf.reshape(tf.convert_to_tensor(values, tf.float64), (-1, 2))
    gamma, beta = source_chart_physical_parameters(flat)
    return tf.reshape(tf.stack((gamma, beta), axis=-1), shape)


def _truth(tf):
    return tf.constant((0.6, 0.4), tf.float64)


def build_spec() -> CellSpec:
    return CellSpec(
        cell_id="KSC-UKF-GAUSSIAN-SUM-T20",
        parameter_dim=2,
        parameter_names=KSC_GAUSSIAN_SUM_UKF_PARAMETER_NAMES,
        target_signature=KSC_GAUSSIAN_SUM_UKF_TARGET_SIGNATURE,
        adapter_factory=_adapter,
        geometry_factory=_identity_geometry,
        physical_transform=_physical,
        truth_factory=_truth,
        recipes=(
            RecipeSpec("ksc_narrow_lr1e3", (8, 8), 1.0e-3),
            RecipeSpec("ksc_narrow_lr5e3", (8, 8), 5.0e-3),
            RecipeSpec("ksc_wide_lr1e3", (16, 16), 1.0e-3),
            RecipeSpec("ksc_wide_lr5e3", (16, 16), 5.0e-3),
        ),
        initial_seed=(20260802, 2871),
        target_description=(
            "KSC T20 seven-component mass-preserving Gaussian-sum UKF "
            "posterior, component cap 32"
        ),
        initial_step_size=0.1,
        common_tuning_status_keys=(
            "status_code",
            "valid_pre_regularized_score",
            "floor_count_value",
            "min_innovation_eigenvalue",
            "innovation_condition_estimate",
        ),
        leapfrog_grid=(6, 10),
        plan_path=(
            "docs/plans/bayesfilter-ksc-ukf-neutra-hmc-continuation-plan-"
            "2026-08-02.md"
        ),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--action", choices=("screen", "cell", "broad-grid", "sequential"), required=True
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--screen-result", type=Path)
    parser.add_argument("--screen-result-sha256")
    parser.add_argument("--screen-steps", type=int, default=500)
    parser.add_argument("--final-steps", type=int, default=5000)
    parser.add_argument("--frozen-transport", type=Path)
    parser.add_argument("--frozen-transport-sha256")
    parser.add_argument("--broad-grid-result", type=Path)
    parser.add_argument("--broad-grid-result-sha256")
    parser.add_argument("--broad-grid-seed", nargs=2, type=int, default=(20260803, 2881))
    parser.add_argument("--screen-results", type=int, default=65)
    parser.add_argument("--chunk-results", type=int, default=65)
    args = parser.parse_args()

    import tensorflow as tf
    from bayesfilter.inference.neutra_end_to_end import (
        EndToEndConfig,
        run_neutra_end_to_end_cell,
    )

    spec = build_spec()
    if args.action == "screen":
        config = EndToEndConfig(
            output_root=args.output_root,
            screen_steps=args.screen_steps,
            final_steps=args.final_steps,
            screen_only=True,
        )
    elif args.action == "cell":
        if args.screen_result is None or args.screen_result_sha256 is None:
            raise SystemExit("cell requires --screen-result and --screen-result-sha256")
        config = EndToEndConfig(
            output_root=args.output_root,
            screen_steps=args.screen_steps,
            final_steps=args.final_steps,
            screen_result_path=args.screen_result,
            expected_screen_result_sha256=args.screen_result_sha256,
        )
    elif args.action == "broad-grid":
        if args.frozen_transport is None or args.frozen_transport_sha256 is None:
            raise SystemExit("broad-grid requires --frozen-transport and --frozen-transport-sha256")
        from bayesfilter.inference.neutra_end_to_end import (
            FrozenTransportBroadGridConfig,
            run_neutra_frozen_transport_broad_grid_cell,
        )
        result = run_neutra_frozen_transport_broad_grid_cell(
            spec=spec,
            config=FrozenTransportBroadGridConfig(
                output_root=args.output_root,
                frozen_transport_path=args.frozen_transport,
                expected_frozen_transport_sha256=args.frozen_transport_sha256,
                root_seed=tuple(args.broad_grid_seed),
                initial_step_size=0.1,
                screen_results=args.screen_results,
            ),
        )
    else:
        if (
            args.frozen_transport is None
            or args.frozen_transport_sha256 is None
            or args.broad_grid_result is None
            or args.broad_grid_result_sha256 is None
        ):
            raise SystemExit(
                "sequential requires frozen transport and broad-grid result paths/hashes"
            )
        from bayesfilter.inference.neutra_end_to_end import (
            BroadGridSequentialConfig,
            run_neutra_broad_grid_sequential_cell,
        )
        result = run_neutra_broad_grid_sequential_cell(
            spec=spec,
            config=BroadGridSequentialConfig(
                output_root=args.output_root,
                frozen_transport_path=args.frozen_transport,
                expected_frozen_transport_sha256=args.frozen_transport_sha256,
                broad_grid_result_path=args.broad_grid_result,
                expected_broad_grid_result_sha256=args.broad_grid_result_sha256,
                chunk_results=args.chunk_results,
            ),
        )
    if args.action in ("screen", "cell"):
        result = run_neutra_end_to_end_cell(spec=spec, config=config)
    print({"decision": result.get("decision"), "passed": result.get("passed"),
           "target_signature": spec.target_signature})
    del tf
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
