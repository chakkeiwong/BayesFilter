#!/usr/bin/env python3
"""Emit the deterministic Lane-B B3 T2 previous-marginal boundary artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import tensorflow as tf  # noqa: E402

from bayesfilter.highdim.sir_latent_preclip_tf import (  # noqa: E402
    latent_preclip_zhao_cui_sir_austria_model,
)
from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_artifact_compat import (  # noqa: E402
    COMPAT_DECODER_ID,
    load_lane_b_t1_artifact_v1_compat,
)
from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_t2_boundary_tf import (  # noqa: E402
    LaneBT1RetainedBoundary,
    independent_total_mass_from_cut,
)
from bayesfilter.highdim.zhao_cui_austria_sir_lane_b_target_tf import (  # noqa: E402
    generate_t1_proposal_cloud,
    tensor_sha256,
)


PROBE_SEED = 73601
PROBE_COUNT = 64
PLAN = "docs/plans/bayesfilter-zhao-cui-austria-sir-lane-b-t1-execution-note-2026-07-30.md"
T1_RESULT = "docs/plans/bayesfilter-zhao-cui-austria-sir-lane-b-t1-result-2026-07-31.md"


def _jsonable(value: Any) -> Any:
    if isinstance(value, tf.Tensor):
        return value.numpy().item() if value.shape.rank == 0 else value.numpy().tolist()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_result(artifact_dir: Path) -> Mapping[str, Any]:
    started = time.monotonic()
    artifact = load_lane_b_t1_artifact_v1_compat(artifact_dir)
    boundary = LaneBT1RetainedBoundary(artifact)
    cloud = generate_t1_proposal_cloud(
        sample_count=PROBE_COUNT, seed=PROBE_SEED, role="b3_boundary_probe"
    )
    z1 = cloud.joint_points[:, :18]
    model = latent_preclip_zhao_cui_sir_austria_model()
    z2 = model.transition_push_from_standard_normal(
        tf.zeros([3], tf.float64),
        z1,
        tf.random.stateless_normal(
            [PROBE_COUNT, 18], seed=[PROBE_SEED, 2], dtype=tf.float64
        ),
        2,
    )
    terms = boundary.t2_log_target(z2, z1)
    marginal_residual = tf.reduce_max(
        tf.abs(terms["previous_api"] - terms["previous_independent"])
    )
    target_residual = tf.reduce_max(
        tf.abs(terms["api_total"] - terms["independent_total"])
    )
    independent_mass = independent_total_mass_from_cut(artifact)
    direct_mass = artifact.density().normalizer()
    mass_residual = tf.abs(independent_mass - direct_mass)
    passed = bool(
        (
            (marginal_residual <= 2e-12)
            & (target_residual <= 2e-12)
            & (mass_residual <= 2e-12)
            & tf.reduce_all(tf.math.is_finite(terms["api_total"]))
        ).numpy()
    )
    source_paths = (
        "bayesfilter/highdim/zhao_cui_austria_sir_lane_b_t2_boundary_tf.py",
        "bayesfilter/highdim/zhao_cui_austria_sir_lane_b_artifact_compat.py",
        "scripts/run_zhao_cui_austria_sir_lane_b_t2_boundary.py",
    )
    return {
        "schema_version": "bayesfilter.zhao_cui_austria_sir_lane_b_t2_boundary.v1",
        "status": "PASS_T2_PREVIOUS_MARGINAL_BOUNDARY" if passed else "BLOCK_T2_PREVIOUS_MARGINAL_BOUNDARY",
        "boundary": boundary.manifest_payload(),
        "source_artifact_identity": artifact.identity.hash.value,
        "artifact_reload_decoder_id": COMPAT_DECODER_ID,
        "probe": {
            "seed": PROBE_SEED,
            "sample_count": PROBE_COUNT,
            "z1_sha256": tensor_sha256(z1),
            "z2_sha256": tensor_sha256(z2),
            "maximum_log_marginal_residual": marginal_residual,
            "maximum_t2_log_target_residual": target_residual,
            "independent_cut_mass": independent_mass,
            "direct_density_mass": direct_mass,
            "absolute_mass_residual": mass_residual,
            "tolerance": 2e-12,
        },
        "gates": {
            "fresh_t1_reload": True,
            "api_independent_marginal": bool((marginal_residual <= 2e-12).numpy()),
            "independent_cut_normalizer": bool((mass_residual <= 2e-12).numpy()),
            "sealed_t2_component_recomposition": bool((target_residual <= 2e-12).numpy()),
            "finite_t2_target": bool(tf.reduce_all(tf.math.is_finite(terms["api_total"])).numpy()),
            "passed": passed,
        },
        "run_manifest": {
            "git_commit": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "command": sys.argv,
            "environment": sys.prefix,
            "host": platform.node(),
            "python": platform.python_version(),
            "tensorflow": tf.__version__,
            "device_policy": "explicit_cpu_hidden_boundary_reference",
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "jit_compile": False,
            "plan": PLAN,
            "t1_result": T1_RESULT,
            "source_sha256": {path: _sha256(ROOT / path) for path in source_paths},
            "wall_time_seconds": time.monotonic() - started,
        },
        "decision_table": {
            "decision": "open_scope_specific_T2_tuning_plan" if passed else "block_T2_fit_and_repair_boundary",
            "primary_criterion_status": "passed" if passed else "failed",
            "veto_diagnostic_status": "passed" if passed else "failed",
            "main_uncertainty": "T1 filtering-density shape remains a finite TT approximation",
            "next_justified_action": "write skeptical scope-specific T2 tuning plan" if passed else "repair retained marginal",
            "not_concluded": "no T2 fit/value, score, T20, HMC, or production readiness",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"output already exists: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(_jsonable(build_result(args.artifact_dir.resolve())), indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
