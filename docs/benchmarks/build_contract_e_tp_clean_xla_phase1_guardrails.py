#!/usr/bin/env python3
"""Build the CPU-hidden Phase 1 source and graph guardrail artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


if os.environ.get("CUDA_VISIBLE_DEVICES") != "-1":
    raise RuntimeError("Phase 1 is CPU-only; set CUDA_VISIBLE_DEVICES=-1 before Python")

import tensorflow as tf


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bayesfilter.highdim import ledh_contract_e_tp_lgssm_tf as lgssm
from bayesfilter.testing.contract_e_tp_clean_xla_guardrails import (
    LoopRole,
    SourceRouteSpec,
    audit_source_path,
    audit_source_text,
    inventory_graph_def,
)
from scripts.filtering_value_gradient_benchmark_generate_p8_datasets import (
    _lgssm_dataset,
)


DTYPE = tf.float64
THETA = tf.constant([0.72, 0.55, 0.35, 0.35, 0.45], DTYPE)
LGSSM_SOURCE = ROOT / "bayesfilter/highdim/ledh_contract_e_tp_lgssm_tf.py"
SCALAR_SOURCE = ROOT / "bayesfilter/highdim/ledh_contract_e_tp_scalar_sv_tf.py"
PREDATOR_SOURCE = ROOT / "bayesfilter/highdim/ledh_contract_e_tp_predator_prey_tf.py"
MODELS_SOURCE = ROOT / "bayesfilter/highdim/models.py"
PREPARATIONS = {
    10: ROOT / "docs/benchmarks/artifacts/contract_e_tp_all_models_2026_07_15/phase8b_lgssm_t10_order5_lookahead8_attempt1_20260715/charts.json",
    50: ROOT / "docs/benchmarks/artifacts/contract_e_tp_all_models_2026_07_15/phase8b_lgssm_t50_order5_lookahead8_attempt1_20260715/charts.json",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _source_audits() -> dict[str, Any]:
    clean = audit_source_path(
        LGSSM_SOURCE,
        SourceRouteSpec(
            roots=("contract_e_tp_lgssm_finite_lookahead_loop_core",),
            loop_roles={},
            required_reachable=("_finite_lookahead_information_parameters_loop",),
        ),
    )
    historical = audit_source_path(
        LGSSM_SOURCE,
        SourceRouteSpec(
            roots=("contract_e_tp_lgssm_score_informed_recursive_core",),
            loop_roles={
                "contract_e_tp_lgssm_score_informed_recursive_core": LoopRole("filter_time", True),
                "_finite_lookahead_information_parameters": LoopRole("lookahead_windows", True),
            },
            required_reachable=("_finite_lookahead_information_parameters",),
        ),
    )
    scalar = audit_source_path(
        SCALAR_SOURCE,
        SourceRouteSpec(
            roots=("contract_e_tp_scalar_sv_recursive_core",),
            loop_roles={
                "contract_e_tp_scalar_sv_recursive_core": LoopRole("filter_time", True),
                "target_continuation_log_likelihood": LoopRole("backward_continuation", True),
            },
            required_reachable=("target_continuation_log_likelihood",),
        ),
    )
    predator = audit_source_path(
        PREDATOR_SOURCE,
        SourceRouteSpec(
            roots=("contract_e_tp_predator_prey_recursive_core",),
            loop_roles={
                "contract_e_tp_predator_prey_recursive_core": LoopRole("filter_time", True),
                "target_continuation_log_likelihood": LoopRole("backward_continuation", True),
            },
            required_reachable=("target_continuation_log_likelihood",),
        ),
    )
    predator_rk4 = audit_source_path(
        MODELS_SOURCE,
        SourceRouteSpec(
            roots=("PredatorPreySSM.transition_mean",),
            loop_roles={
                "PredatorPreySSM.transition_mean": LoopRole("fixed_substep_rk4_solver", True),
            },
        ),
    )
    neutral_source = """
def z(seq):
    r = range
    out = 0
    for j in r(len(seq)):
        out += seq[j]
    return out
def q(values):
    return z(values)
"""
    neutral = audit_source_text(
        neutral_source,
        SourceRouteSpec(
            roots=("q",),
            loop_roles={"z": LoopRole("declared_dynamic_bound", True)},
            required_reachable=("z",),
        ),
        source_id="synthetic_neutral_alias_fixture",
    )
    fixed = audit_source_text(
        "def f(x):\n    for i in range(3):\n        x += i\n    return x\n",
        SourceRouteSpec(
            roots=("f",),
            loop_roles={"f": LoopRole("fixed_small_dimension", False)},
        ),
        source_id="synthetic_fixed_small_fixture",
    )
    return {
        "lgssm_clean": clean,
        "lgssm_historical_unrolled": historical,
        "scalar_sv_current": scalar,
        "predator_prey_current": predator,
        "predator_prey_rk4": predator_rk4,
        "neutral_alias_hidden_helper": neutral,
        "fixed_small": fixed,
    }


def _graph_inventory(time_steps: int) -> dict[str, Any]:
    preparation_path = PREPARATIONS[time_steps]
    preparation = json.loads(preparation_path.read_text(encoding="utf-8"))
    observations = tf.convert_to_tensor(
        _lgssm_dataset(81100)["observations"][:time_steps], DTYPE
    )
    evaluate = lgssm.make_contract_e_tp_lgssm_score_informed_recursive_tf(
        observations,
        tf.constant(preparation["quadrature"]["nodes"], DTYPE),
        tf.constant(preparation["quadrature"]["weights"], DTYPE),
        tf.constant(preparation["active_indices"], tf.int32),
        tf.constant(preparation["row_scales"], DTYPE),
        feature_mode="finite_lookahead",
        lookahead_steps=8,
        jit_compile=True,
    )
    result = inventory_graph_def(evaluate.get_concrete_function(THETA).graph.as_graph_def())
    result.update(
        {
            "time_steps": time_steps,
            "preparation_path": str(preparation_path.relative_to(ROOT)),
            "preparation_sha256": _sha256(preparation_path),
        }
    )
    return result


def build_payload(phase0_registry: Path) -> dict[str, Any]:
    phase0 = json.loads(phase0_registry.read_text(encoding="utf-8"))
    if phase0["schema"] != "contract_e_tp.clean_xla_phase0_inventory.v1":
        raise ValueError("wrong Phase 0 registry schema")
    audits = _source_audits()
    expected = {
        "lgssm_clean": True,
        "lgssm_historical_unrolled": False,
        "scalar_sv_current": False,
        "predator_prey_current": False,
        "predator_prey_rk4": False,
        "neutral_alias_hidden_helper": False,
        "fixed_small": True,
    }
    source_gates = {
        name: audits[name]["approved"] is expected_value
        for name, expected_value in expected.items()
    }
    graphs = [_graph_inventory(10), _graph_inventory(50)]
    by_time = {item["time_steps"]: item for item in graphs}
    ratios = {
        "top_level_nodes": by_time[50]["top_level_nodes"] / by_time[10]["top_level_nodes"],
        "function_nodes": by_time[50]["function_nodes"] / by_time[10]["function_nodes"],
        "graphdef_bytes": by_time[50]["graphdef_bytes"] / by_time[10]["graphdef_bytes"],
    }
    graph_gates = {
        "exact_top_level_nodes_4014": all(item["top_level_nodes"] == 4014 for item in graphs),
        "exact_function_nodes_3712": all(item["function_nodes"] == 3712 for item in graphs),
        "exact_functional_loops_4": all(item["functional_loop_count"] == 4 for item in graphs),
        "top_level_ratio_at_most_1_10": ratios["top_level_nodes"] <= 1.10,
        "function_ratio_at_most_1_10": ratios["function_nodes"] <= 1.10,
        "graphdef_ratio_at_most_1_25": ratios["graphdef_bytes"] <= 1.25,
    }
    if not all(source_gates.values()) or not all(graph_gates.values()):
        raise RuntimeError(f"Phase 1 guardrail gate failed: {source_gates=} {graph_gates=}")
    return {
        "schema": "contract_e_tp.clean_xla_phase1_guardrails.v1",
        "status": "PASS_SHARED_SOURCE_AND_GRAPH_GUARDRAILS",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "device_policy": "CUDA_VISIBLE_DEVICES=-1 before TensorFlow import",
        "phase0_registry": {"path": str(phase0_registry.relative_to(ROOT)), "sha256": _sha256(phase0_registry)},
        "source_audits": audits,
        "source_gates": source_gates,
        "graph_inventories": graphs,
        "graph_ratios": ratios,
        "graph_gates": graph_gates,
        "nonclaims": [
            "not nonlinear loop-core repair",
            "not nonlinear XLA readiness",
            "not scientific accuracy or equivalence",
            "not canonical, default, HMC, or leaderboard readiness",
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase0-registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    registry = args.phase0_registry if args.phase0_registry.is_absolute() else ROOT / args.phase0_registry
    output = args.output if args.output.is_absolute() else ROOT / args.output
    if output.exists():
        raise FileExistsError(output)
    payload = build_payload(registry)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "source_gates": payload["source_gates"], "graph_gates": payload["graph_gates"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
