"""Inventory ordinary and fixed-transport HMC orchestration entry points."""

from __future__ import annotations

import argparse
import ast
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from bayesfilter.inference.tuning_contract import HMC_TUNING_ROUTE_REGISTRY


SCHEMA = "bayesfilter.hmc_tuning_route_inventory.v1"
EXCLUDED_MODULE_PREFIXES = (
    "neutra_",
    "neural_force_",
)
EXCLUDED_FUNCTIONS = frozenset(
    {
        "run_fixed_transport_hmc_candidate_campaign",
        "run_fixed_transport_full_chain_tfp_hmc",
        "run_full_chain_tfp_hmc",
        "run_native_tfp_fixed_kernel_hmc",
        "run_native_tfp_independent_chains",
        "run_staged_fixed_kernel_hmc_estimation",
    }
)
COMPATIBILITY_ALIASES = frozenset(
    {
        "bayesfilter.inference.hmc_kernel_tuning.tune_hmc_kernel",
    }
)
INTERNAL_STAGE_PREFIXES = (
    "run_hmc_",
    "run_bounded_operational_fixed_trajectory_",
    "run_operational_fixed_trajectory_",
)
RUN_ORCHESTRATION_NAME_MARKERS = (
    "hmc_tuning",
    "fixed_metric_grid_search",
    "operational_broad_grid",
    "generic_hmc_tuning",
)


def discover_routes(inference_root: Path) -> tuple[dict[str, Any], ...]:
    rows: list[dict[str, Any]] = []
    for path in sorted(inference_root.glob("*.py")):
        if path.name.startswith(EXCLUDED_MODULE_PREFIXES):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        module = f"bayesfilter.inference.{path.stem}"
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            name = node.name
            if name.startswith("_") or name in EXCLUDED_FUNCTIONS:
                continue
            qualified_name = f"{module}.{name}"
            if qualified_name in COMPATIBILITY_ALIASES:
                continue
            if name.startswith(INTERNAL_STAGE_PREFIXES):
                continue
            lowered = name.lower()
            is_tuner = name.startswith("tune_") and "hmc" in lowered
            is_runner = name.startswith("run_") and any(
                marker in lowered for marker in RUN_ORCHESTRATION_NAME_MARKERS
            )
            is_orchestrator = name.startswith("orchestrate_") and "hmc_tuning" in lowered
            if not (is_tuner or is_runner or is_orchestrator):
                continue
            rows.append(
                {
                    "interface_name": name,
                    "module": module,
                    "qualified_name": qualified_name,
                    "path": str(path),
                    "line": int(node.lineno),
                }
            )
    return tuple(rows)


def inventory_payload(repo_root: Path) -> dict[str, Any]:
    discovered = discover_routes(repo_root / "bayesfilter" / "inference")
    registry = tuple(record.payload() for record in HMC_TUNING_ROUTE_REGISTRY)
    registry_names = {row["qualified_name"] for row in registry}
    discovered_names = {row["qualified_name"] for row in discovered}
    return {
        "schema": SCHEMA,
        "discovered": discovered,
        "registry": registry,
        "unclassified": tuple(sorted(discovered_names - registry_names)),
        "stale_registry_entries": tuple(sorted(registry_names - discovered_names)),
        "exclusions": {
            "module_prefixes": EXCLUDED_MODULE_PREFIXES,
            "functions": tuple(sorted(EXCLUDED_FUNCTIONS)),
            "compatibility_aliases": tuple(sorted(COMPATIBILITY_ALIASES)),
            "internal_stage_prefixes": INTERNAL_STAGE_PREFIXES,
            "separate_algorithm_families": (
                "NeuTra HMC",
                "neural-force HMC",
                "fixed-transport candidate discovery",
                "full-chain execution mechanics",
            ),
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    payload = inventory_payload(args.repo_root.resolve())
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.check and (payload["unclassified"] or payload["stale_registry_entries"]):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
