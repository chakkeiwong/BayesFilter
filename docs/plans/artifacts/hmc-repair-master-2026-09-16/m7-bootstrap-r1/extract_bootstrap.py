"""One-time mechanical move of the recorded bootstrap dependency closure."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parents[4]
source_path = REPO / "bayesfilter/inference/hmc_kernel_tuning.py"
source = source_path.read_text()
assert source == (ROOT / "baseline/bayesfilter/inference/hmc_kernel_tuning.py").read_text()
lines = source.splitlines(keepends=True)
tree = ast.parse(source)
nodes = {node.name: node for node in tree.body
         if isinstance(node, (ast.ClassDef, ast.FunctionDef))}
assignments = {node.targets[0].id: node for node in tree.body
               if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name)}
common = """_validate_band _validate_seed _validate_step_repair_multiplier
_string_tuple _mass_artifact_signature _round_seed _scalar_or_none
_seed_from_mapping _bool_or_none _int_or_none _json_ready
_runtime_seconds_or_none _telemetry_payload _finite_number""".split()
bootstrap = """HMCBootstrapScreenConfig HMCBootstrapRepairRound HMCBootstrapScreenResult
_BootstrapFixedMassLatentValueScoreAdapter _build_bootstrap_fixed_mass_adapter
run_hmc_bootstrap_screen _bootstrap_selected_kernel_payload
_bootstrap_acceptance_relation _bootstrap_diagnostics_payload _bootstrap_error_diagnostics
_bootstrap_leapfrog_payload _bootstrap_repair_action _bootstrap_repair_makes_effective_progress
_bootstrap_reusable_static_contract_payload _bootstrap_screen_config
_bootstrap_update_repair_bracket _classify_bootstrap_screen
_public_bootstrap_hard_veto_category _repair_step_size _resolve_bootstrap_target_scope
_validate_bootstrap_repair_bracket""".split()
constants = """BOOTSTRAP_SCREEN_NONCLAIMS RunFullChainFn BootstrapProgressCallback
PrivateTuningDiagnosticCallback _G2_BOOTSTRAP_ROUND_SEED_DERIVATION_SITE_ID
_G2_BOOTSTRAP_ROUND_SEED_GATE_SITE_ID _G2_BOOTSTRAP_ROUND_SEED_INTERFACE_HOPS""".split()


def block(node):
    start = min([node.lineno] + [n.lineno for n in getattr(node, "decorator_list", [])])
    return "".join(lines[start-1:node.end_lineno])


common_header = '''"""Shared host-side validation and diagnostic helpers for HMC preparation."""
from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, Mapping

from bayesfilter.inference.hmc import PrecomputedMassArtifact
from bayesfilter.inference.hmc_artifact_identity import mass_artifact_signature

'''
bootstrap_header = '''"""Fixed-mass bootstrap screening and bounded epsilon repair.

This preparation stage issues no final tuning or posterior authority.
Historical hmc_kernel_tuning imports remain aliases to these definitions.
"""
from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Mapping

from bayesfilter.inference.batched_value_score import (
    _call_mapping_with_batch_rank_bridge,
    _call_value_score_with_batch_rank_bridge,
)
from bayesfilter.inference.hmc import (
    FullChainHMCConfig, FullChainHMCRunResult, PrecomputedMassArtifact,
    build_reusable_full_chain_tfp_hmc_runner, run_full_chain_tfp_hmc,
    program_signature, stable_adapter_signature,
)
from bayesfilter.inference.hmc_diagnostics import screen_hmc_diagnostics
from bayesfilter.inference.hmc_geometry import (
    HMCGeometryInitializationResult, _GEOMETRY_MAX_LEAPFROG,
    _GEOMETRY_MIN_LEAPFROG, _derive_seed, _validate_max_leapfrog_steps,
)
from bayesfilter.inference.hmc_warmup import (
    G2PreboundarySeedUseRegistry, _G2SeedRegistryError,
    _G2_BOOTSTRAP_ROUND_SEED_INTERFACE_HOPS_CONTRACT,
    g2_preboundary_shared_invalidity_exception,
)
from bayesfilter.inference.posterior_adapter import (
    ValueScoreCapability, value_score_capability,
)
from bayesfilter.runtime import stable_config_hash
'''


def import_block(module, names):
    return f"from bayesfilter.inference.{module} import (\n" + "".join(
        f"    {name},\n" for name in names) + ")\n\n"


common_text = common_header + "\n\n".join(block(nodes[name]) for name in common)
bootstrap_text = (bootstrap_header + import_block("hmc_preparation_common", common)
                  + "\n\n".join(block(assignments[name]) for name in constants) + "\n\n"
                  + "\n\n".join(block(nodes[name]) for name in bootstrap))
# Only the physical registry owner moves; seed identity and derivation are fixed.
assert bootstrap_text.count('owner_file="hmc_kernel_tuning.py"') == 1
bootstrap_text = bootstrap_text.replace('owner_file="hmc_kernel_tuning.py"',
                                        'owner_file="hmc_bootstrap.py"')
bootstrap_text += '''

BootstrapFixedMassAdapter = _BootstrapFixedMassLatentValueScoreAdapter
build_bootstrap_fixed_mass_adapter = _build_bootstrap_fixed_mass_adapter

__all__ = [
    "BOOTSTRAP_SCREEN_NONCLAIMS", "BootstrapFixedMassAdapter",
    "HMCBootstrapRepairRound", "HMCBootstrapScreenConfig", "HMCBootstrapScreenResult",
    "build_bootstrap_fixed_mass_adapter", "run_hmc_bootstrap_screen",
]
'''
removed = set()
for node in [nodes[name] for name in common+bootstrap] + [assignments[n] for n in constants]:
    start = min([node.lineno] + [n.lineno for n in getattr(node, "decorator_list", [])])
    removed.update(range(start-1, node.end_lineno))
remaining = "".join(line for index, line in enumerate(lines) if index not in removed)
anchor = "from bayesfilter.inference.hmc_geometry import ("
remaining = remaining.replace(anchor,
    import_block("hmc_preparation_common", common)
    + import_block("hmc_bootstrap", constants+bootstrap) + anchor, 1)

outputs = {"hmc_preparation_common.py": common_text, "hmc_bootstrap.py": bootstrap_text,
           "hmc_kernel_tuning.py": remaining}
for name, content in outputs.items():
    ast.parse(content)
    (source_path.parent / name).write_text(content.rstrip() + "\n")
comparison = []
for name in common+bootstrap:
    module = "hmc_preparation_common.py" if name in common else "hmc_bootstrap.py"
    current = next(n for n in ast.parse(outputs[module]).body if getattr(n, "name", None) == name)
    before = ast.dump(nodes[name], include_attributes=False)
    after = ast.dump(current, include_attributes=False)
    if name == "run_hmc_bootstrap_screen":
        assert before.replace("value='hmc_kernel_tuning.py'", "value='hmc_bootstrap.py'") == after
    else:
        assert before == after, name
    comparison.append({"name": name, "module": module, "unchanged_ast": before == after,
                       "before_sha256": hashlib.sha256(before.encode()).hexdigest(),
                       "after_sha256": hashlib.sha256(after.encode()).hexdigest()})
(ROOT / "extraction.json").write_text(json.dumps({"definitions": comparison,
    "moved_lines": len(removed), "sole_body_change": "bootstrap seed registry physical owner"}, indent=2)+"\n")
print("Moved", len(comparison), "definitions;", sum(r["unchanged_ast"] for r in comparison), "exact ASTs")
