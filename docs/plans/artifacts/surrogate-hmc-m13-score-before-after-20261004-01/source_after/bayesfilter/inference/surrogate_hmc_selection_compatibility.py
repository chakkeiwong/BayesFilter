"""Revalidate the saved M13 control choice after sequential-verifier changes.

This preserves the original evidence and its source identity. It permits the redundant fallback-health check and reviewed nomination guards
that the saved control probes cannot call; force and selection stay unchanged.
"""
from __future__ import annotations

import ast
import copy
import json
from pathlib import Path

from . import surrogate_hmc_materiality_selection as selection
from .surrogate_hmc_program import ROOT, file_hash

VERIFIER = "bayesfilter/inference/hmc_tensorflow_verified.py"
REVIEWED_CHANGE = (
    "c64b76994a1fe572129ddb370c08b131c1d6344a79f4639c2ad93dc5845b2ec0",
    "46687b1a00bce5c5c515bbc73c0b59dd8a3831748bf5d077b194917c04754f4c",
)


# October 1 resume changes orchestration/counts only; numerical AST audit is saved.
REVIEWED_RESUME_VERIFIER = "73f745b33394339ddf2b558509bdc02f5f90c13b80f20db7dbb9537603084769"

NATIVE = "bayesfilter/inference/hmc_tensorflow_tuning.py"
REVIEWED_NATIVE_CHANGE = (
    "8eaa6ab82acdb615d4459f438d345653a1387d4557c61ee6bdebe1818c0e56dd",
    "d9ee3cd8a05f39ec2a3e37852428c7db84f3672f6b5e97afba67c30b0aa42956",
)
NOMINATION_ONLY_DEFINITIONS = {
    "TensorFlowHMCKernelTuningConfig", "_candidate_verification",
    "_build_tuning_graph", "_write_tuning_artifact",
    "_bounded_fixed_step", "_guard_candidate_energy",
}


def check_native_control_compatibility(old_source, current_source):
    """Control probes call affine transforms, the step loop and tensor writer.

    None calls nomination/config/artifact issuance. Require every other
    definition, import and module statement to be identical, not merely the
    existence of an unchanged loop. Actual control-consumer parity is tested.
    """
    trees = [ast.parse(source) for source in (old_source, current_source)]
    for tree in trees:
        tree.body = [node for node in tree.body
            if not (isinstance(node, (ast.FunctionDef, ast.ClassDef))
                    and node.name in NOMINATION_ONLY_DEFINITIONS)]
    if ast.dump(trees[0]) != ast.dump(trees[1]):
        raise ValueError("materiality native control execution changed")


def check_health_compatibility(old_source, current_source):
    """Prove the only reachable verifier change adds the existing guard."""
    trees = [ast.parse(source) for source in (old_source, current_source)]
    functions = []
    for tree in trees:
        matches = [node for node in tree.body
                   if isinstance(node, ast.FunctionDef) and node.name == "_health"]
        imports = [node for node in tree.body if isinstance(node, ast.Import)
                   and any(alias.name == "tensorflow" and alias.asname == "tf"
                           for alias in node.names)]
        if len(matches) != 1 or len(imports) != 1:
            raise ValueError("unrecognized health function or TensorFlow import")
        functions.append(copy.deepcopy(matches[0]))
    current = functions[1]
    returned = current.body[-1]
    if not (isinstance(returned, ast.Return) and isinstance(returned.value, ast.Call)
            and isinstance(returned.value.func, ast.Name) and returned.value.func.id == "dict"):
        raise ValueError("unrecognized health return")
    added = [key for key in returned.value.keywords if key.arg == "no_force_fallback"]
    expected = ast.parse("~tf.reduce_any(chunk.force_fallback)", mode="eval").body
    if len(added) != 1 or ast.dump(added[0].value) != ast.dump(expected):
        raise ValueError("fallback check changed")
    returned.value.keywords.remove(added[0])
    if ast.dump(functions[0]) != ast.dump(current):
        raise ValueError("materiality health semantics changed")


def load_compatible_selection(path, *, expected_scope, expected_source_hashes):
    """Keep strict validation with two narrowly reviewed unreachable/redundant changes."""
    path = Path(path)
    value = json.loads(path.read_text())
    saved = value.get("source_hashes", {})
    if saved == expected_source_hashes:
        return selection.load_selection(path, expected_scope=expected_scope,
                                        expected_source_hashes=expected_source_hashes)
    changed = [name for name in saved if saved[name] != expected_source_hashes.get(name)]
    if (set(saved) != set(expected_source_hashes)
            or set(changed) not in ({VERIFIER}, {VERIFIER, NATIVE})):
        raise ValueError("unreviewed materiality source change")
    if (saved[VERIFIER], expected_source_hashes[VERIFIER]) not in (
            REVIEWED_CHANGE, (REVIEWED_CHANGE[0], REVIEWED_RESUME_VERIFIER)):
        raise ValueError("unreviewed verifier revision")
    historical = path.parents[1] / "source" / VERIFIER
    if file_hash(historical) != saved[VERIFIER]:
        raise ValueError("historical verifier snapshot mismatch")
    native_compatibility = None
    if NATIVE in changed:
        if (saved[NATIVE], expected_source_hashes[NATIVE]) != REVIEWED_NATIVE_CHANGE:
            raise ValueError("unreviewed native nomination revision")
        native_snapshot = path.parents[1] / "source" / NATIVE
        if file_hash(native_snapshot) != saved[NATIVE]:
            raise ValueError("historical native snapshot mismatch")
        check_native_control_compatibility(native_snapshot.read_text(),
                                           (ROOT / NATIVE).read_text())
        native_compatibility = dict(original_sha256=saved[NATIVE],
            current_sha256=expected_source_hashes[NATIVE],
            historical_source=str(native_snapshot),
            control_execution_unchanged=True,
            reason="nomination guards and summaries are unreachable from fixed control probes")
    for name, digest in expected_source_hashes.items():
        if file_hash(ROOT / name) != digest:
            raise ValueError("current materiality source mismatch: " + name)
    check_health_compatibility(historical.read_text(), (ROOT / VERIFIER).read_text())
    multipliers, partitions, _ = selection.protocol(expected_scope)
    if (value.get("schema") != selection.SCHEMA or value.get("scope") != expected_scope
            or value.get("issuer_sha256") != file_hash(Path(selection.__file__))
            or value.get("exact_score_admission") is not False
            or value.get("posterior_qualified") is not False):
        raise ValueError("wrong historical materiality selection")
    for group in ("evidence", "data"):
        if set(value[group]) != set(partitions):
            raise ValueError("missing selection partition")
        for row in value[group].values():
            if file_hash(Path(row["path"])) != row["sha256"]:
                raise ValueError("selection evidence checksum mismatch")
    index, decisions = selection._evidence(expected_scope, saved,
        {p: row["path"] for p, row in value["evidence"].items()},
        {p: row["path"] for p, row in value["data"].items()})
    if (index is None or index != value.get("selected_index")
            or value.get("multiplier") != multipliers[index]
            or value.get("controls") != selection.protection_controls(multipliers[index])
            or value.get("decisions") != decisions):
        raise ValueError("selection differs from measured decision")
    return dict(value, source_compatibility=dict(
        changed_dependency=VERIFIER, changed_dependencies=sorted(changed),
        native_nomination_compatibility=native_compatibility,
        original_sha256=saved[VERIFIER],
        current_sha256=expected_source_hashes[VERIFIER],
        historical_source=str(historical), original_selection_sha256=file_hash(path),
        reason="control caller already ANDs the identical force-fallback predicate; nomination and resume orchestration changes are unreachable from its fixed kernel",
        selected_controls_unchanged=True, decisions_recomputed=True,
        original_evidence_preserved=True, posterior_qualified=False))
