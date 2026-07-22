#!/usr/bin/env python3
"""Finalize a fresh batch-native LGSSM NeuTra recipe screen."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from bayesfilter.testing.lgssm_neutra_target_specific_protocol_tf import (  # noqa: E402
    BATCH_SIZE,
    FINAL_SEEDS,
    FINAL_STEPS,
    HELDOUT_BATCH_COUNT,
    HELDOUT_BATCH_SIZE,
    HELDOUT_SEEDS,
    SCREEN_SEED,
    SCREEN_RECIPE_ORDER,
    SCREEN_RECIPES,
    SMOKE_SEEDS,
    select_screen_recipe,
)


BINDING_SCHEMA = "bayesfilter.neutra.batch_native_target_binding.v2"
TRAINING_JOB_SCHEMA = "bayesfilter.lgssm_neutra_strict_training_job.v1"
TRUST_BASIS = "owner_designated_managed_session_visible_gpu_trusted"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke-root", required=True)
    parser.add_argument("--screen-root", required=True)
    args = parser.parse_args()
    result = finalize(
        smoke_root=_repo_path(args.smoke_root),
        screen_root=_repo_path(args.screen_root),
    )
    print(
        json.dumps(
            {
                "passed": result["passed"],
                "decision": result["decision"],
                "selected_recipe_id": result["selection"]["selected_recipe_id"],
                "artifact_hash": result["artifact_hash"],
            },
            sort_keys=True,
        )
    )
    return 0


def finalize(*, smoke_root: Path, screen_root: Path) -> Mapping[str, Any]:
    result_path = screen_root / "screen" / "result.json"
    selected_path = screen_root / "selected_recipe.json"
    if result_path.exists() or selected_path.exists():
        raise ValueError("refusing to overwrite fresh screen finalization")

    smoke_entries = tuple(
        _result(smoke_root, "smoke", item) for item in SCREEN_RECIPE_ORDER
    )
    screen_entries = tuple(
        _result(screen_root, "screen", item) for item in SCREEN_RECIPE_ORDER
    )
    smoke_rows = tuple(row for _path, row in smoke_entries)
    screen_rows = tuple(row for _path, row in screen_entries)
    _validate_rows(smoke_entries, expected_kind="smoke", expected_steps=5)
    _validate_rows(screen_entries, expected_kind="screen", expected_steps=500)
    _validate_cross_row_identity((*smoke_entries, *screen_entries))
    selection = select_screen_recipe(screen_rows)
    selected_id = selection["selected_recipe_id"]
    result: dict[str, Any] = {
        "schema": "bayesfilter.neutra.batch_native_screen_result.v1",
        "passed": selected_id is not None,
        "decision": (
            "NOMINATE_ONE_RECIPE_FOR_FRESH_LONG_BUDGET_TRAINING"
            if selected_id is not None
            else "STOP_NO_SURVIVING_SCREEN_RECIPE"
        ),
        "recipe_order": SCREEN_RECIPE_ORDER,
        "finalizer_source": _file_reference(Path(__file__).resolve()),
        "all_predeclared_smokes_passed": True,
        "all_predeclared_screen_arms_processed": True,
        "smoke_rows": tuple(_reference(entry) for entry in smoke_entries),
        "screen_rows": tuple(_reference(entry) for entry in screen_entries),
        "selection": selection,
        "selected_recipe": (
            None if selected_id is None else SCREEN_RECIPES[str(selected_id)].payload()
        ),
        "screen_weights_reused_by_final": False,
        "evidence_role": "proxy_nomination_only_not_transport_promotion",
        "inference_status": {
            "hard_veto_screen": "all_four_candidates_passed",
            "viable_candidates": SCREEN_RECIPE_ORDER,
            "statistically_supported_ranking": False,
            "descriptive_only_differences": (
                "heldout means, paired MCSE values, losses, gradients, and runtimes"
            ),
            "default_readiness": False,
            "next_evidence_needed": (
                "fresh 5000-step training seeds and downstream posterior/HMC validation"
            ),
        },
        "nonclaims": (
            "nomination is not a claim that the selected recipe is best or superior",
            "500-step heldout reverse KL is not posterior correctness evidence",
            "no HMC convergence, robustness, generalization, or default-readiness claim",
        ),
    }
    result = _with_hash(result)
    _write_new(result_path, result)
    if selected_id is not None:
        selected = _with_hash(
            {
                "schema": "bayesfilter.lgssm_neutra_selected_training_recipe.v1",
                "selected_recipe": result["selected_recipe"],
                "selection_result_artifact_hash": result["artifact_hash"],
                "selection_result": _file_reference(result_path),
                "final_steps": FINAL_STEPS,
                "final_seeds": dict(FINAL_SEEDS),
                "screen_weights_reused": False,
                "evidence_role": "proxy_nomination_for_fresh_long_budget_training",
                "nonclaims": result["nonclaims"],
            }
        )
        _write_new(selected_path, selected)
    return result


def _validate_rows(
    entries: tuple[tuple[Path, Mapping[str, Any]], ...],
    *,
    expected_kind: str,
    expected_steps: int,
) -> None:
    for path, row in entries:
        recipe_id = str(row.get("job_id"))
        if recipe_id not in SCREEN_RECIPES:
            raise ValueError(f"unknown recipe identity: {recipe_id}")
        expected_seed = (
            tuple(SMOKE_SEEDS[recipe_id])
            if expected_kind == "smoke"
            else tuple(SCREEN_SEED)
        )
        metadata = row.get("runtime_metadata", {})
        binding = metadata.get("batch_native_target", {})
        records = tuple(row.get("records", ()))
        gpu_manifest = row.get("gpu_manifest", {})
        parity = row.get("frozen_reload_and_score_parity", {})
        closure = row.get("repository_import_closure", {})
        if row.get("schema") != TRAINING_JOB_SCHEMA:
            raise ValueError("fresh recipe row has the wrong training schema")
        if row.get("job_kind") != expected_kind:
            raise ValueError("fresh recipe row has the wrong job kind")
        if row.get("passed") is not True or int(row.get("steps", -1)) != expected_steps:
            raise ValueError("fresh recipe row did not pass its exact step budget")
        if int(row.get("planned_steps", -1)) != expected_steps:
            raise ValueError("fresh recipe row planned-step identity mismatch")
        if tuple(row.get("seed", ())) != expected_seed:
            raise ValueError("fresh recipe row seed identity mismatch")
        if not _same_json(row.get("recipe"), SCREEN_RECIPES[recipe_id].payload()):
            raise ValueError("fresh recipe row recipe identity mismatch")
        if row.get("step_override_debug_only") is not False:
            raise ValueError("fresh recipe row used a debug-only step override")
        _validate_source_artifact(path, row)
        _validate_file_reference(row.get("campaign_contract"), label="campaign contract")
        _require_sha256(
            row.get("campaign_contract", {}).get("contract_hash"),
            "campaign contract hash",
        )
        for label in ("checkpoint", "progress", "payload"):
            _validate_file_reference(row.get(label), label=label)
        if binding.get("schema") != BINDING_SCHEMA:
            raise ValueError("fresh recipe row lacks batch binding v2")
        if binding.get("target_signature") != row.get("target_signature"):
            raise ValueError("fresh recipe row target signature drifted from binding")
        if binding.get("adapter_signature") != row.get("adapter_signature"):
            raise ValueError("fresh recipe row adapter signature drifted from binding")
        _require_sha256(binding.get("dependency_closure_sha256"), "dependency closure")
        if any(
            binding.get(field) is not False
            for field in (
                "scalar_fallback_used",
                "sample_axis_python_loop_used",
                "row_mapped_scalar_target_used",
            )
        ):
            raise ValueError("fresh recipe row binding reports a forbidden fallback")
        if binding.get("jit_compile_required") is not True:
            raise ValueError("fresh recipe row binding does not require XLA")
        if metadata.get("compiled_training_program_invocations") != 1:
            raise ValueError("fresh recipe row lacks one compiled invocation")
        if (
            metadata.get("jit_compile") is not True
            or metadata.get("require_gpu") is not True
            or int(metadata.get("training_batch_size", -1)) != BATCH_SIZE
            or int(metadata.get("program_step_count", -1)) != expected_steps
            or metadata.get("compiled_training_control_flow") != "tf_while_loop"
        ):
            raise ValueError("fresh recipe row lacks the required GPU/XLA batch runtime")
        if any(
            metadata.get(field) is not False
            for field in (
                "scalar_fallback_used",
                "sample_axis_python_loop_used",
                "row_mapped_scalar_target_used",
            )
        ):
            raise ValueError("fresh recipe row runtime reports a forbidden fallback")
        for label in (
            "trainable_variable_devices",
            "adam_moment_devices",
            "compiled_output_devices",
        ):
            _require_gpu_devices(metadata.get(label), label)
        if (
            gpu_manifest.get("jit_compile") is not True
            or gpu_manifest.get("trust_basis") != TRUST_BASIS
            or gpu_manifest.get("training_dtype") != "float64"
        ):
            raise ValueError("fresh recipe row lacks the trusted GPU manifest")
        if not gpu_manifest.get("physical_gpus") or not gpu_manifest.get("logical_gpus"):
            raise ValueError("fresh recipe row lacks visible GPU provenance")
        if parity.get("passed") is not True or parity.get("jit_compile") is not True:
            raise ValueError("fresh recipe row lacks frozen reload/score parity")
        _require_gpu_devices(parity.get("output_devices"), "parity output devices")
        if closure.get("passed") is not True or tuple(closure.get("violations", ())) != ():
            raise ValueError("fresh recipe row failed the NumPy/host-callback closure audit")
        if not records or not all(
            item.get("target_status_all_valid") is True
            and item.get("target_values_finite") is True
            and int(item.get("target_floor_count_total", -1)) == 0
            for item in records
        ):
            raise ValueError("fresh recipe row has an exact-target veto")
        steps = tuple(int(item.get("step", -1)) for item in records)
        if steps != tuple(sorted(set(steps))) or steps[-1] != expected_steps:
            raise ValueError("fresh recipe row has an invalid progress cadence")
        for item in records:
            for field in (
                "loss",
                "raw_gradient_norm",
                "clipped_gradient_norm",
                "learning_rate",
                "mean_log_abs_det_jacobian",
            ):
                if not math.isfinite(float(item.get(field, math.nan))):
                    raise ValueError(f"fresh recipe row has nonfinite {field}")
        _validate_heldout(row, expected_kind=expected_kind)


def _validate_cross_row_identity(
    entries: tuple[tuple[Path, Mapping[str, Any]], ...],
) -> None:
    rows = tuple(row for _path, row in entries)
    for field in ("target_signature", "adapter_signature"):
        values = {str(row.get(field)) for row in rows}
        if len(values) != 1:
            raise ValueError(f"fresh recipe rows disagree on {field}")
        _require_sha256(next(iter(values)), field)
    closures = {
        str(row["runtime_metadata"]["batch_native_target"]["dependency_closure_sha256"])
        for row in rows
    }
    if len(closures) != 1:
        raise ValueError("fresh recipe rows disagree on dependency closure")
    contracts = {
        (
            str(row["campaign_contract"].get("contract_hash")),
            str(row["campaign_contract"].get("file_sha256")),
        )
        for row in rows
    }
    if len(contracts) != 1:
        raise ValueError("fresh recipe rows disagree on campaign contract")


def _validate_heldout(row: Mapping[str, Any], *, expected_kind: str) -> None:
    heldout = row.get("heldout_common_batches")
    if expected_kind == "smoke":
        if heldout is not None:
            raise ValueError("fresh smoke row unexpectedly contains selection evidence")
        return
    if not isinstance(heldout, Mapping):
        raise ValueError("fresh screen row lacks heldout evidence")
    rows = tuple(heldout.get("rows", ()))
    seeds = tuple(tuple(item.get("seed", ())) for item in rows)
    if (
        int(heldout.get("batch_count", -1)) != HELDOUT_BATCH_COUNT
        or int(heldout.get("batch_size", -1)) != HELDOUT_BATCH_SIZE
        or seeds != HELDOUT_SEEDS
        or heldout.get("single_compiled_heldout_invocation") is not True
        or heldout.get("target_status_all_valid") is not True
    ):
        raise ValueError("fresh screen row heldout identity or status mismatch")
    if not math.isfinite(float(heldout.get("mean_reverse_kl_objective", math.nan))):
        raise ValueError("fresh screen row heldout mean is nonfinite")
    if not math.isfinite(float(heldout.get("mcse_across_batches", math.nan))):
        raise ValueError("fresh screen row heldout MCSE is nonfinite")
    if not all(
        item.get("target_status_all_valid") is True
        and math.isfinite(float(item.get("reverse_kl_objective_mean", math.nan)))
        for item in rows
    ):
        raise ValueError("fresh screen row has an invalid heldout batch")
    values = tuple(float(item["reverse_kl_objective_mean"]) for item in rows)
    mean = math.fsum(values) / len(values)
    sample_variance = math.fsum((value - mean) ** 2 for value in values) / (
        len(values) - 1
    )
    mcse = math.sqrt(sample_variance / len(values))
    if not math.isclose(
        float(heldout["mean_reverse_kl_objective"]), mean, rel_tol=0.0, abs_tol=1.0e-12
    ):
        raise ValueError("fresh screen row heldout mean does not match its batches")
    if not math.isclose(
        float(heldout["mcse_across_batches"]), mcse, rel_tol=0.0, abs_tol=1.0e-12
    ):
        raise ValueError("fresh screen row heldout MCSE does not match its batches")


def _result(root: Path, kind: str, recipe: str) -> tuple[Path, Mapping[str, Any]]:
    path = root / kind / "candidates" / recipe / "attempt_1_graph_native" / "result.json"
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise ValueError(f"result is not a mapping: {path}")
    return path, value


def _reference(entry: tuple[Path, Mapping[str, Any]]) -> Mapping[str, Any]:
    path, row = entry
    heldout = row.get("heldout_common_batches")
    return {
        "recipe_id": row["job_id"],
        "passed": row["passed"],
        "steps": row["steps"],
        "result": _file_reference(path),
        "artifact_hash": row["artifact_hash"],
        "heldout_mean": (
            None if not isinstance(heldout, Mapping) else heldout["mean_reverse_kl_objective"]
        ),
        "heldout_mcse": (
            None if not isinstance(heldout, Mapping) else heldout["mcse_across_batches"]
        ),
    }


def _file_reference(path: Path) -> Mapping[str, str]:
    return {
        "path": str(path.relative_to(ROOT)),
        "file_sha256": _file_sha256(path),
    }


def _validate_source_artifact(path: Path, row: Mapping[str, Any]) -> None:
    expected = _stable_artifact_hash(row)
    if row.get("artifact_hash") != expected:
        raise ValueError(f"fresh recipe row artifact hash mismatch: {path}")


def _validate_file_reference(value: Any, *, label: str) -> None:
    if not isinstance(value, Mapping):
        raise ValueError(f"fresh recipe row lacks {label} reference")
    path = ROOT / str(value.get("path", ""))
    if not path.is_file():
        raise ValueError(f"fresh recipe row {label} artifact is missing")
    if value.get("file_sha256") != _file_sha256(path):
        raise ValueError(f"fresh recipe row {label} artifact hash mismatch")


def _require_gpu_devices(value: Any, label: str) -> None:
    devices = tuple(value or ())
    if not devices or not all("GPU" in str(device).upper() for device in devices):
        raise ValueError(f"fresh recipe row {label} are not all GPU")


def _require_sha256(value: Any, label: str) -> None:
    text_value = str(value)
    if text_value.startswith("sha256:"):
        text_value = text_value.split(":", 1)[1]
    if len(text_value) != 64 or any(
        character not in "0123456789abcdef" for character in text_value
    ):
        raise ValueError(f"fresh recipe row {label} is not a SHA-256 digest")


def _same_json(left: Any, right: Any) -> bool:
    return json.dumps(left, sort_keys=True) == json.dumps(right, sort_keys=True)


def _stable_artifact_hash(value: Mapping[str, Any]) -> str:
    payload = {
        key: item
        for key, item in value.items()
        if key not in {"artifact_hash", "artifact_hash_semantics"}
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return f"sha256:{hashlib.sha256(encoded.encode('utf-8')).hexdigest()}"


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _with_hash(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(value)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["artifact_hash"] = f"sha256:{hashlib.sha256(encoded.encode('utf-8')).hexdigest()}"
    return payload


def _write_new(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        raise ValueError(f"refusing to overwrite artifact: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _repo_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


if __name__ == "__main__":
    raise SystemExit(main())
