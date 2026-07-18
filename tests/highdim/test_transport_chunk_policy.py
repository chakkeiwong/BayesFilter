from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from bayesfilter.highdim.transport_chunk_policy import (
    TRANSPORT_CHUNK_POLICY_ID,
    select_transport_chunks,
    validate_transport_chunks,
)


ROOT = Path(__file__).resolve().parents[2]
ARCHIVAL_WRONG_LEDGER = ROOT / "docs/plans" / (
    "bayesfilter-dpf-transport-chunk-policy-"
    "archival-wrong-route-ledger-2026-07-18.json"
)
ACTIVE_CONTRACT_E_DRIVERS = (
    "docs/benchmarks/emit_contract_e_canonical_lgssm_phase8_target_prefix_smoke.py",
    "docs/benchmarks/run_canonical_lgssm_fused_balance_selection.py",
    "docs/benchmarks/run_canonical_lgssm_fused_ot_loop_repair.py",
    "docs/benchmarks/run_canonical_lgssm_balance_schedule_selection.py",
    "docs/benchmarks/run_canonical_lgssm_kalman_certification_arm.py",
    "docs/benchmarks/run_canonical_lgssm_particle_diagnostic_arm.py",
    "docs/benchmarks/run_canonical_lgssm_same_scalar_fd.py",
    "docs/benchmarks/run_contract_e_phase8_lower_rung_node.py",
    "docs/benchmarks/run_contract_e_phase8_paired_audit16.py",
    "docs/benchmarks/run_contract_e_phase8_reset_bias_n_scaling.py",
    "docs/benchmarks/run_latent_preclip_sir_contract_e.py",
)
ARCHIVAL_WRONG_DRIVERS = (
    "docs/benchmarks/diagnose_contract_e_phase8_common_path_identity.py",
    "docs/benchmarks/emit_contract_e_canonical_lgssm_phase5_exact_derivative.py",
    "docs/benchmarks/emit_contract_e_canonical_lgssm_phase8_fd_reclassification.py",
    "docs/benchmarks/emit_contract_e_canonical_lgssm_phase8_rung0b.py",
    "docs/benchmarks/run_contract_e_canonical_lgssm_phase5_certificate.py",
)
ARCHIVAL_WRONG_PREPOLICY_DRIVERS = (
    "docs/benchmarks/benchmark_contract_e_streaming_phase4_gpu_preflight.py",
    "docs/benchmarks/diagnose_contract_e_phase8_common_path_identity.py",
    "docs/benchmarks/diagnose_ledh_pfpf_ot_contract_e_lgssm_value.py",
    "docs/benchmarks/diagnose_ledh_pfpf_ot_lgssm_reset_variants.py",
    "docs/benchmarks/emit_contract_e_canonical_lgssm_phase5_exact_derivative.py",
    "docs/benchmarks/emit_contract_e_canonical_lgssm_phase8_fd_reclassification.py",
    "docs/benchmarks/emit_contract_e_canonical_lgssm_phase8_rung0b.py",
    "docs/benchmarks/emit_contract_e_phase6_raw_diagnostic_baseline.py",
    "docs/benchmarks/run_contract_e_canonical_lgssm_phase5_certificate.py",
    "docs/benchmarks/run_contract_e_phase8_lower_rung_ladder.py",
)


@pytest.mark.parametrize(
    ("num_particles", "expected_chunk", "expected_blocks"),
    (
        (1000, 1000, 1),
        (1024, 1024, 1),
        (10000, 2500, 4),
        (10240, 2560, 4),
    ),
)
def test_required_policy_witnesses(
    num_particles: int, expected_chunk: int, expected_blocks: int
) -> None:
    selection = select_transport_chunks(num_particles)
    assert selection.policy_id == TRANSPORT_CHUNK_POLICY_ID
    assert selection.row_chunk_size == expected_chunk
    assert selection.col_chunk_size == expected_chunk
    assert selection.row_blocks == expected_blocks
    assert selection.col_blocks == expected_blocks


def test_large_prime_fails_closed_instead_of_using_tiny_fallback() -> None:
    with pytest.raises(ValueError, match="refusing a tiny fallback"):
        select_transport_chunks(3001)


@pytest.mark.parametrize(
    ("row_chunk_size", "col_chunk_size", "match"),
    (
        (16, 16, "wrong under"),
        (512, 256, "equal row and column"),
        (333, 333, "divide N exactly"),
        (0, 0, "must be positive"),
    ),
)
def test_validator_rejects_every_nonpolicy_setting(
    row_chunk_size: int, col_chunk_size: int, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        validate_transport_chunks(
            1024 if row_chunk_size == 16 else 1000,
            row_chunk_size=row_chunk_size,
            col_chunk_size=col_chunk_size,
        )


def _imports_policy(tree: ast.AST) -> bool:
    return any(
        isinstance(node, ast.ImportFrom)
        and node.module == "bayesfilter.highdim.transport_chunk_policy"
        for node in ast.walk(tree)
    )


def _has_chunk_cli(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.value.startswith("--")
            and node.value.endswith(("row-chunk-size", "col-chunk-size"))
        ):
            return True
    return False


def _has_numeric_chunk_wiring(tree: ast.AST) -> bool:
    names = {"ROW_CHUNK_SIZE", "COL_CHUNK_SIZE", "row_chunk_size", "col_chunk_size"}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant):
            if not isinstance(node.value.value, (int, float)):
                continue
            if any(isinstance(target, ast.Name) and target.id in names for target in node.targets):
                return True
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            if node.target.id in names and isinstance(node.value, ast.Constant):
                if isinstance(node.value.value, (int, float)):
                    return True
        if isinstance(node, ast.keyword) and node.arg in names:
            if isinstance(node.value, ast.Constant) and isinstance(
                node.value.value, (int, float)
            ):
                return True
    return False


def test_active_contract_e_drivers_use_central_policy_without_chunk_cli() -> None:
    for relative_path in ACTIVE_CONTRACT_E_DRIVERS:
        path = ROOT / relative_path
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative_path)
        assert _imports_policy(tree), relative_path
        assert not _has_chunk_cli(tree), relative_path


def test_contract_e_driver_inventory_is_closed_and_archival_routes_are_explicit() -> None:
    discovered = set()
    tokens = (
        "make_canonical_value_and_score_tf(",
        "prepare_contract_e_lgssm_inputs(",
        "canonical_value_and_score_core(",
        "_canonical_primal_core(",
        "_canonical_manual_jvp_core(",
    )
    for path in (ROOT / "docs/benchmarks").glob("*.py"):
        source = path.read_text(encoding="utf-8")
        if any(token in source for token in tokens):
            discovered.add(str(path.relative_to(ROOT)))
    orchestration_only = {
        "docs/benchmarks/run_canonical_lgssm_fused_ot_loop_campaign.py",
        "docs/benchmarks/run_contract_e_phase8_reset_bias_n_scaling.py",
        "docs/benchmarks/run_latent_preclip_sir_contract_e.py",
    }
    expected = (
        set(ACTIVE_CONTRACT_E_DRIVERS) - orchestration_only
    ) | set(ARCHIVAL_WRONG_DRIVERS)
    assert discovered == expected
    for relative_path in ARCHIVAL_WRONG_DRIVERS:
        assert not _imports_policy(
            ast.parse((ROOT / relative_path).read_text(encoding="utf-8"))
        ), relative_path


def test_shared_streaming_boundaries_import_policy_guard() -> None:
    for relative_path in (
        "experiments/dpf_implementation/tf_tfp/resampling/annealed_transport_tf.py",
        "experiments/dpf_implementation/tf_tfp/filters/experimental_batched_ledh_pfpf_ot_tf.py",
        "experiments/dpf_implementation/tf_tfp/filters/experimental_batched_ledh_pfpf_ot_streaming_tf.py",
        "bayesfilter/highdim/ledh_contract_e_canonical_lgssm_tf.py",
        "bayesfilter/highdim/ledh_contract_e_latent_sir_tf.py",
        "bayesfilter/highdim/ledh_contract_e_lgssm_preparation_tf.py",
        "bayesfilter/highdim/ledh_contract_e_streaming_tf.py",
    ):
        tree = ast.parse(
            (ROOT / relative_path).read_text(encoding="utf-8"),
            filename=relative_path,
        )
        assert _imports_policy(tree), relative_path


def test_independent_chunk_cli_inventory_is_closed_and_archival_wrong() -> None:
    payload = json.loads(ARCHIVAL_WRONG_LEDGER.read_text(encoding="utf-8"))
    assert payload["policy_id"] == TRANSPORT_CHUNK_POLICY_ID
    assert payload["status"] == "historical_wrong_not_diagnostic_not_executable_evidence"
    ledger = set(payload["archival_wrong_independent_chunk_cli_paths"])
    discovered = set()
    for path in (ROOT / "docs/benchmarks").glob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        if _has_chunk_cli(tree):
            discovered.add(str(path.relative_to(ROOT)))
    assert discovered == ledger
    assert not (ledger & set(ACTIVE_CONTRACT_E_DRIVERS))


def _main_is_fail_closed(tree: ast.AST) -> bool:
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        if node.name != "main" or not node.body:
            continue
        first = node.body[0]
        return (
            isinstance(first, ast.Raise)
            and isinstance(first.exc, ast.Call)
            and isinstance(first.exc.func, ast.Name)
            and first.exc.func.id == "RuntimeError"
            and bool(first.exc.args)
            and isinstance(first.exc.args[0], ast.Constant)
            and "ARCHIVAL_WRONG_TRANSPORT_CHUNK_POLICY"
            in str(first.exc.args[0].value)
        )
    return False


def test_every_archival_wrong_route_is_executable_fail_closed() -> None:
    payload = json.loads(ARCHIVAL_WRONG_LEDGER.read_text(encoding="utf-8"))
    ledger_paths = set(payload["archival_wrong_independent_chunk_cli_paths"])
    prepolicy_paths = set(payload["archival_wrong_prepolicy_contract_e_paths"])
    assert prepolicy_paths == set(ARCHIVAL_WRONG_PREPOLICY_DRIVERS)
    for relative_path in sorted(ledger_paths | prepolicy_paths):
        tree = ast.parse(
            (ROOT / relative_path).read_text(encoding="utf-8"),
            filename=relative_path,
        )
        assert _main_is_fail_closed(tree), relative_path


def test_every_numeric_chunk_wiring_route_is_archival_or_centrally_governed() -> None:
    payload = json.loads(ARCHIVAL_WRONG_LEDGER.read_text(encoding="utf-8"))
    archival = set(payload["archival_wrong_independent_chunk_cli_paths"]) | set(
        payload["archival_wrong_prepolicy_contract_e_paths"]
    )
    ungoverned = []
    for path in (ROOT / "docs/benchmarks").glob("*.py"):
        relative_path = str(path.relative_to(ROOT))
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=relative_path)
        if not _has_numeric_chunk_wiring(tree):
            continue
        if relative_path in archival or _imports_policy(tree):
            continue
        ungoverned.append(relative_path)
    assert ungoverned == []


def test_shared_streaming_defaults_are_policy_derived_not_fixed_integers() -> None:
    for relative_path in (
        "experiments/dpf_implementation/tf_tfp/resampling/annealed_transport_tf.py",
        "experiments/dpf_implementation/tf_tfp/filters/experimental_batched_ledh_pfpf_ot_tf.py",
        "experiments/dpf_implementation/tf_tfp/filters/experimental_batched_ledh_pfpf_ot_streaming_tf.py",
    ):
        source = (ROOT / relative_path).read_text(encoding="utf-8")
        assert "row_chunk_size: int = DEFAULT_STREAMING_CHUNK_SIZE" not in source
        assert "col_chunk_size: int = DEFAULT_STREAMING_CHUNK_SIZE" not in source
