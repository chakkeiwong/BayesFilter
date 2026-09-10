"""Governance conformance gates: C-9 (autodiff ban) and G-1 (lane registry).

These run per-commit on CPU; they are the anti-silent-lane layer.
"""

from __future__ import annotations

import importlib
import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parents[2]

CANONICAL_MODULES = (
    "bayesfilter/highdim/ledh_canonical_filter_tf.py",
    "bayesfilter/highdim/ledh_ukf_lifecycle_tf.py",
    "bayesfilter/highdim/ledh_flow_perparticle_tf.py",
    "bayesfilter/highdim/ledh_canonical_score_tf.py",
    "bayesfilter/highdim/ledh_canonical_score_stages_tf.py",
    "bayesfilter/highdim/ledh_canonical_batch_tf.py",
    "bayesfilter/highdim/ledh_canonical_batch_fused_tf.py",
    "bayesfilter/highdim/ledh_canonical_reset_score_tf.py",
    "bayesfilter/highdim/ledh_unified_reset_tf.py",
    "bayesfilter/highdim/ledh_unified_correction_tf.py",
)

AUTODIFF_PATTERN = re.compile(
    r"ForwardAccumulator|GradientTape|tf\.gradients|tf\.autodiff"
)


def test_c9_no_autodiff_in_canonical_modules():
    offenders = []
    for relative in CANONICAL_MODULES:
        text = (REPO / relative).read_text(encoding="utf-8")
        if AUTODIFF_PATTERN.search(text):
            offenders.append(relative)
    assert not offenders, (
        f"autodiff symbols in claim-bearing canonical modules: {offenders}; "
        "autodiff is permitted only under *_oracle_* namespaces (C-9)"
    )


def test_c9_oracle_is_namespaced():
    oracle = REPO / "bayesfilter/highdim/ledh_canonical_autodiff_oracle_tf.py"
    assert oracle.exists()
    assert "_oracle_" in oracle.name


def test_g1_ledh_lane_discovery():
    """Every module matching LEDH canonical naming must be registered in
    the contract's entry-point/known-module registry or be a test/oracle."""

    from bayesfilter.highdim.ledh_alg1_contract import ENTRY_POINTS

    registered_modules = {e.module.split(".")[-1] + ".py" for e in ENTRY_POINTS}
    known_support = {
        "ledh_alg1_contract.py",
        "ledh_canonical_score_stages_tf.py",
        "ledh_canonical_neutra_targets_tf.py",
        "ledh_canonical_reset_score_tf.py",
        "ledh_ukf_lifecycle_tf.py",
        "ledh_flow_perparticle_tf.py",
    }
    unregistered = []
    for path in (REPO / "bayesfilter/highdim").glob("ledh_canonical*.py"):
        if path.name in registered_modules or path.name in known_support:
            continue
        if "_oracle_" in path.name or "_scaffold_" in path.name:
            continue
        unregistered.append(path.name)
    assert not unregistered, (
        f"unregistered LEDH canonical modules (G-1): {unregistered}"
    )


def test_g1_registered_entry_points_resolve():
    """Every registry row must name an importable callable."""

    from bayesfilter.highdim.ledh_alg1_contract import ENTRY_POINTS

    unresolved = []
    for entry in ENTRY_POINTS:
        module = importlib.import_module(entry.module)
        endpoint = getattr(module, entry.callable_name, None)
        if not callable(endpoint):
            unresolved.append(
                f"{entry.lane}: {entry.module}.{entry.callable_name}"
            )
    assert not unresolved, f"unresolved LEDH entry points (G-1): {unresolved}"


def test_g1_unified_score_stage_ledger_is_closed():
    """Every unified score-stage module is declared in the ownership ledger."""

    from bayesfilter.highdim.ledh_alg1_contract import (
        UNIFIED_SCORE_STAGE_MODULES,
    )

    declared = {module.split(".")[-1] + ".py" for module in UNIFIED_SCORE_STAGE_MODULES}
    discovered = {
        path.name
        for path in (REPO / "bayesfilter/highdim").glob("ledh_unified*_tf.py")
    }
    assert discovered == declared, (
        f"unified score-stage ledger drift: discovered={sorted(discovered)}, "
        f"declared={sorted(declared)}"
    )
    for module_name in UNIFIED_SCORE_STAGE_MODULES:
        importlib.import_module(module_name)


def test_g1_score_endpoints_route_shared_reduction_stages():
    """Single and fused endpoints must reach the same reset/correction owners."""

    single = (
        REPO / "bayesfilter/highdim/ledh_canonical_score_tf.py"
    ).read_text(encoding="utf-8")
    fused = (
        REPO / "bayesfilter/highdim/ledh_canonical_batch_fused_tf.py"
    ).read_text(encoding="utf-8")
    reset_adapter = (
        REPO / "bayesfilter/highdim/ledh_canonical_reset_score_tf.py"
    ).read_text(encoding="utf-8")

    assert "ledh_canonical_reset_score_tf" in single
    assert "ledh_unified_reset_tf" in reset_adapter
    assert "ledh_unified_correction_tf" in single
    assert "ledh_canonical_score_tf" in fused
    assert "canonical_value_and_analytical_score" in fused


def test_batch_claim_paths_ban_python_fanout_and_pfor_apis():
    """Batch endpoints use TensorFlow control flow, never Python row/K fanout."""

    forbidden = re.compile(r"tf\.vectorized_map|tensorflow\.vectorized_map|\bpfor\s*\(")
    offenders = []
    for relative in (
        "bayesfilter/highdim/ledh_canonical_batch_tf.py",
        "bayesfilter/highdim/ledh_canonical_batch_fused_tf.py",
    ):
        if forbidden.search((REPO / relative).read_text(encoding="utf-8")):
            offenders.append(relative)
    assert not offenders, f"pfor/vectorized-map APIs in batch claim paths: {offenders}"


def test_production_program_registry_wiring():
    """Owner directive 2026-08-26: the trust region is a REQUIRED
    mechanism of the production program (covariance-explosion control),
    and 'production' must be machine-checkable. This gate pins:
    (1) the registry requires dual_cap + trust_region ON;
    (2) `_restore_cloud_primal`'s dual-cap family defaults equal the
        registry's owner-family constants (the filter relies on them);
    (3) the Q3 leaderboard runner builds its production configs FROM
        the registry (no silent local copies)."""

    import inspect
    import sys as _sys

    from bayesfilter.highdim.ledh_alg1_contract import (
        LEDH_PRODUCTION_PROGRAM_V1 as REG,
    )
    from bayesfilter.highdim.genut_guided_proposal_tf import (
        _restore_cloud_primal,
    )

    assert REG["filter"]["dual_cap_enabled"] is True
    assert REG["filter"]["trust_region_enabled"] is True
    assert REG["score"]["correction_steps"] > 0

    signature = inspect.signature(_restore_cloud_primal)
    for key, value in REG["dual_cap_family"].items():
        default = signature.parameters[key].default
        assert default == value, (
            f"_restore_cloud_primal default {key}={default} != "
            f"registry {value} — family constants drifted"
        )

    _sys.path.insert(0, "docs/benchmarks")
    import run_q3_leaderboard_20260824 as runner

    source = inspect.getsource(runner)
    assert "LEDH_PRODUCTION_PROGRAM_V1" in source, (
        "leaderboard runner does not import the production-program "
        "registry — production configs must come from the registry"
    )
    score_kwargs = runner.production_score_kwargs(2)
    for key, value in REG["score"].items():
        assert score_kwargs.get(key) == value, (
            f"runner score config {key}={score_kwargs.get(key)} != "
            f"registry {value}"
        )
    assert "reset_design" in score_kwargs
    for key, value in REG["filter"].items():
        assert runner.PRODUCTION_FILTER_KWARGS.get(key) == value, (
            f"runner filter config {key} != registry"
        )
