"""Governance conformance gates: C-9 (autodiff ban) and G-1 (lane registry).

These run per-commit on CPU; they are the anti-silent-lane layer.
"""

from __future__ import annotations

import pathlib
import re

REPO = pathlib.Path(__file__).resolve().parents[2]

CANONICAL_MODULES = (
    "bayesfilter/highdim/ledh_canonical_filter_tf.py",
    "bayesfilter/highdim/ledh_ukf_lifecycle_tf.py",
    "bayesfilter/highdim/ledh_flow_perparticle_tf.py",
    "bayesfilter/highdim/ledh_canonical_score_tf.py",
    "bayesfilter/highdim/ledh_canonical_score_stages_tf.py",
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
