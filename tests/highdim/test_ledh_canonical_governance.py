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
