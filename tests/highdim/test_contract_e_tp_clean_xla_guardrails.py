from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import tensorflow as tf

from bayesfilter.testing.contract_e_tp_clean_xla_guardrails import (
    LoopRole,
    SourceRouteSpec,
    audit_source_path,
    audit_source_text,
    inventory_graph_def,
)


ROOT = Path(__file__).resolve().parents[2]


def test_hidden_neutral_aliased_len_bound_loop_is_rejected_by_role() -> None:
    source = """
def z(seq):
    r = range
    out = 0
    for j in r(len(seq)):
        out += seq[j]
    return out

def q(values):
    return z(values)
"""
    result = audit_source_text(
        source,
        SourceRouteSpec(
            roots=("q",),
            required_reachable=("z",),
            loop_roles={"z": LoopRole("declared_dynamic_bound", dynamic=True)},
        ),
    )
    assert result["approved"] is False
    assert result["reachable_symbols"] == ["q", "z"]
    assert result["forbidden_loop_findings"][0]["expression"] == "r(len(seq))"


def test_fixed_small_loop_requires_explicit_permitted_role() -> None:
    source = """
def f(x):
    for i in range(3):
        x = x + i
    return x
"""
    undeclared = audit_source_text(source, SourceRouteSpec(roots=("f",), loop_roles={}))
    permitted = audit_source_text(
        source,
        SourceRouteSpec(
            roots=("f",),
            loop_roles={"f": LoopRole("fixed_small_dimension", dynamic=False)},
        ),
    )
    assert undeclared["approved"] is False
    assert undeclared["forbidden_loop_findings"][0]["role"] == "undeclared"
    assert permitted["approved"] is True
    assert permitted["loop_findings"][0]["disposition"] == "permit_declared_fixed"


def test_unresolved_same_module_helper_fails_closed() -> None:
    result = audit_source_text(
        "def root(x):\n    return not_registered_here(x)\n",
        SourceRouteSpec(roots=("root",), loop_roles={}),
    )
    assert result["approved"] is False
    assert result["unresolved_local_calls"] == [
        {"caller": "root", "callee": "not_registered_here", "line": 2}
    ]


def test_nested_loop_body_is_in_reachable_closure() -> None:
    source = """
def root(seq):
    def inner(values):
        total = 0
        for item in values:
            total += item
        return total
    return inner(seq)
"""
    result = audit_source_text(
        source,
        SourceRouteSpec(
            roots=("root",),
            loop_roles={"root.inner": LoopRole("nested_dynamic_bound", True)},
            required_reachable=("root.inner",),
        ),
    )
    assert result["approved"] is False
    assert result["reachable_symbols"] == ["root", "root.inner"]
    assert result["forbidden_loop_findings"][0]["symbol"] == "root.inner"


def test_functional_loop_graph_inventory_counts_function_library() -> None:
    @tf.function(input_signature=[tf.TensorSpec([], tf.int32)])
    def evaluate(limit: tf.Tensor) -> tf.Tensor:
        return tf.while_loop(
            lambda i: i < limit,
            lambda i: i + 1,
            (tf.constant(0, tf.int32),),
        )[0]

    inventory = inventory_graph_def(evaluate.get_concrete_function().graph.as_graph_def())
    assert inventory["functional_loop_count"] >= 1
    assert inventory["top_level_nodes"] > 0
    assert inventory["function_count"] >= 2


def test_scalar_sv_loop_factory_source_closure_passes() -> None:
    result = audit_source_path(
        ROOT / "bayesfilter/highdim/ledh_contract_e_tp_scalar_sv_tf.py",
        SourceRouteSpec(
            roots=("make_contract_e_tp_scalar_sv_loop_tf",),
            loop_roles={},
            required_reachable=(
                "contract_e_tp_scalar_sv_loop_core",
                "contract_e_tp_scalar_sv_loop_core.cond",
                "contract_e_tp_scalar_sv_loop_core.body",
                "target_continuation_log_likelihood_loop.cond",
                "target_continuation_log_likelihood_loop.body",
                "make_contract_e_tp_scalar_sv_loop_tf.evaluate.poison",
            ),
        ),
    )
    assert result["approved"] is True
    assert result["loop_findings"] == []


def test_scalar_sv_historical_unrolled_root_is_rejected() -> None:
    result = audit_source_path(
        ROOT / "bayesfilter/highdim/ledh_contract_e_tp_scalar_sv_tf.py",
        SourceRouteSpec(
            roots=("contract_e_tp_scalar_sv_recursive_core",),
            loop_roles={
                "contract_e_tp_scalar_sv_recursive_core": LoopRole(
                    "filter_time_python_unroll", dynamic=True
                ),
                "target_continuation_log_likelihood": LoopRole(
                    "backward_continuation_python_unroll", dynamic=True
                ),
            },
            required_reachable=(
                "contract_e_tp_scalar_sv_recursive_core",
                "target_continuation_log_likelihood",
            ),
        ),
    )
    assert result["approved"] is False
    assert {finding["symbol"] for finding in result["forbidden_loop_findings"]} == {
        "contract_e_tp_scalar_sv_recursive_core",
        "target_continuation_log_likelihood",
    }
