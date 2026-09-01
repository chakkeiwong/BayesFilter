from __future__ import annotations

import inspect
import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import tensorflow as tf

from docs.benchmarks import run_ledh_pfpf_genut_sqmc as runner


def test_cell_grid_has_only_distinct_predeclared_semantics() -> None:
    assert runner.ALL_CELLS == (
        ("initial_only", "iid_initial"),
        ("initial_only", "rqmc_initial"),
        ("all_innovations", "iid_existing"),
        ("all_innovations", "hilbert_sort_iid"),
        ("all_innovations", "unordered_rqmc_existing"),
        ("all_innovations", "ranked_joint_rqmc_no_hilbert"),
        ("all_innovations", "ordered_rqmc_hybrid"),
        ("all_innovations", "full_sqmc_halton_mechanics"),
    )
    assert len(set(runner.ALL_CELLS)) == len(runner.ALL_CELLS)


def test_matched_joint_arms_receive_identical_raw_point_sets() -> None:
    spec = runner.diagonal_lgssm_spec()
    arms = (
        "ranked_joint_rqmc_no_hilbert",
        "ordered_rqmc_hybrid",
        "full_sqmc_halton_mechanics",
    )
    inputs = [runner._inputs(spec, arm, "all_innovations", 94301) for arm in arms]  # noqa: SLF001
    for left, right in zip(inputs[:-1], inputs[1:], strict=True):
        tf.debugging.assert_equal(left[0], right[0])
        tf.debugging.assert_equal(left[1], right[1])
        tf.debugging.assert_equal(left[2], right[2])
        assert left[3]["raw_joint_point_sha256"] == right[3]["raw_joint_point_sha256"]


def test_initial_only_changes_only_initial_point_construction() -> None:
    spec = runner.diagonal_lgssm_spec()
    iid = runner._inputs(spec, "iid_initial", "initial_only", 94302)  # noqa: SLF001
    rqmc = runner._inputs(spec, "rqmc_initial", "initial_only", 94302)  # noqa: SLF001
    assert not bool(tf.reduce_all(iid[0] == rqmc[0]).numpy())
    tf.debugging.assert_equal(iid[1], rqmc[1])
    tf.debugging.assert_equal(iid[2], rqmc[2])


def test_runner_is_cpu_xla_standard_score_mechanics_only() -> None:
    source = inspect.getsource(runner)
    assert 'os.environ.get("CUDA_VISIBLE_DEVICES") != "-1"' in source
    assert "jit_compile=True" in source
    assert "repository_standard_pairwise_backward_filtering_score" in source
    assert "experimental_contract_e_chol_primal_not_canonical_identity" in source
    assert "none_not_a_confirmatory_run" in source
    for forbidden in (
        "GradientTape",
        "ForwardAccumulator",
        "finite_difference",
        "manual_score",
        "importance_ratio_w_over_r",
    ):
        assert forbidden not in source

