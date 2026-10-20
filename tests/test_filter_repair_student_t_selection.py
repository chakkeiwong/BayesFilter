"""Independent scalar authority for the existing Student-t bisection rule."""

import pytest
import tensorflow as tf

from bayesfilter.highdim.squared_tt_engine_gaussian_tf import (
    _student_t_nu_criterion_tf,
    student_t_margin,
    student_t_nu_criterion,
)


@pytest.mark.parametrize("alpha,cap", [(0.8, 2.0), (0.4, 12.0), (0.1, 1000.0)])
def test_compiled_bisection_matches_scalar_reference(alpha, cap):
    lo, hi = 1.5, 500.0
    if student_t_margin(hi, alpha) <= cap:
        expected = hi
    else:
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            if student_t_margin(mid, alpha) <= cap:
                lo = mid
            else:
                hi = mid
        expected = lo
    assert student_t_nu_criterion(alpha, cap) == pytest.approx(expected, abs=1e-10, rel=1e-10)
    assert student_t_margin(student_t_nu_criterion(alpha, cap), alpha) <= cap + 1e-10
    concrete = _student_t_nu_criterion_tf.get_concrete_function()
    assert concrete.function_def.attr["_XlaMustCompile"].b
    assert any(node.op in ("While", "StatelessWhile") for node in concrete.graph.as_graph_def().node)


def test_infeasible_margin_and_invalid_alpha_fail_closed():
    with pytest.raises(ValueError, match="no admissible nu"):
        student_t_nu_criterion(0.8, -100.0)
    with pytest.raises(ValueError, match="alpha"):
        student_t_nu_criterion(1.0, 2.0)
    assert _student_t_nu_criterion_tf.input_signature == (
        tf.TensorSpec([], tf.float64), tf.TensorSpec([], tf.float64),
    )
