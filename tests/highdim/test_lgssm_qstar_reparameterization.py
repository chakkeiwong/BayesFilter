from __future__ import annotations

import math

from docs.benchmarks import diagnose_lgssm_qstar_reparameterization as diagnostic


def test_qstar_geometry_and_log_score_chain_rule() -> None:
    geometry = diagnostic.qstar_geometry()
    assert geometry["qstar"] > diagnostic.Q_SCALE
    score = [1.0, -2.0, 0.5, -3.0, 4.0]
    transformed = diagnostic.transform_physical_score(score)
    original_log_q = score[3] * diagnostic.Q_SCALE
    transformed_log_qstar = transformed[3] * diagnostic.hmc_chain_qstar()[3]
    assert math.isclose(
        transformed_log_qstar, original_log_q, rel_tol=1.0e-14, abs_tol=1.0e-14
    )


def test_qstar_physical_score_is_a_relative_bias_preserving_rescale() -> None:
    score = [1.0, -2.0, 0.5, -3.0, 4.0]
    transformed = diagnostic.transform_physical_score(score)
    assert math.isclose(
        transformed[3] / score[3],
        1.0 / diagnostic.qstar_geometry()["sqrt_a"],
        rel_tol=1.0e-14,
        abs_tol=1.0e-14,
    )
