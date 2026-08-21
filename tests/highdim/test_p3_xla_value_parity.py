"""P3.3 gate as a regression test: eager-vs-XLA value parity.

Fixtures: the I-P2-verified n in {1,2} quadrature configs. Declared gate
(P3, plan revision 4): relative value parity <= 1e-12. The XLA route
(`squared_tt_engine_xla_tf`) solves the same scaled augmented systems by
CholeskyQR2 with an eigvalsh condition estimate; backend equivalence is a
MEASURED gate, not bit identity (see the module docstring and
docs/plans/bayesfilter-p3-xla-port-scoping-note-2026-08-18.md).
"""

from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import numpy as np
import pytest

from bayesfilter.highdim.squared_tt_engine_v0_tf import run_value_filter_branch_axis
from bayesfilter.highdim.squared_tt_engine_xla_tf import run_value_filter_branch_axis_xla
from tests.highdim.test_p2_adjoint_engine_fd import _config, _family


@pytest.mark.parametrize("n,seed", [(1, 61), (2, 62)])
def test_p33_xla_value_parity(n: int, seed: int) -> None:
    adapter, _t, _o, _i, ys = _family(n, np.zeros(n), seed)
    config = _config(n)
    value_eager, _d = run_value_filter_branch_axis(adapter, ys, config)
    value_xla, diags = run_value_filter_branch_axis_xla(adapter, ys, config)
    rel = abs(float(value_xla.numpy()) - float(value_eager.numpy())) / max(
        1.0, abs(float(value_eager.numpy()))
    )
    assert rel <= 1e-12, f"n={n}: eager {float(value_eager.numpy())} xla {float(value_xla.numpy())} rel {rel}"
    assert len(diags) == int(ys.shape[0])
