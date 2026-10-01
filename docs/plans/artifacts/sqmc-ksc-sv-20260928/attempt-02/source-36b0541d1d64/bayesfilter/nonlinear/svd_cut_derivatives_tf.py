"""Deprecated SVD-CUT derivative import guard.

Raw TensorFlow ``GradientTape`` SVD-CUT derivatives are testing oracles, not
production analytic derivatives.  Import the testing oracle explicitly from
``bayesfilter.testing.tf_svd_cut_autodiff_oracle``.
"""

from __future__ import annotations

from typing import Callable

import tensorflow as tf

from bayesfilter.structural_tf import TFStructuralStateSpace


TFStructuralModelBuilder = Callable[[tf.Tensor], TFStructuralStateSpace]


def tf_svd_cut4_score_hessian(*args, **kwargs):
    """Raise a migration error for the removed production raw-tape path."""

    del args, kwargs
    raise RuntimeError(
        "tf_svd_cut4_score_hessian is no longer a production nonlinear export. "
        "Use bayesfilter.nonlinear.tf_svd_cut4_score for the analytic score, "
        "or bayesfilter.testing.tf_svd_cut_autodiff_oracle."
        "tf_svd_cut4_score_hessian_autodiff_oracle for testing-only parity checks."
    )
