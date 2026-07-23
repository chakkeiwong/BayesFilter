"""Shared rank-normalized HMC convergence diagnostics.

This module is the compatibility surface used by the shared HMC runners.  The
actual rank normalization and ESS calculations are delegated to
``hmc_posterior_diagnostics`` so that fixed-transport tuning and the older
Phase-29 diagnostic path cannot silently diverge.

The reports are finite-sample operational screens.  They are not proofs of
stationarity, posterior correctness, or convergence.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


RANK_NORMALIZED_SPLIT_RHAT_DEFINITION = (
    "max(rank-normalized split R-hat, folded rank-normalized split R-hat)"
)


@dataclass(frozen=True)
class RankNormalizedHMCThresholds:
    """Thresholds for the deterministic HMC diagnostic screen."""

    rhat_max: float = 1.01
    bulk_ess_min: float = 100.0
    tail_ess_min: float = 100.0

    def __post_init__(self) -> None:
        rhat = float(self.rhat_max)
        if not math.isfinite(rhat) or rhat <= 1.0:
            raise ValueError("rhat_max must be finite and greater than one")
        for name in ("bulk_ess_min", "tail_ess_min"):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be positive and finite")
            object.__setattr__(self, name, value)
        object.__setattr__(self, "rhat_max", rhat)


def _draw_chain_parameter_tensor(samples: Any, *, allow_odd: bool = False) -> Any:
    """Validate and return samples with shape ``[draw, chain, parameter]``."""

    import tensorflow as tf

    tensor = tf.cast(tf.convert_to_tensor(samples), tf.float64)
    if tensor.shape.rank != 3 or any(dim is None for dim in tensor.shape):
        raise ValueError("samples must have fully static shape [draw, chain, parameter]")
    draws, chains, parameters = (int(dim) for dim in tensor.shape)
    if draws < 4 or (draws % 2 and not allow_odd):
        raise ValueError("samples require an even draw count of at least four")
    if chains < 2:
        raise ValueError("samples require at least two chains")
    if parameters < 1:
        raise ValueError("samples require at least one parameter")
    return tensor


def _python_vector(value: Any) -> list[float]:
    if hasattr(value, "numpy"):
        value = value.numpy()
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, (list, tuple)):
        return [float(item) for item in value]
    return [float(value)]


def _modern_arrays(samples: Any) -> tuple[Any, Any, Any, Any]:
    """Return bulk/folded R-hat and bulk/tail ESS tensors."""

    import tensorflow as tf

    from bayesfilter.inference.hmc_posterior_diagnostics import (
        rank_normalized_bulk_tail_ess,
        rank_normalized_split_rhat,
    )

    tensor = _draw_chain_parameter_tensor(samples)
    chain_major = tf.transpose(tensor, perm=(1, 0, 2))
    rhat = rank_normalized_split_rhat(chain_major)
    ess = rank_normalized_bulk_tail_ess(chain_major)
    return rhat["bulk"], rhat["folded"], ess["bulk"], ess


def rank_normalized_split_rhat_summary(
    samples: Any,
    *,
    rhat_max: float = 1.01,
) -> Mapping[str, Any]:
    """Return the compact summary used by fixed-kernel verification.

    ``samples`` must have shape ``[draw, chain, parameter]``.
    """

    import tensorflow as tf

    threshold = RankNormalizedHMCThresholds(rhat_max=rhat_max)
    tensor = _draw_chain_parameter_tensor(samples, allow_odd=True)
    # Retained-sample verification can end on an odd number of valid draws
    # after filtering a chunk mask.  Split R-hat is defined on paired halves;
    # discard only that unmatched final draw for this diagnostic summary.
    if int(tensor.shape[0]) % 2:
        tensor = tensor[:-1]
    bulk, folded, _bulk_ess, _ess = _modern_arrays(tensor)
    values = tf.maximum(bulk, folded)
    finite = tf.math.is_finite(values)
    finite_values = tf.boolean_mask(values, finite)
    finite_count = int(tf.size(finite_values).numpy())
    total_count = int(tf.size(values).numpy())
    max_finite = (
        None
        if finite_count == 0
        else float(tf.reduce_max(finite_values).numpy())
    )
    return {
        "passed": bool(
            finite_count == total_count
            and bool(tf.reduce_all(values <= threshold.rhat_max).numpy())
        ),
        "rhat_definition": RANK_NORMALIZED_SPLIT_RHAT_DEFINITION,
        "max_rank_normalized_split_rhat": (
            None if int(tf.size(bulk).numpy()) == 0 else float(tf.reduce_max(bulk).numpy())
        ),
        "max_folded_rank_normalized_split_rhat": (
            None
            if int(tf.size(folded).numpy()) == 0
            else float(tf.reduce_max(folded).numpy())
        ),
        "max_finite_rhat": max_finite,
        "finite_rhat_count": finite_count,
        "nonfinite_rhat_count": total_count - finite_count,
    }


def rank_normalized_hmc_diagnostics(
    samples: Any,
    *,
    parameter_names: Sequence[str] | None = None,
    thresholds: RankNormalizedHMCThresholds | None = None,
) -> Mapping[str, Any]:
    """Compute the full per-parameter rank-normalized HMC screen.

    ``samples`` has shape ``[draw, chain, parameter]``.  The returned mapping
    is JSON-safe and intentionally contains explicit nonclaims.
    """

    import tensorflow as tf

    tensor = _draw_chain_parameter_tensor(samples)
    thresholds = thresholds or RankNormalizedHMCThresholds()
    draws, chains, parameters = (int(dim) for dim in tensor.shape)
    if parameter_names is None:
        names = tuple(f"parameter_{index}" for index in range(parameters))
    else:
        names = tuple(str(name) for name in parameter_names)
        if len(names) != parameters:
            raise ValueError("parameter_names length must match parameter dimension")
    input_finite = bool(tf.reduce_all(tf.math.is_finite(tensor)).numpy())
    if not input_finite:
        return {
            "schema": "bayesfilter.rank_normalized_hmc_diagnostics.v1",
            "passed": False,
            "input_all_finite": False,
            "diagnostics_all_finite": False,
            "draw_count_per_chain": draws,
            "chain_count": chains,
            "parameter_count": parameters,
            "split_draw_count_per_chain": draws // 2,
            "split_chain_count": chains * 2,
            "definitions": {
                "rhat": RANK_NORMALIZED_SPLIT_RHAT_DEFINITION,
                "bulk_ess": "rank-normalized bulk ESS",
                "tail_ess": "minimum of 5% and 95% indicator ESS",
            },
            "hard_vetoes": ("input_nonfinite",),
            "max_rhat": float("nan"),
            "min_bulk_ess": float("nan"),
            "min_tail_ess": float("nan"),
            "parameter_diagnostics": tuple(),
            "nonclaims": (
                "finite-sample operational screen only",
                "no stationarity or posterior correctness proof",
            ),
        }

    bulk_rhat, folded_rhat, bulk_ess, ess = _modern_arrays(tensor)
    rhat = tf.maximum(bulk_rhat, folded_rhat)
    tail_ess = ess["tail"]
    lower_ess = ess["lower_5pct"]
    upper_ess = ess["upper_95pct"]
    all_finite = bool(
        tf.reduce_all(
            tf.math.is_finite(
                tf.concat(
                    tuple(
                        tf.reshape(value, (-1, 1))
                        for value in (rhat, bulk_ess, tail_ess, lower_ess, upper_ess)
                    ),
                    axis=1,
                )
            )
        ).numpy()
    )
    rows = []
    for index, name in enumerate(names):
        row_rhat = float(rhat[index].numpy())
        row_bulk = float(bulk_ess[index].numpy())
        row_tail = float(tail_ess[index].numpy())
        row_lower = float(lower_ess[index].numpy())
        row_upper = float(upper_ess[index].numpy())
        row_passed = bool(
            math.isfinite(row_rhat)
            and math.isfinite(row_bulk)
            and math.isfinite(row_tail)
            and row_rhat <= thresholds.rhat_max
            and row_bulk >= thresholds.bulk_ess_min
            and row_tail >= thresholds.tail_ess_min
        )
        rows.append(
            {
                "parameter": name,
                "rank_normalized_split_rhat": float(bulk_rhat[index].numpy()),
                "folded_rank_normalized_split_rhat": float(folded_rhat[index].numpy()),
                "rhat": row_rhat,
                "bulk_ess": row_bulk,
                "tail_ess": row_tail,
                "lower_tail_ess": row_lower,
                "upper_tail_ess": row_upper,
                "passed": row_passed,
            }
        )
    max_rhat = float(tf.reduce_max(rhat).numpy())
    min_bulk = float(tf.reduce_min(bulk_ess).numpy())
    min_tail = float(tf.reduce_min(tail_ess).numpy())
    hard_vetoes = () if all_finite else ("diagnostics_nonfinite",)
    return {
        "schema": "bayesfilter.rank_normalized_hmc_diagnostics.v1",
        "passed": bool(all_finite and all(row["passed"] for row in rows)),
        "input_all_finite": True,
        "diagnostics_all_finite": all_finite,
        "draw_count_per_chain": draws,
        "chain_count": chains,
        "parameter_count": parameters,
        "split_draw_count_per_chain": draws // 2,
        "split_chain_count": chains * 2,
        "definitions": {
            "rhat": RANK_NORMALIZED_SPLIT_RHAT_DEFINITION,
            "bulk_ess": "rank-normalized bulk ESS",
            "tail_ess": "minimum of 5% and 95% indicator ESS",
        },
        "thresholds": {
            "rhat_max": thresholds.rhat_max,
            "bulk_ess_min": thresholds.bulk_ess_min,
            "tail_ess_min": thresholds.tail_ess_min,
        },
        "hard_vetoes": hard_vetoes,
        "max_rhat": max_rhat,
        "min_bulk_ess": min_bulk,
        "min_tail_ess": min_tail,
        "parameter_diagnostics": tuple(rows),
        "nonclaims": (
            "finite-sample operational screen only",
            "no stationarity or posterior correctness proof",
            "no sampler superiority or scientific-validity claim",
        ),
    }


__all__ = [
    "RANK_NORMALIZED_SPLIT_RHAT_DEFINITION",
    "RankNormalizedHMCThresholds",
    "rank_normalized_hmc_diagnostics",
    "rank_normalized_split_rhat_summary",
]
