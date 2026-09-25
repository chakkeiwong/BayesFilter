"""Diagnostics for global mode mixing of an exact NeuTra HMC run.

This module does not construct a posterior or assign mode weights.  It only
checks whether retained labels from one common HMC target show the minimum
cross-mode behavior needed before pooled posterior summaries are considered.
Mode-specific chains that never transition are deliberately rejected.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import tensorflow as tf


class GlobalMixingDiagnosticError(ValueError):
    """Raised when retained mode labels do not have the declared shape."""


@dataclass(frozen=True)
class GlobalMixingReport:
    """Tensor diagnostics for retained mode labels.

    ``passed`` is a mechanics/coverage screen only.  It is not a convergence
    proof and must be combined with the repository's R-hat, ESS, finite-target,
    and status gates.
    """

    passed: tf.Tensor
    valid_labels: tf.Tensor
    chain_region_counts: tf.Tensor
    chain_transition_counts: tf.Tensor
    global_region_counts: tf.Tensor
    every_chain_visited_every_region: tf.Tensor
    every_chain_transitioned: tf.Tensor

    def payload(self) -> Mapping[str, Any]:
        """Return JSON-safe diagnostics without introducing a NumPy path."""

        def native(value: tf.Tensor) -> Any:
            raw = value.numpy()
            if getattr(raw, "shape", ()) == ():
                return raw.item()
            return raw.tolist()

        return {
            "passed": bool(native(self.passed)),
            "valid_labels": bool(native(self.valid_labels)),
            "chain_region_counts": native(self.chain_region_counts),
            "chain_transition_counts": native(self.chain_transition_counts),
            "global_region_counts": native(self.global_region_counts),
            "every_chain_visited_every_region": bool(
                native(self.every_chain_visited_every_region)
            ),
            "every_chain_transitioned": bool(native(self.every_chain_transitioned)),
            "role": "global_mixing_coverage_diagnostic_not_posterior_proof",
        }


def assess_retained_mode_mixing(
    region_labels: Any,
    *,
    region_count: int,
    minimum_transitions_per_chain: int = 1,
) -> GlobalMixingReport:
    """Assess whether every retained chain crossed all declared regions.

    Parameters
    ----------
    region_labels:
        Integer tensor with shape ``[chain, retained_draw]``.  Labels are
        retained-state labels only; initial labels are intentionally not used
        to manufacture coverage.
    region_count:
        Number of material regions represented by the labeler.
    minimum_transitions_per_chain:
        Minimum adjacent retained-label changes per chain.  The default one is
        a canary-level lower bound; serious runs should also require the
        mode-indicator R-hat/ESS and declared transition/MCSE evidence.
    """

    if isinstance(region_count, bool) or int(region_count) < 2:
        raise GlobalMixingDiagnosticError("region_count must be at least two")
    if (
        isinstance(minimum_transitions_per_chain, bool)
        or int(minimum_transitions_per_chain) < 1
    ):
        raise GlobalMixingDiagnosticError(
            "minimum_transitions_per_chain must be positive"
        )

    labels = tf.convert_to_tensor(region_labels, dtype=tf.int32)
    if labels.shape.rank != 2:
        raise GlobalMixingDiagnosticError(
            "region_labels must have shape [chain, retained_draw]"
        )
    if labels.shape[0] is not None and int(labels.shape[0]) < 1:
        raise GlobalMixingDiagnosticError("at least one chain is required")
    if labels.shape[1] is not None and int(labels.shape[1]) < 2:
        raise GlobalMixingDiagnosticError("at least two retained draws are required")

    count = int(region_count)
    valid = tf.reduce_all(tf.logical_and(labels >= 0, labels < count))
    chain_region_counts = tf.stack(
        [tf.reduce_sum(tf.cast(labels == index, tf.int32), axis=1) for index in range(count)],
        axis=1,
    )
    transitions = tf.reduce_sum(
        tf.cast(labels[:, 1:] != labels[:, :-1], tf.int32), axis=1
    )
    global_region_counts = tf.reduce_sum(chain_region_counts, axis=0)
    every_chain_visited = tf.reduce_all(chain_region_counts > 0)
    every_chain_transitioned = tf.reduce_all(
        transitions >= int(minimum_transitions_per_chain)
    )
    passed = tf.logical_and(
        valid, tf.logical_and(every_chain_visited, every_chain_transitioned)
    )
    return GlobalMixingReport(
        passed=passed,
        valid_labels=valid,
        chain_region_counts=chain_region_counts,
        chain_transition_counts=transitions,
        global_region_counts=global_region_counts,
        every_chain_visited_every_region=every_chain_visited,
        every_chain_transitioned=every_chain_transitioned,
    )

