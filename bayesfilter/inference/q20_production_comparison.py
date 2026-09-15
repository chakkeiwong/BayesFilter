"""Comparison boundary for q20 posterior artifacts.

The reference method is intentionally unimplemented. This module records that
fact instead of fabricating agreement from a training or sampler diagnostic.
"""
from __future__ import annotations

from bayesfilter.inference.q20_production_config import frozen_scope_hash


def posterior_summary(config, retained, *, target_signature, label, sequential_passed):
    count = None
    shape = None
    if retained is not None:
        shape = [int(v) for v in retained.shape]
        count = int(retained.shape[0])
    return {
        "schema": "bayesfilter.q20.posterior_summary.v1",
        "label": str(label),
        "target_signature": str(target_signature),
        "frozen_scope_hash": frozen_scope_hash(config),
        "retained_shape": shape,
        "retained_draws_per_chain": count,
        "sequential_declared_checks_passed": bool(sequential_passed),
        "reference_status": "unavailable",
        "reference_agreement": "incomplete",
        "method_ranking": "not_estimated",
        "production_qualified": False,
        "nonclaims": [
            "R-hat, ESS or MCSE does not establish target correctness or mode coverage",
            "a missing independent reference cannot be replaced by a training score",
            "one chart or one seed cannot establish ensemble or method superiority",
        ],
    }
