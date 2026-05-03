"""Particle-filter namespace reserved for audited future backends."""

from __future__ import annotations


class ParticleFilterNotAuditedError(NotImplementedError):
    """Raised when particle filtering is requested before an audit gate passes."""


def particle_filter_log_likelihood(*args, **kwargs):
    """Placeholder that fails closed until particle-filter semantics are audited."""

    raise ParticleFilterNotAuditedError(
        "particle filters require a separate proposal, resampling, and target audit"
    )
