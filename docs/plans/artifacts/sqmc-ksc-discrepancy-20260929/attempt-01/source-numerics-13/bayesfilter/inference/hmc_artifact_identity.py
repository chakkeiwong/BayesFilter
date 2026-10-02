"""Dependency-light identities shared by HMC tuning stages."""

from __future__ import annotations

from typing import Any

from bayesfilter.runtime import stable_config_hash


def mass_artifact_signature(mass_artifact: Any) -> str:
    """Hash a mass artifact using the frozen JSON payload semantics."""

    required = ("signature_payload", "position", "covariance", "factor")
    if any(not hasattr(mass_artifact, name) for name in required):
        raise TypeError("mass_artifact does not expose the canonical signature fields")
    return stable_config_hash(
        {
            "signature_payload": mass_artifact.signature_payload(),
            "position": mass_artifact.position,
            "covariance": mass_artifact.covariance,
            "factor": mass_artifact.factor,
        }
    )


__all__ = ["mass_artifact_signature"]
