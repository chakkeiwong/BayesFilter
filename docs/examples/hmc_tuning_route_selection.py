"""Select an HMC tuning route from BayesFilter's capability registry."""

from __future__ import annotations

import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPOSITORY_ROOT))

from bayesfilter.inference import (
    active_hmc_tuning_routes,
    hmc_tuning_interface_capability,
)


def require_artifact_authority(interface_name: str) -> str:
    """Return the qualified public tuner name or reject a lower-level route."""

    capability = hmc_tuning_interface_capability(interface_name)
    if capability.interface_kind != "public_tuner" or not capability.artifact_authority:
        replacement = capability.replacement or "no supported replacement"
        raise ValueError(
            f"{capability.qualified_name} is {capability.interface_kind}, not a "
            f"public artifact-authority tuner; use {replacement}"
        )
    return capability.qualified_name


ordinary = require_artifact_authority("tune_hmc_kernel")
fixed_transport = require_artifact_authority("tune_fixed_transport_hmc_kernel")
assert {record.interface_name for record in active_hmc_tuning_routes()} == {
    "tune_hmc_kernel",
    "tune_fixed_transport_hmc_kernel",
}

try:
    require_artifact_authority("run_full_chain_neural_force_hmc")
except ValueError as error:
    expected_rejection = str(error)
else:  # pragma: no cover - the registry contract requires rejection
    raise AssertionError("a chain runner was incorrectly accepted as a tuner")

try:
    require_artifact_authority("tune_hmc_kernel_robust_broad_grid")
except ValueError as error:
    expected_diagnostic_rejection = str(error)
else:  # pragma: no cover - the registry contract requires rejection
    raise AssertionError("a diagnostic helper was incorrectly accepted as a tuner")

print(ordinary)
print(fixed_transport)
print(expected_rejection)
print(expected_diagnostic_rejection)
