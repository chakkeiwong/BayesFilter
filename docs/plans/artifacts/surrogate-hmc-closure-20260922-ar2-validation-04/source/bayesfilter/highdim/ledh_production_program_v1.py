"""LEDH Production Program V1 Definition.

This module defines the canonical LEDH production program that all claim-bearing
runs must implement. The production program specifies required mechanisms,
wiring gates, and configuration contracts.

Policy Context
--------------
Per Configuration-Status-First Reporting Rule (adopted 2026-08-26):
  "the production program is DEFINED IN CODE, 'production' labels are validated
  against it (wiring gate), and omitting a required mechanism must surface as a
  labeled deviation in the configuration-status table."

Per LEDH Per-Scope Tuning Rule:
  Every claim-bearing LEDH model run requires an offline tuning artifact for the
  exact model/target, route/reset family, horizon/prepared-data regime, particle
  count, dimensions, dtype/backend, chunk policy, and route-specific control family.

Owner Decision (2026-09-02):
  Dual-cap trust-region covariance stabilization is a REQUIRED mechanism for
  LEDH production runs. Trust-region solver (JVP route with Levenberg-Marquardt
  damping) is also required.

Program Version History
-----------------------
- V1 (2026-09-02): Initial definition with dual-cap + trust-region requirement
"""

from __future__ import annotations

from typing import Any, TypedDict


class LEDHProductionProgramV1(TypedDict):
    """LEDH Production Program V1 specification.

    Fields
    ------
    program_id : str
        Unique identifier for this program version.

    reset_policy : str
        Required reset policy. Must be "contract_e".

    dual_cap_required : bool
        Whether dual-cap covariance stabilization is required.

    trust_region_required : bool
        Whether trust-region solver (JVP + LM damping) is required.

    transport_mode : str
        Required transport plan computation mode. Must be "chunked" for production.

    chunk_policy : str
        Required chunk policy identifier. Must be "dpf_transport_exact_divisor_cap3000_v1".

    backend : str
        Required backend. Must be "tensorflow".

    dtype : str
        Required dtype. Must be "float32".

    tf32_enabled : bool
        Whether TensorFlow TF32 execution must be enabled.

    jit_compile : bool
        Whether XLA compilation must be enabled for hot kernels.

    tuning_required : bool
        Whether per-scope tuning artifacts are required for claim-bearing runs.

    diagnostic_observability_required : bool
        Whether full diagnostic payload must be preserved in output artifacts.

    version : str
        Program version string.

    adoption_date : str
        Date this program version was adopted (ISO 8601).
    """
    program_id: str
    reset_policy: str
    dual_cap_required: bool
    trust_region_required: bool
    transport_mode: str
    chunk_policy: str
    backend: str
    dtype: str
    tf32_enabled: bool
    jit_compile: bool
    tuning_required: bool
    diagnostic_observability_required: bool
    version: str
    adoption_date: str


# Canonical program definition
LEDH_PRODUCTION_PROGRAM_V1: LEDHProductionProgramV1 = {
    "program_id": "ledh_pfpf_ot_contract_e_dual_cap_trust_region_v1",
    "reset_policy": "contract_e",
    "dual_cap_required": True,
    "trust_region_required": True,
    "transport_mode": "chunked",
    "chunk_policy": "dpf_transport_exact_divisor_cap3000_v1",
    "backend": "tensorflow",
    "dtype": "float32",
    "tf32_enabled": True,
    "jit_compile": True,
    "tuning_required": True,
    "diagnostic_observability_required": True,
    "version": "v1",
    "adoption_date": "2026-09-02",
}


def validate_ledh_production_configuration(
    *,
    reset_policy: str,
    dual_cap_enabled: bool,
    trust_region_enabled: bool,
    transport_mode: str | None = None,
    chunk_policy: str | None = None,
    dtype: str | None = None,
    program_label: str = "production",
) -> dict[str, Any]:
    """Validate configuration against LEDH Production Program V1.

    This is the wiring gate that enforces production program requirements.

    Parameters
    ----------
    reset_policy : str
        Reset policy being used.
    dual_cap_enabled : bool
        Whether dual-cap is enabled.
    trust_region_enabled : bool
        Whether trust-region solver is enabled.
    transport_mode : str, optional
        Transport plan computation mode.
    chunk_policy : str, optional
        Chunk policy identifier.
    dtype : str, optional
        Tensor dtype.
    program_label : str, default "production"
        Label for this run. Only "production" triggers strict validation.

    Returns
    -------
    dict
        Validation result with fields:
        - valid : bool
            Whether configuration passes wiring gate.
        - deviations : list[str]
            List of deviations from production program.
        - program_id : str
            Production program identifier.
        - configuration_status : str
            One of: "production_compliant", "labeled_deviation", "invalid".

    Raises
    ------
    ValueError
        If program_label is "production" and configuration fails wiring gate.

    Notes
    -----
    Per Configuration-Status-First Reporting Rule, omitting a required mechanism
    when labeled as "production" must fail loudly at the wiring gate, not silently
    define production downward.

    Non-production labels (e.g., "experimental", "baseline", "ablation") bypass
    strict validation but still record deviations.
    """
    program = LEDH_PRODUCTION_PROGRAM_V1
    deviations = []

    # Check required mechanisms
    if reset_policy != program["reset_policy"]:
        deviations.append(
            f"reset_policy='{reset_policy}' (required: '{program['reset_policy']}')"
        )

    if program["dual_cap_required"] and not dual_cap_enabled:
        deviations.append("dual_cap_enabled=False (required: True)")

    if program["trust_region_required"] and not trust_region_enabled:
        deviations.append("trust_region_enabled=False (required: True)")

    # Check optional fields if provided
    if transport_mode is not None and transport_mode != program["transport_mode"]:
        deviations.append(
            f"transport_mode='{transport_mode}' (expected: '{program['transport_mode']}')"
        )

    if chunk_policy is not None and chunk_policy != program["chunk_policy"]:
        deviations.append(
            f"chunk_policy='{chunk_policy}' (expected: '{program['chunk_policy']}')"
        )

    if dtype is not None and dtype != program["dtype"]:
        deviations.append(f"dtype='{dtype}' (expected: '{program['dtype']}')")

    valid = len(deviations) == 0

    if program_label == "production":
        configuration_status = "production_compliant" if valid else "invalid"
    else:
        configuration_status = "labeled_deviation" if deviations else "production_compliant"

    result = {
        "valid": valid,
        "deviations": deviations,
        "program_id": program["program_id"],
        "program_version": program["version"],
        "configuration_status": configuration_status,
        "program_label": program_label,
    }

    # Fail loudly for production-labeled invalid configurations
    if program_label == "production" and not valid:
        deviation_summary = "; ".join(deviations)
        raise ValueError(
            f"Configuration labeled 'production' fails LEDH_PRODUCTION_PROGRAM_V1 "
            f"wiring gate. Deviations: {deviation_summary}. "
            f"Either fix the configuration to match the production program, or use a "
            f"different label (e.g., 'experimental', 'baseline', 'ablation')."
        )

    return result


def get_production_program_summary() -> dict[str, Any]:
    """Return human-readable summary of production program requirements.

    Returns
    -------
    dict
        Summary with required mechanisms and policy references.
    """
    program = LEDH_PRODUCTION_PROGRAM_V1
    return {
        "program_id": program["program_id"],
        "version": program["version"],
        "adoption_date": program["adoption_date"],
        "required_mechanisms": [
            f"Reset policy: {program['reset_policy']}",
            f"Dual-cap covariance stabilization: {'required' if program['dual_cap_required'] else 'optional'}",
            f"Trust-region solver: {'required' if program['trust_region_required'] else 'optional'}",
            f"Transport mode: {program['transport_mode']}",
            f"Chunk policy: {program['chunk_policy']}",
            f"Backend: {program['backend']}",
            f"Dtype: {program['dtype']}",
            f"TF32: {'enabled' if program['tf32_enabled'] else 'disabled'}",
            f"JIT compilation: {'enabled' if program['jit_compile'] else 'disabled'}",
        ],
        "tuning_required": program["tuning_required"],
        "diagnostic_observability_required": program["diagnostic_observability_required"],
        "policy_references": [
            "Configuration-Status-First Reporting Rule (CLAUDE.md)",
            "LEDH Per-Scope Tuning Rule (CLAUDE.md)",
            "Safety Guardrail Reversed Burden (CLAUDE.md)",
        ],
    }
