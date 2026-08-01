"""Repository-owned identity for offline LEDH tuning scopes."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class LEDHTuningScope:
    """All inputs that make a tuned LEDH finite program scope-specific."""

    model_id: str
    target_id: str
    route_id: str
    reset_contract_id: str
    horizon: int
    prepared_data_id: str
    particle_count: int
    state_dimension: int
    parameter_count: int
    dtype: str
    tf32_enabled: bool
    jit_compile: bool
    chunk_policy_id: str
    row_chunk_size: int
    col_chunk_size: int
    row_blocks: int
    col_blocks: int
    control_family_id: str

    def __post_init__(self) -> None:
        if self.horizon < 1:
            raise ValueError("LEDH tuning horizon must be positive")
        if self.particle_count < 1:
            raise ValueError("LEDH tuning particle count must be positive")
        if self.state_dimension < 1 or self.parameter_count < 1:
            raise ValueError("LEDH tuning dimensions must be positive")
        if min(
            self.row_chunk_size,
            self.col_chunk_size,
            self.row_blocks,
            self.col_blocks,
        ) < 1:
            raise ValueError("LEDH tuning chunk sizes and block counts must be positive")
        if self.row_chunk_size != self.col_chunk_size:
            raise ValueError("LEDH tuning scopes require equal row and column chunks")
        if self.row_chunk_size * self.row_blocks != self.particle_count:
            raise ValueError("LEDH tuning row block geometry must exactly tile particles")
        if self.col_chunk_size * self.col_blocks != self.particle_count:
            raise ValueError("LEDH tuning column block geometry must exactly tile particles")
        if self.dtype != "float32" and self.tf32_enabled:
            raise ValueError("TF32 is only valid for float32 tuning scopes")

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def scope_sha256(self) -> str:
        encoded = json.dumps(
            self.as_dict(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def scope_from_mapping(values: Mapping[str, Any]) -> LEDHTuningScope:
    """Build an exact scope from a serialized selection or claim record."""

    return LEDHTuningScope(**dict(values))


def require_scope_match(
    expected: LEDHTuningScope, actual: Mapping[str, Any], *, label: str
) -> None:
    observed = scope_from_mapping(actual)
    if observed != expected or observed.scope_sha256 != expected.scope_sha256:
        raise ValueError(
            f"{label} does not match the repository-issued LEDH tuning scope"
        )


__all__ = ["LEDHTuningScope", "require_scope_match", "scope_from_mapping"]
