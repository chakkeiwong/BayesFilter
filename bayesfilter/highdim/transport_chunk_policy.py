"""Repository-owned DPF transport chunk selection policy."""

from __future__ import annotations

from dataclasses import dataclass


TRANSPORT_CHUNK_POLICY_ID = "dpf_transport_exact_divisor_cap3000_v1"
MAX_EXACT_CHUNK_SIZE = 3000


@dataclass(frozen=True)
class TransportChunkSelection:
    """An exact square block-grid selection issued by repository policy."""

    policy_id: str
    num_particles: int
    row_chunk_size: int
    col_chunk_size: int
    row_blocks: int
    col_blocks: int


def _require_plain_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be a plain Python integer")
    return value


def select_transport_chunk_size(num_particles: int) -> int:
    """Select the only chunk extent eligible under the active DPF policy."""

    num_particles = _require_plain_integer(num_particles, "num_particles")
    if num_particles <= 1:
        raise ValueError("num_particles must be greater than one")
    if num_particles <= MAX_EXACT_CHUNK_SIZE:
        return num_particles
    for candidate in range(MAX_EXACT_CHUNK_SIZE, 1, -1):
        if num_particles % candidate == 0:
            return candidate
    raise ValueError(
        f"{TRANSPORT_CHUNK_POLICY_ID} found no divisor greater than one for "
        f"N={num_particles} under cap {MAX_EXACT_CHUNK_SIZE}; refusing a tiny fallback"
    )


def select_transport_chunks(num_particles: int) -> TransportChunkSelection:
    """Issue the complete, exact square-grid policy selection."""

    chunk_size = select_transport_chunk_size(num_particles)
    block_count = num_particles // chunk_size
    return TransportChunkSelection(
        policy_id=TRANSPORT_CHUNK_POLICY_ID,
        num_particles=num_particles,
        row_chunk_size=chunk_size,
        col_chunk_size=chunk_size,
        row_blocks=block_count,
        col_blocks=block_count,
    )


def validate_transport_chunks(
    num_particles: int, *, row_chunk_size: int, col_chunk_size: int
) -> TransportChunkSelection:
    """Reject any caller setting that differs from repository policy."""

    row_chunk_size = _require_plain_integer(row_chunk_size, "row_chunk_size")
    col_chunk_size = _require_plain_integer(col_chunk_size, "col_chunk_size")
    selection = select_transport_chunks(num_particles)
    if row_chunk_size <= 0 or col_chunk_size <= 0:
        raise ValueError("row_chunk_size and col_chunk_size must be positive")
    if row_chunk_size != col_chunk_size:
        raise ValueError(
            f"{TRANSPORT_CHUNK_POLICY_ID} requires equal row and column chunks; "
            f"got {row_chunk_size} and {col_chunk_size}"
        )
    if num_particles % row_chunk_size != 0:
        raise ValueError(
            f"{TRANSPORT_CHUNK_POLICY_ID} requires K to divide N exactly; "
            f"got N={num_particles}, K={row_chunk_size}"
        )
    if row_chunk_size != selection.row_chunk_size:
        raise ValueError(
            f"chunk K={row_chunk_size} is wrong under {TRANSPORT_CHUNK_POLICY_ID} "
            f"for N={num_particles}; required K={selection.row_chunk_size}"
        )
    return selection


def resolve_transport_chunks(
    num_particles: int,
    *,
    row_chunk_size: int | None,
    col_chunk_size: int | None,
) -> TransportChunkSelection:
    """Select omitted chunks or validate a complete explicit pair."""

    if row_chunk_size is None and col_chunk_size is None:
        return select_transport_chunks(num_particles)
    if row_chunk_size is None or col_chunk_size is None:
        raise ValueError(
            f"{TRANSPORT_CHUNK_POLICY_ID} requires both chunk sizes or neither"
        )
    return validate_transport_chunks(
        num_particles,
        row_chunk_size=row_chunk_size,
        col_chunk_size=col_chunk_size,
    )


__all__ = [
    "MAX_EXACT_CHUNK_SIZE",
    "TRANSPORT_CHUNK_POLICY_ID",
    "TransportChunkSelection",
    "select_transport_chunk_size",
    "select_transport_chunks",
    "resolve_transport_chunks",
    "validate_transport_chunks",
]
