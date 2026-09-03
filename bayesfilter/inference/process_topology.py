"""Validated process-topology contracts for staged research workers.

The topology is intentionally independent of TensorFlow.  It describes logical
CPU allocation for a controller that starts isolated worker processes; it does
not create a pool, import an accelerator framework, or choose a numerical
algorithm.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any


class ProcessTopologyError(ValueError):
    """Raised when a staged worker topology cannot be realized safely."""


@dataclass(frozen=True)
class BarrierTopology:
    """One mutually exclusive process barrier."""

    name: str
    worker_count: int
    cores_per_worker: int
    work_unit_count: int

    def __post_init__(self) -> None:
        if not str(self.name):
            raise ProcessTopologyError("barrier name must be nonempty")
        for field in ("worker_count", "cores_per_worker", "work_unit_count"):
            value = int(getattr(self, field))
            if value <= 0:
                raise ProcessTopologyError(f"{field} must be positive")
            object.__setattr__(self, field, value)

    @property
    def worker_core_total(self) -> int:
        return int(self.worker_count) * int(self.cores_per_worker)

    def payload(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "worker_count": self.worker_count,
            "cores_per_worker": self.cores_per_worker,
            "work_unit_count": self.work_unit_count,
            "worker_core_total": self.worker_core_total,
        }


@dataclass(frozen=True)
class WorkerAssignment:
    """A worker index and its disjoint logical CPU set."""

    worker_index: int
    cpu_ids: tuple[int, ...]

    def __post_init__(self) -> None:
        index = int(self.worker_index)
        if index < 0:
            raise ProcessTopologyError("worker_index must be nonnegative")
        ids = tuple(int(value) for value in self.cpu_ids)
        if not ids or len(set(ids)) != len(ids) or any(value < 0 for value in ids):
            raise ProcessTopologyError("cpu_ids must be unique and nonnegative")
        object.__setattr__(self, "worker_index", index)
        object.__setattr__(self, "cpu_ids", ids)

    def payload(self) -> dict[str, Any]:
        return {
            "worker_index": self.worker_index,
            "cpu_ids": list(self.cpu_ids),
            "cores": len(self.cpu_ids),
        }


@dataclass(frozen=True)
class Q20ProcessTopology:
    """The requested three-barrier 72-worker-core contract."""

    screen: BarrierTopology = BarrierTopology(
        name="screen", worker_count=8, cores_per_worker=4, work_unit_count=48
    )
    selection: BarrierTopology = BarrierTopology(
        name="selection", worker_count=2, cores_per_worker=8, work_unit_count=2
    )
    scope_finalize: BarrierTopology = BarrierTopology(
        name="scope_finalize", worker_count=6, cores_per_worker=4, work_unit_count=6
    )

    def __post_init__(self) -> None:
        barriers = (self.screen, self.selection, self.scope_finalize)
        if any(not isinstance(value, BarrierTopology) for value in barriers):
            raise ProcessTopologyError("all barriers must be BarrierTopology values")
        names = tuple(value.name for value in barriers)
        if len(set(names)) != len(names):
            raise ProcessTopologyError("barrier names must be unique")
        if self.screen.worker_core_total != 32:
            raise ProcessTopologyError("screen barrier must reserve 8x4=32 cores")
        if self.selection.worker_core_total != 16:
            raise ProcessTopologyError("selection barrier must reserve 2x8=16 cores")
        if self.scope_finalize.worker_core_total != 24:
            raise ProcessTopologyError("scope barrier must reserve 6x4=24 cores")
        if self.total_worker_cores != 72:
            raise ProcessTopologyError("staged topology must reserve exactly 72 worker cores")

    @property
    def barriers(self) -> tuple[BarrierTopology, ...]:
        return (self.screen, self.selection, self.scope_finalize)

    @property
    def total_worker_cores(self) -> int:
        return sum(value.worker_core_total for value in self.barriers)

    @property
    def peak_barrier_cores(self) -> int:
        return max(value.worker_core_total for value in self.barriers)

    def barrier(self, name: str) -> BarrierTopology:
        for value in self.barriers:
            if value.name == str(name):
                return value
        raise ProcessTopologyError(f"unknown barrier: {name}")

    def validate_available_cpu_ids(self, cpu_ids: Iterable[int]) -> tuple[int, ...]:
        available = tuple(sorted({int(value) for value in cpu_ids}))
        if any(value < 0 for value in available):
            raise ProcessTopologyError("available CPU IDs must be nonnegative")
        if len(available) < self.total_worker_cores:
            raise ProcessTopologyError(
                f"need {self.total_worker_cores} logical CPUs, found {len(available)}"
            )
        return available

    def assignments(
        self, barrier_name: str, available_cpu_ids: Sequence[int]
    ) -> tuple[WorkerAssignment, ...]:
        barrier = self.barrier(barrier_name)
        available = self.validate_available_cpu_ids(available_cpu_ids)
        required = barrier.worker_core_total
        if len(available) < required:
            raise ProcessTopologyError(
                f"barrier {barrier.name} needs {required} logical CPUs, found {len(available)}"
            )
        selected = available[:required]
        return tuple(
            WorkerAssignment(
                worker_index=index,
                cpu_ids=tuple(
                    selected[
                        index * barrier.cores_per_worker :
                        (index + 1) * barrier.cores_per_worker
                    ]
                ),
            )
            for index in range(barrier.worker_count)
        )

    def payload(self) -> dict[str, Any]:
        return {
            "schema": "bayesfilter.q20.process_topology.v1",
            "barriers": [value.payload() for value in self.barriers],
            "total_worker_cores": self.total_worker_cores,
            "peak_barrier_cores": self.peak_barrier_cores,
            "barriers_are_sequential": True,
            "logical_cpu_ids_not_physical_core_claim": True,
        }


__all__ = [
    "BarrierTopology",
    "ProcessTopologyError",
    "Q20ProcessTopology",
    "WorkerAssignment",
]
