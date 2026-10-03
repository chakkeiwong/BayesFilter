"""Deterministic temperature lineages for reverse-KL transport training.

The controller owns only lineage identity, seed derivation, checkpoint
metadata, and the fixed pre-optimizer validity screen.  It does not infer
posterior mode masses and it never replaces an invalid row with a lucky draw.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import tensorflow as tf

from bayesfilter.inference.tempered_transport_ensemble_tf import (
    InitializationPreflight,
    PreparedTransportInitialization,
    TemperedEnsembleError,
    prepare_transport_initialization,
)


LINEAGE_SCHEMA = "bayesfilter.tempered.reverse_kl_lineage.v1"
LINEAGE_NONCLAIMS = (
    "lineage identity and preflight metadata only",
    "positive-temperature branching is not a mode-discovery guarantee",
    "lineage checkpoints are not posterior samples",
    "no posterior mass or convergence claim",
)


def _hash(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("ascii")
    return hashlib.sha256(encoded).hexdigest()


def _seed_tuple(seed: Any, name: str) -> tuple[int, int]:
    values = tuple(int(item) for item in seed)
    if len(values) != 2:
        raise TemperedEnsembleError(f"{name} must contain exactly two integers")
    return values


@dataclass(frozen=True)
class TemperedLineageConfig:
    """Frozen beta ladder and discovery-arm definition."""

    betas: tuple[float, ...]
    component_ids: tuple[str, ...]
    root_seed: tuple[int, int]
    discovery_arm: str = "pure_continuation"
    positive_branch_betas: tuple[float, ...] = ()
    restart_component_indices: tuple[int, ...] = ()
    preflight_batch_size: int = 32
    repair_scales: tuple[float, ...] = (1.0, 0.5, 0.25)

    def __post_init__(self) -> None:
        betas = tuple(float(value) for value in self.betas)
        if len(betas) < 2 or betas[0] != 0.0 or betas[-1] != 1.0:
            raise TemperedEnsembleError("beta ladder must start at 0 and end at 1")
        if any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in betas):
            raise TemperedEnsembleError("betas must be finite and lie in [0,1]")
        # Adjacent slices intentionally have lengths n-1; strict zip would
        # reject every valid ladder because the two slices are not equal.
        if any(right <= left for left, right in zip(betas, betas[1:])):
            raise TemperedEnsembleError("betas must be strictly increasing")
        ids = tuple(str(item) for item in self.component_ids)
        if not ids or any(not item for item in ids) or len(set(ids)) != len(ids):
            raise TemperedEnsembleError("component_ids must be nonempty and unique")
        if self.discovery_arm not in {"pure_continuation", "positive_temperature_branching"}:
            raise TemperedEnsembleError("unsupported discovery_arm")
        branch_betas = tuple(float(value) for value in self.positive_branch_betas)
        if any(
            not math.isfinite(value) or value <= 0.0 or value > 1.0 for value in branch_betas
        ):
            raise TemperedEnsembleError("positive_branch_betas must lie in (0,1]")
        if any(value not in betas for value in branch_betas):
            raise TemperedEnsembleError("branch betas must be ladder levels")
        if self.discovery_arm == "positive_temperature_branching" and not branch_betas:
            raise TemperedEnsembleError(
                "positive_temperature_branching requires at least one branch beta"
            )
        restart_indices = tuple(int(value) for value in self.restart_component_indices)
        if len(set(restart_indices)) != len(restart_indices) or any(
            value < 0 or value >= len(ids) for value in restart_indices
        ):
            raise TemperedEnsembleError(
                "restart_component_indices must be unique component indices"
            )
        if self.discovery_arm == "pure_continuation" and restart_indices:
            raise TemperedEnsembleError(
                "pure continuation cannot declare restart components"
            )
        if self.discovery_arm == "positive_temperature_branching" and (
            not restart_indices or len(restart_indices) >= len(ids)
        ):
            raise TemperedEnsembleError(
                "positive-temperature branching requires a nonempty proper subset "
                "of restart components"
            )
        batch_size = int(self.preflight_batch_size)
        if batch_size <= 1:
            raise TemperedEnsembleError("preflight_batch_size must exceed one")
        scales = tuple(float(value) for value in self.repair_scales)
        if not scales or any(not math.isfinite(value) or value <= 0.0 for value in scales):
            raise TemperedEnsembleError("repair_scales must be finite and positive")
        object.__setattr__(self, "betas", betas)
        object.__setattr__(self, "component_ids", ids)
        object.__setattr__(self, "root_seed", _seed_tuple(self.root_seed, "root_seed"))
        object.__setattr__(self, "positive_branch_betas", branch_betas)
        object.__setattr__(self, "restart_component_indices", restart_indices)
        object.__setattr__(self, "preflight_batch_size", batch_size)
        object.__setattr__(self, "repair_scales", scales)

    @property
    def component_count(self) -> int:
        return len(self.component_ids)

    def payload(self) -> Mapping[str, Any]:
        payload = asdict(self)
        payload["betas"] = list(self.betas)
        payload["component_ids"] = list(self.component_ids)
        payload["root_seed"] = list(self.root_seed)
        payload["positive_branch_betas"] = list(self.positive_branch_betas)
        payload["restart_component_indices"] = list(
            self.restart_component_indices
        )
        payload["repair_scales"] = list(self.repair_scales)
        payload["schema"] = LINEAGE_SCHEMA
        payload["nonclaims"] = list(LINEAGE_NONCLAIMS)
        return payload

    @property
    def signature(self) -> str:
        return _hash(self.payload())


@dataclass(frozen=True)
class LineageCheckpoint:
    beta_index: int
    beta: float
    component_ids: tuple[str, ...]
    component_seeds: tuple[tuple[int, int], ...]
    parent_indices: tuple[int, ...]
    discovery_arm: str
    bridge_signature: str
    lineage_signature: str
    checkpoint_hash: str

    def payload(self) -> Mapping[str, Any]:
        payload = asdict(self)
        payload["component_ids"] = list(self.component_ids)
        payload["component_seeds"] = [list(seed) for seed in self.component_seeds]
        payload["parent_indices"] = list(self.parent_indices)
        return payload


class TemperedLineageController:
    """Create reproducible lineage checkpoints and preflight receipts."""

    def __init__(self, config: TemperedLineageConfig, bridge: Any) -> None:
        if not getattr(bridge, "signature", None):
            raise TemperedEnsembleError("bridge must expose an immutable signature")
        if not callable(getattr(bridge, "value_score_status", None)):
            raise TemperedEnsembleError("bridge must expose value_score_status")
        self.config = config
        self.bridge = bridge
        self.bridge_signature = str(bridge.signature)
        self._checkpoints: dict[int, LineageCheckpoint] = {}
        self._preflight: dict[tuple[int, str], InitializationPreflight] = {}
        self._admitted_transports: dict[tuple[int, str], Any] = {}

    @property
    def checkpoints(self) -> tuple[LineageCheckpoint, ...]:
        return tuple(self._checkpoints[index] for index in sorted(self._checkpoints))

    @property
    def preflight_receipts(self) -> tuple[InitializationPreflight, ...]:
        return tuple(self._preflight[key] for key in sorted(self._preflight))

    def component_seed(self, beta_index: int, component_index: int, *, role: int = 0) -> tuple[int, int]:
        if not 0 <= int(beta_index) < len(self.config.betas):
            raise TemperedEnsembleError("beta_index is out of range")
        if not 0 <= int(component_index) < self.config.component_count:
            raise TemperedEnsembleError("component_index is out of range")
        seed = tf.constant(self.config.root_seed, tf.int32)
        seed = tf.random.experimental.stateless_fold_in(seed, int(beta_index))
        seed = tf.random.experimental.stateless_fold_in(seed, int(component_index))
        seed = tf.random.experimental.stateless_fold_in(seed, int(role))
        return tuple(int(item) for item in seed.numpy())

    def checkpoint(
        self,
        beta_index: int,
        *,
        parent_indices: Sequence[int] | None = None,
    ) -> LineageCheckpoint:
        index = int(beta_index)
        if not 0 <= index < len(self.config.betas):
            raise TemperedEnsembleError("beta_index is out of range")
        parents = tuple(
            int(item)
            for item in (
                self.branch_parent_indices(index)
                if parent_indices is None
                else parent_indices
            )
        )
        if len(parents) != self.config.component_count or any(
            item < -1 or item >= self.config.component_count for item in parents
        ):
            raise TemperedEnsembleError(
                "parent_indices must index the component bank or use -1 for a restart"
            )
        seeds = tuple(
            self.component_seed(
                index,
                component,
                role=(
                    1 if parents[component] == -1 else 0
                ),
            )
            for component in range(self.config.component_count)
        )
        payload = {
            "beta_index": index,
            "beta": self.config.betas[index],
            "component_ids": list(self.config.component_ids),
            "component_seeds": [list(seed) for seed in seeds],
            "parent_indices": list(parents),
            "discovery_arm": self.config.discovery_arm,
            "bridge_signature": self.bridge_signature,
            "lineage_signature": self.config.signature,
        }
        checkpoint = LineageCheckpoint(
            beta_index=index,
            beta=self.config.betas[index],
            component_ids=self.config.component_ids,
            component_seeds=seeds,
            parent_indices=parents,
            discovery_arm=self.config.discovery_arm,
            bridge_signature=self.bridge_signature,
            lineage_signature=self.config.signature,
            checkpoint_hash=_hash(payload),
        )
        previous = self._checkpoints.get(index)
        if previous is not None and previous != checkpoint:
            raise TemperedEnsembleError("lineage checkpoint identity is immutable")
        self._checkpoints[index] = checkpoint
        return checkpoint

    def preflight_components(
        self,
        transports: Sequence[Any],
        *,
        beta_index: int = 0,
        repair_scales: Sequence[float] | None = None,
        batch_size: int | None = None,
    ) -> tuple[InitializationPreflight, ...]:
        values = tuple(transports)
        if len(values) != self.config.component_count:
            raise TemperedEnsembleError("transport count does not match lineage")
        index = int(beta_index)
        checkpoint = self.checkpoint(index)
        del checkpoint
        scales = self.config.repair_scales if repair_scales is None else tuple(repair_scales)
        size = self.config.preflight_batch_size if batch_size is None else int(batch_size)
        receipts = []
        for component_index, (component_id, transport) in enumerate(
            zip(self.config.component_ids, values, strict=True)
        ):
            prepared = prepare_transport_initialization(
                transport,
                self.bridge,
                component_id=component_id,
                seed=self.component_seed(index, component_index, role=99),
                batch_size=size,
                repair_scales=scales,
                beta=self.config.betas[index],
                reference_center=getattr(self.bridge, "prior_center", None),
                reference_scale=(
                    math.sqrt(float(getattr(self.bridge, "prior_variance")))
                    if getattr(self.bridge, "prior_variance", None) is not None
                    else None
                ),
            )
            receipt = prepared.receipt
            previous = self._preflight.get((index, component_id))
            if previous is not None and previous != receipt:
                raise TemperedEnsembleError(
                    "lineage preflight identity is immutable"
                )
            self._preflight[(index, component_id)] = receipt
            self._admitted_transports[(index, component_id)] = prepared.transport
            receipts.append(receipt)
        return tuple(receipts)

    def require_preflight(self, beta_index: int = 0) -> tuple[InitializationPreflight, ...]:
        rows = tuple(
            self._preflight[(int(beta_index), component_id)]
            for component_id in self.config.component_ids
            if (int(beta_index), component_id) in self._preflight
        )
        if len(rows) != self.config.component_count:
            raise TemperedEnsembleError("all components require a completed preflight")
        if not all(row.valid for row in rows):
            raise TemperedEnsembleError("lineage preflight contains an invalid component")
        return rows

    def admitted_transports(self, beta_index: int = 0) -> tuple[Any, ...]:
        """Return only maps whose exact state passed the fixed preflight bank."""
        self.require_preflight(beta_index)
        index = int(beta_index)
        return tuple(
            self._admitted_transports[(index, component_id)]
            for component_id in self.config.component_ids
        )

    def prepared_initializations(
        self, beta_index: int = 0
    ) -> tuple[PreparedTransportInitialization, ...]:
        receipts = self.require_preflight(beta_index)
        transports = self.admitted_transports(beta_index)
        return tuple(
            PreparedTransportInitialization(transport, receipt)
            for transport, receipt in zip(transports, receipts, strict=True)
        )

    def branch_parent_indices(self, beta_index: int) -> tuple[int, ...]:
        """Return deterministic parents; positive branch levels use fresh seeds."""
        index = int(beta_index)
        if not 0 <= index < len(self.config.betas):
            raise TemperedEnsembleError("beta_index is out of range")
        beta = self.config.betas[index]
        parents = list(range(self.config.component_count))
        if (
            self.config.discovery_arm == "positive_temperature_branching"
            and beta in self.config.positive_branch_betas
        ):
            for component in self.config.restart_component_indices:
                parents[component] = -1
        return tuple(parents)

    def seed_ledger(self) -> Mapping[str, Any]:
        rows = []
        for beta_index in range(len(self.config.betas)):
            parents = self.branch_parent_indices(beta_index)
            for component_index, component_id in enumerate(self.config.component_ids):
                role = 1 if parents[component_index] == -1 else 0
                rows.append(
                    {
                        "beta_index": beta_index,
                        "component_index": component_index,
                        "component_id": component_id,
                        "role": "fresh_restart" if role else "continuation",
                        "seed": list(
                            self.component_seed(
                                beta_index, component_index, role=role
                            )
                        ),
                    }
                )
        seeds = tuple(tuple(row["seed"]) for row in rows)
        return {
            "rows": rows,
            "seed_count": len(seeds),
            "all_seeds_unique": len(seeds) == len(set(seeds)),
        }

    def manifest_payload(self) -> Mapping[str, Any]:
        return {
            "schema": "bayesfilter.tempered.lineage_manifest.v1",
            "lineage": self.config.payload(),
            "lineage_signature": self.config.signature,
            "bridge_signature": self.bridge_signature,
            "checkpoints": [row.payload() for row in self.checkpoints],
            "preflight": [row.payload() for row in self.preflight_receipts],
            "seed_ledger": self.seed_ledger(),
            "nonclaims": list(LINEAGE_NONCLAIMS),
        }


__all__ = [
    "LINEAGE_NONCLAIMS",
    "LINEAGE_SCHEMA",
    "LineageCheckpoint",
    "TemperedLineageConfig",
    "TemperedLineageController",
]
