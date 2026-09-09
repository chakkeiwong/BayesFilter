"""Persistent budget accounting for bounded local research campaigns."""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
import time
from collections.abc import Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


CAMPAIGN_BUDGET_LEDGER_SCHEMA = "bayesfilter.campaign_budget_ledger.v1"


class CampaignBudgetLedgerError(RuntimeError):
    """Raised when a campaign budget cannot be read or safely advanced."""


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CampaignBudgetLedgerError("ledger values must be finite")
        return value
    raise CampaignBudgetLedgerError(
        f"ledger value is not JSON-compatible: {type(value).__name__}"
    )


def _canonical_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            _json_ready(value),
            sort_keys=True,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _finite_seconds(value: Any, label: str, *, allow_zero: bool = True) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise CampaignBudgetLedgerError(f"{label} must be numeric") from exc
    if not math.isfinite(result) or (result < 0.0 if allow_zero else result <= 0.0):
        requirement = "non-negative" if allow_zero else "positive"
        raise CampaignBudgetLedgerError(f"{label} must be finite and {requirement}")
    return result


class CampaignBudgetLedger:
    """Read, reserve, and settle bounded campaign budget in one JSON ledger."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).expanduser().resolve()

    @classmethod
    def create(
        cls,
        path: str | Path,
        *,
        campaign_id: str,
        total_budget_seconds: float,
        source_hash: str,
        plan_hash: str,
        claim_boundary: str,
        initial_consumed_seconds: float = 0.0,
        initial_attempts: Sequence[Mapping[str, Any]] = (),
    ) -> "CampaignBudgetLedger":
        ledger = cls(path)
        if ledger.path.exists():
            raise CampaignBudgetLedgerError(
                f"refusing to replace existing campaign ledger: {ledger.path}"
            )
        total = _finite_seconds(total_budget_seconds, "total_budget_seconds", allow_zero=False)
        consumed = _finite_seconds(initial_consumed_seconds, "initial_consumed_seconds")
        if consumed > total:
            raise CampaignBudgetLedgerError(
                "initial consumed budget exceeds total campaign budget"
            )
        if not str(campaign_id).strip() or not str(source_hash).strip():
            raise CampaignBudgetLedgerError("campaign identity and source hash are required")
        if not str(plan_hash).strip() or not str(claim_boundary).strip():
            raise CampaignBudgetLedgerError("plan hash and claim boundary are required")
        attempts = [_json_ready(dict(item)) for item in initial_attempts]
        payload = {
            "schema": CAMPAIGN_BUDGET_LEDGER_SCHEMA,
            "campaign_id": str(campaign_id),
            "total_budget_seconds": total,
            "consumed_seconds": consumed,
            "reserved_seconds": 0.0,
            "released_seconds": 0.0,
            "source_hash": str(source_hash),
            "plan_hash": str(plan_hash),
            "claim_boundary": str(claim_boundary),
            "created_at_utc": _utc_now(),
            "attempts": attempts,
            "reservations": {},
            "events": [
                {
                    "event": "ledger_created",
                    "at_utc": _utc_now(),
                    "initial_consumed_seconds": consumed,
                }
            ],
        }
        ledger._write(payload)
        return ledger

    def read(self) -> Mapping[str, Any]:
        if not self.path.is_file():
            raise CampaignBudgetLedgerError(f"campaign ledger is missing: {self.path}")
        try:
            payload = json.loads(
                self.path.read_text(encoding="utf-8"),
                parse_constant=lambda value: (_ for _ in ()).throw(
                    ValueError(f"non-finite JSON constant: {value}")
                ),
            )
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise CampaignBudgetLedgerError(
                f"campaign ledger is unreadable: {self.path}"
            ) from exc
        if not isinstance(payload, Mapping):
            raise CampaignBudgetLedgerError("campaign ledger root must be an object")
        self._validate(payload)
        return payload

    def remaining_seconds(self) -> float:
        payload = self.read()
        return float(
            payload["total_budget_seconds"]
            - payload["consumed_seconds"]
            - payload["reserved_seconds"]
        )

    def start_attempt(
        self,
        *,
        attempt_id: str,
        output_root: str | Path,
        seed_namespace: Mapping[str, Any],
        source_hash: str,
        plan_hash: str,
    ) -> None:
        payload = dict(self.read())
        self._check_binding(payload, source_hash=source_hash, plan_hash=plan_hash)
        attempt_id = str(attempt_id).strip()
        output_root = str(Path(output_root).expanduser().resolve())
        if not attempt_id or not output_root:
            raise CampaignBudgetLedgerError("attempt_id and output_root are required")
        attempts = list(payload["attempts"])
        if any(str(item.get("attempt_id")) == attempt_id for item in attempts):
            raise CampaignBudgetLedgerError(f"attempt already exists: {attempt_id}")
        if any(str(item.get("output_root")) == output_root for item in attempts):
            raise CampaignBudgetLedgerError(f"output root already exists in ledger: {output_root}")
        attempts.append(
            {
                "attempt_id": attempt_id,
                "output_root": output_root,
                "status": "running",
                "started_at_utc": _utc_now(),
                "started_monotonic": time.monotonic(),
                "source_hash": str(source_hash),
                "plan_hash": str(plan_hash),
                "seed_namespace": _json_ready(seed_namespace),
                "reserved_seconds": 0.0,
                "consumed_seconds": 0.0,
                "released_seconds": 0.0,
                "arms": [],
            }
        )
        payload["attempts"] = attempts
        payload["events"].append(
            {
                "event": "attempt_started",
                "attempt_id": attempt_id,
                "output_root": output_root,
                "at_utc": _utc_now(),
            }
        )
        self._write(payload)

    def reserve_chunk(
        self,
        *,
        attempt_id: str,
        arm: str,
        chunk_index: int,
        reserve_seconds: float,
        output_root: str | Path,
        seed_namespace: Mapping[str, Any],
        source_hash: str,
        plan_hash: str,
    ) -> str:
        payload = dict(self.read())
        self._check_binding(payload, source_hash=source_hash, plan_hash=plan_hash)
        reserve = _finite_seconds(reserve_seconds, "reserve_seconds", allow_zero=False)
        attempt_id = str(attempt_id).strip()
        arm = str(arm).strip()
        if not attempt_id or not arm or int(chunk_index) < 0:
            raise CampaignBudgetLedgerError("attempt, arm, and chunk identity are invalid")
        attempts = list(payload["attempts"])
        attempt = next((item for item in attempts if item.get("attempt_id") == attempt_id), None)
        if attempt is None:
            raise CampaignBudgetLedgerError(f"attempt is not registered: {attempt_id}")
        expected_root = str(Path(output_root).expanduser().resolve())
        if str(attempt.get("output_root")) != expected_root:
            raise CampaignBudgetLedgerError("reservation output root does not match attempt")
        reservation_id = f"{attempt_id}:{arm}:{int(chunk_index)}"
        reservations = dict(payload["reservations"])
        if reservation_id in reservations:
            raise CampaignBudgetLedgerError(f"chunk reservation already exists: {reservation_id}")
        remaining = self.remaining_seconds()
        if reserve > remaining + 1e-9:
            raise CampaignBudgetLedgerError(
                f"reservation exceeds remaining campaign budget: {reserve:g}s > {remaining:g}s"
            )
        reservations[reservation_id] = {
            "reservation_id": reservation_id,
            "attempt_id": attempt_id,
            "arm": arm,
            "chunk_index": int(chunk_index),
            "output_root": expected_root,
            "reserved_seconds": reserve,
            "status": "reserved",
            "reserved_at_utc": _utc_now(),
            "reserved_monotonic": time.monotonic(),
            "seed_namespace": _json_ready(seed_namespace),
            "source_hash": str(source_hash),
            "plan_hash": str(plan_hash),
        }
        payload["reservations"] = reservations
        payload["reserved_seconds"] = float(payload["reserved_seconds"]) + reserve
        attempt["reserved_seconds"] = float(attempt["reserved_seconds"]) + reserve
        payload["events"].append(
            {
                "event": "chunk_reserved",
                "reservation_id": reservation_id,
                "at_utc": _utc_now(),
                "remaining_seconds": self._remaining_from_payload(payload),
            }
        )
        self._write(payload)
        return reservation_id

    def settle_arm(
        self,
        *,
        attempt_id: str,
        arm: str,
        measured_seconds: float,
        status: str,
        failure_class: str | None = None,
        repair: str | None = None,
    ) -> Mapping[str, Any]:
        payload = dict(self.read())
        measured = _finite_seconds(measured_seconds, "measured_seconds")
        attempt_id = str(attempt_id).strip()
        arm = str(arm).strip()
        reservations = dict(payload["reservations"])
        active = [
            item
            for item in reservations.values()
            if item.get("attempt_id") == attempt_id
            and item.get("arm") == arm
            and item.get("status") == "reserved"
        ]
        reserved = sum(float(item["reserved_seconds"]) for item in active)
        if measured > reserved + 1e-9 and active:
            payload["events"].append(
                {
                    "event": "budget_overrun",
                    "attempt_id": attempt_id,
                    "arm": arm,
                    "at_utc": _utc_now(),
                    "measured_seconds": measured,
                    "reserved_seconds": reserved,
                }
            )
        released = max(0.0, reserved - measured)
        payload["reserved_seconds"] = float(payload["reserved_seconds"]) - reserved
        payload["consumed_seconds"] = float(payload["consumed_seconds"]) + measured
        payload["released_seconds"] = float(payload["released_seconds"]) + released
        for item in active:
            item["status"] = "settled"
            item["settled_at_utc"] = _utc_now()
            item["measured_seconds_allocation"] = (
                measured * float(item["reserved_seconds"]) / reserved if reserved else 0.0
            )
            item["released_seconds"] = (
                float(item["reserved_seconds"]) - item["measured_seconds_allocation"]
            )
            reservations[item["reservation_id"]] = item
        attempts = list(payload["attempts"])
        attempt = next((item for item in attempts if item.get("attempt_id") == attempt_id), None)
        if attempt is None:
            raise CampaignBudgetLedgerError(f"attempt is not registered: {attempt_id}")
        attempt["consumed_seconds"] = float(attempt["consumed_seconds"]) + measured
        attempt["released_seconds"] = float(attempt["released_seconds"]) + released
        attempt["arms"].append(
            {
                "arm": arm,
                "status": str(status),
                "measured_seconds": measured,
                "reserved_seconds": reserved,
                "released_seconds": released,
                "failure_class": failure_class,
                "repair": repair,
                "settled_at_utc": _utc_now(),
            }
        )
        payload["reservations"] = reservations
        payload["attempts"] = attempts
        payload["events"].append(
            {
                "event": "arm_settled",
                "attempt_id": attempt_id,
                "arm": arm,
                "status": str(status),
                "at_utc": _utc_now(),
                "measured_seconds": measured,
                "reserved_seconds": reserved,
                "released_seconds": released,
                "remaining_seconds": self._remaining_from_payload(payload),
            }
        )
        self._write(payload)
        return {
            "attempt_id": attempt_id,
            "arm": arm,
            "measured_seconds": measured,
            "reserved_seconds": reserved,
            "released_seconds": released,
            "remaining_seconds": self._remaining_from_payload(payload),
        }

    def finish_attempt(
        self,
        *,
        attempt_id: str,
        status: str,
        failure_class: str | None = None,
    ) -> None:
        payload = dict(self.read())
        attempts = list(payload["attempts"])
        attempt = next((item for item in attempts if item.get("attempt_id") == attempt_id), None)
        if attempt is None:
            raise CampaignBudgetLedgerError(f"attempt is not registered: {attempt_id}")
        active = [
            item
            for item in payload["reservations"].values()
            if item.get("attempt_id") == attempt_id and item.get("status") == "reserved"
        ]
        if active:
            raise CampaignBudgetLedgerError(
                f"cannot finish attempt with active reservations: {attempt_id}"
            )
        attempt["status"] = str(status)
        attempt["failure_class"] = failure_class
        attempt["ended_at_utc"] = _utc_now()
        attempt["elapsed_wall_seconds"] = max(
            0.0, time.monotonic() - float(attempt["started_monotonic"])
        )
        payload["attempts"] = attempts
        payload["events"].append(
            {
                "event": "attempt_finished",
                "attempt_id": attempt_id,
                "status": str(status),
                "failure_class": failure_class,
                "at_utc": _utc_now(),
                "remaining_seconds": self._remaining_from_payload(payload),
            }
        )
        self._write(payload)

    def _check_binding(self, payload: Mapping[str, Any], *, source_hash: str, plan_hash: str) -> None:
        if str(payload.get("source_hash")) != str(source_hash):
            raise CampaignBudgetLedgerError("campaign source hash changed")
        if str(payload.get("plan_hash")) != str(plan_hash):
            raise CampaignBudgetLedgerError("campaign plan hash changed")

    @staticmethod
    def _remaining_from_payload(payload: Mapping[str, Any]) -> float:
        return float(
            payload["total_budget_seconds"]
            - payload["consumed_seconds"]
            - payload["reserved_seconds"]
        )

    @staticmethod
    def _validate(payload: Mapping[str, Any]) -> None:
        if payload.get("schema") != CAMPAIGN_BUDGET_LEDGER_SCHEMA:
            raise CampaignBudgetLedgerError("campaign ledger schema mismatch")
        for field in (
            "campaign_id",
            "source_hash",
            "plan_hash",
            "claim_boundary",
            "created_at_utc",
        ):
            if not str(payload.get(field, "")).strip():
                raise CampaignBudgetLedgerError(f"campaign ledger field is missing: {field}")
        total = _finite_seconds(payload.get("total_budget_seconds"), "total_budget_seconds", allow_zero=False)
        consumed = _finite_seconds(payload.get("consumed_seconds"), "consumed_seconds")
        reserved = _finite_seconds(payload.get("reserved_seconds"), "reserved_seconds")
        released = _finite_seconds(payload.get("released_seconds"), "released_seconds")
        if consumed + reserved > total + 1e-9:
            raise CampaignBudgetLedgerError("campaign ledger budget is overcommitted")
        if released > consumed + reserved + total:
            raise CampaignBudgetLedgerError("campaign ledger released budget is invalid")
        if not isinstance(payload.get("attempts"), list):
            raise CampaignBudgetLedgerError("campaign ledger attempts must be a list")
        if not isinstance(payload.get("reservations"), Mapping):
            raise CampaignBudgetLedgerError("campaign ledger reservations must be an object")
        if not isinstance(payload.get("events"), list):
            raise CampaignBudgetLedgerError("campaign ledger events must be a list")

    def _write(self, payload: Mapping[str, Any]) -> None:
        self._validate(payload)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        data = _canonical_bytes(payload)
        with tempfile.NamedTemporaryFile(
            "wb",
            dir=str(self.path.parent),
            prefix=f".{self.path.name}.",
            delete=False,
        ) as handle:
            handle.write(data)
            temporary = Path(handle.name)
        try:
            os.replace(temporary, self.path)
        finally:
            if temporary.exists():
                temporary.unlink()

    def checksum(self) -> str:
        return hashlib.sha256(self.path.read_bytes()).hexdigest()
