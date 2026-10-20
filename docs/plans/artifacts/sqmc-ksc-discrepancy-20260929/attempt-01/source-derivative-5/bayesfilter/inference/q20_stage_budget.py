"""Host-side spending checks for replayable q20 numerical chunks."""
from __future__ import annotations

import math
import time

from bayesfilter.runtime.durable_tensor_checkpoint import DurableTensorCheckpoint


class StageBudgetPause(RuntimeError):
    """The next work unit is unfunded; saved work remains inconclusive."""


class BudgetedCheckpoint(DurableTensorCheckpoint):
    def __init__(self, root, identity, *, deadline=None, chunk_seconds=0., safety_factor=1.):
        if not math.isfinite(chunk_seconds) or chunk_seconds < 0:
            raise ValueError("chunk forecast must be finite and nonnegative")
        if not math.isfinite(safety_factor) or safety_factor < 1:
            raise ValueError("chunk safety factor must be finite and at least one")
        super().__init__(root, identity)
        self.deadline = deadline
        self.chunk_seconds = chunk_seconds
        self.safety_factor = safety_factor

    def run(self, key, inputs, compute):
        # Replayed chunks still validate their identities/checksums, but do not
        # need another numerical-work reservation.
        if self.contains(key):
            return super().run(key, inputs, compute)
        if self.deadline is not None and time.monotonic() + self.chunk_seconds >= self.deadline:
            raise StageBudgetPause("next numerical chunk exceeds remaining stage allocation")
        started = time.monotonic()
        result = super().run(key, inputs, compute)
        self.chunk_seconds = max(self.chunk_seconds, self.safety_factor * (time.monotonic()-started))
        return result


def repair_remaining(campaign):
    """One shared repair hold, debited by actual failed infrastructure time."""
    failures = {"failed", "timed_out", "interrupted", "interrupted_upper_charge"}
    spent = campaign.state.get("prior_infrastructure_repair_spent_seconds", 0.) + sum(
        a.get("elapsed_seconds", 0.) for a in campaign.state["attempts"]
        if a["status"] in failures and a.get("failure_classification") not in
        {"resource_unavailable", "budget_pause", "allocation_exhausted"})
    return max(0., campaign.config["budget"]["repair_allocation_seconds"] - spent)


def allocate_stage(campaign, name, ceiling, *, protected_seconds=0.):
    """Return a cumulative allowance; retries cannot renew its original limit."""
    spent = sum(a.get("elapsed_seconds", 0.) for a in campaign.state["attempts"] if a["stage"] == name)
    original = campaign.state.get("stage_limits", {}).get(name, ceiling)
    if name in campaign.state["stages"]:
        return original
    available = min(max(0., original-spent), max(0., campaign.remaining()-protected_seconds))
    if available <= campaign.config["execution"]["termination_grace_seconds"]:
        raise StageBudgetPause("no funded work remains for " + name)
    return spent + available
