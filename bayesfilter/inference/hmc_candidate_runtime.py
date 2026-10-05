"""Shared host-side chunk seeds, deadlines and attempted-work accounting."""
from __future__ import annotations

from contextlib import contextmanager
import hashlib
import json
import time


@contextmanager
def incremental_chunk_checkpoint(runtime):
    """Reuse persisted evidence only inside one native chunk's durable saves.

    Stage completion, exception handling and explicit writes remain full live
    integrity boundaries. Always restore the mode before control returns there.
    """
    previous = getattr(runtime, "_incremental_checkpoint", False)
    runtime._incremental_checkpoint = True
    try:
        yield
    finally:
        runtime._incremental_checkpoint = previous


@contextmanager
def grouped_incremental_chunk_checkpoint(runtime):
    """Defer repeated charge/row saves and flush one durable checkpoint.

    Replicated native batches charge every row before the call and persist all
    returned rows after it.  The controller still records each charge and row;
    this context only coalesces the checkpoint writes.  Its ``finally`` flush
    also covers a budget, serialization, or native interruption after a prefix
    was charged or persisted.
    """
    owner = getattr(getattr(runtime, "_checkpoint_callback", None), "__self__", None)
    previous = getattr(runtime, "_defer_checkpoint", False)
    previous_incremental = getattr(runtime, "_incremental_checkpoint", False)
    previous_owner = getattr(owner, "_defer_checkpoint", False) if owner is not None else False
    runtime._defer_checkpoint = True
    runtime._incremental_checkpoint = True
    if owner is not None:
        owner._defer_checkpoint = True
    try:
        yield
    finally:
        runtime._defer_checkpoint = previous
        if owner is not None:
            owner._defer_checkpoint = previous_owner
        try:
            if not previous and getattr(runtime, "_checkpoint_callback", None) is not None:
                runtime._checkpoint_callback()
        finally:
            runtime._incremental_checkpoint = previous_incremental


def chunk_seed(seed, index: int) -> tuple[int, int]:
    if index == 0:
        return tuple(seed)
    digest = hashlib.sha256(json.dumps([seed, index]).encode()).digest()
    return tuple(int.from_bytes(digest[i:i+4], "big") & 0x7fffffff for i in (0, 4))


def tuning_seed_inventory(evidence, partial=(), *, attempted_chunks=()) -> set[tuple[int, int]]:
    """Include charged attempts even when no completed trace was returned.

    ``attempted_chunks`` contains stage base seeds and chunk indices from the
    durable accounting record, so exporting a member preserves this inventory.
    """
    evidence = tuple(evidence)
    seeds = {tuple(row["seed"]) for row in evidence}
    for row in evidence:
        seeds.update(tuple(chunk["seed"]) for chunk in row.get("chunks", ()))
        if "attempted_seed" in row:
            seeds.add(tuple(row["attempted_seed"]))
    for chunks in partial:
        seeds.update(tuple(chunk["seed"]) for chunk in chunks)
    seeds.update(chunk_seed(seed, index) for seed, index in attempted_chunks)
    return seeds


def before_numerical_chunk(runtime, work, candidate, *, count: int, chains: int, index: int,
                           seed=None, trial_ordinal=None, trial_chunk_index=None):
    from bayesfilter.inference.hmc_candidate_set_tuning import HMCBudgetExhausted
    from bayesfilter.runtime.execution_budget import execution_budget_available
    if (not execution_budget_available() or
            runtime._deadline is not None and time.monotonic() >= runtime._deadline):
        raise HMCBudgetExhausted("wall-time cap reached between numerical chunks")
    charge = getattr(runtime, "_charge_chunk", None)
    if charge is not None:
        details = {} if seed is None else {"seed": tuple(seed), "trial_ordinal": trial_ordinal,
                                          "trial_chunk_index": trial_chunk_index}
        charge(work, candidate, transitions=count * chains, chunk_index=index, **details)
