"""Shared host-side chunk seeds, deadlines and attempted-work accounting."""
from __future__ import annotations

import hashlib
import json
import time


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


def before_numerical_chunk(runtime, work, candidate, *, count: int, chains: int, index: int):
    from bayesfilter.inference.hmc_candidate_set_tuning import HMCBudgetExhausted
    if runtime._deadline is not None and time.monotonic() >= runtime._deadline:
        raise HMCBudgetExhausted("wall-time cap reached between numerical chunks")
    charge = getattr(runtime, "_charge_chunk", None)
    if charge is not None:
        charge(work, candidate, transitions=count * chains, chunk_index=index)
