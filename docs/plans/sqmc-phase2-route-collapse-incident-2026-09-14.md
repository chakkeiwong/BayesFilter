---
title: SQMC Phase 2.2–3–4 Campaign Invalidated — Route Dispatch Failure
date: 2026-09-14
status: incident
---

## Summary

The Phase 2.2–3–4 SQMC campaign launched 2026-09-13 23:50 generated **worthless
artifacts**: all three "per-route" tuning runs produced bit-identical results
because the runner ignored `--routes` in pilot mode and retuned `iid_dual_cap`
three times under different output labels.

**Root cause:** Line 721 of `run_sqmc_tuning.py` hardcoded
`routes_to_run = ['iid_dual_cap']` inside `if args.mode == 'pilot'`, overriding
`--routes` entirely. The campaign wrapper passed
`--mode pilot --routes <route>`, so every "route" silently ran the same route.

**Detection:** Manual log inspection found bit-identical L2 errors (1.4810) and
cosine similarities across three supposedly distinct routes. This class of failure
should be caught by the harness, not by eyeballing.

## Campaign Status

- **Phase 2.2 (tune 3 remaining routes):** INVALID, route collapse
- **Phase 3 (paired comparison):** NOT STARTED
- **Phase 4 (interpretation):** NOT STARTED

All accumulated wall time (~2.5 hours) was wasted. Campaign stopped at
~2026-09-14 01:20.

## Repairs Applied

1. **Dispatch fix:** `--mode` now selects seed budget only, never overrides an
   explicit `--routes`. Default routes still differ by mode.

2. **Path anchor fix:** Artifact paths now anchor to repo root, not caller cwd.
   Previous runs from `docs/benchmarks/` created
   `docs/benchmarks/docs/tuning/...` instead of `docs/tuning/...`.

3. **Regression check:** New `check_routes_are_distinct.py` evaluates all four
   routes at identical controls and asserts pairwise distinct scores. Also
   reports input point-set hashes to distinguish dispatch failure (identical
   inputs) from genuine equivalence (different inputs, same output). Exit 0 =
   routes are distinct and a comparison is meaningful.

## Next Action

Rerun Phase 2.2 with the fixed runner after verifying one route produces the
expected artifact path and route field. Then continue to Phase 3.

## Lessons

- Silent route collapse is not a new failure mode — the original KDM score
  campaign had a similar "all candidates called the same code" defect that
  produced a leaderboard of one entry repeated. This incident log follows that
  precedent.

- Mode switches that override explicit arguments are a recurring trap. The fix
  makes `--mode` orthogonal to `--routes`: mode selects the seed budget, routes
  select what to tune.

- Bit-identical floating-point results across supposedly different algorithms are
  a smoke signal, not a coincidence. Two routes agreeing to 1e-10 is plausible;
  three routes producing `L2=1.4810458808595357` verbatim is a dispatch failure.
