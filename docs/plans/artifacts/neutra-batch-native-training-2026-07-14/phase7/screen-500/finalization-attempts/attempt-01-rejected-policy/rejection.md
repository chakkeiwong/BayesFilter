# Finalization Attempt 01 Rejection

Date: 2026-07-14

Classification: implementation/policy failure in selection-admission tooling.

The first finalization result correctly read the fresh GPU artifacts and applied
the predeclared selection rule, but its validation path called
`campaign_seed_ledger()`. That accessor lazily imported the diagnostic parent
campaign, whose module imports NumPy. The numerical nomination is unchanged,
but this attempt is rejected because selection/admission logic must not depend
on NumPy, even indirectly.

Repair: expose the already-frozen smoke seeds as `SMOKE_SEEDS`, import only that
constant, add a subprocess regression proving finalizer import leaves both
NumPy and TensorFlow unloaded, and bind the finalizer source file hash into the
accepted result.

The preserved JSON files in this directory are historical failed-attempt
evidence only and must not be used as the active Phase 7 handoff.
