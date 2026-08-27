# Phase 6 Repair and Refresh Note

Status: `REPAIR_TRIGGERED_ACCEPTANCE_RECEIPT_DENOMINATOR`

Classify failures in this order: runner/harness, exact-fixture bookkeeping,
proposal support, mutation invariance, seed variability, and only then
scientific candidate quality. An exact-fixture failure blocks q=20 authority
interpretation until the estimator is repaired; it does not by itself reject
the broader particle-flow or NeuTra direction. A one-seed mode imbalance is a
repair trigger, never a continuation veto.

After the run, record each attempted seed, its receipt/hash, MCSE calculation,
and the refreshed next subplan. If the exact fixture passes but the q=20
replications remain variable, refresh a larger independent-seed campaign. If
the exact fixture fails after three focused repairs with no meaningful progress,
record the exact contradiction as a real blocker for this authority route and
stop the program under the master definition.

## Actual first execution

The exact algorithm fixture passed at 64 replicates/N=128: mean normalizer
`2.4812` for the known `2.5`, MCSE `0.0243`, absolute error `0.0188`, and zero
transition-symmetry residuals. The three q=20 N=100 mutation runs (seeds 1001,
1101, 1201) all passed hard receipts. Their weighted negative fractions were
`0.8550`, `0.4620`, and `0.3579`; ESS fractions were `0.7846`, `0.9709`, and
`0.9603`; and log-mass estimates were `-33.7369`, `-34.4274`, and `-34.1186`.
This spread is a seed-variability repair trigger. It is not evidence that the
exact fixture or the broader direction is false.

## Repair action

Run three additional independent N=100 seeds with the same frozen mutation
scope, then compute descriptive mean/MCSE/range tables. If variability remains
large, refresh a particle-count/tail-support scope rather than admitting M0.

## Six-seed result and next repair

The additional three seeds completed with the exact fixture still passing. Across
all six N=100 runs, acceptance ranged `0.1697--0.1797` (mean `0.1739`), ESS
ranged `0.7846--0.9845`, weighted negative fraction ranged `0.2581--0.8550`,
and log-mass ranged `-33.7369-- -34.5201` (mean `-34.2671`, sample MCSE
`0.1211`). The mode spread is too large for authority admission and remains
descriptive rather than a ranking. The next smallest discriminating repair is
one N=300 run with the same mutation scope and a fresh seed, followed by a
second N=300 run only if the first artifact is finite and complete.

The first N=300 run (seed 1701) completed in `1694.7 s` with ESS `0.9772`,
weighted negative fraction `0.4988`, unique signed roots `72/67`, log mass
`-34.4198`, and mutation acceptance `0.1583--0.1817`. This is descriptively
more balanced than the N=100 samples, but it is not a promotion result. The
second N=300 seed is therefore in scope as the declared replication repair.

The two N=300 runs completed in `1694.7 s` and `1603.4 s`. Their weighted
negative fractions were `0.4988` and `0.4838`, root counts were `72/67` and
`74/75`, ESS fractions were `0.9772` and `0.9596`, and log masses were
`-34.4198` and `-34.4603`. This is descriptively more stable than N=100, but
two seeds are still insufficient for an authority or default claim. A third
N=300 seed is the next bounded repair.

## Third N=300 repair and phase disposition

The third N=300 seed (`1901`) completed in `1660.5 s` (pilot wall time) with
finite/status-valid rows, a matching protocol hash, zero invalid proposals,
and zero transition-density residuals. Its ESS fraction was `0.9582`, weighted
negative occupancy was `0.4013`, root counts were `75/68`, log mass was
`-34.1937`, and mutation acceptance ranged `0.1608--0.1817`.

Across the three N=300 seeds, weighted negative occupancy was `0.4613` mean,
`0.0303` MCSE, range `0.4013--0.4988`; ESS was `0.9650` mean and `0.0061`
MCSE; log-mass MCSE was `0.0830`. The exact fixture remained unchanged and
passed. This repairs the immediate particle-count variability trigger, but it
does not establish exhaustive mode discovery, an unbiased q=20 normalizer, or
authority/default status. The phase therefore passes as a role-limited
candidate and refreshes Phase 7 for target-specific NeuTra retuning on the
N=300 bank. HMC remains out of scope.

## Post-phase audit repair

A source audit found that the mutation receipt divided accepted particle
transitions by `N * state_dimension` rather than by the number of particle
proposals `N * steps`. The symmetric transition and state values were not
changed, and acceptance was explanatory only, but every previously reported
acceptance rate is invalid by a factor of four. The pilot now records explicit
accepted/proposal counts and uses the particle-level denominator. Focused tests
pass. The exact fixture and a fresh N=300 replication must be rerun before the
Phase 6 result can be treated as complete; Phase 7 remains role-limited
historical downstream evidence and will not be reinterpreted from the repaired
receipt until that refresh is recorded.
