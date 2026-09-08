# Phase 6 Exact SMC-U and Independent Mutation Replication Subplan

Program: `docs/plans/bayesfilter-ssl-lstm-q20-particle-authority-master-program-2026-08-25.md`  
Status: `REPAIR_TRIGGERED_ACCEPTANCE_RECEIPT`  
Budget cap: `14400 s` transferred from unused completed-phase allocation  
Output root:
`docs/plans/artifacts/ssl-lstm-q20-particle-authority-master-2026-08-25/phase6`

## Objective

Test the actual resampling-plus-mutation normalizing-constant estimator on a
known target, then measure independent-seed variability of the q=20 repaired
mutation branch. This phase addresses the two unresolved authority obligations:
the complete SMC-U bookkeeping identity and seed variability. It does not turn
finite mode occupancy into a discovery theorem.

## Entry gate

Phase 1 contracts, the Phase 2 mutation branch, and downstream role checks have
passed their hard finite/status gates. The mutation implementation test must
also pass its static batch and symmetry checks before the campaign runs.

## Evidence contract

| Field | Predeclared choice |
|---|---|
| Scientific question | Does the implemented resampling/mutation estimator recover a known normalizer, and how variable are q=20 repaired runs? |
| Exact comparator | A normalized one-dimensional Gaussian target with known unnormalized mass `Z=2.5`, using the same fixed-beta/resampling/MH sequence |
| Primary fixture criterion | Across `64` independent fixture replicates with `N=128`, the mean estimator lies within `max(0.15, 4*MCSE)` of `2.5`, all estimates are finite, and all mutation transition residuals are zero |
| q=20 criterion | Each of three fresh N=100 mutation runs has finite/status-valid rows, frozen protocol hash, finite mass, zero invalid proposals, and retained raw mass/mode receipts |
| Vetoes | exact fixture failure, non-finite/status row, hash mismatch, invalid proposal, or overwritten artifact |
| Explanatory diagnostics | acceptance, ESS, mass, signed occupancy, root counts, runtime; no arm ranking from three seeds |
| Nonclaims | no finite-run exhaustive mode discovery, posterior correctness, IID whitening, HMC readiness, or default promotion |
| Artifact | fixture receipt, per-seed pilot roots, aggregate MCSE table, manifest, result, and repair note |

## Numeric/default audit

`N=128`, `64` fixture replicates, three q=20 seeds, one mutation step, and
scale `0.05` are measured-budget hypotheses inherited from the N=100 repair,
not promoted defaults. The first diagnostic is the exact normalizer fixture;
failure triggers a bookkeeping repair before any q=20 interpretation.

## Execution

The fixture runner is
`docs/benchmarks/run_ssl_lstm_q20_particle_authority_smc_u_replication_2026_08_25.py`.
It is CPU-hidden, TensorFlow/XLA, NumPy-free, fail-closed on output paths, and
launches three fresh q=20 pilot commands only after the fixture passes.

## Executed receipts and refresh

The first execution completed the exact fixture and six independent N=100
q=20 runs. The exact fixture passed with mean normalizer `2.4812129`, MCSE
`0.0243`, absolute error `0.0188`, and zero transition-density residuals.
All six N=100 pilots passed finite/status/hash/support gates, but their mode
occupancy spread triggered the declared particle-count repair.

The repaired N=300 scope completed three independent seeds (`1701`, `1801`,
`1901`) with the same frozen beta schedule, defensive mixture, and symmetric
random-walk mutation. All three passed the hard receipts. Across those runs,
weighted negative occupancy had mean `0.4613` (MCSE `0.0303`, range
`0.4013--0.4988`), ESS fraction had mean `0.9650` (MCSE `0.0061`), and
log-mass had mean `-34.3579` (MCSE `0.0830`). Mutation acceptance ranged
`0.1583--0.1817`; invalid proposals and transition-density residuals were
zero. These are descriptive candidate diagnostics, not a finite-run mode
discovery or authority theorem.

The phase result and decision/inference ledgers are recorded in
`...-phase6-result-2026-08-25.md`. The next subplan is a fresh target-specific
NeuTra retuning screen using the N=300 bank; it remains outside HMC and cannot
promote M0 without an explicit unnormalized-law and mode-evidence gate.
