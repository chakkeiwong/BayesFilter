# LGSSM NeuTra Gap Closure Phase 4 - HMC Tuning Admission

Date: 2026-07-15  
Status: `COMPLETE_BLOCKED_NO_ADMISSION`  
Parent: `docs/plans/bayesfilter-lgssm-neutra-scientific-gap-closure-plan-2026-07-15.md`

## Objective

Independently tune a fixed HMC kernel for each frozen candidate and admit only
candidates that pass a fresh 1,000-result modern rank/folded split R-hat gate.

## Entry Conditions

- Both Phase 3 candidate rows pass exact identity and cross-device parity.
- The Phase 3 HMC seed ledger is immutable.
- CPU execution deliberately hides CUDA before TensorFlow import.

## Tuning Procedure

For each candidate, use four chains in one CPU/XLA batch, float64, 10 leapfrog
steps, and deterministic dispersed initial states equal to the canonical
Phase 3 `[4,18]` probe. These starts are near latent zero but not identical;
they expose immobility more readily than four identical zeros and are a fixed
campaign choice, not a tuned parameter.

Run 64 retained results after 128 burn-in steps at primary step sizes
`(0.025, 0.05, 0.1, 0.2, 0.4)`. Derive each grid seed by adding
`10,000 + grid_index` to the frozen probe-seed second component. Reject any
nonfinite, target-status-invalid, unmoved, or energy-error-divergent probe.
Among healthy probes in acceptance band `[0.60,0.90]`, nominate the one closest
to 0.75, then grid order.

Verify the nominee with a disjoint frozen seed, 1,000 retained results per
chain after 1,000 burn-in. Compute rank-normalized split and folded
rank-normalized split R-hat in raw model coordinates. Admission requires all
health gates and maximum R-hat `<=1.01`; acceptance alone cannot admit.

If primary admission fails only through finite acceptance nomination or modern
R-hat, run one repair grid `(0.0125,0.025,0.05,0.1,0.2,0.4,0.8)` with its
disjoint seeds and repeat one verification. Any nonfinite/status/identity
failure is a hard veto and cannot trigger the grid repair.

## Required Artifacts And Checks

- per-grid health/acceptance rows without raw sample arrays;
- TensorFlow-serialized verification latent and raw draws with hashes;
- selected fixed-kernel payload/hash, or complete rejection reason;
- exact target/artifact/transport/adapter identities;
- R-hat definition, per-parameter rank/folded values, max value, and threshold;
- test that high acceptance cannot bypass the R-hat gate;
- no NumPy or legacy HMC import in the active route.

## Evidence Contract

Phase 4 admission establishes a viable fixed HMC candidate under a 1,000-draw
convergence screen. It does not establish confirmatory convergence, bulk/tail
ESS, posterior agreement, recovery, or superiority.

## Forbidden Claims And Actions

No adaptive or unplanned grid, leapfrog change, target change, non-JIT fallback,
post-admission retuning, seed reuse, or selection using training loss. Do not
rank admitted candidates by acceptance, runtime, or R-hat magnitude.

## Handoff Conditions

Process both Phase 3 candidates, freeze all admission decisions and kernel
hashes, then draft Phase 5. At least one admitted candidate is required to
continue. A rejected candidate does not stop another candidate.

## Stop Conditions

Stop for zero admitted candidates after the declared repair, common target or
sampler invalidity, corrupted artifacts, or the 6-hour tuning budget. Local
serialization/process/XLA defects trigger focused repair within the same
contract.

## Suitability Review

The procedure separates acceptance nomination from convergence admission,
uses fresh disjoint seeds, checks the actual raw parameter coordinates, and
predeclares its only repair. Verdict: `SUITABLE_TO_EXECUTE`.
