# Randomized-HMC SGQF feasibility continuation — 15 October 2026

**Terminal, 16 October:** All eight roots qualified and were assessed. Five
passed posterior/reference checks; three deliveries failed, with tuning
membership preserved. The integrity audit and settlement passed. See the
[terminal result](bayesfilter-modern-hmc-randomized-terminal-2026-10-16.md).
The following original design and live notes are retained as history.

This is the next executable phase of
`bayesfilter-modern-hmc-implementation-plan-2026-10-07.md`. It follows the
terminal fixed-HMC continuation and uses the storage cleanup's recovered disk
capacity for the same program. It does not start a separate data-retention
project.

## Current question and evidence contract

The question is whether the frozen randomized-HMC SGQF consumer can execute
the public candidate-set lifecycle on the full predator–prey target under the
same qualification and posterior contract as fixed HMC. The baseline is the
immutable `full-model-source-e68cdf1c4` snapshot and
`designs-02/randomized_hmc.json`, with all eight declared roots and every
repair descendant retained in the denominator.

Acceptance/health verification determines tuning membership. R-hat, ESS, MCSE
and finite-target reference agreement are posterior diagnostics only; they
cannot reject, rank or repair a tuning candidate. A candidate failure is
preserved and does not suppress its peers. A changed source, target, geometry,
seed, policy, missing evidence or corrupt checkpoint is an implementation
veto. Exhausting the allocation is a resource-censored result, not scientific
rejection.

This phase cannot establish a method ranking, broad model validity, stopped
interval coverage, a new default or release readiness. Its bounded outcome is
useful only for the declared target, seed and frozen source.

## Verified prerequisites and skeptical audit

The 14 October continuation settled all eight fixed-HMC members: seven
delivered and one precision-limited, with all tuning identities retained. Its
inclusive charge was 31,838.804891 seconds from the 169,200-second reservation,
leaving 137,361.195109 worker-seconds in the new grant after the held local
reserve. The prior 1,579.592954 seconds are separate and are not reused here.

The frozen randomized design has eight roots `(L, epsilon)` equal to
`(1,1), (3,.5), (3,1), (7,.5), (7,1), (15,.5), (15,1), (31,.5)`, the
uniform-duration transition profile, fresh qualification streams, four-chain
posterior assessment and the unchanged 18-quantity reference screen. Existing
randomized prices show a minimum-count scenario of about 377,630 seconds,
which exceeds the available balance. Therefore this is explicitly a bounded
feasibility attempt, not a promised complete cohort.

The skeptical review passes after two preparation repairs: the preflight now
loads the same independent reference compatibility report as the worker and
validates the frozen search against the bound execution; the accounting below
clarifies that the local reserve was already held at the fixed-HMC settlement.
The design and source are frozen; the comparator
is not selected from the fixed-HMC outcome; the full root denominator and
repair allowance remain explicit; cost forecasts are not treated as vetoes;
the numerical supervisor has a single inclusive 137,000-second cap; and
unfinished roots/checkpoints will be settled without fabrication. The inherited summary reported 103 focused CPU tests, but its precise log has not been located; that count is not newly verified evidence. A trusted GPU probe selected idle GPU
1, with GPU memory growth required before TensorFlow initialization. The actual
source preflight must pass before numerical work.

## Allocation and execution

The existing 3,600 local worker-second reserve remains held for inspection and
terminal accounting; it is not subtracted again from the settled balance. The
randomized child receives at most 137,000 inclusive wall seconds, leaving
361.195109 seconds of the settled balance unallocated. This is a ceiling, not
a completion estimate. TensorFlow/TFP/XLA runs on GPU 1 with
`TF_FORCE_GPU_ALLOW_GROWTH=true` and two host threads. The numerical source is
read-only; all output is in the fresh
`.../priced-comparison-01/randomized-feasibility-01/` directory.

Run `preflight.py` from that directory under the trusted GPU environment before
launch. It must record the frozen eight-pair search, source and design hashes,
bound target/data/geometry identity, reference compatibility, verified GPU
memory growth, and `sampling_performed: false`. A failed preflight is a launch
stop and a local repair trigger. The immutable numerical consumer then runs
`scripts/run_modern_hmc_frozen_full_model.py --design
.../designs-02/randomized_hmc.json --output
.../randomized-feasibility-01/run --max-wall-seconds 137000` in the same GPU
environment. The output path must not exist at launch.

**Launch record, 15 October:** The preflight passed without sampling after
361.575 seconds. The numerical consumer launched as
`bayesfilter-modern-randomized-feasibility-20261015-01.service`, and its
receipt was captured at 14:22:57 UTC. A separate CPU-only read-only observer,
`bayesfilter-modern-randomized-monitor-20261015-01.service`, records progress
and terminal state; it has no numerical launch, retry or stop authority. The
launch receipt binds the source/design hashes, binding, data and geometry
identities, GPU 1, memory-growth policy and 137,000-second cap. The worker is
currently qualifying with eight primary candidates created and no candidate
yet verified; its checkpoint label `partial_budget` is interim while the
service is active, not a terminal resource disposition.

The source supervisor checkpoints qualification and posterior work and
finalizes every known disposition at an allocation stop. The launch service
does not retune, restart with another seed, alter count limits, or kill a
healthy long chunk for inactivity. Monitoring records progress and resource
contention separately. At terminal exit, verify source/input hashes, all eight
requested roots, created/verified/assessed/delivered counts, candidate failures,
checkpoint integrity, and inclusive elapsed cost exactly once.

## Decision table

| Decision | Primary criterion | Veto | Next action | Not concluded |
| --- | --- | --- | --- | --- |
| Launch bounded randomized-HMC feasibility | Actual-source preflight and public consumer | Source/identity mismatch or invalid GPU policy | Settle terminal cohort and refresh the master | Complete delivery, ranking, generality or release |

| Inference status | Required interpretation |
| --- | --- |
| Hard veto screen | Numerical/source/checkpoint failures are reported per affected identity |
| Statistically supported ranking | None from this one-target, one-seed bounded attempt |
| Descriptive evidence | Runtime, acceptance, retained counts and censoring only |
| Default-readiness | Not established |
| Next evidence | Terminal randomized audit, then feasible SNAPER/MAMS/classical cells and independent seeds |

