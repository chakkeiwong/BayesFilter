# M18: numerical parity, current-source integration and official book

Refreshed after M17's terminal reconciliation. Engineering preparation and source review
are already recorded in the M18 engineering note. The phase ceilings remain
8000 CPU-reference and 2000 GPU worker seconds, including that preparation.
The September 21 engineering note reallocates 2000 unused campaign CPU seconds
to finish the broad regression continuation; total authorization is unchanged.
M17's source/evidence audit is complete; its result now informs this design.

## Question and evidence contract

Does the scoped checkpoint-hash optimization preserve the numerical procedure,
and do the current implementation, tests, capability registry and official book
describe the same supported behavior? The numerical comparator changes only
the hash hook between generic normalization and JSON-native hashing, on one
identical frozen source with identical target, source closure, seeds and search.
Changing the source directory between arms would change stream identity and
would be an invalid comparator.

The primary numerical criterion is exact equality of candidate settings,
states, verified membership, work order, starts, seeds, samples, transition
traces, analyses and subsequent retained draws. Compare the complete numerical
projection; explicitly exclude elapsed-time and device-runtime metadata and
hashes containing that metadata. Preserve the original full evidence and
validate its checksums independently. Empty candidate sets, unfinished work,
hash discrepancies or lost corruption checks veto the optimization's parity
claim and trigger diagnosis. A numerical mismatch must be explained or the
optimization removed; a faster host microbenchmark alone cannot promote it.

Engineering evidence consists of affected mechanism/integration tests, replay
and tamper tests, independently verified evidence inventories, current-source
dependency reconciliation and a rebuilt official book. CPU tests are reference
exceptions with GPUs hidden. They cannot substitute for the matched GPU/XLA
comparison. No inference of posterior correctness, burn-in sufficiency, broad
default readiness or algorithmic speed superiority follows from these checks.

## Numerical comparison and budget

Use Gaussian and beta-binomial (7 successes, 12 trials with the declared prior)
targets, four dispersed starts, supplied identity geometry and L=(3,5,9).
Initial epsilon .5, domain [.005,3], five repairs per family, 64 pilot,
measurement and verification draws, eight discarded diagnostic transitions,
evidence rungs (1,2,4), total work 72 and repair reserve 12 are explicit small
mechanics fixtures. They test the hash intervention and durable public replay,
not automatic preparation. Pause after two work items and resume; preselect
the first verified member by identity, export/reload it, then discard 64 and
retain 128 draws on fresh explicit streams. No posterior accuracy is assessed.
Root seed 2026092198 has a case-specific `hash-parity` domain; both arms use
the same derived seed. Each of four GPU workers has a 400-second ceiling,
leaving 400 seconds for a localized repair within the phase GPU allocation.
The cap is a bounded engineering allowance informed by earlier prepared-path
searches, not a calibrated runtime guarantee. Run at most two concurrently
with verified memory growth and trusted GPU/XLA access.

The already executed host study compared 3160 saved JSON-native records and
196 MB of serialized input; every checksum matched. Five alternating timing
repeats were descriptive and did not measure end-to-end sampler speed.
Retain rehashing of every live record and all current corruption checks; do
not introduce an unchecked cache or change checkpoint cadence.

Within the CPU ceiling, preparation is bounded by 1800 seconds, affected
HMC/inference-validation tests by 4400, book build and inspection by 900,
and reconciliation/localized repair by 900. Actual unused costs carry forward.
Record every failed check and retry once. Use fresh output directories under
`artifacts/hmc-repair-master-2026-09-16/m18-r1/`, preserve test XML and logs,
and snapshot source before the GPU comparison. Exact resolved commands and
environment belong in each run record.

## Documentation, tests and remaining evidence

Run the affected HMC mechanism, public entry-point, whole-procedure,
checkpoint/reload, warmup/precision and validation-engine tests, including
the multi-model integrations added in earlier phases. Inventory actual target,
route, device and source coverage; a historical different source does not
become current-source evidence by relabeling it. Distinguish source differences
outside the numerical dependency closure from changes needing parity evidence.

Synchronize the API reference and validation README with the official
`docs/main.tex` chapter, including the conditional sequential test, preparation
recovery, metric policy, all-candidate retention and separate posterior checks.
Use measured M15--M17 denominators and power limits. Keep the absent exact
MacroFinance target/reference cell explicit. Record deferred learned-transport
training and other unexecuted coverage; a fixed nonlinear codec is not training.

The bibliography/source audit has checked the Gorinova paper and official code,
Pakman--Paninski paper and author truncated-Gaussian code, and Afshar--Domke
paper and supplement. The adjacent discontinuity chapter now distinguishes
continuous gradient kinks from potential jumps and frozen preprocessing from
within-chain adaptation. Preserve source anchors and the corrected retrieval
record. Build and install `docs/main.pdf`; inspect the changed rendered pages
and compile diagnostics. Preserve the prior PDF as the comparison baseline.

## Skeptical review to complete at the phase refresh

Already addressed: repeated generic normalization is a measured host cost,
not a sampler-speed claim; only JSON-copied records satisfy the specialization's
precondition; source-bound random streams require the same source in both
arms; timing-bearing hashes cannot be expected to match; corruption checks
must remain active. The source audit repaired an initially wrong paper URL
before using it, and the rejected file remains retrieval-failure evidence only.

At M17 close, record its substantive outcomes, any needed local repairs, exact
remaining budget and any source changes before accepting this draft. At M18
close, write a terminal result with decision/inference tables, refresh the
progress record to bounded completion, and list unresolved scientific evidence
separately from repaired engineering defects. A bounded completed program does
not require every difficult model to pass or every statistical question to be
settled.

## Completed skeptical refresh after M17

M17 completed all 21 matrix and four matched-reference fits, plus the six
conditional field repairs. Eleven matrix fits and three reference fits passed
the full posterior screen. Centered funnels returned empty sets, mixtures
hit warmup caps, and rotated/noncentered/CPU eight-schools fits hit retained
precision caps. All inventories, 3028 receipts and 3006 tensors passed integrity
checks. These outcomes do not change M18's exact numerical comparator, tuning
rules, posterior thresholds or scope. The exact MacroFinance reference remains
unavailable for that cell alone. Opening remaining budget is
82393.40737798327 CPU and 18362.589490781014 GPU worker seconds.

The broad regressions repaired stale test assumptions and one real lazy public
export: resolving the uncertainty-admission function unnecessarily imported
unrelated numerical modules. Direct routing now returns the same callable
without loading TensorFlow. The final frozen source is `m18-r1/source-r2`,
identity `bb4090ef90025ce9441a7dfb1b6d3e3a2116258b5c9416dfff291536f56606ac`.
It differs from preserved source-r1 only in that export. Relative to M17 the
only numerical change remains JSON-native checkpoint hashing. Both parity
arms use this identical source and harness; generic/native is the sole hook
intervention. Work IDs are ordinal and selected-member streams are explicitly
fixed, so timing-bearing receipt hashes cannot alter the compared streams.
Inspect the complete numerical projections and validate original evidence.

The source reader, required policy constructor arguments and fresh output
paths have been checked before launch. Run `run_hash_parity.py --source
<source-r2> --case <gaussian|beta_binomial> --arm <generic|native> --seconds 400`
on trusted GPU1 with memory growth, at most two workers. The full expanded
commands and environment are saved per arm. The 8000/2000-second ceilings fit
the remaining ledger; every regression failure/repair is charged. This design
passes the skeptical review for bounded engineering execution and book repair.
