# FAB+IAF campaign terminal checkpoint

Active checkout: `/tmp/BayesFilter-neutra-fab-20260925`, branch `main`.
The shared `/home/ubuntu/python/BayesFilter` checkout remains on
`preserve/shared-main-before-fab-20260926` with unrelated dirty changes.
Do not reset, clean or restore the shared checkout.

Owner request: commit/merge/push and test whether FAB plus the canonical IAF
improves whitening while covering modes, training only. Integration and the
bounded execution are recorded in the terminal result:
`bayesfilter-fab-iaf-training-results-2026-09-27.md`.
Governing plan: `bayesfilter-fab-iaf-training-campaign-2026-09-26.md`.

Execution has STOPPED; both queues have finished and no campaign worker remains.
Two q20 pairs reached 240 updates; FAB seed 0 remains at 96, while its RKL arm
reached 240. It must not be included as a matched comparison. All saved maps
have complete finite 1,000-point probes and 4,096-point coverage checks.
The 11 scientific/prefix result packages pass terminal hash, source, pairing,
update-count, finite, precision, XLA and memory-growth checks.

Result: no map meets the desired joint whitening/coverage goal. In completed
q20 pairs, FAB score RMS is 196.5/147.2 versus RKL 7.01/6.39. The known
three-mode control shows FAB visits all three regions while RKL collapses,
but FAB geometry remains poor. Independent q20 prior-bank ESS is only 20--22;
all-mode coverage is unverified. These are descriptive, not statistical
rankings. No clipping or scale-saturation signal was recorded. Terminal AIS
ESS around 1.3/32 and short training are repair hypotheses. Do not promote a
new default or claim HMC/posterior readiness.

Budget: q20 plus pricing charged 15,861.008 of 16,200 worker seconds; 338.992
remain. Smokes/control charged 161.995 of a separate 600 seconds. Finishing
seed-0 FAB needs about 1,625 seconds including cached diagnostics. Do not
silently relaunch under the exhausted effective allocation.

Evidence root: `docs/plans/artifacts/neutra-fab-iaf-training-2026-09-26`.
Read `summary.json` for tables/checks/accounting. Full evidence is also bundled
in `campaign-evidence.tar.gz` with a SHA-256 inventory. Latest source integration
before terminal reporting: `52888d771` (main and origin/main); later commits
record the report, audit script and evidence.

Next research decision: complete replication if desired, then calibrate AIS
exploration and continued training with stronger independent q20 coverage
checks. A proposed FAB-to-RKL refinement has not been implemented or tested.
Read the terminal result before planning; do not repeat the already repaired
serialization, importance denominator, width or timing investigations.
