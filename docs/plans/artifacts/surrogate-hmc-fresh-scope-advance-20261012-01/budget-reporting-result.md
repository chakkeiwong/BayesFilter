# Master reporting repair result, 2026-10-13

The master now keeps compute time and attempt slots separate. At an attempt cap
it returns the true remaining seconds and reports ATTEMPT_LIMIT_REACHED. Both
run and advance enforce both bounds before launching. A phase timeout is
classified as a time limit rather than a rejected scientific candidate.
Completed tuning is retained if mechanics cannot start.

Validation: 128 CPU-only tests passed in 0.40 seconds in the full master test
module. CUDA_VISIBLE_DEVICES=-1 and BAYESFILTER_PRELOAD_CUSTOM_OP=0 were set
before imports; no GPU time was used. Exact command/environment and wrapper
wall time are in budget-reporting-cpu.json; full output in budget-reporting-cpu.log.
Coverage includes both launch boundaries, exhaustion between phases, failure
charging, unfinished reservations, malformed limits and timeout with/without
a partial worker result. Existing fake ledgers now include valid attempt fields.

The repair changes orchestration/reporting only, not scientific runtime,
filter tuning, force source, accepted results or limits. Completed scopes
retain their numerical source identity. Historical advance.json files remain
unchanged. Their erroneous zero-time or generic repair statuses are interpreted
alongside the authoritative ledger and new terminal next-phase-plan.json.

Skeptical post-run review: removing the count gate alone would have allowed
extra attempts; the tested repair retains it at both launch boundaries.
There is one sequential driver, as in the campaign. This is not a concurrent
multi-supervisor ledger and does not add a locking/launch-token mechanism.
The observed 650-second timeout had 0.328 seconds of termination/recording
overhead, still within the 36000-second campaign. At most 11.009 seconds remain.

| Decision | Primary criterion | Veto | Uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Retain reporting repair |128 orchestration tests pass|Both limits enforced|Fake workers do not validate numerical kernels|Update terminal master state|Scientific or sampler improvement|

| Inference status | Finding |
|---|---|
| Hard veto screen |No new GPU run; real compute limit prevents continuation|
| Statistically supported ranking |N/A; deterministic harness repair|
| Descriptive differences |Corrected remaining-time and stop-reason reports|
| Default readiness |Orchestration repair only; numerical defaults unchanged|
| Next evidence needed |Remaining numerical and posterior phases under adequate compute|

The predator timeout preserves full candidate 0 and 1 JSON files and per-seed
precision checks. Both are finite and pass the one-percent arithmetic screen;
maximum scaled discrepancies are 0.0348313% and 0.0351927%. Candidate 2 is incomplete
and no full filter-selection.json exists. These are viable partial candidates,
not a final selection or HMC authority. The original discovery used a
gitignore-aware search and missed these ignored result files; explicit
artifact paths and parsed JSON resolved that reporting mistake. No tuning
persistence change was needed or made.
