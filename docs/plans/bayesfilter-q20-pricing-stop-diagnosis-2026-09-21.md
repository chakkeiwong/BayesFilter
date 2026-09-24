# q20 pricing stop diagnosis, September 21 at 21:24 Shanghai

The campaign has not obtained an estimate and is not running. `campaign-02`
stopped at 17:58:54 with `ESTIMATION_BUDGET_PAUSED`. The settled balance is
175,624.1062574485 seconds (48.7845 hours), including 2,537.861876872681 diagnostic
seconds (42.30 minutes). The calendar deadline remains September 25 at 18:00
Shanghai. No new numerical work was launched for this inspection.

## What happened

Attempt 00008 finished the numerical ensemble-pricing calculation in 1,865.04
seconds and wrote `worker/data/result.json`. Its process cost was 1,872.98
seconds. The file contains all eight HMC prices, actual two-chart exchange
prices, training prices, reference prices and reporting prices.

After dispatch returned, `q20_master_stages.run_worker` ran its final GPU
contention check. GPU 1 had been available at launch, but the final check found
other compute PIDs 2745694 and 2858690. The exception replaced the terminal worker
result with `waiting_for_gpu`. The full numerical pricing file remains on disk;
it is not an admitted timing result. It cannot establish uncontended throughput
or a performance comparison.

The supervisor treated this as a resource wait, charged the actual process cost,
then retried pricing on GPU 2. Pricing has no checkpoint-resume path in
`Campaign.numerical_stage`, so attempt 00009 repeated its work. Only 1,872.02
seconds remained under the cumulative allocation. That attempt wrote all eight
individual HMC price rows, but no complete result, and the timeout stopped it
after 1,867.96 seconds. This was a terminal-contention/restart problem rather
than another failure to make the original 62-minute cap large enough. The
previous cap-only repair did not handle this case.

## Affordability audit

The complete but contended result can expose the current forecast arithmetic;
it remains a diagnostic input, not a passed pricing stage. Running the existing
host-only `forecast_campaign` with this result, and conservatively assigning all
process overhead to startup, gives:

| Quantity | Hours | Interpretation |
| --- | ---: | --- |
| Training floor allocation | 19.04 | Includes the inherited factor-two reserve |
| First ensemble posterior assessment allocation | 96.57 | Uses actual mixture timing at L=25 and the factor-two reserve |
| First independent-reference assessment allocation | 1.34 | Includes the same reserve |
| Sum before tuning | 116.94 | Exceeds the remaining 48.78 hours |

This is not an expected runtime or proof that the ensemble cannot fit. Pricing
builds both positive-temperature mixtures at the maximum declared L=25.
`forecast_campaign` uses that mixture for the earliest assessment; the master
then protects this amount before it will allocate training. Consequently even
a clean rerun with similar timings would pause before training. Shorter verified
trajectories might cost less, but no actual multi-chart timing at shorter L was
recorded. Scaling the longest mixture by 3/25 would be an unsupported substitute
for that missing measurement.

The next execution plan needs to address both problems: distinguish a final
contention result from a prelaunch capacity wait and preserve reusable completed
work; and measure the actual ensemble at the shortest declared trajectory for a
conditional initial allocation, then price the actual verified mixture before
sampling. Cumulative funds, scientific checks, fresh verification, source
identities and the deadline must remain enforced. Existing contended prices
must retain their qualification and cannot silently become clean benchmarks.
A repeat of the current full-price command would not answer the affordability
gap, so it is not launched.

## Evidence and decision

Exact source inspected: preserved snapshot
`/tmp/BayesFilter-q20-staged-budget-20260920-r2`, modules
`q20_master_stages.py`, `q20_gpu_runtime.py`, `q20_campaign_runtime.py` and
`q20_master_program.py`. The unchanged run configuration and all prior attempts
remain in `docs/plans/artifacts/q20-master-operations-2026-09-21/campaign-02/`.
The main evidence is attempts `00008-price-ensemble` and `00009-price-ensemble`,
their manifests and logs, and `phase-events.jsonl`. The installed status command
confirmed both supervisor and numerical master inactive at 21:24 Shanghai.

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Keep current campaign paused for a pricing/resume repair | No posterior estimate; full clean pricing pending | Terminal resource check failed; retry allocation exhausted | Cost of a shorter actual mixture and adequate handling of co-resident GPU work | Repair and test the pricing continuation and staged allocation before another numerical launch | Ensemble rejection, budget exhaustion, or a reliable finish ETA |

There is no statistically supported method ranking. The plain candidate still
has no verified kernel; its movement failure remains a candidate veto. The
ensemble has not reached training or kernel verification. The observed runtimes
are descriptive, and the affordability calculation is a conditional engineering
reservation. The weakest evidence is the missing uncontended mixture timing at
the trajectory lengths that might actually be selected. A measured affordable
mixture and successful downstream checks would overturn the present inability
to continue under the executable plan.
