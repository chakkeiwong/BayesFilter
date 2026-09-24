# q20 recovery implementation and execution

September 22, 11:58 Shanghai update: the owner-requested
[Gaussian-step canary](bayesfilter-q20-gaussian-epsilon-canary-result-2026-09-22.md)
completed. All six epsilon values around 1.2–1.35 failed in both starting banks
(0/192 accepted; numerical vetoes), while epsilon .0620027091/L3 controls passed
health checks and accepted 29/32. No tuning or posterior qualification resulted.
The setup repair plus successful run consumed 577.665953 seconds, leaving
155,934.801794 campaign seconds (43.3152 hours) and 1,144.952755 diagnostic
seconds (19.0825 minutes). The existing affordability pause remains in place;
the canary did not change numerical settings or the estimator's scientific
contract. Campaign and terminal balance records include these costs.

At September 22, 11:01 Shanghai, a read-only control status check confirms that
campaign-05 reached `ESTIMATION_BUDGET_PAUSED` at 10:19:51, with reason
`no funded work remains for train-ensemble`. Both current-source GPU
qualifications passed and pricing returned; the timing imports that previously
crashed completed with their original provenance preserved. The defect was
reproduced and fixed, and 45 focused checks pass. Plain midpoint tuning completed
without a verified kernel; there is still no posterior estimate. Remaining
allowances are 156,512.467747 campaign seconds (43.4757 hours), including
1,722.618708 diagnostic seconds (28.7103 minutes). The conservative tuning
reservation exceeds that remaining allowance. Its cost basis needs reconciliation
before ensemble training can be funded; the pause does not prove the scientific
work intrinsically requires the forecast cost. The step-size/whitening guide
audit is recorded in the September 15 production parameter ledger. That audit
changes no active numerical configuration. Earlier execution details follow below.

The master now includes recovery phases R1–R6. The owner requested refresh,
thorough review and execution on September 22. The repaired source is isolated
at `/tmp/BayesFilter-q20-recovery-20260922-r1`, copied from the actual September
20 execution snapshot. Unrelated main-worktree HMC edits were not imported.

## Engineering and saved-evidence review

Completed worker results are written before the final resource probe. Numerical
completion, terminal resource observations and timing quality are distinct.
Pricing records its training, HMC-setting, ensemble, reference and reporting
blocks atomically with scope and content checks. Interruption resumes eligible
blocks; historical imports retain original source and resource qualifications.
First-call elapsed time uses whole-call seconds, while steady HMC/exchange
time uses seconds per transition. The initial ensemble template explicitly uses
the smallest declared width and L=3. Actual verified maps and per-chart kernels
are priced separately before sampling and can veto an unaffordable allocation.

The supervisor quotes missing work before retry, applies cumulative diagnostic
phase limits and records next actions. A contended final selected-procedure
price permits at most one remeasurement of that block, within its original
cap. It does not repeat training or tuning. Historical spent time and the
exhausted 7,200-second infrastructure repair hold remain charged. Allocation
exhaustion is distinguished from infrastructure failure for new work.

The plain repair uses six supplied pairs at epsilon 0.062002709114199195 and
L=(3,5,9,13,18,25), with pilot, directional children and refinement disabled.
The public fixed-transport tuner retains its health, movement, acceptance,
evidence-rung and independent verification rules. The explicit cohort's wall
ceiling follows its recorded funding, rather than silently reverting to the
old eight-hour ceiling. Neither mass adaptation nor comparison prerequisites
were introduced.

The saved endpoint audit checked tensor/evidence checksums, finite values and
status, accepted/rejected state consistency, log-acceptance arithmetic and
transport forward/inverse parity. All twelve saved epsilon/L endpoint records
passed these checks. At the larger epsilon, extremely low Metropolis
probabilities accompany rejected proposals and frozen chains. The traces do
not establish posterior whitening or rule out poor learned geometry.

Sixty-eight distinct engineering checks have passing latest results. The first
run passed 63 checks. A subsequent strengthened funding check exposed an old
fixture that funded a cheap sampling chunk but not the mandatory tuning cohort;
its expectation was repaired and a negative funding test added. Follow-ups
passed, including real interrupted-pricing resume without repeated HMC/training,
actual finite-cohort fresh verification, exact selected ensemble timing, full
known-target estimation and unchanged replay. Numerical tests intentionally hid
GPUs; they establish engineering behavior only. Logs preserve the failed fixture
and passing repairs. Seven test invocations consumed 384.5026 CPU process-seconds;
the separate saved-trace audit consumed 2.0651 seconds. Both are charged to the
separately authorized 48 CPU-hour reserve, not the GPU allowance.

## Execution decision

The active successor is `campaign-04` under
`docs/plans/artifacts/q20-recovery-and-affordability-2026-09-22/`.
It inherits the original total ceilings and cumulative spending, preserves old
attempts separately, and imports the trained maps/optimizer/RNG with the checked
training importer. Exports are cleared for current assessment. Old qualifications
are preserved but current-source qualification runs before numerical continuation.
The fixed `status`/`ensure` commands now follow this explicit successor.

Initial activation at 00:40 Shanghai completed the current GPU/XLA beta-one
qualification in 136.54 seconds and reused historical pricing through the
checked import in 7.51 seconds. Inspection then found an additional forecast
defect: full cached-loss credit made the training quote zero although restoring
and reassessing maps still costs time. The first assessment phase saved its
checkpoint and paused cooperatively after 23.01 seconds under a 29.58-second
allocation. No training, qualification or pricing result was lost.

The host supervisor now adds a separately recorded reassessment quote without
altering original pricing files: measured setup plus two measured heldout first
calls per imported map, with the inherited factor-two margin. For twelve maps
this allocates 787.08 seconds plus process overhead. Dedicated tests verify this
cost is preserved even with cached losses, and that an explicitly repaired
attempt retains prior spending and the cumulative ceiling. This is an engineering
allowance hypothesis; actual time is charged. The numerical source remains
unchanged, so current qualification and pricing remain reusable.

`campaign-04` preserves the settled `campaign-03` ledger and provides one
recorded continuation of this corrected assessment allocation. Service
`bayesfilter-q20-master-operations-20260921-04.service` started at 00:45:13
Shanghai and resumed `cohort-00002.json`; its worker manifest verifies GPU memory
growth before initialization. No extra total or diagnostic funds were added.
Reassessment completed in 55.52 seconds, preserving the twelve beta-one maps at
512 updates and issuing current assessments/exports. Tuning began at 00:46:10.
The saved controller declares exactly the six midpoint pairs, pilot disabled,
zero refinement rounds, fresh evidence rungs (1,2,4), and four batched chains.
Its funded wall ceiling is 39.7247 hours after protecting about nine hours for
downstream work. This ceiling is not an expected tuning duration. The estimated
base measurement/verification work remains approximately 4.75 hours, with a
9.49-hour initial engineering reservation; inconclusive evidence can extend it.

After the rounded-up one-second trusted NVIDIA readiness charge, 175,623.1063
campaign seconds (48.7842 hours) remain, including 2,536.8619 GPU diagnostic
seconds (42.2810 minutes). The six plain candidates' base measurement and
verification reserve is 34,165.5682 seconds (9.4904 hours), derived from observed
same-L work with the inherited factor-two engineering allowance and startup.
All evidence rungs could require up to 193,511.5530 reserved seconds; that is a
conditional cap, not an expected duration or funds already granted. The master
clips it to actual available funds after protecting posterior/reference work.
Inconclusive candidates may therefore remain incomplete at the affordable cap.
The September 25 18:00 Shanghai deadline is unchanged.

Final skeptical review: current-source import and reassessment prevent source
restamping; pricing hypotheses cannot become kernel qualification; cheap L=3
cannot silently replace an expensive verified kernel; mandatory tuning scopes
receive protected funds; all allocation and resource retries conserve balances.
The strongest remaining risks are weak NeuTra geometry, expensive evidence
extensions, unknown uncontended ensemble cost and reference precision. None is
resolved by these engineering tests. The two permitted methods remain viable
research directions, with no statistical method ranking supported.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Execute repaired master | Latest focused recovery and actual CPU workflow checks pass | Current GPU qualification, training assessment and affordability still required | Whether interior kernels qualify and posterior checks fit | Trusted fixed ensure; observe qualification and plain repair admission | Convergence, whitening, production qualification or a finish ETA |

The full source diff, import audit, test receipts, saved-trace audit and ledger
remain under the versioned artifact root. No prior campaign or numerical result
was overwritten.

## R7 status recovery, September 22

At 05:49 Shanghai all six midpoint settings were terminally rejected by the
declared acceptance screen. The initial L=9,18,25 observations were inconclusive;
their predeclared second evidence rungs completed. Final mean acceptance was
0.90786, 0.96588, 0.95882, 0.96442, 0.96202 and 0.98373 for L=3,5,9,13,18,25.
Every final decision requested a larger epsilon. The fixed cohort admitted no
directional children, as planned, and produced no verified kernel. All completed
measurements had chain movement and no recorded hard veto or engineering
invalidity. Initial L=18 and L=25 had large log-acceptance energy-proxy alerts;
these were explanatory under the frozen policy. This does not establish
whitening, posterior convergence, an acceptable alternative epsilon or a method
ranking. The narrow tested cohort failed; the NeuTra research direction was not
rejected.

The master correctly continued to the permitted ensemble, passed beta-0.5
qualification, then crashed in pricing at 05:52. `price_complete` reused `origin`
for map metadata; its nested import helper subsequently supplied that metadata
as historical timing provenance. `PricingLedger.run` raised `KeyError:
timing_quality` at the second positive temperature. This is an implementation
failure in cost bookkeeping, not evidence against the target or ensemble.

The repair separates `historical_origin` from `map_origin`. A CPU regression
reproduced the exact exception on the baseline and passes with the repair. The
test exercises actual two-temperature ensemble pricing and proves that imported
training/HMC timings are not recomputed, the exchange measurement runs, and
original resource quality survives. A checked completed-failure import carries
the terminal plain rejection forward; tests reject partial, successful and
changed evidence. The focused suite passed 45 tests. Actual CPU usage was
69.544292 seconds, charged to the existing CPU reserve, leaving 172,343.888138
seconds. CPU fixtures intentionally hide GPUs and are engineering evidence only.

Source `/tmp/BayesFilter-q20-recovery-20260922-r2` differs from the preserved r1
snapshot only in `q20_master_stages.py` and `q20_master_program.py`. Numerical
target, optimizer, HMC kernels and statistical criteria are unchanged. The
checked training import preserves map/optimizer/RNG state and cache identities.
Old failed pricing blocks contained only imports, so their regeneration repeats
no measured numerical block. Historical receipts retain their original sources;
new GPU work requires current-source qualification.

Campaign-05 preserves total spending of 92,798.304105 seconds, diagnostic spending
of 6,228.585362 seconds, and the original limits. Its 25.011566-second failed
pricing attempt remains charged. The new price cap is 1,174.443065 seconds:
the existing 1,114.419933-second exchange reservation plus twice the observed
25.011566-second failed-worker setup and two inherited five-second stop graces.
This is an explicit allocation within the remaining diagnostic balance, not a
refund or a renewal of the exhausted infrastructure repair hold. Including both
current-source qualifications, the next diagnostic reservation is 1,725.611433
seconds out of the remaining 2,228.768696 seconds. Actual worker time is charged.

The trusted fixed `q20_campaign_control.py ensure` launched service
`bayesfilter-q20-master-operations-20260921-05.service` at 10:11:24 Shanghai.
Its first worker records XLA enabled and verified incremental GPU memory growth.
Preparation, exact test commands, logs, source diff, source audit and CPU charges
are under `artifacts/q20-recovery-and-affordability-2026-09-22/pricing-origin-repair-01/`.

An additional budget concern is visible before the new exchange measurement:
the current mandatory tuning reserve is 63.8284 hours across four scopes. It
uses the historical L=25 cost for L=5,9,13,18 as well as L=25, and its resource
history is unknown. This is a conservative planning hypothesis, not a measured
63.8-hour requirement. It nevertheless exceeds the 43.6163-hour remaining
campaign before training or sampling. The price and affordability decision
must be recorded honestly; this repair does not authorize relaxing the scientific
contract or increasing the budget to overcome that obstacle.

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Reject finite plain midpoint cohort | No verified member; acceptance screen failed | No recorded hard numerical veto; initial energy-proxy alerts preserved | Untested epsilon interval and learned geometry | Permitted ensemble fallback | Rejection of plain NeuTra as a research direction |
| Repair and resume cost estimation | Baseline exception reproduced; 45 checks pass | Original failure preserved; source and budget checks pass | Current ensemble cost and whether required remaining work fits | Current-source qualification, missing exchange timing, staged affordability | Stable posterior, kernel qualification, finish ETA |

Post-repair red-team note: an executing worker is not evidence of posterior
progress, a cheap mixture timing does not qualify its kernel, and a conservative
forecast is not proof of intrinsic computational impossibility. The weakest
remaining planning evidence is the inherited worst-length tuning reservation.

At 10:19, beta-0.5 qualification completed in 139.545501 seconds and beta-one
qualification in 135.540681 seconds. The actual ensemble pricing ledger has
imported training, all eight historical HMC rows across both temperatures, and
the reference price with `resource_history_unknown` preserved. Its exchange
measurement is in progress. No new numerical failure is recorded at this point.
