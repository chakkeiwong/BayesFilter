# Terminal NeuTra causal-repair checkpoint

The owner's request to plan, review and execute all audited gap repairs is
complete for the declared benchmark scope. Read
`bayesfilter-neutra-causal-repair-results-2026-10-01.md` and the plan of the same
prefix. No worker or planned phase remains active. No commit/push was requested.
Preserve unrelated dirty changes on branch
`preserve/shared-main-before-fab-20260926` (original HEAD
`de80aaff5812ebfbed551977476c0868551a2c88`).

The six repaired consumers cover HMC map-based starts, effective 8/12 epsilon
repair bounds, optional calibrated temporal contrasts, real Adam/walker
continuation, effective SMC mutation work/proposal rescue, and retained clipping,
progress and 1000-point diagnostics. Continuation validates the original teacher
particles/weights. Failed numerical probes also veto refined maps; supervisor
and qualification consumer reject upstream failures. The global legacy HMC
policy and canonical IAF architecture remain unchanged.

All 18 originally failed cases received repairs; 15 have new passing posterior
screened runs. Twelve earlier passes are preserved, giving passing examples in
27 of the original 30 groups. This is not a reliability estimate. Open groups:
mixture/Gabrié seeds 11 and 37 (Gaussian plateau under all three controls), and
wiggle/Gabrié seed 23 (precision/search limitations). Ordinary mixture seed 37's
lower-LR map still failed shape after 131072 updates without supported progress;
other maps for that same seed passed. All three frozen-map funnel teachers and
resulting SMC-trained IAFs have passing downstream examples, but physical
mutation-only teachers failed at 4/16/64 mutation settings. Their populations
actually executed 12/48/192 steps; transported populations needed none.

A review found and repaired the omitted warped-mixture/Gabrié seed-37 case.
It passed after continuation to 65536, a frozen-sampler check and the declared
HMC retry. Case enumeration now derives from the original ledger.

The exact case table, all attempts, costs and checks live under:
`artifacts/neutra-warm-start-master-2026-09-29/campaign-r1/causal-repair-20261001-r1/`:
`terminal-review.json`, `case-coverage.md`, `state.json`, and
`final-source-verification/`. The terminal review checks 3362 archived source
hashes, with no mismatch; all completed GPU workers recorded memory growth;
no invalid terminal probe was found. Regression counts/scopes are in the result
note, including the final installed-source 16-test pass.

GPU 1 was assigned; unrelated GPUs 0/2 were left alone. ALWAYS use trusted
execution for GPU work. Two initial sandbox failures are archived. Shared tuner
work seeds depend on source hashes, so even a comment edit can change streams;
the saved funnel repeat records both earlier failures and a later pass, and is
not an isolated algorithmic improvement. See `source-replay-sensitivity.json`.

Repair workers used 2.257 GPU process-hours and 3.524 CPU core-hours.
Conservative remaining allocation: 24.032 GPU process-hours and 49.134 CPU
core-hours, including the old orphan-job uncertainty reserve. Use the shared
ledger as authority. Routine CPU regression cost is reported separately.

`repair-next-phase.json` now marks terminal review complete, with no blind resume
command. Further research needs a focused experiment for the explicit open
training/kernel questions; q20/default readiness and method ranking remain
unestablished. Existing campaign authorization and remaining budget persist,
but do not silently substitute a new architecture or relax a scientific screen.
