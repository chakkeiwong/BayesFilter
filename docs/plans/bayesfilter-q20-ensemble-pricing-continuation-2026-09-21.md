# q20 ensemble pricing timeout repair

September 21, 21:24 update: the retry campaign is stopped. A completed pricing
calculation failed its terminal GPU contention check; the restarted calculation
then exhausted the remaining cumulative allocation. The
[pricing stop diagnosis](bayesfilter-q20-pricing-stop-diagnosis-2026-09-21.md)
records the evidence and the additional ensemble affordability gap. The launch
observation below is historical, not a claim that execution remains active.

At September 21, 16:44 Shanghai time the campaign is stopped. Plain NeuTra's
declared search completed at 12:34 with 36 observations and no verified member.
Thirty observations requested a larger epsilon; the last six had conflicting
acceptance across chains and failed the movement screen. These settings cannot
support posterior sampling. They do not reject all plain NeuTra settings or the
NeuTra research direction. The existing master correctly advanced to its planned
tempered-ensemble attempt.

Ensemble qualification passed. Pricing then exhausted a fixed 1,200-second
attempt allocation at 12:56, after writing seven of eight HMC price rows and its
training prices. The external timeout charged 1,195.78 seconds. This is an
infrastructure allocation failure, not evidence against the ensemble. The master
has 179,365.04 campaign seconds and 6,278.80 included diagnostic seconds remaining.
The September 25, 18:00 Shanghai deadline and both total limits stay unchanged.

## Intent, evidence, and skeptical audit

The research question remains stable estimation of the fixed q20/T30 posterior
using plain NeuTra HMC or the permitted tempered ensemble. The independent
integration reference and all existing training, numerical-health, convergence,
precision, start-group and reference criteria remain required. This repair asks
only whether the unchanged ensemble pricing program can finish with a measured
time allocation. Its baseline is the timed-out pricing attempt. Success is a
complete checked pricing result followed by the master's existing affordability
and training decisions; neither pricing nor service activity proves convergence.

Vetoes are changed sources or request identities, corrupted completed evidence,
live predecessor workers, exhausted total/diagnostic funds, and the deadline.
A second pricing timeout stops for investigation rather than renewing itself.
No scientific threshold, numerical method, proposal, seed, training protocol,
source snapshot, device class or GPU parallelism changes. All spent time and
completed stage receipts carry forward. Prior numerical attempts stay intact.

Audit verdict: a blind rerun would repeat the short cap and fail the one-attempt
rule. Editing the bound numerical protocol would also invalidate prior evidence.
Instead, record one host-side stage allocation repair, preserve the predecessor
in `campaign-01`, and resume in fresh `campaign-02` with a separately versioned
supervisor. Replay completed stages under their original request identities.
Pricing has no restart checkpoint, so only this short pricing phase is rerun;
prior training and tuning are reused. The current shared repair hold is already
exhausted by recorded timeouts; this explicit local repair draws on the remaining
diagnostic allocation, not a renewed hold or new allowance.

## Numerical choices and execution

The retry allocation is derived from inspected timings:

| Component | Seconds | Provenance and limitation |
| --- | ---: | --- |
| Completed part of failed attempt | 1,195.78 | Measured supervised wall time; rerun includes this work |
| Missing HMC timing row | 230.13 | Largest completed first-call plus four-transition steady-call cost |
| Ensemble transition timing | 431.08 | Hypothesis: two positive temperatures times two calls times four transitions times the largest measured per-transition cost; compilation may cost more |
| Reference and reporting timing | 15.46 | Earlier current-scope measured reference and analysis costs |
| Safety multiplier | 2 | Inherited engineering reserve, not a calibrated probability bound |
| Retry cap | 3,745 | Ceiling of twice the component sum; about 62.4 minutes |

The cumulative price-stage limit is the old 1,195.78 seconds plus 3,745 seconds.
Allow exactly two pricing attempts in total (one failed attempt, one repair), a
convenience bound against repeated failures. Both attempts count toward the
existing six-diagnostic-attempt ceiling. The full retry would leave 2,533.80
diagnostic seconds and about 48.78 campaign hours. A partial pricing result is
never promoted; missing exchange costs remain unmeasured until the retry writes
the full result. Downstream reservations still use the actual new prices.

Before activation, check actual completed-stage request replay against the
unchanged 490-file numerical snapshot, cumulative cap/accounting enforcement,
deadline and diagnostic clipping, refusal of unrelated/repeated repairs, and
selection of the fresh campaign by `status` and `ensure`. CPU engineering checks
hide GPUs. Freeze the updated host supervisor separately; run the existing fixed
trusted `q20_campaign_control.py ensure` command. The worker still chooses an
available GPU and verifies memory growth before TensorFlow initialization.

Artifacts remain under `docs/plans/artifacts/q20-master-operations-2026-09-21/`:
`repairs/ensemble-pricing-timeout-01/` preserves the old job/status, allocation,
source/request audit, tests and repair receipt; `campaign-02/` holds the fresh
ledger and new attempts; `runtime-supervisor-r2.py` is the revised host driver.
The existing master handles later phases and stops at the first valid estimate.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Reject tested plain settings for sampling | No verified member | Final six settings fail movement | Other settings or trained maps may work | Existing ensemble branch | Plain NeuTra is impossible |
| Repair ensemble pricing allocation | Full pricing result pending | No code exception recorded; timeout alone | Unmeasured exchange compilation cost | One funded retry after engineering checks | Affordability or posterior validity |

| Inference status | Finding |
| --- | --- |
| Hard veto screen | The plain trial lacks required all-chain movement at its final tested settings |
| Statistically supported ranking | None; this is not a method comparison |
| Descriptive-only differences | Acceptance and observed timing do not rank samplers |
| Default-readiness | Not established |
| Next evidence needed | Complete ensemble pricing, assessed training, fresh kernel verification and posterior/reference checks |

The strongest alternative explanation is that the timing proxy underestimates
ensemble compilation. A repeated timeout would refute the adequacy of this
allocation. It would require inspecting the unfinished timing section, not
calling the ensemble numerically invalid or extending funds automatically.

## Verification and resumed execution

Seventeen focused host-engineering tests passed in 1.90 seconds with GPUs
intentionally hidden and imports from the preserved numerical snapshot. These
cover the new cumulative/diagnostic/calendar bounds and one-retry limit, plus
existing resource recovery, repeated-failure stops, numerical vetoes, master
progression and controller behavior. Actual request replay reached ensemble
pricing with its original request hash; every completed stage replayed. All 490
numerical source files and 325 completed artifacts matched. No prior spending,
attempt, result receipt, scientific setting or numerical source changed.

The fixed trusted `ensure` command started
`bayesfilter-q20-master-operations-20260921-02.service` at September 21, 16:56
Shanghai time, supervisor PID 2695720. Its new pricing attempt began at 16:56:03
in `campaign-02/attempts/00008-price-ensemble/`. At 16:56:22 the master reports
`running:price-ensemble`; the worker manifest records host GPU 1, verified memory
growth and XLA. The 3,745-second external bound ends around 17:58:29, including
stop grace. This is an execution limit, not a completion forecast.

About 49.82 campaign hours remain at that observation. The numerical parent and
first continuation remain preserved. The new runtime driver SHA-256 is
`43fbbcf5a0896182647d9b7463784ac197d8efee959b951f56f3f404dd891a20`.
`source-and-request-audit.json`, `repair.json`, `tests.log`, the prior job/status
copies, and `launches/002.json` preserve verification and activation evidence.
The current job points `status` and `ensure` at `campaign-02`. Pricing, training,
kernel verification and posterior/reference checks remain unfinished.
