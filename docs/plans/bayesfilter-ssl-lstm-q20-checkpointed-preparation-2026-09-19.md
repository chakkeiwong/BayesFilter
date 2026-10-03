# q20 checkpointed preparation and measured-cost scheduling

Status: `MASTER_STARTUP_PASSED_MASS_DEFERRED`, terminal September 19 13:42:13
Asia/Shanghai. Both GPU/XLA qualifications and both bootstrap scopes passed;
mass adaptation was deferred before launch on the measured-cost check. All four
worker attempts exited normally, the service is inactive, and accounting is
settled. This completes the bounded startup repair, not the overall campaign.
Owner authorization: September
19, “refresh the master program. and continue with your suggestion”. This
allocates a new repair phase from the settled remainder; it does not renew the
campaign or diagnostic budget. Predecessor:
`ssl-lstm-q20-bootstrap-repair-2026-09-18/retry-01/campaign`.

## Research intent and evidence contract

The question is whether the same classical q20/T30 target can complete a healthy
bootstrap without spending its preparation allocation refining acceptance before
mass adaptation, and whether subsequent preparation is affordable. The comparator
is the two preserved timeout attempts, using their frozen source. The target,
exact score, prior-scale coordinates, XLA, initial states, bootstrap draw counts,
mass adaptation requirements, candidate verification, and posterior gates stay
fixed. The bootstrap random stream changes explicitly with chunking.

| Role | Requirement |
| --- | --- |
| Engineering pass | Completed chunks survive interruption; unchanged-source replay continues the same chunk stream without repeating completed chunks; insufficient predicted time defers work before launch |
| Startup pass | Existing finite state/value/log acceptance/status checks plus proposed-state/score health over all 40 transitions; observed acceptance at least the inherited repair floor .55; no upper-acceptance rejection |
| Preparation pass | Existing full mass adaptation and actual metric update, then unchanged handoff checks |
| Promotion veto | Any numerical health failure; incomplete preparation or absent candidate verification/posterior evidence |
| Continuation veto | Corrupt checkpoint, changed target/source/start/config, invalid shared target, runtime exception, exhausted allowance |
| Repair trigger | Low finite bootstrap acceptance; deadline or resource interruption with intact checkpoint |
| Explanatory | Acceptance, movement, compile/runtime, predicted mass cost; none establishes posterior whitening or convergence |
| Not concluded | Startup is not a tuned kernel, posterior sample, complete preparation, method ranking, or production readiness |

## Repair and acceptance-rule review

Use an opt-in warmup-startup bootstrap role for both q20 preparation consumers.
The prior .65–.75 rule repeatedly increases a healthy high-acceptance starting
epsilon before the mass changes. The resulting pre-adaptation acceptance cannot
qualify the final mass/epsilon pair. Retain the existing lower repair floor .55
as an explicitly uncalibrated startup hypothesis, all 8 discarded and 32 measured
transitions, the bounded geometric downward repair, and all numerical vetoes.
High acceptance nominates a starting step only. The ordinary public bootstrap
default and final tuning acceptance policy remain unchanged. Reject lower-floor
exhaustion rather than silently using an unscreened geometry fallback.

Run bootstrap through the existing stable-signature TF/TFP XLA runner in chunks
of four transitions, carrying the last position, fixed epsilon/mass, and separate
deterministic chunk seeds. Refreshing HMC momentum at each transition makes this
a fixed-kernel continuation; bitwise equality to the old single-call seed stream
is not claimed. Archive warmup too and check its health before discarding it from
the acceptance statistic. Python performs checkpoint I/O between compiled chunks;
it does not implement leapfrog or iterate over samples inside a compiled kernel.
Persist samples, trace, configuration, source/start identity and ordinary checksums.
Resume into a fresh output directory, leaving previous evidence intact.

Before each chunk reserve its predicted runtime using the maximum observed cost
per transition and the protocol's existing forecast factor. Before mass adaptation
reserve the complete required warmup at the measured cost. An unaffordable stage
is explicitly deferred with its bootstrap checkpoint; do not launch a doomed
partial warmup or replace 1,000 serious transitions with a smoke count.

## Numbers, assumptions and budget

The settled allowance is 88,545.36473172551 campaign seconds, including
12,785.796950445605 diagnostic seconds. Charge measured CPU verification and a
180-second setup/accounting allocation (inherited engineering envelope), then
fund this phase within the remaining diagnostic balance. Previous costs remain
charged. The old 4,000-second preparation allocations are historical, not renewed.

Four-transition chunks are a convenience choice: prior scalar L25 rounds measured
33.15–38.21 seconds/transition, giving approximately 2.2–2.6 minutes/chunk before
the existing safety factor. Use the maximum measured predecessor cost as the
initial forecast; update with current actual costs. Qualification allocations
derive from the previous maximum 192.1 seconds and the same factor plus startup
reserve. A preparation allocation reserves initializer cost, one complete
40-transition screen and cleanup, derived from predecessor observations. Allow
local interrupted-stage continuation only while both phase and campaign balances
fund the next chunk. No automatic extra budget or changed scientific criteria.
Each stage permits at most two infrastructure attempts, sharing its measured
envelope; this is an engineering retry limit, not renewed compute authority.

The 1,000-transition serious mass minimum is inherited. At previous scalar cost
it forecasts about 9.2–10.6 hours per temperature before safety overhead. Geometry
and batching can change this, so it is a conservative scheduling hypothesis, not
a measured complete preparation price. Record remaining price gaps explicitly.

Artifact root: `docs/plans/artifacts/ssl-lstm-q20-checkpointed-preparation-2026-09-19/`.
Execution uses a new copy of `/tmp/BayesFilter-q20-bootstrap-repair-20260919-r2`
with only the reviewed changes. GPU workers use trusted execution and memory
growth; CPU fixtures hide GPUs before TensorFlow import. The existing master
supervisor records exact commands, seeds, sources, hardware and charged time.

## Skeptical audit and verification

The audit rejects simply raising the old timeout, accepting initializer probes
as completed bootstrap, lowering serious adaptation counts, or treating successful
startup as tuning. It also rejects combining concurrent main-branch changes with
the experiment. Those would respectively preserve waste, use the wrong evidence,
change scientific scope, or confound the comparison.

The plan passes with the explicit startup-only role and independent downstream
gates. The weakest assumption is that .55 realized acceptance is a useful startup
screen; it is not calibrated convergence evidence. Chunk boundary target
reinitialization adds overhead, so measure it before predicting later costs.
New streams prevent exact historical trajectory comparison. Unit/CPU-XLA checks
must cover interruption/resume equality to uninterrupted chunk execution,
checkpoint mismatch/corruption, discarded-transition numerical vetoes, budget
deferral, high-acceptance startup versus ordinary bootstrap, and both q20 consumers.
The master must preserve settled arithmetic and reusable completed work. A fresh
GPU qualification precedes the bounded startup run. No full mass launch is
authorized by a partial bootstrap result or an unaffordable forecast.

Integration review found a material handoff dependency: operational warmup
treated every bootstrap-selected epsilon as an upper bound. A startup-floor
nomination cannot supply that qualification. The new role is therefore recorded
in the selected-kernel payload and dispatches to warmup's existing reasonable-step
search, without the inherited startup ceiling. A CPU Gaussian integration run
that reached zero accepted metric updates exposed this issue; that failure is
charged and retained. The full metric-update requirement remains binding.

## Final implementation review and execution allocation

The frozen source passed 70 focused tests, including CPU XLA resume equality,
scope/corruption rejection, health checks during discarded startup, budget
deferral, bootstrap regressions, both q20 consumers, real classical tuning,
pricing, and chart/replica dispatch. The main-branch integration passed another
25 focused checks. The corrected Gaussian preparation applied three metric
updates at beta .5 and one at beta1. These are mechanics fixtures, not q20
scientific evidence. Verification consumed 260.144601 seconds, including failed
prechecks; the separate setup allocation is 180 seconds.

Final review checked the source overlay against the predecessor, both public
consumer paths, startup-to-mass handoff, selected-kernel role, discarded/proposed
health, deterministic chunk seeds, checkpoint continuity, cache reuse, ordinary
bootstrap default, and cumulative retry accounting. No target, score, data or
downstream acceptance/posterior threshold change entered the isolated source.
The startup rule change is explicitly opt-in and does not grant tuning authority.
This was a self-review with executable checks, not an independent reviewer verdict.

Derived worker allocations are 484.107165 seconds per qualification and
3,717.133465 seconds per preparation, shared across at most two attempts per
stage. Cooperative preparation limits subtract the inherited 100-second reserve.
The total worker envelope is 8,402.481259 seconds (2.334 hours). The service gets
8,463 seconds including 60 seconds for coordinator settlement. These are maximum
allocations; expensive unfunded mass adaptation is not launched merely because
the service still has time. Execution starts with 88,105.220131 campaign seconds,
including 12,345.652350 diagnostic seconds, after all verification/setup debits.

Exact source: `/tmp/BayesFilter-q20-checkpointed-preparation-20260919-r1`.
The executable master command, environment, deadlines and allowance are in
`artifacts/ssl-lstm-q20-checkpointed-preparation-2026-09-19/launch.json`.
The durable service is `bayesfilter-q20-checkpointed-preparation-20260919-r1.service`.
Use `campaign/campaign.json`, worker manifests and each
`worker/data/bootstrap-checkpoints/` directory for live progress; terminal
allowance appears only after all attempts settle.

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Execute bounded repaired startup | Focused CPU/XLA and real-consumer checks passed | No engineering/source/accounting veto | Actual q20 chunk health and cost | Fresh GPU qualification and bootstrap, followed by full-mass affordability check | Complete preparation, tuning or posterior validity |

| Inference status | Finding |
| --- | --- |
| Hard veto screen | Original timeouts remain incomplete; all new numerical health vetoes remain binding |
| Statistically supported ranking | None; no method comparison performed |
| Descriptive-only differences | Predecessor timings and future startup acceptance/runtime |
| Default-readiness | Not established; q20 explicitly selects the reviewed startup role |
| Next evidence needed | Real q20 startup, affordable completed mass adaptation, fixed-kernel verification and posterior checks |

Historical launch check: the service started on GPU 2 with XLA requested and
memory growth verified before initialization. Its hard deadline was September
19 14:30:12 Asia/Shanghai; it completed normally before that deadline.

## Terminal assessment, September 19

The executed question was whether the same q20/T30 classical target could finish
healthy bootstrap with checkpointing and startup-only acceptance, and whether
full mass adaptation fit its allocation. Both startup screens passed. No numerical
health veto occurred in these screens, and no worker timed out. The cost test
deferred the unfunded mass stage; no mass transitions or posterior draws ran.

| Temperature | Bootstrap rounds | Completed transitions | Selected epsilon / L | Selected-round realized acceptance | Preparation worker wall time |
| --- | --- | --- | --- | --- | --- |
| .5 | 1 | 40 in 10 chunks | .011048543456039808 / 25 | 1.0 | 1,890.943 s |
| 1 | 2 | 80 in 20 chunks | .011048543456039808 / 25 | .84375 | 3,321.369 s |

At beta1, the first round at epsilon .022097086912079615 had realized acceptance
.28125 and triggered the declared downward repair. It had no hard numerical
veto. Each round includes eight discarded and 32 measured transitions; all
transitions and their proposal/score/status checks are checkpointed. Beta .5
passed the explicit startup floor with acceptance 1; it did not pass the old
.65–.75 acceptance rule, which no longer defines this startup role.

The recorded maximum chunk costs were 40.165089 and 41.908213 seconds per
transition. Multiplying by the unchanged 1,000-transition mass minimum gives
11.16 and 11.64 hours before overhead. With the predeclared factor2 scheduling
margin, the master requested 80,330.177 and 83,816.427 seconds (22.31 and 23.28
hours). Only 1,750.405 and 319.833 seconds remained in the respective preparation
allocations. These are extrapolated scheduling estimates, not measured mass
adaptation times; its metric changes, extra step searches and numerical behavior
could change the actual cost. Full preparation remains unpriced.

Qualification cost 186.547 and 183.551 seconds. Total settled worker cost was
5,582.408528 seconds (93.04 minutes), in addition to the already charged
260.144601 verification and 180 setup seconds. The remaining allowance is
82,522.811603 campaign seconds (22.9230 hours), including 6,763.243822 diagnostic
seconds (1.87868 hours). The complete allowance and checksum are in
`campaign/settled-allowance.json`. On terminal inspection, the ledger checksum,
sum of worker charges, remaining-balance arithmetic, and all recorded stage
artifact checksums were verified. No source or result was overwritten.

The result does not invalidate the target, implementation, or research direction.
It resolves startup completion under the revised role and exposes the cost of
the remaining unchanged work. The overall campaign budget is not exhausted;
the planned preparation allocations cannot fund mass adaptation. There is no
additional funded runtime-repair phase in this completed master invocation.
The next discriminating work is bounded profiling of the expensive target/score
and integrator path, followed by a measured estimate for the actual warmup route.
Keep the prepared startup checkpoints and all prior training evidence.

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Close startup repair; defer mass adaptation | Both bootstrap scopes passed; complete mass criterion untested | No bootstrap health veto; mass-cost admission failed | Actual adaptation cost and posterior geometry | Profile target/score work and revise affordable continuation from settled budget | Completed preparation, convergence, whitening, tuned kernel or production readiness |

| Inference status | Finding |
| --- | --- |
| Hard veto screen | No health veto in completed startup; beta1 first pair rejected by startup acceptance floor only |
| Statistically supported ranking | None |
| Descriptive-only differences | Observed acceptance, runtime and repair counts; no uncertainty-supported performance comparison |
| Default-readiness | Not established; startup has no candidate or posterior authority |
| Next evidence needed | Completed operational mass adaptation, independent candidate verification, posterior and reference checks |

Post-run red-team: the strongest alternative explanation for apparent progress
is the deliberately changed acceptance role and chunk seed stream. Passing this
screen does not show the old rule would pass, or that mixing or target evaluation
became faster. Resume equality was tested on CPU fixtures; these GPU workers
needed no restart, so actual interrupted GPU recovery remains unobserved. The
weakest cost evidence is extrapolating short fixed-mass scalar chains to adaptive
warmup. A properly bounded measurement of the actual warmup path could overturn
that forecast; lowering scientific work counts would not validate it.
