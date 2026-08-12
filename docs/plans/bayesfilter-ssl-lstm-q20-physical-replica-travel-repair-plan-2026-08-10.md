# SSL-LSTM q=20 physical replica-travel repair plan (2026-08-10)

Status: `AUDITED_24X1_EIGHT_HOUR_MATERIAL_CAMPAIGN_AUTHORIZED`

## Research intent ledger

| Field | Declaration |
|---|---|
| Main question | Can a fixed physical-coordinate replica-exchange HMC transition produce repeated cold-hot-cold travel and converged cold chains at a measured, bounded cost? |
| Candidate mechanism | The existing exact TensorFlow/TFP six-temperature fixed-HMC replica-exchange kernel in the checked physical affine chart, followed by a distributed exact-HMC repair only if the current graph is too slow.  No NUTS and no NeuTra coordinates. |
| Baseline | The exact 12-transition physical run: six betas `(1,.5,.25,.125,.0625,.03125)`, step `0.05/sqrt(beta)`, `L=8`, two sign-separated chains, five hot local-HMC sign changes, two cold sign transitions, zero round trips, and `4600.29 s` undifferentiated wall time. |
| Expected failure mode | The ladder may communicate locally without completing round trips; hot replicas may retain sign; the four-thread graph may make a statistically useful chain unaffordable; chunk restarts may corrupt identity accounting if swap matrices are not recomposed globally. |
| Promotion criterion | Material transition admission requires repeated round trips, hot sign forgetting from local HMC, valid finite target/status telemetry, and modern rank-normalized split/folded R-hat plus declared bulk/tail ESS on retained cold physical draws. |
| Promotion veto | Any non-finite retained state, target, score, or log acceptance; invalid target status; broken swap permutation; target/chart identity mismatch; missing trace/checkpoint; or failed modern cold-chain diagnostic. |
| Continuation veto | Harness invalidity, failed source representative parity, failed checkpoint state/identity composition, output collision, wall cap, or a measured cost that cannot fit a reviewed material budget.  Zero round trips in a valid short canary is a repair trigger, not a research-direction veto. |
| Repair trigger | Slow cached transition cost triggers a process-distributed exact value/score design.  Weak adjacent communication triggers ladder repair.  Good swaps but weak hot sign forgetting triggers a hotter or longer hot trajectory. |
| Explanatory diagnostics | HMC and adjacent-swap acceptance, raw cold occupancy, individual sign paths, runtime, compile time, and point estimates.  These cannot promote the sampler alone. |
| Must not be concluded | Full mode discovery, full-posterior authority, correct posterior weights, convergence from raw occupancy, NeuTra repair, predictive validity, sampler superiority, or default readiness. |

The independent mass comparator is the accepted two-known-region annealed-SMC
result and its interval `[0.40573,0.53602]`.  SMC is retained only as the weight
authority over those proposal-supported regions.  Replica exchange addresses
transition travel and cold sampling; its raw occupancy is not a replacement mass
estimate.

## Evidence contract

### Timing/profile canary

| Item | Contract |
|---|---|
| Question | How much of the old wall time is first-call compilation, what is cached seconds per exact transition, and does 32-thread execution improve the same graph descriptively over four threads? |
| Comparator | Same source target, chart, betas, steps, `L=8`, two chains, and one-transition graph at four versus 32 pinned CPU threads. |
| Primary criterion | Both topologies preserve exact target/chart identity, finite traces, valid target status, one XLA trace, and durable first/cached call receipts. |
| Vetoes | Identity/parity failure, non-finite trace, invalid accepted state, wrong affinity, retracing, overwrite, child timeout, or campaign cap. |
| Explanatory only | Compile-inclusive seconds, one cached seconds-per-transition observation per topology, acceptance, swaps, and signs. |
| Nonclaim | One cached observation cannot statistically rank topologies or establish capacity, mixing, convergence, or posterior validity. |
| Artifact | `docs/plans/artifacts/ssl-lstm-q20-physical-replica-travel-repair-2026-08-10/r1-timing/` |

### Later material transition campaign

The later campaign is conditional and not authorized by this timing canary alone.
It must freeze its ladder, step sizes, leapfrog count, chain count, chunk size,
warm-up rule, retained-draw budget, round-trip threshold, hot-forgetting threshold,
R-hat threshold, bulk/tail ESS thresholds, and total compute after timing and a
checkpoint-composition test.  The concatenated full trace, not per-chunk reset
diagnostics, is the diagnostic authority.

At minimum, the later plan must use the repository modern diagnostic definition
`max(rank-normalized split R-hat, folded rank-normalized split R-hat)`.  Thresholds
will not be invented before the achievable draw count is measured.  If a useful
ESS target cannot fit, the campaign is under-budgeted and must not run.

### Four-chain timing/travel checkpoint

The measured 12-worker distributed canary changed the engineering baseline.  Its
exact `L=8` transition took `9.47256 s`, versus `235.08511 s` for the cached
four-thread monolithic graph, with a descriptive ratio of `24.82`.  All target,
status, cache-parity, XLA-worker identity, invalid-path self-rejection, and swap
permutation gates passed.  This opens one four-chain checkpoint before material
sampling.

| Item | Frozen contract |
|---|---|
| Question | Does the exact distributed backend remain valid and affordable for four cold chains, and do 25 transitions preserve communication and global identity accounting? |
| State shape | Six temperatures x four chains x four parameters; starts alternate the checked plus/minus representatives, then evolve independently through stateless momentum. |
| Kernel | Betas `(1,.5,.25,.125,.0625,.03125)`, steps `0.05/sqrt(beta)`, `L=8`, deterministic even/odd adjacent swaps, seed `(20260810,7201)`. |
| Target topology | 24 persistent one-row CPU/XLA workers pinned to CPUs `32--55`; no scalar fallback, no training update, GPU hidden. |
| Budget | 25 transitions, chunks of five, `900 s` cap, one attempt plus a localized unchanged-settings harness retry. |
| Primary pass | All retained states/cached values/scores finite and independently cache-checked; all accepted-state statuses valid; invalid proposal paths self-reject; every swap matrix is a permutation; exact target/chart/worker identity matches; wall cap passes. |
| Nomination screen | Cached transition cost plus 50% margin supports at least 300 discarded and 1,000 retained transitions within the remaining `20,000 s`; every adjacent pair accepts at least one swap; no temperature-chain mean Metropolis probability is below `0.35` or above `0.99`. |
| Explanatory only | Raw signs/occupancy, acceptance differences, hot sign changes, cold transitions, and any round trip count in only 25 transitions. |
| Nonclaim | No convergence, posterior, mass, predictive, superiority, or default-readiness conclusion. |

The acceptance band combines the inherited BayesFilter confirmation lower bound
`0.35` with the generic HMC diagnostic upper limit `0.99`.  It is a checkpoint
nomination screen, not a universal scientific constant.  A miss triggers tuning
or budget repair before material sampling; it does not invalidate the target or
replica-exchange direction.

### Conditional material contract

If and only if the four-chain checkpoint passes and its measured projection fits,
freeze the following material policy before launch:

- four chains and the checkpoint kernel/ladder, with no retuning after retained
  sampling begins;
- chunks of at most ten transitions with immutable tensor receipts and atomic
  cumulative progress;
- at least 300 and at most 500 discarded warm-up transitions per chain;
- warm-up readiness requires a recent-window modern rank-normalized split/folded
  R-hat at most `1.05`, repeated global identity travel, and all hard validity
  screens;
- retained sampling starts at 1,000 draws per chain and may extend in 250-draw
  blocks to at most 1,500;
- retained admission requires all-parameter modern R-hat at most `1.01`, bulk ESS
  at least `1,000`, tail ESS at least `400`, repeated cold-hot-cold identity round
  trips, hot local-HMC sign forgetting, valid target/status/finite traces, and the
  checkpoint acceptance band;
- SMC remains the independent two-known-region mass authority.  Cold negative-sign
  occupancy is reported against SMC `[0.40573,0.53602]` as an explanatory
  consistency diagnostic and cannot replace the convergence/travel gates; and
- total material wall cap is `20,000 s`.  If 1,000 retained draws cannot fit after
  measured warm-up cost, stop as under-budgeted rather than weaken ESS or R-hat.

The R-hat/ESS thresholds are inherited from the repository
`RankNormalizedHMCThresholds` defaults and the physical-coordinate computation is
required.  They are finite-sample health criteria, not proofs of exhaustive mode
coverage.

### Checkpoint-triggered topology repair

The `24 workers x 1 row` checkpoint passed every hard gate and the communication
and acceptance nomination screens, but failed only the prospectively frozen cost
screen: `11.0810 s` checkpoint cost per transition projected to `21,607.97 s`
for 1,300 transitions with 50% margin.  The cap is `20,000 s`; the miss is `7.44%`.

This is a performance repair trigger, not a sampler rejection.  The next topology
canary uses the same 24-row state and exact kernel with `12 workers x 2 rows`.
Batch size two may amortize per-process overhead while preserving concurrent XLA
evaluation.  Its frozen pass rule is:

- all exact target/status/cache/swap/invalid-path/worker identity gates pass;
- one transition plus one cache wave is measured, with checkpoint-equivalent cost
  estimated as `transition_seconds + cache_seconds / 5`;
- `1.5 * 1300 * estimated_seconds <= 20,000 s`; and
- no scientific claim or topology ranking is made from the single timing draw.

If it passes, run a fresh 25-transition `12x2` checkpoint under the same screens.
If it fails, stop as under-budgeted.  Do not reduce warm-up, retained count, margin,
R-hat, ESS, or travel gates after observing timing.

The canary failed the originally declared terminal-cache tolerances.  A bound
follow-up diagnosis showed that changing only a row's batch-two companion changed
its computed value by as much as `3.70e-7` and score by as much as `4.66e-6`.
Relative to observed target magnitudes near `35--54` and score magnitudes up to
`116`, these are approximately `1e-8` relative perturbations and are plausibly
ordinary floating-point/XLA eigensolver effects.  The inherited absolute `1e-9`
value and `1e-8` score tolerances had no target-specific calibration and were
wrongly elevated into a continuation veto.  They cannot reject the `12x2`
topology by themselves.

Before a fresh checkpoint, a bounded identical-randomness materiality canary must
compare `24x1` against `12x2` and changed batch-two pairings.  Its primary screen
is actual HMC and swap decision agreement; it must also report value/score,
leapfrog endpoint, Hamiltonian/log-acceptance, and forward/reverse error.  These
continuous differences are explanatory unless they change decisions or violate a
scale-aware numerical screen declared before the run.  MCSE contextualizes
scientific materiality but does not by itself prove detailed balance.

The pre-launch skeptical audit passes for that canary.  The comparator is the
valid `24x1` route, not a weak proxy; target, chart, initial state, ladder, steps,
`L=8`, and stateless random seeds are identical.  Contiguous and shifted `12x2`
pairings distinguish ordinary batching effects from a single favorable pairing.
Finite/status failures and decision disagreements veto nomination.  The
forward/reverse scaled-error screen is `100*sqrt(binary64 epsilon)`, a conservative
convenience-chosen engineering bound rather than a posterior-accuracy requirement
or universal HMC constant; raw errors remain reported.  One transition cannot
establish invariant-measure equality, convergence, or superiority, and a passing
canary only authorizes the already planned fresh 25-transition checkpoint.  The
run is CPU-only/XLA, hash-bound to the prior evidence, uses a unique output root,
and has a `360 s` runner budget plus a detached service cap.

The unchanged-contract `r7` canary passed every numerical-materiality screen.
Across the `24x1` reference, contiguous `12x2`, and shifted-pair `12x2` routes,
all target statuses and outputs were valid/finite, HMC path validity was
identical, every HMC accept/reject decision was identical, every swap decision
was identical, and scaled forward/reverse errors passed.  The largest observed
log-acceptance perturbation was `7.52e-8`, acceptance-probability perturbation
`6.92e-8`, proposal-state perturbation `1.40e-8`, and swap-log-ratio perturbation
`3.58e-8`.  This clears the observed rounding differences as immaterial for this
canary; it does not prove invariant-measure equality.

The topology nevertheless fails its intended performance role.  Contiguous
`12x2` measured `25.389 s` per transition plus a `3.095 s` cache wave, giving a
checkpoint-equivalent `26.008 s/transition` and a `50,715.80 s` conservative
material projection.  The same-run `24x1` reference measured `11.373 s` per
checkpoint-equivalent transition.  Therefore no fresh `12x2` checkpoint is
justified: it is numerically viable but descriptively much slower than the
already valid `24x1` route and cannot fit the frozen cost screen.

### Eight-hour material authorization

The user authorized eight additional hours on 2026-08-11.  This changes only the
campaign wall budget from `20,000 s` to a hard `28,800 s`; it does not change the
target, chart, `24x1` topology, ladder, steps, `L=8`, chain count, warm-up,
retained-draw ladder, travel, R-hat, ESS, acceptance, validity, or nonclaim policy.
The previous 50% projection margin was a planning admission buffer, not a
scientific gate.  The measured raw 1,300-transition estimate of about `14,405 s`
fits the new hard budget, so the margin no longer vetoes launch.  The runner stops
starting transitions at `28,500 s`, preserving `300 s` for final artifacts; these
are respectively derived from the user budget and a convenience-chosen bounded
finalization reserve.

The material execution uses the valid `24 workers x 1 row` CPU/XLA topology and
the frozen conditional contract.  Warm-up readiness is checked at exactly
`300,350,400,450,500` discarded transitions using the latest 300 cold physical
draws and R-hat `<=1.05`.  Retained admission is checked at exactly
`1,000,1,250,1,500` cold physical draws using R-hat `<=1.01`, bulk ESS `>=1,000`,
and tail ESS `>=400`.  The travel threshold is at least one complete
cold-hot-cold return in each of the four independent chains, hence at least four
global returns; hot forgetting requires both signs and at least one local-HMC hot
sign change in every chain.  These choices instantiate the previously frozen
"repeated travel" and "hot forgetting" requirements without using occupancy as a
mass estimate.

The skeptical pre-launch audit passes after repairing and testing the milestone
scheduler.  The exact checkpoint and materiality artifacts are SHA-bound; no weak
baseline or proxy is promoted.  Every transition checks finite retained
state/target/score, finite log acceptance or declared invalid-path `-inf`, invalid
self-rejection, and swap permutations.  Every ten-transition chunk has immutable
tensor receipts, a terminal state/cache checkpoint, an atomic progress update,
and overwrite refusal.  Warm-up is excluded from retained diagnostics.  The hard
wall cap, warm-up failure at 500, retained failure at 1,500, and hard numerical
invariants are continuation vetoes.  Acceptance, runtime, sign paths, and cold
occupancy remain explanatory except for the already frozen acceptance-band
admission screen.  A passing run establishes a finite-sample two-known-region
candidate only; it cannot establish exhaustive mode discovery, full posterior
authority, predictive validity, superiority, or default readiness.

### Warm-up-triggered ladder repair

The `r8` candidate stopped cleanly at the frozen 500-transition warm-up veto.  It
was not a budget or implementation failure: all 50 manifests and 650 tensor
receipts verified, all 12,000 row-transitions were target-valid, no proposal path
was invalid, all swap permutations passed, HMC acceptance was `0.76--0.89` by
temperature, and adjacent swap acceptance was `44.5--52.4%`.  Global travel also
passed strongly at the terminal milestone, with 56 complete returns and
per-chain counts `[18,16,10,12]`.  The candidate failed because recent-window
modern R-hat was `1.1389 > 1.05` and chain four had zero hottest-temperature local
sign changes despite twelve round trips.  This fires the predeclared hotter/longer
hot-trajectory repair trigger; it does not invalidate the target, harness, or
replica-exchange direction.

The smallest discriminating repair keeps six temperatures, four chains, physical
coordinates, exact fixed HMC, `L=8`, `24x1` CPU/XLA evaluation, and
`step=0.05/sqrt(beta)`, and changes only the geometric ladder ratio.  Two
100-transition canaries run concurrently on disjoint CPUs: ratio `0.40`, with hot
beta `0.01024`, and ratio `0.35`, with hot beta `0.0052521875`.  The inherited
ratio `0.50` is the measured failed baseline.  The candidate grid is a bounded
target-specific hypothesis: `0.40` is the smallest meaningful heat increase;
`0.35` brackets a substantially hotter endpoint without adding replicas or
changing trajectory length.

The tuning selection screen requires every chain to observe both signs and at
least one hottest-temperature local-HMC sign change, every adjacent pair to
communicate, every temperature-chain mean acceptance probability to remain in
`[0.35,0.99]`, finite/status-valid retained quantities, invalid-path
self-rejection, valid swap permutations, and the wall cap.  R-hat, round-trip
count beyond basic communication, raw occupancy, and runtime are explanatory at
100 transitions and cannot select or rank.  If only one arm passes, select it.  If
both pass, select ratio `0.40` as the prospectively declared smallest intervention,
not as a statistically supported performance ranking.  If neither passes, stop
the repair campaign rather than weaken the material gates.  A selected arm must
start a fresh warm-up; tuning draws are never retained.

The skeptical audit passes after aligning invalid-proposal handling with the
material policy: invalid proposals may occur but must self-reject and cannot kill
unrelated chains.  The failed `r8` artifact is SHA-bound, outputs are versioned and
overwrite-refusing, each arm has a 2,400-second cap, and the two arms use disjoint
CPU sets.  A tuning pass authorizes one fresh material attempt under the remaining
eight-hour campaign budget; it cannot establish convergence, posterior validity,
superiority, or default readiness.

Neither hotter ratio passed.  Both arms remained finite/status-valid with zero
invalid paths, all adjacent pairs communicated, and all temperature-chain mean
acceptance probabilities remained in `[0.35,0.99]`.  Ratio `0.40` produced local
hot sign changes `[4,3,1,0]` and ratio `0.35` produced `[1,0,0,0]`; neither
repaired all-chain hot forgetting, and the hotter/coarser ladders reduced complete
travel.  This rejects additional heat as the repair mechanism under this grid,
not replica exchange itself.

The remaining predeclared repair branch is a longer hot trajectory.  The original
ratio-0.50 ladder is restored because it had strong adjacent swap acceptance and
56 returns at 500 transitions.  Two final 100-transition canaries multiply only
the hottest replica's step size by `1.5` and `2.0`; all other steps, `L=8`, target,
chart, chains, and topology remain fixed.  These multipliers are derived bounded
hypotheses: the failed hot trajectory length was about `2.26`, so they test about
`3.39` and `4.53`, while the observed hot acceptance `0.81--0.91` leaves room for
larger integration steps.  They add no target-evaluation waves and use the same
selection screens as the ratio canaries.  If both pass, select `1.5x` as the
smallest intervention; if neither passes, stop the campaign.  The ratio-canary
artifacts are SHA-bound, the arms use disjoint CPUs and 2,400-second caps, and no
tuning draw can enter a posterior artifact.

### Exact baseline continuation under the original eight-hour deadline

The post-tuning audit found that the transition-500 warm-up maximum was inherited
from the former `20,000 s` campaign rather than justified as a scientific mixing
limit.  Under the already authorized eight-hour campaign, stopping the research
direction there would incorrectly upgrade failure of the current warm-up candidate
into a continuation veto.  None of the hotter-ladder or longer-hot-step canaries
passed its nomination screen, so they remain excluded tuning evidence.  The only
justified continuation is the exact ratio-0.50 `r8` chain from its verified
transition-500 checkpoint.

The continuation preserves state, target cache, score cache, replica identities,
master seed `(20260811, 8101)`, stateless transition indexing, six betas, step
sizes, `L=8`, four chains, and `24x1` CPU/XLA evaluation.  It checks the latest
300 discarded draws at global transitions `600,700,800,900,1000`; readiness still
requires modern R-hat `<=1.05`, one round trip per chain, and all-chain hot
forgetting.  If ready, retained sampling starts after that global cutoff and is
checked at `1000,1250,1500` retained draws using the previously frozen admission
gates.  No `r9` or `r10` tuning draw enters warm-up or retained diagnostics.

The evidence contract is unchanged: the exact ratio-0.50 `r8` run is the baseline;
warm-up and retained convergence/travel/forgetting gates are promotion criteria,
while nonfinite state/target/score, invalid target status, invalid-path acceptance,
malformed swap permutations, or invalid log acceptance are hard vetoes.  Runtime,
acceptance within a valid run, sign paths, cache residual magnitudes, and raw cold
occupancy are explanatory except where the existing retained acceptance band is
explicitly an admission gate.  Passing would establish only a finite-sample
candidate over the two known regions, not exhaustive mode discovery, posterior
mass authority, predictive validity, superiority, or default readiness.

The skeptical continuation audit passes after two harness repairs.  The loader now
requires exactly 50 contiguous prior chunks ending at transition 500 and verifies
all 650 manifest-bound tensor receipts before using the terminal checkpoint.  The
runner independently re-evaluates the terminal cache, restores the original
finite-or-declared-invalid log-acceptance check and per-chunk target-status audit,
writes immutable 10-transition chunks, and uses the absolute campaign end
`2026-08-11T13:10:18+08:00` with a 300-second finalization reserve.  Thus tuning
time cannot extend the user's eight-hour budget and a successful command cannot
silently answer a different research question.

### Target-free warm-up failure decomposition

The exact candidate subsequently failed all five reviewed warm-up windows and
ended at transition 1000 with zero retained draws.  Before another sampler arm is
proposed, the smallest discriminating question is whether cold-chain disagreement
is caused mainly by unequal occupancy of the two known sign regions or by
disagreement after removing the measured source-region centers.  This diagnosis
loads only verified trace tensors and performs zero target evaluations.

The baseline is each failed 300-draw window ending at transitions
`600,700,800,900,1000`.  The primary explanatory classification compares modern
R-hat of the binary physical-parameter-2 sign indicator with modern R-hat after
subtracting the corresponding measured plus/minus source center from every draw.
Per-chain occupancy, cold sign changes, longest dwell lengths, chain means, and an
exact between-chain sum-square decomposition are explanatory.  Receipt mismatch,
nonfinite trace, wrong terminal status, or any retained draw vetoes the diagnostic.
No result can admit samples, validate posterior mass, establish an exact symmetry,
rank sampler repairs, or support predictive/default claims.

The skeptical audit passes because the diagnostic does not fold modes through an
assumed reflection: the two measured local approximations are similar but not an
exact symmetry, so only source-center subtraction is used and is labeled heuristic
explanation.  Both occupancy and residual components may fail simultaneously; the
classification preserves that mixed outcome rather than forcing one cause.  The
existing `1.05` threshold is inherited solely to compare against the failed warm-up
screen, not promoted as a new scientific equivalence threshold.

The first execution completed computation but failed strict JSON serialization
because at least one binary sign R-hat scalar was nonfinite.  This is a localized
reporting failure, not a trace or sampler retry.  The repair serializes nonfinite
explanatory scalars as `null` while preserving the report's finite/nonfinite counts
and indeterminate classification; the scientific inputs and calculations are
unchanged.

### Dense-mass local-geometry repair canaries

The target-free decomposition classified the terminal failure as mixed, but its
between-chain sum-square magnitudes were `0.165` for occupancy and `3.778` after
source-center subtraction; terminal residual R-hat was `1.1299`.  The current HMC
uses identity mass despite measured mapped local precision eigenvalues spanning
approximately `0.5--367`.  This nominates a fixed dense mass repair before another
ladder intervention.

The fixed mass is the arithmetic mean of the two checked mapped local precision
matrices.  This is a source-derived symmetric compromise hypothesis, not an adapted
or posterior-estimated default.  It changes the measured worst local generalized
frequency from about `19.15` to `1.413`, giving leapfrog stability ceiling about
`1.415`.  Two 100-transition canaries use cold steps `0.35` and `0.70`, respectively
one-quarter and one-half of that measured ceiling, with the unchanged ratio-0.50
ladder, `1/sqrt(beta)` scaling, `L=8`, four chains, and `24x1` CPU/XLA evaluator.

Hard validity, invalid-path self-rejection, adjacent communication, all-chain hot
forgetting, and acceptance `[0.35,0.99]` remain the nomination screens.  R-hat,
occupancy, runtime, and jump distances remain explanatory at 100 transitions.  If
only one passes, it is nominated.  If both pass, `0.35` is the prospectively chosen
smaller intervention, not a statistically supported winner.  If neither passes,
stop rather than weakening screens.  The two arms use disjoint CPUs, unique roots,
and a `2400 s` cap each; their draws cannot enter a posterior artifact.

The skeptical audit passes after adding optional fixed dense mass to the shared
helper with identity as the unchanged default.  Focused tests prove identity
backward compatibility, anisotropic leapfrog algebra, matched Gaussian momentum
and kinetic energy, SPD rejection, and existing TFP parity.  A successful canary
only nominates one fresh warm-up candidate; it does not repair or retroactively
admit the failed ratio-0.50 chain.

The `0.35` arm passed every viability screen with zero invalid paths, acceptance
means `0.515--0.878`, communication on every adjacent pair, and hot local sign
changes `[3,1,3,7]`.  The `0.70` arm failed acceptance and all-chain hot forgetting
and is rejected.  Both results are tuning evidence, not a stochastic ranking.

The admitted `L=8` arm costs about `11.08 s/transition`, so a fresh claim run no
longer fits the original absolute deadline.  One bounded cost repair changes only
`L` from 8 to 4 at the admitted mass and step.  `L=4` remains valid HMC, halves
target waves, and gives nominal hottest trajectory length `4*1.9799=7.9196`, near
the measured mass-metric separation `8.234`.  This geometric comparison is a
source-derived hypothesis, not proof of crossing.  The 100-transition canary keeps
all screens; passing nominates one fresh warm-up under the remaining absolute
campaign budget, while failure ends the campaign without threshold relaxation.

## Defaults and assumptions audit

| Choice | Provenance/status | Justification | Failure mode and early diagnostic |
|---|---|---|---|
| Exact target and physical chart | Measured accepted baseline | Target signatures and representative value/score parity already passed; chart is symmetric over the two known regions | Source drift; recheck signatures, geometry hash, and representative parity in each child |
| Six geometric betas | Baseline hypothesis, not default | All adjacent pairs communicated and hot HMC changed sign in the 12-step run | No round trips; global identity travel remains the material gate |
| `0.05/sqrt(beta)`, `L=8` | Baseline hypothesis | Finite two-region local checks and descriptive acceptance passed | May be inefficient or invalid in tails; finite/status/log-accept vetoes apply |
| Two chains in timing canary | Convenience shape matching baseline | Measures the exact old graph shape | Insufficient for material convergence; timing artifact cannot promote |
| Four threads | Historical baseline | Exact prior run used four threads | May underuse CPU; compare same graph at 32 threads |
| 32 threads on CPUs `32--63` | Convenience topology hypothesis | One NUMA-local 32-CPU slice and ample machine headroom | Thread overhead or interference; compare cached wall time, record affinity and active-service context |
| First plus one cached transition | Smallest timing diagnostic | Separates compile-inclusive from reused-graph cost | One cached timing is noisy; descriptive only and material plan includes margin |
| `2400 s` per child, `5000 s` campaign | Derived from old `4600.29 s` 12-step run plus bounded compile uncertainty | Prevents another unbounded foreground run while allowing both exact topologies | Old wall lacks phase split; timeout is a hard engineering stop, not scientific evidence |
| CPU-only XLA | Explicit diagnostic exception | Matches the established multicore target lane and avoids GPU 0 concerns | Not the repository default production target; artifact must label the exception |
| 12/24 one-row XLA workers | Measured repair hypothesis | One worker per replica-chain row exposes embarrassingly parallel target/score waves; 12-row canary passed at `1.12--1.19 s` per wave | Size-one shards are less efficient per worker and sampling-only; 24-row checkpoint measures scaling before material |
| Four material chains | Repository convergence diagnostic requirement | Modern R-hat requires multiple independent chains and prior staged HMC policy uses four | Doubled target rows could increase wave cost; 25-transition checkpoint stops before material if projection fails |
| Warm-up `300--500`, retained `1000--1500` | Derived from `9.47 s` measured transition and `20,000 s` remaining budget | Minimum retained count makes default bulk/tail ESS thresholds attainable in principle | Autocorrelation may make ESS unattainable; stop under-budgeted at cap |

## Execution plan

1. Repair historical SMC receipt provenance and future schema before transition
   work.  Preserve historical tensors and reproduce terminal estimates without
   target reevaluation.
2. Refactor the diagnostic replica helper to expose one reusable XLA sampler.
   Verify a continued state uses one trace and remains finite on analytic mixtures.
3. Run the four-thread and 32-thread exact one-transition profiles sequentially
   inside one detached user service.  Each child writes atomic progress, immutable
   call traces, and a terminal result.  The supervisor writes a terminal comparison.
4. Use cached timing to choose the next engineering branch.  If the current graph
   is affordable, implement and test globally recomposed checkpoint diagnostics.
   If it is not, implement a process-distributed exact HMC value/score coordinator
   and prove parity against TFP on analytic and exact-target one-step fixtures.
5. Run the four-chain 25-transition timing/travel checkpoint and verify cumulative
   identity composition, exact cache/status validity, acceptance nomination, and
   measured material projection.
6. If only the checkpoint cost screen fails, run the frozen `12x2` topology canary
   and, conditionally, a fresh 25-transition checkpoint.  Stop under-budgeted if
   the same 50%-margin material projection remains above `20,000 s`.
7. Write a frozen material subplan only after the repaired backend has measured
   cost and exact checkpoint semantics.  Run warm-up and retained chunks under a
   detached service; exclude warm-up and diagnose only the concatenated cold trace.
8. Issue no posterior archive until the material travel and convergence gates pass.

Execution paused at step 6, but the resulting stop was retracted after review
found that the cache tolerances were uncalibrated and disproportionate to the
observed numerical scale.  Steps 7--8 remain conditional on the transition-level
materiality canary and a fresh `12x2` checkpoint.  No convergence, travel, or
scientific threshold is being relaxed.

## Compute and attempt budget

- Receipt recovery and focused tests: completed, under one minute of target-free
  CPU diagnostics.
- Timing profile: two sequential children, at most two exact transitions per child,
  `2400 s` child cap and `5000 s` total cap; no retry unless a localized harness
  failure leaves scientific settings unchanged.
- Distributed-backend mechanics work, if triggered: tests and at most one exact
  one-step parity canary under a separately recorded `3600 s` cap.
- Four-chain checkpoint: 25 transitions in five-transition chunks, `900 s` cap.
- Topology repair: one `12x2` transition canary under `300 s`; conditional fresh
  25-transition `12x2` checkpoint under `900 s`.
- Conditional material campaign: `20,000 s` cap, only after checkpoint projection
  and material harness tests pass; no threshold relaxation or extra attempt beyond
  localized unchanged-contract harness repair.

## Skeptical plan audit

| Risk | Resolution |
|---|---|
| Wrong baseline | The profile matches the exact prior target, chart, graph shape, ladder, step, `L`, and two-chain initialization. |
| Proxy promoted | Acceptance, swaps, signs, raw occupancy, and runtime are explicitly explanatory.  Material admission requires travel plus modern cold convergence. |
| Missing stop | Source parity, finite/status, swap identity, retracing, receipt, child, and campaign caps are explicit. |
| Unfair comparison | Four and 32 threads run the same code and state shape sequentially on the same machine; timing remains descriptive because there is one cached replication. |
| Hidden compile cost | First and cached calls use the same returned `tf.function`, and trace count must remain one. |
| Broken chunk semantics | No material use until concatenated swap matrices reconstruct global identities and round trips in focused tests. |
| Stale context | The SMC recovery inventory, physical-transition artifact, local gate, geometry, target signature, and helper sources are hash-bound in the canary. |
| Misleading successful run | A fast finite transition could still have no global mixing; timing cannot promote and the later campaign retains travel/convergence vetoes. |
| Failure for engineering rather than science | A slow graph triggers distributed evaluation; it does not reject tempering, HMC, the target, or the physical chart. |

The audit passes for the four-chain checkpoint.  Material execution remains
conditional on its cost, identity-composition, and nomination result; the stated
conditional thresholds are frozen now so checkpoint observations cannot tune them.

### Topology-repair pre-launch audit

The `12 workers x 2 rows` canary also passes the skeptical pre-launch audit.  It
is SHA-bound to the rejected `r4` checkpoint and checked geometry, and changes
only target-evaluation sharding from `24 x 1` to `12 x 2`.  The physical chart,
four-chain state, six-temperature ladder, step sizes, `L=8`, exact HMC and swap
mechanics, invalid-path behavior, cache tolerances, `1,300`-transition projection,
50% margin, and `20,000 s` cap are unchanged.  CPU affinity, hidden GPU status,
worker XLA/signature identity, finite/status checks, swap permutations, terminal
cache parity, unique output root, and the `300 s` wall cap fail closed.  A single
passing timing draw can nominate a fresh same-topology checkpoint only; it cannot
support a topology ranking or any travel, convergence, posterior, mass,
predictive, or default-readiness claim.

## Pre-mortem

The canary could appear faster at 32 threads because of transient system load rather
than topology.  The result will therefore be descriptive and used only to select
the next mechanics profile.  It could appear valid while chunk identity accounting
resets; the later checkpoint test must recompute identities from concatenated swap
matrices.  A material run could eventually obtain round trips yet retain correlated
cold samples; modern R-hat and ESS remain independent vetoes.  Conversely, a slow
four-thread graph is an implementation result, not evidence against replica
exchange: the measured multicore batch target lane provides a concrete distributed
repair path.
