# SSL-LSTM q=20 physical replica-travel repair terminal result (2026-08-10)

## Current verdict

Update, 2026-08-11: the valid `24x1` route was authorized under one eight-hour
campaign and run to a reviewed transition-1000 warm-up cap.  The implementation
remained engineering-valid, but the candidate failed warm-up and produced zero
retained draws.  The final 300-draw cold window had modern R-hat `1.141610 >
1.05`; travel passed with 20 round trips and per-chain returns `[6,4,5,5]`, and
hot forgetting passed with local sign changes `[1,6,2,3]`.  No posterior archive
or predictive validation is authorized.  The detailed terminal addendum below
supersedes older statements that no material run had launched or that launch
remained blocked by the former budget margin.

The failure has two separate causes and two completed engineering repairs.

1. The original physical replica-exchange implementation is correct enough for
   mechanics evidence but computationally wrong for a material chain: it evaluates
   the 12 replica-chain rows in one TensorFlow graph.  Cached one-transition cost was
   `235.085 s` with four threads and `320.533 s` with 32 threads.  More intra-op
   threads did not expose row parallelism.
2. The process-distributed exact-HMC repair passed one exact SSL transition in
   `9.47256 s` after worker startup.  Twelve persistent one-row XLA workers evaluated
   every leapfrog wave in `1.12--1.19 s`, a descriptive `24.82x` ratio relative to
   the four-thread monolithic cached call.

The `12 workers x 2 rows` topology repair failed the originally declared cache
tolerances.  It
passed worker identity, CPU/XLA, finiteness, target status, invalid-path
self-rejection, and swap-permutation gates, but failed terminal cache parity.  A
targeted immutable-state diagnosis then changed only the batch-two companion of
each row.  The same row's computed target changed by up to `3.70e-7` and its score
by up to `4.66e-6`.  The post-swap cache mismatch was up to `4.07e-7` in value and
`2.12e-6` in score, above the frozen `1e-9` and `1e-8` tolerances.

Those absolute cache tolerances were inherited without target-specific numerical
calibration and were wrongly used as a continuation veto.  The observed target
magnitudes are approximately `35--54` and score magnitudes reach `116`, making the
largest observed differences approximately `1e-8` relative.  This is consistent
with minor floating-point/XLA eigensolver variation and does not by itself show a
material Markov-transition error.

The prior *numerical-invalidity* verdict is retracted.  The valid `24x1` topology
projects to `21,607.97 s` under the prospectively frozen 50% margin, exceeding the
`20,000 s` cap.  The identical-randomness canary clears `12x2` numerically but
shows that it is much slower: `26.008 s` checkpoint-equivalent cost versus
`11.373 s` for the same-run `24x1` reference.  Its conservative material
projection is `50,715.80 s`.  Therefore `12x2` is rejected only as a performance
repair.  No material run, posterior archive, predictive validation, or NeuTra
action was launched.

The four-chain checkpoint subsequently passed every hard gate but was not nominated
because the frozen conservative material-cost screen missed: `21,607.97 s` projected
versus the `20,000 s` cap.  The sampler mechanics and communication screens passed.

## Failure dossier

| Failure | Classification | Evidence | Repair/status |
|---|---|---|---|
| SMC receipt count `782` | Reporting/provenance defect | Flat v1 stage maps overwrote six pre receipt entries at each of 35 nonterminal stages | Recovery verified all 990 child tensors plus two aggregates; estimates reproduced; future stage v2 nests pre/post receipts |
| Original 12-step wall `4600.29 s` | Missing cost attribution | Artifact did not split compilation from cached transition | Reusable XLA sampler measured compile-inclusive and cached calls separately |
| Four-thread monolithic graph | Performance failure | Cached exact transition `235.085 s` | Distributed target/score waves implemented |
| 32-thread monolithic graph | Performance repair failure | Same HLO; cached transition `320.533 s`; only about two cores used during observation | Do not extend monolithic graph; process-distribute rows |
| Distributed `r2` | Harness failure before HMC | Geometry representative key `value` did not exist; actual key is `log_prob` | Failed artifact preserved, SHA-256 `bfc3b2...c6b30f`; localized same-contract `r3` retry |
| Distributed `r3` | Passed mechanics/timing canary | Exact target/status/cache/worker/swap gates all passed | Proceed to four-chain 25-transition checkpoint only |
| Four-chain `r4` | Valid candidate, cost screen failed | `25` transitions, one round trip, all adjacent pairs communicated, all acceptance means in `[0.35,0.99]`, all hard gates passed | Test `12 workers x 2 rows`; do not relax the 50% margin |
| Topology `r5` | Uncalibrated cache screen failed | Post-swap reevaluation disagreed with cached value/score despite all rows remaining status-valid | Preserve evidence; test whether differences materially alter the transition |
| Cache diagnosis | Small batch-pair numerical dependence measured | Changing only the pair companion changed a row's value by up to `3.70e-7` and score by up to `4.66e-6` | Run identical-randomness decision/reversibility canary; do not reject from absolute cache error alone |
| Materiality `r6` | Reporting-only harness failure | Subtracting intentionally unproposed `-inf` swap-log slots produced `NaN` during JSON serialization | Preserve failure and mask non-proposed slots; no target/kernel/seed change |
| Materiality `r7` | Numerical concern cleared; cost failed | All decisions identical; maximum log-acceptance difference `7.52e-8`; `12x2` cost `26.008 s/transition` | Reject `12x2` only as a performance repair; do not run a fresh `12x2` checkpoint |

## Claimed and computed quantities

| Item | Classification |
|---|---|
| Claimed target | Exact row-independent physical log posterior and score inside fixed HMC, followed by exact power-tempered adjacent swaps. |
| Quantity computed | `24x1` computes each row independently and passed cache parity; `12x2` computes two rows together, and the computed result changes when the companion row changes. |
| Relation | `12x2` differs from `24x1` at floating-point scale, but the bounded identical-randomness canary found identical HMC path validity, HMC decisions, and swap decisions for both tested pairings.  The observed numerical difference is immaterial for that canary. |
| Source/artifact anchor | `r5-cache-parity-diagnosis/diagnosis.json`, SHA-256 `1a29bd118fb75481aa86dde0dd6a3353d4f7b729b6e9c6cf0bf55ac2e5774363`. |
| Not proved | Mathematical source of the batch-dependent floating-point result, repeated travel, convergence, correct posterior occupancy, exhaustive mode coverage, posterior predictive validity, or default readiness. |

## Timing evidence

| Backend | Compile/start inclusive | Cached/exact transition | Status |
|---|---:|---:|---|
| Monolithic, 4 threads | `267.219 s` | `235.085 s` | Valid but unaffordable |
| Monolithic, 32 threads | `329.977 s` | `320.533 s` | Valid; no observed speed repair |
| Distributed, 12 x one-row workers | startup `13.824 s` | `9.47256 s` | All canary gates passed |
| Four-chain checkpoint, 24 x one-row workers | startup included in total `310.65 s` | mean `10.8116 s`; cache-adjusted `11.0810 s` | Hard gates passed; cost nomination failed |
| Topology canary, 12 x two-row workers | wall `50.06 s` through screen failure | timing provisional pending transition materiality | Not yet eligible for cost nomination |
| Cache diagnosis, 12 x two-row workers | first evaluation `15.80 s` | cached reevaluation `2.62 s` | Diagnostic only; measured small pair-group dependence |
| Materiality `r7`, 24 x one-row | first evaluation `17.10 s` | transition `11.096 s`; checkpoint-equivalent `11.373 s` | Numerically valid reference |
| Materiality `r7`, 12 x two-row | first evaluation `18.71 s` | transition `25.389 s`; checkpoint-equivalent `26.008 s` | Numerically viable; performance repair failed |

The continuous timing difference is descriptive only: there is one cached
observation per topology and one distributed transition.  It is sufficient to
reject a long monolithic launch on resource grounds, not to claim statistically
supported performance superiority.

## Distributed gate table

| Gate | Result |
|---|---|
| Persistent workers/signatures/XLA/affinity | Pass, 12 distinct workers on CPUs `32--43` |
| Initial representative value/score parity | Pass; value residual `1.70e-9`, latent score max `2.28e-8` |
| Proposal/retained finiteness | Pass |
| Invalid proposal self-rejection semantics | Pass; zero invalid paths observed |
| Terminal accepted-state status | `12/12` valid |
| Cached value/score vs independent terminal evaluation | Exact zero residual |
| Swap matrix permutation | Pass |
| HMC acceptance by temperature | `[1.0,1.0,0.5,1.0,1.0,1.0]`, descriptive one-step values |
| Adjacent accepted/proposed swaps | `[2/2,0/0,1/2,0/0,2/2]`, even parity only in transition zero |
| Hot local sign changes | `0`, descriptive one-step result |

## Decision table

| Decision | Primary criterion status | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Reject long monolithic execution | Cached costs directly measured | No validity veto; resource repair trigger fired | None material to this implementation decision | Retain only as independent mechanics comparator | Sampler or target invalidity |
| Admit distributed backend to checkpoint | Analytic/TFP and exact-target mechanics gates passed | No finite/status/cache/swap/identity veto | Four-chain scaling and repeated transition accounting | Run 25-transition four-chain checkpoint | Convergence or posterior validity |
| Keep posterior archive blocked | No material travel/convergence run exists | Repeated round-trip and modern cold diagnostic gates untested | Whether 1,000--1,500 draws achieve ESS/R-hat within budget | Freeze and execute material only if checkpoint projects within cap | Full posterior authority or predictive equivalence |
| Retain replica candidate after `r4` | Every hard, communication, and acceptance checkpoint screen passed | Frozen cost screen failed (`21,607.97 > 20,000 s`) | Whether batch-two worker shards recover at least 7.44% | Run same-kernel `12x2` topology canary | Sampler failure, convergence, or posterior validity |
| Clear `12x2` numerical concern | All validity and identical-randomness decision screens passed for two pairings | No numerical-materiality veto observed | One transition cannot prove invariant-measure equality | Treat raw cache discrepancies as explanatory, not a hard veto | Convergence or posterior validity |
| Reject `12x2` performance repair | Checkpoint-equivalent cost was `26.008 s`, versus `11.373 s` for same-run `24x1` | Frozen cost screen failed decisively | Timing remains descriptive, but no plausible 7.44% repair was observed | Do not spend 25 transitions on `12x2` | Numerical invalidity or sampler invalidity |
| Keep material launch blocked under the frozen campaign | Valid `24x1` misses the prospectively frozen 50%-margin screen | Cost continuation veto remains | Raw point projection is below the hard wall cap, but changing the predeclared margin is a separate budget-policy decision | Preserve evidence; no automatic material launch | Posterior failure or research-direction rejection |

## Inference-status table

| Inference item | Status |
|---|---|
| Hard veto screen | Numerical screens passed for `24x1` and both `12x2` pairings; `12x2` failed only the cost screen |
| Statistically supported ranking | None |
| Descriptive-only differences | All runtime ratios, one-step acceptance/swaps/signs, CPU utilization, and raw cache discrepancies |
| Default readiness | Not ready |
| Next evidence needed | Human budget-policy decision on whether the already valid `24x1` raw projection may run under the hard `20,000 s` cap without the predeclared 50% prelaunch margin |

## Artifacts

| Artifact | SHA-256 |
|---|---|
| SMC receipt recovery | `3aea988e7b27381a6b62e7a2d452db8251b9bd7d8b9f5e68ad08fcbe711b6d97` |
| Timing supervisor result | `d4a0be4b4ac0a8fe5d4daf1a4a3bfb1425f774e393231a2f987aa5fe248ed4ed` |
| Distributed failed `r2` | `bfc3b2a4b4afcea87010cbf434d21b911171dfa155e86e8f979f799ac9c6b30f` |
| Distributed passed `r3` | `bfcbb5840622e761e052b5dfe398c6ae194570765294a4f1d159091b1569d471` |
| Four-chain checkpoint `r4` | `8276947db5785786567c5194b469c0938907820faf8d1bafd0265b1d4f87adab` |
| Failed `12x2` canary `r5` | `08e9d29fee2af56aeadc3622f01a6f97487384c4446e01f16fc00dedb2ecb3ac` |
| Batch-pair cache diagnosis | `1a29bd118fb75481aa86dde0dd6a3353d4f7b729b6e9c6cf0bf55ac2e5774363` |
| Materiality reporting failure `r6` | `408645656995f123f334a4c92e1c8eb779cd9dd2633540a89e3029b0cd93caa9` |
| Passed materiality/cost-failed `r7` | `5de1e5d217abd9ae293aff81356955c799ed6328e6a66670b019220f6d27aad2` |

## Run manifest addendum

| Field | Value |
|---|---|
| Git commit at execution | `9ebaecc59f792f49bf7b946342ea512e71f5b3e4` with a dirty shared worktree |
| Commands | Detached `systemd-run --user` services invoking the `12x2` canary and cache diagnosis runners, each under `taskset -c 32-47` |
| Environment | `/home/ubuntu/anaconda3/envs/tfgpu`; TensorFlow CPU/XLA; `CUDA_VISIBLE_DEVICES=-1` |
| GPU status | Intentionally hidden; GPU 0 was not used |
| Worker topology | 12 persistent workers pinned to CPUs `32--43`, batch size two |
| Seeds | Canary HMC master seed `(20260810,7301)`; diagnosis has no randomness |
| Wall/caps | Canary failed at `50.06 s` under `300 s`; diagnosis completed at `28.87 s` under `180 s` |
| Output roots | `r5-topology-12x2-canary/` and `r5-cache-parity-diagnosis/` under the plan artifact root |
| Plan/result | This plan and terminal result file |

## Post-run red team

The evidence supports ordinary floating-point/XLA eigensolver variation rather
than a material transition defect: two batch pairings produced identical decisions
and small forward/reverse errors.  The result would be overturned by broader
states or longer paired replay showing decision disagreement or growing
reversibility error; one transition is not a universal guarantee.  The topology
still fails as a speed repair, independently of numerical validity.
The weakest evidence remains scientific coverage: every
chain starts in one of two known regions, so even a future converged two-region
campaign would not prove there is no third mode.  SMC and replica exchange answer
separate two-region weight and transition questions; neither establishes
exhaustive posterior coverage.

## Transition-1000 material continuation addendum (2026-08-11)

### Verdict

The exact ratio-0.50 candidate is rejected for failure to equilibrate the cold
chains by its reviewed transition-1000 warm-up cap.  This is a sampler-tuning or
mixing failure of the current candidate, not an implementation, target, numerical,
or budget failure.  The continuation used only `5,344.60 s` and ended with
`12,893.14 s` remaining because further draws would have crossed the frozen
candidate cap.  All 1,000 transitions are discarded warm-up; retained draws per
chain are exactly zero.

The claimed target was the exact physical SSL-LSTM log posterior and score inside
fixed `L=8` HMC at six inverse temperatures, with exact adjacent replica swaps.
The quantity actually computed matches that claimed target on the one-row route:
all per-chunk terminal cache reevaluations had exact zero value and score residual,
target status remained valid, and worker target/adapter signatures remained fixed.
What was not obtained is an equilibrated cold-chain sample.  Therefore the run
cannot support posterior estimates, predictive checks, mode-mass estimates,
exhaustive-mode claims, superiority, or default readiness.

### Warm-up evidence

| Global transition | Latest-window max modern R-hat | Round trips by chain | Hot local sign changes by chain | Readiness |
|---:|---:|---|---|---|
| 600 | `1.133129` | `[7,5,1,7]` | `[4,5,2,1]` | Fail: R-hat |
| 700 | `1.056483` | `[2,3,2,12]` | `[1,5,1,8]` | Fail: R-hat |
| 800 | `1.159304` | `[3,2,3,10]` | `[0,0,1,8]` | Fail: R-hat and hot forgetting |
| 900 | `1.073534` | `[6,3,4,7]` | `[1,1,0,9]` | Fail: R-hat and hot forgetting |
| 1000 | `1.141610` | `[6,4,5,5]` | `[1,6,2,3]` | Fail: R-hat |

Every window passed the per-chain round-trip requirement and every chain saw both
hot signs.  The nonmonotone R-hat path and intermittent absence of local hot sign
changes show that simply extending this unchanged candidate did not establish
stable cold equilibration.  Round trips alone are insufficient: they prove global
identity travel across the ladder, not equality of the cold marginal distributions.

### Decision table

| Decision | Primary criterion status | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Reject exact ratio-0.50 candidate | Warm-up readiness failed at all five new milestones; terminal R-hat `1.141610 > 1.05` | No hard engineering or numerical veto; reviewed warm-up cap fired | Whether between-chain disagreement is mode occupancy or within-mode geometry | Run a target-free decomposition on the verified trace before proposing another sampler arm | Replica-exchange direction, target, or implementation invalidity |
| Keep posterior archive blocked | Zero retained draws | Warm-up admission veto fired | No retained R-hat/ESS or downstream evidence exists | Do not run predictive validation | Posterior correctness or predictive equivalence |
| Preserve distributed backend | All 100 continuation manifests and 1,200 total tensors across `r8+r11` verify; caches exact | No target/status/finiteness/swap veto | Longer-run performance ranking remains descriptive | Reuse backend only under a new reviewed sampler hypothesis | Default readiness or superiority |

### Inference-status table

| Inference item | Status |
|---|---|
| Hard veto screen | Pass for engineering and numerical validity; no hard failure recorded |
| Statistically supported ranking | None; this run tests one fixed candidate |
| Descriptive-only differences | R-hat trajectory, round-trip counts, hot sign-change counts, runtime, and raw occupancy |
| Default readiness | Not ready; candidate rejected and no retained posterior exists |
| Next evidence needed | Decompose between-chain disagreement by mode occupancy versus within-mode distributions, then review the smallest targeted repair |

### Run manifest

| Field | Value |
|---|---|
| Git commit at launch | `9ebaecc59f792f49bf7b946342ea512e71f5b3e4`, dirty shared worktree |
| Command | Detached transient user service running `taskset -c 32-63 /home/ubuntu/anaconda3/envs/tfgpu/bin/python docs/benchmarks/resume_ssl_lstm_q20_physical_distributed_replica_material_2026_08_11.py` |
| Environment | `/home/ubuntu/anaconda3/envs/tfgpu`, Python `3.13.13`, TensorFlow `2.20.0` |
| Device | CPU-only, `CUDA_VISIBLE_DEVICES=-1`; GPU 0 not used |
| Topology/XLA | 24 persistent one-row workers on CPUs `32--55`; XLA true; parent CPUs `32--63` |
| Seed/kernel | master seed `(20260811,8101)`, six fixed betas/steps, `L=8`, four chains |
| Wall/budget | `5,344.60 s` continuation; absolute campaign `05:10:18--13:10:18 +08:00`; `12,893.14 s` remained |
| Output | `r11-material-24x1-resumed/`; 50 immutable continuation manifests and 550 continuation tensor receipts verified independently |
| Result SHA-256 | `0fbec0c372008d406953908a30b6aa66a27d843781a93dba5ae52cd98235c66b` |

### Post-run red team

The strongest alternative explanation is that recent-window R-hat is responding
to unequal mode occupancy among otherwise valid chains rather than inadequate
within-mode integration.  That would still invalidate ordinary cold-chain
admission, but it would call for a different repair than changing the local mass
matrix or trajectory length.  Conversely, repeated round trips can coexist with
poor cold equilibration when replicas traverse the ladder without sufficiently
decorrelating within each region.  A target-free decomposition of the verified
trace is therefore the next smallest discriminating artifact.  It cannot itself
promote a sampler or posterior.

## Dense-mass repair addendum (2026-08-11)

The target-free diagnosis found a mixed failure at the terminal window.  Sign
occupancy fractions were `[0.333,0.727,0.660,0.447]` with sign-indicator R-hat
`1.2325`; after subtracting the corresponding measured source-region center,
residual R-hat still reached `1.1299`.  The between-chain sum-square decomposition
was `0.165` for source-center occupancy and `3.778` for residual disagreement.
This does not prove a unique causal partition, but it rejects occupancy-only and
local-geometry-only explanations.  The larger observed residual term nominated
local preconditioning first.

The shared exact HMC helper now supports an optional fixed dense mass while
preserving identity mass as the default.  The implementation draws momentum from
`N(0,M)`, advances position with `M^-1 p`, and uses kinetic energy
`p^T M^-1 p/2`.  Sixteen focused distributed/TFP tests pass, including bitwise
identity backward compatibility, nonidentity closed-form leapfrog algebra,
momentum/kinetic matching, and non-SPD rejection.

The source-derived mass is the arithmetic mean of the two checked mapped local
precision matrices.  It is a tuning hypothesis, not a default or posterior-adapted
mass.  Three 100-transition tuning arms completed:

| Arm | Hard gates | Acceptance screen | Hot changes by chain | Travel by chain | Decision |
|---|---|---|---|---|---|
| dense mass, `step=0.35`, `L=8` | Pass; zero invalid paths | Pass, means `0.515--0.878` | `[3,1,3,7]` | `[0,1,0,3]` | Viable tuning candidate |
| dense mass, `step=0.70`, `L=8` | Pass; zero invalid paths | Fail, several means below `0.35` | `[0,6,4,0]` | `[0,1,0,0]` | Rejected |
| dense mass, `step=0.35`, `L=4` | Pass; zero invalid paths | Pass, means `0.629--0.935` | `[0,4,5,4]` | `[2,2,2,5]` | Rejected: chain 1 hot forgetting failed |

The `L=4` arm was descriptively fast at `5.523 s/transition` and had strong travel,
but passing travel cannot override the failed all-chain forgetting screen.  The
only viable arm is `step=0.35, L=8`, measured at `11.081 s/transition`.  At the
time it was identified, a fresh 300-transition warm-up plus 1,000 retained draws
would require about `14,405 s` before startup/finalization, exceeding the remaining
absolute campaign time.  No full repair run was launched and no threshold was
weakened.

### Dense-mass decision table

| Decision | Primary criterion status | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Keep dense `0.35/L8` as a viable candidate | All 100-transition viability screens passed | No hard veto | Short tuning run cannot establish warm-up or posterior quality | In a new budgeted campaign, run fresh warm-up from balanced sign-separated starts, then retained gates unchanged | Improvement, convergence, superiority, or default readiness |
| Reject dense `0.70/L8` | Acceptance and hot forgetting failed | Candidate selection veto | A different step might work, but `0.35` is already viable | Do not extend this arm | Dense mass direction invalidity |
| Reject dense `0.35/L4` | Acceptance/communication/travel passed; hot forgetting failed | Candidate selection veto | Failure may be stochastic, but no remaining reviewed replication budget | Do not use speed to override forgetting | That `L=4` is globally impossible |
| End current campaign without posterior | Viable `L8` full run is under-budgeted at the absolute deadline | Budget continuation veto | Actual fresh warm-up length and ESS remain unknown | Preserve candidate and start a new reviewed campaign only with new budget | Posterior correctness or predictive validity |

### Dense-mass artifacts

| Artifact | SHA-256 |
|---|---|
| Target-free warm-up decomposition | `4946e5b33bf3258d00ef86ea2bf6067b7928e74ce85c871990975e760c15832c` |
| Passed dense `0.35/L8` canary | `3b785f78ca2e18e44162756cde7a69088c8bb3723f2549dee1106f4567dc63f0` |
| Rejected dense `0.70/L8` canary | `8faf518826660a2816053a54e62cc2f249fd60081ae7969bcfe64a9754a4cd7a` |
| Rejected dense `0.35/L4` canary | `2eead8ce2fd4ca8ffdb9f340184dcf67cb7b3df29b7a5e442a34b87e2fdc0fe6` |

All 30 tuning state receipts independently verify.  These draws are tuning-only
and ineligible for posterior use.
