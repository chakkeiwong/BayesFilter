# Simpler-model NeuTra continuation: execution record

The planned queue completed at **18:39:16 Asia/Shanghai on 2026-09-29**.
`neutra-warm-start-continuation-20260929.service` exited successfully and no
campaign worker remains active. Its final state and accumulated results are in
`artifacts/neutra-warm-start-master-2026-09-29/campaign-r1/continuation-queue-state.json`
and the adjacent `state.json`, `summary.json` and `result.md`. The executed plan is
`bayesfilter-neutra-warm-start-continuation-2026-09-29.md`. The first-allocation
closeout remains historical. Earlier progress snapshots below are preserved
with their original timing; the terminal review immediately below governs the
current interpretation.

Subsequent explanation, 2026-09-30: direct inspection of the saved oracle warm
maps confirms their nearly affine, broad Gaussian shape on the tested grids.
The result and the qualified mathematical explanation for forward-fit
stagnation and RKL collapse are in
`bayesfilter-unwarped-mixture-failure-results-2026-09-30.md`.

## Terminal review

All **111 base training runs**, **15 bounded training repairs**, and **6
duration/capacity controls** completed, producing 132 finite maps. Every map
that passed the declared training screens received downstream qualification.
The result is **56 distinct maps passing the HMC checks**: 53 base maps and
3 repaired maps. A successful queue exit means the planned work completed;
it does not mean every candidate passed or that reliable multimodal training
has been established.

The read-only aggregation and all supporting per-map paths are in
`campaign-r1/terminal-review-r1.json` beneath the artifact root above. It counts
multiple searches for one frozen map once and excludes obsolete mixture
diagnostic revision 1.

| Target | Base maps tested | Base maps passing HMC checks | Additional repaired maps passing |
|---|---:|---:|---:|
| Gaussian | 24 | 24 | 0 |
| Separated unequal mixture | 24 | 0 | 1 |
| Warped mixture | 24 | 15 | 2 |
| Wiggle | 18 | 1 | 0 |
| Ten-dimensional funnel | 21 | 13 | 0 |

The ordinary-SMC warm-start arm passed on the warped mixture at all three
seeds, using the canonical IAF followed by classic RKL and frozen-map HMC.
Every Gaussian arm also passed at all three seeds. These are replicated
passes of finite operational checks. The repaired ordinary-SMC mixture map at
seed 23 passed, as detailed below, but the ordinary-mixture base runs produced
no passing map across the three seeds. Wiggle passed only for plain RKL at seed
37. Funnel outcomes also depended on the training seed. Thus there are useful
trained transports, but a reliable training procedure across the harder
fixtures remains unestablished.

Across all 132 maps, 42 failed coarse coverage, 27 passed training screens but
had no verified kernel in the bounded searches, and 7 obtained verified kernels
but failed downstream checks. The last group contains four numerical chunk
health failures, one AFT-mixture warm-up failure, one waste-free-mixture failure
of retained R-hat/information/precision, and one AFT-wiggle precision failure.
These categories distinguish coverage and training problems, bounded tuning
failures, and actual sampling failures. They are not all evidence against the
warm-start idea. The six longer/wider oracle controls all lost coverage after
RKL; neither tested width nor the 8,192-update endpoint repaired that protocol.

The terminal audit found no missing downstream qualification for an eligible
map. It checked 298 completed GPU training/qualification manifests with no
device-assignment or pre-initialization memory-growth mismatch. The canonical
transport, numerical core and weighted-trainer source hashes were unchanged.
The focused implementation suite passed 44 tests before durable launch.
Actual arithmetic remains FP64 with TF32 enabled as a device setting; these
results do not measure an FP32 training implementation. Selected author
operation parity is checked; full AFT/CRAFT feature equivalence remains
unestablished.

The campaign used 18,050 GPU process-seconds (5.01 hours) and an estimated
31,838.53 CPU core-seconds (8.84 hours). Conservative CPU charging, including
the earlier missing counter, is 32,752.53 seconds (9.10 hours). Remaining
allocation is **46,750 GPU process-seconds (12.99 hours)** and **89,647.47 CPU
core-seconds (24.90 hours)**. No global resource ceiling was exhausted.

The stop is completion of the declared matrix and five-repair-per-seed scope,
not a research-direction rejection or numerical continuation veto. Forty-three
failed base candidates received no training repair because that per-seed cap
was reached: 14 at seed 11, 15 at seed 23 and 14 at seed 37. The terminal review
reconstructs all three seeds; the last master invocation's `deferred_jobs`
contains only the latter two. No incomplete work is described as a pass.

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Close this bounded execution phase | All declared matrix jobs, controls and capped repairs completed | No missing eligible qualification or active worker | Scientific failures remain | Preserve results and plan the next discriminating repair | The whole research objective is solved |
| Retain the 56 passing maps as viable | Declared frozen-map convergence, information, precision and reference checks passed | No declared veto in those accepted runs | Limited seeds, fixed fixture/reference protocol and operational stopping rules | Replicate the successful difficult-target configuration before generalization | Method superiority, default readiness or q20 transfer |
| Reject promotion of the 76 failing maps | Coverage, bounded tuning or downstream criterion failed as recorded | Candidate-specific failures, including four numerical chunk vetoes | Whether initialization, objective geometry, parameterization or tuning is causal | Diagnose the failing stage using preserved checkpoints and matched controls | IAF or forward warm starts are generally ineffective |

| Inference status | Terminal finding |
|---|---|
| Hard veto screen | Candidate-specific coverage and numerical failures preserved; passing maps satisfy declared checks |
| Statistically supported ranking | None; passing counts over different targets/arms do not rank methods |
| Descriptive-only differences | Loss, score tails, region estimates, runtime and success counts |
| Default-readiness | Not established |
| Next evidence needed | A matched explanation and repair of ordinary-mixture forward-fit stagnation/RKL collapse, separately assessed wiggle tuning, then independent downstream replication |

Post-run challenge: initialization and optimizer dynamics can explain why
some teacher clouds yield useful maps while exact-example fits plateau. The
present results do not identify that causal mechanism. A matched derivative,
parameter-update and function-shape diagnostic on saved successful and failed
checkpoints is the next small discriminating study. The common reference bank,
three seeds, coarse reference screen and adaptive repairs limit broader
inference. There is no q20 result or production-readiness claim.

## Progress at 17:36 Asia/Shanghai

The background queue completed seed 11, seed 23 and all four additional oracle
capacity controls. Seed 37 is executing. The read-only snapshot
`campaign-r1/progress-review-20260929T093643Z.json`, beneath the artifact root
above, preserves the per-map results and qualification paths used for these
counts. Earlier sections below retain their original snapshots.

| Base training seed | Completed / eligible | Coarse coverage passed | Maps passing downstream HMC checks |
|---|---:|---:|---:|
| 11 | 37 / 37 | 29 | 18 |
| 23 | 37 / 37 | 28 | 17 |
| 37, incomplete | 16 / 37 | 12 | 7 |

There are 43 distinct trained maps with passing downstream checks, including
one repaired map. Multiple searches on the same frozen map are counted once;
obsolete mixture diagnostic revision 1 is excluded. These counts combine
different targets and methods and do not constitute a ranking. The active job
at the snapshot was ordinary-mixture SMC qualification for seed 37.

The repaired ordinary-SMC seed-23 mixture map is a useful positive result.
It passed the unchanged checks with four chains, 2,000 warm-up and 3,000
retained draws per chain. The maximum retained R-hat over the declared
quantities was 1.00260, region ESS was 2,269, the minimum continuous tail ESS
was 539, and the largest mean MCSE/SD ratio was .0211. The independent reference
screen passed. Its minority responsibility after training was .3435, against
target 1/3. The standard score probe still had large outliers (p99 142.8,
maximum 802.9); this map is not close to a Gaussian everywhere. Its HMC success
is one frozen-map result and does not prove robust training or general coverage.
The exact evidence is in
`campaign-r1/attempts/qualify-repair-mixture-smc-s23-search-repair-r1/` and its
training parent `campaign-r1/attempts/repair-mixture-smc-s23-r1/`.

The duration/capacity intervention did not fix the exact-example warm fits.
Across seeds 11, 23 and 37, both width-8 and width-16 8,192-update candidates
had terminal warm forward KL between .9470 and .9482 and lost minority coverage
after RKL. The width-8 seed-11 case is exact continuation of its original
optimizer, while the other five are fresh candidates. This rejects those
specific duration/width remedies under the tested training protocol. It does
not establish that the canonical IAF cannot represent or learn the target.

Review of 235 completed GPU training/qualification manifests found no missing
memory-growth verification or device-assignment mismatch. Their canonical
transport, shared numerical core and weighted trainer source hashes agree.
The target-module revisions are the already documented wiggle-region repair
and exposure of the unchanged density body for outer XLA composition. No new
infrastructure failure occurred during this durable continuation as of review.

The conservative remainder was 50,535.14 GPU process-seconds (14.04 hours) and
95,845.52 CPU core-seconds (26.62 hours), before charging the active job. Matched
target/arm costs from seeds 11/23 predict about 56 minutes for the 21 remaining
base jobs and their qualifications. Final repairs and any pending qualification
of an already trained map add time; roughly 60--90 minutes is a descriptive
planning estimate, not a deadline or resource allocation. Its forecast basis
is preserved in the snapshot. The existing service continues the work under
the same global budget and five-repair-per-seed limit.

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Keep the repaired SMC-mixture candidate as viable | Frozen-map accuracy/precision/convergence screens passed for seed 23 | Declared numerical and reference checks passed | Only one successful repaired seed; large geometry tails | Finish independent planned seeds | Robust training, method superiority or q20 transfer |
| Reject the tested longer/wider oracle remedies | All six tested endpoints lost coverage during RKL | Coarse coverage veto applies | Initialization, objective geometry and capacity interactions remain unresolved | Complete current matrix before choosing another targeted repair | IAF representational impossibility |
| Continue the seed-37 queue | Valid execution and budget remain | No continuation veto observed | Pending candidates and limited replication | Continue the existing service | Campaign completion |

No stochastic ranking is statistically supported. Passing screens identify
viable candidates; the loss, geometry, runtime and counts above are descriptive.
Default-readiness remains unestablished. A defensible ranking would require a
predeclared paired comparison with adequate uncertainty evidence; it is not
needed to finish the present viability study.

## Completed training and engineering checks

All 37 eligible base training jobs for seed 11 completed. All 37 produced finite
maps; 29 passed the coarse coverage screen. Six lost coverage during RKL. These
counts describe this seed and exclude the separate repairs and capacity controls.
They neither establish posterior accuracy nor rank methods.

The AFT/CRAFT performance repair composes the unchanged analytic target tensor
body into the outer compiled gradient step. The controlled interleaved-update
check reproduced the old slowdown and found normalized gradient differences
below 7e-17 and particle/weight differences below 4e-16. Full-loop pricing then
completed 64 gradient evaluations and six population advances in 24.93 process
seconds. All five AFT and five CRAFT base jobs subsequently completed. This
establishes the tested performance repair; the exact internal TensorFlow cache
mechanism remains unproved.

Before durable launch, the queue audit repaired failed-check exit propagation
and made the five-repair limit per seed cumulative across filtered resumes.
The expanded suite passed 44 tests in
`campaign-r1/attempts/checks-1c51fe5a277b-r1/` beneath the campaign artifact root.
The test attempt charged 35.996 wall seconds and 54.537 CPU core-seconds.
Host GPU 1 was idle before launch, and the first GPU worker verified incremental
memory allocation before device initialization. Other workloads were preserved.

## AFT ordinary mixture, seed 11

The completed AFT warm start followed by classic RKL retained the minority
component: the generated mean responsibility was .32856, against target 1/3.
Heldout forward KL was .17234. The standard 1,000-base-point score probe was
finite, but its residual norm had median .575, p95 37.36 and maximum 232.27.
These are descriptive diagnostics. They nominated a map for downstream testing;
they did not establish that it whitened the posterior sufficiently for HMC.

The initial kernel search found no verified pair. The declared wider search
found a verified L=18 kernel, then four native batched chains completed the
10,000-transition warm-up ceiling. Numerical health passed, but the latest
1,000-transition window had coordinate R-hat 1.260 and binary-region R-hat
1.487, above the 1.05 readiness requirement. No retained posterior draws were
produced. This is a failure of warm-up readiness for that map/kernel and start
configuration, with equilibration inconclusive; it is not an implementation
crash or evidence that every AFT warm start must fail.

The authoritative outputs are
`campaign-r1/attempts/qualify-mixture-aft-s11-search-repair-r1/qualification.json`
and its `posterior.json`, beneath the campaign artifact root. All warm-up chunks
were archived. The declared fresh training-budget repair completed automatically
in `campaign-r1/attempts/repair-mixture-aft-s11-r1/` in 68.58 process seconds.
That candidate stayed finite but lost coverage during RKL: its final minority
responsibility was .000219 and heldout forward KL was 2.977. It was excluded
from downstream qualification. The queue then advanced to the remaining
seed-11 qualifications, beginning with the pending wider Gabrié-wiggle search.
The oracle capacity controls and seeds 23/37 remain queued.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Keep repaired execution route | Controlled deterministic equivalence and focused checks passed | No observed engineering validity failure | Full author feature equivalence remains unchecked | Continue the recorded campaign | Complete AFT/CRAFT author-code equivalence |
| Reject promotion of the seed-11 AFT mixture map/kernel and its repair | Original warm-up readiness failed at 10,000 per chain; repair lost coverage during RKL | Original numerical health passed; repair failed the coverage screen | Dependence on training trajectory, starts and selected kernel | Continue independent seeds and other candidates | AFT or canonical IAF is generally ineffective |
| Preserve the other training candidates | Coarse coverage passed for 29 of 37 | Eight fail that screen; other downstream checks vary or remain pending | Single-seed results and unexecuted qualifications | Complete downstream checks and replications | Coverage alone establishes valid posterior estimation |

| Inference status | Finding |
|---|---|
| Hard veto screen | See candidate finite/coverage screens and per-qualification numerical health; AFT-mixture warm-up failed the declared convergence screen |
| Statistically supported ranking | None |
| Descriptive-only differences | Loss, responsibility estimates, score residuals and timing |
| Default-readiness | Not established |
| Next evidence needed | Independent seeds with frozen-map posterior accuracy, precision and convergence checks |

At the completion of the first AFT-mixture qualification and wider search,
the ledger retained 60,175.54 GPU process-seconds and 111,466.70 conservative
CPU core-seconds; this excludes the then-active training repair. The earlier
lost CPU counter remains estimated at 960 seconds with 1,874 charged for launch
capacity. This uncertainty has not been erased. Current remaining resources
must be read from the live ledger, not this snapshot.

Post-run challenge: the strongest alternative explanation is a useful
coverage-preserving initializer followed by a map/kernel combination with poor
mixing in the region separating the modes. A different independently trained
map that passes the unchanged downstream checks would overturn a general
failure interpretation. The weakest evidence is the single seed and one
selected verified kernel. The campaign makes no q20 or production-readiness
claim.
