# Adaptive R replication ladder

Status: ACTIVE, reviewed continuation of the owner's48 CPU-hour/48 GPU-hour
campaign. Phase15 completed400 evaluations and729 checks; see
`artifacts/iapf-adaptive-score-reference-20260922-01/result.md`.
Root: `docs/plans/artifacts/iapf-adaptive-replication-ladder-20260922-01/`.
Remaining45.226470 CPU and47.745145 GPU hours. This phase uses at most36 CPU
worker-wall hours, two single-thread workers, no GPU, and520 total launches.
Each batch has a900-second cap; one timeout retry per batch may use1800 seconds
in a new directory within the same total budget. The phase may finish under
budget; budget is a ceiling, not a spending target. No new approval is needed.

## Question and frozen scientific contract

Estimate the conditional distribution of fresh final likelihoods from complete
adaptive score/tail8 and QR/tail8 reconstructions at100,300 and1000 independent
learner replicates per paper dimension. Is the apparent stability sustained,
and how large are the Monte Carlo uncertainties around bias, SD and RMSE?
The diagonal score approximation and the cloud-dependent QR fit are distinct
extensions, neither the paper's Eq15 optimizer. Do not infer an original-author
implementation or full paper replication from matching published summary numbers.

Use the same real `iapf_iterate` and `iapf_apf` call chain, paper LG model,
T100,N0=1000,k5,tau.5,ESS.5,sample-SD and six-likelihood stopping window.
Keep max12 iterations/N4000. Every cap/rejection remains a failed learner,
with no replacement or easier retry. Infrastructure timeout retries use exactly
the same data/method/seeds. The final likelihood is a fresh post-stop run.

Only the already declared `after_k` doubling convention is scaled. This is
the existing reconstruction hypothesis, not an accuracy-based selection:
phase15 showed that early doubling changes particle counts before the first
permitted stopping check. Both fitters are retained, including the score arm
that lost its d40 descriptive heuristic screen. The original convention
remains unknown; early-doubling evidence remains in phase15 and is not erased.

Generate one fresh observation set per d5,10,20,40,80 using seed `971000+d`.
For learner label r1..1000, reset RNG to `972000+10000*d+r` for each fitter.
For every heuristic, reset RNG to `973000+10000*d+r`. The larger stride prevents
overlap of dimension-specific1000-seed streams. All stages retain the same
data and accumulate predetermined labels; no reselection after a stage.
No equal-cost or common-random-number equivalence claim follows from labels.

Constructed controls in each dimension/label: bootstrap N10000 (naive prior
proposal), FA-APF N5000 (analytic one-step adaptation, published particle count),
current-observation guide N1000 (cheap myopic twist), full Gaussian guide N1000
(exact backward-information control), and Kalman likelihood (exact certifier).
This includes a strong classical comparator and both enhanced reconstructed
methods. Exact Gaussian controls are analytically optimal here and need no
hyperparameter tuning. Learned methods' numerical settings are frozen hypotheses,
not default policies. No production or model-parameter-score claim is sought.

## Evidence and decision roles

Primary engineering criteria: frozen reference hashes and prior evidence verify;
same-seed replay reproduces phase15 selected d5/d80 likelihoods, counts, statuses
and guides; controller wiring proves a fresh final call; every scheduled label
has exactly six method records; finite states/log weights; full oracle/Kalman
log difference<=1e-8; budgets and source snapshots valid. Failures are continuation
vetoes requiring localized repair. Timeouts get one automated fresh-directory
retry; repeat timeout or another unexplained harness failure pauses the runner
with an explicit checkpoint. Budget exhaustion is a true continuation veto.

Primary scientific quantities: whole-arm completion proportion, ratio mean/SD,
relative RMSE and log-error MSE conditional on each dimension. Do not report a
whole-arm accuracy statistic when any learner is missing or capped. Failed
candidates block promotion, not later scheduled measurements. Ratio underflow
must remain visible; SD alone cannot establish accuracy. The recorded heuristic
promotion veto is observed conditional relative MSE exceeding any cheap control.
No candidate becomes a default and no statistical superiority is automatically
declared. Exact-oracle differences remain visible.

At each stage compute deterministic-seed empirical bootstrap95% intervals for
ratio mean, sample SD and relative RMSE, with2000 bootstrap resamples. Compute
paired-label bootstrap intervals for mean squared relative-error differences
against each of the three cheap controls. These are pointwise Monte Carlo
intervals conditional on this data, not simultaneous or data-population
inferences. Report maximum squared-error contribution and observed ratio tails
to expose under-resolved extremes. Heavy-tail or interval-precision concerns
block strong interpretation, not continuation to the next planned sample size.
No claim of superiority or equivalence to the published paper is automatically
issued at1000. Published SD/resampling/particle counts are contextual checks,
not tuning targets; the original data and implementation are unavailable.

Explanatory diagnostics: every score fit retains target Gaussian min/mean weights,
cloud/precision margins and the same-cloud negligible-floor coefficient
difference; QR retains fit diagnostics. Every controller history, fitted guide,
fresh final diagnostic, seed, count and per-arm CPU time is saved. R timings
include score diagnostic refits and cannot rank algorithm speed. Outer Python
monotonic time charges the budget. Raw particle arrays are reproducible from
seeds and are not retained in every learner record.

## Default audit, pre-mortem and skeptical review

| Choice | Provenance/status | Failure mode | Diagnostic |
|---|---|---|---|
| Paper model and counts | §5.2, inspected local source | Incorrect timing or initial law | Oracle/Kalman and unchanged consumer |
| Two reconstructed fits | Phase15 hypotheses | Objective/diagonal approximation differs from paper | Explicit identity; no replication claim |
| after_k | Existing controller hypothesis | Different N/cost from author convention | Phase15 contrast retained; counts reported |
| tail8, no clipping/ridge | Existing hypothesis with known Gaussian-contour limit | Remote clouds distort fit, singular fit rejects | Per-fit floor effect and relative guards |
|12 iterations,N4000 | Bounded convenience | Valid slow adaptation can cap | Preserve failure; no completion-only selection |
| One new data set/d | Paper-like conditional design | Cannot infer across-data performance | State conditional scope; do not pool d |
|100/300/1000 labels | Bounded published-count ladder | Repeated looks invite selection | Same arms/seeds; no stage selection |
| Bootstrap2000 | Conventional diagnostic uncertainty | Rare tails may be absent | Max loss share and pointwise-only interpretation |
| CPU R and two workers | Authorized independent reference | Not default TF/GPU execution | Explicit hidden GPU and pinned thread counts |

Review PASS: the earlier ten-replicate result cannot justify a ranking. This
ladder retains the failing arm and meaningful classical/oracle controls, freezes
settings before fresh data, and measures complete fresh-final runs. It answers
uncertainty rather than selecting a method to match the paper. The iteration
and particle caps are explicit limitations. Source/timing and actual-controller
checks precede scaling. Phase15 observed CPU demand projects about25.5 core hours
plus overhead for this two-fitter design;36 charged worker hours leaves headroom
and a separate campaign reserve. No change of hardware, data privacy boundary,
target, default, package, or scientific promotion is involved.

Pre-mortem: near-zero bootstrap estimates can have misleading SD; early stopping
can select noisy histories; mean final N can hide occasional expensive runs;
source drift can invalidate a long run; missing replicas can look like success.
Ratio means/RMSE, independent final calls, all count histories, per-launch source
checks and complete-label validation address these risks. Passing every code
check still does not establish Eq15 fidelity or nonlinear performance.

## Execution and automatic continuation

Extend the existing R batch harness with explicit label ranges, seed bases and
controller convention; preserve its default phase15 behavior. Run two same-seed
replays (d5,r1 and d80,r1, later doubling) against preserved phase15 results before
launching the ladder. Batches contain ten labels. At most two are active; each
reserves its full time cap before launch. Fresh versioned attempt directories
preserve retry evidence. The supervisor verifies each batch, charges elapsed
worker time, and proceeds through100,300,1000 without further permission.
Stage summaries, bootstrap intervals, manifests and checkpoint files are written
automatically. Scientific candidate losses do not stop the supervisor. Final
interpretation awaits inspection of the completed evidence, not human permission
to execute subsequent authorized batches.

```text
CUDA_VISIBLE_DEVICES=-1 /home/chakwong/anaconda3/envs/tftwogpu/bin/python docs/benchmarks/diagnose_iapf_adaptive_replication_ladder.py --mode preflight
CUDA_VISIBLE_DEVICES=-1 /home/chakwong/anaconda3/envs/tftwogpu/bin/python docs/benchmarks/diagnose_iapf_adaptive_replication_ladder.py --mode run
```
