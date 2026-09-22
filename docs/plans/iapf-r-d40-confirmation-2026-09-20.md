# Independent d40 confirmation of the optional R iAPF

2026-09-20. Owner instruction: refresh the master program and continue. This
bounded stage follows the completed positive-floor validation. Its unused
budget and results remain attached to that closed stage.

## Research intent and evidence contract

Does the previously observed d40 mean likelihood ratio of 1.041181 reproduce
with independent filter randomness, and does the frozen candidate retain
accuracy on fresh data? The candidate is the independent R iAPF with optional
`log_quadratic` fitting and positive-floor power8. It is different from the
paper's equation15 objective. The comparator is the exact Kalman likelihood
on the identical T100 observations. No numerical policy changes are planned.

Two separately evaluated batches each contain32 repetitions:

| Batch | Dimension | Data seed | Repetition IDs | Maximum worker seconds |
| --- | --- | --- | --- | --- |
| Same-data confirmation | 40 | 80000040 | 801:832 | 600 |
| Fresh-data confirmation | 40 | 82000040 | 901:932 | 600 |

The primary accuracy criterion remains a2000-resample percentile bootstrap95
interval for the mean likelihood ratio wholly within[.9,1.1], separately for
each dataset. The bootstrap seed is data_seed+900. No pooling can rescue a
failed dataset. Whether an interval includes1 is explanatory, not a new veto
or a proof of unbiasedness. These small conditional studies cannot resolve
rare-tail bias or support an efficiency/method ranking. Every repetition and
failure is retained; a success-only subset is invalid evidence.

Promotion vetoes: failed/non-finite filter, invalid QR fit, incomplete records,
Gaussian-limit second-moment margin failure, or conditional heuristic failure.
Continuation vetoes: invalid data/source identity, broken numerical identities,
corrupted evidence, missing required diagnostics or exhausted budget. Candidate
inaccuracy instead triggers diagnosis; it does not invalidate the research
direction. Infrastructure repairs may retry within the same total allocation.

The constructed cheap adversaries are BPF10000 (prior propagation with adaptive
resampling), fully-adapted APF5000 (exact one-observation Gaussian proposal),
SIS10000 (prior propagation without resampling), plus the exact Kalman oracle.
Use the existing per-prefix log-likelihood MSE veto against each stochastic
adversary separately on ordinary and large-innovation observations, defined
before filter execution by the Kalman chi-square90% innovation threshold.
Unequal particle counts/costs prevent an efficiency comparison. ESS, fit loss,
floor probability, controller history and timings explain outcomes only.

## Implementation, budget and artifacts

Use the existing captured-source R runner and tail/record validation. Add only
a bounded driver schedule and reporting for these two batches. Core SHA256
must remain c7152fea96d917bf90218bbbbc2bb31c01b0831b439625ff922454b01d748fd0.
Keep N0=1000,k=5,tau=.5,kappa=.5,max_iterations=20,max_particles=16000,
first_full_window doubling and fit_maxit200 fixed. Method seeds remain
53000000+d*100000+replication*10+method_id. All method seeds are fresh.

Budget:1500 summed worker seconds, comprising1400 filtering and100 diagnostics
and reporting;120 additional seconds for focused mechanics tests. Two planned
scientific launches and at most one localized infrastructure retry, never a
new candidate or selected-seed replacement. At most two R workers. CPU-only
base R4.1.2, CUDA_VISIBLE_DEVICES=-1, OPENBLAS_NUM_THREADS=1, OMP_NUM_THREADS=1.
This is the owner's independent reference exception, not a GPU/TF/XLA lane.

Root:docs/plans/artifacts/iapf-r-d40-confirmation-20260920-01/.
Attempt names:attempt01-same-data and attempt02-fresh-data. Each manifest stores
the actual command, source closure, Git commit, environment, seeds, observation
hash, CPU choice, wall time and output paths. The checkpoint records remaining
budget and next action. Terminal summary.json/result.md preserve the separate
decisions, conditional heuristic tables and inference status.

Command template (Python is /home/chakwong/anaconda3/envs/tftwogpu/bin/python):
`python docs/benchmarks/run_iapf_r_replication.py --campaign d40_confirmation
--output <root>/<attempt> --dimension 40 --repeats 32 --first <801|901>
--data-seed <80000040|82000040> --mode replication --fit-mode log_quadratic
--floor-power 8 --doubling-mode first_full_window --fit-maxit 200 --timeout 600`.
Invoke the captured summarize_iapf_r_d40_confirmation.py with --diagnose
<attempt> and then --finalize, passing the explicit repository/campaign roots.

Alongside execution, recheck the paper's algorithms3--5 and equations5--6,
15--16 against the actual runner/controller/proposal/weight call chain. Record
the conditional unbiasedness argument and the unresolved author solver/floor/
data/provenance choices. Do not invent an author setting. More log-quadratic
repetitions cannot close equation15 conformance. No paper-scale replication,
TensorFlow parity, canonical LEDH, KDM, score or HMC conclusion follows.

## Default audit and skeptical review

| Choice | Provenance and justification | Failure mode; earliest check | Status |
| --- | --- | --- | --- |
| log_quadratic/power8 | Previously checked optional fit and calibrated floor | Changed method mistaken for author replication; bind identity/hash | Frozen candidate |
| Original d40 data | Dataset with the unexplained1.041181 mean | Selected-dataset overinterpretation; independent randomness plus fresh data | Diagnostic confirmation |
| New seed82000040/IDs901:932 | Preregistered unused data/method streams | Accidental reuse; compare observation hash and method seeds | Convenience, no tuning |
| 32 repeats and10% tolerance | Same bounded screen as prior stage | Missed rare tails/low power; save all ratios and second-moment margins | Limited screen |
| Controller and positive floor | Unchanged previously tested implementation | Adaptation/normalization error; trace fresh final draw and telescoping identity | Reconstructed settings |
| Heuristic particle counts | Paper BPF/FA counts; prior diagnostic SIS count | Unequal cost mistaken for superiority; conditional veto only | Comparators |

Pre-execution skeptical self-review: PASS for this limited question. The exact
baseline and fresh final draw are preserved; seeds cannot be selected after
seeing outcomes. An interval excluding1 did not become a retrospective veto.
The same-data batch is explicitly selected for diagnosis, and the second data
batch is untouched. Missing observations/fit records cannot be hidden by a
successful summary. Finite-repeat agreement is not unbiasedness proof. The
remaining original-paper fitting gap is substantive and cannot be closed by
this experiment. No independent reviewer or agent is required for this bounded
continuation. Post-run review must consider rare tails and data dependence.
