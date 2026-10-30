# Expanded surrogate-HMC component campaign: terminal result

The authorized 48-hour worker budget is exhausted. The master exited normally on 8 October 2026 at 16:25 Hong Kong time (08:25 UTC), reporting **INCOMPLETE**. No campaign worker remains running. Total charged worker time was 172800.219 seconds; the 0.219-second excess is deadline shutdown overhead. The experiment completed 27 of 90 planned model/horizon/particle-count jobs: 17 with the current implementation and 10 explicitly retained from the earlier implementation. Completion means the scheduled comparison ended, not that every scientific criterion passed.

The goal remains **testing surrogate HMC**. This phase compares likelihood values and analytical scores needed by that sampler. It has not run or certified posterior sampling. The candidate is marginal-mixture weighting; the paired baseline is ancestor-conditioned weighting in the same filter. Exact Kalman, refined KSC integration, and range-bearing quadrature/replicated particle references provide the independent accuracy comparisons.

## Coverage and findings

| Particle count | Complete, including retained | Partial | Started without a completed comparison | Not started |
|---|---:|---:|---:|---:|
| 1024 | 22 (10 retained) | 8 | 0 | 0 |
| 4096 | 5 | 1 | 1 | 23 |
| 16384 | 0 | 0 | 0 | 30 |
| Total | 27 | 9 | 1 | 53 |

The nine partial jobs retain dataset/design rows for resumption. Long-horizon timeouts and the exhausted total budget explain unfinished coverage; these do not establish numerical failure. The defined 90-job grid uses LGSSM dimensions 1/3/10/20, native scalar KSC, and native four-state range bearing. Requested non-native nonlinear dimensions still lack agreed model equations.

| Model | Completed jobs | Interpretation of completed parameter-point comparisons |
|---|---:|---|
| LGSSM | 17 | All 51 points meet the declared value and score noninferiority margins. Some, but not all, support lower errors than ancestor weighting. |
| KSC | 6 | Points 0 and 2 meet both margins in each job; point 1 remains inconclusive in all six jobs. |
| Range bearing (M13) | 4 | Six points support lower value and score errors; two comparisons have invalid ancestor baselines and cannot support paired accuracy conclusions. |

Across completed jobs, 69 of 77 parameter-point comparisons meet both noninferiority criteria, six are inconclusive, and two have invalid baselines. These are counts of tested cells, not independent success probabilities or evidence about untested cells. The two invalid-baseline cells are M13 T20/N1024 (five ancestor failures) and M13 T10/N4096 (two ancestor failures); neither has a marginal-candidate or reference invalidity in its recorded comparisons.

The last completed job was LGSSM d=10, T=10, N=4096, with 16 datasets. All three parameter points support lower value and score errors under the declared simultaneous intervals. The full per-cell estimates, uncertainty intervals, reference sensitivities and comparator tables remain in [results.md](results.md) and [analysis.json](analysis.json).

## Decisions and limits

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | What is not concluded |
|---|---|---|---|---|---|
| Preserve completed component evidence | Per-cell classifications above | Engineering and precision checks passed; two invalid baseline cells excluded | Dataset uncertainty and reference approximation | Retain explicit source and scope identities | Universal improvement |
| Keep marginal weighting as an optional candidate | Promising supported comparisons; KSC evidence incomplete | Observed simple-comparator screen blocks promotion in all 75 valid cells | Practical value beyond ancestor comparison; incomplete equal-runtime matches | Continue the planned evidence collection only with renewed compute budget | Default readiness or computational superiority |
| Campaign execution | Budget exhausted, grid incomplete | Terminal stop is the declared budget condition | 63 unfinished jobs and undefined nonlinear extensions | Preserve partial rows; budgeted resumption is part of the existing plan | Research-direction rejection |
| Surrogate HMC | Downstream test not performed in this phase | Cross-process FP32/TF32 repeatability remains unresolved | Posterior correctness and sampling behavior | Address component limitations before the planned downstream sampler tests | HMC convergence or inference validity |

| Inference status | Terminal interpretation |
|---|---|
| Hard veto screen | Engineering and precision checks passed. Two ancestor baselines are invalid; those cells cannot establish accuracy ranking. No new campaign-wide numerical continuation veto caused the stop. |
| Statistically supported ranking | Only cells explicitly marked by the predeclared paired dataset intervals support lower error relative to ancestor weighting. |
| Descriptive-only differences | Heuristic errors, runtime and incomplete equal-runtime comparisons remain descriptive; their promotion-veto role does not establish a statistical ranking. |
| Default-readiness | Not established. The optional candidate remains viable for further testing, with no default change. |
| Next evidence needed | Remaining reference/replication/particle-count coverage, resolution of inconclusive and invalid cells, repeatability assessment, and eventual surrogate-HMC sampling tests. |

The primary materiality margins remain 0.05 absolute log-likelihood-error nats and 0.05 for 0.1 times physical-score RMS error. They are component criteria, not calibrated posterior error tolerances. The strongest alternative explanation for a broad improvement claim is that gains relative to ancestor weighting do not provide adequate accuracy per unit cost against simpler filters. Sparse completed coverage at N4096, no N16384 comparisons, reference uncertainty, and the unresolved cross-process FP32/TF32 behavior are the weakest parts of the evidence. Completing missing cases or a downstream posterior discrepancy could change the conclusion; timeouts alone do not reject the weighting idea.

## Reproducibility and terminal review

The run used the recorded TensorFlow/TFP GPU/XLA path with FP32/TF32 and verified GPU memory growth, on the designated RTX 4080 SUPER. Full commands, environment, seeds, dataset identities, wall times and result paths are preserved in each attempt manifest and in [campaign.json](campaign.json). The budget extension and master command are in [the extension record](budget-extensions/20261007-01/status.json). The exact source is preserved in executed-source.tar.gz.

The terminal standard-library review checked all 837 source hashes against the campaign, found all 27 completed-result files, and verified no campaign worker remained alive. See [terminal-integrity-20261009.json](terminal-integrity-20261009.json). Existing engineering/precision evidence and the 12 controller checks were inspected; no additional numerical experiment or independent external review was performed during this terminal review.

Plan: [expanded execution](../../surrogate-hmc-marginal-expanded-execution-2026-10-06.md), with the [24-hour extension](../../surrogate-hmc-marginal-budget-extension-2026-10-07.md). The master and partial rows are resumable, but the authorized compute budget has been used. Resuming unfinished jobs would continue this plan; it would require additional budget rather than a change of scientific direction.
