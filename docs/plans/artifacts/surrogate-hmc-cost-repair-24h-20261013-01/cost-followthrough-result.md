# Surrogate-HMC cost followthrough, 2026-10-15

The five-model scientific posterior campaign remains unfinished. The owner will
supply model-specific scientific priors. The master now rejects scientific
inference with mechanics-only benchmark priors and restores selected precision
and seed scope before validating a handoff. All five accepted-OT tuning, mechanics
and consumer checks passed; these are computational checks, not posterior results.

## Paired cost result

The candidate carries the final Sinkhorn normalizer and its analytical tangent
out of the loop instead of recomputing them. The claimed target is the same
finite likelihood and total analytical score. The full-horizon diagnostic
compares both implementations on identical two-row inputs and fixed designs.
Nominal/offset values, scores, clouds and tangents are exactly equal. Both
additional initial-region comparisons have healthy rows and equal values/scores.

| Model | Baseline mean seconds | Candidate mean seconds | Runtime change | Value/score parity |
|---|---:|---:|---:|---|
| LGSSM | 8.4715 | 8.9305 | +5.42% | Exact on tested rows |
| SIR | 17.7777 | 17.6978 | -0.45% | Exact on tested rows |

Three warmed alternating timings per arm at fixed positions do not support a
statistical speed ranking. The candidate has not demonstrated practical cost
benefit. Its source is archived and the protected batch source is restored
byte-for-byte. The remaining three planned candidate attempts are omitted under
the recorded plan amendment. This decision does not reject surrogate HMC.

The component diagnostics locate LGSSM staged cost mainly in transport/reset
(57.3%). SIR staged cost is proposal/observation 48.2%, density 23.0%, reset 28.8%.
Staged/fused ratios are 1.0154 and 1.2931. Instrumentation changes fusion, so the
fractions are descriptive and are not exact fractions of fused consumer cost.

| Decision | Primary criterion | Veto diagnostics | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Restore protected cost implementation | Parity passed; useful speed benefit not demonstrated | No numerical/health veto on tested rows | Small fixed-position timing sample | Preserve both sources and evidence | Candidate scientifically wrong |
| Retain master repairs | 42 tests pass; all five real scopes validate | Benchmark-prior inference rejected | Scientific priors not supplied | Bind owner priors and refresh affected tuning | Posterior ready or converged |

| Inference status | Finding |
|---|---|
| Hard veto screen | No numerical or health failure in the paired comparisons |
| Statistically supported ranking | None |
| Descriptive-only differences | Runtime ratios and instrumented component fractions |
| Default-readiness | Cost candidate not adopted; posterior not assessed |
| Next evidence needed | Specified priors, matching tuning, actual retained HMC diagnostics |

Post-run red team: XLA may already remove duplicate work, or additional live
loop state may offset the saved reduction. Compiler IR has not been checked, so
these remain possible explanations. Reproducible workload-wide speed gains would
overturn the cost decision. The weakest evidence is the small timing sample.

## Reproducibility and continuation

See cost-followthrough-result.json for exact numbers and comparison paths;
accepted-ot-components-lgssm-01 and accepted-ot-components-sir_d18-01 preserve
component results. budget.json preserves exact commands and wall times. Each
versioned run preserves source snapshots, seeds, device/memory provenance and
its manifest. The protected source and rejected candidate are both preserved
under accepted-ot-normalizer-reuse-baseline-01.

Post-restore engineering checks intentionally hid GPUs: 42 tests passed with two
warnings in 51.27 seconds (post-restore-tests.log), and all five real selected
scopes validated (post-restore-real-selections.json). These checks add no
posterior or statistical performance evidence.

The campaign has 47,849.122 seconds (13.29 hours) and 11 of 140 attempts remaining.
No GPU worker is running. Next remains part of the existing master: implement
the supplied scientific prior densities, supports, coordinate transformations
and Jacobians; refresh affected scope tuning; perform adaptive warmup followed
by cumulative retained sampling to R-hat <=1.01 or the configured 20,000-draw cap,
subject to remaining wall budget. There is no extra validation chain. Waiting
is due to missing scientific target definitions, not an exhausted budget.
