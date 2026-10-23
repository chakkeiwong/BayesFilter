# LEDH flow guard placement: terminal engineering review

**DISQUALIFIED, 23 October 2026:** owner rejects the N64 workload. All numerical
and speed conclusions below are historical only; do not use them as evidence or
baselines. Replacement: `docs/plans/ledh-sir-n1008-runtime-20261023.md`.

Status: implementation and terminal review complete. Correctness passes in the
tested scope; the overall speed criterion fails.
Plan: [reviewed execution plan](../../ledh-sir-guard-placement-repair-20261022.md).
Design: [mathematical and call-chain analysis](../../ledh-sir-guard-placement-repair-design-20261022.md).
Commands and attempts: [execution record](execution.md).

## Change and checked target

The shared analytical flow now stores determinant matrices and their analytical
tangents over bounded substep groups, then applies the unchanged checked
pivoting determinant and derivative solve to the stacked history. This removes
the factorization dependency from each state/tangent update. Log determinants
and their tangents still accumulate chronologically, with every finite-prefix
check and the original epsilon placement.

Moving innovation Cholesky, state/tangent finite checks, and absorbing failure
status remain inside the substep loop. First failing substeps are observable.
Rejected particles cannot become valid later. Masking after a determinant
failure can be delayed within its group; later placeholders for an already
rejected particle may differ. Accepted computations and rejection coverage are
the correctness target.

There is no unchecked PSD shortcut: negative UKF weights mean that
`valid_predict` does not certify the newly predicted covariance PSD. The history
cap is 16 MiB for matrix/tangent pairs; an unknown static particle count uses
one substep. OT epsilon/iterations remain (1,5), terminal balance remains 32,
and no guide, score objective, model, or numerical tolerance changed.

## Correctness evidence

- 145 distinct CPU tests passed, one skipped. CUDA was deliberately hidden;
  these are mechanics/reference checks. Coverage includes FP32/FP64,
  determinant orders 1/2/9, analytical finite differences, reduced coordinates,
  transient singular/nonfinite and derivative-only faults, absorbing rejection,
  prefix overflow, first-failure ordering, partial groups and stable tracing.
- 61 focused GPU/XLA tests passed. These overlap the CPU tests. Memory growth
  was configured and verified before logical-device initialization.
- All 2928 frozen SIR replay rows are bitwise identical to the retained checked
  baseline: 2496 holdout rows and 432 directional rows, including raw moments
  and analytical scores. Exact validity flags agree. Source/input hashes are
  unchanged and all six functions trace once.
- Plain, conditional and physical-mixture endpoints emit actual grouping
  diagnostics. Tests also exercise generic, reduced-coordinate, affine and
  annealed consumers of the shared implementation.

The baseline is `../ledh-sir-merge-repair-20261022/candidate-02-final-source`.
The candidate is [candidate-01-source](candidate-01-source/).
The full comparison is [replay-analysis.json](replay-analysis.json).
An initially failing diagnostic recorder was repaired to preserve nested return
structures; its numerical assertions were unchanged and pass. This was a test
harness repair, not a change in the accepted numerical computation.

## Endpoint timing

Ratios are candidate/baseline synchronized time. Intervals are paired whole-block
percentile bootstrap 95% intervals, with 20000 resamples and seed 20261022.
Four exposed datasets are pooled within each block. The fixed criterion requires
all six pooled upper limits below one and no dataset/regime lower limit above
one. The two attempts are analyzed separately; results are not pooled.

| Consumer / direction | Attempt 04: eight blocks |
|---|---:|
| Plain / zero | 0.9561 [0.9158, 0.9943] |
| Plain / nonzero | 0.9457 [0.8966, 1.0028] |
| Conditional / zero | 0.9788 [0.9049, 1.0556] |
| Conditional / nonzero | 0.9824 [0.9292, 1.0453] |
| Physical mixture / zero | 0.9915 [0.9522, 1.0331] |
| Physical mixture / nonzero | 0.9329 [0.8734, 0.9945] |

Attempt 04 **fails the speed criterion**. Four pooled intervals include one;
plain/zero on dataset 263043004 has a regression veto, ratio 1.0801
[1.0100, 1.1637]. All 216 paired outputs are identical, source/trace checks pass,
and actual endpoint grouping is verified. Peak allocator bytes are 10033664
before and 10033920 after; current bytes are 1618944 for both workers.
See [attempt 04 analysis](attempt-04-paired-timing/guard-placement-analysis.json).

Attempt 05 completed the independent fixed 48-block precision continuation.
All 1176 paired calls have identical numerical outputs and exact validity flags;
all actual consumer calls use 24-substep groups at each of five time steps.
Sources are unchanged, traces remain one, and neither lane contains live QR.
Allocator peaks are again 10033664 and 10033920 bytes, a 256-byte difference;
current bytes are 1618944 in both lanes.

| Consumer / direction | Attempt 05: 48 blocks |
|---|---:|
| Plain / zero | 0.9470 [0.9195, 0.9766] |
| Plain / nonzero | 0.9243 [0.8974, 0.9517] |
| Conditional / zero | 1.0032 [0.9753, 1.0309] |
| Conditional / nonzero | 0.9921 [0.9634, 1.0209] |
| Physical mixture / zero | 0.9703 [0.9434, 0.9984] |
| Physical mixture / nonzero | 0.9605 [0.9301, 0.9920] |

Attempt 05 **also fails the overall speed criterion**: both conditional-guide
intervals include one. No individual dataset/regime has a lower limit above one
in this attempt. Plain and physical-mixture pooled intervals lie below one,
under the approximately independent-block assumption. These are pointwise
intervals on four exposed inputs, not simultaneous guarantees or cross-model
evidence. The eight-block regression remains recorded; the independent run
neither deletes it nor makes it a reproducible regression. The two attempts
were not pooled, and no interim timing estimates changed the run or candidate.
See [attempt 05 analysis](attempt-05-precision-timing/guard-placement-analysis.json).

## Provenance and scope

Checkout: `ledh-refactor-with-policy-fix`, HEAD
`90a3acebd3965da5bdd64cb165dfdfe2d8cf0649` plus preserved working changes.
Frozen source closure and exact commands are in the per-attempt manifests and
execution record. Replay environment: `tftwogpu`, TensorFlow
`2.20.0-dev0+selfbuilt`, RTX 4080 SUPER, FP64, TF32 off, XLA on,
verified GPU memory growth. The frozen harness records its command-buffer
exception. CPU reference checks do not establish GPU performance.

Replay scope: SIR T=5, N=64, state dimension 18, observation dimension 9,
parameter dimension 3, 24 flow substeps. The 64 datasets and four replications
are frozen; particle seed base is 283030000 and reset seed is 263040002. Timing
datasets are 263043001, 263043004, 263043021 and 263043052, with plain,
conditional and physical-mixture endpoints and zero/nonzero directions.

GPU process walls: focused tests 112.552501 seconds (charged 118), full replay
1064.046347 seconds (charged 1070), attempt 04 resident-worker sum
515.297222 seconds. Attempt 05 charges 1579.730353 seconds: baseline worker
839.582365, candidate worker 740.147988. Charges count complete resident-worker
lifetimes, not just timed calls. This phase used 3283.027575 of its 4200 seconds;
916.972425 remain unused. The local campaign ledger charges 11658.575674 of
12800 seconds, leaving 1141.424326; the parent balance is 1177.193849 seconds.
No process is running and no further timing retry is planned.
Manifest links: [GPU tests](attempt-02-gpu-tests/manifest.json),
[full replay](attempt-03-full-replay/manifest.json),
[timing 04 baseline](attempt-04-paired-timing/before/manifest.json),
[timing 04 candidate](attempt-04-paired-timing/after/manifest.json),
[timing 05 baseline](attempt-05-precision-timing/before/manifest.json),
[timing 05 candidate](attempt-05-precision-timing/after/manifest.json),
[timing 05 completion and charges](attempt-05-precision-timing/result.json).

## Terminal decision and skeptical review

Retain the shared checked batching implementation and first-failure diagnostics
as an engineering repair. Its accepted outputs match the retained checked
baseline, and its rejection coverage survives injected transient faults. This
retention is not promotion as a uniformly faster route: the prespecified overall
speed criterion failed twice. No reduced lane or unchecked determinant path was
introduced to obtain the partial speed result.

| Decision | Primary criterion status | Veto status | Main uncertainty | Next justified action | What is not concluded |
|---|---|---|---|---|---|
| Retain the checked shared refactor | Correctness passes: 145 CPU tests, 61 overlapping GPU tests, 2928 replay rows | No correctness, call-chain, tracing or memory veto observed | Finite fixtures and one replay scope do not cover every covariance or dtype regime | Keep the shared implementation and regressions | Universal numerical validity |
| Reject an overall speed-promotion claim | Failed in attempts 04 and 05 | Attempt 04 has one regression veto; attempt 05 has none under its pointwise checks | Conditional-guide intervals include one; block dependence and exposed inputs limit inference | Preserve both results; any further performance work should first localize conditional endpoint cost | Uniform speed improvement or full pre-merge speed recovery |
| Close this repair phase | All planned implementation, tests, replay and fixed precision work complete | No continuation veto fired; allocation respected | No exclusive GPU cost attribution is available | Hand off the remaining conditional performance question with frozen evidence | Guide quality, posterior validity or HMC readiness |

| Inference status | Finding |
|---|---|
| Hard veto screen | Numerical/parity, exact flags, source stability, single tracing and bounded memory pass; attempt 04 speed regression is preserved |
| Statistically supported ranking | No overall ranking passes the prespecified criterion; attempt 05 pointwise pooled intervals favor batching for plain and physical-mixture consumers only, conditional on the block assumption |
| Descriptive-only differences | Conditional point estimates and individual timing observations; worker lifetime differences include untimed work |
| Default-readiness | This engineering-equivalent refactor supplies no new scientific or HMC admission; scope-specific tuning and prior scientific vetoes remain |
| Next evidence needed | A discriminating conditional-endpoint cost diagnostic before further optimization; fresh broader-scope comparisons before any broader performance claim |

Terminal skeptical review: **PASS for the scoped correctness conclusion; REJECT
the overall speed claim**. The strongest alternative explanation for the partial
timing benefit is device/host scheduling or correlation between adjacent blocks.
A fresh well-controlled timing result whose intervals cross one would overturn
the cell-level speed findings. A missed transient failure, downstream mismatch,
or unsupported consumer path would overturn retention of this implementation.
The weakest evidence is performance generalization: four exposed datasets, one
GPU, FP64 reference mode and pointwise bootstrap intervals. Conditional timing
does not identify which operation now dominates; no exclusive cost attribution
or claim that every slowdown root cause is fixed is supported.

No result here establishes guide quality, posterior validity, HMC readiness,
pre-merge numerical parity, full pre-merge speed recovery, or cross-model
performance. Earlier guide and pre-merge scientific vetoes remain unchanged.
