# Controlled-SMC RBF full backward-refit diagnostic — 30 September 2026

## Question and evidence contract

Does retaining all positive coefficients throughout the backward recursion
produce a numerically valid controlled-SMC policy whose normalizer agrees with
the independent quadrature? Stage65 changed only the final proposal; it did
not repair the fitting targets. This is the next discriminating mechanism
test, not another repetition of that completed branch.

Use exactly the two saved training clouds from57, seeds20261009/20261010,
N512,H101,R1,bandwidth factor0.3,one refinement. Fit through the existing
RBFBackwardFit with components512, so each earlier target integrates the
complete next-time fitted policy. Keep the unregularized QR NNLS, capacity128,
iteration cap2560 and stopping factor0.1. No coefficient clipping, ridge,
bandwidth search, training-seed search or extra refinement is introduced.
The source basis and recursion are anchored at author supplement6.1,
local text2817–2846, and ToyMultimodal/cSMC_Resample.m203–215.
Full retention differs from the source top16 compression; label it accordingly.

Primary engineering criteria: all202 fits valid, finite nonnegative
coefficients with positive total weight, no capacity/rank/iteration failure,
every positive coefficient retained, stable single traces, GPU/XLA and
verified memory growth. Recompute the backward targets independently with
the closed Gaussian-product formula and check max absolute error<=1e-10.
Check terminal observation-only target and constants as well. Audit the
actual RBFBackwardFit.fit -> _fit -> NNLS call and subsequent forward consumer.

Primary downstream criterion: fresh normalizer estimates versus the existing
converged independent quadrature. Preserve the previous nominal familywise99%
bootstrap rule with20000 resamples: disagreement if1 excluded, resolved10%
if1 included and the interval lies in[0.9,1.1], otherwise under-resolved.
Evaluate two refitted QR policies, the two previous QR full-saved-fit policies,
BPF512 and BPF6553. Old saved fits are mechanism controls, not source-M16 runs.
No stochastic ranking is inferred from these marginal intervals.

Constructed heuristic set: BPF512 (same particles), BPF6553 (source R1 count,
not current GPU cost matching), zero conditional-mean predictor (sign
ambiguity), and independent quadrature (certifying 1D reference).
Preserve conditional filtering MSE groups y<5,5<=y<15,y>=15. Loss to a
heuristic vetoes filtering/default promotion; it does not veto the normalizer
investigation. MSE, ESS, fitting residuals, runtime and cross-solver coefficient
drift are explanatory only. No tuning against heuristic outcomes.

A failed numerical guard or oracle check invalidates the affected execution
and stops dependent phases. A normalizer discrepancy rejects this candidate
under the diagnostic, not controlled-SMC theory. A passing fixed-policy
diagnostic does not complete source Table2, demonstrate performance at other
R values, establish superiority, validate gradients/HMC, or change a default.

## Default/assumption audit

| Choice | Provenance and justification | Failure mode | Early diagnostic | Status |
|---|---|---|---|---|
| All512 basis coefficients during recursion | Exact full NNLS expansion before source compression; isolates mechanism63 | Larger basis could retain unreliable tails or alter backward targets badly | Positive-coefficient preservation and recurrence identity, then downstream oracle | Repair hypothesis |
| Same two saved clouds | Paired mechanism intervention, independent of new forward draws | Only two fitted seeds, one data series | Report scope; no population-level replication claim | Diagnostic control |
| Stopping factor0.1 | Prior calibration56, primitive and full-fit checks58 | Does not guarantee exact coefficients when design nearly singular | Original rank/capacity/KKT guards at all times; preserve margins | Reviewed diagnostic setting, not new default |
| Capacity128/no ridge | Prior active sizes<=92; preserves NNLS objective | Full refit could exhaust capacity or fail rank | Fail closed, no automatic clipping or ridge | Bounded implementation hypothesis |
| FP64/TF32off | Source-equivalence reference lane | Not default production precision evidence | Recorded environment and traces | Reference exception |
|128 then4096 fresh draws | Fixed diagnostic budget65 | Bootstrap may miss rare tails | Fixed final stage; preserve maxima and under-resolution | Precision budget, not selection |
| Same H101,R1,bandwidth0.3 | Source-data convention and source R1 setting | Does not reproduce proseH100 or other R values | Manifest, shapes and data hashes | Frozen diagnostic scope |

## Execution and budget

Attempt66 c2_rbf_full_refit: max900 GPU seconds, one attempt; save every fitted
array and diagnostic. Attempt67 c2_rbf_refit_pilot:128 draws per six arms,
max300 seconds. If its engineering checks pass, project final cost as timed
pilot arm seconds*32+30; run attempt68 c2_rbf_refit_precision once with4096
fresh draws per arm only if projected cost<=1500 seconds and within the
remaining allocation. No optional stopping or added repetitions after68.

The existing CPU300/GPU18000-second allocation is unchanged. At65 close,
5275.748128 GPU seconds remain; these phases cap total use at2700 seconds.
Raise the attempt ceiling65->68 for these named phases only. Inherited
remaining GPU32.840847h is not a new48-hour grant. Each attempt gets a fresh
versioned output under artifacts/controlled-smc-kdm-replication-20260928/.

Forward seed bases202609340 (pilot),202609350 (final); +1 for BPF6553,
+2 for zero-policy identity, +9 for bootstrap. Training clouds are reused
deliberately to isolate the recursion change. No new fitting randomness.

Commands use the approved dispatcher prefix:

    /home/chakwong/anaconda3/envs/tftwogpu/bin/python -B scripts/run_controlled_smc_kdm.py run --phase c2_rbf_full_refit
    /home/chakwong/anaconda3/envs/tftwogpu/bin/python -B scripts/run_controlled_smc_kdm.py run --phase c2_rbf_refit_pilot
    /home/chakwong/anaconda3/envs/tftwogpu/bin/python -B scripts/run_controlled_smc_kdm.py run --phase c2_rbf_refit_precision

## Skeptical audit and pre-mortem

PASS for a bounded repair diagnostic. The changed coefficient count is named
explicitly and no exact-source Table2 claim is possible. Reusing clouds is
intentional mechanism control, not independent training replication. The
oracle controls separate a broken normalizer harness from a failed candidate;
filtering MSE cannot silently become the source normalizer criterion.
High ESS, small fitting residuals and success of the solver do not qualify
the downstream method. The run could pass while unseen tails remain or while
generalization to other fitting clouds fails; those remain nonclaims.
It could fail because full recursion changes the NNLS conditioning or because
capacity128 is insufficient; saved per-time margins and counts distinguish
those from evidence against the mathematical method. A validity failure stops
this branch for diagnosis; it does not invite silent parameter changes.

## Executed outcome,30 September2026

Closed after68 as planned. Full backward refit66 passed all202 fit guards and
all10 recurrence/representation checks; pilot67 passed118 execution checks and
the cost gate. Final68 completed24,576 estimates and3094 execution checks.
Both fully refitted policies passed the predeclared normalizer screen
(mean ratios1.00188069 and1.00204889, intervals containing1 and within10%).
Both old saved-fit controls failed the fresh normalizer screen. Filtering
heuristic underperformance remains a promotion veto for every controlled arm.
No additional repetitions are authorized by this branch.
Full interpretation, budgets and manifests are in the
[terminal result](controlled-smc-rbf-fitting-execution-result-20260929.md#full-backward-refit-terminal-assessment-attempts6668).
