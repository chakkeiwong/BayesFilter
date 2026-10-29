# Controlled-SMC RBF retention repair diagnostic — 30 September 2026

The authorized replication campaign continues. This plan tests a named repair
diagnostic after the deterministic mechanism branch closed at63. It does not
alter a public default or claim the paper's M=16 Table2 experiment is reproduced.

## Question and checked basis

Does preserving every positive coefficient in each saved NNLS fit remove the
downstream normalizer discrepancy caused by coefficient truncation? Compare the
four saved full fits from57/58 with independent quadrature and BPF512/BPF6553.
The previous retained16 result61 is historical comparison evidence on the same
observations and policies, with independent forward draws, not a paired sample.

Source anchors: author manuscript/supplement section6.1 and equations13–17;
local text2817–2846 describes full natural-scale NNLS followed by topM.
ToyMultimodal/cSMC_Resample.m203–215 fits all particle knots then deletes all
but M coefficients. TableS1 usesN512,M16. We use the same fixed fitted
coefficients with all positive components retained. This is a compression
repair counterfactual within the controlled-SMC proposal family, not the
published16-component setting and not a full-basis backward refit.

Stage63 establishes that top16 deletion is the principal severe-coverage
mechanism in these saved fits: posterior path mass in q/p_smoothing<1e-8
regions is10.55%/6.73% with retained16, versus0.0066%/0.0038% with the local
full fits. Hundreds of fitting particles occupy the removed regions.
The full fits themselves used downstream targets from the retained recursion.
Preserving them now therefore changes the proposal but does not reconstruct
what an entirely untruncated backward fit would have produced.

## Evidence contract and intent

Primary engineering criterion: actual existing RBFControlledSMC forward
consumer must accept every positive fitted coefficient, with an exact
zero-coefficient-only storage reduction, stable input signatures, GPU/XLA,
verified memory growth, finite output, and zero-policy BPF identity<=1e-10.
Dropping a positive coefficient, failed oracle agreement, missing diagnostics,
invalid samples, extra traces, corrupted inputs or exhausted budget veto
continuation. No tolerance or coefficient clipping is introduced.

Primary diagnostic criterion: natural-scale normalizer mean compared with
the same converged independent quadrature (logZ=-250.62302165662575).
Use the existing20000 bootstrap resamples and nominal familywise99% percentile
intervals across six arms. Classify as disagreement if1 is excluded, resolved
within10% if the interval includes1 and lies within[0.9,1.1], otherwise
under-resolved. This decides only this fixed-policy diagnostic. The unseen-tail
limitation remains; deterministic coverage evidence reduces but does not
eliminate it.

Heuristic adversaries, constructed before these runs:
- BPF512: equal particle count, no learned/twisted proposal.
- BPF6553: the source's R=1 count; historical source-time comparator, not
  contemporary GPU cost matching.
- Zero conditional-mean predictor: tests whether a fitted filter earns its
  complexity when observations leave the state sign ambiguous.
- Independent quadrature: numerical certifying reference in this 1D scope.

Report filtering-mean MSE conditional on y<5,5<=y<15,y>=15, exactly as61.
Heuristic underperformance is a promotion veto, not a reason to hide or tune
away a result. MSE and ESS differences remain descriptive without uncertainty
analysis; no stochastic ranking follows. Runtime comparisons require this
machine's measurements, not the source's particle-count equivalence.

Promotion is not requested. Passing nominates full retention for a separately
specified full backward-refit and empirical replication protocol. It does not
complete Table2, validate other R values, establish filtering superiority,
support HMC, select a default, or certify the paper's exact reported results.

## Defaults, assumptions and stop conditions

| Choice | Provenance/justification | Failure mode | Early diagnostic | Status |
|---|---|---|---|---|
|All positive saved coefficients|63 before/after truncation mechanism|May retain tiny spurious terms or fail for backward-target inconsistency|Exact coefficient preservation; fixed-policy downstream checks|Repair hypothesis|
|Remove only exact zeros|Zero terms contribute exactly zero in a Gaussian mixture|Accidental removal of positive terms|Sum of every discarded coefficient exactly zero and all coefficients nonnegative|Exact storage simplification|
|Original101 observations,R1,N512|Frozen57/58 source-data scope|Not the prose100-observation convention|Fixture hashes and shape/value checks|Frozen diagnostic baseline|
|128 then4096 forward runs|Existing bounded diagnostic ladder|Pilot cannot establish agreement; tails can remain unseen|Final fixed stage plus62/63 coverage evidence|Precision budget, not tuned count|
|Fresh seed bases202609320/330|Separate repair-pilot/final forward randomness|Reuse would undermine independence|Manifest and saved per-row seeds|Convenience choice, no seed search|
|FP64,TF32off,GPU/XLA|Current source-equivalence diagnostic lane|Not repository default precision/production evidence|Device/memory/trace metadata|Reference exception|

Pilot64 uses128 runs per arm, max300 GPU seconds. If numerical controls pass,
estimate final runtime as pilot timed-arm seconds*32+30 seconds. Continue once
to4096 fresh runs per arm only if projected time<=2400 seconds and within the
remaining allocation. Pilot classification is not a superiority or selection
gate. Final65 is fixed, cap2400 GPU seconds, one attempt. No adaptive extra
repetitions or selection among components, bandwidths, stopping factors or seeds.
Localized infrastructure repair is allowed within the same total caps.

Allocation remains CPU300/GPU18000 seconds from the inherited campaign balance;
attempt ceiling63->65 for these two phases. No new GPU-hour grant. Output roots
attempt64_c2_rbf_full_retention_pilot and attempt65_c2_rbf_full_retention_precision.
Every run snapshots source, fixture, saved coefficients and prerequisite results.

Exact commands:
/home/chakwong/anaconda3/envs/tftwogpu/bin/python -B scripts/run_controlled_smc_kdm.py run --phase c2_rbf_full_retention_pilot
/home/chakwong/anaconda3/envs/tftwogpu/bin/python -B scripts/run_controlled_smc_kdm.py run --phase c2_rbf_full_retention_precision

## Skeptical audit

PASS for this bounded diagnostic. The change in M is explicitly named; it cannot
masquerade as source Table2 reproduction. It preserves the saved full fits
without selecting on new forward results. The oracle and heuristic comparisons
are retained; high ESS and small fitting residuals cannot promote the method.
Fresh draws and a fixed terminal count avoid repeatedly sampling until a
favorable result. Early zero-policy identity checks the enlarged consumer path;
discarded-weight equality checks the exact storage reduction. Potential failure
despite correct code is backward-target inconsistency or remaining rare tails.
A pass is still conditional on two fitting seeds and one observation series.

## Executed outcome

Attempts64 and65 completed; all3094 final execution checks passed. All four
controlled normalizer intervals include the oracle; three satisfy the declared
10% precision screen, while QR seed20261009 remains under-resolved.
Conditional filtering errors remain larger than both BPF controls in all groups.
This fixed-policy branch is complete, with no extra repetitions. It nominates
a separately specified full backward refit, not source-M16 or default promotion.
See the terminal assessment in
[the RBF execution result](controlled-smc-rbf-fitting-execution-result-20260929.md).
Charged GPU time:36.257507s for64 and519.198207s for65.
