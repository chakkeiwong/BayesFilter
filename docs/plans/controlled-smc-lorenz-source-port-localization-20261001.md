# Matched author-source / TensorFlow localization — 1 October 2026

The next question is where the TensorFlow port departs from the successful
native adaptive runs in attempt86. It is not yet known whether a new precision
protection is needed. Preserve all failed port and safeguard results.

## Evidence contract and research intent

Comparator: the six preserved original-author Octave runs at F=2.5,4.8801,8.5,
seeds20260929/30, N512, d8, T100, full quadratic, adaptive maximum4. The author
paper §4.3 and §6.2, equations42–44, and pinned Lorenz/cSMC_Resample.m:187–234
define backward targets, normal equations and policy accumulation. No source
parameter or numerical operation is changed. The source observation convention
omits its Gaussian constant; compare in that same convention.

Phase A records X, Value and BetaHat immediately after the author's regression
line in a diagnostic copy. Re-execute the six seeds and require exact equality
of every returned policy and SMC field with attempt86. This is an instrumented
reference, not a claim that the copy is byte-for-byte unchanged. Preserve the
original, the minimal insertion diff and both hashes. Require 16 refinements,
101 captures per refinement, matching times, finite arrays, and captured X equal
to the saved source clouds. A numerical or callable-path change invalidates the
capture; no TensorFlow interpretation is allowed from an invalid fixture.

Phase B uses those identical X and Value arrays in the actual shared TF
regression endpoint, with both raw normal equations and centered QR. It also
calls the shared cloud-weight and complete backward-fit endpoints on identical
saved old policies/clouds/weights. Compare all A,b,c, fitted values and source
weights/logZ/ESS. Record the first differing refinement and descending time,
conditioning, residuals, precision margins, trace counts and exact callables.
Thus single-regression differences are separated from accumulated-recursion
differences. No substitute TF implementation is introduced.

The diagnostic completes when every declared comparison has a finite-or-invalid
classification and preserved outputs. Numerical disagreement is a repair trigger,
not a harness failure and not promotion evidence. Missing data, a changed source
replay, wrong endpoint, unstable tracing, timeout or missing required diagnostics
are continuation vetoes. Nonfinite outputs or failed validity flags veto the
affected port route. Existing coefficient comparison tolerances remain
1e-7 + 1e-9*abs(reference); no tolerance search. These are localization screens,
not evidence that the ill-conditioned source coefficients are a mathematical
oracle. Residuals and condition estimates are explanatory only. A PASS for
diagnostic execution cannot satisfy batch-validation or empirical-replication
dependencies. Do not select a solver by these descriptive residuals.

Not concluded: cause of the failing attempt76 random trajectory, general solver
superiority, successful forward replay on common random draws, complete Figure6,
MATLAB equivalence, precision-repair validity, or whole-master completion.
The next repair must be specified from the first mismatching operation; if no
mismatch is found, compare common forward draws before tuning safeguards.

## Default audit, review and execution

| Choice | Provenance / justification | Failure mode / earliest diagnostic | Status |
|---|---|---|---|
| Six source cases | attempt86, source endpoints and generating F | Passing source streams may miss TF failing stream; forbid that inference | Fixed diagnostic scope |
| Instrumented source | Additive capture only at source line193 | Logging changes result; require exact equality of every returned field | Reference only |
| FP64, TF32 off, GPU/XLA | Existing Lorenz parity contract | Compilation/device or finite failure; record placement, traces and memory growth | Reference parity setting |
| Raw equations and centered QR | Author operation vs current port | Conditioning and cancellation; identical targets and complete fits | Two hypotheses, no tuning |
| Existing tolerances | Prior parity screens | Ill-conditioning magnifies coefficient error; preserve prediction errors and do not equate mismatch with a bug | Localization screen |
| No precision projection | Original full-quadratic author route | Indefinite fit; report failed validity without clipping | Source-comparison requirement |

Skeptical review: PASS for localization. Unlike a fresh random comparison this
holds actual inputs fixed, and checks that observation does not alter the source.
It tests claim-bearing shared endpoints. Baselines, stop conditions, environment
and normalization conventions are explicit. Its strongest limitation is that
these are successful source trajectories; they cannot explain the failing TF
trajectory by themselves. This is deterministic code localization, so stochastic
ranking and heuristic efficacy comparisons are not applicable. Independent
review has not been performed; this is the executor's skeptical audit.

Existing authorization covers local diagnostic repair and execution. Allocate
at most90 additional CPU wall seconds and900 GPU wall seconds, two attempts per
phase including infrastructure retries, within the existing CPU500/GPU18000
tranche and inherited grant. Raise attempt ceiling86 to90; do not increase compute
allocations. At opening, tranche has152.346 CPU and3602.135 GPU seconds left.
Use unique attempt directories under the existing campaign root. Each has the
command, source/data/input hashes, seeds, environment, device, wall time, plan,
summary, full log and saved diagnostic tensors. CPU reference hides GPUs.

Commands (same approved dispatcher prefix):

```text
/home/chakwong/anaconda3/envs/tftwogpu/bin/python -B scripts/run_controlled_smc_kdm.py run --phase c1_lorenz_author_capture
/home/chakwong/anaconda3/envs/tftwogpu/bin/python -B scripts/run_controlled_smc_kdm.py run --phase c2_lorenz_source_port
```

Phase B requires a completed Phase A. GPU execution needs trusted/escalated
access and verified memory growth before tensor initialization. This plan
authorizes localization only; batch/full empirical commands stay blocked by
their existing failed numerical-validation dependencies.

Preflight: Python syntax, phase dependencies, original-source completion, free
budget and no overlapping campaign job checks passed. Escalated NVIDIA query
found both local GPUs. Attempt87 captured all six cases with exact equality to
the saved source returns, then failed at SciPy MAT decoding: multidimensional
cell entries remained mat_struct objects. This is a serialization failure,
not source/algorithm failure. Retain its captures and10.539320001 CPU seconds.
The localized decoder now accepts either dictionaries or mat_struct fields;
retry once within the unchanged90-second aggregate cap and phase criteria.

Attempt88 completes all1616 captured regressions,22 source stages and16 refits;
every source result is exactly unchanged. Its CPU charge is11.191161323 seconds.
Attempt89 fails before numerical comparison: strict GPU placement was also
applied to DecodeRaw, which supports CPU only. Charge5.729742912 GPU seconds.
Move binary decoding and tensor serialization/file writing explicitly to CPU;
keep numerical endpoints on GPU with soft placement disabled. This repairs the
harness boundary without changing a solver, criterion, datum or compute cap.
The remaining retry is attempt90.

Terminal90: all354 harness checks pass. Weight/logZ/ESS parity passes all22
source stages;224 coefficient/prediction comparisons fail at unchanged screens.
All11 used XLA endpoints trace once; no nonfinite/precision veto fires on these
source clouds. See controlled-smc-lorenz-source-port-result-20261001.md.
Diagnostic phases are complete; exact repair remains conditional on matrix/
solver localization. Do not treat this execution PASS as a port parity PASS.
