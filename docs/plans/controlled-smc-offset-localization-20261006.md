# Controlled-SMC fitted-policy discrepancy localization

The active question is why the unperturbed F=4.8801, seed 20260930 port
differs from the author by about 1.1% in likelihood when the author's measured
1e-8 random-input response is much smaller. This is the next step authorized
by the owner's instruction to continue. Replication remains incomplete.

## Evidence contract and audit

Compare the unchanged author outputs in attempt147, conditional TF fits in
attempt149, and autonomous TF fits in attempt154. Attempt155 resumed only the
remaining trials; its row file contains earlier results, but their policy
tensors remain in attempt154. Verify input hashes before numerical inspection.
Use d=8, N=512, T=100, the original data, and native seed 20260930. These are
localization inputs, not fresh qualification data.

Follow the actual call chain: saved author cloud/weights -> backward regression
target -> quadratic coefficients -> next fitted policy -> matched forward
result. Check the last backward time first, where the correction is zero,
then earlier times. Distinguish policy coefficient error from error in the
quadratic evaluated on its fitting cloud. A constant policy offset can cancel
in the filter and is not itself a likelihood error. Check ancestor equality
and cloud/weight errors before blaming the fit. Matching total likelihoods
alone does not prove matching particle clouds.

Primary diagnostic: locate the first operation whose difference is supported
by saved numerical evidence and a checked source/TF formula. Diagnostics are
explanatory, not new acceptance tolerances. Input hash mismatch, wrong stage,
nonfinite data, or a source control that cannot reproduce its own result
invalidates the affected diagnosis. A port mismatch is a repair trigger, not
a veto on continuing replication. No ranking, no-worse acceptance, precision
protection, solver default change, or E0 repair follows from this diagnosis.

Assumptions: the pinned Octave implementation defines the current source
baseline; ordinary backslash has conditional Cholesky/LU/SVD behavior, verified
in attempt151. The raw TF route is an unconditional solve, and QR solves the
unformed design problem. Both are existing comparison hypotheses, not proved
translations of the source solver. Inspect condition estimates and downstream
effects rather than selecting the smallest coefficient error. Cheap controls
are the author fixed policy, APF stage zero, and zero-correction terminal fit;
an unrelated filter leaderboard would not answer this implementation question.

Source anchors: StateSpaceModels/Lorenz/cSMC_Resample.m:185-231 (regression and
Gaussian correction), 55-141 (forward quantities), and controlled SMC paper
sections 4.2 and 6.2. This stage checks the source implementation and saved
inputs; it makes no new theorem or claim about the paper's exact likelihood.

Skeptical audit: passed for localization. The plan avoids treating coefficient
agreement, a final likelihood match, or an observed maximum sensitivity as
proof of full replication. The key unresolved alternative is that both ports
agree with the intended least-squares problem but differ from the author's
ill-conditioned finite-arithmetic solve. The saved-input comparison tests this.

## Execution

Start with CPU-only saved-artifact analysis, CUDA deliberately hidden before
TensorFlow import. NumPy/SciPy are independent diagnostic tools only. No
candidate runtime is altered. Preserve inputs, per-time comparisons, source
hashes, command, commit, elapsed time and conclusions under
artifacts/controlled-smc-kdm-replication-20260928/offset-localization-20261006-*/.
Commands use the already approved tftwogpu Python interpreter and the new
docs/benchmarks/diagnose_lorenz_saved_policies.py script.

Budget: at most 1,800 CPU seconds and 600 GPU seconds, within the existing
151,428.37 CPU and 873.45 GPU seconds. Three CPU diagnostic stages and up to two GPU attempts, with fresh
output directories. Start with a saved-data inspection; run a GPU check only
if it answers a remaining call-chain question. GPU work must use escalation,
memory growth, float64, TF32 off, XLA, and the frozen core. Stop on budget,
invalid controls, or a material scope change. Update this plan and the master
checkpoint with the localized repair or next discriminating test.

## Saved-data result and focused follow-up

The first CPU check finished in 1.8142 seconds. Stage-zero particles agree to 7.11e-15, incremental weights to 6.98e-10, and all ancestors match. The largest policy/prediction discrepancy is at time 28; the raw design condition reaches 1.63e8. Source and TF terminal regression residuals are both around 1e-10.

Run capture_lorenz_fit_trace.py on the single case, retaining the original solve expression and only adding observation. Require bit-for-bit agreement of the entire cSMC result and final RNG states with the saved source. Export every design, target, Gram matrix, RHS and solved coefficient. Compare a centered QR independent reference on each identical target. This separates local solver error from propagation of a changed future target. A lower QR residual is explanatory only: the source remains the replication comparator. Timeout 180 seconds; existing CPU budget and stop rules apply. Audit passed: the zero control prevents instrumentation from silently changing the baseline.

## GPU discrimination step

The observed author trace is exactly unchanged, including its RNG. The Octave-linked reference exactly reproduces all 101 solves; 14 use SVD fallback, each retaining rank 45. At time 28 the source residual RMS is 0.0843 versus 0.00706 for centered QR on the same target, with a 0.4516 maximum prediction difference. This demonstrates a finite-arithmetic solver effect but not its complete downstream effect.

Execute dispatcher phase p0_offset_localization: regress every saved source target with both actual TF endpoints, compare QR predictions to an independent centered QR reference, and run six matched stage-one forward policies: source, saved QR, source with QR at the 14 SVD times, the reverse swap, and both swaps at time 28 alone. These hybrid policies are causal diagnostics only; they are not candidate methods. Freeze all choices before the GPU run. Require unchanged-source ancestry and abs(delta logZ)<1e-6; require the saved QR likelihood within 1e-6. These are controls inherited in scale from prior forward checks, not no-worse margins. An independent QR prediction difference over 1e-7 flags a remaining implementation discrepancy. Neither direction of a likelihood change is an accuracy ranking.

Use the RTX 4080 SUPER from the previous experiment, even if physical enumeration has changed. Escalated nvidia-smi found it idle as physical index 1; index 0 is now an RTX 5080. Select by recorded device name before logical-device initialization. Verify growth for all visible devices, then float64/GPU/XLA with TF32 off and the exact frozen core. Per-attempt 300 seconds, two attempts, total 600 seconds from the existing GPU balance. CPU preparation used 3.63 seconds plus the separately manifested reference preparation. Audit passed: input identity, source zero control, direct target comparison and reciprocal swaps discriminate solver behavior without promoting a hybrid or replacing the original method.

Prelaunch supervisor check exposed a stale test enumerating retry limits; the newly registered phase has the declared two-attempt bound. The expectation was updated to include that phase. No numerical kernel was changed.

The first dispatcher start stopped before launching because the historical attempt ceiling was 155. It consumed no run budget. Allocate attempts 156–157 to the already authorized diagnostic within the existing compute balance; no compute is added.

## Completion

Attempt156 completed with all controls passing in24.227988 GPU seconds. The reciprocal substitution removes99.68% of the selected log-likelihood gap, and TF QR matches independent QR to4.72e-11 on all101 source targets. See controlled-smc-offset-localization-result-20261006.md for the limits, decision tables and next compatibility check. No retry is needed; no numerical job remains running. Replication and KDM remain incomplete.
