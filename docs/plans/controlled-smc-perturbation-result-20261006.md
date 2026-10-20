# Controlled-SMC perturbation result, 6 October 2026

The experiment is complete. Small random-input perturbations explain the scale
of some author/port discrepancies, but they do not justify accepting the current
full replication as numerically no worse. Both existing solvers remain viable:
all 174 source trials and all 348 unique TF trials returned valid finite results,
with no additional TF failure. Each solver's unperturbed difference is within
the predeclared median author sensitivity in three of six cases. The strongest
unresolved case is F=4.8801, seed 20260930, where both differ by approximately 1.1%.

## What was measured

These are full adaptive runs, starting from the author's APF policy, fitting
each implementation's own clouds, and applying its own native ESS stopping rule.
They replace single-refit comparisons as the primary evidence for this question.
The observations, N=512, d=8, T=100, maximum four refinements, quadratic basis,
unprotected numerical routes, and native normal/uniform tapes were fixed.
For each case we used four independent Rademacher directions and both signs,
separately for normal draws, systematic-resampling uniforms, and both, at 1e-8.
One direction also ran at 1e-10 and 1e-6. The signs within each pair are dependent;
normal and combined trials sometimes coincide. 174 is not an independent sample
size for statistical inference.

Define delta=logZ_T-logZ_A. Signed likelihood percentage is 100*expm1(delta),
not a percentage change in logZ. Author sensitivity below is the median absolute
likelihood change in eight combined 1e-8 trials relative to its zero control.
The formal scale comparison uses absolute log differences, as frozen in the plan.

| F | Native seed | Author median absolute perturbation response | Raw normal-equation TF difference | Centered-QR TF difference |
|---|---|---:|---:|---:|
| 2.5 | 20260929 | 0.5622% | +0.3185% | +0.3422% |
| 2.5 | 20260930 | 0.2161% | -0.1690% | -0.2842% |
| 4.8801 | 20260929 | 0.4648% | -0.1993% | -0.1984% |
| 4.8801 | 20260930 | 0.0223% | +1.0914% | +1.0993% |
| 8.5 | 20260929 | 0.9350% | -1.5135% | -0.6631% |
| 8.5 | 20260930 | 4.2825% | -9.0831% | -5.6066% |

The earlier maximum 2.4% discrepancy was conditional on an author cloud. Full
recursion produces a 9.08% raw-solver discrepancy and 5.61% QR discrepancy in the
second F=8.5 case. Reporting only the earlier conditional results would conceal
this feedback.

For F=4.8801/seed 20260930, the author median absolute log response is 0.0002228804,
and the largest observed 1e-8 response is 0.0005244388. The raw/QR discrepancies
are 0.01085525/0.01093282: 48.7/49.1 times the median and 20.7/20.8 times that maximum.
Even the largest observed 1e-6 source response for the checked direction is only
0.0006996900. These finite probes do not bound all possible directions, but none
of the tested directions explains this discrepancy.

Uniform-only 1e-8 perturbations left every final likelihood exactly unchanged
(48 source and 96 TF trials); no uniform reflection was needed. Normal-only and
combined source results coincided. The normal perturbations changed ancestry
at a later stage in every source 1e-8 normal or combined trial; some F=8.5 trials also changed the
adaptive stopping stage. Thus the final response includes fitting, resampling
and stopping effects, not just smooth arithmetic noise.

The epsilon ladder is not monotone even for the same signed direction. For
example, F=2.5/seed 20260929 has signed log changes +0.00394225, +0.03796390,
-0.01473102 for the positive direction at 1e-8, 1e-10, 1e-6, respectively.
A 1e-8 probe therefore supplies an empirical scale, not a universal roundoff
bound or a derivative-based stability certificate.

QR showed smaller median local sensitivity than the author in all six cases;
the raw route did in five. That is descriptive conditioning evidence, not
evidence of greater likelihood accuracy. A locally stable implementation can
still have a stable offset. No exact-likelihood oracle was evaluated.

## Log-scale detail

S_A and S_T are medians of absolute changes at 1e-8 in the combined arm. The
last column tests abs(delta)<=S_A, a diagnostic compatibility screen only.

| Case (zero based) | Solver | Baseline delta logZ | S_A | S_T | S_T/S_A | Within typical source scale |
|---|---|---:|---:|---:|---:|---|
| 0 | source_normal_equations | +0.00318006271 | 0.00562495892 | 0.00029085017 | 0.0517071 | True |
| 0 | centered_qr | +0.003416162 | 0.00562495892 | 2.07188768e-05 | 0.00368338 | True |
| 1 | source_normal_equations | -0.00169137269 | 0.00215840116 | 0.000556059988 | 0.257626 | True |
| 1 | centered_qr | -0.00284568612 | 0.00215840116 | 9.04037734e-10 | 4.18846e-07 | False |
| 2 | source_normal_equations | -0.00199504655 | 0.00462849869 | 0.000311034864 | 0.0671999 | True |
| 2 | centered_qr | -0.00198623844 | 0.00462849869 | 0.000198667813 | 0.0429227 | True |
| 3 | source_normal_equations | +0.0108552454 | 0.000222880432 | 0.000365035849 | 1.63781 | False |
| 3 | centered_qr | +0.0109328244 | 0.000222880432 | 2.00316208e-10 | 8.98761e-07 | False |
| 4 | source_normal_equations | -0.0152506187 | 0.00939406228 | 0.00249404551 | 0.265492 | False |
| 4 | centered_qr | -0.00665292753 | 0.00939406228 | 5.16592991e-10 | 5.49914e-08 | True |
| 5 | source_normal_equations | -0.0952242921 | 0.0437809778 | 0.031578088 | 0.721274 | False |
| 5 | centered_qr | -0.0576985691 | 0.0437809778 | 1.89902494e-09 | 4.33756e-08 | False |

## Execution, checks and repairs

Attempt 152 stopped before filtering because pinned Octave 6.4 lacks JSON support.
Attempt 153 used MAT bookkeeping without modifying the author's filtering,
fitting or stopping code. It completed 174/174 valid trials. All six zero controls
were exactly equal to saved author results and final RNG states. 44 binary
stage/kind checks reproduced every previously captured normal/uniform tape.
All direction coordinates are exactly +1 or -1; uniforms stayed in their domain.

Attempt 154 completed 330 unique TF trials before its 1800-second limit.
Attempt 155 verified the frozen core hash and prior row hash, exactly replayed
the interrupted case's zero result, and completed the remaining 18. The native
random tapes and fitting/forward endpoints were unchanged. The first attempt's
partial evidence remains preserved; the final 348-row file records resume
provenance. There were no invalid policies or forward results.

TF used float64, GPU 0 (RTX 4080 SUPER), XLA, TF32 disabled, and verified memory
growth on both visible GPUs. Peak allocator memory recorded in the continuation
was 142,965,248 bytes. Stable signatures and one engine per forcing/solver were
used. The continuation recorded one trace each for fit, forward, and perturbation.
The timed-out first process did not emit its terminal trace-count summary; those
earlier trace counts and its peak allocator value are not available. Do not
claim that the continuation's measurements cover the entire first process.

Twenty focused CPU tests passed, covering reporting units, a failure veto,
outlier-resistant tolerance reporting, antithetic design, and existing supervisor/
forward/fitting reporting. Saved tape integrity and zero controls provide the
executable end-to-end common-input check. Review found no changed numerical
kernel or author-method protection.

CPU attempts charged 299.1697117799 seconds; GPU attempts charged 1938.990619596
seconds (32.32 minutes). Remaining active budget: 151428.36565556822 CPU seconds
and 873.4461873673717 GPU seconds (14.56 minutes). No numerical job remains active.
Routine reporting/tests are CPU diagnostics and recorded separately. No grant
was added.

Plan: [controlled-smc-perturbation-experiment-20261006.md](controlled-smc-perturbation-experiment-20261006.md).
Exact commands, Git 89065bc6354801cb368d5e163ba49fd9c4372d10, environment, seeds,
code/data hashes, wall times, and output paths are in attempts 152-155 manifest.json.
Interpreter: /home/chakwong/anaconda3/envs/tftwogpu/bin/python.
TF 2.20.0-dev0+selfbuilt. Source CPU intentionally hid CUDA; NumPy/SciPy were
limited to independent-reference MAT decoding after Octave execution.

Main evidence:

- artifacts/controlled-smc-kdm-replication-20260928/attempt153_p0_perturb_source/
- artifacts/controlled-smc-kdm-replication-20260928/attempt155_p0_perturb_tf/
- artifacts/controlled-smc-kdm-replication-20260928/perturbation-analysis-20261006-01/
- artifacts/controlled-smc-kdm-replication-20260928/perturbation-preflight-20261006-01/tests-repaired.log
- artifacts/controlled-smc-kdm-replication-20260928/perturbation-analysis-20261006-01/final-validation/summary.json

## Decision

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Sensitivity diagnosis complete | 174 source and 348 unique TF trials; exact zero controls; common tapes verified | No invalidity or additional TF failure | Four directions per case; six reused calibration cases | Preserve the empirical per-case scales | Population-wide error bounds |
| Keep both ports viable; do not accept full replication yet | Typical-scale compatibility 3/6 cases per solver | Both fail the explanatory scale screen at F=4.8801/seed 20260930; this is not proof of inferior exact-likelihood accuracy | A persistent implementation/backend offset versus intrinsic source conditioning remains unresolved | Localize that shared offset on saved clouds and policies, then test a frozen no-worse rule on fresh matched seeds | Source is mathematically correct, TF is wrong, or either solver is superior |
| Keep source arithmetic and replication scope unchanged | No protections or default changes | No E0 work or KDM inference from this experiment | Larger source intrinsic failure region and KDM training remain outside these trials | Resume only relevant existing repair phases after localization | Both replications complete |

| Inference status | Finding |
|---|---|
| Hard veto screen | No additional invalidity in these 348 TF trials; control and input-integrity checks pass |
| Statistically supported ranking | None; no independent-seed non-inferiority comparison |
| Descriptive-only differences | All per-case magnitudes, medians, ranges, sensitivity ratios and stopping/ancestry counts |
| Default readiness | Not established or changed |
| Next evidence needed | Resolve or explain the persistent shared offset; predeclare acceptable downstream margins and use untouched matched seeds |

Post-run skeptical review: a single maximum would make an arbitrary tolerance;
the predeclared median avoids that but is itself a descriptive reference, not a
coverage guarantee. The largest remaining uncertainty is that neither source
nor port has been compared here to an exact likelihood. Reduced sensitivity
could favor the port's numerical behavior while concealing a systematic offset.
A matched independent-seed comparison with uncertainty, or a checked high-accuracy
reference showing the offset is source error, could change the conclusion.
The current finding does not reject controlled SMC or prove the paper defective.
