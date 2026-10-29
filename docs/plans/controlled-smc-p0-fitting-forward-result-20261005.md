# P0 fitting and downstream result, 5 October 2026

Attempt149 completed all 32 fit/forward comparisons in 168.896125 charged GPU seconds. All 16 saved-author-policy controls passed. Every raw-normal-equation and centered-QR fit, and every subsequent forward call, was numerically valid; neither implementation passed the strict matched-draw comparison on any of its 16 cases.

| Existing fit | Valid fits / forward calls | Policy screen | Exact ancestry | Largest absolute end-of-series logZ difference across all refinements |
|---|---:|---:|---:|---:|
| Raw normal equations | 16/16 each | 0/16 | 0/16 | 207.242341 |
| Centered QR | 16/16 each | 0/16 | 0/16 | 187.465877 |

These are descriptive paired differences, not statistical evidence of inferiority. Resampling makes trajectories sensitive to small coefficient changes. The strict screen was predeclared for localization and is not the owner's final no-worse criterion. A failed equality screen therefore does not reject the replication direction or establish that QR is better than raw.

The numerical consumer ran in float64 GPU/XLA, with TF32 disabled and verified memory growth on both visible GPUs. Each fit and forward graph traced exactly once. TensorFlow peak allocator use was 64,745,472 bytes. No execution error, additional invalid policy, retracing, Python particle loop, or NumPy numerical backend was observed in this checked path. Python loops select cases and save diagnostics; time recursion remains in the shared TensorFlow kernels.

The same forward consumer passes when supplied the author's saved policy. This localizes the observed differences to fitted policies rather than a failed fixed-policy forward translation. Prior attempts91–94 already showed that Octave conditionally changes from ordinary solves to minimum-norm least squares on ill-conditioned Gram matrices, whereas the raw TF endpoint uses an unconditional solve. Centered QR solves the unformed least-squares problem instead. This is a known arithmetic/solver distinction; how much of each full recursive policy difference it explains is not yet quantified.

| Decision | Primary criterion | Veto diagnostic | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Diagnostic execution complete | 16 controls and32 fits accounted for | None invalidates the harness | Six source cases only | Preserve saved policies/forward outputs | Full replication |
| Both implementations fail strict fitting equality | 0/16 strict screens per solver | No additional invalid fit/forward call | Sensitivity, conditional solver semantics and recursive propagation | Verify a source-compatible conditional solver against existing matrices before full recursion | Statistical no-worse failure |
| Continue replication work | Existing source inputs and downstream controls are usable | No continuation veto fired | Untouched P1 criteria and KDM training protocol remain incomplete | Resolve solver compatibility and freeze P1 margins before claims | Author-method repair or E0 authorization |

| Inference status | Finding |
|---|---|
| Hard veto screen | No new numerical-invalidity veto on these source-valid inputs; strict equality fails |
| Statistically supported ranking | None |
| Descriptive-only differences | LogZ differences, ancestry changes, policy-evaluation differences |
| Default-readiness | Not established |
| Next evidence needed | Source-compatible solver/recursion checks; independently calibrated P1 margins and untouched multi-seed comparisons |

Evidence: artifacts/controlled-smc-kdm-replication-20260928/attempt149_p0_fitting_forward/{manifest.json,summary.json,command.log,progress.json}; all fitted policy and downstream tensors are retained. Manifest binds the commit, frozen code, plan, command, source-array hashes, seeds and timing. The frozen harness invokes the actual shared fit and forward endpoints.

Focused prelaunch validation: 51 tests and10 subtests passed in3.674702 CPU seconds, CUDA intentionally hidden. Log: artifacts/controlled-smc-kdm-replication-20260928/p0-fitting-review-20261005-01/tests.log.

Post-run review: the strongest alternative explanation for strict mismatch is amplified finite-arithmetic difference through resampling, not an implementation error in every differing trajectory. Source controls substantially constrain a forward-harness explanation. The weakest evidence is the absence of an untouched no-worse comparison; exact ancestry is not a replacement for that test. No threshold was changed after inspection.

Active balance after this run: CPU151728.165652 seconds, GPU2812.436807 seconds (46.87 minutes), before separately reported routine CPU checks. No worker remains active.

Follow-up completed: [conditional solver result](controlled-smc-p0-solver-semantics-result-20261005.md) reproduces all128 saved Octave solves bit for bit using the Octave-linked library. All60 SVD fallback solves retain rank45. This resolves the independent reference semantics; the TF fitting and downstream qualification gaps remain.

Stage clarification, 6 October: the large maxima above include early refinement stages. At each of the six cases' last captured stages, actual likelihood differences are -2.3973% to +1.0589% for raw and -1.8542% to +1.0993% for QR, using 100*expm1(delta logZ). These are conditional single-refit checks, not independent full TF runs. See the [scale calculation and proposed perturbation calibration](controlled-smc-likelihood-scale-and-perturbation-20261006.md).
