# KDM/IWSG sequential consumer and update validation

This is the next bounded K2 correctness phase in the authorized master. It
checks a full sampling-to-loss-to-update computation. It does not reproduce
the Appendix B training curves: the true coefficients, initial perturbations,
prior/action conventions and optimizer schedule have not been recovered.

## Question, target and evidence contract

Does a native-batched TensorFlow/GPU/XLA implementation of Appendix B equations
(20)–(22) preserve IWSG gradients through five filter steps and one plain SGD
update? Use 25 particles, posterior bandwidth .5 and resampling bandwidth .05,
as specified in Appendix B. Compare against an independent CPU PyTorch
reference executing the unchanged author `resample_particles` and
`weigh_particles` methods plus the real Gaussian KDE density/sampler. Dynamics,
observation likelihood and terminal-state loss follow the paper equations.
The diagnostic supplies their analytic models, not the neural bearings models.

Primary pass: every resampled cloud, propagated cloud, normalized weight,
terminal density, mean loss, seven-parameter gradient and SGD update agrees
within 1e-8 absolute plus 1e-8 relative tolerance on the same realized draws.
Check both IWSG and truncated-gradient controls, identical forward values,
finite normalized densities, and a nonzero IWSG-minus-truncated gradient.
Check the actual TensorFlow update entrypoint calls the shared filtering and
gradient kernels; a detached unused primitive is insufficient. The gradient
oracle is the author autograd computation with identical fixed samples, not
finite differences through stochastic resampling.

The actual author sampler executes, with its categorical choices recorded by a
diagnostic wrapper which calls the original method. Convert those choices to
interior CDF uniforms and infer the realized standard-normal residuals; replay
those inputs through the TensorFlow categorical/Gaussian sampler. This checks
the entire local sampling consumer without requiring identical framework RNGs.
The author reference uses an explicitly declared FP64 diagnostic default to
avoid its sample-output buffer silently downcasting to FP32.

Invalid values, nonpositive bandwidth, nonunit weights, the author's 1e-8
normalization floor firing, inconsistent replay ancestry, gradient discrepancy,
extra traces, CPU fallback or missing source/provenance diagnostics veto the
correctness claim. A failure triggers local diagnosis; it does not reject KDM.
Elapsed time and memory are explanatory. No speed or stochastic-quality ranking
is attempted. A pass supports this constructed analytic consumer only, not
bearings learning, Appendix B parameter recovery, unbiased finite-particle
learning gradients, training convergence or full replication.

## Assumption audit and skeptical review

| Choice | Provenance/status | Failure mode and early check |
|---|---|---|
| H5, N25, posterior .5, resampling .05 | Appendix B | Wrong source settings; record in fixture and assert in both consumers |
| Coefficients [.8,.4,1,-1.2,.2,-.3,.4], sigma .3, gamma .4 | Constructed mechanics hypothesis, shared with exact-reference control | Mistaken author provenance; explicit labels, no convergence interpretation |
| Initial N(.1,1.2), dynamics then observation, resampling before each transition | Constructed timing/prior convention | Does not establish the unknown author convention; preserve inputs and scope |
| Batch four | Small native-batch correctness fixture | Scalar fallback; inspect leading dimension and require shared tensor operations |
| IWSG density/injection epsilons 1e-8 | Author KDE and resampling code | Modified paper expression; match and report source conventions explicitly |
| SGD learning rate .001, no momentum/decay | Diagnostic update hypothesis | Mistaken trained/default status; one update only, compare exact same update |
| FP64 with XLA | Source-comparison numerical convention | Compiler or gradient-loop failure; bounded check, trace counts, memory growth |
| Positive weight normalization floor 1e-8 | Author weight code | Subnormalized densities; fail closed when floor would affect output |

This is an engineering parity study. It does not select a method or claim
learned-method adequacy. IWSG and truncated gradients are the constructed
mechanism/control pair. The master retains the exact mixture-Kalman and
heuristic baseline requirements for subsequent learning comparisons.

Skeptical audit: PASS for the stated correctness question. It would be wrong to
use a constructed five-step update as evidence of paper training replication,
to compare against pathwise finite differences, or to call a bearings neural
consumer the Appendix B analytic model. Those interpretations are excluded.
The reference enters the actual resampling and normalized-weight methods;
full empirical training remains a separate unfinished phase. No new backend
or NumPy exception applies to the TensorFlow implementation.

## Execution and preservation

Implement `k2_sequential` in the existing dispatcher after `k1_iwsg` and the
exact mixture-Kalman check pass. Maximum 180 GPU-charged seconds per attempt,
360 seconds total across at most two attempts, within the inherited GPU balance
and current 12000-second execution allocation. Reference CPU work is a child of
this conservatively GPU-charged diagnostic. It deliberately hides GPU devices.
The total campaign attempt limit remains 18. Run numerical jobs sequentially.

    /home/chakwong/anaconda3/envs/tftwogpu/bin/python -B scripts/run_controlled_smc_kdm.py run --phase k2_sequential

Each fresh attempt saves source snapshots, author commit/file hashes, exact
commands and environment, constructed fixture and sampled replay, complete
checks, device/growth/trace/memory diagnostics and elapsed wall time. Write a
result note with supported decision, vetoes, uncertainties and next action.
The result does not authorize choosing unknown training defaults by convenience.

## Local harness repair after attempt17

All 34 within-mode source value, gradient and update checks passed. The final
IWSG/TG equality check incorrectly called tf.cast on host Python lists, which
first rounded them to float32. Its reported error was 5.41922e-8; direct
FP64 values differ by at most 4.4408920985006262e-16. This invalidates that harness
comparison, not the filter. Use explicit FP64 conversion and add a host-value
roundtrip regression; keep every original tolerance and kernel unchanged.
The failed attempt and 14.452126 GPU seconds remain charged. Rerun within the
original two-attempt/360-second phase cap. Skeptical repair review: PASS; no
changed target, comparator, randomness, numerical method or promotion rule.
