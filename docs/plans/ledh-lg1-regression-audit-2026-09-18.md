# LG1 fixed-input regression audit — 2026-09-18

Active question: did the graceful-failure changes cause the LG1 calibration
failure, or does the starting implementation already fail on the same inputs?
The user's regression question authorizes this bounded diagnostic. It is not a
fourth calibration candidate and does not reopen the completed tuning campaign.

Compare baseline commit `6fbcf3147660c40d5d5644bbcbcc9fadbcb06aef` with execution
commit `8a4cffa2`, using an existing failed candidate: calibration replicate 2,
phi=-0.9, N=1024, T=10, float32/TF32/XLA, scrambled Halton inputs, flow steps 3,
coordinate cap 16, and all remaining frozen controls. Materialize inputs once
with the current fixture and give their identical float32 values to both
implementations. Use each revision's canonical analytical recursion with
explicit stationary initial states, covariances, and their analytical tangents.
Replay the current consumer if the direct recursion comparison requires it.

Evidence contract: the comparator is the pre-change code on identical inputs,
not a different historical model or successful finite-value screen. Compare
likelihood, analytical score, per-time filtering moments and reset moments.
A scaled difference above 1e-4 triggers localization; small FP32 reduction
differences alone do not establish a regression. Nonfinite values, unexpected
device/configuration, changed inputs, or a missing baseline path invalidate
the comparison. Exact Kalman errors and particle ESS explain the already
observed failure; they select no controls and support no stochastic ranking.
No posterior, default-readiness, convergence, or whole-repository equivalence
claim follows from this one fixed-input diagnostic. Heuristic dominance is
inapplicable: this is implementation parity, not method promotion.

Skeptical audit: comparing the earlier 3D T50 tuning SUCCESS label directly to
this 1D prior-wide Kalman screen would use the wrong baseline and proxy success
criterion. The fixed-input comparison removes that confound. Both revisions
must load the same existing native Sylvester library and use the same GPU.
All controls are inherited here solely to reproduce the failed candidate;
none is endorsed or retuned. The earliest diagnostic is input checksums and
the first-observation value/moment comparison. Both versions use a full N-by-N
transport block at N=1024, which obeys K=N. The baseline's dense N4096 route
violates the current exact-divisor cap3000 policy and will not be rerun. This
limits the comparison to N1024; it cannot clear the changed N4096 block path.

Budget: at most 600 GPU wall seconds and 180 CPU wall seconds, with no more
than three GPU evaluations (baseline recursion, current recursion, and current
consumer). Stop on budget, unexpected resource contention, invalid inputs, or
nonfinite arithmetic; record incomplete evidence rather than adding candidates.
Use the idle RTX 4080 SUPER, GPU-68251639-fe82-8f81-3ccc-2953c32e805b,
with verified TensorFlow memory growth. CPU preparation intentionally hides GPUs.
Environment: /home/chakwong/anaconda3/envs/tftwogpu/bin/python. Artifacts and
complete logs: docs/plans/artifacts/ledh-lg1-regression-audit-20260918-01.
Separate manifest records commands, commits, input/library hashes, device,
memory policy, elapsed time, and results. Prior campaign evidence is preserved.

Completed: both recursions and the actual consumer agree within the stated
tolerance; the starting revision already has the same 16% covariance error.
Three filter evaluations completed. One consumer preparation attempt failed
before evaluation because CPU/GPU whitening differed by a few ULPs; the local
harness repair supplied the frozen inputs directly. See the
[result](ledh-lg1-regression-audit-result-2026-09-18.md), manifest, and active
[checkpoint](artifacts/ledh-lg1-regression-audit-20260918-01/checkpoint.md).
