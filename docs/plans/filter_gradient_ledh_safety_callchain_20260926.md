# LEDH safety helper and analytical stage call-chain repair

The KDM endpoint audit found that the shared reset calls
`ledh_numerical_safety_tf.safe_cholesky`, which is outside the exact-source
guard and contains a Python rank-expansion loop. Two LGSSM reference functions
in `ledh_canonical_score_stages_tf` also retain Python flow/time recurrences.
The stage library supplies the UKF prediction/update and Gaussian-density
analytical helpers to the public score, so outer endpoint coverage alone is
insufficient. This is execution repair, not canonical LEDH rebuild/admission.

Replace the two fixed appended axes in the Cholesky mask with direct tensor
broadcasting, preserving the existing squeezed validity shape (including B=1
becoming scalar). Correct the helper documentation: detecting a failed
factorization does not diagnose all ill-conditioned matrices. Explicitly reject
nonfinite factors, including infinity, without changing finite accepted
factors. Test frozen pre-change factors/masks for unbatched and singleton/mixed
batch shapes, indefinite and singular inputs, and a new infinity regression
inside stable-signature XLA. No condition-number threshold or ridge is added.

Convert the stage-library reference recurrences to native `tf.while_loop`
without changing the finite numerical update order or analytical score. Preserve
the Python-double lambda schedule rounding before conversion to the configured
dtype. Standalone reference status remains explicit; consumers continue to use
the shared UKF helpers. Extend the exact guard through the reset adapter,
numerical safety, stage helpers and correction adapter. Any new exception must
be a reviewed fixed schema or enclosing graph, never a numerical-loop waiver.

Baseline is Git `04643213e` for these unchanged numerical sources. Primary gates
are complete output/validity equality on accepted inputs, analytical finite
differences and the existing independent reference tests, stable trace count,
native control-flow/HLO, CPU and GPU execution, and unchanged existing score
consumer results. Nonfinite-factor rejection is a fail-closed guard with an
explicit new rejection test. Timing and memory are explanatory diagnostics;
measure original versus native stage cold/repeated calls and allocator/host
peak separately in fresh processes before making cost claims. No NumPy enters
runtime, no pfor/autodiff becomes the score, and no threshold is loosened.

Budget inside the existing 56 CPU / 52 GPU hour caps: at most 10 CPU workers /
2400 process-seconds and 6 GPU workers / 1800 process-seconds for this unit.
Use `scripts/run_filter_repair_campaign.py`, versioned run directories under
the existing raw root, explicit CPU references and an available non-display
GPU with verified memory growth. Freeze numerical sources during workers.
An unexplained numerical difference blocks adoption and triggers localization;
source drift, missing artifacts, or the budget ceiling stops that comparison.

Skeptical review: broadcasting and squeezing are distinct operations, and B=1
must not silently change shape. A NaN-only check can accept an infinite factor;
finite output does not prove good conditioning. Moving the flow schedule to
FP32 division could change the finite program, so construct it in FP64 and
cast only afterwards. Existing reference functions are not target-scale
performance baselines. No result here closes HMC, LEDH tuning/admission,
posterior validity or the master program's terminal gates.
