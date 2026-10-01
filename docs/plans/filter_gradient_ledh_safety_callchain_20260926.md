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

Budget inside the existing 56 CPU / 52 GPU hour caps: at most 12 supervised
CPU invocations / 2400 process-seconds and 8 supervised GPU invocations / 1800
process-seconds for this unit. A supervised lifecycle worker may launch its
declared serial child checks; its parent wall time is the charged quantity.
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

03987 passes all 11 CPU safety/wiring tests. 03988 finds an inherited XLA
incompatibility in the reference stage routines: `MatrixDeterminant` has no
XLA CPU kernel. Preserve this failure and candidate r1 bytes. Use the already
guarded native `bayesfilter.ops.slogdet_tf.determinant_tf` shared authority,
preserving determinant-product-then-log arithmetic and the existing analytical
inverse-trace derivative. A direct switch to `slogdet` would change that finite
product arithmetic and is not used. This is an operation compatibility repair;
accepted-output and derivative gates remain unchanged. Explicitly record the
original XLA compile attempt separately and use the original graph as the
reference where the frozen original cannot compile. Do not create a runtime
fallback or fabricate original-XLA cost evidence; use separate original-graph,
native-graph and native-XLA workers instead. This replaces the planned failed
original-XLA timing arm within the same worker/compute ceilings.

03989 confirms that native flow compiles and matches, then exposes TensorFlow
losing the uniform-weight vector's static extent inside the time body. Preserve
candidate r2 and assert the unchanged vector shape with `tf.ensure_shape`; do
not relax the owner signature or pad/recompute a different weight vector.

03990 passes the nonempty flow/recursion, complete records and derivatives,
then finds that XLA validates the unexecuted body even for an empty horizon.
Return the original neutral zero/empty score at that static configuration
boundary before tracing an invalid slice. Preserve candidate r3. Extend guard
coverage also to the correction's higher-moment and LM helpers and the mandated
configuration-time transport chunk selector. Its divisor search is a host
configuration operation required by owner policy; it is not a filter recurrence.
The fixed eight optional-target presence check is also host validation.

Allocate up to 12 CPU supervised invocations, within the unchanged 2400 CPU
seconds, to cover the three localized failures plus the required checks/cost
arms. The six GPU/1800-second and global caps are unchanged. Static import
discovery also finds conditional/reference modules beyond these direct helper
dependencies. Those remain audit debt; seven additional guarded modules do
not establish that every transitive optional route is repaired.

03991/03992 and 03995/03996 pass FP64/FP32 stage qualification on CPU/GPU.
03994 passes GPU safety/wiring checks. The original determinant fails enclosing
XLA on both devices; retain its graph comparator. Complete three fresh-process
cost arms per device (original graph, native graph, native XLA), with 20 exact
replays each. This needs seven GPU supervised invocations including the four
qualification/consumer groups, so set that local invocation ceiling to eight
within the unchanged 1800-second allowance. These single-process cohorts are
descriptive cold/warm/allocator measurements, not a statistical speed ranking.
The changed stage file has four identical pre-existing Ruff findings against
the frozen source; new/touched test/runner/safety sources introduce none.
