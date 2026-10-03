# Full post-training diagnostic on saved q20 maps

Owner request: execute the newly installed 1,000-point procedure on the saved
repaired maps. This is a diagnostic phase of the existing authorized campaign.

## Research intent and evidence contract

Question: how far do the four retained repaired maps depart from standard-normal
geometry at beta one, and do all 1,000 transformed target/score evaluations
remain numerically valid? Use the saved q20/T30 float64 UKF target approximation,
the exact frozen map checkpoints from the earlier 32-point diagnostic, and the
tested v2 `PostTrainingProbe`. Restore maps without optimizer updates.

The primary completion criterion is a complete, saved 1,000-row report for each
of control, repaired clipping, depth four, and batch 128. Their shared parent is
an optional fifth diagnostic under the same budget, evaluated last. The parent
is the training baseline; previous 32-point results are descriptive context and
are not substituted for full reports. No candidate ranking or recipe selection
is part of this run.

For theta=T(z), calculate
`g(z) = J_T(z)^T grad_theta log pi(theta) + grad_z log|det J_T(z)| + z`
and `r(z)=log pi(T(z))+log|det J_T(z)|+||z||^2/2` on standard-normal base draws.
Record residual vector-norm min/median/mean/RMS/p95/p99/max, exceedance fractions
above 1 and above the corresponding norm(z), r range, per-coordinate RMS and
existing scale diagnostics. Use global summary statistics across all batches.

Hard numerical screens are valid target statuses, finite values/scores and
unchanged restored parameters. A failed map remains recorded and triggers
repair; it does not cancel other maps. Changed source/target/checkpoint identity,
missing GPU memory growth, broken artifacts or exhausted budget stop execution.
Large finite residuals and saturated scale derivatives are explanatory signals,
not calibrated HMC rejection thresholds. Neither small residuals nor numerical
health establishes posterior coverage, adequate training or HMC convergence.

## Choices, provenance and budget

- Exactly 1,000 points per map: owner directive, diagnostic sample size rather
  than a tail-precision guarantee. Fixed batches of 20: the implemented standard
  and the previously successful 1,000-point GPU procedure. Stable [20,4] XLA
  graphs evaluate native batches. No scalar target fallback or pfor.
- Use the existing scoped geometry seed `(1843264305,1053964650)` on every map.
  It is reproducibility/common-bank provenance, not an optimized seed. The bank
  extends a previously inspected seed, so differences are descriptive. Gaussian
  base draws reach different physical locations for different maps.
- Numerical source stays frozen at
  `/tmp/BayesFilter-q20-training-repair-20260923-r1`. Copy the exact tested v2
  diagnostic module into the new artifact root; record its hash separately.
  Preserve the source snapshot and all original checkpoints/exports unchanged.
- Use trusted GPU1, confirmed available by the installed readiness probe. Other
  GPUs are occupied. TensorFlow memory growth must be set before import and
  verified before numerical work; record actual device, TF32 and XLA settings.
- Current balances: 144386.38292394514 campaign seconds and
  418.80836451620416 diagnostic seconds. The stage cap is the smaller current
  balance, not additional compute. Charge actual worker wall time to both
  ledgers with the existing Campaign supervisor. At most two attempts share
  this cap; a retry uses only unspent time and saved batches. Five-second process
  termination grace is inherited from the campaign. Stop starting batches ten
  seconds before the remaining cap, allowing the inherited grace plus time for
  summary/artifact writes. This is an engineering margin, not a runtime bound.
- Save each batch before starting the next; a timeout cannot erase a completed
  prefix. A new attempt receives a new versioned output directory and can copy
  the prior prefix. Mark all uncompleted maps explicitly.

Outputs: `docs/plans/artifacts/q20-saved-maps-1000-point-2026-09-23/run-01/`.
The adjacent runner, frozen probe and request preserve exact inputs. Command:
the tfgpu interpreter runs the runner's `supervise` entry point with that
request/output root; the supervisor launches a bounded worker with
`TF_FORCE_GPU_ALLOW_GROWTH=true`. The manifest records the actual argv,
environment, source/checkpoint hashes, seeds, wall time and allocator use.

## Skeptical audit before execution

The baseline maps and target match the earlier experiment; no workspace source
drift is substituted. Full-bank statistics replace sparse quantiles, while
geometry remains explanatory. The tested core includes the log-Jacobian score
and global r centering. GPU compatibility and runtime are still empirical;
early per-batch timings will show whether all maps fit. Persisted batches and a
shared cap prevent an unbounded retry. An expected poor-whitening result leads
to repair evidence, not a stop for the research direction. The strongest way
this check could mislead is missing posterior modes outside the map's reachable
base region; it therefore cannot certify convergence or global coverage. Audit
passes. Run the already tested shared core and record a terminal review.
