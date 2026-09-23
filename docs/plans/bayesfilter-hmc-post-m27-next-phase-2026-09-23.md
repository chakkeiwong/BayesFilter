# HMC continuation after M27

Status: reviewed next-phase design. M27 is terminally audited; its ledger leaves
76517.20 CPU and 82259.94 GPU worker-seconds. Numerical launch follows a frozen
validated suite. This replaces the post-M26 agenda after
M27 closes. It does not reopen successful candidate-retention or process-parity
checks. The M27 result supplies the actual GPU price and remaining allowance.

## What the next phase must answer

M27's two rotated-Gaussian complete fits retained all 44 verified candidates.
The four predeclared L=3/L=4 members met .05 model-coordinate mean/median MCSE
at 10000 warmup and 30000 retained draws per chain. This demonstrates possible
delivery under that allocation, not coverage, superiority or a selection default.
The accidental duplicate is replay evidence only. The planned warmup maximum
of 30000 was not tested because the generated suite capped at 10000; record
that limit instead of rewriting the design. Future sibling suites must verify
each count against their plan before launch.

Large confirmation still needs adequate independent fit counts. M26's exact
binomial calculation requires 358/384 successes to pass one pointwise lower-.90
screen and has .951 passing probability at hypothetical true coverage .95.
The four-quantity union bound is .804; it is not simultaneous 95% confidence
and excludes the separate delivery criterion. CPU costs already make the
three-model inventory unaffordable. M27 GPU pairs took 648.78/651.47 seconds for Gaussian and 587.24/596.09
seconds for beta-binomial. At the larger prices, 384 fits require 69.49 and
63.58 GPU hours respectively. These are descriptive prices from one design
each, not timing upper bounds or speed rankings. Do not
launch an underpowered 64/128-fit substitute and call it closure.

The next affordable scientific question is whether a supplied partially
whitened map with bounded residual conditional scale avoids the specific funnel
tail defect seen in M25, while retaining every verified candidate and reporting
model-coordinate precision separately. Learning the map remains upstream.

## M28: supplied residual scale and posterior delivery

Reserve at most **1800 CPU and 4800 GPU worker-seconds**, taken from the terminal
remaining campaign allowance, with at most two numerical workers. These are
convenience development ceilings; no authorization increase. At most four
serious GPU fit attempts (two maps by two new seeds), 1000 seconds each, leave
800 GPU seconds for readiness and a localized infrastructure retry. A numerical
candidate failure remains a result, not permission to retry the same seed until
it passes. Keep every planned cell and failure. CPU checks hide devices before
import; GPU/XLA checks require trusted access and verified memory growth.

Use the existing noncentered funnel adapter and supported dense-IAF frozen
codec, composing the map with the adapter's model chart. The exact control has
`v=3*z0`, `x_i=exp(v/2)*z_i`. The optional residual fixture has

`delta(v)=0.5*tanh(v/6)`,
`v=3*z0`, `x_i=exp(v/2 + delta(v))*z_i`, for the two children.

The scale 3 and two children are inherited from M25 and preserve its model law.
The residual amplitude .5 is a convenience test hypothesis with a derived
conditional curvature ratio bounded by exp(2); width 6 equals twice the scale
and avoids an arbitrarily sharp local turn. Neither is a transport-training or
posterior default. The transform is triangular and invertible, using the same
model start bank and explicit inverse mapping. It uses existing codec operations:
affine scaling of z0 followed by an autoregressive tanh log-scale in the
noncentered child coordinates. No unsupported arbitrary force or custom tuner
is introduced. A distinct target/preparation/transport scope is required.

With `S=sum(z_i^2)`, the model-map log determinant is
`log(3)+v+2*delta(v)`. Substitution in the normalized funnel density gives

`U(z) = z0^2/2 + exp(2*delta(v))*S/2 - 2*delta(v) + constant`.

Thus the child-coordinate curvature is exp(2*delta) in [exp(-1),exp(1)]
for every v. This removes the old maps' exponential blowup in negative-v tails.
It does **not** bound the entire Hessian globally: derivatives with respect to
z0 include terms proportional to S. Neither finite conditional curvature nor
correct Jacobian algebra guarantees usable posterior trajectories or precision.
The residual's zero-amplitude limit must reproduce the exact Gaussian latent
control. Treat the preceding equations as a derivation to verify in code before
launch, not evidence that the new fixture already exists.

1. Implement the fixture as an optional diagnostic map family. Independently
   check forward/inverse roundtrips, normalized density plus total Jacobian,
   analytic score, directional finite differences, and the child-curvature
   bounds. Use extreme v and large children to expose falsely claimed global
   bounds. Check the zero-amplitude limit. Tests must reject target/scope and
   start-coordinate mismatches. Use scalar directional derivatives or a native
   TensorFlow loop; no implicit pfor. Then test both real public supplied-map
   pipelines, candidate receipts, complete retention, archive/restart and a
   failed-first-member continuation. Fixed `L=3` and `L=5` mechanics checks are
   controls only, not replacement public tuning.
2. Freeze four new GPU/XLA designs: exact and residual maps with root seeds
   2026092381 and 2026092382 (convenience identifiers, no seed optimization).
   Call `tune_fixed_transport_hmc_kernel` through the shared validation route.
   Use M25's declared broad L grid, per-candidate evidence and fresh verification;
   inspect all transferred search controls, and explicitly preserve or revise
   each in the generated plan before launch. Preserve common model starts.
   Use the optional `shortest_verified_l`, count 2, selected posterior scope.
   It is a cost hypothesis supported by M27 engineering checks, not a mixing
   default. Report shortages, budget-stopped slots and unassessed siblings.
3. Assess the unchanged .05 model-coordinate mean/median precision requirement
   with declared lugsail mean MCSE, not latent-Gaussian precision. Use warmup
   minimum/window 10000, chunks 5000 and maximum 30000; retained minimum 30000,
   chunks 5000 and maximum 60000. These are bounded allocation hypotheses
   informed by prior exact-map precision caps and M27 delivery; they are not
   transferred adequacy claims. The independent fixed arm uses 10000 warmup
   and 60000 retained draws per selected member. An explicit 60000 count budget
   and reason must be serialized. Freeze an exact count table and automatically
   compare it with all resolved designs before launch; permit no prose/JSON
   mismatch. At most 1000 seconds per complete fit, including teardown, with
   a 980-second child timeout as an unproven resource margin.
4. Audit the saved source, map/start algebra, all candidates and independent
   receipts, tensor/checkpoint integrity, normal exits and separate complete-fit
   slot denominators. Report each map and seed; never choose a posterior winner
   from these assessment draws. Reconcile actual cost, classify each failure,
   repair confirmed implementation errors, and refresh the following phase.

Create `m28-r1/` and an immutable source snapshot, and validate the suite with
`python -m bayesfilter.testing.inference_validation plan <suite> --output <plan>`.
Launch with the public `run` CLI, serial GPU jobs and at most one independent
CPU diagnostic worker. Use `/home/ubuntu/anaconda3/envs/tfgpu/bin/python`;
`TF_FORCE_GPU_ALLOW_GROWTH=true` and bounded numerical threads must be in the
environment before import. All exact commands go in metered attempt manifests.
No long run begins until the concrete suite and source match this reviewed plan.

### Frozen M28 count table

The M28 suite builder compares every resolved design with this table before
launch. All values are per chain. These are allocation hypotheses, not adequacy
claims; the 60000 count budget requires an explicit reason in each design.

| Design option | Count |
| --- | --- |
| posterior_settings.warmup_min_results | 10000 |
| posterior_settings.warmup_check_window_results | 10000 |
| posterior_settings.warmup_chunk_results | 5000 |
| posterior_settings.warmup_max_results | 30000 |
| posterior_settings.retained_min_results | 30000 |
| posterior_settings.retained_chunk_results | 5000 |
| posterior_settings.retained_max_results | 60000 |
| fixed_comparator.warmup_results | 10000 |
| fixed_comparator.retained_results | 60000 |
| posterior_count_budget.max_results_per_chain | 60000 |

## Evidence contract and skeptical review

The primary engineering criteria are correct transformed target and score,
correct starts, per-candidate verification, complete retention, unchanged resume,
valid archives and normal bounded exits. The posterior criterion is each
predeclared member's existing health/readiness/precision policy. R-hat, ESS and
MCSE do not affect tuning membership, epsilon repair or candidate retention.
Runtime, per-seed interval coverage, curvature and estimator values explain
failures; they do not establish a ranking. Expected posterior caps or numerical
health failures veto promotion of that member and trigger the next geometry or
allocation diagnosis. Corrupt source/archives, wrong target/Jacobian/score,
missing inputs, unavailable trusted devices or exhausted budget veto continuation
of the affected cell.

The constructed comparison set is: the exact noncentered positive control;
the independent fixed-count arm for each selected kernel; and exact analytic
model means/medians. Evaluate all three separately by map, seed and slot.
Any proposal to promote this fixture as a general policy additionally requires
simple fixed-step HMC and tuned affine controls, sufficient replication and
uncertainty; this development tranche cannot establish heuristic dominance.

Skeptical review: the main risks are using a bounded log-scale as if it were
bounded residual scale, claiming the whole Hessian is bounded, importing
latent-Gaussian precision into heavy model-coordinate tails, counting siblings
or repeated runs as independent fits, and selecting the favorable member after
seeing its posterior. The analytic substitution, tail tests, model-coordinate
assessment and frozen slot rule address those risks. Means of the funnel
children are finite but can have high sampling variability; two seeds cannot
certify coverage or reveal every rare event. A cheap test of the fixture algebra
precedes expensive tuning. Failure of a map is evidence about that map, not a
reason to reject supplied whitening or the tuning program.

## Remaining program after M28

| Gap | Next evidence or repair | Closure boundary |
| --- | --- | --- |
| General stopping/MCSE coverage | Profile current ordinary preparation/search costs, price unchanged confirmation inventory, or justify a new test with independent null/power calibration | Small pilots and saved-window reanalysis do not close coverage |
| Global exploration | Declared mode occupancy/crossings and an independent same-target reference on fresh multimodal fits; test numerical and global quantities separately | Local R-hat/MCSE cannot show unknown modes were found |
| Subtle full-fit defects | Inspect a cheaper statistical design's source/derivation, then fresh null and power calibration before confirmation | Existing 384-fit by at least 17-experiments-per-arm design remains under-budgeted |
| Exact MacroFinance integration | Newly qualified matching target/source, data, prior, coordinates and uncertainty-bearing independent reference | Synthetic models cannot manufacture missing consumer inputs |
| Measured maintenance and cost | Profile stage costs and graph/cache lifetimes under identical designs; change one confirmed bottleneck, then require exact replay and resource checks | No blanket refactor or unreviewed eager/non-XLA fallback |

After every phase use the same sequence: reconcile workers and costs, classify
candidate versus implementation failure, repair and test, record the result,
freeze the next discriminating design, audit assumptions, and execute within
the remaining authorization. No new approval ceremony or per-retry token applies.
