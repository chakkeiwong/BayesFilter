# SSL-LSTM q=20 seed-B NeuTra mode-failure root-cause plan (2026-08-10)

## Objective

Identify why the frozen seed-B NeuTra plus fixed-HMC run produced 4,000
retained draws with positive observation weight and no retained draw in the
half-space containing the known negative stationary point.

This investigation separates three mechanisms:

1. `transport_training_failure`: the reverse-KL flow proposal places negligible
   observable mass in the negative half-space;
2. `transformed_geometry_failure`: the exact Jacobian-corrected target retains
   separated latent attraction regions or a material observed path barrier; and
3. `initialization_or_kernel_failure`: the original local starts and tuned
   fixed-HMC kernel do not transition between the two regions even when both
   regions are used as explicit starting points.

The diagnostic does not create an independent posterior authority and does not
estimate exact posterior mode weights.

## Research intent ledger

| Item | Frozen statement |
|---|---|
| Main question | Which of transport proposal coverage, exact transformed geometry, and initialization/fixed-kernel behavior explains the missing negative-region draws? |
| Candidate under test | Frozen seed-B terminal dense-IAF transport at optimizer step 6,250 and its frozen `L=3`, step-size `0.8115211181271775`, identity-mass HMC kernel. |
| Baseline/comparator | The two highest-log-density sign-separated stationary representatives in the existing target-only multistart artifact, plus the original four local latent starts. |
| Expected failure mode | Reverse KL learned the positive region; the inverse image of the negative region is remote from the standard-normal training base, and local-start HMC does not cross the resulting transformed barrier. |
| Promotion criterion | None. Root-cause classification only. |
| Promotion veto | None. No sampler or transport is promoted by this diagnostic. |
| Continuation veto | Wrong source/target/transport/kernel identity; nonfinite transform, value, score, Hessian, or path result; failed transform round trip; invalid target status; visible GPU in the explicit CPU diagnostic; artifact overwrite; or wall-cap breach. |
| Repair trigger | Flow omission triggers multimode-aware training or a global mode-discovery input; exact separated transformed regions trigger tempering/global transitions; same-region convergence from split starts triggers kernel redesign rather than more local-start draws. |
| Explanatory diagnostics | Flow sign counts, binomial zero-count upper bound, inverse-mapped locations and base log densities, exact transformed values/scores, transformed local optima, Hessians, two observed path barriers, kinetic-energy availability, original-start mapping, split-start sign transitions, acceptance, and runtime. |
| Must not conclude | Exact posterior basin weights, minimum possible energy barrier, exhaustive mode discovery, posterior correctness, predictive validity, NeuTra failure in general, or that a finite zero-transition count proves mathematical impossibility of crossing. |

## Evidence contract

1. Reconstruct the frozen transport and q=20 target through
   `docs/benchmarks/ssl_lstm_q20_neutra_seed_b_terminal.py`. Require the exact
   target and base-adapter identities, checkpoint identity, unmigrated trainer
   source-state hash, optimizer step, and underlying trained state recorded by
   the seed-B loader. Migrated/restored-state hashes are recorded but cannot be
   equality gates because the live migration adds two new null chart fields.
   The live shared tree adds chart/output fields to regenerated transport
   manifest hashes, so the historical transformed-adapter hash no longer
   reproduces. This metadata-schema drift is admissible only through the
   numerical compatibility gate in item 2; it is not silently normalized.
2. Verify the exact transformed value and score against the authenticated
   August 7 retained trace at 16 deterministic accepted latent states, evaluating
   each at the historical persistent-worker static shape `[1,4]`. Require
   maximum absolute value and score residual `<=5e-7`, finite
   current values/scores, valid current target status, and hash-valid archived
   state/value/score tensors. Record both current and historical hashes plus the
   source diffs causing drift. Any numerical mismatch is a continuation veto.
3. Select the positive and negative source-coordinate representatives from the
   hash-bound target-only multistart artifact using finite coordinates, finite
   target value, score infinity norm `<=1e-5`, observation-weight sign, and
   maximum target value within sign.
4. Map both representatives through the frozen inverse and require forward
   round-trip error `<=1e-10`. Evaluate the exact transformed target
   `log pi_z = log pi_theta(T(z)) + log|det J_T(z)|` and its score.
5. Standard-normal draws test only the learned proposal
   `q_phi = T_# N(0,I)`. Use 100,000 deterministic TensorFlow draws and report
   physical sign counts. If one sign has zero observations, report the exact
   one-sided 95% binomial upper bound `1 - 0.05^(1/n)`; this is a proposal-mass
   bound, not a posterior-mass bound.
6. Because source-coordinate stationary points need not remain stationary after
   the nonlinear Jacobian correction, optimize the exact transformed target
   independently from both inverse-mapped representatives. An endpoint is a
   usable transformed stationary representative only if it is finite and its
   transformed score infinity norm is `<=1e-5`.
7. Compute stable symmetric finite-difference negative Hessians at distinct
   usable transformed representatives. Hessian eigenvalues and local Laplace
   log masses are explanatory only; they do not provide exact global weights.
8. Evaluate two deterministic connecting paths: straight interpolation in
   latent coordinates and straight interpolation in source coordinates mapped
   through the inverse transport. The sampled maximum potential rise is a path
   heuristic only: the path need not minimize the barrier, and a finite grid
   can miss a narrow peak. Convert each sampled rise to the identity-mass, four-
   dimensional probability `P[K > delta] = exp(-delta)(1+delta)` only as an
   energy-availability diagnostic.
9. Map the original material-run starts through the transport and record their
   physical observation-weight signs and distances to both transformed
   representatives.
10. Run the frozen numeric kernel from a split bank initialized at both inverse-
   mapped sign-separated source representatives. The canary uses 16 replicated starts per region and 8
   transitions solely to validate mechanics and measure cost. If valid and
   within budget, the material diagnostic uses 32 replicated starts per region
   and 64 transitions. Every draw and sign transition is retained. These counts
   diagnose this fixed kernel; they do not estimate stationary basin weights or
   establish convergence.
11. TensorFlow/TFP and XLA are required for target, optimization, and HMC
    computations. GPU is deliberately hidden because this reuses the reviewed
    CPU/XLA diagnostic lane; it provides no production-target performance
    evidence.

### Post-canary repair

The inverse-source-MAP canary completed in `2,952.831` seconds. Positive-start
binary acceptance was `0.7265625`, negative-start acceptance was `0`, and no
chain crossed sign regions. The original material rung is under-budgeted:
linear scaling by the doubled chain bank and eightfold transition count gives
`47,245.3` seconds, versus approximately `8,084` seconds remaining under the
12,000-second campaign cap. It will not be launched.

That zero-acceptance result is ambiguous because an inverse-mapped source MAP is
not stationary under the exact transformed target. The transformed optimizer
found two usable stationary endpoints, with the negative endpoint materially
farther into the tail than the inverse source MAP. The repaired next phase uses
8 replicated chains at each exact transformed stationary endpoint and 4 frozen-
kernel transitions. It archives samples, signs, acceptance flags,
log-acceptance ratios, accepted/proposed target values, and accepted scores.
This is the smallest test separating bad negative initialization from a kernel
that is locally invalid for the negative transformed region.

Skeptical audit verdict: **PASS FOR TARGETED REPAIR**. Target, transport,
kernel, hardware class, and campaign cap remain unchanged. The phase does not
retune or rank kernels, and its only decision is whether the frozen kernel can
make accepted local moves from the transformed negative stationary region.

The stationary canary again produced zero negative-region acceptance (`0/32`)
while positive-region acceptance was `0.71875`. Its negative-region proposed
target values ranged from about `-1.5e4` to `-1e100`. The negative transformed
precision has largest eigenvalue `91.72047`, giving the quadratic leapfrog
stability scale `2/sqrt(lambda_max)=0.2088319`; the frozen step `0.8115211` is
3.89 times that scale. A final causal control therefore holds the stationary
starts, `L=3`, chain count, transition count, target, and random-seed domain
fixed but uses `epsilon=0.1`, less than half the derived stability scale. This
number is derived from measured curvature, not tuned for acceptance. Recovery
of finite accepted negative-region moves identifies step-size/curvature
mismatch; continued zero acceptance would leave another negative-region
mechanism unresolved.

## Numeric provenance and default audit

| Choice | Provenance/status | Justification | Failure mode | Early diagnostic |
|---|---|---|---|---|
| Two source representatives | Measured existing target-only artifact | Known sign-separated stationary regions under the exact source target | Other modes may exist; source MAPs are not transformed MAPs | Optimize exact transformed target from both inverse images |
| 16 archived parity states at batch size 1; tolerance `5e-7` | States span four chains and separated retained indices; batch size reproduces the archived one-chain-worker static target shape. Tolerance is derived after exact historical-commit reconstruction measured maxima `1.522e-7` for value and `2.080e-7` for score in standalone versus fused-HMC XLA contexts; `5e-7` is slightly above twice the larger measured residual | Detects material numerical drift while admitting measured XLA compilation-context rounding under identical source/checkpoint identities | Sparse parity could miss state-dependent drift elsewhere; tolerance was revised after the prospective tighter values vetoed both live and exact-historical reconstructions | Bind exact tensor hashes and historical commit/checkpoint; record residuals; retain no global equivalence claim |
| 100,000 base draws | Derived diagnostic budget | Zero observations gives one-sided 95% proposal-mass upper bound about `3e-5` while transport mapping is cheap | Rare proposal mass below the bound remains unseen | Report count and bound without posterior claim |
| 95% zero-count bound | Standard exact binomial inversion, explanatory | Quantifies what zero observed proposal draws means | Dependence would invalidate it | Stateless base draws are iid by construction |
| 65 path points | Convenience diagnostic grid, not reviewed default | Resolves broad barriers with one batch-native target call | Narrower saddle/barrier can be missed; path is not optimized | Report both coordinate paths as heuristics with no bound claim |
| Hessian steps `1e-3,3e-4,1e-4` | Inherited from the prior q=20 reference diagnostic | Existing target-specific finite-difference ladder | Curvature instability or non-SPD point | Require finite SPD result and last-two relative change `<=1e-3` for Laplace reporting |
| Split canary 16 per region, 8 transitions | Convenience mechanics/cost choice | Large leading batch tests both starts without pretending to sample a distribution | Too short for transition-rate inference | Explicit mechanics-only role; material rung required |
| Split material 32 per region, 64 transitions | Bounded diagnostic hypothesis | 4,096 transition opportunities across a balanced start bank; comparable total state count to the failed archive | Autocorrelation and start conditioning prohibit stationary-weight claims | Report complete paths, region-specific transitions, and acceptance only |
| Stationary repair canary 8 per region, 4 transitions | Derived from the completed inverse-MAP canary and remaining campaign budget | Resolves whether zero negative acceptance came from nonstationary inverse-MAP starts or the frozen kernel itself | Still too short for stationary weights or transition-rate inference | Archive full kernel trace; compare local acceptance by transformed stationary start |
| Stationary step control `epsilon=0.1`, `L=3` | `0.1 < 0.5 * 2/sqrt(91.72047) = 0.104416`; derived from measured negative-mode curvature with a factor-two stability margin | Tests whether the frozen `0.8115211` step causes negative-region proposal blow-up | A short successful control does not tune an optimal kernel or solve global transitions | Require finite trace and report region-specific local acceptance only |
| Frozen `L=3`, epsilon `0.811521...` | Measured selected kernel | The question concerns why the actual run failed | A better kernel might cross and this test would miss it | Classification is fixed-kernel failure, not universal HMC failure |
| 12,000-second total cap | User-provided headroom and prior measured q=20 HMC cost | Bounded root-cause campaign | Shared CPU load or large-batch compilation may exhaust cap | Canary records compile/runtime before material rung |

All numeric choices above remain diagnostic settings. None becomes a sampler,
training, or repository default.

## Skeptical pre-execution audit

| Audit question | Finding |
|---|---|
| Wrong baseline? | No. The exact failed transport/kernel identities and the two known exact-target stationary regions are the objects under question. |
| Proxy promoted? | No. Base-flow draws concern `q_phi`, Laplace concerns local geometry, observed path profiles are heuristics rather than barrier bounds, and split starts concern fixed-kernel transitions only. |
| Missing stop condition? | No. Identity, finite/status, round-trip, transformed-stationarity, curvature, artifact, device, and wall-cap vetoes are explicit. A scientific failure in one diagnostic does not invalidate later discriminating diagnostics. |
| Unfair comparison? | No ranking is attempted. Both regions use the same target, transport, kernel, leading-batch size, random mechanism, dtype, and XLA route. |
| Hidden assumptions? | Exposed: only two known regions; sign half-space is not a formal basin; local Laplace is approximate; deterministic paths are not optimized minimum-energy paths; finite transitions cannot prove impossibility. |
| Stale context? | Found and repaired: the live shared tree adds chart/output metadata to regenerated transport hashes, so historical adapter-hash equality fails. A hash-bound archived-state value/score parity gate now distinguishes numerical drift from metadata-schema drift before any scientific diagnostic. Existing 4,000 retained draws are otherwise used only for the established missing-region fact. |
| Environment mismatch? | CPU/XLA is the same explicit diagnostic exception used for the failed material HMC archive; GPU is hidden before TensorFlow import. |
| Could the run pass while misleading us? | Yes, if a narrow peak is missed, another lower path exists, or more modes exist. Path semantics, finite-transition limits, and mode-incompleteness are mandatory nonclaims. |
| Could it fail for engineering reasons? | Yes. Source drift, transform inversion, XLA compilation, target status, and artifact failures are continuation vetoes and cannot be interpreted scientifically. |
| Do artifacts answer the stated question? | Jointly yes: proposal sign mass locates training omission, exact transformed modes/path geometry locate residual geometry, and split starts isolate local initialization/kernel behavior. Exact posterior weights remain unanswered. |

Audit verdict: **PASS AFTER TWO REPAIRS**. The initial idea of evaluating only the
inverse-mapped source MAPs was wrong for the transformed-mode question because
the nonlinear log-Jacobian changes stationarity. The repaired plan first
optimizes the exact transformed target and restricts barrier/Hessian claims to
usable transformed stationary representatives. The first two launch attempts
then exposed stale manifest/signature identity after shared chart-governance
changes. The second repair adds an authenticated archived-state transformed
value/score parity gate and records, rather than hides, current versus historical
hashes. The initial `1e-10/1e-9` parity tolerances vetoed both the live tree and
an exact detached reconstruction of commit `9ebaecc...` by the same approximately
`2e-7`; this localizes the difference to standalone-versus-fused XLA numerical
context rather than source/checkpoint drift. The reviewed compatibility
tolerance is therefore revised to the measured-derived `5e-7`, with the exact
historical commit and evidence hashes still mandatory. No remaining known flaw
prevents this bounded root-cause diagnostic. The August 7 run recorded a dirty
worktree, so its transformed manifest hash is not reproducible from the recorded
commit alone; `historical_identity_exact=false` remains a mandatory limitation,
not a caller-stamped replacement identity.

## Execution and artifacts

Versioned output root:

`docs/plans/artifacts/ssl-lstm-q20-seed-b-neutra-mode-failure-root-cause-2026-08-10/r1/`

Planned commands:

```bash
CUDA_VISIBLE_DEVICES=-1 /home/ubuntu/anaconda3/envs/tfgpu/bin/python -m pytest -q tests/test_ssl_lstm_q20_seed_b_neutra_mode_failure_root_cause.py
CUDA_VISIBLE_DEVICES=-1 /home/ubuntu/anaconda3/envs/tfgpu/bin/python docs/benchmarks/run_ssl_lstm_q20_seed_b_neutra_mode_failure_root_cause_2026_08_10.py --mode geometry
CUDA_VISIBLE_DEVICES=-1 /home/ubuntu/anaconda3/envs/tfgpu/bin/python docs/benchmarks/run_ssl_lstm_q20_seed_b_neutra_mode_failure_root_cause_2026_08_10.py --mode split-canary
CUDA_VISIBLE_DEVICES=-1 /home/ubuntu/anaconda3/envs/tfgpu/bin/python docs/benchmarks/run_ssl_lstm_q20_seed_b_neutra_mode_failure_root_cause_2026_08_10.py --mode split-material
CUDA_VISIBLE_DEVICES=-1 /home/ubuntu/anaconda3/envs/tfgpu/bin/python docs/benchmarks/run_ssl_lstm_q20_seed_b_neutra_mode_failure_root_cause_2026_08_10.py --mode stationary-canary
CUDA_VISIBLE_DEVICES=-1 /home/ubuntu/anaconda3/envs/tfgpu/bin/python docs/benchmarks/run_ssl_lstm_q20_seed_b_neutra_mode_failure_root_cause_2026_08_10.py --mode stationary-step-control
```

The geometry artifact decides whether the split canary remains interpretable.
The canary records measured runtime before the material rung. A result note will
record the actual commands, environment, git commit, CPU/GPU status, identities,
seeds, wall time, artifact paths, decision table, inference-status table,
alternative explanation, and next repair.
