# SSL-LSTM q=20 tempered reverse-KL ensemble Phase 8 calibration subplan

Date: 2026-08-29  
Status: `C0_TO_C5_COMPLETE_K2_COMPACT_HIGH_L3_PURE_FROZEN_K4_NOT_RETAINED_PHASE9_SUBPLAN_REQUIRED`

Parent program:
`docs/plans/bayesfilter-ssl-lstm-q20-tempered-rkl-transport-ensemble-implementation-plan-2026-08-28.md`.

GPU boundary note:
`docs/plans/bayesfilter-gpu-default-execution-boundary-2026-08-29.md`.

This is the active serious-campaign subplan required by Phase 8. It governs
calibration and candidate freeze only. It does not consume untouched Phase 9
confirmation streams and cannot establish posterior convergence, mode
discovery, statistical superiority, high-dimensional scaling, or a new default.

## Research question and evidence contract

Question: can independently initialized, fresh-Gaussian reverse-KL transports
trained along a proper prior-to-posterior bridge produce a numerically reliable
set of distinct frozen charts worth testing with exact multi-chart HMC and
replica exchange?

The exact baseline ladder for this phase is:

1. one cold reverse-KL chart at `beta=1`;
2. `K=2`, three bridge levels, pure continuation;
3. `K=4`, three bridge levels, pure continuation;
4. `K=4`, five bridge levels, pure continuation; and
5. `K=4`, five bridge levels, with half the charts freshly restarted at
   `beta=0.5`.

The primary Phase 8 pass criterion is engineering and numerical viability, not
posterior promotion: every retained chart has a complete immutable lineage,
finite batch-native XLA training, exact checkpoint replay, a passing learned-map
reliability receipt, and measured cost within the campaign envelope. Phase 8
may nominate and freeze at most one `K=2` representative and one `K=4`
representative for Phase 9.

Hard vetoes are a target/bridge identity mismatch, failed endpoint/properness
receipt, invalid target row, nonfinite loss or gradient, scalar or row-mapped
training, pfor, missing/incorrect checkpoint replay, failed inverse/Jacobian/
score reliability, GPU memory-growth failure, wrong device/XLA route, artifact
overwrite, or an exhausted campaign cap. A hard failure rejects that attempt;
it rejects the research direction only when the bounded repair ladder is
exhausted.

Training loss, pullback-density residuals, pullback-score residuals, sign-region
occupancy, component separation, clipping frequency, HMC acceptance, swap rate,
and short-chain movement are explanatory or nomination diagnostics. They do not
promote a sampler. Phase 9 R-hat, ESS, MCSE, region travel, downstream agreement,
and uncertainty-aware comparator results are the promotion evidence.

The preserved result is a versioned run manifest plus a Phase 8 result note
under the output root below. Even a pass establishes only that a frozen
candidate is ready for Phase 9 tuning and confirmation.

## Budget, attempts, and output ownership

The user supplied an additional 18-hour budget earlier in this program and on
2026-08-29 explicitly instructed the refreshed program to continue. This
subplan treats `64,800` command-wall seconds as the hard cap beginning with the
first command executed under this subplan. The number is user supplied, not a
performance estimate. The budget is allocated as follows:

| Work | Cap (seconds) | Provenance and purpose |
|---|---:|---|
| Full compatibility and focused checkpoint tests | 3,600 | Bounded engineering gate |
| GPU compile, batch, and cost pilot | 5,400 | Measured feasibility, no selection |
| Architecture/optimizer and lineage calibration | 28,800 | Main Phase 8 search |
| Reliability, uncertainty analysis, freeze, and repair | 9,000 | Candidate freeze |
| Reserved for the first justified Phase 9 tuning/pilot | 18,000 | Prevent Phase 8 from consuming the full grant |

Phase 8 has at most ten launched attempts: two compatibility/localization, two
GPU cost, four training/search, and two repair/freeze attempts. A command that
does not launch because the platform approval service fails is recorded as an
infrastructure denial but consumes no GPU allocation. Every actual attempt uses
a fresh directory and counts its measured wall time. No prior output is
overwritten.

Output root:

```text
docs/plans/artifacts/ssl-lstm-q20-tempered-rkl-transport-ensemble-2026-08-29/phase8-calibration/
```

Phase 10 has no allocation in this grant. Reaching Phase 10 is progress, but a
new bounded scaling budget is then required.

## Frozen environment

- Target: current q=20 SSL-LSTM likelihood with four sampled parameters and
  60 internal filtering coordinates.
- Bridge: repository-issued Gaussian-prior likelihood bridge with the current
  properness receipt and exact target signature.
- Backend: TensorFlow/TFP, `float64` for this target, XLA on for GPU training,
  TF32 state recorded, and no NumPy runtime path.
- Device: the repository-default launcher selects one visible GPU (GPU 0 unless
  `BAYESFILTER_GPU_ID` is explicitly set), with
  `TF_FORCE_GPU_ALLOW_GROWTH=true` set before TensorFlow import and verified on
  every visible physical device before logical initialization. The launcher
  does not call an idle-GPU or per-run approval probe. An outer Codex execution
  boundary, if present, is platform provenance rather than a scientific gate.
- GPU allocator stop: terminate the attempt if TensorFlow peak allocation
  exceeds 4 GiB in the cost pilot or if a contemporaneous resource check shows
  material contention. The 4 GiB pilot cap reserves most of the measured
  31 GiB free device memory for other work; it is a resource-sharing bound, not
  a scientific threshold.
- Training: every optimizer update uses one fresh stateless Gaussian batch with
  more than one row and one batch-native target call. No invalid row is dropped
  or replaced.
- New paths must pass a static scan for `tf.map_fn`, `tf.vectorized_map`, pfor,
  and `GradientTape.jacobian`/`batch_jacobian`.

## Compatibility gate

Run these CPU-hidden compatibility commands before any selection-bearing GPU
training. CPU use here is an explicit engineering-test exception, not NeuTra
training evidence.

```text
CUDA_VISIBLE_DEVICES=-1 TF_CPP_MIN_LOG_LEVEL=3 timeout 1200s \
/home/ubuntu/anaconda3/envs/tfgpu/bin/python -m pytest -q --disable-warnings \
tests/test_ssl_lstm_complexity_target_tf.py

CUDA_VISIBLE_DEVICES=-1 TF_CPP_MIN_LOG_LEVEL=3 timeout 1200s \
/home/ubuntu/anaconda3/envs/tfgpu/bin/python -m pytest -q --disable-warnings \
tests/test_ssl_lstm_predictive_tf.py
```

A failing assertion blocks training until repaired. A timeout with no failure
is a compile/infrastructure repair trigger: use pytest duration/collection
localization, run the smallest stalled test, and retry the full suite once
within the remaining 3,600-second allocation. If the second full run still
times out, preserve the focused passes but do not mark the full suite passed;
GPU selection waits unless the timeout is shown to be wholly outside the call
chain used by this campaign.

The focused route suite is:

```text
CUDA_VISIBLE_DEVICES=-1 TF_CPP_MIN_LOG_LEVEL=3 timeout 600s \
/home/ubuntu/anaconda3/envs/tfgpu/bin/python -m pytest -q --disable-warnings \
tests/test_tempered_transport_ensemble.py \
tests/test_tempered_lineage_transitions.py \
tests/test_neutra_hmc_route_policy.py \
tests/test_ssl_lstm_protocol.py
```

## Immutable temperature checkpoint protocol

One mutable transport must never stand for several temperature charts. At each
`(arm, root seed, beta, chart)` boundary, the runner writes an immutable state
containing:

- transport configuration and reference-affine wrapper;
- all trainable TensorFlow tensors in stable order;
- component, parent, beta, bridge, target, data, dtype, backend, and XLA
  identities;
- update count, stateless training seed derivation, and validation-bank IDs;
- transport tensor hash and checkpoint payload hash.

Before proceeding to the next beta, a fresh transport object is reconstructed
from the checkpoint. Forward value, inverse, forward log determinant, and every
trainable tensor must replay on a fixed bank, and both hashes must match. The
restored clone continues training; the earlier-beta object is frozen and never
mutated. Tampering or a replay mismatch is a hard engineering veto.

Adam state is reset at each new beta in this candidate. This is an explicit
hypothesis: the objective changed, and the current trainer binds beta at graph
construction. It is not evidence that optimizer reset is universally better.
The manifest records the reset, and an optimizer-carry ablation remains a
future repair if continuation stalls while maps and targets remain valid.

## Candidate and seed protocol

Candidate component counts are `K in {2,4}`. Two is the smallest ensemble;
four is the smallest tested expansion that can restart half its charts while
retaining two continuations. These are bounded pilot hypotheses, not defaults.

Bridge ladders are:

```text
L3 = (0.0, 0.5, 1.0)
L5 = (0.0, 0.25, 0.5, 0.75, 1.0)
```

`L3` is the smallest proper path with an interior law. `L5` halves that spacing
to test overlap at bounded cost. The branching arm restarts component indices
`K/2, ..., K-1` at `beta=0.5`; the remaining half continue from `beta=0.25`.

The target-specific architecture screen uses:

| ID | Hidden layers | Stages | Activation | Learning rate |
|---|---|---:|---|---:|
| `compact-high` | `(16,16)` | 2 | `tanh` | `1e-3` |
| `compact-low` | `(16,16)` | 2 | `tanh` | `5e-4` |
| `wide-high` | `(32,32)` | 2 | `tanh` | `1e-3` |
| `wide-low` | `(32,32)` | 2 | `tanh` | `5e-4` |

These values are q=20 legacy warm starts used only to bound a target-specific
screen. They are not inherited defaults. Other Adam parameters remain the
current library values and are recorded as convenience hypotheses; failure of
all four candidates triggers a bounded activation/scale-cap diagnosis before
rejecting the architecture family.

Batch hypotheses are `B in {8,32}`. The cost pilot chooses `B=32` only if it is
finite, its peak allocation is at most 4 GiB, and its median steady update time
is at most four times the `B=8` median; otherwise it chooses `B=8`. Four is the
linear sample-count ratio, so this rule rejects a worse-than-linear throughput
increase without pretending to optimize scientific quality.

The update ladder at each positive beta is `(8,32,128)`. All viable calibration
arms reach 32 updates. An arm extends to 128 only when at least half of its
components lower their fixed-bank held-out reverse-KL estimate from update 8 to
32, no hard reliability veto fires, and the forecast fits the remaining
allocation. This is an adaptive calibration rule; none of these banks is used
for confirmation.

There are no optimizer updates at `beta=0`. The reference-affine initialization
must first pass an analytic density/value check showing that its pushforward law
is exactly the Gaussian bridge endpoint. This preserves independent internal
initialization while avoiding a shared endpoint optimization that can erase it.
Failure of that equality is a target/map implementation veto.

Stateless roots are frozen as follows:

```text
initialization roots: (20260829,11001), (20260829,11002), (20260829,11003)
training root:        (20260829,21001)
calibration roots:   (20260829,31001), (20260829,31002)
validation roots:    (20260829,41001), (20260829,41002), (20260829,41003)
stress root:          (20260829,51001)
Phase 9 tuning root:  (20260829,71001)
Phase 9 confirmation: (20260829,91001), (20260829,91002), (20260829,91003)
```

The runner derives every beta/component/update seed with
`tf.random.experimental.stateless_fold_in` in the fixed order
`arm, root, beta index, component, role, update`. Seed collisions are a hard
veto. Phase 9 roots are reserved and cannot be read by Phase 8 selection code.

## Calibration diagnostics

For each frozen chart and beta, use three disjoint 256-row standard-Gaussian
validation banks. This size gives a worst-case sign-probability standard error
of about `1/(2 sqrt(256)) = 0.03125`; it is enough for calibration description,
not a posterior occupancy claim. A separate 4,096-row base bank estimates each
chart's forward sign-region occupancy without another target evaluation, with
worst-case Monte Carlo standard error about `0.0078`.

For `z ~ phi`, calculate

```text
r(z) = log tilde_pi_beta(T(z)) + log|det J_T(z)| - log phi(z)
u(z) = J_T(z)^T score_beta(T(z)) + grad_z log|det J_T(z)| + z.
```

Record the held-out reverse-KL mean, centered `r` RMS, median absolute centered
`r`, q90 absolute residual, `u` RMS per coordinate, maximum row norm,
target-status counts, and gradient-clipping frequency. Exact Gaussianization
would make `r` constant and `u=0`; finite deviations measure local pullback
density/score mismatch under the base law. They do not inspect target regions
that every chart misses.

For each root and beta, compare the per-row held-out reverse-KL contribution at
the frozen checkpoint with the same rows at that beta's starting checkpoint.
The 768 paired rows from the three 256-row banks provide the Monte Carlo mean,
standard error, and a two-sided normal 95% interval for the mean difference.
The normal interval is a calibration approximation justified by IID base rows;
the three initialization roots are reported separately so row-level precision
is not confused with training-seed uncertainty. A chart is training-viable only
when its mean difference is negative and the interval upper endpoint is below
zero at its final positive beta. If fewer than two of three roots are viable,
that architecture/lineage arm enters the bounded repair ladder rather than
being nominated. This establishes improvement over its own start, not closeness
to the posterior or superiority over another architecture.

Extreme empirical q99/max density-residual values from these small banks are
recorded only as unstable tail nomination signals and never enter the freeze
rule. The fixed radial-stress bank and learned-map reliability receipt carry
the bounded tail engineering checks.

The fixed stress bank contains coordinate axes at radii `sqrt(d)`,
`2 sqrt(d)`, and `3 sqrt(d)`, the prior center, the two target-bound MAP
representatives, self-component draws, and all cross-component draws. The
existing reliability screen supplies dtype/scale-derived tolerances for
round-trip, log-determinant cancellation, transformed score, and conditioning.
No threshold is tuned after q=20 results are observed.

Forward input moments of the generated `z` are forbidden as whitening evidence:
they are Gaussian by construction. In Phase 9, exact retained physical draws
will be pulled back through each selected chart; only then are latent mean,
covariance, radial tails, rank-normalized R-hat, ESS, and MCSE informative about
the sampled pullback law.

Exact duplicate tensor hashes are a repair trigger. Near-duplicate maps are
described with pairwise cross-density disagreement and sign-occupancy intervals;
they are not rejected by a post hoc distance threshold. If every chart in an
arm is descriptively indistinguishable and occupies only one declared region,
that arm remains mathematically valid but is not nominated as the diversified
representative.

## Search sequence and freeze rule

1. **C0, checkpoint correctness:** add snapshot/restore support and exact replay
   tests before any serious training.
2. **C1, cost pilot:** run `K=2`, `L3`, `compact-high`, one compile update and
   four timed steady updates at `B=8` and `B=32`. Measure target calls, compile
   time, median step time, allocator current/peak, and cross-density work.
3. **C2, architecture/optimizer screen:** at `beta=0.5`, train all four
   architecture rows to 32 updates for the first two initialization roots at
   the selected batch size. Preserve every result and reliability receipt.
4. **C3, lineage ladder:** retain architectures that are hard-valid and satisfy
   the paired start-to-final improvement rule for at least two roots. If both
   capacities remain viable without a statistically supported difference,
   choose the smaller one by predeclared parsimony. Within equal parameter
   count, use lower mean held-out reverse-KL only as an operational nomination
   rule, not evidence of superiority. Run the five baseline/mechanism arms
   listed in the evidence contract with three roots.
5. **C4, optional joint arm:** admit `K=4` joint refinement only if measured
   `K^2 B` work forecasts at most 3,600 seconds for its planned updates and a
   peak below 4 GiB. Otherwise record `SKIP_JOINT_ARM_RESOURCE_ENVELOPE`.
6. **C5, freeze:** retain at most one hard-valid `K=2` and one hard-valid `K=4`
   representative. If multiple candidates are not statistically distinguishable
   under paired validation-bank uncertainty, the lower parameter count, fewer
   bridge levels, and lower measured target-call/wall cost win by predeclared
   parsimony. This selects a representative for confirmation; it does not rank
   scientific performance.

No Phase 9 confirmation seed or stream is consumed in C0--C5.

## C1 timeout and bounded localization repair

The first repository-default GPU launch was
`attempt-02-default-gpu`. It initialized one visible GPU 0 with the required
on-demand allocator, completed the `B=8`, chart-0 `beta=0` preflight,
diagnostic, immutable checkpoint, and replay, and then reached the 1,800-second
launcher cap before producing a `beta=0.5` result. Its exit code was `124`; the
partial checkpoint and timeout record are preserved under that attempt
directory. Because the process was terminated externally, the Python exception
handler did not write `failure.json`.

This is classified as `available_evidence_only` invalidation and a localization
repair trigger. It does not establish a target, mathematics, implementation,
candidate, whitening, or mode-travel failure. The attempt consumed 1,800 seconds
of the 5,400-second GPU cost-pilot allocation, leaving 3,600 seconds for the
bounded repair and one cost-pilot retry. The amount is measured from the
launcher timeout, not a performance estimate.

The next repair uses the same repository-default launcher in
`target-localization` mode, `B=8`, and a 900-second cap (a predeclared quarter
of the remaining C1 allocation). It writes a start marker before each direct
q20 target call, beta-half preflight, diagnostic, and first optimizer update.
The marker that remains after any external timeout identifies the operation
being localized. This mode is diagnostic only and cannot select a candidate.
After localization, a fresh cost-pilot retry may use the reviewed strict
`tensorflow_eigh_strict` backend only if the marker identifies a backend/graph
compile issue; changing that backend would be a new recorded repair hypothesis,
not a silent default change.

The localization completed both `B=8` target calls with finite values and
scores, but timed out during the `B=256`, `beta=0` call. This confirms a large
static-batch compile/evaluation bottleneck without proving a numerical target
failure. The remaining C1 allocation is 2,700 seconds after the measured
1,800-second cost pilot and 900-second localization attempts. One final
cost-pilot retry is therefore allowed with `--validation-size=8` and a
2,700-second cap. This is a feasibility-only repair arm: the original
`validation-size=256` remains the frozen default, and the reduced bank cannot
support whitening, candidate selection, or posterior claims. If it completes,
its manifest and checkpoint scope identify the reduced bank; if it times out or
fails, C1 is closed as an unresolved large-batch resource limitation and
C2--C5 do not launch under this budget.

The final small-bank retry (`attempt-04-cost-smallbank`) reached GPU 0,
completed both `B=8` charts and the `B=32` chart-0 beta-0.5 checkpoint, and
then exited with code `124` at the 2,700-second cap before the complete
two-batch receipt. Its timeout record and all immutable partial checkpoints are
preserved in the attempt directory. The C1 allocation is exhausted:
`1,800 + 900 + 2,700 = 5,400` seconds after C0. C2--C5 therefore do not launch
under this subplan. This is a real continuation blocker for the current budget
and graph route, not evidence that the reverse-KL ensemble idea, whitening, or
mode exploration is mathematically invalid. Resumption requires a newly
authorized budget or a reviewed target-graph optimization, with a new subplan
and fresh output root.

## Frozen Phase 9 diagnostics

The first Phase 9 pilot uses four chains, the repository sequential policy,
at least 2,000 discarded warmup transitions per chain, the latest 1,000 warmup
window, warmup maximum rank-normalized split/folded R-hat `<=1.05`, and retained
admission R-hat `<=1.01`. The controller may grow warmup and retained sampling
to 10,000 transitions per chain under the remaining budget.

Bulk and tail ESS must each be at least 400 per sampled parameter and for the
declared-region indicator. The value 400 derives the approximate relative
Monte Carlo standard error bound `MCSE/SD <= 1/sqrt(400)=0.05`; direct MCSE is
also reported and takes precedence when autocorrelation estimates disagree.
The declared-region occupancy MCSE must be at most 0.025, its worst-case value
at ESS 400.

Each cold chain must visit both strict sign regions and make at least four
retained sign crossings. Four is a minimal per-chain observability guard, not a
convergence proof; ESS/MCSE is the quantitative gate. Start-stratified mean
differences must have 95% MCSE intervals contained within `+/-0.1` pooled
posterior standard deviations for each parameter, and the sign-occupancy
difference interval must lie within `+/-0.05`. These equivalence margins are
q=20 campaign hypotheses and cannot become cross-model defaults.

For replica travel, require at least 16 aggregate completed cold-to-hot-to-cold
and 16 hot-to-cold-to-hot round trips and at least one completed round trip per
replica identity. Sixteen corresponds to an approximate Poisson relative count
uncertainty of `1/sqrt(16)=25%`; failure rejects promotion but does not invalidate
the exact kernel.

## Commands

The planned Phase 8 runner is
`docs/benchmarks/run_ssl_lstm_q20_tempered_rkl_transport_ensemble_phase8_2026_08_29.py`.
The repository-default GPU command is:

```text
bash scripts/run_ssl_lstm_q20_tempered_rkl_phase8_gpu_default.sh
```

The launcher sets GPU visibility and memory growth before importing TensorFlow,
refuses to overwrite an output directory, and delegates numerical and artifact
checks to the Python runner. Operational overrides are recorded in the
manifest. The runner records `external_approval_is_runner_gate=false`; an outer
service denial is an execution interruption, not a change to the repository's
GPU default. Later commands must retain this launch boundary and the subplan's
candidate sets and budget.

For the bounded repair, invoke the same launcher with
`BAYESFILTER_PHASE8_MODE=target-localization`,
`BAYESFILTER_PHASE8_TIMEOUT_SECONDS=900`, and a fresh
`BAYESFILTER_PHASE8_ATTEMPT_LABEL`. No idle-GPU probe or per-run Luna reviewer
is part of either mode. The final small-bank retry uses
`BAYESFILTER_PHASE8_MODE=cost-pilot`,
`BAYESFILTER_PHASE8_VALIDATION_SIZE=8`,
`BAYESFILTER_PHASE8_TIMEOUT_SECONDS=2700`, and another fresh label.

## Pre-mortem and repair ladder

| Apparent result | Strong alternative explanation | Cheapest discriminator | Action |
|---|---|---|---|
| Low reverse-KL loss and flat base-bank residual | Every chart covers the same local basin | Cross-chart bank plus declared-region occupancy | Repair branching/ladders; do not promote |
| Poor residuals | Undertraining, capacity, learning rate, or invalid score | Update ladder, two capacities, two rates, score parity fixture | Repair the numerical/training cause first |
| Reliable charts but no travel | Temperature overlap or HMC tuning is poor | Adjacent energy overlap and per-scope tuner results | Repair ladder/tuning in Phase 9 |
| Good short movement | Short-chain transient or favorable starts | Canonical four-chain retained protocol | No promotion from short movement |
| Full-suite timeout | Compile latency rather than assertion failure | Single-test localization and duration report | One bounded retry |
| GPU allocation failure | Shared-resource contention or batch too large | Trusted device snapshot and `B=8` retry | Preserve failure; retry smaller batch once |
| Checkpoint mismatch | Earlier-beta state was overwritten or incompletely serialized | Fresh-object replay fixture | Hard engineering repair before training |
| All architectures unreliable | Architecture family or target numerics are incompatible | Bounded activation/scale-cap diagnostic | Continuation veto only after repair exhaustion |

## Skeptical pre-execution audit

- **Wrong baseline:** repaired by the explicit single-chart, component-count,
  ladder, branching, physical-HMC, and physical-replica-exchange ladder. Phase 8
  trains/nominates; Phase 9 compares samplers.
- **Proxy promoted:** repaired. Loss and whitening residuals nominate and
  explain only. Paired loss improvement merely prevents an unchanged map from
  being nominated. Exactness, reliability, and later posterior diagnostics
  retain their distinct roles.
- **Hidden state overwrite:** repaired by immutable per-temperature snapshots
  and fresh-object replay before continuation.
- **Tautological whitening:** repaired. Generated base moments are forbidden;
  density/score residuals are computed now and retained-pullback diagnostics
  later.
- **Unfair comparison:** target calls, `K^2 B` transport work, compile time,
  peak memory, and wall time are recorded separately. Mechanism arms change one
  main feature at a time.
- **Search contaminates confirmation:** repaired with disjoint frozen roots and
  a Phase 8 code path that cannot access Phase 9 roots.
- **Unsupported numbers:** every number above is classified as user supplied,
  derived from controller policy, derived from Monte Carlo error, measured from
  Phase 7, legacy warm start, or bounded campaign hypothesis. Hypotheses are not
  promoted across models.
- **No-learning candidate:** repaired by paired held-out comparison against the
  exact beta-start checkpoint across three disjoint banks and separate reporting
  across initialization roots.
- **Unstable extreme quantile:** repaired by excluding q99/max residuals from
  selection; radial stress and reliability checks carry the bounded tail test.
- **Environment mismatch:** GPU/XLA/memory growth is a hard training gate;
  CPU-hidden commands are compatibility tests only. The repository launcher
  supplies the GPU default directly and does not rely on an idle-GPU probe.
- **Commands fail to answer the question:** C0 tests state identity, C1 tests
  feasibility, C2 tests target-specific training choices, C3 tests the proposed
  mechanism, C4 tests the optional quadratic arm, and C5 freezes without using
  confirmation evidence.
- **Missing stop conditions:** target invalidity, reliability repair exhaustion,
  checkpoint mismatch after repair, campaign exhaustion, or platform denial are
  explicit stops. Candidate mode lock is a Phase 9 rejection/repair trigger, not
  an implementation-invalidity claim.

Audit verdict:
`PASS_FOR_C0_COMPATIBILITY_AND_IMPLEMENTATION; C1_BUDGET_EXHAUSTED_GRAPH_LIMIT`.
C0 subsequently passed all
three full/focused compatibility invocations: 9 complexity-target tests, 38
predictive tests, and 39 focused route tests. After checkpoint scope was added
to the immutable hash, the affected transport suite passed again with 13 tests.
The cost-pilot harness parses the JUnit receipts, passes its forbidden-route
scan, and persists beta-start and final checkpoints in a parent-linked chain.
GPU C1 is now `BUDGET_EXHAUSTED_LARGE_BATCH_GRAPH_LIMIT` after attempts 02--04;
C2--C5 remain pending and are not authorized by this subplan. The scientific plan has no idle-probe
or per-launch
approval requirement. Historical approval-service denials remain preserved as
platform evidence, not candidate failures. If the outer service refuses the
launcher, that is recorded as an execution interruption and no indirect route
is attempted. Conservatively debiting the complete 3,600-second C0 allocation
and the measured 1,800-second cost pilot, 900-second localization, and
2,700-second small-bank retry leaves 55,800 seconds of the user-supplied
campaign cap outside the closed C1 allocation. That remainder is reserved by
the parent program and cannot be silently reassigned to C2.

## Required result tables

The Phase 8 result note must contain:

| Decision | Primary criterion | Hard veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|

and

| Inference status | Result |
|---|---|
| Hard veto screen | |
| Statistically supported ranking | |
| Descriptive-only differences | |
| Default readiness | |
| Next evidence needed | |

Every failure record states whether it invalidated the harness, implementation,
target, data, mathematics, candidate, or only the available evidence.

## 2026-08-30 strict-backend repair amendment

The original C1 allocation remains closed with status
`C1_BUDGET_EXHAUSTED_LARGE_BATCH_GRAPH_LIMIT`. A separately reviewed repair
arm tested the existing `tensorflow_eigh_strict` backend after a same-input
q=20 parity check. Parity passed, and the full-256 K=2 two-batch cost pilot
completed in `261.52175762609113` seconds with B=32 selected and peak
TensorFlow allocation below 4 GiB. The amendment is documented in
`docs/plans/bayesfilter-ssl-lstm-q20-phase8-c1-strict-backend-cost-subplan-2026-08-30.md`
and
`docs/plans/bayesfilter-ssl-lstm-q20-phase8-c1-strict-backend-cost-result-2026-08-30.md`.
This result reopens feasibility investigation only. It does not silently
change the frozen backend or authorize C2--C5; a refreshed C2 calibration
subplan must bind the strict backend, include a larger-batch parity check, and
state its own budget before training search.

## Current closeout superseding historical pending text, 2026-08-31

The strict-backend C2 calibration, C3A lineage/overlap and diversity repair,
C3B L5 ladder, C4A K=4 feasibility pilot, and fresh C4B K=4 replication have
all completed under their own fresh subplans and receipts.  C4B passed its
implementation/resource hard screen in `876.4273084410233` seconds, but its
unpaired objective contrasts had opposite signs and did not support retaining
the optional joint arm.

C5 then completed as a metadata-only freeze.  It selected one K=2
`compact-high` representative with `L3=(0,.5,1)`, pure continuation, and
fixed state-independent uniform gamma.  K=4 is preserved as diagnostic
evidence and marked `NOT_RETAINED_FOR_PHASE9`.  The active C5 result is
`docs/plans/bayesfilter-ssl-lstm-q20-phase8-c5-freeze-result-2026-08-31.md`.
All historical paragraphs above that say C2--C5 are pending describe their
state at the time they were written and are superseded by this closeout.
Phase 9 remains closed until a new reviewed tuning/validation subplan is
written and survives the skeptical pre-execution audit.
