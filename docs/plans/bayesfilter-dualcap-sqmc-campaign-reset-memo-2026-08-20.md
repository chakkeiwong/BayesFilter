# Dual-Cap SQMC Campaign Reset Memo

Date: 2026-08-20
Status: `BLOCKED_AWAITING_LEDH_COVARIANCE_REPAIR__HANDOFF_TO_CONTINUATION_AGENT`

Owner: chakwong
Primary Lane: GenUT score accuracy for high-dimensional nonlinear filtering
Secondary Lane: SQMC particle-count feasibility and numerical stability

---

## Executive Summary for Continuation Agent

This memo documents a 48-hour campaign (2026-08-18 to 2026-08-20) that
established N=16128 feasibility for four SQMC (Sequential Quasi-Monte
Carlo) variants of the dual-cap + trust-region GenUT filtering algorithm on
a consumer 16 GB GPU, repaired a critical score-recursion memory bug
blocking large-N evaluation, characterized three distinct
perturbation-amplification mechanisms (TF32 matmul error, FP32 near-tie
Hilbert swaps, self-anchored moment restoration), and discovered a major
algorithm-identity defect in the LEDH (Localized Exact Daum-Huang)
invertible particle-flow route that invalidates its interpretation as
"UKF-initialized Li(17) flow."

The lane is **blocked** pending the owner's repair of the LEDH covariance
wiring (Austria SIR model callbacks currently use placeholder `eye(18)`
identities instead of UKF/GenUT predicted covariances). Once repaired, the
continuation agent should: (1) validate the repaired route's smoke rows at
N=1008/4032 under a fresh tuning scope, (2) analyze the completed
stage-gain ledger artifact (sitting unanalyzed at
`docs/benchmarks/artifacts/dualcap-stage-gain-ledger-20260820/`), (3) build
an LEDH-route tangent-growth probe to A/B test repaired vs
identity-covariance proposals, and (4) design the multi-seed
variance-scaling ladder for the score-accuracy campaign.

**Key technical insight from this session:** the filter's catastrophic
perturbation amplification (800–1500× over T=20 while the physical SIR
dynamics contract to 0.36×) is structural, not numerical — the algorithm
lacks a damping mechanism. When the model becomes contractive (eigenvalue <
1), the filter should inherit that contraction; instead, the per-step map
has gain ≥ 1 built into its self-anchored moment-restoration scheme. The
missing UKF covariance (which would shrink along contracting directions)
and the missing damping mechanism may be the same defect.

---

## Campaign Background and Objectives

### Starting Context (2026-08-18)

The owner directed a particle-count feasibility campaign after an earlier
N=8064 attempt OOM'd (out-of-memory) during score evaluation. The goal:
establish whether N=16128 — the next ladder rung for the score-accuracy
investigation — is computable on the available hardware (NVIDIA RTX 5080
16GB, RTX 4080 SUPER 16GB) under the modified dual-cap + trust-region GenUT
algorithm, and if so, across which SQMC ancestry/reset variants.

The dual-cap correction (diagonal + pairwise higher-moment matching with
rowwise radial cap rms=2.0 and coordinate power-8 cap at 0.98 quantile) was
selected 2026-08-07/08 as the default family to stabilize the pairwise
gradient updates, which showed large directional instability and made N=4032
score variance *worse* than N=1008 in the 2026-08-01 diagnostic. The
trust-region reset variant was promoted 2026-08-17 after the Austria
GenUT/NeuTra root-cause campaign established that the dual-cap +
trust-region route passed finite-difference validation (gate C) where
earlier higher-moment routes failed.

### Scientific Questions

1. **Feasibility:** can N=16128 particles be propagated through T=20 SIR
   filtering steps with dual-cap correction, trust-region reset, and the
   O(N²) all-parent-marks score estimator on 16 GB consumer GPUs?
2. **Memory ceiling:** what is the allocator peak, and which operation
   (transport, score recursion, correction loop) is the binding constraint?
3. **SQMC variant comparison:** do all four ancestry/reset policies
   (`repaired_permutation`, `iid_dual_cap`, `previous_inverse_cdf`,
   `repaired_fixed_previous_controls`) remain feasible at this scale?
4. **Numerical stability surfaces:** which perturbation mechanisms drive
   trajectory divergence (TF32 matmul precision, ancestry ordering
   instability, smooth chaos amplification), and at what scales do they
   dominate?
5. **Performance:** row wall time, streaming vs dense transport mode,
   XLA-unrolled vs while-loop score recursion.

### Lane Constraints and Policies

- **One seed, no ranking:** all campaign rows use frozen seed 97701; values
  are descriptive only, no accuracy claim, no route promotion. Multi-seed
  evidence is reserved for the score-accuracy campaign.
- **Frozen model/observations:** Austria SIR d=18, T=20, β=0.55 γ=0.35
  σ_obs=100·exp(θ₂), observations and noise from the baseline seed.
- **TF32-on by default:** `enable_tensor_float_32_execution(True)` for
  performance; TF32-off variants explicitly labeled.
- **Streaming transport preferred:** K=2688 row/col chunks (6×6 tiling of
  N=16128) to bound allocator spike; dense mode as reference only.
- **Verified memory growth:** fail-closed policy per repo TensorFlow GPU
  Memory Rule.
- **LEDH per-scope tuning discipline:** each algorithmic change (new
  ancestry policy, new reset variant, new proposal) is a distinct tuning
  scope; ε/balance/ridge/strength controls are scope-local and not
  comparable across scopes without re-establishing parity.

---

## Technical Achievements

### 1. Score-Recursion While-Loop Repair (Unblocks Large-N)

**Problem:** The N=8064 OOM during score evaluation traced to XLA unrolling
the T=20 score backward recursion (`tf.while_loop` with
`maximum_iterations=horizon`), creating 20 graph copies of the O(N²)
pairwise operations and exhausting compilation memory before runtime
allocation even began.

**Root cause:** The recursion while-loop lacked `parallel_iterations=1`,
allowing XLA to unroll by default.

**Fix** ([plan
2026-08-19](docs/plans/bayesfilter-score-recursion-while-loop-repair-plan-2026-08-19.md)):
added `parallel_iterations=1` to force sequential execution through one
graph copy. Validated at N=4032 (baseline parity: value exact-equal to
unrolled reference, score δ=7e-5 / 0.3% relative, well within the expected
O(N²T) error accumulation from reordered floating-point reductions).

**Impact:** N=16128 score evaluation compiles and runs; allocator peak
drops from >16 GB (OOM) to 7.36 GB. The while-loop score path is now the
default for large-N rows.

**Artifact:**
`docs/benchmarks/artifacts/genut-sqmc-streaming-n16128-20260818/smoke_attempt14/`
(N=4032 parity validation),
`docs/plans/bayesfilter-score-recursion-while-loop-repair-plan-2026-08-19.md`.

### 2. N=16128 Four-Variant Feasibility Table (All Routes Pass)

**Campaign execution:** 5.8 GPU hours (of 8 h ceiling) across two
artifacts, four SQMC variants, K=2688 6×6 streaming, TF32+XLA, seed 97701,
while-loop score recursion.

**Variants tested:**
- `repaired_permutation` (Hilbert one-to-one, trust-region reset)
- `iid_dual_cap` (IID uniform ancestry, dual-cap reset)
- `previous_inverse_cdf` (Hilbert inverse-CDF resampling, trust-region)
- `repaired_fixed_previous_controls` (Hilbert one-to-one, fixed RQMC
  controls, trust-region)

**Results** (all pass route-aware gates):

| Route | TV | Sat | Unique anc | Value | Score (j0, j1, j2) | Wall s |
|---|---:|---:|---:|---:|---|---:|
| `repaired_permutation` | 2.9e-5 | 0 | 16128 (perm valid) | −681.61 | (−34.49, −68.11, 7.34) | 4991 |
| `iid_dual_cap` | 4.0e-5 | 0 | 16128 | −682.61 | (−458.76, 38.99, 9.93) | 6201 |
| `previous_inverse_cdf` | 2.0e-5 | 0 | 15872 (98.4%) | −681.73 | (−281.72, −13.76, 6.86) | 4871 |
| `repaired_fixed_previous_controls` | 2.4e-5 | 0 | 16128 (perm valid) | −681.18 | (−189.48, −36.42, 7.50) | 4891 |

**Gate semantics:** the one-to-one permutation/full-uniqueness requirement
binds only `hilbert_*_one_to_one` ancestry policies; `previous_inverse_cdf`
legitimately resamples with duplication (multinomial-style), and its 98.4%
unique ancestors with harness `row_valid=True` is the correct contract.

**Allocator peak:** 7.36 GB across both artifacts (well under 16 GB limit).

**Verdict:** **N=16128 feasibility established for all four SQMC variants**
on consumer 16 GB GPUs. Values agree across variants to ~0.2% (descriptive
only, one seed); scores realization-scrambled as expected from chaos
(range: hundreds in magnitude, across three digits of variation per
component — the score's Monte Carlo noise at this N/T is enormous, and
single-run scores are meaningless).

**Artifacts:**
- `docs/benchmarks/artifacts/genut-sqmc-streaming-n16128-20260818/smoke_attempt16/`
  (N=16128 repaired_permutation, 83 min)
- `smoke_attempt17/` (three variants, 266 min)
- `docs/plans/bayesfilter-genut-sqmc-streaming-n16128-result-2026-08-18.md`
  (result note with full table)
- `docs/plans/bayesfilter-genut-sqmc-streaming-n16128-reboot-reset-memo-2026-08-18.md`
  (closed campaign memo)

### 3. Perturbation-Surface Localization (Three Mechanisms Identified)

**Objective:** the N=4032 / N=16128 campaign rows showed extreme
sensitivity to arithmetic perturbations — switching GPU (5080 vs 4080
SUPER), toggling TF32, or changing transport tiling flipped values by
0.01–0.04% and fully scrambled score coordinates (hundreds of absolute
delta on components of size ~100). Localize which algorithmic surfaces
inject and amplify these perturbations.

**Method:** seed-matched perturbation pairs at N=4032 (TF32-on 5080 stream
vs 4080 SUPER stream; TF32-on vs TF32-off on 5080 stream; stream vs dense
on 5080 TF32-off). Measure step-0 and step-1 particle-cloud divergence (max
coordinate distance, median/99th-percentile Euclidean distance), permutation
rank displacements (how many positions each particle moves in the Hilbert
sort), and step-1 value/score divergence.

**Findings** (three surfaces, scale-dependent):

#### Surface 1: Near-Tie Hilbert Ordering Swaps (FP32-scale dominant)

At **FP32 precision** (TF32-off dense reference vs stream):
- Step-0 clouds agree to 1.3e-6 (smooth chaos seed still negligible).
- Step-1: 3498/4032 particles swap Hilbert ranks, **median displacement 2
  ranks**, 99% moved ≤8 ranks — classic near-tie signature.
- Step-1 divergence: 7.3e-4, far above what 1e-6 smooth amplification would
  produce in one step.

**Mechanism:** the two runs' pre-sort clouds are nearly identical, but the
Hilbert sort resolves thousands of near-ties differently (quantization
boundaries, floating-point key comparison). Each swap is a *discrete jump*
in which RQMC point and reset slot a particle receives, injecting an error
of inter-particle-spacing size (~1/4032 = 2.5e-4, ~100× the smooth
perturbation). The 7.3e-4 step-1 delta is signature-consistent with this
discrete mechanism dominating over smooth amplification.

**Practical implication:** under TF32-off (or FP64-matmul), **ordering
stabilization** (robust tie-breaking on persistent particle IDs, deeper
Hilbert bit quantization, or soft-rank OT-based pairing) is a justified
investment — it attacks the dominant onset amplifier and buys extended
trajectory agreement (better FD validation, tighter parity tests). It does
not restore bit-stability (the 1e-6 smooth seed still wins eventually), and
it does not rescue score reproducibility (both regimes scrambled scores
fully by T=20).

#### Surface 2: Smooth Chaos (All Scales, Sublinear Growth Phase)

Both TF32 and FP32 arms show **sublinear growth of max-coordinate
divergence through steps 1–3** (max delta grows ~3×–5× per step while
median/99th Euclidean grow ~10×–30×, then saturation/resampling-decorrelation
take over). This is the classical signature of smooth dynamical-system
chaos: a small perturbation along an unstable manifold amplifies
exponentially under the flow, with Lyapunov exponent estimated ~1.2–1.6 per
step in the early expansive phase.

The SIR model's early epidemic phase (steps 1–7, reproduction number R_t >
1, Jacobian eigenvalue > 1) is inherently expansive, and the filter
inherits that instability — this is not a bug, but it does mean any
perturbation (TF32 error, ordering swap, or roundoff) enters a
multiplication regime and rapidly saturates the ~0.01–0.1 scale where the
filter's own moment corrections and coordinate caps become load-bearing.

#### Surface 3: TF32 Matmul Coherent Error (TF32-scale dominant)

At **TF32 scale** (ε ≈ 5e-4 in matmuls):
- Step-0 clouds **already diverged to 9.8e-4** (max coordinate) before any
  ancestry sort — TF32 error enters during the first trust-region /
  dual-cap reset (Cholesky, LM solve, moment standardization).
- Step-1: 4018/4032 swaps, **median displacement 115 ranks**, 94% moved >8
  ranks — the clouds are macroscopically different, so the sort correctly
  orders them differently.

**Mechanism:** TF32 degrades matmuls to ~5e-4 relative precision, and the
reset/correction stack routes every step through **shared global
statistics** (empirical mean/covariance → Cholesky → standardization,
Sinkhorn transport kernel, LM/trust-region solve, flow covariance
propagation). An error in a shared statistic is *coherent*: it displaces
**all N particles simultaneously**, so nothing averages out. A 5e-4 error
in the Cholesky factor shifts the entire standardized cloud by ~5e-4 at
step 0, exactly as measured. Then the closed loop (this step's cloud is
next step's input, through nonlinear SIR + moment corrections + power-8
caps) amplifies multiplicatively — measured saturation to percent level by
step 2.

**Root cause clarification:** this is not "TF32 is bad for Monte Carlo" in
the averaging sense — per-particle independent roundoff *does* average out
at ε/√N. The defect is that this filter is an **interacting particle
system** where particles couple through cloud-level empirical moments, and
TF32 degrades the matmuls that compute those coupling operators. Roundoff
in coupling operators enters at full ε, coherently, and compounds through
the temporal feedback loop. The value estimator (sum of per-step logsumexp
averages) survives; the score (T-step path functional) does not.

**Discriminating test (proposed, not run):** force FP32 *only* on the
moment/Cholesky/solve path (keep TF32 elsewhere) and check whether step-0
agreement returns to ~1e-6. If yes, root cause confirmed and we know
exactly which ~5 matmuls must never run in TF32.

**Artifact:**
`docs/plans/bayesfilter-dualcap-perturbation-surface-localization-diagnostic-2026-08-20.md`
(full three-surface analysis with step-by-step divergence tables and
mechanism attributions).

### 4. Per-Step Growth Diagnostic Recall (2026-08-01 Baseline)

During the damping-mechanism discussion, the owner recalled an earlier
analysis showing the full filter amplifying perturbations at +0.1–0.27 per
step (log growth rate) during the contracting phase (steps 8–20) where the
physical SIR dynamics damp at −0.25/step. I located and verified:

**Source:**
`docs/plans/bayesfilter-genut-score-growth-n4032-comparison-result-2026-08-01.md`
and
`docs/benchmarks/artifacts/genut_score_variance_repair_validation_20260801_n4032/derived_physical_vs_full_20260801_v2/growth_by_step.csv`.

**Key table** (N=4032, 8 tangent probes, steps 8–20 mean):

| Arm | Steps 1–7 | Steps 8–20 | Cumulative T=20 |
|---|---:|---:|---:|
| Physical (RK4 transition only) | +0.2 to +0.63 (R_t > 1) | **−0.25** (R_t < 1) | **0.36× contraction** |
| Full filter, diagonal-only | +0.5 to +1.57 | **+0.096** (7/13 negative) | **819× amplification** |
| Full filter, diagonal+pairwise | +0.5 to +1.0 | **+0.266** (0/13 negative) | **1510× amplification** |

**Interpretation:** the physical SIR transition becomes contractive after
step ~7 (susceptible depletion, eigenvalue < 1); an error injected early
would shrink to 0.36× by T=20 if only the model acted. The filter
manufactures 800–1500× amplification by pinning per-step growth at +0.1–0.3
*even while the model it's filtering contracts at −0.25*. Pairwise is the
larger marginal amplifier (+0.17/step over diagonal), but diagonal-only
still fails to contract (+0.35/step gap vs physics), so substantial gain
lives in the shared reset/transport/reweighting machinery.

**Owner's insight:** "the algorithm is missing a damping mechanism." When
the model becomes contractive, the filter should inherit that contraction
and forget perturbations at nearly the physical rate. Instead, the per-step
filter map has gain ≥ 1 built into its structure, primarily via
**self-anchored moment restoration**: every step, the dual-cap and
Contract-E resets re-standardize the cloud against its own perturbed
empirical moments (compute `target_mean = Σ wₙ xₙ`, `target_cov = Σ wₙ
(xₙ−m)(xₙ−m)ᵀ`, then affinely restore the cloud to match), recycling
moment-space errors at unit gain rather than damping them against any
external or predicted reference.

This session's finding — that the LEDH flow's `transition_covariance` is
wired to `eye(18)` instead of a UKF-predicted covariance that would shrink
along contracting directions — directly connects to this mechanism: the
missing UKF covariance and the missing damping may be the same defect.

---

## Critical Algorithm-Identity Defect (Blocks LEDH Lane)

### Discovery Context

While verifying that the stage-gain ledger experiment (see Artifacts §5
below) was testing the correct algorithm, I traced the LEDH-PFPF-OT
campaign route's initialization to confirm it matched the owner's
definition: "Li(17) invertible particle flow with UKF initialization and
Corenflos OT resampling."

### The Defect

The Austria SIR model callbacks
([ledh_pfpf_genut_model_callbacks_tf.py:448-461](bayesfilter/highdim/ledh_pfpf_genut_model_callbacks_tf.py#L448))
wire the LEDH flow's Gaussian linearization inputs as **placeholder
identity matrices**:

```python
initial_covariance   = lambda theta: tf.eye(18, dtype=theta.dtype)
transition_covariance = lambda theta: tf.eye(18, dtype=theta.dtype)
transition_matrix     = lambda theta: tf.eye(18, dtype=theta.dtype)
```

The transition *mean* is faithful (per-ancestor float64 RK4 via
`sir_score._transition_mean_and_parameter_tangent` — the Zhao–Cui
author-source path). The LEDH flow algebra (`batched_ledh_flow_core_tf`),
its forward log-det Jacobian correction (the defining feature of Li(17)
PF-PF), and all downstream stages (Corenflos-lineage entropic OT reset via
Sinkhorn, dual-cap corrections, trust-region) are present and correctly
wired. But the covariance structure the flow linearizes against — and
critically, the **Cholesky factorization through which pre-flow process
noise is injected** ([ledh_pfpf_genut_initial_rqmc_tf.py:738-741](bayesfilter/highdim/ledh_pfpf_genut_initial_rqmc_tf.py#L738):
`pre_flow = prior_mean + noise @ chol(transition_covariance)ᵀ`) — is a
unit identity.

### Consequences

1. **Algorithm-identity drift confirmed.** The N=16128 feasibility campaign
   and all prior LEDH-route results implement the **identity-covariance
   variant of Li(17) flow**, not the UKF/GenUT-initialized version. The
   finite programs are valid (correct importance weights, no
   bootstrap-filter contamination), but any interpretation claiming
   "LEDH-PFPF-OT with UKF initialization" is false. This is drift of the
   class the owner specifically warned against after debugging a
   bootstrap-PF contamination in another lane.

2. **Directly implicates the damping problem.** An identity transition
   covariance cannot shrink along the directions where SIR compartment
   dynamics contract in steps 8–20. A UKF/GenUT sigma-point predicted
   covariance would capture that contraction (via cubature propagation
   through the RK4 dynamics or its linearization) and pull perturbed
   particles back toward the predicted manifold at the physics' rate;
   `eye(18)` is invariant, wildly mis-scaled for raw-count compartment
   magnitudes, and ignores the correlation structure between S/I
   compartments. The missing UKF covariance and the missing damping
   mechanism are plausibly **the same defect**.

3. **Blast radius appears Austria-specific.** The same file's other model
   constructors (diagonal LGSSM at line 249, generalized SV at 288/327)
   build `initial_covariance` and `transition_covariance` from model
   parameters; the identity-stub pattern is isolated to the Austria
   callbacks (line 454).

### What Survives and What Doesn't

**Survives:**
- N=16128 feasibility conclusions (memory ceiling, streaming transport,
  while-loop score recursion, chunk compliance) — these are properties of
  the transport/score-evaluation machinery, not the proposal quality.
- The perturbation-surface diagnostics (TF32 vs FP32, near-tie swaps,
  smooth chaos) remain valid as measurements *of the route that ran*, with
  the caveat that the identity covariance's contribution to amplification
  is unmeasured.
- The four-variant gates (TV, saturation, unique ancestors, finite) — all
  rows pass their contracts under the identity-covariance semantics.

**Does not survive:**
- Any claim that the campaign rows are "LEDH-PFPF-OT with UKF
  initialization" or that they validate Li(17)'s method as described in the
  literature.
- Direct comparability of the identity-covariance campaign's value/score
  baselines to a repaired route's results — the repaired proposal will
  produce a different particle distribution and may need retuning.
- The assumption that further tuning or precision/ordering fixes will
  restore damping — the structural defect (unit-gain moment feedback +
  invariant flow covariance) must be addressed algorithmically.

### Owner Repair Scope (Blocking)

Fix the three Austria callback lambdas
([ledh_pfpf_genut_model_callbacks_tf.py:454](bayesfilter/highdim/ledh_pfpf_genut_model_callbacks_tf.py#L454))
to return:

1. **`initial_covariance(theta)`**: UKF/GenUT sigma-point predicted
   covariance from the model's prior distribution (Austria SIR: likely from
   the stationary distribution of the initial S/I equilibrium, or from the
   `zhao_cui_sir_austria_model().initial_covariance` if it exists).

2. **`transition_covariance(theta)`**: sigma-point predicted covariance
   propagated per ancestor through the RK4 SIR dynamics (cubature
   third-degree or higher, or unscented transform — the repo already has
   the machinery in `cubature_genut_adapters.py` and the generic
   higher-order cubature rules).

3. **`transition_matrix(theta)`**: the RK4 transition Jacobian
   (finite-difference or analytical, or its sigma-point linearization
   equivalent — the LEDH flow core uses this as the `a_matrix` input for
   its local linearization).

The `transition_mean` path is already faithful and needs no change. The
LGSSM/SV callback constructors (same file, lines 230–360) provide reference
wiring for model-informed covariances.

**Critical note:** because `transition_covariance` feeds the Cholesky that
injects pre-flow process noise into the particle distribution, this fix
changes the realized particles' joint distribution, not just the flow's
internal linearization algebra — it is a **new finite program** requiring
its own tuning scope under the LEDH per-scope rule. The repaired route's
parity/value/score baselines are not comparable to the identity-covariance
campaign's results without re-establishing smoke/gate validation at
N=1008/4032 first.

---

## Artifact Inventory

### Primary Campaign Artifacts (N=16128 Feasibility)

1. **`docs/benchmarks/artifacts/genut-sqmc-streaming-n16128-20260818/`**
   - `smoke_attempt16/result.json`: N=16128 repaired_permutation, 83 min,
     7.36 GB peak, while-loop score, seed 97701, PASS.
   - `smoke_attempt17/result.json`: three variants (iid_dual_cap,
     previous_inverse_cdf, repaired_fixed_previous_controls), 266 min, 7.36
     GB peak, PASS (route-aware gates).
   - Combined closeout table in result note (see References).

2. **`docs/plans/bayesfilter-genut-sqmc-streaming-n16128-result-2026-08-18.md`**
   - Full campaign result note with four-variant table, gate semantics,
     budget closeout (5.8 h / 8 h ceiling).

3. **`docs/plans/bayesfilter-genut-sqmc-streaming-n16128-reboot-reset-memo-2026-08-18.md`**
   - Campaign reset memo covering the OOM ladder, streaming migration,
     status transitions, closed with all four rows passing.

### Score-Recursion Repair Artifacts

4. **`docs/plans/bayesfilter-score-recursion-while-loop-repair-plan-2026-08-19.md`**
   - Plan and validation for `parallel_iterations=1` fix; N=4032 parity
     (value exact-equal, score 7e-5 delta, 0.3% relative).

5. **`docs/benchmarks/artifacts/genut-sqmc-streaming-n16128-20260818/smoke_attempt14/`**
   - N=4032 while-loop validation artifact (TF32 unrolled, dense).

### Perturbation-Surface Diagnostic Artifacts

6. **`docs/plans/bayesfilter-dualcap-perturbation-surface-localization-diagnostic-2026-08-20.md`**
   - Three-surface analysis (TF32 coherent error, FP32 near-tie swaps,
     smooth chaos) with step-by-step divergence tables, mechanism
     attributions, and practical decision tree (precision policy first,
     ordering stabilization second, multi-seed statistics as floor).

7. **Scratchpad perturbation arms** (N=4032, seed 97701):
   - `arm_tf32_on.json` (5080 stream, TF32-on reference)
   - `arm_tf32_off.json` (5080 stream, TF32-off)
   - `arm_fp32_dense.json` (5080 dense, TF32-off)
   - Located in session scratchpad, disposable after analysis complete.

### Stage-Gain Ledger Artifact (Unanalyzed, Ready)

8. **`docs/benchmarks/artifacts/dualcap-stage-gain-ledger-20260820/result.json`**
   - Seven-arm stage-gain decomposition (base reset → +diagonal →
     +pairwise uncapped/capped → cap ladder → +trust region), N=1008, 8
     tangent probes × 8 seeds, per-step log growth through T=20.
   - **Status:** task completed 2026-08-21 00:12 UTC; artifact sits
     unanalyzed pending algorithm repair.
   - **Scope caveat:** this harness propagates through the
     cubature-GenUT adapter's raw RK4 transition + reweight, NOT through
     the LEDH flow — it interrogates the correction/reset stages (shared
     code) but is not an end-to-end LEDH statement. Internally consistent
     with the 2026-08-01 baseline it extends.

9. **`docs/plans/bayesfilter-dualcap-stage-gain-ledger-plan-2026-08-20.md`**
   - Experiment plan, arm definitions, scope caveat, next steps after
     repair.

### Historical Growth Baseline

10. **`docs/plans/bayesfilter-genut-score-growth-n4032-comparison-result-2026-08-01.md`**
    - Per-step tangent growth at N=4032 (diagonal vs pairwise, physical
      transition reference, cumulative T=20 factors).
    - Key finding: physical −0.25/step in contracting phase vs full filter
      +0.10 (diagonal) / +0.27 (pairwise) — the 800–1500× amplification
      source.

11. **`docs/benchmarks/artifacts/genut_score_variance_repair_validation_20260801_n4032/`**
    - `derived_physical_vs_full_20260801_v2/growth_by_step.csv`: per-step
      growth data for all arms.
    - `derived_physical_vs_full_20260801_v2/austria_physical_vs_full_particle_growth.png`:
      owner-recalled graph showing damping crossover.

---

## Open Technical Questions and Next Steps

### Immediate (Post-Repair)

1. **Validate the repaired LEDH route** (owner delivers UKF-covariance
   Austria callbacks):
   - Smoke rows at N=1008 and N=4032 under a **fresh tuning scope** (the
     repaired proposal is a new finite program; ε/balance/ridge/strength
     may need retuning for finite/valid).
   - Establish value baseline and score baseline (single seed, descriptive)
     for the repaired route before comparing to identity-covariance
     campaign results.
   - Verify the flow covariance actually shrinks in the contracting phase
     (steps 8–20): log the condition number or min eigenvalue of
     `transition_covariance(theta)` per step and confirm it drops as
     susceptibles deplete and R_t < 1.

2. **Analyze the completed stage-gain ledger** (artifact §8 above):
   - Confirm pairwise as the largest marginal amplifier and measure whether
     the dual-cap coordinate/radial caps reduce tangent *gain* or only
     bound *magnitude* (different things).
   - Quantify the shared-stage gain (base reset + transport + reweighting)
     to understand how much damping must come from other mechanisms if we
     keep the self-anchored mean/covariance policy.
   - Assemble the per-stage per-step growth table against the physical
     −0.25 reference (CSV format: `[time_index, base_reset,
     diagonal_uncapped, ..., physical_reference]`).
   - **State the scope caveat prominently** in any writeup: this measures
     correction/reset stages under the cubature lane's transition, not
     end-to-end LEDH.

3. **Build the LEDH-route tangent-growth probe** (new JVP path required,
   ~1 day scope):
   - The existing probe harness
     (`run_genut_score_variance_repair_validation.py`) has no forward-mode
     AD through the LEDH flow core (`batched_ledh_flow_core_tf`); it only
     supports the cubature adapter's `transition_tangent`.
   - Extend the flow core with a JVP path (or build a finite-difference
     tangent probe as a cheaper MVP — perturb θ or particles, measure
     δ-response, approximate Jacobian).
   - Run a 3-arm covariance A/B: (a) repaired UKF-predicted covariance, (b)
     `eye(18)` (identity baseline for direct comparison against campaign
     rows), (c) shrinkage blend `λ·predicted + (1−λ)·eye(18)` with λ=0.5.
   - This directly measures how much per-step damping the UKF covariance
     restores, isolating the covariance input's effect on gain.

### Medium-Term (Score-Accuracy Campaign Design)

4. **Multi-seed variance-scaling ladder** (after repaired route validated
   and per-stage gain understood):
   - K seeds (16–25) × N ∈ {1008, 4032, 16128}, compute mean and SD of
     value and score per (N, component).
   - Fit `SD ∝ N^(−α)` per score component to measure whether the repaired
     route under capped pairwise shows canonical 1/N scaling or degraded
     T-dependence (effective N^(−α) with α < 0.5, signaling path-space
     variance growth).
   - Expected cost at current performance: ~14 s/seed·N=1008, 100
     s/N=4032, 83 min/N=16128 per row → the N=16128 tier dominates; 16
     seeds × 83 min = 22 GPU hours. Argues for the performance work
     (score-block fusion, retune) before the big ladder, or running the
     ladder at N ∈ {1008, 4032, 8064} first.
   - Output: empirical variance constant per (N, component), evidence for
     or against 1/N scaling, and the number of seeds K needed to bring
     score relative SD below ~20% (for a ranking/accuracy claim).

5. **Damping mechanism design** (if UKF covariance alone insufficient):
   - The owner's keep-self-anchoring decision (mean/covariance targets
     remain the filter's own weighted empirical moments, no external-filter
     bias) means substantial gain may remain in the shared reset stage even
     after the flow covariance is repaired.
   - Candidate damping schemes that preserve self-anchoring:
     - **Temporally smoothed self-moments**: restore toward an exponential
       moving average of the last K steps' weighted moments rather than the
       instantaneous step-t cloud — still self-anchored (no external
       filter), but the EMA pole introduces damping.
     - **Gain-scheduled correction strength**: scale
       `diagonal_strength`/`pairwise_strength` against an online
       contraction estimate (e.g., the min eigenvalue of the empirical or
       predicted transition covariance) — reduce correction gain when the
       model is already contractive.
     - **Shrinkage on moment updates**: restore toward a blend of
       perturbed empirical + predicted moments, with blend weight λ(t)
       chosen so the restoration loop gain drops below 1 in the contracting
       phase.
   - Each is a new algorithmic change requiring its own scope and
     validation ladder; proper sequencing is: mechanism attribution
     (stage-gain ledger) → single-mechanism test (covariance A/B) →
     residual-gain measurement → damping design → multi-seed accuracy
     validation.

### Long-Term (Ordering Stabilization, Conditional on TF32 Policy)

6. **Ordering stabilization** (only if TF32-off becomes the policy):
   - At FP32 scale, near-tie Hilbert swaps are the onset amplifier (median
     displacement 2 ranks, step-1 divergence 7.3e-4 from 1.3e-6 step-0
     agreement) — stabilizing buys extended trajectory agreement and tighter
     FD validation.
   - Tier 1 (cheap, preserves estimator): FP64 ordering path, deeper
     Hilbert bit quantization (20 bits/axis instead of 12).
   - Tier 2 (weeks, changes finite program): entropic soft-ranking
     (Cuturi et al. 2019 OT-based sorting, TF implementation exists in
     `google-research/soft_sort`), or full OT resampling (Corenflos et al.
     2021 `filterflow` — already literature-adjacent to the Contract-E OT
     reset).
   - **Do not invest until the TF32 policy decision is final** — under
     TF32-on, clouds diverge macroscopically at step 0 before any sort, so
     ordering stabilization is irrelevant. Under TF32-off (or
     FP64-isolated matmuls), it's load-bearing.

---

## Known Limitations and Honest Caveats

1. **All campaign results are single-seed, descriptive only.** No ranking,
   no accuracy claim, no route promotion. The four-variant N=16128 table
   establishes feasibility and mechanical functioning, not statistical
   superiority. Multi-seed evidence is mandatory for any score-accuracy
   claim.

2. **Score variance at N=16128 is enormous.** The four-variant scores span
   hundreds in magnitude per component (j0 range: −458 to −34; j1: −68 to
   +39; j2: 6.9 to 9.9) on a single seed. Pseudo-ensemble SDs at N=1008
   and N=4032 (from perturbation arms) are O(100–550) on components of
   size O(100), i.e. **relative SD near or above 100%** — single-run
   scores are noise. Extrapolating 1/√N scaling suggests even N=16128
   leaves score SD at O(100) on components of O(100), meaning ~25 seeds
   are needed to bring relative SD to ~20%. The score-accuracy campaign
   must be multi-seed and expectation-level by design.

3. **The identity-covariance defect's quantitative impact is unmeasured.**
   We know the LEDH flow was fed `eye(18)` instead of UKF-predicted
   covariances, and we know the full filter amplifies at +0.1–0.27/step
   where the physics contracts at −0.25/step, but we have not isolated how
   much of that +0.35–0.52/step excess gain is due to the invariant
   covariance vs other mechanisms (self-anchored moment restoration,
   pairwise instability, transport/reweighting). The covariance A/B probe
   (Next Steps §3) will measure this.

4. **The stage-gain ledger has a scope gap.** It interrogates the
   correction/reset stages (Contract-E, dual-cap, trust-region) but
   propagates through the cubature adapter's raw transition, not the LEDH
   flow. The stages it measures are literally the shared code, so as
   mechanism evidence about *those* stages it's valid. But the LEDH flow
   step's per-step gain is unknown — it could damp (if the predicted
   covariance shrinks appropriately) or amplify (if the flow's
   linearization or log-det correction injects gain). The end-to-end LEDH
   statement requires the follow-up LEDH-route probe (Next Steps §3).

5. **TF32's root cause is confirmed descriptively, not via ablation.** The
   perturbation diagnostic established that TF32-scale arms diverge at step
   0 (9.8e-4, before any sort) via coherent error in shared matmuls, and
   the mechanism derivation (coupling operators, Cholesky, moment
   standardization) is code-supported. But the discriminating experiment
   (isolate FP32 on the moment/Cholesky path, keep TF32 elsewhere, confirm
   step-0 agreement returns to ~1e-6) was proposed but not run. That test
   would name the specific ~5 matmuls that must never use TF32 and confirm
   the mechanism causally rather than correlationally.

6. **Feasibility does not imply efficiency.** N=16128 rows run in 81–103
   min (1.4–1.7 hours) on current code; a 25-seed ladder at that tier is
   ~35 GPU hours. The while-loop score recursion removed the compilation
   blocker but is slower than the unrolled path at small N (adds Python
   loop overhead). Performance work (score-block fusion, retune K for
   while-loop, trust-region solve optimization) is feasible and would
   directly reduce the ladder's cost, but it's not on the critical path
   until the algorithm repair and mechanism attribution are complete.

---

## Algorithm-Identity Verification Checklist (For Future Campaigns)

The identity-covariance defect was caught during a manual
"is-this-the-algorithm" audit triggered by the owner's drift warning. To
prevent recurrence, propose adding this preflight checklist to campaign
launch procedures:

### Before declaring a route "validated" or "production-ready":

1. **Trace the proposal's input sources** (not just that the proposal
   function is called):
   - Mean: where does it come from? (model callback, fixed, adaptive)
   - Covariance: model-informed or placeholder?
   - For LEDH/flow-based routes: transition matrix, observation Jacobian —
     analytical, finite-diff, or identity stub?

2. **Trace the reset's target statistics** (not just that the reset runs):
   - Mean/covariance: self-anchored (cloud's own moments), external
     (parallel filter), fixed, or blended?
   - Higher moments: cloud's own, external, or zeros?

3. **Trace the ancestry mechanism** (not just that resampling happens):
   - Hilbert ordering: what coordinate map? (fixed supplied vs adaptive
     empirical — the state-map-policy fix from the ordering path)
   - RQMC design: stateless frozen or per-step fresh randomness?
   - Permutation vs sampling: does the route contract require one-to-one?

4. **Verify model callbacks against author source** (where "faithful to
   X" is load-bearing):
   - For Austria SIR: does the transition use the half-step RK4 from
     Zhao–Cui source, in float64, with the author's adjacency matrix?
   - For LGSSM: is the observation matrix the declared one?
   - For any "teacher" or "ground truth": is the model frozen or
     parameterized? If frozen, to what values?

5. **Check for placeholder stubs** (common during development, forgotten at
   promotion):
   - `eye(d)` covariances
   - `zeros()` tangents or higher-moment targets
   - `lambda theta: constant` callbacks that ignore their input

6. **One-sentence algorithm summary → code audit round-trip**:
   - Write what you believe the algorithm is in one sentence (e.g.,
     "Li(17) PF-PF with UKF-predicted covariance and Corenflos OT reset").
   - Audit each noun in that sentence: is there a line of code that
     computes it as described, with no placeholder?
   - If any noun traces to a stub or a "TODO: replace with X" comment, the
     algorithm identity does not match the description.

This checklist is **not** a substitute for reading the code (I did read the
callbacks; I failed to recognize that `eye(18)` was a defect rather than a
design until the owner's "should inherit model contraction" framing made it
obvious), but it structures the audit so critical inputs don't fall through
the gap between "the function is wired" and "the function is wired to the
right source."

---

## Handoff Protocol for Continuation Agent

### What the Owner Will Deliver

The owner is repairing the Austria SIR model callbacks
([ledh_pfpf_genut_model_callbacks_tf.py:448-461](bayesfilter/highdim/ledh_pfpf_genut_model_callbacks_tf.py#L448))
to replace the `eye(18)` stubs with UKF/GenUT sigma-point predicted
covariances. Expect:

1. **`initial_covariance(theta)`**: returns a [18, 18] tensor from the
   prior distribution (stationary S/I equilibrium or model's initial
   covariance if it exists).
2. **`transition_covariance(theta)`**: returns a [18, 18] tensor from
   sigma-point propagation through the RK4 SIR dynamics (cubature
   third-degree or unscented transform).
3. **`transition_matrix(theta)`**: returns the [18, 18] RK4 transition
   Jacobian (analytical, finite-difference, or sigma-point linearization).
4. Likely a **new tuning scope** (e.g., `austria_sir_T20_ukf_covariance`)
   since the repaired proposal is a distinct finite program.

### First Actions on Resumption

1. **Verify the repair landed:**
   ```bash
   grep -n "eye(18)" bayesfilter/highdim/ledh_pfpf_genut_model_callbacks_tf.py
   ```
   Should return no hits in the Austria constructor (line ~448–461). If it
   still returns hits, the repair is incomplete.

2. **Inspect the repair** (understand what was changed):
   ```bash
   sed -n '448,475p' bayesfilter/highdim/ledh_pfpf_genut_model_callbacks_tf.py
   ```
   Read the new `initial_covariance`, `transition_covariance`, and
   `transition_matrix` lambdas. Confirm they depend on `theta` or call
   model functions (not constants). Look for comments indicating the
   sigma-point method used (cubature degree, unscented parameters).

3. **Run smoke validation** (N=1008, new tuning scope, one seed):
   - Use the repaired route's new scope name in the harness (likely in
     `run_ledh_pfpf_genut_initial_rqmc_all_models.py` or a new smoke
     script the owner provides).
   - Target: `finite=True`, `program_valid=True`, `TV <= 1e-4`,
     `saturation=0`, and a finite value/score. The *magnitudes* will differ
     from the identity-covariance campaign (different particle
     distribution) — that's expected and correct.
   - If nonfinite or invalid: the predicted covariance may be
     ill-conditioned (too small eigenvalues, Cholesky failure). Check the
     covariance condition number per step; may need a ridge or
     eigenvalue-floor mitigation.

4. **Smoke at N=4032** (reuse the established ladder):
   - Same gates, ~100 s expected wall time.
   - Establish the repaired route's value baseline (one seed, descriptive)
     before any cross-route comparisons.

5. **Log the flow covariance's contraction** (verify the repair's intent):
   - Add instrumentation to log `tf.linalg.eigvalsh(transition_covariance)`
     or its condition number per step.
   - Confirm the min eigenvalue or condition number *changes* as the
     epidemic progresses (should shrink in steps 8–20 as susceptibles
     deplete and R_t < 1, if the UKF prediction is working).
   - If the eigenvalues stay roughly constant, the covariance is not
     capturing the model's time-varying dynamics — flag to the owner for
     re-inspection.

6. **Analyze the stage-gain ledger** (artifact §8, sitting unanalyzed):
   - `docs/benchmarks/artifacts/dualcap-stage-gain-ledger-20260820/result.json`
   - Aggregate per-step growth across arms, build the table comparing
     (base_reset, diagonal_uncapped, diagonal_pairwise_capped_current,
     cap_tight, cap_loose, full_with_trust_region) against the physical
     −0.25 reference.
   - Confirm pairwise as the largest marginal amplifier; measure whether
     caps reduce *gain* or only *magnitude*.
   - **State the scope caveat** prominently: cubature-lane transition, not
     end-to-end LEDH.

7. **Build the LEDH-route tangent probe** (next diagnostic, ~1 day):
   - Extend the flow core with a JVP path or build a finite-difference
     tangent approximation (perturb particles by δ, measure response, form
     Jacobian).
   - 3-arm A/B: (a) repaired UKF covariance, (b) `eye(18)`, (c) λ=0.5
     shrinkage blend.
   - Compare per-step growth in steps 8–20 against the physical −0.25
     reference to measure how much damping the UKF covariance restores.

### Key Technical Details to Preserve Across Handoff

- **Frozen campaign inputs** (seed 97701, Austria SIR β=0.55 γ=0.35,
  observations/process-noise/initial-noise, Halton RQMC design, state-map
  fixed location/scale) — reuse for cross-route comparisons.
- **While-loop score recursion is the default** for N ≥ 8064 (`genut_sqmc`
  harness after the 2026-08-19 repair).
- **K=2688 streaming transport** (6×6 tiling) is the feasible mode at
  N=16128 (allocator peak 7.36 GB vs >16 GB dense).
- **TF32-on is the current policy** (performance > bit-reproducibility);
  TF32-off variants require explicit labeling and lose ~20–30% speed.
- **One seed = no ranking.** Multi-seed is mandatory for any route
  promotion or accuracy claim (score SD is O(100–550) at N=1008–4032, still
  O(100) projected at N=16128).
- **Tuning-scope discipline:** every algorithmic change (new proposal, new
  reset, new ancestry) is a new scope; ε/balance/ridge/strength are
  scope-local and not comparable across scopes without re-parity.

### Where to Find Things

- Campaign result notes: `docs/plans/bayesfilter-genut-sqmc-*.md`
- Artifacts: `docs/benchmarks/artifacts/genut-sqmc-streaming-n16128-20260818/`,
  `dualcap-stage-gain-ledger-20260820/`
- Harnesses: `docs/benchmarks/run_genut_sqmc_particle_trust_austria_20260817.py`
  (N=16128 campaign), `run_genut_score_variance_repair_validation.py`
  (tangent-growth probe)
- Core routes: `bayesfilter/highdim/ledh_pfpf_genut_initial_rqmc_tf.py`
  (LEDH-PFPF-OT), `dual_cap_genut_primal_tf.py` (dual-cap corrections),
  `genut_guided_proposal_tf.py` (Contract-E reset + trust region)
- Model callbacks (repair target):
  `bayesfilter/highdim/ledh_pfpf_genut_model_callbacks_tf.py`

### Communication with Owner

If the smoke validation fails (nonfinite, invalid, or unreasonable values):
- Report the failure mode (which validity gate fired, condition numbers,
  eigenvalue ranges).
- Do NOT attempt algorithmic fixes or hyperparameter retuning without owner
  direction — the repaired covariance is the owner's design, and
  mitigation strategy (ridge, eigenvalue floor, fallback to identity in
  ill-conditioned steps) is a policy decision.

If the logged covariance eigenvalues do not contract in steps 8–20:
- The UKF prediction may not be capturing the time-varying dynamics —
  report the eigenvalue trace and ask the owner to re-inspect the
  sigma-point propagation.

If everything validates cleanly:
- Proceed to stage-gain ledger analysis (§6 above) and LEDH-route tangent
  probe design (§7 above) per the Next Steps ladder.

---

## Closing Summary for Owner

This 48-hour campaign established the platform (N=16128 feasibility, memory
ceiling, streaming transport, while-loop score recursion) and mechanism map
(three perturbation surfaces, per-step growth decomposition, self-anchored
moment restoration as unit-gain feedback) needed for the score-accuracy
investigation — but it also surfaced a load-bearing algorithm-identity
defect that blocks interpreting any LEDH-route result as the claimed
Li(17)+UKF method.

The finding validates your instinct: the filter's failure to inherit the
model's contraction (steps 8–20, eigenvalue < 1, physics damps at
−0.25/step while filter amplifies at +0.1–0.27/step) and the missing UKF
covariance initialization are plausibly the same defect. An `eye(18)`
covariance is invariant and cannot shrink along contracting directions; a
UKF-predicted covariance propagated through the RK4 SIR dynamics would
capture that time-varying structure and pull perturbations back toward the
predicted manifold at the physics' rate.

Once you deliver the covariance repair, the continuation agent's first
priority is validating the repaired route's smoke rows (N=1008/4032, fresh
scope, confirm finite/valid and that the logged covariance eigenvalues
actually contract in the late phase), then analyzing the completed
stage-gain ledger (sitting unanalyzed, ready), then building the LEDH-route
covariance A/B probe to measure how much per-step damping the UKF repair
restores. That measurement will tell us whether the damping problem is
fully addressed by proper initialization or whether additional mechanisms
(temporally smoothed moments, gain scheduling, shrinkage blends) are needed
to bring the filter's perturbation gain below 1 in the contracting phase.

The score-accuracy campaign (multi-seed variance ladder, 1/N scaling
measurement, route ranking) waits on that mechanism-attribution work —
because if the per-step map still has gain > 1 after the covariance repair,
score variance will grow exponentially in T regardless of how many
particles or how much precision we throw at it, and no amount of seeds will
rescue a structurally unstable estimator.

**Status for handoff:** lane blocked, repair in owner's hands,
continuation agent has clear sequencing (smoke validation → ledger analysis
→ LEDH probe → variance ladder), all artifacts documented and ready.

---

**Document Metadata**

- **Author:** Fable-5 (Claude Code session 2026-08-18 to 2026-08-20)
- **Lane Owner:** chakwong
- **Handoff Target:** Codex / Grok / continuation agent with access to this
  repo and the artifact inventory above
- **Last Campaign Commit:** (not recorded; see artifact sha256 hashes in
  result notes)
- **Blocking Repair File:**
  `bayesfilter/highdim/ledh_pfpf_genut_model_callbacks_tf.py` lines
  448–461 (Austria constructor)
- **Primary Artifact Root:**
  `docs/benchmarks/artifacts/genut-sqmc-streaming-n16128-20260818/`,
  `dualcap-stage-gain-ledger-20260820/`
- **Continuation Plan:**
  `docs/plans/bayesfilter-dualcap-stage-gain-ledger-plan-2026-08-20.md`
