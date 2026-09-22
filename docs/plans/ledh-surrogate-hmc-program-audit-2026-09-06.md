# LEDH Surrogate-Force HMC Program — Skeptical Pre-Execution Audit

**Date:** 2026-09-06
**Auditor:** Claude Code (self-audit of own program, at owner instruction)
**Scope:** Every factual and methodological claim in
`ledh-surrogate-force-hmc-master-program-2026-09-04.md` and its phase results,
plus the pre-existing runners the program would have used.
**Method:** Executable verification against the code on branch
`ledh-refactor-with-policy-fix`. No claim accepted from a document.

**Verdict: 12 findings. 4 blocking, 5 major, 3 minor.**
The program must not execute as written.

---

## Blocking findings

### B1 — The pre-existing Phase 3 runner silently disables the reset

`docs/benchmarks/surrogate_force_lgssm_three_arm_v2.py` (committed 2026-09-01
in 2478d3da, 421 LOC) passes:

```python
reset_policy="transport_affine_cumulant_trust"
```

The dispatch in `ledh_canonical_score_tf.py:413` tests exactly one literal:

```python
if reset_policy == "contract_e":
    ...   # Sinkhorn + Contract-E reset, dual-cap, trust-region
else:
    states, d_states = children, d_children   # NO RESET
```

Verified by enumeration: the only comparison literal in the module is
`contract_e`; the parameter default is `none`; the runner's string matches
neither. There is no validation and no error — the unrecognised policy falls
through to the bare `else`.

So the runner executes the branch the authority's own docstring marks:

> **ESTIMAND WARNING** … with `reset_policy="none"` and `annealed_stages=1` …
> the returned VALUE is **NOT a log-likelihood estimator** for horizons T > 1
> (measured on the frozen LGSSM anchor: N-independent score bias +4.2 at T=50).
> That slice is a derivative-parity/diagnostic object ONLY.

The runner sets `annealed_stages=1` and `HORIZON=50`. It therefore hits the
exact configuration measured at +4.2 bias, while passing 11 reset-family
arguments (`reset_epsilon`, `reset_sinkhorn_steps`, `correction_steps=15`,
`correction_trust_radius=0.5`, …) that are **all silently ignored** because the
branch consuming them never executes.

This reproduces confound **C2** from the 2026-08-29 audit — "the purportedly
matched score cell uses [wrong control family] … These are numerics-altering
controls, not aliases that can be silently equated" — in a file written three
days after that audit was filed.

Consequence: had Phase 3 run this, it would have produced a converged chain
targeting `exp(-U)` for a `U` that is not a likelihood, with every reset
diagnostic reported as if configured. Corollary 5.2 tolerates arbitrary force
bias but requires the *value* be the exact scalar. This breaks the value.

**Required fix:** fail closed on unrecognised `reset_policy` (Class B guard,
adopt-by-default under the Safety Guardrail Reversed Burden policy). Then
correct the runner to `contract_e` with the tuned control family.

### B2 — The runner's observations are placeholder noise

```python
def make_observations():
    """Generate observations matching historical fixture."""
    rng = np.random.default_rng(OBS_SEED)
    # Simple: just noise for now (actual fixture would run Kalman forward)
    return tf.constant(rng.standard_normal((HORIZON, DIM)), DTYPE)
```

The docstring claims the historical fixture; the body returns i.i.d. standard
normal draws. The real frozen dataset exists at
`ledh_canonical_neutra_targets_tf.py:340` (`_lgssm_frozen_observations`, seed
81100, φ=[0.72,0.55,0.35], q=0.35, r=0.45, with the 3×3 observation matrix and
the stationary initialisation).

Against pure noise there is no likelihood signal, so θ is unidentified and the
posterior is the prior. Any "posterior coverage" result would be meaningless —
and could easily *look* like a pass, because a diffuse posterior covers
everything.

**Required fix:** call `_lgssm_frozen_observations()` (or an explicit equivalent
with a recorded hash).

### B3 — Phase 3's ESS criterion has no baseline to compare against

Phase 3 states: *"Arm 2 ESS/gradient > 0.5× Arm 1 (descriptive)"*, and
`Baseline: Exact score as force (current HMC infrastructure, Arm 1)` —
"current HMC infrastructure" implying something already measured.

Verified: no such artifact exists. `docs/benchmarks/artifacts/` contains exactly
one HMC directory, `multidim_lgssm_serious_hmc_tuning_2026_07_09`, which is a
different target (multidim LGSSM Kalman tuning: `geometry.json`, `mass.json`,
`kernel_tuning.json`, `xla_compile_gate.json`). Fifteen `ledh_*` artifacts
exist; none is an HMC run.

So Arm 1 is not a reference to preserved evidence — it must be *executed*, and
its cost is not in the budget. The word "baseline" in the evidence contract
implied a comparator that does not exist.

**Required fix:** Arm 1 becomes an explicit deliverable with its own compute
line, or the ESS criterion is withdrawn and replaced by an absolute floor.

### B4 — The λ/δ damping recipe does not map onto real parameters

The program's scientific question is stated in terms of λ (process-covariance
ridge) and δ (observation-covariance ridge): exact (1e-5, 1e-5) for value,
damped (1e-3, 1e-3) for force. Inherited verbatim from
`implementation_plan_surrogate_force_hmc_2026-08-29.md`, which sketches
`make_adapter(lambda_ridge=1e-3, delta_damp=1e-3)`.

Verified: no such parameters exist. `canonical_value_and_analytical_score`
exposes `reset_ridge` (the Contract-E ridge) and `correction_lm_damping` /
`correction_lm_scale_floor` (trust-region LM controls). Neither λ nor δ is a
process- or observation-covariance ridge. Grep for `lambda_ridge|delta_damp`
across the repository returns 10 hits, all inside the two surrogate runners —
i.e. the names exist only in the drivers that invented them.

v2's mapping is:

| Recipe | v2 maps to | What that parameter actually is |
|---|---|---|
| λ (process ridge) | `reset_ridge` | Contract-E reset ridge |
| δ (observation ridge) | `correction_lm_scale_floor` | trust-region LM scale floor |

The second is a **tuned control**: the 2026-09-03 campaign selected
`higher_moment_lm_scale_floor = 1e-06` for LGSSM T50. v2's damped arm would
overwrite it with 1e-3 — a 1000× change to a numerics-altering trust-region
setting. Under the Per-Scope Tuning Rule that is a new tuning scope requiring
its own artifact, which does not exist. Under the Class C policy it is a
numerics-altering protection changed without calibration.

**Required fix:** derive what "damping the force" means in the parameters that
exist, before running anything. Candidates — the reset ridge, the LM damping,
the flow substep count, the Sinkhorn ε — are *not* interchangeable: they
perturb different stages and have different bias signatures. Choosing among
them is a mathematical decision, not a configuration one, and it changes what
the experiment measures.

---

## Major findings

### M1 — Phase 3's stated criterion is 5-dimensional; the runner is 1-dimensional

Phase 3 requires *"posterior 95% intervals cover true θ (all 5 parameters)"*.
The runner initialises `init_state = true_theta[0] + 0.05*randn(n_chains)` —
a scalar chain over θ₀ only — and its gradient zeroes four of five components:

```python
grad_full = tf.zeros(5, DTYPE)
grad_full = tf.tensor_scatter_nd_update(grad_full, [[0]], [score_tf])
```

A 1-D run cannot satisfy a 5-D criterion. Either the criterion drops to θ₀
(and says so), or the runner is extended to 5 directions — which is precisely
what the refactored kernel's K-direction capability is for, and is the
performance argument for unification.

### M2 — Phase 4's model list belongs to a different program

Phase 4 names Tier A as LGSSM-KF, PP-UKF, PP-SGQF, SIR-SGQF, STR-UKF. Those
are the cells of `bayesfilter-hnn-surrogate-hmc-master-program-2026-07-17.md`
— the neural-force program the owner explicitly identified as *not* this work.

The models with LEDH trust-region tuning artifacts are: Austria SIR T20,
LGSSM T50, KSC SV T10, Predator-Prey T20. Different set, different horizons.
Phase 4's 50 GPU-hours was budgeted against the wrong matrix.

### M3 — The Heuristic Dominance Gate is absent

`CLAUDE.md` mandates it before interpreting any learned or optimised method:
construct (not cite) 3–7 cheap practitioner adversaries, evaluate
**conditionally on each salient situation**, and treat losing to any of them
as the headline and a promotion veto. Its provenance note records a campaign
that passed every internal gate and still lost to buy-and-hold.

The program contains no adversary set. For surrogate-force HMC the obvious
constructed set is: (i) exact-score HMC — Arm 1; (ii) random-walk
Metropolis at matched cost, no gradient at all; (iii) a coarser but *unbiased*
cheaper filter, e.g. fewer particles or fewer flow substeps at matched wall
time; (iv) exact Kalman HMC where available, as the correctness oracle for
LGSSM; (v) fixed-preconditioner HMC ignoring θ-dependence. Salient
situations: high vs low curvature regions, θ near the boundary vs interior.

The third adversary is the sharp one. If spending the same wall clock on an
unbiased-but-coarser filter mixes as well, the whole surrogate-force
construction is unnecessary. That comparison must be planned in, not
discovered later.

### M4 — Seed discipline is ambiguous and its implementation is unsound

Corollary 5.2 requires all Monte Carlo seeds inside the force frozen **across
the trajectory**; a per-leapfrog-step reseed breaks the involution. v2 derives
the seed from θ:

```python
return hash((self.seed_base, theta_tuple)) % (2**31)
```

Since θ changes at every leapfrog step, the noise changes at every leapfrog
step. This is θ-reproducible but *not* frozen along the trajectory — arguably
the exact failure Remark 5.3 warns about. The two readings ("frozen per
trajectory" vs "deterministic function of θ") are genuinely different
estimators and the program never chose between them. This is a correctness
prerequisite, not a detail.

Separately, `hash()` is the wrong primitive: it is not a documented stable
hash across processes, and the repository already has a memory recording that
naive seed derivation produced pseudo-replication
(`tf-consecutive-from-seed-is-one-stream`). Use `SeedSequence`.

The program's own Phase 3 task list says *"verify TFP HMC doesn't re-invoke
`log_prob_and_grad` per leapfrog step"* — never executed.

### M5 — The 100× damping ratio is an uncalibrated Class C hyperparameter

Damping alters the computed numerics, so it is Class C under the Safety
Guardrail Reversed Burden policy: it requires a derivation, a measured
calibration curve, or a recorded owner rationale. "Inherited" and "convenient"
are explicitly not justifications, and the policy states the same burden
applies to the *off* setting.

The 100× ratio traces to a single line in the Aug-29 note with no calibration
behind it. The program's own ladder (1e-3, fallback 1e-4) is a two-point
guess. A bias-vs-robustness curve over the chosen parameter is the normal
evidence form and does not exist.

---

## Minor findings

### N1 — Phase 0's success criterion contradicts the owner's decision

Phase 0 lists `Code on main branch`. The owner subsequently directed that only
fully successful, fully tested work merges, to avoid interfering with other
agents. The criterion is stale and should read: work isolated on the branch
until the program completes.

### N2 — Budget omits the unification phase

The program states ~61 GPU-hours and 23 attempts. Unification is 6–9 days of
predominantly CPU test work, absent from both figures. Phase 4's 50 GPU-hours
is also priced against the wrong model set (M2).

### N3 — Phase 4 promotion rests on single-seed Phase 3 evidence

Phase 3 runs one seed and its contract correctly says *"Statistical Evidence:
None (descriptive comparison only)"*. Phase 4 then gates on Phase 3's
promotion. Under Statistical Evidence Discipline a single-seed descriptive
result can *nominate* but cannot support a ranking; Phase 4's entry condition
should state explicitly that it proceeds on a nomination, not on evidence of
superiority.

---

## What the audit did not find

Stated so the negative results are on record:

- The pfor repair (6 × `tf.vectorized_map` → `tf.while_loop`) is genuine. No
  `tf.vectorized_map` remains in the fused kernel; the 6 parity tests pass.
- `reviewed_value_score_target_fn` exists and is correct
  (`batched_value_score.py:173`): `tf.custom_gradient`, `stop_gradient` on both
  value and score, shape validation, and it closes the captured-variable path
  so HMC cannot accidentally update transport parameters. Phase 2's T3 does not
  need a new wrapper — it needs to *use* this one.
- The frozen LGSSM T=50 fixture is real and reproducible
  (`_lgssm_frozen_observations`, seed 81100).
- The Sept-3 trust-region tuning artifacts are real, complete, and contain the
  full 16-control record for LGSSM T50.
- The two-engine duplication diagnosis holds: `single_cloud` 617 LOC,
  `batch_fused` 736 LOC, `batch` a 94-line Python row loop with no
  mathematics, `neutra_target` a forwarder.

---

## Disposition

| # | Finding | Class | Fix belongs in |
|---|---|---|---|
| B1 | Runner silently disables reset | blocking | fail-closed guard + Phase 3 rewrite |
| B2 | Observations are placeholder noise | blocking | Phase 3 rewrite |
| B3 | No Arm 1 baseline exists | blocking | Phase 3 scope + budget |
| B4 | λ/δ do not exist as parameters | blocking | pre-Phase-3 derivation |
| M1 | 1-D runner vs 5-D criterion | major | Phase 3 scope |
| M2 | Wrong Tier A model list | major | Phase 4 rewrite |
| M3 | No heuristic adversary set | major | Phase 3 evidence contract |
| M4 | Seed discipline unresolved | major | pre-Phase-3 derivation |
| M5 | Uncalibrated damping ratio | major | pre-Phase-3 calibration |
| N1 | Stale merge criterion | minor | Phase 0 correction |
| N2 | Budget omits unification | minor | budget revision |
| N3 | Single-seed → Phase 4 gate | minor | Phase 4 entry condition |

Four of these (B1, B4, M4, M5) are **mathematical or estimand** questions that
must be settled on paper before code. Three (B2, B3, M1) are scope and
deliverable corrections. The rest are bookkeeping.

None of them argues against surrogate-force HMC as a research direction. All of
them argue that the program as written would have produced a confident,
publishable-looking, wrong answer.

---

## Note on provenance

This audit examines a program I wrote, and B1–B4 include defects I would have
executed. The two Phase results corrected on 2026-09-04 (Phase 1's five
unmeasured diagnostics, Phase 2's substituted promotion criterion) are the same
failure mode caught one step earlier. The pattern in all six cases is the same:
a document asserting a verified quantity that was never measured. The
structural fix is that no phase closes on a document — it closes on an artifact
containing the measurement.

---

**END OF AUDIT**
