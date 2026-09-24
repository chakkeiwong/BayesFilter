# RECOVERY MEMO — LEDH Surrogate-Force HMC Damping Sweep

**Date:** 2026-09-13
**Branch:** `surrogate-hmc`
**Read this first in a fresh session.** Companion detail:
`ledh-surrogate-hmc-phase4a-execution-status-2026-09-13.md` (defects 1–10).
Governing plan: `ledh-surrogate-hmc-program-reconciliation-2026-09-12.md`.

---

## Owner directive (2026-09-13)

**Run the calibration in TF graph mode.** Already implemented and verified:
`EXECUTION_MODE` defaults to `"graph"` in
`docs/benchmarks/ledh_surrogate_hmc_damping_calibration_pilot.py`, which calls
`DualParameterLEDHTarget.as_graph_callable(P)`. Set `EXECUTION_MODE=eager` only
for comparison. Do not re-adopt eager: it was a defect (see below).

---

## Bottom line

The damping sweep was **not launched**, and the reason is no longer cost — it is
that **the sweep as specified has nothing to measure**. Two facts, both measured:

1. The damped force is **not cheaper** than the exact force (cost ratio within
   6.8% of 1.0 at 1×/10×/100×/1000×; noise floor is ±7%).
2. At plan scale the damped force is **not meaningfully different**: 3.0e-05
   relative separation at 100× damping, which *shrinks ~22×* from small scale.

Corollary 5.2's value proposition is exact value + *cheaper* force. With
`reset_ridge` and `correction_lm_damping` as the knobs, neither half holds. A
4-arm sweep would cost 14.6 days and **pass vacuously** — matching acceptance,
ESS ratios near 1.0, W₁ inside tolerance — because the arms are nearly the same
sampler. That reads as "damping up to 1000× is harmless, promote it". The true
cause would be that damping never changed the force.

**A vacuous pass is the dangerous outcome, harder to catch than a failure.**

---

## Measured cost law (trust these numbers)

| varied | effect on time |
|---|---|
| N ×10.5 (24→252) | **×0.99** — independent |
| substeps ×2 (2→4) | **×1.01** — independent |
| T ×2 (5→10) | **×2.00** — exactly linear |

Time is linear in **T** and in call count; nothing else. Per-timestep eager cost
4.544 s (identical to 3 decimals at T=5 and T=10).

Memory is a **separate axis**: O(N²) through the N×N transport tangent at
`ledh_unified_reset_tf.py:142` (`d_transport`). **N=1008 OOMs in 13 GB VRAM in
both eager and graph.** N=252 fits (~3 GB). So small N was never a time
optimization — it is a memory necessity.

### Eager vs graph, measured (not extrapolated)

| config | eager | graph steady | trace | speedup |
|---|---|---|---|---|
| T=5, substeps=2, N=24 | 22.3 s | 2.6 s | 30.7 s | ×8.69 |
| **T=50, substeps=8, N=252** | **268.9 s** | **39.3 s** | 485.1 s | **×6.85** |

Speedup degrades with scale (×8.69 → ×6.85) as arithmetic takes a larger share.
Trace cost is negligible across 32,000 calls.

### Campaign cost at the measured graph figure (39.3 s/call)

| scope | graph | eager |
|---|---|---|
| pilot 4 arms × 2 chains × 400 steps × 10 leapfrog = 32,000 calls | **14.6 days** | 99.6 days |
| plan 4 × 2 × 1000 × 10 = 80,000 calls | **36.4 days** | 249 days |

4 GPU-hours buys **366 calls = 4.6 HMC steps per chain**. Phase 4a's stated
budget was 4 GPU-hours.

My earlier 9.7-day estimate was extrapolated from small scale and was 0.67×
optimistic; **14.6 days is the measured figure.**

---

## Knob search: which damping knob is actually cheaper?

`docs/benchmarks/ledh_cheap_force_knob_search.py`, N=24, T=5, substeps=2,
sinkhorn=8, 3 repeats. Exact params identical in every row; only biased params
change. Exact reference cost 16.118 s.

| biased-param override | cost | cost/exact | force rel |
|---|---|---|---|
| identical params (noise floor) | 17.223 s | 1.069 | 0 |
| `reset_ridge` ×100 (magnitude) | 15.627 s | 0.970 | 3.506e-04 |
| `correction_lm_damping` ×100 (magnitude) | 15.180 s | 0.942 | 3.756e-04 |
| `reset_sinkhorn_steps` 8→4 | 15.253 s | 0.946 | **4.257e-13** |
| `reset_sinkhorn_steps` 8→2 | 14.703 s | 0.912 | **4.400e-09** |
| `correction_steps` 4→2 | 12.570 s | 0.780 | 3.444e-04 |
| `correction_steps` 4→0 | 10.221 s | **0.634** | 9.599e-04 |
| `pairwise_steps` 4→0 | 10.915 s | **0.677** | 7.668e-04 |

**The script printed "viable knobs: 0", and that verdict is an artifact of my own
gate, not a finding.** I required `force_rel > 1e-3`; the best candidates land
just under it (9.6e-04, 7.7e-04). Read the table, not the verdict.

### What the table actually shows

**Iteration-count knobs cut cost; magnitude knobs do not.** `correction_steps`
4→0 saves **37%**, `pairwise_steps` 4→0 saves **32%**, `correction_steps` 4→2
saves 22% — all well outside the ±7% noise floor. The magnitude knobs save 3–6%,
i.e. nothing. This confirms the defect-8 diagnosis mechanically: magnitudes
change *what* the arithmetic computes, iteration counts change *how much* runs.

So a re-parameterized sweep has real candidates: `correction_steps` and
`pairwise_steps` both deliver a genuinely cheaper force.

### Two findings that are NOT about damping

**1. `reset_sinkhorn_steps=8` is 4× more transport work than the score needs.**
Cutting 8→4 changes the score by **4.257e-13** relative — float64 roundoff
against a score of magnitude ~4.8. Cutting 8→2 changes it by 4.400e-09. The
Sinkhorn iteration has converged well before step 2, so **6 of the 8 iterations
are waste in the EXACT baseline**, worth ~9% of score cost. This is not a
damping knob; it is unnecessary cost in the production configuration. Worth
re-tuning independently of the damping question. (Caveat: measured on this
LGSSM fixture at N=24, T=5. Convergence rate may differ at larger N or on a
harder target — check before changing a default.)

**2. Deleting whole stages barely moves the score — unresolved.**
`correction_steps` 4→0 removes the correction stage entirely, and the score
moves only **9.6e-04** relative (0.096%). `pairwise_steps` 4→0 likewise: 7.7e-04.
Two explanations, not distinguished by this data:

- **(a)** The LGSSM fixture is well-conditioned enough that these stages have
  little to correct — plausible, since it is a linear-Gaussian target.
- **(b)** The score's dependence on these stages is not fully wired — i.e. the
  stages change the particle cloud but their contribution does not propagate
  into the analytical score.

Both stages measurably change **cost** (37%, 32%), so they are definitely
executing. If (b) held it would matter well beyond this campaign: the exactness
of the value/score seam is what Corollary 5.2's acceptance ratio rests on.

**Cheapest discriminating test:** compare the *value* (not the score) at
`correction_steps=4` vs `0`. If the value moves materially while the score does
not, that points at (b). If neither moves, that points at (a) — the stage is
genuinely inactive on this target. This was designed but **not run** (the shell
died first). It is the highest-value next diagnostic and takes minutes.

---

## Ten defects found (detail in the status note)

| # | defect | state |
|---|---|---|
| 1 | GPU whole-device preallocation, ~13.5 GB; no memory growth | FIXED, fail-closed |
| 2 | Wrong GPU selected (`CUDA_DEVICE_ORDER` unset → FASTEST_FIRST) | FIXED |
| 3 | `np.savez` would write pickled object arrays | FIXED |
| 4 | No run manifest | FIXED |
| 5 | Screen measured mixing (accept/ESS) but never agreement | FIXED (W₁ added) |
| 6 | Exact-value call ran K=P passes, discarded P−1 | FIXED, bitwise verified |
| 7 | θ[3], θ[4] have identically zero score → random-walk | FIXED in diagnostics |
| 8 | **Damped force is not cheaper — premise fails** | **OPEN, owner decision** |
| 9 | Adapter's 7 tests all error at fixture setup; never ran | **OPEN, pre-existing** |
| 10 | Monitoring: `grep \| tail` buffered a multi-hour run to silence | FIXED (`python -u`) |

Defects 1 and 2 were project-rule violations (CLAUDE.md TensorFlow GPU Memory
Rule). Defect 7 had corrupted a diagnostic added earlier in the same session:
the W₁ max ran over all 5 coordinates, two of which compare one random walk to
another, which would have swamped the max and made every arm look divergent.

### Defect 9 detail (matters for Phase 3's completion claim)

`tests/inference/test_ledh_dual_parameter_target.py` — all 7 tests error at
fixture setup:

```
TypeError: PerPointScoreModel.__init__() got an unexpected keyword argument
'transition_fn'
```

The fixture passes `transition_fn`, `transition_score`, `observation_score`. The
real dataclass (`ledh_canonical_batch_fused_tf.py:30-43`) requires
`transition_mean_fn`, `transition_mean_tangent_fn`, `observation_fn`,
`observation_jacobian_fn`, `observation_tangent_fn`, `process_covariance`,
`observation_covariance`. Only `observation_fn` overlaps.

So the adapter underpinning the whole Corollary 5.2 validation has **no working
automated test coverage** — including tests named `_value_is_exact`,
`_score_is_biased`, `_determinism`, which are exactly the properties the
corollary needs. Phase 3's "verified" status rests on ad-hoc benchmark scripts.
Not fixed here (separate work from the execution task); recorded because it
weakens the evidence base for Phase 3 being complete.

---

## Why eager mode was there (answers the owner's question)

A defect, and a violation of the repo TensorFlow Graph policy. The runner carried
only `# NO tf.function - let it run in eager mode to avoid graph issues` — no
named issue. There was **zero** `tf.function` in the whole call chain.

Provenance, from the Task 3.2 status doc:

| Attempt | N | Steps | Mode | Outcome |
|---|---|---|---|---|
| 1 | 5000 | 10k+10k | tf.function | OOM (~45 GB RAM) |
| 2 | 1000 | 10k+10k | tf.function | OOM (~45 GB RAM) |
| 3 | 1000 | 500+500 | **eager** | OOM (~45 GB RAM) |
| 4 | 1008 | 500+500 | **eager** | GPU OOM (~13 GB VRAM) |

Attempts 1–2 wrapped **`tfp.mcmc.sample_chain`** in `tf.function`, unrolling
10,000 HMC steps × 400 stages into one graph. That had to OOM. The conclusion
drawn was "avoid tf.function"; the correct one was **"compile at the right
granularity"** — around one force evaluation, where the signature is a fixed
`[P]` vector and the call repeats 32,000 times.

Eager never fixed it: attempts 3–4 OOM'd in eager, and so did my plan-scale probe
(eager warmup at N=1008 died at `d_transport`). The real cause was the autodiff
tape through the LEDH filter, fixed later by `tf.custom_gradient` — at which
point the eager workaround was obsolete and should have been deleted. It stayed
and silently cost ×6.85–8.69.

### Graph fix, verified — not just timed

`DualParameterLEDHTarget.as_graph_callable(P)` compiles at the force-evaluation
boundary; `sample_chain` stays deliberately uncompiled (reason recorded at the
call site).

This touches `tf.custom_gradient`, the mechanism Corollary 5.2 rests on. Had
tracing bypassed the registered gradient and differentiated through the filter,
HMC would still run and still accept **while using the wrong force** — a silent
wrong-science failure. So it was verified against the analytical score:
`docs/benchmarks/ledh_graph_mode_gradient_verification.py`.

| check | T=5, N=24 | T=50, N=252 |
|---|---|---|
| V1 graph value == eager value | PASS 2.08e-16 | PASS 5.01e-16 |
| V2 graph grad == analytical **biased** score | PASS 9.52e-16 | PASS 2.34e-16 |
| V3 graph grad != exact score | PASS 6.71e-04 | PASS 3.00e-05 |
| control: eager grad == biased score | PASS 0.00e+00 | PASS 0.00e+00 |

V3's shrinkage (6.71e-04 → 3.00e-05) is defect 8's evidence, not a graph problem.

**Known trap:** wrapping `target(theta)` in `tf.function` from a *free variable*
makes AutoGraph try to convert the object and fail with
`tf____init__() missing 4 required positional arguments`. Use
`as_graph_callable`, where `self` is closed over. My probe script
`ledh_plan_scale_graph_probe.py` hit this at T=50; the class helper did not.

### Evaluation-count arithmetic (the plan undercounted 50×)

The Phase 3 doc used `4 arms × 2 chains × 1000 steps × 2 calls = 16,000`. That
counts 2 LEDH calls per **HMC step**, but each step runs `num_leapfrog_steps=10`
force evaluations, and each expands to K=5 sequential filter passes for the
score. Correct pilot count: **32,000 value+gradient calls** = 192,000 filter
passes (after the K=1 fix; 320,000 before).
