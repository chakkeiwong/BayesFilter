# LEDH Surrogate-Force HMC — Execution Status (Damping Sweep)

**Date:** 2026-09-13
**Branch:** `surrogate-hmc`
**Governing plan:** `ledh-surrogate-hmc-program-reconciliation-2026-09-12.md`
(authoritative program: `ledh-surrogate-hmc-executable-master-program-2026-09-07.md`,
with the reconciliation amendment folding damping calibration into Phase 4a)

**Status of this note:** execution log for the pilot-scale damping sweep.
No scientific claim is made or certified here.

---

## Configuration status (read before any number)

- **Program:** PILOT variant, not the plan-spec calibration run.
  Diff vs plan: N=252 (plan 1008), 200+200 steps (plan 500+500).
- **Tuning:** UNTUNED for this scope. No per-scope tuning artifact exists for
  (LGSSM d=3, T=50, N=252, contract_e, float64, dual-parameter target).
  Under the LEDH Per-Scope Tuning Rule this run carries **no per-model claim**.
- **Do not conclude** from this run: a damping ranking, W₂/W₁ agreement,
  posterior correctness, HMC convergence, or Corollary 5.2 validation.
  The pilot can only *nominate* damping ratios and *veto* broken ones.

---

## Defects found and fixed before launch

### 1. GPU whole-device preallocation (project rule violation) — FIXED

The pilot script never enabled TensorFlow GPU memory growth. Measured
evidence: with the script running, `nvidia-smi` reported **15640 MiB** used on
the selected GPU; killing the process dropped it to **2098 MiB**. So ~13.5 GB
was preallocated.

This violates the CLAUDE.md TensorFlow GPU Memory Rule, which requires memory
growth to be enabled *and verified* on every visible physical GPU before device
initialization, and requires serious runs to **fail closed** if it cannot be.

Fix: a module-level block after `import tensorflow` (before any bayesfilter or
TFP import, so it precedes device init) enables growth on each visible GPU,
re-reads `get_memory_growth`, and raises if the policy did not take effect. The
verified policy string is recorded in the run manifest.

Post-fix measurement: 2361 MiB in use on the run's GPU, growth reads back
`True`.

### 2. Wrong GPU selected — FIXED

`CUDA_VISIBLE_DEVICES=1` alone selected the **RTX 5080**, not the 4080 SUPER.
Cause: `CUDA_DEVICE_ORDER` was unset, so CUDA used its default
`FASTEST_FIRST` ordering, which does not agree with the `nvidia-smi` index
order (`PCI_BUS_ID`). The intended card was also the busier one at the time
(GPU 0 held ~2 GB from another process).

Fix: launch with `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1`,
verified to resolve to `RTX 4080 SUPER, pci bus id 0000:09:00.0`. Both env
vars are recorded in the manifest.

### 3. `np.savez` would have written pickled object arrays — FIXED

Diagnostics were passed as **dict-valued** kwargs to `np.savez`. NumPy stores a
dict as a 0-d object array, which cannot be re-read without
`allow_pickle=True`; the artifact would have been awkward and fragile.

Fix: flatten to plain numeric arrays (`acceptance_rate_*`, `ess_per_param_*`,
`wall_time_*`, `is_accepted_*`, `step_size_*`, `marginal_w1_vs_1x_*`), so the
`.npz` loads without `allow_pickle`.

### 4. No run manifest — FIXED

Added a `run_manifest.json`: git commit and dirty flag, command, conda env,
Python/TF/TFP versions, verified GPU memory policy, GPU device list, both CUDA
env vars, execution mode, scale, all seeds (including per-arm chain seeds),
damping base values, per-arm wall times, artifact paths, and an explicit
statement of the pilot's deviation from the plan scale.

### 5. Screen measured mixing but not agreement — FIXED

The pilot's screen was acceptance ≥ 0.15 and ESS ratio ≥ 0.3. Both are mixing
diagnostics. The reconciled Phase 4a promotion criterion is distributional
agreement between damped arms and the 1× baseline, which the script did not
compute at all.

Fix: added per-coordinate Wasserstein-1 between each arm's pooled draws and the
1× baseline, using the exact 1-D identity W₁ = ∫|F_a − F_b| evaluated on a
common quantile grid, reported both raw and in units of the baseline marginal
SD. Computed post-hoc from saved draws, so it costs no extra LEDH evaluations.
This follows existing repo precedent (`tests/hardbound/test_phase2_joint_hmc.py`
compares an HMC marginal to a reference by empirical-CDF W₁) and stays inside
the backend rule, which permits NumPy in post-run diagnostics.

Interpretation, stated up front: Corollary 5.2 makes the θ-marginal invariant
to force quality **by construction**, so a large W₁ here would indicate an
implementation defect, not a failure of the corollary. With ~400 pooled draws
per arm the Monte Carlo floor on W₁/SD is O(1/√ESS), so small nonzero values
are expected even under exact agreement. W₁ is therefore a **repair trigger**
and an explanatory diagnostic in this pilot, not a promotion criterion.

### 6. Exact-value call did 5× the necessary work — FIXED AND VERIFIED

`DualParameterLEDHTarget.__call__` passed `directions = eye(P)` to **both** the
exact-value call and the biased-score call. The engine evaluates directional
derivatives one at a time (`tf.map_fn(..., parallel_iterations=1)` over K), so
each extra direction costs a full filter pass. The exact-value call needs only
the primal value and discarded all P directional scores: 4 of its 5 filter
passes were pure waste.

The engine itself asserts the primal value is direction-invariant
(`assert_near(..., rtol=1e-12)`, `ledh_canonical_batch_fused_tf.py:245`), so
K=1 returns the same value at 1/P the cost.

Fix: the value call passes a single direction; the score call still passes all P.
One value+gradient drops from **10 filter passes to 6** (−40%). The same
one-line defect in `value_only` (which broadcast `eye(P)` for a value-only
computation) is fixed too.

Verified executably, not by argument:
`docs/benchmarks/ledh_value_direction_invariance_parity.py` compares K=P vs K=1
on the same frozen fixture.

```
value @ K=P : -17.062971640105108
value @ K=1 : -17.062971640105108
abs diff    : 0.000e+00      bitwise identical: True
score[0] @ K=P == score[0] @ K=1, abs diff 0.000e+00
RESULT: PASS
```

### 7. Two of five θ coordinates receive no score — FIXED IN DIAGNOSTICS

The gradient verification printed both analytical scores at θ = [1,1,1,0.5,0.3]:

```
exact  score: [-1.4830369  -3.42790933 -2.97842445  0.  0.]
biased score: [-1.4832196  -3.42982463 -2.98098902  0.  0.]
```

θ[3] and θ[4] are **identically zero**, not small. The fixture model explains
why — `_diagonal_lgssm_fused_model`
(`ledh_canonical_neutra_targets_tf.py:484-513`) uses only `theta_rows[:, :3]`,
and its docstring states: *"Frozen-scope noise scales (q=0.35, r=0.45) pinned as
flow/weight inputs; phi directions carry the score."* Both `process_covariance`
and `observation_covariance` are hardcoded constants.

So the runner samples a 5-vector against a model that reads 3 components. θ[3]
and θ[4] receive no HMC restoring force and random-walk without bound. The
`true_theta = [1, 1, 1, 0.5, 0.3]` entries 0.5 and 0.3 are inert.

Three consequences, one of which hit diagnostics added earlier in this session:

1. Summary ESS averaged over all 5 coordinates was partly measuring a random
   walk.
2. **The W₁ agreement diagnostic took a max over all 5 coordinates.** A dead
   coordinate compares one unconstrained random walk against another, so its W₁
   is arbitrarily large and unrelated to damping — it would have swamped the max
   and made every arm look like it disagreed with the baseline. The diagnostic
   introduced to fix defect 5 would itself have produced a spurious verdict.
3. Two of the K=5 score filter passes compute identically zero, so ~33% of the
   remaining score cost is waste for this fixture.

Fix: `score_support()` determines live/dead coordinates at runtime from the
analytical scores (not hardcoded — another model may identify all P). Summary ESS
and the W₁ max are restricted to live coordinates; per-coordinate ESS and the
dead-coordinate W₁ are still printed and saved, labelled as not interpretable.
The runner prints the live/dead split with a warning before any number.

What this does **not** resolve: whether sampling a 5-vector was intended at all.
If θ[3:] were meant to be noise-scale parameters, the fixture model does not
implement that and the intended target is not the one being sampled. If they
were not intended, the sweep should sample a 3-vector, which also cuts the score
from K=5 to K=3 passes. That is a scope question for the owner, not an
implementation detail — recorded, not silently decided.

### 8. The damped force is not cheaper — THE CAMPAIGN'S PREMISE FAILS

This is the most consequential finding in this note, and it is about the
experiment's design rather than its implementation.

Corollary 5.2's value proposition is asymmetric cost: pay for an **exact value**
(required for the Metropolis acceptance ratio) but obtain the **force cheaply**.
The mixing penalty from a biased force is worth accepting only because the biased
force costs less. If it does not cost less, the dual-parameter target performs
two filter evaluations where plain exact-gradient HMC performs one, and buys
nothing.

Measured (`docs/benchmarks/ledh_damping_premise_check.py`, N=24, T=5,
substeps=2, 3 repeats, identical K in both calls — only the damping *values*
differ):

| damping ratio | exact score | biased score | biased/exact |
|---|---|---|---|
| 1× | 15.823 s | 16.317 s | 1.031 |
| 10× | 17.078 s | 15.913 s | 0.932 |
| 100× | 14.958 s | 14.719 s | 0.984 |
| 1000× | 16.013 s | 15.821 s | 0.988 |

Maximum deviation from parity: **0.068**. Every ratio is within measurement
noise of 1.0, and the deviations are not even monotone in the damping ratio,
which is what noise looks like. **The damped force costs the same as the exact
force.**

The mechanism is straightforward once stated: `reset_ridge` and
`correction_lm_damping` are numerical **magnitudes**, not iteration counts.
Changing `reset_ridge` from 1e-5 to 1e-2 changes *what* the arithmetic computes;
it does not change *how much* arithmetic runs. The same Sinkhorn iterations, the
same correction steps, the same pairwise steps, over the same T and the same N.

A second measurement points the same way, and it gets worse at plan scale. At
100× damping the biased score differs from the exact score by:

| scale | rel separation | |
|---|---|---|
| T=5, N=24 | 6.711e-04 | 0.067% |
| **T=50, N=252** (plan horizon) | **3.000e-05** | **0.003%** |

Per-coordinate at plan scale (live coordinates only):

```
theta[0]: exact -63.25427823   biased -63.25438226   rel 1.645e-06
theta[1]: exact -42.70347390   biased -42.70087043   rel 6.097e-05
theta[2]: exact -72.39587139   biased -72.39409073   rel 2.460e-05
```

The separation **shrinks by ~22× toward plan scale**. So a 100-fold change in
both damping parameters perturbs the plan-scale force by 0.003%.

This makes the design problem sharper than "no compute saving": **the sweep has
no effective independent variable.** Four arms at 1×, 10×, 100×, 1000× would run
forces that are identical to within ~1e-5 relative, producing HMC trajectories
indistinguishable within Monte Carlo error at 400 draws per arm.

Note the failure mode this creates, because it is the dangerous kind: such a
sweep would **pass** its screen. Acceptance rates would match, ESS ratios would
sit near 1.0, and W₁ agreement would be well inside tolerance — and the natural
reading of that table is "damping up to 1000× is harmless, promote it." The true
explanation would be that the damping never meaningfully changed the force. A
vacuous pass is harder to catch than a failure, and 9.7 days of GPU time would
have been spent manufacturing one.

Hypothesis for the shrinkage, not verified: `reset_ridge` and
`correction_lm_damping` are regularizers whose influence depends on the
conditioning of the systems they stabilize. With more particles and a longer
horizon those systems appear better conditioned, so the ridge contributes
proportionally less. If so, damping magnitudes are the wrong knob precisely in
the regime the campaign targets. Testing that would need a conditioning
diagnostic on the reset/correction solves, which is cheap relative to the sweep.

**What this invalidates:**

- The 4-arm damping sweep over `reset_ridge` and `correction_lm_damping` as a
  test of Corollary 5.2's practical benefit. It would measure mixing degradation
  against a compute saving that does not exist. Spending 9.7 days (graph) or 84
  days (eager) of GPU time on it would answer a question whose premise is false.

**What this does NOT invalidate:**

- Corollary 5.2 as mathematics. Invariance of the θ-marginal under a biased force
  is a statement about the Metropolis correction, and nothing here bears on it.
- The dual-parameter adapter implementation. It computes exactly what it claims:
  exact value, analytical biased score, verified bitwise and against the
  analytical reference.
- Surrogate-force HMC as a research direction. The idea needs a damping knob that
  actually reduces work.

**What would make the premise true:** damping parameters that cut computation
rather than perturb magnitudes — `reset_sinkhorn_steps`, `reset_balance_steps`,
`correction_steps`, `pairwise_steps`, or `substeps`. Halving
`reset_sinkhorn_steps` in the biased call would produce a genuinely cheaper
force, and the measured cost law (independent of N and substeps, linear in T)
predicts which knobs move cost at all. Whether to re-parameterize the sweep this
way is an owner decision about the experiment's design; it is recorded here, not
silently adopted.

Caveat: measured at small scale (N=24, T=5) with 3 repeats. The cost law is
independent of N and substeps, so the parity result should transfer, but a
plan-scale confirmation would be cheap insurance before acting on it.

### 9. The adapter's test file has never run — PRE-EXISTING, NOT FIXED

`tests/inference/test_ledh_dual_parameter_target.py` contains 7 tests. **All 7
error at fixture setup**, so none has ever executed:

```
TypeError: PerPointScoreModel.__init__() got an unexpected keyword argument
'transition_fn'
```

The fixture builds `PerPointScoreModel(transition_fn=..., observation_fn=...,
transition_score=..., observation_score=...)`. The real dataclass
(`ledh_canonical_batch_fused_tf.py:30-43`) requires `transition_mean_fn`,
`transition_mean_tangent_fn`, `observation_fn`, `observation_jacobian_fn`,
`observation_tangent_fn`, `process_covariance`, `observation_covariance`. Only
`observation_fn` overlaps.

So the dual-parameter adapter — the component the whole Corollary 5.2 validation
rests on — has **no working automated test coverage**, including the tests named
`..._value_is_exact`, `..._score_is_biased`, and `..._determinism`, which are
exactly the properties Corollary 5.2 requires. Prior claims of "spot-check
verification passed" rest on ad-hoc benchmark scripts, not on this suite.

This is pre-existing and independent of the changes in this note (it fails at
setup, before any modified code runs). It is not fixed here because repairing
the fixture is a separate piece of work from the execution task, and doing it
inside this change would mix concerns. It is recorded as an open defect because
it materially weakens the evidence base for Phase 3's completion claim.

The two correctness properties I depended on for the K=1 change were verified
directly by the parity script above instead.

### 10. Monitoring defect (mine, not the script's) — FIXED

The first launches piped through `grep | tail -60`, which buffers the whole
stream until exit, so a multi-hour run showed zero progress output. Real runs
must use `python -u` and write to a log file directly.

---

## BLOCKER: the campaign as scoped is ~505× over its stated budget

This is the headline finding. It is arithmetic on measured timings, not a
projection from the plan's assumptions.

### Two independent errors compound

**Error 1 — the evaluation count is undercounted 50×.** The Phase 3 status doc
estimated cost as `4 arms × 2 chains × 1000 steps × 2 calls = 16,000
evaluations`, counting two LEDH calls per *HMC step*. But each HMC step runs
`num_leapfrog_steps=10` force evaluations, and each force evaluation expands to
K=5 sequential filter passes for the analytical score (plus one for the value).
Correct count for the pilot: 4 × 2 × 400 × 10 = **32,000 value+gradient calls**
= 192,000 filter passes after the K=1 fix (320,000 before it).

**Error 2 — cost does not scale with particle count.** Measured, T=5,
substeps=2:

| N | time for one value+gradient |
|---|---|
| 24 | 22.18 s |
| 252 | 21.96 s |

N ×10.5 → time **×0.99**. The cost is bound by the sequential stage count
(T × substeps), not by N. The pilot's entire reason for choosing N=252 over the
plan's N=1008 — that fewer particles cost less — is **false**. The pilot is not
a cheaper run; it is the same run with weaker statistics.

### Measured cost law

`docs/benchmarks/ledh_cost_structure_probe.py`, one value+gradient call
(= 6 filter passes after the K=1 fix), N=24 unless noted, sinkhorn=2:

| config | time | scaling |
|---|---|---|
| T=5, substeps=2, N=24 | 22.722 s | reference |
| T=5, substeps=2, N=252 | 21.956 s | N ×10.5 → **×0.99** |
| T=10, substeps=2 | 45.444 s | T ×2 → **×2.00** |
| T=5, substeps=4 | 22.917 s | substeps ×2 → **×1.01** |

The cost law is **linear in the observation horizon T, and independent of both
particle count N and flow substeps**. Per-timestep cost is 4.544 s, identical to
three decimals at T=5 and T=10.

Independence of N and substeps says the per-stage work is already vectorized on
the GPU; what costs time is the *number of sequential Python-dispatched stages*,
which T alone sets here.

At the plan horizon T=50, eager: **227 s ≈ 3.8 min per value+gradient call**.

- pilot (32,000 calls): **2,020 h ≈ 84 days**
- plan scale (80,000 calls): **5,049 h ≈ 210 days**

Against the Phase 4a budget of 4 GPU-hours, the pilot is **~505× over**.

(An earlier draft of this note extrapolated with a T × substeps stage model and
reported ~329 days. The substeps-independence measurement above supersedes that;
84 days is the correct eager figure. The probe script's own printed
extrapolation carries the same superseded stage model.)

### The one lever that helps: graph compilation

Q3 of the same probe, identical config, eager vs `tf.function` with a stable
`input_signature`:

| | time |
|---|---|
| eager, best | 21.157 s |
| graph, trace (one-time) | 30.687 s |
| graph, steady-state best | 2.446 s |

**Steady-state speedup ×8.65**, with the one-time trace cost amortizing after
~2 calls. Graph mode also returned the same value, so this is a genuine
speedup rather than skipped work. This directly confirms the diagnosis: the
cost was Python op-dispatch latency, not arithmetic.

With graph mode at T=50: 0.525 s per timestep → **26.3 s per value+gradient
call**.

- pilot (32,000 calls): **234 h ≈ 9.7 days** — still 58× over a 4-hour budget
- plan scale (80,000 calls): **584 h ≈ 24.3 days** — 146× over

Vectorizing the K=5 directions as well (independent by construction, but
`tf.vectorized_map`/pfor requires prior written owner approval under the repo
TensorFlow policy) would give a further ~5×: pilot ≈ 47 h, plan ≈ 117 h.

**What actually fits in 4 GPU-hours with graph mode at T=50:** ~548
value+gradient calls, i.e. 4 arms × 2 chains × **7 HMC steps**. Seven steps per
chain supports no damping conclusion whatsoever.

So graph compilation is necessary but not sufficient. Even with both
optimizations the campaign is a multi-day run, and the gap to the stated budget
has to be closed by an owner decision about compute or scope, not by an
implementation choice.

### Why eager mode was being used at all — a granularity error

Asked directly, and the answer is that eager mode is a **defect**, not a design
choice, and it violates this repo's own TensorFlow Graph policy ("repeated
TensorFlow scientific kernels must execute through `tf.function` with an
explicit, stable `input_signature`; eager is reserved for diagnostics, smoke
checks, and documented exceptions").

The runner carried only `# NO tf.function - let it run in eager mode to avoid
graph issues` — no named issue. There is **zero** `tf.function` anywhere in the
call chain (`ledh_canonical_batch_fused_tf.py`, `ledh_dual_parameter_target.py`).

Provenance, from the Task 3.2 status doc attempt table:

| Attempt | N | Steps | Mode | Outcome |
|---|---|---|---|---|
| 1 | 5000 | 10k+10k | tf.function | OOM (~45 GB RAM) |
| 2 | 1000 | 10k+10k | tf.function | OOM (~45 GB RAM) |
| 3 | 1000 | 500+500 | **eager** | OOM (~45 GB RAM) |
| 4 | 1008 | 500+500 | **eager** | GPU OOM (~13 GB VRAM) |

Attempts 1–2 wrapped **`tfp.mcmc.sample_chain`** in `tf.function`. That unrolls
every HMC step into a single graph — 10,000 steps × 400 stages — so of course it
exhausted host memory. The conclusion drawn was "avoid `tf.function`". The
correct conclusion was "compile at the right granularity": around **one force
evaluation**, where the signature is a fixed `[P]` vector and the call repeats
32,000 times.

Two further facts confirm eager was never the fix:

- Attempts 3 and 4 **OOM'd in eager mode too**, and so did my own plan-scale
  probe (eager warmup at N=1008 died with `ResourceExhaustedError` at
  `d_transport`, `ledh_unified_reset_tf.py:142`). Eager never solved the OOM it
  was adopted for.
- The actual OOM cause was the autodiff tape through the LEDH filter, which
  `tf.custom_gradient` fixed later. At that point the eager workaround was
  obsolete and should have been removed. It stayed, and silently cost ~8.65×.

**Memory and time are separate axes**, which the original diagnosis conflated:

- **Time** scales with T and call count; independent of N (×0.99) and substeps
  (×1.01).
- **Memory** scales as O(N²) through the N×N transport tangent in the Sinkhorn
  reset; independent of the time budget.

So small N was never a time optimization — but it *is* a memory necessity, and
N=1008 does not fit in 13 GB VRAM in either mode.

### Fix applied and verified

Graph compilation is now the default via
`DualParameterLEDHTarget.as_graph_callable(P)`, compiled at the force-evaluation
boundary; `sample_chain` stays deliberately uncompiled, with the reason recorded
at the call site. `EXECUTION_MODE=eager` remains for comparison only.

Because this touches `tf.custom_gradient` — the mechanism the whole Corollary
5.2 setup rests on — it was verified against the analytical score rather than
merely timed. If tracing had bypassed the registered gradient and differentiated
through the filter instead, HMC would still run and still accept while using the
**wrong force**: a silent wrong-science failure.
`docs/benchmarks/ledh_graph_mode_gradient_verification.py`:

| check | result |
|---|---|
| V1 graph value == eager value | PASS (rel 2.08e-16) |
| V2 graph gradient == analytical **biased** score | PASS (rel 9.52e-16) |
| V3 graph gradient != exact score | PASS (rel 6.71e-04) |
| control: eager gradient == biased score | PASS (rel 0.00e+00) |

Steady-state speedup **×8.69** (eager 22.3 s → graph 2.6 s), independently
reproducing the ×8.65 from the cost probe.

Measured at the plan horizon T=50 (N=252): eager **274 s** per value+gradient
call (warmup 278.8 s, rep 1 274.0 s) — about 21% above the 227 s linear
extrapolation, so the linear law slightly understates cost at large T.

### Why it is this slow

`canonical_batch_fused_value_score` has **no `tf.function` anywhere** (grep:
zero occurrences in `ledh_canonical_batch_fused_tf.py`). Every one of the
~2,400 sequential stages per call dispatches its ops eagerly from Python.
Measured GPU utilization was 10–47%, consistent with launch-latency-bound
execution rather than compute-bound work. The kernel also evaluates directional
derivatives strictly one at a time (`tf.map_fn(..., parallel_iterations=1)`
over K), so the K=5 score is 5 serial filter passes.

This is a **performance-architecture** blocker, not a numerical or scientific
one. Nothing here contradicts Corollary 5.2 or invalidates the dual-parameter
adapter; the adapter computes what it claims to compute (verified bitwise, see
defect 7 below). The campaign simply cannot be executed at the planned scale on
this code path.

### What this does NOT establish

- It does not show the LEDH filter is wrong, or that the damping question is
  ill-posed. The blocker is performance architecture, not mathematics.
- It does not invalidate the dual-parameter adapter. The adapter computes what
  it claims to compute; the K=1 value substitution is verified bitwise.
- It does not justify silently shrinking the scientific scope. Reducing arms,
  chains, steps, or leapfrog depth to fit a budget changes what the run can
  conclude, and that is an owner decision, not an implementation detail.
- The measured ×8.65 graph speedup was obtained at T=5, substeps=2, N=24. It is
  not established that the same factor holds at T=50, substeps=8, N=1008, where
  trace cost grows with the unrolled stage count and per-stage arithmetic is
  heavier. The speedup could be larger (more dispatch overhead to remove) or
  smaller (arithmetic-bound at large N). This needs one measurement at plan
  scale before any schedule is built on it.

### Options, with measured costs

**These options address the cost gap only. Read defect 8 first: with the current
damping parameterization there is nothing worth measuring at any budget, so
every row below is contingent on re-parameterizing the sweep.**

| # | option | pilot cost | preserves the question? | needs approval? |
|---|---|---|---|---|
| 1 | graph-compile the kernel | 9.7 days | yes | no — **done** |
| 2 | 1 + vectorize K directions | 1.9 days | yes | yes (pfor policy) |
| 3 | reduce horizon T=50→10 | 1.9 days | **no** — changes the target | scope decision |
| 4 | reduce leapfrog 10→3 | 3.2 days | **no** — mixing is under test | scope decision |
| 5 | spend the compute as-is | 84 days | yes | yes (compute) |

Option 1 is implemented and verified (see "Fix applied and verified" above); it
was required regardless, being what the repo TensorFlow Graph policy already
asks for. Option 2 remains the largest untapped no-scope-cost lever.

Options 3 and 4 look attractive on cost but are not free: T is part of the
target definition, and leapfrog depth is one of the things the damping sweep is
supposed to measure. Trading either away to fit a budget would produce a run
that cannot answer the question it was commissioned to answer.

Option 5 is now the weakest row, not merely the most expensive: spending 84 days
to sweep a parameter that moves the force by 0.003% would buy a vacuous pass.

### The real decision: re-parameterize the sweep, or drop it

Defect 8 reorders everything. The question is no longer "how do we afford this
run" but "is there a run worth affording".

| # | option | what it costs | what it buys |
|---|---|---|---|
| A | re-parameterize damping onto iteration-count knobs, then sweep | knob search (minutes) + re-planned sweep | a sweep with a real independent variable and a real compute saving to trade against |
| B | sweep the magnitude knobs anyway | 9.7 days (graph) | a table that passes vacuously; arms differ by ~1e-5 in force |
| C | drop the damping sweep from Phase 4a | nothing | Phase 4a reverts to plain LGSSM certification; the Corollary 5.2 *cheap-force* claim goes untested |
| D | test Corollary 5.2 invariance only, not its benefit | short run | confirms the θ-marginal is force-invariant (the mathematical claim) without claiming any speed benefit |

A is the only option that tests what the campaign set out to test. Its
prerequisite is cheap: `docs/benchmarks/ledh_cheap_force_knob_search.py`
measures, for each candidate knob, both the cost ratio and the force change, and
reports which knobs move both. That table is what a re-planned sweep needs, and
it takes minutes rather than days.

D deserves separate mention because it is cheap and answers a real question. The
invariance half of Corollary 5.2 — that the θ-marginal is preserved under a
biased force — is testable with a *deliberately crude* force (say
`correction_steps=0`, which does change the computation) on a short chain. That
would validate the adapter and the corollary's mechanism without pretending the
magnitude knobs deliver a speedup.

### Recommended sequence

1. Graph-compile the LEDH target path behind `tf.function` with a stable
   `input_signature`, with a numerical-equivalence check against the eager path
   (the equivalence check is cheap; Q3 already showed graph and eager agree at
   small scale).
2. Re-measure the cost law at plan scale (T=50, substeps=8, N=1008) to replace
   the extrapolated ×8.65 with a measured figure.
3. Bring the resulting budget to the owner with options 2–5 priced from that
   measurement, and let the owner choose between more compute and reduced scope.

Step 1 and step 2 are authorized infrastructure repair under the campaign
retry rule: same target, data, method, criteria, and hardware class. Step 3 is
the decision boundary.

---

## Status: sweep NOT launched — stopped at a decision boundary

The damping sweep was not launched. Launching it would commit 9.7 days of GPU
time in the best available configuration (graph-compiled), against a Phase 4a
budget of 4 GPU-hours. Materially expanded compute is an owner decision, and
the alternatives that fit the budget (options 3 and 4) trade away part of the
scientific question. So the run waits on direction.

Everything that did not depend on that decision is finished: six defects fixed
(five in the runner, one in the library), two correctness claims verified
executably, the cost law measured, and the feasibility gap quantified.

## Next actions

1. **Owner decision** on the options table above: more compute, pfor approval,
   or reduced scope.
2. Graph-compile the LEDH target path (`tf.function`, stable `input_signature`)
   with an eager-vs-graph numerical-equivalence check. Needs no new approval and
   is required regardless — the repo TensorFlow Graph policy already asks for it.
3. Re-measure the cost law at plan scale (T=50, substeps=8, N=1008) to replace
   the extrapolated ×8.65 speedup with a measured one.
4. Repair `tests/inference/test_ledh_dual_parameter_target.py` (defect 7) so the
   adapter has real coverage of the Corollary 5.2 properties.
5. Then launch the sweep at whatever scale the owner authorizes, with
   `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1`, `python -u`, logging
   to a file, and record results plus an inference-status table in a result note.

## Artifacts from this session

| path | what it is |
|---|---|
| `docs/benchmarks/ledh_cost_structure_probe.py` | cost-law + eager/graph probe (Q1–Q3) |
| `/tmp/ledh_cost_structure.json` | its raw timings |
| `docs/benchmarks/ledh_value_direction_invariance_parity.py` | K=P vs K=1 parity check |
| `docs/benchmarks/ledh_dual_target_scaling_probe.py` | first N-scaling probe (superseded) |
| `docs/benchmarks/ledh_surrogate_hmc_damping_calibration_pilot.py` | runner, with defects 1–6 fixed |
| `bayesfilter/inference/ledh_dual_parameter_target.py` | K=1 value-path fix |

Reproduce the cost law:

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 COST_REPEATS=2 \
  conda run -n tftwogpu --no-capture-output python -u \
  docs/benchmarks/ledh_cost_structure_probe.py
```
