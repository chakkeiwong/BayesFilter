# LEDH Surrogate-Force HMC — Pre-Mortem and Execution Readiness

**Date:** 2026-09-06
**Question posed:** List every way the program can still fail while all tests
pass; check whether the tests cover those cases; audit the logical consistency
and mathematical correctness of the tests; identify missing steps; state the
allowlist and upfront approvals needed for uninterrupted execution.

**Headline:** 19 residual failure modes. Phase 2C covers 6. **Three are
mathematical errors in the program itself, one of which guarantees a false
negative.** Two are missing validation rungs without which a Phase 4 pass is
uninterpretable. One test tolerance is probably unachievable as specified.

---

# PART 1 — How the program can fail with every test green

Organised by what the failure attacks. `[2C]` = covered by the planned
contract suite. `[GAP]` = not covered by anything currently planned.

## Class A — The estimand and the criterion are mismatched

### A1. The promotion criterion measures the wrong thing `[GAP]` — **BLOCKING**

Phase 4's primary criterion is *"Arm 2 posterior 95% intervals cover true θ."*
What surrogate-force HMC actually targets is

$$\pi_N^\omega(\theta) \;\propto\; \exp\!\big(-U_N^\omega(\theta)\big)$$

the **finite-particle pseudo-posterior at the frozen noise draw ω**. Coverage of
the true θ confounds three independent questions:

1. does the sampler correctly sample $\pi_N^\omega$ (the surrogate-force claim);
2. is $\pi_N^\omega$ close to the exact posterior $\pi_\infty$ (a finite-N
   question, and the object of the +4.2 bias measurement);
3. does $\pi_\infty$ cover the true θ for this dataset (a property of the data,
   not of any algorithm).

A **correct** sampler fails this criterion whenever (2) or (3) fails. An
**incorrect** sampler passes it whenever over-dispersion happens to widen the
intervals over the truth. The criterion is therefore neither necessary nor
sufficient for the claim being tested.

No Phase 2C test catches this, because nothing is miswired — the criterion is
simply not a test of the hypothesis. This is the same defect class as Phase 2A's
substituted promotion criterion, one level up.

**What the criterion should be:** agreement between the surrogate-force chain
and a reference sampler *on the same frozen target* $\pi_N^\omega$ — long-run
exact-force HMC on identical (ω, N, controls), compared by two-sample test on
the marginals plus a Kolmogorov–Smirnov or energy-distance statistic. That
isolates (1) from (2) and (3). Coverage of true θ then becomes an explanatory
diagnostic about finite-N bias, which is what it actually measures.

### A2. Single ω does not generalise `[GAP]`

With frozen noise, every draw ω defines a *different* target $\pi_N^\omega$.
Results at one ω are one draw from a distribution over pseudo-posteriors. The
program never states whether the claim is about a specific ω or about
$\mathbb{E}_\omega$, and single-ω evidence cannot support the latter. This is
not fixed by more HMC seeds — it needs multiple ω.

### A3. Value and force may not share a particle cloud `[GAP]` — **BLOCKING**

The construction is: exact value at λ_value, damped force at λ_force, "same
frozen noise." But if λ enters the **transport** (reset ridge is the current
candidate), then the same noise driven through two different reset
configurations produces **two different particle clouds**. The force is then
not a perturbed gradient of the value — it is the gradient of a different
functional evaluated on a different cloud.

Corollary 5.2 survives this (any deterministic F is admissible), so the chain
still targets exp(−U) correctly. But three consequences follow that the program
does not state:

- cost is **2× per leapfrog step**, not 1× — every step runs the filter twice.
  The ESS/gradient criterion is measuring a 2× cost arm against a 1× baseline
  without saying so;
- "damping" is a **different transport**, not a small perturbation of one, so
  the bias/mixing tradeoff is not the smooth curve Phase 3.2 assumes;
- as λ_force → λ_value the arms do not continuously converge unless the
  transport does.

Phase 3.1 must resolve this before Phase 3.2's calibration curve means
anything.

## Class B — HMC correctness holes

### B1. No involution / reversibility test `[GAP]` — **BLOCKING**

The single most standard HMC correctness check is absent from every phase:
integrate L steps forward, flip momentum, integrate L steps back, assert return
to the start within tolerance.

$$\Psi_{-p}\circ\Psi_{p}=\mathrm{id}\quad\text{to integrator tolerance}$$

This is far stronger than the determinism test in Phase 2A (T1), which only
checks $f(\theta)$ twice in a row at the same θ. Reversibility fails if the
force depends on **anything** besides θ — trajectory index, call count, a cached
cloud, a reseed, an adaptive branch, accumulated float state. Every one of those
breaks detailed balance while leaving T1, T2 and T4 green.

v2's θ-derived seed would very likely fail it, which is exactly why it must run.

### B2. No volume-preservation check `[GAP]`

Corollary 5.2 asserts the leapfrog map is volume-preserving for any
deterministic F. That holds for a *textbook* kick-drift-kick. It does not hold
if the executed map contains clipping, a trust-region projection inside the
force, a state-dependent step size, or an early exit. Some of those live inside
the LEDH reset. Nothing tests the executed map's Jacobian determinant.

### B3. No exact-Kalman surrogate-force rung `[GAP]` — **BLOCKING**

The validation ladder jumps from a 3-D quadratic toy (Phase 2A) straight to the
full LEDH filter at T=50 (Phase 4). The decisive intermediate rung is missing:

> Run surrogate-force HMC where the **value is the exact Kalman
> log-likelihood** and the force is that same gradient, deliberately corrupted
> by a known factor. The posterior is analytically known.

If this recovers the Kalman posterior, the surrogate-force mechanism is
validated *independently of any filter approximation*. If it does not, the
mechanism is broken and Phase 4 cannot possibly succeed — but Phase 4 would
report the failure as "LEDH score bias too large," misattributing a mechanism
bug to the science.

`tf_svd_kalman_log_likelihood` already exists in `bayesfilter/linear/kalman_svd_tf.py`.
This rung is roughly half a day and it is the highest-value missing test in the
program.

### B4. Seed semantics unresolved — and one reading is fatal `[2C partial]`

Phase 3.3 defers the choice between "frozen per trajectory" and "deterministic
function of θ." These are not two conventions; **one of them is broken**.

If the seed is a hash of θ, then $U(\theta)$ is **discontinuous** on every hash
boundary. A discontinuous potential has no gradient, leapfrog energy error is
unbounded, and acceptance collapses in a way that looks like "damping too
aggressive." No test in the plan distinguishes discontinuity-driven rejection
from bias-driven rejection.

Phase 2C Class 6 specifies a seed-freeze test but cannot write it until the
semantics are chosen — so the test is blocked on Phase 3.3, which sits *after*
2C in the current order. Ordering defect.

### B5. TFP may call the target more than once per step `[GAP]`

The program lists *"verify TFP HMC doesn't re-invoke `log_prob_and_grad` per
leapfrog step"* as a Phase 3 task and it has never been run. Worse than
per-step reseeding: if value and gradient are separately traced and each
regenerates noise, then within a **single** step the value and the force come
from different clouds. Instrument the call count; assert exactly one filter
evaluation per intended evaluation.

### B6. Step-size adaptation can mask a broken chain `[GAP]`

`SimpleStepSizeAdaptation` drives ε toward a target acceptance. If acceptance is
NaN or the force is pathological, ε → 0 produces a chain that accepts
essentially everything, reports acceptance ≈ 1.0, and does not move. Acceptance
looks excellent; ESS is ~1. Nothing vetoes ε below a floor, and nothing vetoes
NaN acceptance.

## Class C — Test-design holes

### C1. Golden-master pins behaviour, not correctness `[2C partial]`

Golden-master tests preserve whatever `single_cloud` + `contract_e` currently
computes. The August 29 audit's verdict on that route is `BLOCKED_FOR_CLAIM`.
So Phase 2B can pass every criterion and faithfully preserve a wrong estimand.
"Behaviour preserved" ≠ "behaviour correct."

Two independent anchors are needed alongside the golden master:

1. FD-vs-JVP at multiple θ (already in Phase 1 — must actually run);
2. **N-convergence**: on an LGSSM where exact Kalman is available,
   $U_N \to -\log p(y_{1:T})$ as N grows, at the expected Monte Carlo rate.
   Nothing currently checks that the value converges to the right limit at all.

### C2. Row-independence tests can miss symmetric segmentation errors `[2C partial]`

The three planned assertions (identical rows → identical; distinct rows →
distinct; row *i* invariant to row *j*) are good, and the third is the strong
one. But a segmentation bug that is **symmetric** across rows — e.g. every row
receiving the *pooled* moments rather than its own — passes assertions 1 and 2,
and passes 3 only if the perturbation test is done per-row rather than as an
aggregate. Specify: perturb row *j*, assert row *i* output is **bitwise**
unchanged, for every ordered pair (i,j) at B=4. Not a norm difference — bitwise.

### C3. The two tolerance regimes are mutually inconsistent `[GAP]` — **BLOCKING**

Phase 2B sets golden-master parity at **rtol 1e-12** and cross-lane parity at
**rtol 5e-4**. These cannot both be right for the same object:

- 1e-12 in float64 essentially demands **bitwise-identical operation order**.
  Replacing a Python unroll with `tf.while_loop`, or flattening `[B,N]`, changes
  reduction order. Over T=50 steps of a chaotic-ish recursion, 1e-12 is very
  likely unachievable — and it was 1e-12 that was offered as the guarantee
  protecting tuning scope.
- Meanwhile 5e-4 is 8 orders looser. A refactor introducing 1e-5 drift fails the
  golden master and passes cross-lane parity. Which verdict governs?

This must be resolved **before** Step 1 fixtures are generated, because it
determines what the fixtures are for. The honest resolution is a derived
tolerance: bound the accumulated round-off from the recursion's condition number
over T steps, and separately establish the band within which the tuned controls
are insensitive. If measured drift sits inside both, tuning scope survives; if
not, retune is owed. "1e-12" as an assertion of convenience is not a derivation.

### C4. Contract tests verify presence, not adequacy `[2C, by design]`

Schema validation can check that `adversary_set` has ≥3 entries. It cannot check
that they are the *right* adversaries. Class 7 catches omission, never
inadequacy. Stated in the 2C plan; restated here because it bounds what a green
suite means.

### C5. Enum ledger is one-directional `[2C gap]`

Class 1 checks *string → is it ledgered*. It does not check *ledger entry → does
a branch implement it*. A ledger listing `"contract_e"` and `"annealed"` where
only the first has a branch passes. Add the reverse assertion.

### C6. Fixture hash pins values, not properties `[2C gap]`

A golden hash over `_lgssm_frozen_observations()` catches B2 (noise substituted
for the fixture). It does **not** catch a fixture that was wrong when the hash
was taken — e.g. if the stationary initialisation
$q/\sqrt{1-\phi^2}$ were mis-transcribed, the hash pins the error permanently.
Add a property test: sample variance of the generated states matches
$q^2/(1-\phi^2)$ within Monte Carlo error.

### C7. "Tests must be seen to fail" is unenforced `[2C gap]`

The 2C plan requires each test be demonstrated failing pre-repair, by git stash.
A manual step nobody can audit later. Make it mechanical: pair each contract
test with a negative fixture (a deliberately defective config in
`tests/contracts/negative/`) and assert the checker rejects it. Then the
negative case is itself a test.

## Class D — Statistical validity

### D1. The 5-parameter coverage criterion is mathematically wrong `[GAP]` — **BLOCKING**

Phase 4 requires *"95% intervals cover true θ (all 5 parameters)."* Read as five
marginal 95% intervals all covering simultaneously, the probability for a
**correct** sampler is at most

$$0.95^5 \approx 0.774$$

and lower under positive dependence. So a perfectly correct implementation fails
this criterion roughly **one run in four**. As written it is a false-negative
trap that would reject a working method and send the program into redesign.

Three admissible repairs — pick one explicitly:

1. **Marginal, with expected failures:** per-parameter 95% coverage, expected
   ~0.25 failures per run, veto only on ≥2 misses;
2. **Joint region:** one 95% joint credible region via Mahalanobis distance
   under the posterior covariance, then "covers" is a single well-defined event;
3. **Bonferroni:** 99% marginals (0.99⁵ ≈ 0.951), preserving "all five" at the
   cost of wider intervals.

(2) is the cleanest statement of the intended claim.

### D2. Acceptance-rate threshold — checked, and it is sound `[OK]`

Worth recording a negative result. At 4 chains × 5000 draws, the Monte Carlo SE
on an acceptance estimate near 0.2 is $\sqrt{0.2\cdot0.8/20000}\approx0.003$;
at Phase 2A's 3 × 500 it is ≈ 0.010. Autocorrelation inflates this somewhat.
The 0.2 threshold is far enough from plausible operating points (0.6–0.8) that
MC error will not flip the verdict. No repair needed.

### D3. ESS ratio at one seed each is not a comparison `[program-acknowledged]`

ESS itself has ~4.5% relative SE at ESS≈1000. A ratio of two such estimates has
~6.4%. The 0.3× threshold is coarse enough to survive, but "Arm 2 achieves
0.31×" versus "0.29×" is noise. The contract already marks this descriptive;
keep the language strictly to *nomination*.

### D4. R-hat variant unspecified `[minor]`

"R-hat < 1.05" without naming rank-normalised split R-hat. The repository's own
NeuTra policy mandates max(rank-normalised split R-hat, folded version). Name it.

## Class E — Execution and environment

### E1. No checkpointing on 12h / 48h GPU runs `[GAP]`

Phase 4 is 12 GPU-hours, Phase 5 is 48. A crash at hour 11 loses everything and
consumes attempt budget. Persist per-chain draws incrementally; support resume.

### E2. Determinism is unverified on the execution device `[GAP]` — **BLOCKING**

Every correctness argument rests on bitwise determinism of the force. Phase 4
runs on GPU. On GPU:

- TF32 has a ~10-bit mantissa;
- reduction order across thread blocks is **not guaranteed** run-to-run;
- non-associativity means the same graph can give different last bits.

If the force is not bitwise reproducible on the actual device, Corollary 5.2's
premise fails *in execution* even though it holds in theory, and the toy tests
(CPU, float64) will never reveal it. A determinism test must run **on the GPU,
in the production dtype**, before Phase 4.

### E3. The dtype/precision story is incoherent across phases `[GAP]`

- production program declares `dtype: float32`, `tf32_enabled: True`;
- tests and fixtures are float64;
- golden master is specified at 1e-12, which is only meaningful in float64;
- Phase 4 executes on GPU where TF32 gives ~1e-3 relative precision.

So the artifact certifying the engine is float64/CPU/1e-12 while the scientific
run is float32/TF32/GPU. Different numerical regimes; the certificate does not
cover the run. Either Phase 4 runs float64, or the golden master must also be
established in the production regime with a defensible tolerance.

### E4. TF32 enablement not wired in the LEDH path `[GAP]`

`LEDH_PRODUCTION_PROGRAM_V1` declares `tf32_enabled: True`, but grep finds
`enable_tensor_float_32` only under `bayesfilter/nonlinear/ssl_lstm_*`. Nothing
in `bayesfilter/highdim/` enables it. Either the declaration is unimplemented
(a wiring defect of exactly the audited class) or it is set externally and
undocumented. A Class 2 wiring test should assert the declared policy is
actually applied.

### E5. Memory-growth policy not in any phase `[GAP]`

The TensorFlow GPU Memory Rule requires memory growth verified on every visible
GPU **before** logical-device init, failing closed, and recorded in the
manifest. No phase does this. `set_memory_growth` helpers exist only under
`bayesfilter/testing/`.

### E6. No CI exists — 2C's gate cannot be wired as written `[GAP]`

Phase 2C says `validate_contracts.py` is "wired into CI." There is no
`.github/workflows`. The gate must be a pre-commit hook or an explicit
pre-execution step, or it will silently not exist.

### E7. Environment ambiguity `[minor]`

Project settings allowlist `envs/tf-gpu/bin/python`; the recorded BayesFilter
environment is `tftwogpu` (with `CUDA_VISIBLE_DEVICES=1` → 4080 SUPER). Phase 0
ran under `tftwogpu`. Pin one and record it in every manifest.

## Class F — Governance

### F1. Self-audit blind spot `[GAP]`

The 2026-09-06 audit, this pre-mortem, and the program are all mine. The same
judgment that produced the 12 defects is assessing them. A1, C3, and D1 are
precisely the class I already got wrong twice (substituted criterion,
unmeasured measurement). One independent Codex review of *this* document before
execution is proportionate.

### F2. No defined recovery state for a mid-refactor abort `[GAP]`

If Phase 2B fails at step 3c, the branch holds a half-ported engine. Define the
abort state: which commit to reset to, and a rule that each 3c stage lands as
one atomic green commit.

---

# PART 2 — Coverage: does Phase 2C catch these?

| # | Failure mode | Severity | Covered? |
|---|---|---|---|
| A1 | Criterion measures wrong object | **BLOCKING** | ❌ no |
| A2 | Single ω doesn't generalise | major | ❌ no |
| A3 | Value/force clouds may differ | **BLOCKING** | ❌ no |
| B1 | No involution test | **BLOCKING** | ❌ no |
| B2 | No volume-preservation check | major | ❌ no |
| B3 | No exact-Kalman rung | **BLOCKING** | ❌ no |
| B4 | Seed semantics fatal reading | major | ⚠️ partial (blocked on 3.3) |
| B5 | TFP multi-call per step | major | ❌ no |
| B6 | Step-size adaptation masks failure | major | ❌ no |
| C1 | Golden master ≠ correctness | major | ⚠️ partial |
| C2 | Symmetric segmentation error | major | ⚠️ partial |
| C3 | Tolerance regimes inconsistent | **BLOCKING** | ❌ no |
| C4 | Presence ≠ adequacy | bounded | ✅ acknowledged |
| C5 | Enum ledger one-directional | minor | ❌ no |
| C6 | Fixture hash pins wrong values | minor | ❌ no |
| C7 | "Seen to fail" unenforced | major | ❌ no |
| D1 | 0.95⁵ coverage criterion | **BLOCKING** | ❌ no |
| D2 | Acceptance MC error | — | ✅ sound as-is |
| D3 | ESS ratio noise | acknowledged | ✅ |
| D4 | R-hat variant | minor | ❌ no |
| E1 | No checkpointing | major | ❌ no |
| E2 | GPU determinism unverified | **BLOCKING** | ❌ no |
| E3 | dtype incoherence | **BLOCKING** | ❌ no |
| E4 | TF32 not wired | major | ⚠️ would catch if asserted |
| E5 | Memory growth absent | major | ❌ no |
| E6 | No CI | major | ❌ no |
| E7 | Environment ambiguity | minor | ❌ no |
| F1 | Self-audit blind spot | major | ❌ no |
| F2 | No abort state | minor | ❌ no |

**Score: 6 of 19 substantive modes covered.** Phase 2C is a good suite for the
defect class it was designed against — wiring and document-code drift. It does
nothing about estimand design, HMC-specific correctness, tolerance coherence, or
device-level determinism, because those are different classes of question.

**Eight blocking items, and seven of them are not tests.** A1, A3, C3, D1, E3
are *decisions* — what to measure, how tight, in which precision. B1, B3, E2
are *missing checks*. Only the second group is buildable; the first must be
settled on paper first, which is why they belong in Phase 3 (now Phase 2D) and
not in a test suite.

---

# PART 3 — Missing steps

Beyond the covered items, in priority order:

1. **Exact-Kalman surrogate-force rung** (B3) — new Phase 2A.5, ~0.5 day. The
   highest-value missing test in the program.
2. **Involution + volume-preservation tests** (B1, B2) — into Phase 2A
   alongside T3, ~0.5 day.
3. **Reference-sampler comparison** replacing coverage as Phase 4's primary
   criterion (A1) — long-run exact-force HMC on identical (ω, N, controls).
   Adds ~6 GPU-hours; makes the result interpretable.
4. **N-convergence check** (C1) — $U_N \to$ exact Kalman as N grows. Half a day,
   and it is the only planned check that the value converges to the *right*
   limit rather than merely reproducibly.
5. **Tolerance derivation** (C3) — before any fixture generation.
6. **Coverage-criterion repair** (D1) — before Phase 4 is written.
7. **GPU determinism + dtype coherence** (E2, E3) — before Phase 4 executes.
8. **Checkpointing** (E1) — before the 12h run.
9. **Memory-growth preflight** (E5) — before any GPU run.
10. **Independent Codex review** of this document (F1).

Revised sequence:

| Phase | Content | Duration |
|---|---|---|
| 2A | Toy mechanics: T3 + involution + volume preservation | 1 day |
| **2A.5** | **Exact-Kalman surrogate-force rung** | **0.5 day** |
| 2B | Engine unification (tolerance derived first) | 6–9 days |
| 2C | Contract-integrity suite + C5/C6/C7 repairs | 2–3 days |
| **2D** | **Decisions: A1, A3, C3, D1, E3 + damping + seed** | **2 days** |
| 3 | Damping calibration curve | 1 day + 4 GPU-h |
| 4 | LGSSM validation vs reference sampler | 1 day + 18 GPU-h |
| 5 | Tier A suite | 2 days + 48 GPU-h |

**Total: 15–20 days + ~70 GPU-hours.** Up from 13–18 and 60.

---

# PART 4 — Allowlist for uninterrupted execution

Current settings already cover much of this (`git -C`, `grep`, `find`, `ls`,
`wc`, `pip install *`, `conda run -n tftwogpu python -c '*'`). Missing entries,
by phase:

## CPU phases (2A, 2A.5, 2B, 2C, 2D)

```
Bash(CUDA_VISIBLE_DEVICES=-1 TF_CPP_MIN_LOG_LEVEL=3 conda run -n tftwogpu python -m pytest tests/*)
Bash(CUDA_VISIBLE_DEVICES=-1 TF_CPP_MIN_LOG_LEVEL=3 conda run -n tftwogpu python -m pytest tests/contracts/*)
Bash(CUDA_VISIBLE_DEVICES=-1 TF_CPP_MIN_LOG_LEVEL=3 conda run -n tftwogpu python scripts/validate_contracts.py*)
Bash(CUDA_VISIBLE_DEVICES=-1 TF_CPP_MIN_LOG_LEVEL=3 conda run -n tftwogpu python tests/highdim/fixtures/generate_golden_master.py*)
Bash(git stash*)
Bash(git stash pop*)
Bash(mkdir -p tests/contracts/negative)
Bash(mkdir -p tests/highdim/fixtures/ledh_golden_master_20260906)
Bash(mkdir -p bayesfilter/contracts)
Bash(mkdir -p docs/schemas)
Bash(mkdir -p docs/derivations)
Bash(sha256sum *)
```

`git stash` is needed for the "seen to fail" demonstrations. If C7 is adopted
(negative fixtures), it becomes optional.

## GPU phases (3, 4, 5) — escalated

```
Bash(CUDA_VISIBLE_DEVICES=1 TF_CPP_MIN_LOG_LEVEL=3 conda run -n tftwogpu python docs/benchmarks/run_ledh_surrogate_*)
Bash(CUDA_VISIBLE_DEVICES=1 TF_CPP_MIN_LOG_LEVEL=3 conda run -n tftwogpu python docs/benchmarks/run_ledh_damping_*)
Bash(CUDA_VISIBLE_DEVICES=1 TF_CPP_MIN_LOG_LEVEL=3 conda run -n tftwogpu python -m pytest tests/highdim/test_ledh_gpu_determinism.py*)
Bash(nvidia-smi*)
Bash(mkdir -p docs/benchmarks/artifacts/ledh_*_20260906)
```

Per the GPU/CUDA policy these must be escalated, and a non-escalated failure is
sandbox evidence only.

## Long-run management

```
Bash(nohup conda run -n tftwogpu python docs/benchmarks/run_ledh_surrogate_* > *.log 2>&1 &)
Bash(tail -* docs/benchmarks/artifacts/*/run.log)
```

Needed so a 12h run is not tied to one foreground call. With checkpointing (E1)
this also gives resume-after-crash.

## Not requested

No network, no `git push`, no changes outside
`bayesfilter/`, `tests/`, `docs/`, `scripts/`.

---

# PART 5 — Upfront approvals for no mid-execution questions

## Campaign authorisation

**A1. CPU campaign (2A → 2D).** Authorise implementation, tests, fixtures,
ledgers, schemas, and localised repair-and-retry under an unchanged scientific
contract, without per-step approval. Failures repaired and recorded; each
attempt consumes budget.

**A2. GPU campaign (3 → 5).** Authorise ~70 GPU-hours and 12 attempts total,
escalated, `CUDA_VISIBLE_DEVICES=1` (4080 SUPER) with 0 as fallback. Every
launch gets a fresh versioned output directory; no prior evidence overwritten.

## Code-change authority

**A3. Fail-closed `reset_policy` guard.** Add validation that raises on an
unrecognised value. Class B under the Safety Guardrail Reversed Burden policy
(adopt-by-default after a no-fire regression). It **will** break any existing
caller passing a non-ledgered string — that is the intent, and it is why this
needs your explicit sign-off rather than my judgment.

**A4. Delete the duplicate engine** at Phase 2B step 3f, after all adapters are
migrated and green, as a separate commit. Recoverable from git.

**A5. Retire `surrogate_force_lgssm_three_arm{,_v2}.py`** — move to
`docs/benchmarks/archive/` with a banner recording B1/B2. Not deleted; blocked
from use.

## Decisions I should not make alone

These change what the program concludes. My recommendation given, but yours to
set:

**A6. Coverage criterion (D1).** Recommend joint 95% Mahalanobis region.
Alternatives: marginal-with-expected-failures, or Bonferroni 99%.

**A7. Primary criterion (A1).** Recommend reference-sampler agreement on the
frozen target, with true-θ coverage demoted to explanatory. This is the single
most consequential change in this document — it alters what Phase 4 proves.

**A8. Tolerance policy (C3).** Recommend deriving from condition number and
tuning-insensitivity band rather than asserting 1e-12. If a derived tolerance
exceeds the tuning-insensitivity band, retune is owed and the program lengthens.

**A9. Precision regime (E3).** Recommend Phase 4 in float64 on GPU, so the
golden-master certificate covers the scientific run, with TF32 deferred to a
separate performance phase. Alternative: certify in the production regime
instead.

**A10. Seed semantics (B4).** Recommend one trajectory-level seed, frozen,
never reseeded — the reading Corollary 5.2 actually requires. θ-hashed seeding
makes U discontinuous and should be rejected, not merely documented.

## Review

**A11. Independent Codex review of this pre-mortem** before Phase 2A resumes.
Advisory, not blocking, per Review Proportionality — but F1 is a real blind
spot and this is the cheap correction.

---

# PART 6 — What I would not proceed without

Ranked. Items 1–4 are cheap and change what the program can conclude:

1. **D1** — the coverage criterion as written fails a correct sampler ~25% of
   the time. A one-line decision.
2. **A1** — without it, a Phase 4 pass or fail is uninterpretable, because three
   questions are confounded in one number.
3. **B3** — the exact-Kalman rung. Half a day, and it separates "mechanism
   broken" from "LEDH bias too large." Without it a Phase 4 failure cannot be
   attributed.
4. **B1** — the involution test. Hours. It is the standard HMC correctness check
   and its absence is conspicuous.
5. **C3** — resolve before fixtures are generated, or Step 1 has to be redone.
6. **E2/E3** — before spending 12 GPU-hours on a run whose determinism premise
   is unverified in the regime it executes in.

---

**END OF PRE-MORTEM**
