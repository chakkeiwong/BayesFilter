# LEDH Surrogate-Force HMC — Gap Verification Audit

**Date:** 2026-09-06  
**Task:** Verify the 19-item pre-mortem gap list is correct and complete before
building the repair plan.

**Method:** Independent re-audit from first principles, structured by what could
go wrong. Compare against the pre-mortem list, mark additions and corrections.

---

## Audit 1: Is the mathematical claim well-formed?

### G1. What is the theorem? `[PRE-MORTEM: missing]` — **BLOCKING**

The program cites "Corollary 5.2 (variance note lines 954-976)" four times. I
have never read that document in this conversation. The claim being tested is:

> "Biased force → correct sampling, if force is deterministic and value exact."

But I have not verified:
- whether that corollary exists,
- whether it says what the program claims,
- what its premises are,
- whether we satisfy them,
- what "biased force" means formally (∇U + ε? ∇Ũ where Ũ ≠ U?),
- whether it applies to the specific construction (damped ridge in reset).

**This is a show-stopper.** The entire program rests on a theorem I have not
inspected. If the citation is wrong, or the premises don't hold, or the
construction doesn't match the theorem's form, the science is invalid
independent of any implementation quality.

**Action required:** Read the variance note, extract the exact statement of
Corollary 5.2, verify its premises, map them onto the LEDH surrogate-force
construction, and document the binding. If any premise fails, the program stops.

**Pre-mortem status:** Not listed. This is a **new blocking item**.

---

### G2. Arm 1 (baseline) is underspecified `[A3 partial, new detail]`

The program says "long-run exact-force HMC" but never defines:
- Fixed-L leapfrog or NUTS? (Grep finds `num_leapfrog_steps` in pilot code,
  suggesting fixed-L, but the program doesn't say.)
- What "long-run" means quantitatively — 10× the draws? 100×?
- Whether Arm 1 uses the **same** ω and N as Arm 2, or independent draws. If
  independent, the comparison is between two **different targets** and agreement
  is not expected even if both samplers are perfect.

Pre-mortem A3 identified that value and force may use different clouds (a cost
and interpretability issue). This is the dual: do Arm 1 and Arm 2 target the
same $\pi_N^\omega$?

**Pre-mortem status:** Partially covered by A3. The "same ω, N?" question is new.

---

### G3. What does "damped" mean? `[A3, expanded]`

Pre-mortem A3 says "if λ enters the transport, the clouds differ." The program
never states **which parameter is λ**. Candidates:

- `reset_ridge` (current suspicion)
- `correction_lm_damping`
- `correction_lm_scale_floor`
- Process covariance Q(θ) or observation covariance R(θ) directly

Phase 3.1's task is to resolve this, but the program can be *executed* in the
current ordering without that resolution — Phase 2B comes first. If 2B
hard-codes a choice (say, `reset_ridge`), and 3.1 later determines that was
wrong, 2B is invalidated.

**Pre-mortem status:** Covered as A3, but the sequencing hazard is new.

---

## Audit 2: Are the promotion/veto criteria sound?

### G4. Coverage criterion is mathematically wrong `[D1]` — **BLOCKING**

Pre-mortem D1 identified this. Verified: $0.95^5 = 0.774$, a correct sampler
fails ~1 in 4. **Confirmed.**

---

### G5. The criterion measures the wrong object `[A1]` — **BLOCKING**

Pre-mortem A1: coverage of true θ confounds sampler correctness, finite-N bias,
and data properties. Verified: this is correct. **Confirmed.**

The recommended fix is reference-sampler agreement on the frozen target. But G2
asks: is the reference sampler specified enough to implement? Currently no.

---

### G6. ESS criterion is one-seed descriptive `[D3, acknowledged]`

Pre-mortem D3: ESS at one seed each is descriptive only. Program already
acknowledges this. **Confirmed as acknowledged.**

---

### G7. No divergence veto `[new]` — **major**

Standard HMC diagnostics (Betancourt et al): divergences are a **hard veto** for
sampler correctness, not a soft metric. The program lists "monitor divergences"
as an artifact field but never states a veto threshold.

If Arm 2 produces any divergences, the result is "sampler broken" regardless of
coverage or ESS. The Phase 4 decision table has no row for this.

**Pre-mortem status:** Not listed. **New major item.**

---

### G8. No E-BFMI check `[new]` — **major**

Energy Bayesian fraction of missing information (E-BFMI) < 0.2 is a standard
indicator of poor exploration or geometric pathology. It is the second-most
standard HMC diagnostic after divergences. The program never mentions it.

**Pre-mortem status:** Not listed. **New major item.**

---

### G9. Bulk vs tail ESS not distinguished `[new]` — **minor**

The program says "ESS" without specifying bulk or tail. Tail ESS can be much
lower. For a filtering posterior where tail behaviour matters (outlier data,
rare regimes), tail ESS is the relevant one.

**Pre-mortem status:** Not listed. New minor item (easily fixed by naming it).

---

## Audit 3: Are the test rungs complete?

### G10. No exact-Kalman surrogate-force rung `[B3]` — **BLOCKING**

Pre-mortem B3. Verified: the ladder jumps from 3-D quadratic (Phase 2A) to full
LEDH T=50 (Phase 4). The exact-Kalman rung with deliberately corrupted gradient
is the decisive middle test. **Confirmed.**

---

### G11. No involution test `[B1]` — **BLOCKING**

Pre-mortem B1. Standard HMC correctness. **Confirmed.**

---

### G12. No volume-preservation check `[B2]` — **major**

Pre-mortem B2. Corollary 5.2 (whatever it says) asserts volume-preservation for
deterministic F, but only if the executed map is actually the textbook
kick-drift-kick. **Confirmed.**

---

### G13. No multi-ω test `[A2]` — **major**

Pre-mortem A2: single ω is one draw from a distribution over pseudo-posteriors.
**Confirmed.**

---

### G14. No N-convergence check `[C1]` — **major**

Pre-mortem C1: nothing checks that $U_N$ converges to the **right** limit (exact
Kalman log-likelihood) as N grows. **Confirmed.**

---

### G15. Toy test uses wrong target signature `[new detail on Phase 2A]`

Phase 2A's toy potential is a standalone Python function. But Phase 4's LEDH
target is a `tf.function` with `input_signature`. The `tf.custom_gradient`
wrapper for surrogate-force wiring is **graph-mode only** — it doesn't work in
eager Python. So Phase 2A's T3 acceptance test, even if we write it correctly,
tests a calling convention that Phase 4 never uses.

**Pre-mortem status:** T3's gap was listed under Phase 2A incomplete. The
graph-vs-eager mismatch is a **new detail** that makes T3 less valuable than
stated — it's a correctness check of a wiring path that isn't the production one.

---

## Audit 4: Are tolerances and precision consistent?

### G16. Tolerance regimes inconsistent `[C3]` — **BLOCKING**

Pre-mortem C3: golden master 1e-12, cross-lane parity 5e-4. Eight orders apart.
**Confirmed.**

---

### G17. dtype incoherence `[E3]` — **BLOCKING**

Pre-mortem E3: certificate is float64/CPU, run is float32/TF32/GPU. **Confirmed.**

---

### G18. GPU determinism unverified `[E2]` — **BLOCKING**

Pre-mortem E2: TF32 + non-guaranteed reduction order. **Confirmed.**

---

### G19. Golden master can preserve wrong behaviour `[C1 partial]`

Pre-mortem C1: golden master pins behaviour, not correctness. The August 29
audit verdict on `single_cloud` + `contract_e` is `BLOCKED_FOR_CLAIM`.
**Confirmed.**

---

## Audit 5: Is the seed/noise story coherent?

### G20. Seed semantics unresolved, one reading fatal `[B4]` — **BLOCKING**

Pre-mortem B4: if seed is hash(θ), U is discontinuous. **Confirmed.**

---

### G21. TFP may call target multiple times per step `[B5]` — **major**

Pre-mortem B5: if value and gradient are separately traced and each regenerates
noise, they come from different clouds within one step. **Confirmed.**

---

### G22. Seed-freeze test is blocked on 3.3, but 2C runs first `[B4, sequencing]`

Pre-mortem B4 noted the Phase 2C seed-freeze test is blocked on the Phase 3.3
seed-semantics decision. Current program order is 2C before 3. So 2C cannot
write that test. **Confirmed as a sequencing defect.**

---

## Audit 6: HMC-specific correctness

G7, G8 (divergences, E-BFMI) already covered above.

### G23. Step-size adaptation can mask failure `[B6]` — **major**

Pre-mortem B6: if force is pathological, ε → 0 produces acceptance ≈ 1 and
ESS ≈ 1, looks excellent but doesn't move. **Confirmed.**

---

### G24. No check that force is actually used `[new]` — **major**

What if the `tf.custom_gradient` decorator is silently ignored (wrong TF
version, wrong trace mode, wrong function signature) and TFP falls back to
autodiff of the value? The chain works, converges, has good diagnostics — and
tests nothing about surrogate-force, because it never used the supplied force.

Catch: instrument `target_log_prob_fn` and the custom gradient, assert the
custom branch was called, count invocations, check they match expected.

**Pre-mortem status:** Not listed. **New major item.**

---

## Audit 7: Test-design soundness

### G25. Row-independence can miss symmetric errors `[C2]` — **major**

Pre-mortem C2: a segmentation bug that pools moments passes aggregate
perturbation tests. Needs per-row bitwise check. **Confirmed.**

---

### G26. Enum ledger is one-directional `[C5]` — **minor**

Pre-mortem C5: ledger → code branch not checked. **Confirmed.**

---

### G27. Fixture hash pins values not properties `[C6]` — **minor**

Pre-mortem C6: a mis-transcribed stationary variance stays pinned. **Confirmed.**

---

### G28. "Seen to fail" is unenforced `[C7]` — **major**

Pre-mortem C7: manual demonstration, not mechanised. **Confirmed.**

---

### G29. Contract tests verify presence not adequacy `[C4, by design]`

Pre-mortem C4: acknowledged limitation of schema-based validation. **Confirmed.**

---

## Audit 8: Execution environment

### G30. No checkpointing `[E1]` — **major**

Pre-mortem E1: 12h/48h runs, crash loses everything. **Confirmed.**

---

### G31. Memory growth not in any phase `[E5]` — **major**

Pre-mortem E5: TensorFlow GPU Memory Rule. **Confirmed.**

---

### G32. TF32 enablement not wired `[E4]` — **major**

Pre-mortem E4: `LEDH_PRODUCTION_PROGRAM_V1` declares it, grep finds no
implementation. **Confirmed.**

---

### G33. No CI, gate cannot be wired `[E6]` — **major**

Pre-mortem E6. **Confirmed.**

---

### G34. Environment ambiguity `[E7]` — **minor**

Pre-mortem E7: `tf-gpu` vs `tftwogpu`. **Confirmed.**

---

## Audit 9: Are Phase 1 diagnostics ever run?

### G35. Phase 1 five diagnostics deferred and never scheduled `[new]` — **major**

Phase 1 (route identity baselines) was marked INCOMPLETE with five diagnostics
deferred to "Phase 2B Step 5." I read Phase 2B Step 5 (golden-master parity
tests). It never mentions:

- FD-vs-JVP residual < 1e-6
- Sinkhorn marginal TV < 1e-3
- Contract-E moment residual
- Cholesky condition number
- dual-cap convergence counts

So Phase 1's diagnostics are not merely deferred — they are **not planned
anywhere**. The Phase 1 result document I wrote on 2026-09-04 claimed them
"deferred to 2B Step 5" without checking that 2B Step 5 would do them.

**Pre-mortem status:** Phase 1's incompleteness was noted, but the fact that the
deferred diagnostics have no home was not. **New major item.**

---

## Audit 10: Governance and review

### G36. Self-audit blind spot `[F1]` — **major**

Pre-mortem F1. **Confirmed.**

---

### G37. No abort state for mid-refactor failure `[F2]` — **minor**

Pre-mortem F2. **Confirmed.**

---

## Audit 11: Are the allowlist and approval lists sufficient?

### G38. Long runs need nohup/background, not in allowlist `[new]` — **minor**

Pre-mortem Part 4 lists `nohup ... &` as needed for 12h runs. Current allowlist
does not have it. **Confirmed as incomplete allowlist.**

---

### G39. Git stash not in allowlist `[listed in Part 4]`

Pre-mortem Part 4 says `git stash` needed for "seen to fail" demos. Not
currently allowed. **Confirmed in pre-mortem Part 4.**

---

### G40. Escalation for GPU commands not explicit `[policy statement, not gap]`

Pre-mortem Part 4 says GPU commands "must be escalated." This is a reminder of
the policy, not a gap — the user escalates, not the allowlist.

---

## Consolidation: New items found

| # | Item | Severity | Pre-mortem |
|---|---|---|---|
| G1 | Corollary 5.2 never read, premises unverified | **BLOCKING** | ❌ new |
| G2 | Arm 1 baseline underspecified (same ω,N?) | major | ⚠️ A3 partial |
| G3 | "Damped" parameter unresolved, 2B precedes 3.1 | major | ⚠️ A3, sequencing new |
| G7 | No divergence veto | major | ❌ new |
| G8 | No E-BFMI check | major | ❌ new |
| G9 | Bulk vs tail ESS | minor | ❌ new |
| G15 | Toy test wrong calling convention | minor | ⚠️ detail on T3 |
| G22 | Seed test blocked on 3.3, but 2C first | major | ⚠️ noted, sequencing new |
| G24 | No check force actually used | major | ❌ new |
| G35 | Phase 1 diagnostics have no home | major | ❌ new |
| G38 | nohup not in allowlist | minor | ⚠️ listed Part 4 |
| G39 | git stash not in allowlist | minor | ⚠️ listed Part 4 |

**Summary:**
- Pre-mortem: 19 items, 8 blocking
- This audit: **adds 6 new substantive items** (G1, G7, G8, G24, G35, plus
  sequencing/detail on 4 more)
- **New blocking: G1** (the foundational theorem unverified)
- **Revised total: 25 items, 9 blocking**

The pre-mortem was 68% complete. The biggest miss: **I never read the theorem
the entire program depends on.**

---

## What this means for "correct and complete"

The gap list is now more correct, but I cannot certify it **complete** without:

1. Reading Corollary 5.2 and checking every premise against the construction
2. Reading the Betancourt HMC diagnostics paper to confirm the standard checklist
3. Reading TFP's `HamiltonianMonteCarlo` and `tf.custom_gradient` docs to verify
   calling conventions and trace-mode assumptions
4. Checking whether any LEDH-specific property (T steps of recursion, Sinkhorn
   non-differentiability, dual-cap non-smoothness) creates a failure mode I
   haven't listed

**Your instinct is exactly right.** We cannot proceed until:
1. The gap list is verified complete (requires reading the source documents)
2. A plan exists that fixes **all** gaps, not a priority subset
3. That plan is audited for internal consistency before execution

---

**END OF VERIFICATION AUDIT**
