# Handoff to Codex — Audit request: UKF discontinuous-gradient HMC program

Date: 2026-09-02
From: Claude (author of both objects under audit)
To: Codex
Type: **Adversarial audit request.** Not an implementation task. Not a repair task.

---

## 0. Why you are being asked, and what failure looks like

I wrote both artifacts under audit: survey section §7 (`sec:ukf-gradient`) and the
research plan `ukf-discontinuous-gradient-hmc-research-plan-2026-08-26.md`. I have
since found what look like material defects in **both**, including in my own
mathematics. I am not asking you to confirm my findings. I am asking you to
independently adjudicate them, and to find the ones I missed.

The known failure mode for this audit is **agreeable skimming**: reading the plan
instead of the source, restating the survey's prose as if it were evidence for the
survey, marking derivations "correct" without writing them, and returning a
summary that is fluent and unfalsifiable. That output is worse than no audit,
because it will be cited as clearance. If you can only do part of the work, do
part of it properly and mark the rest `not checked`. A short honest audit beats a
long confident one.

Highest priority by a wide margin: **Part A, the mathematics.** If you run out of
budget, run out of it in Part D, not Part A.

---

## 1. Deliverable

Write exactly one file:

```
docs/plans/ukf-discontinuous-gradient-hmc-codex-audit-reply-2026-09-02.md
```

Do not edit the survey. Do not edit the plan. Do not write code. Do not create
other files. This is an audit; repair is a separate, later task under separate
authority.

### Required structure of your reply

1. **Verdict table** — one row per question M1–M25, columns:
   `ID | verdict | anchor (file:line or eq label) | one-line basis`.
   All 25 rows present. No blanks.
2. **Per-question sections** M1–M25, each answered on its own, in order.
3. **Findings I missed** — defects you found that are not covered by M1–M25.
4. **Completeness self-audit** — see §6.

### Verdict vocabulary (from `CLAUDE.md` "Plain Scientific Language And Non-Evasion")

Use exactly one of these per question. No other words:

| verdict | meaning |
|---|---|
| `correct` | follows from a derivation you wrote out, or a source section you read |
| `wrong relative to the stated target` | the claimed object differs from the computed/asserted object |
| `unsupported` | no inspected derivation, citation, or artifact supports it |
| `not checked` | you did not inspect enough evidence to decide |
| `heuristic only` | may be useful; no correctness claim established |

---

## 2. Rules of engagement (binding)

**R1. Every answer carries an anchor.** A `file:line` range, an `eq:` label, or a
pasted command output. An answer without an anchor is not an answer.

**R2. The survey is the object under audit, not evidence for itself.** You may not
support a verdict on a survey claim by quoting adjacent survey prose. Support it
with a derivation you wrote, the source code, or a cited paper you read.

**R3. `correct` on a mathematical question requires the derivation in your reply.**
Symbols defined, steps shown. If you did not write it, the verdict is
`not checked`.

**R4. `not checked` is an acceptable and expected answer.** State what evidence you
would need. Bluffing is the only disallowed response.

**R5. No batching.** "M4–M6: all fine" is a non-answer. One verdict per ID.

**R6. Banned words in verdict positions:** *reasonable, sensible, approximately
correct, largely correct, broadly right, seems fine, no major issues, looks good,
minor.* These are the vocabulary of the failure mode. Say `correct`, say
`wrong relative to the stated target`, or say `not checked`.

**R7. Read the `.tex` and the `.py`, not the plan.** The plan is a downstream
artifact under audit. Answering a math question from the plan's restatement is a
sourcing error.

**R8. Do not propose new work until the audit is complete.** Recommendations go in
one short closing section, after all 25 answers.

**R9. Do not defer to me.** Where I have pre-stated a competing claim below, I may
be wrong. Several of these are my own errors and I have flagged them as suspicions,
not conclusions. Adjudicate independently and say so if I am wrong about being
wrong.

---

## 3. Objects under audit

| # | Artifact | Scope |
|---|---|---|
| 1 | `docs/surveys/zlb_discontinuous_hmc/zlb_discontinuous_hmc_survey.tex` | **§7 lines 1048–1419** primarily; cross-check §2 `sec:geometries` lines 186–240, §6 `sec:kink-validity` lines 979–1046, §13 case study line 2012 ff., multiplicity derivation line 2782 ff. |
| 2 | `docs/plans/ukf-discontinuous-gradient-hmc-research-plan-2026-08-26.md` | whole file |
| 3 | `docs/plans/hardbound-kink-hmc-master-program-2026-08-21.md` | governing authority — read §1 objective, non-goals, pre-approved decisions |
| 4 | `docs/plans/hardbound-g2-3-leapfrog-ladder-result-2026-09-01.md` | most recent executed evidence |
| 5 | `bayesfilter/hardbound/` | `model_tf.py`, `dns_curve_tf.py`, `joint_target_tf.py`, `hmc_runner.py` |

### Ground truth I verified before writing this memo

Stated so you do not waste budget rediscovering it, and so you cannot claim it was
unavailable. **Verify anything you rely on** — I may have mis-read.

- `bayesfilter/hardbound/` implements the two-country shadow-rate DNS model with
  two targets: `TARGET_S1 = "mf_s1_k40_softplus"` and
  `TARGET_C1 = "mf_c1_k40_hardmax"` (`model_tf.py:16-18`).
- The softplus map is `s(u) = ℓ + a·softplus((u−ℓ)/a)` with `a = alpha`
  (`dns_curve_tf.py:45-49`). Fixture: `alpha_d = 1.5e-3`, `alpha_f = 1.0e-3`
  (`model_tf.py:33-34`).
- **There is no UKF in `bayesfilter/hardbound/`.** The executed lane is joint
  non-centred HMC over latent states (`x0_raw`, `eta_raw`), not a filter marginal
  likelihood.
- The executed G2.3 runs used `TARGET_C1` — the **hard max**, not the softplus.
- The master program declares as **binding non-goals**: filter-free operation
  ("If a marginal likelihood is ever needed, that is a new program") and
  "No event-aware / reflection-refraction HMC".
- `docs/.localresources/` **does not exist**. No papers are stored locally.
- "Tran and Kleppe 2025" appears at survey lines 1270 and 1343 and **nowhere in
  the bibliography**.
- Owner positions, not up for re-litigation: particle filters are off the table for
  production (cost; LEDH score still biased); dense mass matrix does not scale to
  the T=120, d=100 target (≈1.44×10⁸ entries).

### Suggested commands

```bash
sed -n '1048,1419p' docs/surveys/zlb_discontinuous_hmc/zlb_discontinuous_hmc_survey.tex
sed -n '979,1046p'  docs/surveys/zlb_discontinuous_hmc/zlb_discontinuous_hmc_survey.tex
sed -n '186,240p'   docs/surveys/zlb_discontinuous_hmc/zlb_discontinuous_hmc_survey.tex
sed -n '2782,2844p' docs/surveys/zlb_discontinuous_hmc/zlb_discontinuous_hmc_survey.tex
grep -n -i 'kleppe' docs/surveys/zlb_discontinuous_hmc/zlb_discontinuous_hmc_survey.tex
sed -n '1,120p' bayesfilter/hardbound/dns_curve_tf.py
grep -rn 'ukf\|UKF' bayesfilter/hardbound/
```

---

## 4. MUST-ANSWER QUESTIONS

### Part A — Mathematics (highest priority)

Where I state a "competing claim", it is my suspicion about my own text. Adjudicate
it; do not adopt it.

---

**M1. Is the central claim of §7 mathematically false?**

§7.3 (lines 1120–1128) asserts, for the softplus/UKF composition:

  lim_{θ→θb⁻} ∇log p(y|θ) ≠ lim_{θ→θb⁺} ∇log p(y|θ)

Competing claim: `softplus` is C^∞, and if the UKF recursion is built from smooth
primitives (matrix products, Cholesky of a positive-definite matrix, log-det), then
θ ↦ log p(y|θ) is C^∞ and **both limits are equal**, making the asserted inequality
false and the "continuous-kink" classification of §7 wrong for the softplus case.

Required:
- Differentiate the code's map `s(u) = ℓ + a·softplus((u−ℓ)/a)`. Give `s'(u)` and
  `s''(u)` in closed form. State `sup_u s''(u)` and its scaling in `a`.
- State whether any step of the UKF recursion introduces a non-smooth operation in
  θ (min/max, sort, branch, `tf.where`, pivoted or fallback Cholesky, clipping,
  an iteration count that depends on θ). Search for them; name any you find.
- Give the verdict on §7.3's displayed inequality.
- If the map is smooth, state the *correct* characterization of the difficulty and
  which survey geometry case it belongs to.

**M2. Does §7 contradict itself?**

§7.5 Step 5 (lines 1226–1235) estimates a **finite** Lipschitz constant `L_est`
growing linearly in α. A function with a genuine gradient jump has `L_est → ∞` as
δ → 0. Are §7.3 and §7.5 Step 5 simultaneously satisfiable? If not, which one
survives, and what must the other say?

**M3. Is `eq:56a` invoked outside its proven scope?**

`eq:56a` (lines 1028–1033) gives `ΔH = O(ε‖∇U⁺ − ∇U⁻‖)`. Lines 1000–1003 restrict
the enclosing proposition to a **fixed piecewise-regular partition of ℝ^d** and
explicitly exclude "a potential evaluated through an implicit multi-root
equilibrium solver, a state-dependent iteration count, or any branch rule that is
not a fixed partition". §7 invokes `eq:56a` at lines 1134–1135 and 1268 for the
UKF/softplus composition.

Required:
- Is the UKF/softplus composition inside or outside the stated scope? Justify.
- If the map is smooth (M1), then `‖∇U⁺ − ∇U⁻‖ = 0` and the bound is vacuous.
  Is §7's use of `eq:56a` therefore meaningless as written?
- If the operative problem is **stiffness** rather than a kink, state the correct
  leapfrog statement: the local error order for a smooth potential, and the
  linear stability threshold on ε in terms of the largest Hessian eigenvalue.
  Relate that eigenvalue to `a` (equivalently α) using your M1 result.
- Which prescription follows — event detection, or step size / preconditioning?
  These are different research programs. Say which one the mathematics selects.

**M4. Is the variance-collapse mechanism stated backwards?**

§7.1 lines 1073–1085 argues: data pin `i_t` near ℓ → likelihood is uninformative
about shadow depth → "The filtering distribution **thus** develops a one-sided
collapse: variance in the direction that affects the observed rate shrinks to near
zero".

Competing claim: the inference marked "thus" runs the wrong way. With observation
map `h(x)` through the softplus, the observation Jacobian carries the factor
`s'(·) = σ((u−ℓ)/a) ∈ (0,1)`. Deep below the bound `s' → 0`, so `H → 0`, so the
Kalman gain `K = P Hᵀ(HPHᵀ + R)⁻¹ → 0`, so `P_{t|t} = (I − KH)P_{t|t−1} →
P_{t|t−1}` — the variance does **not** collapse there; it is the *uninformative*
regime and variance is preserved or grows under the transition. Collapse is a
property of the *informative* regime (`s' → 1`, above the bound).

Required:
- Write `H_t`, `S_t`, `K_t`, `P_{t|t}` explicitly for this observation map.
- Take both limits (`s' → 0` and `s' → 1`) and state what happens to `P_{t|t}`.
- Adjudicate the sentence at lines 1078–1085. Is "thus" a valid inference?
- If the mechanism is misstated, does anything downstream in §7 survive?

**M5. Is the amplification mechanism bounded away from the claimed pathology?**

§7.2 (lines 1095–1113) attributes large parameter gradients to variance collapse.
But `S_t = H P Hᵀ + R ⪰ R ≻ 0`, so `‖S_t⁻¹‖ ≤ ‖R⁻¹‖` **regardless of how far P
collapses**. Fixture measurement noise is `noise_scale_truth = 5e-4`
(`model_tf.py:37`).

Required:
- Bound `∂/∂θ` of the innovation quadratic form and the log-det term, in terms of
  `R`, `H`, `P`.
- Is the operative small quantity the filtering variance `P`, or the measurement
  noise `R`? These imply different remedies.
- Does the §7.2 mechanism survive the `S ⪰ R` bound as stated?

**M6. Does the sigma-point argument hold?**

§7.2 says collapse causes sigma points to "change regime contribution
simultaneously". UKF sigma points sit at `m ± √((n+λ)P)`. As `P → 0` all sigma
points converge to the mean `m`, so the unscented approximation converges to the
**linearization at `m`** — arguably *more* regular, not less.

Required: state what `∂ŷ/∂θ` actually does as `P → 0`, and whether the stated
mechanism holds, fails, or holds for a different reason than the one given.

**M7. Is the §7.4 value-jump claim true in general?**

Lines 1168–1172 claim that when an integer regime solver switches paths, the
one-sided likelihoods "generally do not match".

Competing claim: at an OccBin/guess-and-verify switch boundary the constraint binds
with **equality**, so the solution path is typically continuous in θ with a kink,
not a jump. Genuine value jumps require multiplicity or non-existence.

Required:
- Adjudicate the blanket claim.
- Read the survey's own derivation at line 2782 ("A fully derived multiplicity
  example and the discontinuity mechanism"). Does it attribute discontinuity to
  regime switching *per se*, or to multiplicity/selection? Cite equations.
- Is §7.4 consistent with that derivation, or does it overclaim?

**M8. Is the particle-filter smoothing claim correct?**

§7.6 lines 1244–1247 displays
`p(y_t|y_{1:t-1},θ) ≈ Σ_i W^i_{t-1} p(y_t|X^i_{t-1},θ)`.

Required:
- (a) Is `p(y_t | X^i_{t-1}, θ)` the bootstrap incremental weight? Write the
  correct bootstrap expression. Is the displayed density available in closed form
  for this model?
- (b) Standard multinomial resampling makes the PF likelihood estimate
  **discontinuous** in θ (ancestor indices jump). Does the unqualified claim that
  PFs "smooth the gradient" hold? What conditions are required — common random
  numbers, differentiable/OT resampling? Name them.
- (c) Is the claim consistent with the repository's own position that the LEDH
  score remains biased? If §7.6 is right, why is the LEDH score biased?

**M9. Is the geometry cross-reference correct?**

§7.8 lines 1332–1335 says value-jump is "**Case 3**" and continuous-kink is
"**Case 1**" of `sec:geometries`. Read lines 186–240 and state the correct case
numbers for (i) censored max with ΔU = 0, (ii) deterministic branch map with ΔU ≠ 0
per `eq:5b`, (iii) genuine mixed support, (iv) multiplicity. Is the §7.8
cross-reference right?

**M10. Is there an α convention collision between survey and code?**

Survey `eq:ukf1` (lines 1069–1072) writes `ℓ + (1/α)log(1 + e^{α(i*−ℓ)})` — α is a
**temperature**, large = sharp. Code writes `ℓ + a·softplus((u−ℓ)/a)` with
`a = 1.5e-3` — a **scale**, small = sharp.

Required:
- Are these the same map under `α = 1/a`? Show it.
- What is the survey's α for the fixture?
- Do "gradient jumps of size O(α)" (line 1110) and "L_est grows approximately
  linearly with the temperature α" (line 1234) read correctly under **both**
  conventions, or does one invert?
- Is "jump of size O(α)" the same claim as "curvature of size O(α)"? If not, which
  one does the mathematics support?

---

### Part B — Document and sources

**M11. The Tran and Kleppe citation.** Cited at lines 1270 and 1343; absent from
the bibliography. §7.9 line 1343 asserts their integrator is "mathematically valid
for continuous-kink targets".
- Confirm the dangling citation.
- Identify the actual paper (search; I believe there is real Tran–Kleppe work on
  numerical generalized randomized HMC for piecewise-smooth target densities —
  verify the authors, year, venue, and title rather than trusting me).
- Read enough of the method to answer: does it address **ΔU = 0 kinks**, **ΔU ≠ 0
  jumps**, or both? Is the line 1343 characterization supported?
- `CLAUDE.md` requires a local copy of any paper materially affecting decisions.
  `docs/.localresources/` does not exist. State the compliance gap. (Do not create
  it in this audit; report it.)

**M12. Uninspected sources.** Line 1042 states of Pakman and Paninski (2014) "we
were unable to obtain the full text". Does §7, or the plan, rest on any claim that
requires that text? The plan cites it as reference #4.

**M13. Internal contradictions.** List every contradiction between §7 and §2
(`sec:geometries`), §6 (`sec:kink-validity`), §13 (case study), and the DPF section.
Line anchors for both sides of each contradiction.

**M14. Does §7's premise refer to anything that exists?** Lines 1060–1062 claim this
is "the geometry encountered in the two-country shadow-rate model of
Section~\ref{sec:shadow} when a UKF replaces the particle filter". Does
`\ref{sec:shadow}` resolve to a real label? Does a UKF variant of that model exist
anywhere in the survey or the repository? If not, is §7 describing a measured
phenomenon or a hypothesized one — and does the text say which?

---

### Part C — The research plan

**M15. Conflict with the governing authority.** The plan never cites
`hardbound-kink-hmc-master-program-2026-08-21.md`, which is marked "approved for
autonomous execution" and declares binding non-goals including filter-free
operation and "No event-aware / reflection-refraction HMC". Enumerate every
conflict between the plan and that authority. State whether the plan is executable
under current governance or requires a new program.

**M16. Is the plan's central factual premise true?** Plan Phase 1.2 asserts G2_3 is
a UKF-filtered test case with "2 states near ZLB". Verify against
`bayesfilter/hardbound/model_tf.py`. Is there any UKF in that package? Which target
did the executed G2.3 runs use? Is the premise correct?

**M17. Is the plan's problem premise supported by executed evidence?**
`hardbound-g2-3-leapfrog-ladder-result-2026-09-01.md` locates the mixing bottleneck
at θ8 (log FX noise scale), non-monotone in L, with a **funnel** explanation
offered — not a kink-crossing explanation. The plan's entire premise is that
kink/gradient geometry is the binding obstacle.
- Is the plan's premise supported, contradicted, or untested by the executed
  evidence?
- State the competing explanation precisely.
- What is the smallest diagnostic that discriminates funnel geometry from
  kink-crossing geometry? Be concrete.

**M18. Governance violations in the plan.** Phase 4.3 names
`phase4_surrogate_trained_model.pt` — a PyTorch checkpoint — against the `CLAUDE.md`
Backend Rule (TF/TFP; PyTorch requires reviewed exception). Audit the plan against:
Backend Rule; NeuTra Batch-Native Training Rule (Phase 4 trains a network);
TensorFlow GPU Memory Rule; required run manifest; required decision table;
required inference-status table; required pre-mortem. List every violation and
every missing mandatory artifact.

**M19. Budget arithmetic.** Phase 4.2 budgets 200 GPU-hours for 20,000 evaluations
≈ 36 s each. Phase 4 totals 800 of 1,000 GPU-hours. Audit against measured
runtimes in the ladder result and the G2.3 gate run. Is the budget defensible, and
is the 80% concentration in the least-validated phase a defensible allocation?

**M20. Plan/survey method mismatch.** Survey §7.7 item 3 (lines 1276–1281)
recommends a surrogate for **proposals with true-target Metropolis correction** —
posterior remains exact, a bad surrogate costs acceptance only. Plan Phase 4.4 runs
HMC **on the surrogate target** and then measures posterior bias — posterior is not
exact. Are these the same method? Which is correct? Does the plan silently
substitute a biased method for an exact one?

**M21. Promotion criteria.** The plan sets `ESS/grad > 0.1`. The ladder screen
reports min ESS/grad ≈ 4.5×10⁻⁴ at the selected rung — roughly two orders of
magnitude below. Is the plan's criterion attainable, correctly defined, and
measured the same way as the ladder's? If not, every promotion gate in the plan is
mis-set.

**M22. Target scale.** Phase 5 uses a "NAWM II proxy" and the plan targets
T=120, d=100. Does NAWM II exist in this repository? Does any code support that
scale? Is the plan's stated target model reachable, or aspirational?

---

### Part D — State of execution

**M23. What has actually been run?** Produce a factual ledger separating: (i) work
executed under the master program (Program A, hard-max kink target, filter-free);
(ii) work executed for the UKF program (I believe: only survey §7 was written);
(iii) work the plan assumes exists but does not. Cite artifacts.

**M24. Is §7 correctly placed?** Given §2 already classifies the geometries and §6
already proves kink validity and states `eq:56a`, does §7 add a result, duplicate
existing material, or contradict it? Should §7 be revised, split, narrowed, or
retracted pending measurement? Give one recommendation.

**M25. What is the smallest discriminating experiment?** If a UKF-filtered variant
of this model were built, what is the *smallest* diagnostic establishing whether a
gradient-geometry obstacle exists **at all** — as opposed to stiffness, funnel
geometry, or ordinary tuning failure? State the measurement, the discriminating
prediction of each hypothesis, and whether it is authorized under current
governance or needs a new program.

---

## 5. What I am not asking

- Not asking you to fix anything. Report only.
- Not asking whether particle filters should return. Owner has decided; out of
  scope.
- Not asking for a rewrite of §7. If it needs one, say so in one line under M24.
- Not asking for agreement. If M1–M10 are all `correct` as written and I am wrong
  about my own errors, say that plainly and show why.

---

## 6. Completeness self-audit (required closing section)

End your reply with exactly this, filled in:

```
Questions answered with anchor + verdict: __ / 25
Verdicts by class: correct __ | wrong __ | unsupported __ | not checked __ | heuristic only __
Derivations written out in full (M1, M3, M4, M5, M6 minimum): __ / 5
Files opened: <list>
Commands run: <list>
Papers read (not just located): <list>
Questions I could not answer and why: <list with IDs>
Claims in this reply I could not anchor: <list, or "none">
```

If `not checked` exceeds 8 of 25, say so in your first paragraph rather than
burying it — a partial audit is useful, a partial audit presented as complete is
not.

---

## 7. Closing note

Two of the objects under audit were written by me in sequence, the second built on
the first without re-verifying it, and neither was checked against the executed
evidence in this repository or against the governing master program. That is the
specific mechanism I want caught. The mathematics in Part A is where a wrong answer
propagates furthest: if M1 resolves against §7.3, then §7's classification, the
survey's prescription, and the plan's entire five-phase structure are aimed at a
geometry that does not exist in the softplus model — and the real obstacle,
whatever it is, remains unmeasured.
