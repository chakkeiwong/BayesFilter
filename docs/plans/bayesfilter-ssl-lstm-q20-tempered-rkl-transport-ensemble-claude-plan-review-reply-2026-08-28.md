# Stage 2: Implementation-Plan Review Response

Date: 2026-08-28
Reviewer: Claude Code (Opus 5), read-only bounded review
Target: `docs/plans/bayesfilter-ssl-lstm-q20-tempered-rkl-transport-ensemble-implementation-plan-2026-08-28.md`
Mathematical authority accepted in Stage 1:
`docs/plans/bayesfilter-ssl-lstm-q20-adaptive-replay-neutra-mathematical-note-2026-08-21.tex`
(Stage 1 reply: `...-claude-math-review-reply-2026-08-28.md`, VERDICT: AGREE)

## Scope

I read only the implementation plan and the mathematical note it cites as
authority. I did not edit files, run commands, launch agents, or inspect the
wider repository. Line anchors below are to the implementation plan unless the
anchor says `math note`.

I did not open the sibling MathDevMCP audit or document-alignment audit, so this
reply says nothing about whether their findings were closed.

## Verdict summary

The plan is faithful to the accepted mathematics on every point the handoff
named as a risk. It does not reintroduce particle circularity, it does not
confuse exactness with discovery, it keeps `alpha`, `gamma`, and `p_i`
separated, it rejects the pure-power shortcut, it requires the `beta=0`
diversity ablation, and it routes multi-chart sampling through the canonical
sequential controller and route ledger rather than a standalone chain loop.

Two defects block implementation. Both are additive repairs to existing phases,
not redesigns:

- Phase 0 proves the bridge *identity* but never checks that the bridge defines
  a *proper law* at intermediate `beta`. Theorem 9.1 of the math note needs
  `0 < Z_beta < infinity` at every ladder level, and the plan inherits that
  assumption without a diagnostic.
- The fail-closed invalid-row policy (contract item 4) and blind initialization
  under `g0` (default audit line 149) can deadlock: a blind start may make the
  first update fail closed forever, with no stated screen or recovery.

Six further defects block only the serious campaign, above and beyond the
component count, ladder, ESS/MCSE target, and budget that the plan
*deliberately* leaves open. I do not count those four deliberate gaps against
the plan, per the handoff boundary, except where Phase 8 uses one of them
before it is frozen (finding S3).

VERDICT: REVISE

The revision is small and local. I would expect it to close in one pass.

## What the plan gets right

These are load-bearing, so I record them explicitly rather than leaving the
review to read as only a defect list.

**Bridge properness, as far as identity goes.** Lines 57–66 read the actual
target expression, `log tilde_pi = log L - ||theta - PRIOR_CENTER||^2 / (2*16)`,
and derive `g0 = N(PRIOR_CENTER, 16 I)` with the likelihood-tempered path
`log tilde_pi_beta = log g0 + beta log L`. That is exactly math note eq. (17)
with a proper prior, and it correctly avoids the inadmissible unnormalized
uniform endpoint the math note rejects at math note lines 373–374. Line 66's
instruction not to reconstruct the likelihood "by subtracting two rounded
outputs" is the right engineering call and prevents a real cancellation bug.

**No particle circularity.** Line 400 and Phase 2 line 217 together require the
trainer to record and forbid replay, scalar fallback, row mapping, and
invalid-row filtering. Training stochasticity is IID Gaussian base draws plus
exact target calls. This is the correction the math note's Propositions 4.1 and
4.2 demand, implemented as a recorded receipt rather than an assertion.

**Exactness is not sold as discovery.** Line 45 forbids concluding exhaustive
mode discovery. Line 98 makes replica round trips, hot-level basin forgetting,
cold direct mode transitions, and initialization forgetting *separate*
promotion gates. That matches math note Proposition 9.2 and lines 550–554,
which is where a weaker plan would have leaned on acceptance and swap rate.

**Weight semantics.** Contract item 5 (lines 126–128), default audit line 153,
and especially Phase 6 lines 293–295 — which requires a fixture with unequal
regional mass *and* deliberately unequal component error, checking that fitted
`alpha` follows the biased-weight formula — directly exercise math note eq. (19)
and Corollary 5.1. This is stronger than restating the caveat in prose.

**Controller integration.** Lines 254–256 forbid a standalone chain loop,
require either an exact-transition abstraction inside the shared sequential
controller or a shared-core mode within it, and require the route-ledger update
in the same phase. Contract item 7 (lines 132–134) gives every
`(target, beta, chart, dtype, backend, XLA mode)` scope its own artifact and
forbids runtime retuning and cross-scope reuse. Both match repository policy.

**Backend and execution policy.** Phase 1 line 190 (TensorFlow only), Phase 2
lines 214–217 (fresh stateless Gaussian batch, batch size above one, XLA, no
row mapping or pfor), Phase 7 lines 304–305 (memory growth enabled and verified
before device initialization). Compliant.

**Attribution design.** The ladder at lines 74–82 includes
single-chart-per-temperature replica exchange and cold multi-start multi-chart
HMC. Those two arms are what make the eventual result interpretable: they
separate work done by tempering from work done by having several charts. Many
plans omit exactly these.

## Implementation blockers

These must be addressed before Phase 0 begins.

### I1: Bridge law properness at intermediate beta is not checked

**Lines:** 57–66 (target bridge), 169–181 (Phase 0), contract item 8 line 136,
Phase 5 lines 267–286.

**Issue:** Phase 0 proves that the decomposition `log tilde_pi_beta = log g0 +
beta log L` is exact at `beta=0` and `beta=1` (lines 170–174). Contract item 8
and Phase 5 line 274 require the exact bridge value at arbitrary `beta`.
Theorem 9.1 of the math note (lines 520–534) needs `0 < Z_beta < infinity` at
every ladder level. The plan inherits that assumption without ever checking it.

If the likelihood `L(theta)` is unbounded above on `R^4`, or if `L` and `g0`
have mismatched fast-decay directions, then for some `beta in (0,1)` the
unnormalized bridge density `tilde_pi_beta` may be nonintegrable. This would
make `pi_beta` improper, the swap ratio undefined, and the cold marginal claim
void. The q=20 filtering likelihood is bounded conditional on state but the
state integrates out over a 60-dimensional space. I did not inspect whether the
result is bounded.

A weaker failure mode: if the bridge is proper but intermediate `beta` produces
extreme log densities, training and swapping can hit numerical overflow or
underflow without the plan's current finite/status gates catching it before a
long run.

**Why it blocks implementation:** Without a bridge properness diagnostic,
Phase 0 can pass and Phase 8 can consume its full budget before discovering
that the ladder has an improper or numerically unsuitable intermediate
temperature. The blocker is not hypothetical: power tempering on an improper
uniform endpoint is exactly the inadmissible case the math note lines 373–374
reject.

**Repair:** Add to Phase 0 exit criteria (after line 181):

- For a small grid of `beta in (0,1)`, verify that `tilde_pi_beta(theta)` is
  finite and positive at a held-out Gaussian sample under `g0` and at least one
  known mode location. If any evaluation is nonfinite, the bridge is rejected.
- Compute the dynamic range `max log tilde_pi_beta - min log tilde_pi_beta` on
  that sample. If the range exceeds a declared dtype-dependent threshold (e.g.,
  `600` for float64, `80` for float32), flag a numerical-overflow risk and
  either reject the ladder or require stabilized log-sum-exp in all swap and
  mixture-density calls.

This diagnostic does not prove the bridge is integrable (Proposition 6.2 of the
math note shows that is impossible with finite queries), but it rules out the
most common breakage: nonfinite values at typical draws or known modes.

---

### I2: Blind initialization under g0 can deadlock with fail-closed invalid-row policy

**Lines:** contract item 4 (lines 123–125), default audit line 149 (blind
initialization), Phase 3 line 235 (only from reference law and stateless rules).

**Issue:** Contract item 4 forbids silently dropping, replacing, or resampling
an invalid target row within an update: "The update fails closed and triggers
an initialization, target, or numerical repair." Default audit line 149 says
initial physical locations come only from the reference law `g0` without
target-mode oracle leakage. Phase 3 line 235 says "Initial physical locations
and scales for the blind candidate come only from the reference law and
declared stateless rules."

If `g0` and the likelihood have mismatched support or if typical `g0` draws
land in a numerically hostile region (state-space singularity, extreme
likelihood curvature, filter divergence), then the first training batch under
the blind start returns invalid status for every row. Contract item 4 fails
closed. The plan says this "triggers an initialization, target, or numerical
repair" but does not say *what* repair, and Phase 3 forbids non-blind
initialization for the claim-bearing candidate. The blind candidate can
therefore deadlock: it cannot take its first update and the rules forbid the
oracle start that would escape.

The handoff boundary line 88 forbids demanding posterior samples as an input to
training, so a defensive-support mixture `m = epsilon * q_candidate +
(1-epsilon) * tail_guard` is not available before `q_candidate` exists.

**Why it blocks implementation:** A deterministic rule (blind init under `g0`,
fail-closed on invalid rows, no oracle escape for the main arm) can produce a
state from which no forward progress is possible. That is a design bug, not a
tuning or scientific failure.

**Repair option A (preferred):** Amend contract item 4 to allow a capped retry
with a fresh stateless Gaussian batch when the *entire* batch is invalid on the
*first* update of a newly initialized component. Record the retry and the
fraction of invalid batches; if it exceeds a small cap (e.g., 3 retries), fail
closed and trigger the target or bridge repair. After the first successful
update, any invalid row fails closed as currently specified. This lets blind
discovery escape an unlucky initial batch without allowing silent filtering of
rare target failures during training.

**Repair option B:** Before Phase 3, add a Phase 2.5 that draws a pilot sample
under `g0`, evaluates bridge targets at a grid of `beta`, and records the
fraction of invalid or nonfinite returns. If the invalid fraction at any `beta`
exceeds a threshold (e.g., `0.5`), the bridge or target is rejected and either
`g0` is replaced by a reviewed defensive alternative or the target is repaired.
This is a target-validity screen, not initialization tuning.

Either repair closes the deadlock. Option A is less expensive and treats the
problem where it occurs (first update). Option B is safer if the target can be
structurally invalid on a large `g0` mass.

---

## Serious-campaign blockers

These do not prevent Phase 0–7 implementation and mechanics tests. They prevent
a long GPU campaign whose results would be used for a research conclusion. The
plan already blocks the serious campaign (lines 29–32, 412–415) pending
component count, ladder, ESS/MCSE, and budget. I do not repeat those four
deliberate gaps. The findings below are defects the plan did not recognize.

### S1: No alpha/gamma semantics test in Phase 6

**Lines:** 288–301 (Phase 6).

**Issue:** Phase 6 lines 293–295 require an analytic fixture with unequal
regional mass and unequal component error, testing that fitted `alpha` follows
the biased-weight formula (math note eq. 19). That is correct. But the phase
never tests the *other* weight: fixed state-independent `gamma`, which controls
chart-kernel selection frequencies.

Math note Proposition 8.2 (lines 455–463) and Counterexample 8.1 (lines
471–477) prove that `gamma` must be state-independent for invariance.
Contract item 5 (lines 126–128) says `gamma` is a fixed state-independent
frequency affecting efficiency but not the invariant posterior. Contract item 6
(lines 129–131) refers to "that chart" (singular) per physical kernel call,
but the plan never verifies that runtime chart selection respects the fixed
`gamma` or that it rejects a state-dependent proposal.

**Why it blocks a serious campaign:** If Phase 4 or the final controller
accidentally implements state-dependent chart selection (e.g., `gamma_i =
f(theta)`), the resulting kernel does not preserve `pi_beta` and Theorem 9.1
is void. A mechanics-only smoke (Phase 7) would not catch this because it does
not check posterior moment agreement. Phase 6 is where exact analytic-target
moment fixtures live, and it should include one.

**Repair:** Add to Phase 6 exit criteria (after line 301):

- On a two-component separated Gaussian-mixture target with known unequal
  regional masses, verify that fixed uniform `gamma = [0.5, 0.5]` and fixed
  nonuniform `gamma = [0.2, 0.8]` both preserve the exact marginal moments
  within finite-sample uncertainty, and that a deliberately state-dependent
  `gamma(theta)` produces biased moments and is rejected by the controller.

---

### S2: Cross-density quadratic cost is measured but never used to gate joint refinement

**Lines:** 23–26 (decision summary), Phase 1 line 199, default audit line 151.

**Issue:** The plan correctly identifies that joint mixture refinement has
`O(K^2 B)` cost (line 199, line 26, line 151). Phase 1 line 199 says
"cross-density tensor shapes are fixed and `O(K^2 B)` cost is measured." But
the plan never says what measurement would make joint refinement *infeasible*,
and Phase 8 line 320 lists "K candidates" without bounding them.

If `K=10` and `B=1024`, then `K^2 B ~ 100K` cross-density evaluations per
update. For a nonlinear `O(1000)`-call filtering target like q=20, that is
`~100M` target calls per update, which is absurd. But the plan would discover
this only during Phase 8, after the independent-training route has already
consumed budget.

**Why it blocks a serious campaign:** Without a declared `K^2 B` threshold,
Phase 8 could select a `K` that makes joint refinement GPU-OOM or
thousand-hour, discover this mid-campaign, and be forced to drop the optional
joint arm without ever testing it. The dropped arm is optional, so this does
not invalidate independent training, but it wastes the implementation of Phase 2
joint training and Phase 1's cross-density primitives.

**Repair:** In the Phase 8 subplan entry criteria (before line 316), require:

- Declare the maximum tolerable `K^2 B` cost for joint refinement (e.g., `K^2 B
  <= 50000` or `joint update wall time <= 10 * independent update wall time`).
- If any `K` candidate would exceed this threshold, label joint refinement as
  infeasible for that `K` and run only the independent-training route.

This makes the optional arm's scope explicit before the campaign.

---

### S3: Phase 8 uses "K candidates" before K is frozen

**Lines:** 315–331 (Phase 8), default audit line 146.

**Issue:** The plan correctly identifies `K` (component count) as an unproven
hypothesis with no numeric default (line 146) and correctly says the serious
campaign is blocked until `K` is frozen (lines 29–32, 412–415). But Phase 8
line 320 says the subplan must include "K candidates" (plural). If Phase 8
*is* the serious campaign, then it cannot also be the phase that explores `K`.

This is either a naming confusion (Phase 8 is not the serious campaign, only
the subplan for it) or a scope leak (Phase 8 tests `K` and also runs the full
confirmation ladder on untouched data).

**Why it blocks a serious campaign:** If Phase 8 explores `K` on the
confirmation partition, then there is no untouched data left for the terminal
validation. If Phase 8 explores `K` on a pilot partition and then runs the
confirmation ladder on a separate partition, that is fine, but the phase
description does not say so.

**Repair:** Clarify Phase 8 scope (lines 315–331):

- Either split Phase 8 into Phase 8a (pilot `K` and ladder search on a cheap
  surrogate or small-data calibration partition) and Phase 8b (full
  confirmation ladder on the untouched partition with frozen `K`), or
- Explicitly state that "K candidates" (line 320) means a small pilot set
  (e.g., `K in {2, 4, 8}`) selected by heuristic or prior work, tested on a
  disjoint calibration partition, and that the confirmation run uses exactly
  one selected `K`.

The current text reads as if Phase 8 is both the search and the confirmation,
which violates the calibration/validation/confirmation partition mentioned at
line 322.

---

### S4: No continuation-veto screen for transport inverse/Jacobian reliability

**Lines:** continuation veto line 43, Phase 1 lines 183–203, Phase 4 lines
246–264.

**Issue:** The continuation veto (line 43) includes "the transport cannot
provide a reliable inverse and log determinant" but the plan has no screen that
would fire this veto. Phase 1 tests density/inverse/Jacobian on analytic
Gaussian components and a small diagnostic fixture (lines 191–199). Phase 4
tests "physical/latent round-trip" and "transformed value/score parity" on
analytic targets (lines 257–260). Neither phase checks whether a *learned*
transport's inverse is numerically unstable, non-bijective in practice, or
produces a Jacobian with extreme conditioning.

For a neural IAF, the inverse is computed by a forward pass of an internal
network. If that network has poor conditioning or if the learned map has
near-singular Jacobian regions, then `theta = T(T_inv(theta))` can have
large error or `log |det DT_inv|` can be inaccurate or nonfinite. The analytic
fixtures cannot catch this because analytic maps are numerically exact.

**Why it blocks a serious campaign:** If a learned transport is numerically
unreliable but Phase 4 passed on analytic targets, the controller will use it
in Phase 9, and the sampler will produce biased or invalid results. The plan
would not discover this until terminal posterior checks fail, long after Phase 8
training. The continuation veto says transport unreliability should stop the
direction, but there is no diagnostic to trigger it.

**Repair:** Add to Phase 4 exit criteria (after line 264):

- After Phase 3 produces at least one trained transport, add a numerical
  reliability check: draw a batch of physical `theta` under the mixture `q_alpha`,
  compute `theta_recovered = T(T_inv(theta))`, and verify `||theta_recovered -
  theta|| / ||theta||` is below a dtype-dependent threshold (e.g., `1e-6` for
  float64). If any component fails, reject that component and trigger a repair
  (architecture, regularization, or training adjustment). If all components
  fail, fire the continuation veto.
- Similarly, verify that `log |det DT(z)|` and `log |det DT_inv(theta)|` sum to
  near zero (within `1e-6` absolute) on a held-out batch. This is the adjugate
  identity and catches Jacobian implementation bugs.

These are parity checks, not proofs, but they catch the most common neural-flow
numerical failures before they contaminate a long posterior run.

---

### S5: No "direct cold mode transition" definition or measurement protocol

**Lines:** research intent line 41, evidence roles line 98, Phase 9 line 342.

**Issue:** The promotion criterion (line 41) and tempering-specific gates (line
98) require "direct cold mode transitions" and "initialization forgetting."
Phase 9 line 342 says "freeze mode-transition and replica-travel requirements"
before launch. But the plan never defines what counts as a mode transition or
how it will be measured.

For the q=20 target, the user and prior plans have mentioned a known four-mode
structure (sign flips). Does a "mode transition" mean a cold chain visits more
than one of those four regions in a single run? Does it mean a Markov-chain
label-switching diagnostic? Does it mean a bimodal marginal histogram? The
mathematical note's Proposition 9.2 says invariance does not imply mixing, so
this gate is essential, but the plan gives no protocol.

**Why it blocks a serious campaign:** Phase 9 line 342 requires these
requirements to be frozen before launch, but Phase 8 or earlier never derives
them. The campaign cannot begin because the gate is undefined. If the gate is
defined during Phase 9, after the campaign has started, then it is not
predeclared and cannot be used as a promotion criterion.

**Repair:** Add to Phase 8 entry criteria (before line 316) or as a separate
Phase 7.5:

- Define the mode-transition and initialization-forgetting protocol for the q=20
  target. If the four-sign-mode structure is accepted knowledge, define a mode
  as a connected region in parameter space and require a cold chain to visit at
  least two regions separated by a likelihood barrier under the prior. Measure
  this with a label-switching diagnostic or a bimodal-marginal test on a
  predeclared parameter. If the mode structure is not known, define "transition"
  as a chain that traverses from initial state to a state far from it (e.g.,
  `||theta_final - theta_initial|| > 3 * prior_std`) and then returns within
  that distance, indicating global exploration rather than local drift.
- For "initialization forgetting," define a permutation-test or ANOVA-like
  diagnostic: run multiple chains from deliberately distinct starts, and reject
  the sampler if the terminal means remain significantly different beyond MCSE.

Without this, Phase 9 cannot proceed.

---

### S6: Physical replica-exchange baseline uses the same bridge but the plan does not verify its proper-reference implementation

**Lines:** comparator line 77, Phase 0 lines 169–181, Phase 5 lines 267–286,
skeptical audit line 402.

**Issue:** Comparator line 77 says "Physical-coordinate replica exchange:
Classical tempering baseline under the identical proper bridge." The skeptical
audit line 402 confirms "Improper power-tempering endpoint repaired by Phase
0's exact proper Gaussian-prior bridge and endpoint parity."

Phase 5 implements the proper-reference replica exchange for the multi-chart
candidate (lines 267–286). But the plan never says whether the physical
baseline *also* uses the proper bridge or whether it is allowed to use the
old pure-power route. If the physical baseline uses pure power and the candidate
uses the proper bridge, the comparison is unfair: they do not operate on the
same temperature family. If the physical baseline is updated to use the proper
bridge, that is a change to an existing production route and needs a separate
validation (it is not merely a "classical baseline" anymore, it is a corrected
baseline).

Current implementation findings line 110 says the diagnostic
`distributed_replica_exchange_tf.py` uses "pure power target" and is not
production inference authority. Correct. But the comparator ladder (line 77)
needs a production physical replica-exchange baseline, and the plan does not
say where that comes from or whether it is already proper-bridge compliant.

**Why it blocks a serious campaign:** If the physical baseline is run with the
old pure-power route, the comparison violates the plan's own tempering repair
(line 402). If it is run with the new proper bridge, that is new code for a
production route, and the plan must test it separately (not just as a
multi-chart dependency) and update its route ledger.

**Repair:** Add to Phase 5 exit criteria (after line 286):

- Verify that the physical-coordinate replica-exchange baseline uses the same
  proper-reference bridge as the multi-chart candidate, with the same endpoint
  parity, bridge identity, and swap ratio. If the existing production
  physical-coordinate route does not support parameterized bridges, implement
  and test it in Phase 5 as a separate arm. Update the route ledger if a new
  production physical-coordinate proper-bridge route is created.

This ensures the comparator ladder is internally consistent.

---

## Non-blocker findings: clarifications and minor corrections

These do not prevent implementation or a serious campaign, but they improve
clarity, reduce misinterpretation risk, or close minor gaps.

### N1: "Oracle upper-bound arm" is mentioned but never specified (Phase 3 line 239)

Phase 3 lines 237–239 say "Known q=20 sign modes may be used only in an
explicitly labeled oracle upper-bound arm." This is correct scoping (the
oracle cannot contaminate the blind candidate). But the plan never says
whether the oracle arm will be run, what its purpose is, or how its results
will be interpreted. If it is run, it should appear in the comparator ladder
(lines 73–82) so its role is explicit. If it is optional and may be skipped,
the plan should say so.

**Suggested repair:** Either add the oracle arm to the comparator table (line
82a: "Oracle mode-informed initialization: Upper bound for mode-discovery
probability"), or delete the mention at line 239 if the oracle arm is not part
of this plan.

---

### N2: "Reject" ambiguity in contract item 7 (line 134)

Contract item 7 lines 132–134 forbids runtime retuning and cross-scope
artifact reuse. The sentence "No artifact may be reused across beta or
transport identity" could mean (a) a single artifact may not be used for
two different (beta, chart) pairs, or (b) the same (beta, chart) pair run
twice must generate a fresh artifact each time. Interpretation (a) is correct
per repository policy. Interpretation (b) would forbid checkpoint recovery,
which is not intended.

**Suggested repair:** Clarify line 134: "A tuning artifact is valid only for
the exact (target, beta, chart, dtype, backend, XLA mode) scope it was
generated for. Different scopes require separate artifacts. Checkpoint recovery
of a prior tuning run for the same scope is allowed."

---

### N3: Phase 2 joint training is "optional" but its failure mode is not scoped (lines 223–224)

Phase 2 lines 223–224 say "Joint training is a separate optional exit and may
be rejected without stopping the main route." This is correct. But the default
audit line 151 lists joint refinement's failure modes as "K^2 B memory/time,
component collapse, unequal-error weight bias" without saying which of these
would cause rejection versus which are merely recorded findings.

For example: if joint training hits GPU-OOM due to `K^2 B`, that is an
infeasibility rejection and correctly drops the optional arm. But if joint
training runs successfully and the fitted `alpha` follows the biased formula
(which is the *correct* behavior per math note eq. 19), that is not a
rejection, it is confirmation of the mathematics. The plan should distinguish
implementation failure from expected behavior.

**Suggested repair:** Amend default audit line 151 or Phase 2 exit (after line
224): "Joint training is rejected if it exceeds memory/time budgets or produces
numerically invalid results (NaN, infinite loss, component collapse to
identical parameters). Fitted alpha that follows the biased-weight formula
(eq. 19 of the math note) is expected and correct, not a rejection."

---

### N4: No mention of what "basin forgetting" means or how to measure it (line 98, line 342)

Evidence roles line 98 and Phase 9 line 342 require "hot-level basin
forgetting" as a tempering-specific gate. The intuition is clear (a hot chain
should not be mode-locked), but the plan gives no measurement protocol. This
is similar to finding S5 but less severe because "basin forgetting" could be
inferred from autocorrelation or effective-sample-size diagnostics. Still, an
explicit definition would help.

**Suggested repair:** Add to Phase 9 line 342 or Phase 8 entry: "Hot-level
basin forgetting is measured by the hot replica's ESS or autocorrelation in a
predeclared parameter. If the hot chain's ESS per step is below a threshold
(e.g., `ESS/N < 0.01`), the hot level is too cold and the ladder is rejected."

---

### N5: Swap accounting could be clearer about cached-denominator reuse (line 86)

Lines 84–86 say target calls, score calls, swaps, compile time, and wall time
are "reported separately" and "A compute-matched comparison must cap total
target evaluations, not merely optimizer updates or retained draws."

Each adjacent swap needs four bridge densities: `tilde_pi_beta_ell(theta_ell)`,
`tilde_pi_beta_ell(theta_ell+1)`, `tilde_pi_beta_ell+1(theta_ell)`,
`tilde_pi_beta_ell+1(theta_ell+1)`. If the denominator pair (current states)
is cached from the most recent within-temperature transition, only the
numerator pair (two new evaluations) is needed. But the plan does not say
whether cached reuse is allowed, and if it is, whether the comparison counts
two target calls per swap or four.

This is not a blocker if all arms use the same rule, but it should be explicit.

**Suggested repair:** Clarify line 86: "Swap accounting: each adjacent swap
attempt consumes two new bridge target evaluations (the numerator states in
eq. 24 of the math note) when denominator states are reused from the most
recent within-temperature transition. Cached-denominator reuse is allowed
provided exact value/score/status are preserved and swap-ratio identity is
verified in Phase 5 fixtures. All arms use the same swap-accounting rule."

---

## Reviewer boundaries and what I did not check

Per the handoff:

- I did not demand posterior samples as an input to the reverse-KL training
  objective (handoff line 75–76). The plan correctly uses IID Gaussian base draws.
- I did not accept an arithmetic average of maps as a mixture distribution
  (handoff line 78). The plan has Counterexample 8.1 coverage in Phase 1 line 197.
- I did not accept the current diagnostic pure-power replica-exchange module as
  the proper-reference implementation (handoff line 79–81). Finding S6 flags this.
- I did not accept loss, whitening, acceptance, or swap rate as posterior
  convergence or mode-mass evidence (handoff line 82–83). Lines 95, 400 correctly
  classify them as explanatory only.
- I did not reject the overall direction merely because an optional joint-mixture
  arm fails (handoff line 84–85). Finding S2 and N3 scope the optional arm correctly.
- I treated missing component count, ladder, ESS/MCSE target, and serious compute
  budget as deliberate serious-campaign blockers, not routine implementation
  defects (handoff line 86–88). I did not count them against the plan except
  where Phase 8 uses one before it is frozen (finding S3).

I did not check:

- Whether MathDevMCP obligations were closed. The handoff line 73–74 says
  "eleven obligations were not checkable and the full rigor reporter hit a tool
  error," so I did not assume zero findings certify the proofs. The math review
  was my independent check, not a delegation to MathDevMCP.
- Whether a document-alignment audit exists or passed. The handoff mentions one
  (lines 8–9 of the plan itself) but does not name it as required reading.
- Whether the repository policy documents (`hmc-tuning-interface.md`,
  `neutra_hmc_policy.py` route ledger, etc.) have been updated to reflect the
  new multi-chart route. The plan correctly requires those updates in Phase 4
  and Phase 5; I did not verify they are done.
- Whether TF32, XLA, memory growth, or GPU policies are correctly implemented in
  the current code. The plan requires them; I took that as the contract.

---

## Summary of findings by severity

**Implementation blockers (must fix before Phase 0):**

- I1: Bridge law properness at intermediate beta is not checked.
- I2: Blind initialization under g0 can deadlock with fail-closed invalid-row policy.

**Serious-campaign blockers (must fix before Phase 8 or Phase 9):**

- S1: No alpha/gamma semantics test in Phase 6.
- S2: Cross-density quadratic cost is measured but never used to gate joint refinement.
- S3: Phase 8 uses "K candidates" before K is frozen (scope ambiguity).
- S4: No continuation-veto screen for transport inverse/Jacobian reliability.
- S5: No "direct cold mode transition" definition or measurement protocol.
- S6: Physical replica-exchange baseline uses the same bridge but the plan does
  not verify its proper-reference implementation.

**Non-blocker clarifications:**

- N1: "Oracle upper-bound arm" is mentioned but never specified.
- N2: "Reject" ambiguity in contract item 7 (reuse vs recovery).
- N3: Phase 2 joint training failure mode is not scoped.
- N4: No mention of what "basin forgetting" means or how to measure it.
- N5: Swap accounting could be clearer about cached-denominator reuse.

---

## Faithfulness to the accepted mathematics

The plan is faithful. Specific checks:

- **No particle circularity:** Training uses IID Gaussian draws, not
  self-generated particles. Lines 23, 217, 400.
- **Proper bridge required:** Lines 57–66, 170–174, Phase 0. Finding I1 adds
  the missing intermediate-beta check.
- **Maps not averaged:** Phase 1 line 197, contract item 2 line 118.
- **Weights distinguished:** Contract item 5, Phase 6 lines 293–295, default
  audit line 153. Finding S1 adds the `gamma` fixture.
- **Beta=0 diversity:** Default audit line 148, Phase 3 lines 228–232,
  skeptical audit line 401.
- **Exactness not discovery:** Lines 45, 98, contract item 9 line 138, Phase 9
  lines 341–342. Findings S4 and S5 strengthen the diagnostics.
- **Frozen charts:** Contract item 6, Phase 4 lines 246–256, Phase 8 line 328.
- **Fixed gamma:** Contract item 5 line 127, default audit line 152. Finding S1
  adds the fixture.
- **Adjacent swaps with exact ratio:** Contract item 8, Phase 5 lines 273–274,
  fixtures line 278. Finding S6 ensures the baseline uses the same bridge.
- **Cold marginal only:** Contract item 9, Phase 5 line 281.
- **Canonical controller integration:** Lines 254–256, contract item 7.
- **No pfor:** Default audit line 155, Phase 2 line 217.

The plan does not reintroduce the rejected replay foundation, does not confuse
exactness with discovery, keeps the three weight types distinct, rejects the
improper power endpoint, requires the beta=0 diversity ablation, and routes
sampling through the canonical controller and route ledger.

---

## VERDICT: REVISE

Repair findings I1 and I2 before implementation. Repair findings S1–S6 before
the serious campaign (they can be fixed during Phase 0–7 implementation). The
non-blocker findings N1–N5 improve clarity but are not required for AGREE.

The revision is local and additive. I expect it to close in one pass.

After repair:

- Implementation (Phase 0–7 mechanics and fixtures) can proceed.
- Serious campaign (Phase 8–10) remains deliberately blocked pending component
  count, ladder, ESS/MCSE targets, and budget, per lines 29–32 and 412–415.
  That block is intentional and correct.

The plan is well-constructed, thorough, and faithful to the accepted
mathematics. The findings are real gaps, not rejections of the approach.
