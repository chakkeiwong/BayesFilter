# SSL-LSTM q=20 Adaptive-Replay NeuTra Mathematics: Fable Review Reply

Date: 2026-08-23

Status: `READ_ONLY_MATHEMATICAL_REVIEW_COMPLETE`

Auditor: Fable (claude-fable-5), independent mathematical reviewer

Audited artifact (primary):
`docs/plans/bayesfilter-ssl-lstm-q20-adaptive-replay-neutra-mathematical-note-2026-08-21.md`

Context artifacts read in full:
`docs/plans/bayesfilter-ssl-lstm-q20-adaptive-replay-neutra-fable-handoff-2026-08-21.md`,
`docs/plans/bayesfilter-ssl-lstm-q20-adaptive-replay-neutra-mathdevmcp-audit-2026-08-21.md`,
`docs/plans/bayesfilter-ssl-lstm-q20-adaptive-replay-neutra-mathematics-review-plan-2026-08-21.md`.

Review basis: read-only inspection at commit
`68ba5271989fe35740416dff599bb61c83dfa099` with pre-existing unrelated
modified HMC-tuning files preserved and untouched. Every theorem,
proposition, lemma, corollary, and counterexample in the primary note was
rederived by hand; the cited NeuTra and Del Moral--Doucet line regions and
both implementation anchors were re-inspected directly. No file was edited,
no training or HMC command was run, no GPU process was launched, and no
MathDevMCP status was accepted as proof. The only write is this reply.

Summary: all five requested verdicts are AGREE. Every theorem in the note is
correct under its stated assumptions. The findings below are, in order:
one claim-boundary sharpening ((30) provably fails for the implemented dense
IAF on any admissible set containing two symmetry-equivalent stationary
points, so Theorems 2/2A are at best basin-local for that family), two
proof-presentation gaps whose missing derivations are supplied here, and a
set of assumption-wording repairs. None invalidates a theorem, changes an
estimator, or changes the research direction. All empirical nonclaims are
preserved.

---

## Findings ordered by severity

### F1. Assumption (30) is not merely unverified for the implemented dense IAF: it provably fails on any admissible set containing two symmetry-equivalent stationary points

Note anchors: Theorem 2 assumption 7, lines 489-497; scope paragraph, lines
614-620; Theorem 2A inheritance of the same condition, lines 641-643.

Classification of the note's own labeling ("a strong-stability hypothesis,
not a property established for the finite dense IAF objective"): `correct`.
Classification of (30) itself for the implemented family on any `C`
containing two distinct symmetry-equivalent stationary points:
`wrong relative to the stated target` (it cannot hold there). On a
symmetric-copy-free local basin it remains `unsupported` (neither proved nor
refuted).

Derivation. The implemented conditioner is the masked MLP of
`bayesfilter/inference/neutra_weighted_training.py`. Two exact parameter
symmetries leave the transport function, hence `q_phi`, hence
`J_{a,b}(phi) = a Z F(phi) + b R(phi)`, invariant:

```text
(i) Same-degree hidden-unit permutation, any activation.
    _dense_masks (neutra_weighted_training.py:190-216) assigns hidden degree
    1 + index % 3 for d = 4, so a width-32 layer holds 10-11 units per
    degree. Swapping two same-degree units swaps two columns of the incoming
    kernel, two bias entries, and two rows of the outgoing kernel. The
    corresponding mask columns/rows are identical, so every masked affine
    map, hence T_phi, is unchanged: T_{S phi} = T_phi for the induced linear
    involution S.

(ii) Tanh sign flip. The active q=20 runner sets activation="tanh"
     (run_ssl_lstm_q20_neutra_global_mixing_training_2026_08_19.py:365).
     Negating one hidden unit's incoming column and bias negates its
     preactivation; tanh(-x) = -tanh(x) negates its activation; negating its
     outgoing row restores every downstream preactivation. Again
     T_{S phi} = T_phi.

Consequence. J(S phi) = J(phi) for all phi, so
grad J(S phi) = S grad J(phi). Let phi_star be any stationary point with
S phi_star != phi_star (generic: any two same-degree units carry different
parameters). Then grad J(S phi_star) = 0. If both copies lie in C, testing
(30) at phi = S phi_star gives

  0 = <S phi_star - phi_star, grad J(S phi_star)>
    >= mu ||S phi_star - phi_star||^2 > 0,

a contradiction. Hence (30) cannot hold on any admissible C that contains
two distinct symmetry-equivalent stationary points.
```

Impact: changes no theorem and no estimator; (30) is an assumption and the
note never claims it for the implemented family. It sharpens the claim
boundary from "unverified" to "provably unattainable except on
symmetric-copy-free local sets": for the implemented architecture, Theorems
2 and 2A can only ever be applied as local-basin statements, and any future
campaign document that cites them must say so.

Weaker sufficient version (not claimed for the current dense IAF): drop
(30) entirely and keep the remaining assumptions; the standard
Kushner--Clark / Robbins--Monro set-convergence argument then yields
almost-sure convergence of `phi_t` to the chain-recurrent set of the
projected mean ODE (a subset of the stationary set of `J_{a,b}` on `C`),
not to a point. That version is compatible with parameter symmetries but
proves correspondingly less.

### F2. Theorem 2 assumption 4 asserts, without derivation, that Lemma 1 makes the Poisson solution uniformly Lipschitz in phi; the claim is true but the obvious proof attempt fails, so the two-line argument should be recorded

Note anchors: assumption 4, lines 480-482; series (32), lines 516-519;
Poisson equation (33), lines 522-524.

Classification of the claim (bounded, uniformly Lipschitz `u_phi`):
`correct` (verified below). Classification of its support in the note as
written: `unsupported` (asserted via "Lemma 1 then makes", with no
derivation; the naive term-by-term bound does not sum).

Verification. Write `pi = P_B^K`, `h_phi = G_F(phi,.) - Z grad F(phi)`, and
let `tau` be the coupling time of Lemma 1, with
`P(tau > j) <= K (1-eps)^j` by (25). The coupling is built from refresh
coins and replacement draws only, so it does not depend on `phi`.

```text
Boundedness:
  ||P^j G_F(phi,B) - pi(G_F(phi,.))||
    <= 2 sup||H|| P(tau > j) <= 2 sup||H|| K (1-eps)^j,
  so ||u_phi(B)|| <= 2 K sup||H|| / eps.

Lipschitz continuity: the naive route bounds each series term by
2 L_H ||phi - phi'|| (L_H the Lipschitz constant of H in phi), which does
not decay in j and cannot be summed. Apply the coupling a second time, to
the phi-difference. With (B_j) started at B and (B_j^st) the coupled
stationary copy,

  [P^j G_F(phi,B) - pi G_F(phi)] - [P^j G_F(phi',B) - pi G_F(phi')]
    = E[ (G_F(phi,B_j) - G_F(phi',B_j))
         - (G_F(phi,B_j^st) - G_F(phi',B_j^st)) ; tau > j ],

because the integrand vanishes identically on {B_j = B_j^st}. Each
difference is bounded by L_H ||phi - phi'||, so the j-th term is at most
2 L_H ||phi - phi'|| K (1-eps)^j, and summing gives

  ||u_phi(B) - u_phi'(B)|| <= (2 K L_H / eps) ||phi - phi'||.
```

Impact: proof-presentation repair only. No assumption changes; the theorem
stands. Recommended edit: add the displayed four lines (or a citation to a
uniform-ergodicity Poisson-regularity lemma) after (32).

### F3. Assumption A4 licenses the forward-side interchange but its wording and example do not cover the base-side interchange used to prove (9)

Note anchors: Assumption set A item 4, lines 72-74; Proposition 2 proof step
"differentiate under the base expectation", line 162; equation (9), lines
141-148.

Classification of (9) as a formula: `correct` (rederived; the
`grad_phi log rho(z)` term is zero because `z` does not depend on `phi`, the
chain rule produces the target score at the current mapped point times
`grad_phi T_phi(z)`, and no boundary term arises because the domain is all
of `R^d` and no integration by parts in `theta` is used). Classification of
the base-side interchange under the stated A4: `unsupported` as written --
A4 says "differentiation and target integration may be interchanged" and its
example envelope covers `||s_phi(theta)||` under `pi` only, which is the
forward case (8).

Smallest exact repair (changes only a proof assumption): extend A4 to

```text
Differentiation may be interchanged with the target-side integral in (4)
and with the base-side integral in (7); for example, ||s_phi(theta)|| has a
pi-integrable envelope, and

  ||grad_phi log abs det D T_phi(z)||
  + ||grad_theta log tilde_pi(T_phi(z))|| * ||grad_phi T_phi(z)||

has a rho-integrable envelope, locally uniformly in phi on the parameter
set being considered.
```

Impact: proof-assumption wording only. It also covers the identical
interchange needed for (28). The note's Section 10 consequence -- that
reusing a target value or score evaluated at an old mapped point is wrong
relative to (9) -- is `correct` and unaffected.

### F4. The (33f) absorption should state why the r_{t+1} cross terms are harmless: r_{t+1} is neither history-measurable nor conditionally centered, but it has a deterministic summable envelope

Note anchors: (33c)-(33d), lines 557-574; (33f) and the absorption sentence,
lines 589-599.

Classification of (33f): `correct`. Verification: from (33c),

```text
V_{t+1} = ||y_t - phi_star - eta_t g(phi_t) - eta_t M_{t+1} + r_{t+1}||^2.

r_{t+1} = a[(eta_t - eta_{t+1}) u_{phi_t}(B_{t+1})
            + eta_{t+1} (u_{phi_t}(B_{t+1}) - u_{phi_{t+1}}(B_{t+1}))],

so ||r_{t+1}|| <= beta_t
   := a U (eta_t - eta_{t+1}) + a L_u D eta_t eta_{t+1},

with U = sup||u||, L_u the constant from F2, and D a bound on the update
direction (assumptions 4-5). beta_t is deterministic and summable, which is
(33d). r_{t+1} depends on B_{t+1} and phi_{t+1}, so it is not measurable
with respect to the history through t and E[r_{t+1} | history] != 0 in
general. This does not matter: every cross term containing r_{t+1} --
2<y_t - phi_star, r_{t+1}>, -2 eta_t <g, r_{t+1}>, -2 eta_t <M_{t+1},
r_{t+1}>, and ||r_{t+1}||^2 -- is bounded pathwise by a constant times
beta_t, because y_t - phi_star, g, and M_{t+1} are all uniformly bounded
(compact C, ||y_t - phi_t|| <= a U eta_t, assumptions 4-5). These bounds go
into the deterministic summable d_t. The martingale cross term
-2 eta_t <y_t - phi_star, M_{t+1}> vanishes in conditional expectation
because y_t is history-measurable, and the drift term splits as

  <y_t - phi_star, g(phi_t)>
    = <e_t, g(phi_t)> + <y_t - phi_t, g(phi_t)>
    >= mu ||e_t||^2 - c eta_t,

whose O(eta_t) displacement is absorbed into c_3 eta_t^2 exactly as the
note says. The Robbins--Siegmund step and the concluding limit argument
(lines 601-612) are correct as written.
```

Impact: proof-presentation repair only (one added sentence). Also verified
here: both `M^B_{t+1}` and `M^R_{t+1}` are martingale differences with
respect to the history sigma-field generated by
`(B_0..B_t, base batches 0..t-1)` -- `M^R` because the base batch at `t` is
drawn after `phi_t` is fixed, and `M^B` because the refresh randomness is
independent of the base draws (assumption 2) and `phi_t` is
history-measurable, then the tower property.

### F5. Theorem 2A statement tightenings: the stochastic status of lambda_t should be declared; the history-measurability of R_t is not actually needed

Note anchors: statement of `R_t` and (33g), lines 627-638; (33h), lines
645-648; (33i), lines 670-675.

Classification of Theorem 2A: `correct` under its stated assumptions.
Verification of (33i): expanding `||e_{t+1}||^2` with the projection
inactive, the `<e_t, xi_{t+1}>` term vanishes conditionally because
`xi_{t+1}` is a bounded martingale difference given the pre-draw history
(Theorem 1 applies because `phi_t` and the frozen proposals are pre-draw
measurable and `s_{phi_t}` is an `f` fixed before the draws); (30) bounds
the drift; `|2 eta_t lambda_t <e_t, R_t>| <= 2 diam(C) sup||R|| eta_t
lambda_t` gives `c_5 eta_t lambda_t`; and the squared update is
`c_4 eta_t^2` by the uniform boundedness assumptions plus
`sup_t lambda_t < infinity`. Robbins--Siegmund then concludes as in F4.

Two tightenings, both wording-level:

1. (33h) does not say whether `lambda_t` may be random. The proof as
   written wants `lambda_t` deterministic, or adapted with
   `sum_t eta_t lambda_t < infinity` almost surely (Robbins--Siegmund
   accepts adapted nonnegative summable perturbations). State one.
2. `R_t` is introduced as "history-measurable", but the bound uses only
   `sup_t ||R_t|| < infinity` pathwise; measurability of `R_t` is never
   used. Either drop the qualifier or keep it as hygiene -- but do not let
   a future reader believe the proof depends on it.

Also verified: the note's boundary claim that constant-weight stale replay
from an evolving proposal is outside both theorems is `correct`, and the
classification of the controlled-Markov, increasing-buffer, and
cross-fitting routes as separate additional proofs rather than consequences
(lines 683-690, 898-911) is `correct`: Theorem 2's Poisson construction
requires a phi-independent buffer kernel, which fails once block generation
adapts to `phi_t` (the buffer becomes a controlled Markov chain), and
Theorem 2A's mechanism is summability, which constant weights violate
(`sum_t eta_t lambda_t = lambda sum_t eta_t = infinity`).

Impact: changes only proof-assumption wording.

### F6. Proposition 3 is correct but proves non-equality only by non-assumption; add the one-line explicit witness

Note anchors: Proposition 3 and proof, lines 345-366.

Classification: `correct`. The self-normalized estimator (18) is biased at
finite `N` in general, the law of large numbers in (22) converges to the
finite-`N` expectation `E[H_1(phi)]`, and averaging blocks removes
between-run variance, not finite-`N` bias. The proof, however, supports
"in general not equal" only by observing that equality was never assumed.
Smallest strengthening -- an explicit witness:

```text
N = 1: the single normalized weight is W_1 = 1 regardless of tilde_pi, so
pi_hat(f) = f(X) with X ~ r, and E[pi_hat(f)] = E_r[f] != E_pi[f] whenever
r != pi on the relevant f. With f = s_phi this gives
E[H_1(phi)] = -E_r[s_phi] != -E_pi[s_phi] = grad F(phi) generically.
```

Impact: presentation only. The downstream classification of the existing
`N=100` normalized-weight artifacts as SMC-N finite empirical blocks that
"cannot be upgraded to SMC-U by equalizing each run's total replay mass"
(lines 368-370) is `correct`; the loader applies a uniform `-log 7` shift
and the trainer's joint softmax gives each training bank total mass 1/6,
which is an aggregation convention, not an unnormalized estimator.

### F7. Proposition 4's smoothness hypothesis should say the conditional CDFs are C^1 jointly in the conditioned and conditioning variables

Note anchors: Proposition 4, lines 748-757; proof, lines 759-784.

Classification: `correct` under the intended reading. The
Knothe--Rosenblatt construction (37)-(38) is standard and the pushforward
algebra is right. For `R_p`, hence `T_star = R_pi^{-1} o R_rho`, to be a
`C^1` diffeomorphism of `R^d`, each component
`P_p(X_j <= x_j | X_{1:j-1} = x_{1:j-1})` must be continuously
differentiable jointly in `(x_j, x_{1:j-1})`, not merely in `x_j` for each
fixed conditioning value. The current phrase "successive conditional
cumulative distribution functions are continuously differentiable" admits
the weaker coordinatewise reading. One clause fixes it.

The stated boundary -- existence in an unrestricted triangular class, no
membership claim for the implemented finite dense IAF (lines 786-789) -- is
`correct` and must be kept.

Impact: proof-assumption wording only.

### F8. The uniform-boundedness assumptions are honest but strong; record them as campaign-design constraints

Note anchors: Theorem 2 assumptions 4-5, lines 480-485; Theorem 2A uniform
boundedness over adaptive histories, lines 643-644.

Classification: `correct` as explicitly stated assumptions; `unsupported`
as properties of the active runtime, which the note itself says.
Specifics worth recording for any future campaign:

- `H(B,phi)` uniformly bounded requires
  `ess sup [tilde_pi(X)/m_b(X)] ||s_phi(X)||` finite over the block law.
  Gaussian-type targets with proposals whose tails do not dominate, or
  unbounded scores, violate it. Tail-dominant proposals plus the bounded
  `s_max`-capped transport family make it plausible but unproved.
- The fresh reverse estimator uniformly bounded (assumption 5) generally
  fails for unbounded target scores under Gaussian base draws unless the
  estimator is truncated or clipped; a truncated variant needs its own bias
  accounting before it can claim (28). The note already flags the
  second-moment generalization as available but unclaimed (lines 484-485),
  which is the honest disposition.

Impact: no change required; these are assumption-realism records, not
errors.

---

## Item-by-item adjudication of the twelve required review items

| Handoff item | Verdict | Basis |
|---|---|---|
| 1. Target/transport identities | `correct` | (1), (2), (6), (7) rederived; Jacobian cancellation `D T^{-1}(T(z)) = [D T(z)]^{-1}` checked; `pi_phi^z` integrates to one; (7) equals `KL(rho \|\| pi_phi^z)` exactly |
| 2. Gradient identities | `correct` with F3 | (8) is the forward interchange; (9) rederived with total derivative through `T_phi(z)`; no boundary term (no phi-dependent domain, no integration by parts in theta); missing regularity = base-side envelope, repair in F3 |
| 3. Deterministic-mixture estimator | `correct` | Theorem 1 is the balance-heuristic/deterministic-mixture identity with samples from `r_bh` and mixture denominator `m_b`; conditional unbiasedness holds for proposals frozen after arbitrary history and `f` fixed pre-draw; each `h`-term is dominated via `alpha_h r_h / m_b <= 1`, so `tilde_pi`-integrability of `f` suffices (state this measure explicitly; minor); vector case componentwise; finite variance correctly not claimed in Theorem 1 and deferred to Theorem 2's boundedness assumption (F8) |
| 4. SMC boundary | `correct` | SMC-U vs SMC-N split matches the stored Del Moral--Doucet text: normalized estimators and consistency at lines 400-426; unnormalized ratio unbiasedness under unbiased resampling at lines 428-470. Proposition 3's averaging claim verified (F6); demanding an implementation-level Feynman--Kac proof for (19) is the right bar |
| 5. Replacement unit | `correct` | Whole-population replacement is required for SMC-N semantics (Counterexample 2); row-level replay can be valid only for known-density blocks with bound source densities and content-independent row selection with known inclusion probabilities, which reduces it to another known-density MIS block -- matching the Section 10 table |
| 6. Buffer Markov chain | `correct` | Lemma 1's coupling (shared coins, shared replacement blocks) preserves marginal laws; slots agree forever after first refresh; union bound gives (25); invariance of `P_B^K` and uniqueness verified; Corollary 1's stationary mean field (26) follows from Theorem 1 slot-wise; the note's caveat that this is not conditional unbiasedness for a buffer-trained `phi` is correct and important |
| 7. Theorem 2 | `correct`; eight assumptions sufficient | Poisson series and equation verified (F2 supplies the missing Lipschitz derivation); time ordering update-then-refresh consistent with both martingale differences (F4); (33c) substitution checked term by term; (33d) summability verified; (33e)-(33f) verified with the cross-term accounting in F4; Robbins--Siegmund conclusion and the limit-identification argument correct. Strength of (30): see F1 -- provably unattainable on symmetric-copy-containing sets for the implemented family |
| 8. Theorem 2A | `correct` with F5 | Conditional unbiasedness after history-dependent proposal selection follows from Theorem 1's freeze-before-draw structure; (33i) verified; (33h) does make arbitrary bounded stale replay a summable perturbation; constant-weight stale replay from an evolving proposal is genuinely outside both theorems; the three alternative routes are correctly classified as separate proofs |
| 9. Exact minimizer | `correct` | Lemma 2's Jensen argument with `A = {p > 0}`, `c = integral_A q` verified including the equality case (`c = 1` and `q/p = 1` p-a.e., forcing `q = 0` a.e. off `A`); Theorem 3 verified; `q_phi > 0` everywhere for a full-support Gaussian base through an `R^d` diffeomorphism, so `pi << q_phi` is automatic; infinite-KL cases do not break nonnegativity or the zero-sum argument; existence is cleanly separated from optimizer convergence |
| 10. Expressivity | `correct` with F7 | Rosenblatt existence verified; the unrestricted-triangular-class boundary statement is exactly right and says nothing about finite dense IAF membership |
| 11. Gaussianization and HMC | `correct` | Corollary 2 is (2) plus (6) at `pi = q_{phi_star}`; Corollary 3's rotation solution `z(tau) = z(0) cos tau + p(0) sin tau` gives full position-momentum exchange at `tau = pi/2` with acceptance one under exact integration; the note's scope sentences (lines 800-802, 824-826) correctly deny any finite-leapfrog, tuned-trajectory, or sign-mode-crossing implication |
| 12. Counterexamples | `correct`; none defeats a theorem | CE1 violates content-independent refresh (state the within-region law of retained rows to make it self-contained; currently illustration-grade); CE2 arithmetic checked; CE3 violates the support assumption; CE4 concerns fixed replay (10), which no convergence theorem covers; CE5 lives in a family with no exact member, which Theorem 3 requires. Each counterexample breaks exactly one stated assumption, so none contradicts any theorem under its full assumption set |

## Source and implementation anchor checks

- NeuTra paper, lines 87-125 of the stored text: contains the
  change-of-variables identity (paper eq. 2) and the reparameterized ELBO
  (paper eq. 3) with the unbiased Monte Carlo estimate statement. The
  note's Section 2 attribution is accurate, and (9) is the negative-ELBO
  gradient in the note's sign convention.
- Del Moral--Doucet stored preprint: lines 400-426 give the normalized
  estimators and the consistency statement; lines 428-470 give the
  normalizing-constant ratio estimator and "if the resampling scheme used
  is unbiased, then (4) is also unbiased." The note's Section 5 split rests
  on exactly this distinction and cites the right regions.
- `bayesfilter/inference/neutra_weighted_training.py`: the class docstring
  and `_train_step_impl` (lines 604-714; softmax normalization and loss at
  lines 668-673) compute exactly (10) with `W_i` the joint softmax weights,
  under `stop_gradient` on the weights. The note's Section 3 description is
  `correct`, including the boundary that (10) targets the fixed empirical
  measure (11), not (4).
- `docs/benchmarks/run_ssl_lstm_q20_neutra_global_mixing_training_2026_08_19.py`:
  `_run_arm` (lines 343-414) feeds the identical `train_rows`,
  `train_weights` tensors to every update and selects on the held-aside
  rows; data assembly (lines 653-658) slices rows 0:600 for training and
  600:700 for selection from the seven verified 100-row banks loaded by
  `_load_replay` (canary module lines 228-264, uniform `-log 7` shift).
  The note's "six 100-particle SMC training populations" framing and both
  cited line regions are accurate.

## What this review does not establish

This reply certifies conditional mathematics only. It does not establish
that the active q=20 target satisfies Assumption set A, that any proposal
library satisfies the boundedness assumptions, that (30) holds on any
useful basin for the implemented dense IAF (F1 shows it cannot hold
globally), that the optimizer reaches such a basin, that a proof-bearing
block generator has acceptable variance, or that any finite HMC kernel
crosses the sign barrier. The note's Section 13 evidence gates and Section
14 open questions remain the governing empirical boundary, unchanged.

## Correction-impact summary

| Finding | Impact class |
|---|---|
| F1 | Claim-boundary sharpening; no theorem, estimator, or direction change |
| F2 | Proof-presentation repair (derivation supplied); no assumption change |
| F3 | Changes only a proof assumption (extend A4 to the base side) |
| F4 | Proof-presentation repair (one sentence); no assumption change |
| F5 | Changes only proof-assumption wording (lambda_t status; R_t qualifier) |
| F6 | Proof-presentation strengthening (explicit witness); no change |
| F7 | Changes only a proof assumption (joint C^1 wording) |
| F8 | Assumption-realism record; no change required |

No finding invalidates a theorem, changes an estimator, or changes the
research direction. The recommended edits are textual and can be applied in
a follow-up revision of the note without reopening the mathematics.

```text
ESTIMATOR_VERDICT: AGREE
REPLAY_CONVERGENCE_VERDICT: AGREE
GAUSSIANIZATION_VERDICT: AGREE
SSL_LSTM_CLAIM_BOUNDARY_VERDICT: AGREE
OVERALL_VERDICT: AGREE
```
