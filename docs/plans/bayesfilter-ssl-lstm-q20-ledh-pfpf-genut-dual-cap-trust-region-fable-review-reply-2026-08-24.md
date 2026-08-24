# SSL-LSTM q=20 LEDH-PFPF-GenUT Dual-Cap/Trust-Region: Fable Review Reply

Date: 2026-08-24

Status: `READ_ONLY_BOUNDED_REVIEW_COMPLETE`

Auditor: Fable (claude-fable-5), independent mathematical reviewer

Handoff request:
`docs/plans/bayesfilter-ssl-lstm-q20-ledh-pfpf-genut-dual-cap-trust-region-fable-handoff-2026-08-24.md`

Audited artifacts (primary):

- Pass A:
  `docs/plans/bayesfilter-ssl-lstm-q20-ledh-pfpf-genut-dual-cap-trust-region-literature-solution-plan-2026-08-24.md`
- Pass B:
  `docs/plans/bayesfilter-ssl-lstm-q20-ledh-pfpf-genut-dual-cap-trust-region-mathematical-note-2026-08-24.md`

Review basis: read-only inspection at commit
`14e4618749c9e04e8c4d2398becadb0206b30599`. Every proposition, corollary,
counterexample, and displayed derivative in the Pass B note was rederived by
hand. The cited local paper text anchors (Ebeigbe GenUT `:84-95,114-164`;
Li--Coates `:236-270`; Cornuet AMIS `:68-110,208-222,428-465`; Hesterberg
`:295-340`), all four implementation anchors
(`genut_shape_lm_tf.py:121-167`, `dual_cap_genut_primal_tf.py:193-245`,
`genut_guided_proposal_tf.py:900-1045`,
`ledh_pfpf_genut_initial_rqmc_tf.py:500-527`), the monograph anchor
(`ch19c_dpf_implementation_literature.tex:156-183,477-512`), and the prior
replay-note anchor
(`bayesfilter-ssl-lstm-q20-adaptive-replay-neutra-mathematical-note-2026-08-21.md:298-320`)
were re-inspected directly. No file was edited, no command that trains,
samples, or launches GPU work was run, and no MathDevMCP abstention was
converted into agreement. The only write is this reply.

## Verdicts

- **Pass A (plan): VERDICT: AGREE.**
- **Pass B (mathematical note): VERDICT: AGREE.**

Scope of these verdicts, exactly as the handoff requests: they validate the
note's **scoped conditional and no-go claims** — the separation of finite-cloud
moment conditioning from density correction, replay validity, and mode
exploration, and the conditional route of Proposition 10 under its stated,
unestablished assumptions. They do **not** validate the current
implementation, any empirical result, NeuTra/HMC admission, or any
default-route change. The author's central negative position — that the
dual-cap reset alone cannot provide a density-faithful global whitening map —
is `correct` as stated and is preserved by this review (Proposition 3 plus
Corollary 4.1 jointly refute the stronger claim).

---

## Pass A: plan review

The plan contains a complete research intent ledger, an evidence contract with
explicit veto/explanatory separation, a skeptical audit completed before
execution, a default-and-assumption audit with provenance columns, stop/repair
rules, and an execution record. The comparator (current fixed normalized
replay, used as historical/descriptive context rather than as a density
authority) is correctly scoped for a documentary execution: no numerical
comparison is claimed, so no baseline ladder is owed. Promotion vetoes and
continuation vetoes are stated separately and are not conflated. The
MathDevMCP outcome is correctly recorded as partial abstention rather than
certification. No long run is authorized, matching the stated scope.

No blocking, major, or minor findings. One editorial item:

| Field | Content |
|---|---|
| Location | Plan, "Planned mathematical results," item 4 |
| Severity | `editorial` |
| Classification | `project_derivation` |
| Claim checked | "Frozen known-density or proven SMC-U blocks give a conditionally unbiased estimator" |
| Reason | `SMC-U` is defined only in the prior replay note (`bayesfilter-ssl-lstm-q20-adaptive-replay-neutra-mathematical-note-2026-08-21.md`, Route SMC-U, near line 318). The plan's scope inventory points at "the existing q=20 replay mathematics/result notes" generically. |
| Repair | Optional: cite the defining note path beside the first `SMC-U` use. |

## Pass B: mathematical note review

### The six requested checks

1. **Selected GenUT moments are not presented as density identification:**
   `correct`. Proposition 1's counterexample is exact: for
   `Q = (2/3)δ_0 + (1/6)δ_{+√3} + (1/6)δ_{−√3}`, the moments are
   `E[X]=0, E[X²]=2·(1/6)·3=1, E[X³]=0, E[X⁴]=2·(1/6)·9=3`, matching the
   standard normal through fourth order while the measures differ. The GenUT
   boundary matches the source: the Ebeigbe anchor (`:114-164`) states the
   `(2n+1)`-point rule matches mean, covariance, and the *diagonal* components
   of the skewness/kurtosis tensors, with constrained variants keeping at
   least second-order accuracy — exactly the "selected moments" reading.
2. **Caps are classified as bounded finite maps, not normalizing flows:**
   `correct`. I rederived both derivatives. Radial:
   `h(r)=r(1+r²/(dρ²))^{-1/2}` gives `h'(r)=(1+r²/(dρ²))^{-3/2}>0` with
   `h(r)→√d·ρ`. Coordinate: `g(u)=u(1+(u/b)^p)^{-1/p}` gives
   `g'(u)=(1+s)^{-1/p-1}>0`, `s=(u/b)^p`, with `|g|<b`. Both match the
   implementation exactly: `smooth_rms_cap_value` computes
   `rsqrt(1+mean_square/ρ²)` with `mean_square=‖u‖²/d`
   (`genut_shape_lm_tf.py:128-131`), and the coordinate cap and its
   derivative appear verbatim at `dual_cap_genut_primal_tf.py:238-241`.
   Bounded image ⇒ non-surjective ⇒ not a full-support flow; the inference is
   valid.
3. **Li--Coates Jacobian pairing:** `correct` and `source_faithful`. The
   paper anchor (`:236-270`) shows eq. (17) (proposal density =
   pre-flow density divided by the Jacobian determinant of the invertible
   per-particle map) and eq. (18) (weight containing post-flow transition
   `p(η₁|x_{k-1})`, observation `p(z_k|η₁)`, determinant, divided by the
   pre-flow density), plus the explicit warning that non-invertible flows
   break (18). Proposition 4 and the note's weight display reproduce this
   pairing; the monograph anchor
   (`ch19c_dpf_implementation_literature.tex:156-183`) carries the same
   identity, and its `:477-512` section separates proposal correction from
   exact filtering as the note claims. Corollary 4.1 is sound: the
   implementation forms PF-PF logits as
   `target_initial + target_observation − proposal_log + forward_log_det`
   (`ledh_pfpf_genut_initial_rqmc_tf.py:521-526`) *before* the cloud reset
   (`_restore_cloud_primal`, line 537), and the Contract-E-then-dual-cap
   sequence (`genut_guided_proposal_tf.py:900-1044`) records diagnostics
   only, no joint `Nd×Nd` density correction. The stored determinant
   therefore belongs to `Φ`, not to `R`, as the note states.
4. **Deterministic-mixture replay is conditioned on frozen proposals:**
   `correct` and `source_faithful`. Definition 1 and Propositions 5 and 7
   freeze the proposal schedule before the draws to which unbiasedness is
   applied. The Cornuet anchor confirms both halves: eq. (2)-(4) give the
   deterministic-mixture weight and its unbiasedness in the non-adaptive
   case, and Section 5 (`:428-465`) states plainly that with adaptation "the
   estimator is no longer unbiased" without compactness/bounded-target
   conditions — exactly the caveat the note attaches after Proposition 7.
   Proposition 6's `N=1` ratio-bias argument is elementary and correct, and
   its classification of the six stored normalized 100-particle populations
   matches the prior replay note's own Route SMC-U/SMC-N distinction.
5. **Replay denominator positivity and defensive-mixture hypotheses:**
   `correct`. Definition 1 states `m_b(θ)>0` wherever `π̃(θ)|f(θ)|>0` as part
   of the block contract; Proposition 8 states
   `0<ε_min≤ε_t≤1`, derives the second-moment bound from
   `m_t ≥ ε_min·r_safe`, and explicitly says finiteness of the right-hand
   integral is an *additional* assumption. The Hesterberg boundary is drawn
   faithfully: the anchor (`:295-340`) shows the bounded-weight statement
   `W(x) ≤ 1/h` requires the target `f` as a mixture component, which the
   note correctly refuses to claim for a merely evaluable target.
6. **Tempering is not advertised as a finite mode-discovery guarantee:**
   `correct`. Proposition 9 claims only the ideal Feynman--Kac targeting
   under invariant mutation kernels, and Section 8 closes with the exact
   elementary bound `1−(1−p_A)^n` and the statement that no finite
   deterministic coverage guarantee follows.

Propositions 2, 5, 8, 9, and 10 were also rederived line-by-line; each proof
is valid under its stated hypotheses, and Proposition 10 is properly
conditional — its closing paragraph does not upgrade the conditional theorem
into a claim about the current code.

### Findings ordered by severity

No `blocking` or `major` findings. All findings below leave every proposition,
estimator, boundary, and nonclaim intact.

#### F1. Proposition 2: state the whitening property instead of "square-root convention"

| Field | Content |
|---|---|
| Location | Note §4, Proposition 2 |
| Severity | `minor` |
| Classification | `project_derivation` |
| Claim checked | Restored cloud has exactly mean `μ_X` and covariance `Σ_X` "up to the chosen matrix square-root convention" |
| Reason | The identity requires the standardization factor `W = C_Y^{-1/2}` to satisfy `W C_Y Wᵀ = I` (true for the symmetric root and for inverse-Cholesky whitening, which is what `_standardize_uniform` followed by right-multiplication with `target_cholesky` at `dual_cap_genut_primal_tf.py:242-245` implements). The convention caveat is correct but does not name the property that any admissible convention must satisfy. |
| Repair | Optional wording: "for any factor `W` with `W C_Y Wᵀ = I`," replacing the convention clause. No mathematical change. |

#### F2. Forward-gradient remark after Proposition 5 uses an interchange stated only in Proposition 10

| Field | Content |
|---|---|
| Location | Note §7, display after Proposition 5 (`∇F(φ) = −E_π[s_φ]`) |
| Severity | `minor` |
| Classification | `project_derivation` |
| Claim checked | `−γ̂_b(s_φ)` targets `Z∇F(φ)` for the forward KL objective |
| Reason | The identity `∇_φ KL(π‖q_φ) = −E_π[∇_φ log q_φ]` requires differentiation/integration interchange (dominated convergence or equivalent). The note assumes this only in Proposition 10, assumption 6, four sections later. The scale calculation itself is correct: `Z∇F = −∫π̃ s_φ`, and `γ̂_b(s_φ)` is conditionally unbiased for `∫π̃ s_φ` by Proposition 5 at fixed `φ`. |
| Repair | Add a forward pointer: "assuming the interchange justified in Proposition 10, assumption 6." |

#### F3. Reference 8 author error

| Field | Content |
|---|---|
| Location | Note, References, item 8 |
| Severity | `editorial` |
| Classification | `source_faithful` |
| Claim checked | Attribution of *Annealed Importance Sampling* (2001) |
| Reason | "A. S. S. Neal" is not the author; the paper is by Radford M. Neal. |
| Repair | Replace with "R. M. Neal". |

#### F4. Grammar in §2

| Field | Content |
|---|---|
| Location | Note §2, first paragraph |
| Severity | `editorial` |
| Classification | `project_derivation` |
| Claim checked | N/A (prose) |
| Reason | "existing local copies … is retained" — subject/verb disagreement. |
| Repair | "are retained". |

#### F5. Proposition 3 understates the injectivity actually proved

| Field | Content |
|---|---|
| Location | Note §5, Proposition 3 statement ("injective along each ray") |
| Severity | `editorial` |
| Classification | `project_derivation` |
| Claim checked | Injectivity of the radial cap |
| Reason | Since `C_ρ` preserves direction and its radial profile `h` is strictly increasing, the map is globally injective on `ℝ^d`, not merely ray-wise. The weaker statement is true and suffices for the non-surjectivity conclusion, so nothing downstream changes. |
| Repair | None required; may strengthen to "a global injection with bounded image" if desired. |

---

## Remaining evidence gaps and the exact next artifact

Per the handoff's request, since no finding rises above `minor`:

1. **Gap:** Proposition 10 is conditional and none of its six assumptions is
   established by current artifacts (the note says so; this review confirms
   no artifact inspected here establishes them). **Next artifact:** a
   reviewed implementation-phase experiment plan for assumption 2 alone — a
   per-proposal LEDH artifact (pre-flow density, transition, observation,
   covariance state, pseudo-time factors, determinant product, support
   declaration) plus an affine known-map density-identity test, which is the
   cheapest discriminating check of the proposal contract.
2. **Gap:** the defensive component's score-class second moment
   (Proposition 8's integrability hypothesis) is unquantified for the actual
   q=20 target. **Next artifact:** a tail second-moment estimate for a
   concrete `r_safe` candidate against the forward score class, as a bounded
   diagnostic run under its own small plan.
3. **Gap:** replay metadata sufficiency for Proposition 7 (recomputable
   historical log densities) is a code property this bounded review did not
   audit. **Status:** `not checked`. **Next artifact:** a wiring/parity test
   that recomputes `log m_b` for one stored block from retained metadata.

## Non-authorizations

This reply is a mathematical review only. It does not approve implementation
work, sampler changes, default-route changes, GPU campaigns, replay-buffer
changes, or NeuTra/HMC admission. Those require their own plans under the
repository's evidence-contract and per-scope tuning rules.
