# SSL-LSTM q=20 adaptive-replay NeuTra: audit adjudication and plan amendment

Date: 2026-08-23
Status: `FABLE_ADJUDICATED_AB_AMENDMENT_IN_SEPARATE_EXECUTION_PLAN`

## Scope and evidence boundary

This amendment adjudicates the available independent review:

`docs/plans/bayesfilter-ssl-lstm-q20-adaptive-replay-neutra-fable-review-reply-2026-08-23.md`

Fable reviewed commit `68ba5271989fe35740416dff599bb61c83dfa099`, which is the
current repository `HEAD`. The second path supplied by the user,
`/home/ubuntu/python/BayesFilter/docs/analysis/bayesfilter-ssl-lstm-q20-adaptive-replay-neutra-fable-handoff-2026-08-21-analysis.md`,
does not exist in this checkout: `docs/analysis/` is absent, it is not present
in the current Git tree, and no alternate copy was found under
`/home/ubuntu/python` or the Codex attachment store. This amendment therefore
does not claim to adjudicate the Grok review. Any disagreement or additional
finding in that artifact remains open until its contents are supplied.

Primary mathematical artifact:

`docs/plans/bayesfilter-ssl-lstm-q20-adaptive-replay-neutra-mathematical-note-2026-08-21.md`

Governing prior plan:

`docs/plans/bayesfilter-ssl-lstm-q20-adaptive-replay-neutra-mathematics-review-plan-2026-08-21.md`

## Adjudication verdict

Fable's five terminal verdicts are accepted for the **conditional note**:
the estimator identities, fixed-law replay theorem, adaptive fresh-block
theorem with summable stale replay, exact Gaussianization result, and explicit
finite-SSL-LSTM nonclaims are viable under their stated assumptions.

That verdict is not a runtime admission. In particular, F1 establishes that
the strong monotonicity condition (30) cannot hold on any admissible parameter
set containing two distinct symmetry-equivalent stationary points of the
implemented tanh masked network. Thus Theorems 2 and 2A are local-basin
conditional results for the implemented parameterization unless a quotient or
gauge-fixed parameterization is introduced.

No substantive Fable finding is rejected. Two interpretations are explicitly
rejected:

1. `OVERALL_VERDICT: AGREE` must not be read as evidence that the active q=20
   target, proposal library, dense IAF, optimizer, or HMC kernel satisfies the
   theorem assumptions.
2. F8's boundedness discussion is a risk and campaign-design constraint, not
   evidence that boundedness holds in the runtime.

## Finding ledger

| Finding | Classification | Agreement | Required disposition |
|---|---|---|---|
| F1: parameter symmetries obstruct global strong monotonicity | Conditional mathematical finding | Agree, conditional on `C` containing both distinct copies | Add a local-basin/symmetry boundary; do not claim global dense-IAF convergence; optionally investigate quotient/gauge fixing as a separate hypothesis |
| F2: Poisson-series Lipschitz support is asserted too briefly | Proof-presentation gap | Agree | Add the coupling bound and uniform Lipschitz derivation after (32) |
| F3: A4 covers target-side but not base-side differentiation | Proof-assumption gap | Agree | Extend A4 with a locally uniform rho-integrable base-side derivative envelope |
| F4: stale-replay remainder cross terms need pathwise handling | Proof-presentation gap | Agree | State the deterministic beta bound and explain why all remainder cross terms enter a summable perturbation |
| F5: lambda schedule status should be explicit | Assumption-wording repair | Agree | Use a deterministic nonnegative `lambda_t` in the theorem; retain history-measurable `R_t` as a safe sufficient condition |
| F6: finite-N normalized bias deserves an explicit witness | Presentation strengthening | Agree | Add the `N=1` witness to Proposition 3, labeled illustrative rather than a claim about the actual N=100 run |
| F7: Rosenblatt smoothness must be joint | Proof-assumption repair | Agree | Require conditional CDFs and inverses to be jointly C1 in conditioned and conditioning variables |
| F8: boundedness assumptions are strong | Assumption-realism risk | Agree as a risk, not as evidence | Add campaign gates for tail/support/envelope checks; do not promote boundedness from diagnostics |

## Revised research intent ledger

| Field | Revised declaration |
|---|---|
| Main question | Which replay route has a defensible target and convergence argument for adaptive SSL-LSTM q=20 NeuTra training? |
| Candidate A | Fixed-law, content-independently refreshed whole-block replay covered by Theorem 2 |
| Candidate B | Fresh history-dependent known-density/SMC-U block at every update plus stale replay with `sum eta_t lambda_t < infinity`, covered by Theorem 2A |
| Excluded candidate | Constant positive-weight stale replay from a proposal law that changes with the learned transport; it requires a new controlled-Markov or finite-error proof |
| Baseline | Existing fixed empirical loss on rows 0:600 of the verified replay banks, reused on every update |
| Primary mathematical promotion | All theorem assumptions are explicit, the note contains the Fable repairs, and no theorem is applied outside its route |
| Empirical promotion | Separate target/proposal validity, heldout transport diagnostics, and one common transformed-target HMC gate pass; no pooled mode-locked chains |
| Hard veto | Missing support, unbound proposal density, failed target status, finite normalized SMC block labeled unbiased, non-summable adaptive stale replay, or global-convergence language for the symmetric dense IAF |
| Nonclaim | No finite-sample guarantee, exact finite-IAF representation claim, optimizer success claim, whitening claim, mode-discovery claim, or HMC convergence claim follows from this amendment |

## Revised execution plan

### Phase 1: repair and provenance

1. Apply the note edits listed in the revision ledger below.
2. Record the Fable review and the missing Grok artifact as separate evidence
   classes; do not merge their verdicts.
3. Re-run Markdown and focused symbolic checks. No training or HMC is part of
   this phase.

### Phase 2: route and assumption preflight

1. Classify a proposed training run as Candidate A or Candidate B before any
   particle generation.
2. For Candidate A, verify a fixed block law, content-independent refresh,
   and proposal/source-density provenance.
3. For Candidate B, verify a fresh block at every update, conditional
   freeze-before-draw semantics, deterministic `lambda_t`, bounded stale
   replay, and a recorded proof of `sum eta_t lambda_t < infinity`.
4. Measure proposal support, importance-ratio tails, score/Jacobian envelopes,
   target statuses, and finite-moment diagnostics on calibration data. These
   diagnostics can veto a route or trigger a repair; they cannot certify the
   uniform boundedness assumptions by themselves.
5. Treat the dense IAF symmetry inventory as a local-basin constraint. Either
   freeze a symmetric-copy-free basin for a conditional diagnostic or open a
   separately reviewed quotient/gauge-fixing hypothesis. Do not call (30)
   global for the current parameterization.

### Phase 3: empirical campaign, only after Phase 2

1. Keep whole SMC populations as units; never delete arbitrary rows from a
   normalized SMC-N population and retain its original normalized semantics.
2. Use disjoint calibration, validation, and audit blocks. Preserve source
   densities, proposal mixtures, ancestry, seeds, target signatures, and
   replacement probabilities.
3. For Candidate B, make the fresh block the persistent forward-gradient lane;
   stale replay is an explicitly weighted perturbation, not the claimed target
   estimator.
4. Evaluate fresh pullback moments and dependence, heldout losses, and
   proposal-tail diagnostics. Treat them as explanatory or veto diagnostics,
   not proof of Gaussianity.
5. Freeze one admitted transport and run one common transformed-target HMC
   procedure. Initialization from multiple modes is allowed only as an
   overdispersed start; pooled occupancy from mode-locked chains is invalid.
   Require chain-level forgetting and cross-mode transitions under the same
   target before any posterior or predictive claim.

## Mathematical-note revision ledger

The following edits are authorized by this amendment and affect documentation,
not the target estimator or research direction:

1. **A4 base-side regularity:** include a locally uniform rho-integrable
   envelope for the derivative of the log Jacobian and the composed target
   density in (9) and (28).
2. **Theorem 2 Poisson regularity:** insert the coupling argument showing
   boundedness and Lipschitz continuity of `u_phi`, using the geometric tail
   `K(1-epsilon)^j` rather than a non-summable termwise Lipschitz bound.
3. **Theorem 2 remainder:** state the pathwise `beta_t` envelope for `r_t` and
   explain that its cross terms are bounded by a deterministic summable
   sequence; no centering of `r_t` is assumed.
4. **Theorem 2A schedule:** make `lambda_t` deterministic, nonnegative,
   bounded, and summably weighted by `eta_t`; retain `R_t` as history
   measurable and uniformly bounded.
5. **Proposition 3:** add the `N=1` self-normalized witness and label it as a
   generic finite-N counterexample, not an estimate of the actual N=100 bias.
6. **Rosenblatt proposition:** state joint C1 regularity in conditioned and
   conditioning variables, including the corresponding inverse maps.
7. **Symmetry boundary:** state that (30) is impossible on a set containing
   two distinct hidden-unit permutation or tanh-sign copies of a stationary
   point, and therefore only supports local-basin conditional reasoning for
   the implemented dense IAF.
8. **Assumption-realism section:** record uniform boundedness as an unverified
   campaign constraint; if only finite second moments are observed, do not
   silently substitute the stronger theorem. Either use a separately proved
   second-moment stochastic-approximation result or downgrade the claim.

## Stop conditions and next decision

Stop the mathematical route if any revision reveals a counterexample under all
stated assumptions. Stop an empirical route if support, target status,
proposal-density provenance, adaptive-summability, or common-target HMC gates
fail. A failed boundedness diagnostic is a repair trigger or a theorem-scope
downgrade, not evidence against the general replay idea.

The revised mathematical note and revision result record are complete. The
paired A/B execution amendment is recorded in
`docs/plans/bayesfilter-ssl-lstm-q20-adaptive-replay-neutra-ab-comparison-plan-2026-08-24.md`.
That plan authorizes only its bounded route preflight and training screen; no
HMC run or public claim is authorized by this mathematical adjudication alone.
