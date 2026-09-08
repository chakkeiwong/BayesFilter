# SSL-LSTM q=20: LEDH-PFPF-GenUT dual-cap/trust-region literature and solution plan

Date: 2026-08-24  
Status: `DOCUMENTARY_EXECUTION_BOTH_REVIEWS_RECEIVED_IMPLEMENTATION_PENDING`

## Research intent ledger

| Field | Declaration |
|---|---|
| Main question | Can the canonical LEDH-PFPF-GenUT dual-cap/trust-region construction repair the current q=20 NeuTra failures in variance retention, proposal-density faithfulness, replay weighting, and mode coverage? |
| Candidate mechanism | Use LEDH-PFPF with an explicit change-of-variables correction, UKF/GenUT local moment control, dual smooth caps, trust-region LM steps, and a proposal-aware replay/tempering layer. |
| Exact comparator | The current fixed normalized SMC replay and the existing finite-capacity NeuTra transport, treated as historical/descriptive context rather than a valid density authority. |
| Expected failure modes | Moment matching mistaken for density matching; capped maps losing full support or invertibility; omitted/incorrect Jacobian or covariance terms; finite self-normalized replay bias; adaptive-proposal denominator drift; mode non-discovery; poor tail second moments; local caps suppressing bridge motion. |
| Primary decision criterion | A proposition-proof note must identify a conditional route whose estimator targets the stated measure, and must state exactly which assumptions are not established by the current code/artifacts. |
| Promotion veto | Any claim that GenUT sigma points are IID samples, that moment restoration implies density restoration, that a capped map is automatically a normalizing flow, or that fixed-size normalized replay is unbiased. |
| Continuation veto | A source-faithful change-of-variables identity or replay identity is contradicted by the inspected code/math, or the target/proposal density cannot be defined on a common support. |
| Repair trigger | MathDevMCP finds an algebraic inconsistency, a source anchor does not support a claimed operation, or the local implementation violates an assumption used in a proposed theorem. |
| Explanatory diagnostics | Whitening moments, ESS, cap-active fractions, covariance gaps, determinant residuals, mode occupancy, R-hat, and validation loss. They do not establish density correctness or HMC readiness by themselves. |
| Must not be concluded | No proof of global mode discovery, exact nonlinear filtering, NeuTra optimizer convergence, HMC convergence, or superiority of dual-cap over other proposals. |

## Evidence contract

**Question.** Which parts of LEDH-PFPF-GenUT dual-cap/trust-region are mathematically
useful for the q=20 problem, and what additional construction would be required
for a valid learned-transport/replay route?

**Literature comparator.** GenUT moment quadrature; invertible particle-flow
importance correction; deterministic-mixture/AMIS recycling; defensive mixtures;
tempered SMC/SMC mutation; SQMC/RQMC; and ensemble-transform OT.

**Primary pass criterion.** Every proposed positive claim has either a checked
derivation in the note or a primary source/precise local-code anchor. Every
negative claim has an explicit counterexample or support argument.

**Veto diagnostics.** Missing source anchor; non-invertible or bounded-support
map used as if it were a global flow; stale replay denominator; normalized-only
block treated as an unnormalized estimator; or a theorem whose conditioning/adaptivity
is unspecified.

**Explanatory only.** Finite-run loss, ESS, whitening, cap activity, and mode
counts can nominate a repair but cannot promote a method.

**Nonconclusion.** A conditional theorem is not evidence that the present q=20
implementation satisfies its assumptions. A literature result is not a guarantee
for this target or architecture.

**Artifacts.**

1. Mathematical note:
   `docs/plans/bayesfilter-ssl-lstm-q20-ledh-pfpf-genut-dual-cap-trust-region-mathematical-note-2026-08-24.md`
2. MathDevMCP audit record:
   `docs/plans/bayesfilter-ssl-lstm-q20-ledh-pfpf-genut-dual-cap-trust-region-mathdevmcp-audit-2026-08-24.md`
3. Fable review memo:
   `docs/plans/bayesfilter-ssl-lstm-q20-ledh-pfpf-genut-dual-cap-trust-region-fable-handoff-2026-08-24.md`
   Its completed result is recorded separately at
   `docs/plans/bayesfilter-ssl-lstm-q20-ledh-pfpf-genut-dual-cap-trust-region-fable-review-reply-2026-08-24.md`.
4. Grok review handoff:
   `docs/plans/bayesfilter-ssl-lstm-q20-ledh-pfpf-genut-dual-cap-trust-region-grok-handoff-2026-08-24.md`
   Its requested result path is the separate
   `docs/plans/bayesfilter-ssl-lstm-q20-ledh-pfpf-genut-dual-cap-trust-region-grok-review-2026-08-24.md`.
5. Source/literature ledger and command manifest in the mathematical note and
   this plan; no existing result artifact is overwritten.

## Scope and source inventory

The local implementation and source anchors to inspect are:

- `bayesfilter/highdim/genut_shape_lm_tf.py` (scaled LM solve, smooth RMS cap,
  feasibility diagnostics);
- `bayesfilter/highdim/dual_cap_genut_primal_tf.py` (diagonal/pairwise
  correction, coordinate cap, restandardization, affine restoration);
- `bayesfilter/highdim/genut_guided_proposal_tf.py` (Contract-E reset and
  trust-region wiring);
- `bayesfilter/highdim/ledh_pfpf_genut_initial_rqmc_tf.py` (PF-PF proposal
  density terms, determinant, recursive score, and reset order);
- `docs/fable-rewrite/monograph/chapters/ch19c_dpf_implementation_literature.tex`
  (Li--Coates PF-PF equations and covariance lifecycle);
- `docs/genut-dual-cap-default-algorithm-integration-note-2026-08-07.md` and
  `docs/plans/bayesfilter-genut-feasible-trust-region-repair-result-2026-08-15.md`
  (local claims and explicit nonclaims);
- the existing q=20 replay mathematics/result notes, which define the current
  target and distinguish normalized replay from an unnormalized estimator.

Primary literature to inspect at technical-method level:

- Ebeigbe et al., *A Generalized Unscented Transformation for Probability
  Distributions*, arXiv:2104.01958 / PMC 8043458;
- Li and Coates, *Particle Filtering with Invertible Particle Flow*,
  arXiv:1607.08799, especially the proposal-density/Jacobian construction;
- Cornuet et al., *Adaptive Multiple Importance Sampling*, arXiv:0907.1254;
- Hesterberg, *Weighted Average Importance Sampling and Defensive Mixture
  Distributions*, Technometrics 37 (1995);
- Fearnhead and Taylor, *An Adaptive Sequential Monte Carlo Sampler*,
  arXiv:1005.1193;
- Gerber and Chopin, *Sequential Quasi-Monte Carlo*, arXiv:1402.4039;
- Reich, *A non-parametric ensemble transform method for Bayesian inference*,
  arXiv:1210.0375;
- local copies of Neal (AIS), Del Moral--Doucet (SMC samplers), and
  Parno--Marzouk (transport-map MCMC).

The note will distinguish source-faithful statements, direct derivations in the
project notation, and empirical/implementation hypotheses.

## Planned mathematical results

The note will state and prove, or explicitly refute, the following propositions.

1. **Selected-moment proposition.** GenUT/dual-cap clouds can preserve selected
   empirical mean/covariance moments and, before caps, selected diagonal or
   pairwise moments under feasibility conditions; this does not imply equality
   of probability measures or IID sampling.
2. **Cap-support proposition.** The implemented smooth radial and coordinate
   caps are differentiable and numerically bounded, but their global images are
   bounded. They cannot be used alone as a full-support global normalizing flow.
3. **PF-PF change-of-variables proposition.** For a genuinely invertible map,
   the post-flow proposal density and importance ratio are correct when the
   post-flow transition, pre-flow proposal, observation, and matching Jacobian
   are all evaluated. A post-hoc/non-invertible cap breaks this identity.
4. **Unnormalized block proposition.** Frozen known-density or proven SMC-U
   blocks give a conditionally unbiased estimator of the unnormalized target
   integral. Normalized terminal weights alone do not establish this result.
5. **AMIS/replay proposition.** Reusing old samples can be valid only when all
   retained samples are reweighted by the deterministic mixture of every
   proposal that generated them (or by a separately proved finite-window target).
6. **Defensive-mixture proposition.** A fixed positive full-support component
   supplies support and a second-moment bound conditional on an explicit
   integrability assumption; it does not guarantee finite variance by support
   alone.
7. **Tempering/mode proposition.** A bridge sequence with invariant mutation
   kernels gives a valid SMC route and can improve mode traversal, but finite
   local caps or finite mutation budgets do not prove mode discovery.
8. **No-go proposition.** Moment retention plus trust-region stability cannot
   imply density faithfulness, global mode coverage, or IID Gaussian whitening.
9. **Conditional solution proposition.** A viable future route is possible under
   explicit assumptions: full-support defensive proposals, exact/certified
   map-density evaluation, AMIS or unnormalized mass-carrying replay, tempering
   with mutation, and a separately valid batch-native NeuTra objective. The
   current artifacts do not yet establish those assumptions.

## Default and assumption audit

| Choice | Provenance | Why used | Failure mode | Early check | Status |
|---|---|---|---|---|---|
| Dual-cap GenUT | Owner integration note and local implementation | Controls local higher-moment correction and finite solves | Changes finite objective; caps alter tails | Post-cap moments and density audit | Candidate proposal mechanism |
| Trust-region LM | Local repair route and LM literature | Limits ill-conditioned moment updates | Suppresses needed bridge displacement | Cap activation and mode-transition diagnostics | Numerical hypothesis |
| Exact PF-PF correction | Li--Coates source and local monograph | Restores ratio for an invertible proposal map | Wrong determinant/covariance lifecycle | Affine known-map density identity | Required theorem assumption |
| Defensive mixture | Hesterberg | Prevents zero proposal density on covered support | Still infinite variance if heavy-tail integral diverges | Tail second-moment estimate | Candidate coverage mechanism |
| AMIS denominator | Cornuet et al. | Makes replay proposal-aware | Cost grows with proposal history; adaptive schedule conditions matter | Recompute all historical log densities | Candidate replay mechanism |
| Tempering/mutation | Neal; Del Moral--Doucet; Fearnhead--Taylor | Supplies a bridge between separated modes | Poor kernels leave modes isolated | Stage ESS and cross-mode transitions | Candidate mode mechanism |
| RQMC/SQMC | Gerber--Chopin and local SQMC code | Reduces integration variance under regularity | Does not create missing modes | Randomized replicate discrepancy | Variance-reduction arm only |
| GPU/XLA route | Repository owner policy | Required for serious NeuTra/LEDH training | Resource or backend mismatch | Memory-growth/device manifest | Execution requirement, not validity evidence |

## Skeptical plan audit (completed before execution)

| Audit question | Finding | Disposition |
|---|---|---|
| Is the proposed baseline the object under discussion? | The active dual-cap route is a finite reset/proposal component, not a complete density estimator. | State it as a proposal mechanism and compare against the fixed normalized replay only as context. |
| Could a proxy become a promotion criterion? | Whitening, ESS, and cap residuals are tempting shortcuts. | Keep them explanatory; require density identities and independent target checks. |
| Are stop conditions explicit? | A failed local cap can be a tuning failure, while a failed support identity is a theorem veto. | Separate implementation/tuning failure from research-direction failure. |
| Does a cap remain a valid flow? | The coordinate cap has bounded image; the radial cap also saturates. | Prohibit full-support flow claims unless a separate invertible chart and mixture density are supplied. |
| Is replay mathematically identified? | Existing q=20 blocks are normalized finite populations. | Require AMIS deterministic-mixture reweighting or an SMC-U proof before an unbiasedness claim. |
| Does local transport solve mode discovery? | No; caps bound local motion and seeded clouds can remain disconnected. | Add defensive full-support proposals and a tempered mutation lane. |
| Are comparisons fair? | A/B screen and prior LEDH artifacts use different targets/scopes. | Do not combine their numerical metrics; use local artifacts only for provenance/nonclaims. |
| Are literature claims source-grounded? | GenUT, Li--Coates, AMIS, defensive mixtures, SQMC, and ETPF have distinct targets. | Cite each only for the operation it actually supports. |
| Can the requested artifact answer the question without a long run? | The request is mathematical/literature analysis; a run would not prove the missing identities. | Perform static source inspection and MathDevMCP audit; defer implementation experiments to a new plan. |

The audit passes for documentary execution after these boundaries are made
explicit. No long numerical campaign is authorized by this plan.

## Execution record (2026-08-24)

The documentary execution completed with the following artifacts:

| Step | Artifact/outcome | Status |
|---|---|---|
| Local source and literature inspection | Primary technical papers copied under `.localresources/papers/ledh_replay_solution_20260824/`; local code anchors recorded in the note | completed |
| Proposition-proof note | `docs/plans/bayesfilter-ssl-lstm-q20-ledh-pfpf-genut-dual-cap-trust-region-mathematical-note-2026-08-24.md` | completed; conditional/no-go conclusion |
| Machine-auditable appendix | `docs/plans/bayesfilter-ssl-lstm-q20-ledh-pfpf-genut-dual-cap-trust-region-mathematical-note-2026-08-24.tex` and compiled PDF under `docs/plans/artifacts/ledh-pfpf-genut-literature-solution-20260824/latex/` | completed |
| MathDevMCP audit | `docs/plans/bayesfilter-ssl-lstm-q20-ledh-pfpf-genut-dual-cap-trust-region-mathdevmcp-audit-2026-08-24.md` plus JSON artifacts under `.../mathdevmcp/` | no contradiction found; substantive rows partly abstained |
| Fable review | `docs/plans/bayesfilter-ssl-lstm-q20-ledh-pfpf-genut-dual-cap-trust-region-fable-review-reply-2026-08-24.md` | received; written-claims `AGREE`; no blocking/major findings; implementation not authorized |
| Grok review and cross-review adjudication | `docs/plans/bayesfilter-ssl-lstm-q20-ledh-pfpf-genut-dual-cap-trust-region-grok-review-2026-08-24.md` and `...grok-adjudication-2026-08-24.md` | received; both reviewers agree on written claims; implementation alignment pending |

The MathDevMCP result is deliberately not treated as a blanket certification:
the Markdown and TeX deep passes found no findings, but their typed obligation
coverage was limited, and the isolated label pass marked the measure-theoretic
rows `unverified` or `inconclusive`. The narrow SymPy identities were
`equivalent`. This preserves the distinction between a tool abstention, a
checked algebraic identity, and a proof about the running code.

During the final proof pass, two implicit hypotheses were made explicit before
closeout: defensive-mixture coefficients obey
`0 < epsilon_min <= epsilon <= 1`, and every deterministic-mixture denominator
is positive on the target integrand support. The Markdown and TeX audits were
rerun after those repairs; the result remains no contradiction found with the
same typed-parser abstentions.

The independent documentary reviews are now complete. Grok and Fable both
return `AGREE` for the plan and mathematical note. Fable records two minor
derivational clarifications and three editorial repairs; none changes the
conditional/no-go conclusion. Both reviews explicitly withhold implementation,
default-route, replay, GPU, and NeuTra/HMC authorization. The next phase must
therefore begin with a separately planned per-proposal density identity and
affine known-map test, followed by the remaining replay, support, tail, and
mode gates.

## Execution sequence

1. Inspect the local source anchors and the technical sections of the primary
   papers listed above.
2. Write the proposition-proof mathematical note, including a literature table,
   explicit counterexamples, assumptions, and a conditional implementation
   route.
3. Run MathDevMCP doctor and a bounded applied-math/proof audit on the note;
   preserve the exact command and outcome. An `unverified` or `inconclusive`
   result is recorded as abstention, not converted into agreement.
4. Write the Fable handoff memo containing one bounded review path, the exact
   questions, source anchors, and the requested verdict format.
5. Run static checks (Markdown/LaTeX readability, referenced-path existence,
   and git diff review) and write a short audit record.

## Stop/repair rules

- Stop the documentary execution if a cited paper does not support the claimed
  operation; narrow the claim and record the omission.
- If MathDevMCP finds an algebraic error, repair the note and rerun the bounded
  audit before closeout.
- Do not implement a new sampler, alter defaults, launch GPU training, or admit
  HMC from this note. Those actions require a separate reviewed experiment plan
  with fresh scope-specific tuning and an evidence contract.

## Planned closeout decision

The expected outcome is **conditional solution, current implementation not yet
admissible**: dual-cap/trust-region can be retained as a numerically controlled
local proposal component, while replay validity and mode coverage require the
additional full-support deterministic-mixture/unnormalized and tempered-mutation
construction. If a proof obligation fails, the note will state the corresponding
no-go result rather than claim a repair.
