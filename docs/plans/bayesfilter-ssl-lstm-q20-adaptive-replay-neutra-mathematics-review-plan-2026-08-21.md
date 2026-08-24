# SSL-LSTM q=20 adaptive-replay NeuTra mathematics and review plan

Date: 2026-08-21
Status: `FABLE_REVIEW_ADJUDICATED_TEXTUAL_REPAIRS_APPLIED_GROK_ARTIFACT_MISSING`

## Research intent ledger

| Field | Declaration |
|---|---|
| Main question | Under what explicit assumptions can refreshed, proposal-aware replay combined with fresh NeuTra target queries learn a transport whose pullback is Gaussian, and what does that establish for the SSL-LSTM q=20 campaign? |
| Mechanism under test | Population-block target replay for a forward-KL term, plus fresh base draws and exact target evaluations for a reverse-KL term. |
| Exact baseline | The current fixed empirical loss on six weighted 100-particle SMC populations, reused for every update. |
| Expected failure mode | Arbitrary row eviction, stale or missing proposal densities, self-normalized finite-population bias, mode-weight distortion, no sign-crossing mutation, reverse-KL mode collapse, inadequate flow capacity, or optimizer failure. |
| Promotion criterion | A correct conditional theorem with all assumptions exposed, Fable-reviewed proof repairs, an explicit local-basin/symmetry boundary for the dense IAF, a derivation that distinguishes population, estimator, and finite-run claims, and a MathDevMCP audit that finds no unaddressed contradiction in the encoded obligations. |
| Promotion veto | Calling a finite-buffer estimator unbiased when it is not; claiming convergence with a fixed genealogically dependent SMC population; claiming the finite IAF contains an exact transport; or claiming that mathematical possibility establishes SSL-LSTM empirical success. |
| Continuation veto | The proposed replay estimator cannot be written with source-bound weights, or a claimed equality is refuted. |
| Repair trigger | MathDevMCP abstention or gap findings trigger narrower claims and explicit human-review obligations; they do not become proof failures automatically. |
| Explanatory diagnostics | Existing SMC ancestry, sign transitions, replay ESS, heldout losses, pullback moments, R-hat, and ESS. None is used as a theorem premise unless stated. |
| Must not be concluded | No finite-sample guarantee, no proof of mode discovery, no guarantee that the current dense IAF can represent the exact map, no global optimizer-convergence claim for its symmetric parameterization, no claim that (30) holds globally, and no HMC convergence claim. |

## Evidence contract

The claimed mathematical target is the normalized UKF-defined parameter
posterior `pi`, not the unavailable exact nonlinear-state likelihood. The
transport density is `q_phi = (T_phi)_# rho`, where `rho` is the standard
Gaussian base. The note must establish, conditionally:

1. the forward- and reverse-KL formulas and their gradients;
2. an unbiased unnormalized deterministic-mixture importance estimator for the
   forward-KL gradient when proposal densities are known;
3. the corresponding consistency-only statement for weighted SMC population
   blocks;
4. the conditions under which block refresh preserves the intended estimator;
5. nonnegativity and the common exact minimizer of the hybrid objective; and
6. exact Gaussian pullback and HMC invariance if an exact diffeomorphic
   transport is attained.

MathDevMCP is explanatory/audit evidence. A `diagnostic_only`, `gap_found`,
`not_encodable`, or `human_review_required` status is not a proof. The durable
artifacts will be:

- the mathematical note;
- the MathDevMCP audit ledger with exact commands and outputs summarized; and
- a read-only Fable handoff requesting an independent thorough review.

## Default and assumption audit

| Choice | Provenance | Status | Failure mode | Early check |
|---|---|---|---|---|
| Hybrid forward/reverse KL | Proposed repair motivated by the fixed-replay and historical reverse-KL failures | Hypothesis, not default | Weight choice changes optimization geometry and can still collapse or over-cover | Prove common exact minimizer for every strictly positive pair of coefficients; leave coefficient tuning empirical |
| Population-block replacement | Existing replay consists of separately normalized SMC populations | Reviewed mathematical requirement | Rowwise eviction destroys within-population normalization and can impose selection bias | Derive block estimator and contrast with arbitrary row deletion |
| Known-density proposal MIS | Standard importance identity | Mathematical baseline | Missing support or proposal-density error invalidates the estimator | State absolute-continuity and finite-moment assumptions explicitly |
| SMC replay | Existing q=20 artifacts | Warm-start evidence only | Self-normalization, ancestry, and finite particle count prevent an exact unbiasedness claim | State consistency under a standard SMC limit, not finite-N unbiasedness |
| Exact transport exists in model family | Idealized theorem premise | Unproven hypothesis | Dense IAF capacity/topology may exclude or poorly approximate it | Separate existence in a broad diffeomorphism class from membership in the implemented finite IAF |
| Refreshed buffer capacity and cadence | User-proposed experience-replay analogy | Unchosen target-specific controls | Fixed convenient values could dominate results | Do not assign numerical defaults in this mathematical note |
| Dense-IAF strong stability | Theorem 2 sufficient condition | Conditional local-basin hypothesis only | Hidden-unit permutation and tanh-sign symmetries create equivalent stationary copies | Restrict `C` to a symmetry-free basin or open a separately reviewed quotient/gauge-fixing route |
| Uniform estimator boundedness | Theorem 2/2A proof premise | Unverified campaign constraint | Proposal-ratio, target-score, and Jacobian tails may violate essential boundedness | Run disjoint support/tail/envelope diagnostics; downgrade to a separately proved finite-second-moment theorem if needed |

## Planned derivation

1. Define the target, base law, diffeomorphic transport, pushforward density,
   and pulled-back target.
2. Derive forward KL, reverse KL, and the hybrid objective.
3. Derive a proposal-history-aware deterministic-mixture importance estimator.
4. Prove its conditional unbiasedness for the unnormalized forward gradient.
5. State the normalized-SMC consistency boundary under an explicit SMC law of
   large numbers, and prove why rotating fixed-size normalized blocks does not
   by itself remove finite-particle bias. Do not reprove a general SMC theorem.
6. Formalize content-independent block refresh and show what it preserves.
7. Prove a continual adaptive-generation route in which fresh unbiased blocks
   carry the persistent gradient and stale replay has summable influence.
8. Prove the hybrid exact-minimizer theorem and Gaussian pullback consequence.
9. Give counterexamples for arbitrary eviction, missing proposal support,
   fixed finite replay, and pure reverse-KL mode seeking.
10. Translate the result into an implementable but not-yet-authorized q=20
   protocol and empirical vetoes.

## MathDevMCP audit commands

Use the installed executable
`/home/ubuntu/anaconda3/envs/tfgpu/bin/mathdevmcp`. Run `doctor`, then bounded
`assumptions-for`, `debug-derivation`, `prove-or-counterexample`,
`classify-math-claim`, and `prepare-review-packet` queries. Record exact command
text and returned status. Do not claim machine certification for measure-theory
steps that the tool cannot encode.

## Skeptical pre-execution audit

| Audit question | Finding |
|---|---|
| Wrong baseline? | No. The note compares against the actual full-batch fixed weighted loss in `WeightedForwardKLNeuTraTrainer`, not an imagined unweighted buffer. |
| Proxy promoted to criterion? | No. MathDevMCP output and replay diagnostics are audit evidence only; the deliverable is a checked conditional derivation. |
| Missing stop condition? | No. A refuted equality or inability to bind proposal weights stops the positive theorem and requires a counterexample/negative result. |
| Unfair comparison? | No empirical method ranking is planned. Paper-faithful reverse KL, refreshed forward replay, and the hybrid are classified separately. |
| Hidden assumptions? | Support, differentiability, integrability, proposal evaluation, SMC consistency, flow expressivity, and optimization are explicit proof obligations. |
| Stale context? | The active q=20 runner and weighted trainer were inspected on 2026-08-21; no historical plan is treated as executable authority. |
| Environment mismatch? | This is a CPU/read-only mathematical audit. No TensorFlow, GPU, training, HMC, or package mutation is required. |
| Do artifacts answer the question? | Yes, within the conditional scope. They can establish mathematical viability and failure boundaries, but not finite SSL-LSTM success. |

Audit verdict: `PASS_FOR_DOCUMENTATION_AND_BOUNDED_MATH_AUDIT`.

## Execution steps

1. Write the mathematical note under `docs/plans`.
2. Run the bounded MathDevMCP commands and write the audit ledger.
3. Revise any overclaim exposed by the audit.
4. Write a single-primary-path, read-only handoff memo to Fable requesting a
   theorem, estimator, counterexample, and SSL-LSTM claim-boundary review.
5. Verify links, ASCII/Markdown structure, and touched-file scope. Do not launch
   a training or HMC campaign under this plan.

## Execution result

The mathematical note, bounded MathDevMCP audit ledger, and read-only Fable
handoff were created on 2026-08-21. MathDevMCP certified only the scoped
deterministic-mixture algebra; higher-level integral and stochastic-
approximation obligations require human review. No training, HMC, package
mutation, or GPU process was launched. Final document and worktree-scope checks
are recorded in the audit ledger and repository command history.

## Fable adjudication and amendment

On 2026-08-23, Fable returned `AGREE` for the conditional estimator,
replay-convergence, Gaussianization, and claim-boundary verdicts. The review
did not establish runtime applicability. Its symmetry finding shows that the
strong stability condition (30) is impossible on any admissible set containing
two distinct hidden-unit permutation or tanh-sign copies of a stationary point;
the dense-IAF theorem is therefore local-basin or quotient-parameterization
conditional. The note was revised to add this boundary, the base-side
differentiation envelope, the Poisson-Lipschitz coupling derivation, the
pathwise remainder bound, deterministic `lambda_t`, the finite-N witness, the
joint-C1 Rosenblatt condition, and the assumption-realism gate.

The adjudication and revised execution contract are recorded in
`docs/plans/bayesfilter-ssl-lstm-q20-adaptive-replay-neutra-review-adjudication-plan-2026-08-23.md`.
The user-supplied Grok analysis path was not present in the checkout, so no
Grok conclusion is silently merged into this plan.

The next authorized work is a documentation consistency check and then a new
reviewed route preflight. No training or HMC run is authorized by this
amendment. Constant-positive-weight stale replay from an evolving proposal
remains outside the proved routes; only fresh adaptive blocks with summable
stale influence are currently covered.
