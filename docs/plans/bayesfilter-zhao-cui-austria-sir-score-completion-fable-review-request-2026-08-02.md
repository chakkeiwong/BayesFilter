# Fable Review Request: Zhao-Cui Austria SIR Score Completion Plan

Date: 2026-08-02

Review role: read-only mathematical, scientific, and implementation-plan
review. Fable is an independent reviewer; Codex remains executor and final
integration authority.

## Primary Request

Review exactly this plan first:

```text
docs/plans/bayesfilter-zhao-cui-austria-sir-score-completion-plan-2026-08-02.md
```

Question: Does the plan define a mathematically correct, source-honest,
statistically adequate, and feasible route to close the Austria `T=20` finite
score computation, without upgrading the current T1/T2 finite-difference
mechanics into claims they do not support?

Do not edit files, run experiments, launch agents, or review the whole repo.
You may inspect only the exact supporting paths and source anchors listed below
when needed to answer a numbered question. Report what you inspected and what
you did not inspect. End with one of:

```text
VERDICT: AGREE
VERDICT: AGREE-WITH-CHANGES
VERDICT: REVISE
```

A material mathematical, target, source-classification, numerical,
statistical, XLA, measure, tail-derivative, or budget defect is blocking.
Formatting preferences and superseded launch ceremony are advisory only.

## Why Review Is Needed

The current active result reports finite T1/T2 origin scores, but both are
centered finite-difference estimates and T2 adds a first-core radial projection
that enforces scalar consistency by construction. Nothing beyond T2 or away
from `theta=0` is admitted. The proposed plan changes the derivative
architecture: it uses offline Zhao-Cui-derived proposal artifacts but computes
the runtime score with a manual normalized-weight recursion on a literal
theta-independent frozen branch.

It also changes the finite scalar. The current T1/T2 values are trained-TT
normalizers; the proposed value is a frozen importance-filter likelihood
estimate. This is disclosed in the plan and requires a new route/row. Fable
must decide whether that replacement is an acceptable answer to “complete the
Zhao-Cui Austria score,” or whether the task requires differentiation of the
trained-TT normalizer instead. Do not blur these objects because both use TT
artifacts.

This is a material mathematical and scientific choice. Review must test the
claimed scalar/derivative equality, the boundary between source-faithful
operations and repository invention, the proposal/measure and tail contracts,
and whether the evidence ladder can support the stated terminal status.

## Minimum Ground Truth

### Active state

1. `docs/plans/bayesfilter-zhao-cui-austria-sir-material-replay-xla-repair-result-2026-08-02.md`
2. `docs/plans/bayesfilter-zhao-cui-austria-sir-material-replay-xla-reset-memo-2026-08-02.md`
3. `docs/plans/bayesfilter-zhao-cui-austria-sir-t2-scalar-consistency-repair-note-2026-08-02.md`

### Passed value/proposal evidence

4. `docs/plans/bayesfilter-zhao-cui-austria-sir-lane-b-t2-result-2026-07-31.md`
5. `docs/plans/bayesfilter-zhao-cui-austria-sir-lane-b-b2-sampler-result-2026-07-31.md`
6. `docs/plans/bayesfilter-zhao-cui-austria-sir-conditional-reference-t1-result-2026-08-01.md`

### Negative evidence that the plan must not erase

7. `docs/plans/bayesfilter-zhao-cui-austria-sir-observed-data-score-result-2026-07-30.md`
8. `docs/plans/bayesfilter-zhao-cui-austria-sir-measure-bridge-blocker-result-2026-08-01.md`
9. `docs/plans/bayesfilter-zhao-cui-austria-sir-parameter-density-t1-reset-memo-2026-08-01.md`

### Paper and author source

10. `.localresources/papers/zhao-cui-tensor-train-sequential-learning-jmlr-2024.txt`
11. `third_party/audit/zhao_cui_tensor_ssm_p10/source/models/full_sol.m`
12. `third_party/audit/zhao_cui_tensor_ssm_p10/source/eg3_sir/mainscript.m`
13. `third_party/audit/zhao_cui_tensor_ssm_p10/source/deep-tensor.dev/src/@TTSIRT/marginalise.m`
14. `third_party/audit/zhao_cui_tensor_ssm_p10/source/deep-tensor.dev/src/@TTSIRT/eval_irt_reference.m`

### Local model and proposed reusable mechanics

15. `bayesfilter/highdim/models.py` (`ParameterizedZhaoCuiSIRSSM` and
    `parameterized_zhao_cui_sir_austria_model` only)
16. `bayesfilter/highdim/sir_latent_preclip_tf.py`
17. `bayesfilter/highdim/zhao_cui_austria_sir_lane_b_sampler_tf.py`
18. `docs/plans/bayesfilter-zhao-cui-austria-sir-observed-data-score-implementation-handoff-2026-07-30.md`

Start with the plan alone. Open supporting paths only for the question they
answer. This is a bounded review packet, not a request for a repository audit.

## Required Review Questions

### A. Current Verdict And Gap Inventory

1. Does the plan correctly characterize the current T1/T2 result as local
   finite-difference mechanics at `theta=0`, rather than a completed score
   route?
2. Does it preserve the important negative evidence: T2 radial projection,
   historical APF proposal-quality failure, measure-bridge tail failure,
   missing later horizons, and absent HMC evidence?
3. Is any claimed current gap already closed by an artifact listed above, or
   is any material gap omitted?
4. Is “finite-program score completion” defined narrowly enough that a pass
   cannot be mistaken for exact physical likelihood or posterior correctness?
   Is the target-replacement disclosure prominent and complete enough that the
   new row cannot inherit the old T1/T2 values or status?

For each answer, classify `correct`, `wrong relative to the stated target`,
`unsupported`, `not checked`, or `heuristic only`.

### B. Source-Faithfulness Boundary

5. Verify from the paper that general Algorithm 2 uses the adjacent target
   `pi_hat(x_{t-1},theta) f_theta g_theta`, squared-TT approximation, and
   marginal carry.
6. Verify from Section 6.3 that the Austria experiment fixes kappa and nu and
   performs 18-dimensional state inference, and from `eg3_sir/mainscript.m`
   that the author route sets `d=0`.
7. Is the plan therefore correct to reject the label “source-faithful
   Zhao-Cui Austria parameter score” while allowing source-faithful
   classifications for individual target/marginal/KR operations?
8. Is `fixed_hmc_adaptation` plus `extension_or_invention` the right route
   classification for a frozen proposal evaluated over the repository's
   external three-parameter surface?
9. Does any cited paper or author-source anchor actually derive the manual
   external likelihood score? If not, confirm it remains a project derivation.
10. Identify any source anchor in the plan that is wrong, insufficient, or
    supports less than claimed.

Do not issue `AGREE` on source status without inspecting both the paper anchors
and the author files.

### C. Finite Scalar And Total-Derivative Derivation

11. Independently derive the normalized-weight score recursion for the exact
    scalar written in the plan. Check initial marks, per-time marks,
    normalized-weight derivative carry, increment derivative, and total
    additivity.
12. Is `D_{t-1}^{A_t^i}` the correct carried quantity for fixed ancestry, or
    is an unnormalized/path score required instead?
13. Under literal theta-independent samples, maps, ancestry, proposal
    densities, auxiliary probabilities, shifts, and Jacobians, are all their
    derivatives exactly zero for this finite scalar?
14. List every term that becomes nonzero if any part of the proposal or branch
    is theta-dependent. Does the plan reliably fail closed rather than hide
    those terms behind stopped gradients?
15. Check the proposed multi-branch scalar. Is the likelihood-weighted branch
    score the derivative of the log-mean-exp scalar? Confirm that plain score
    averaging would be wrong.
16. Does the manual recursion compute the derivative of the finite importance
    scalar actually reported, or has the plan silently changed probability
    measure, conditioning event, or baseline?
17. Can this manual route validly replace differentiation of the TT optimizer
    for the stated finite-program claim? Explain precisely what is gained and
    what is no longer claimed.
    Separately decide whether target replacement is acceptable for the user's
    completion request. If not, specify the smallest viable plan for the
    trained-TT-normalizer derivative instead.

Any missing term or scalar/derivative mismatch is a blocking finding.

### D. Score Tolerance And Independent Checks

18. Is the plan correct not to reuse a pure relative five-significant-digit
    rule for near-zero scores?
19. Are the proposed initial hypotheses `score_atol=5e-6` and
    `score_rtol=5e-5` defensible as calibration starting points, or are they
    too loose/tight relative to the expected FP64/FP32 and Monte Carlo errors?
20. Does step halving at `h`, `h/2`, and `h/4` adequately diagnose FD
    truncation/rounding for the bounded tests? What additional diagnostic is
    essential?
21. Are omitted-term mutants a valid way to show the gate is discriminating?
    Propose the minimum mutant set if the plan's set is incomplete.
22. Are `GradientTape` and FD correctly restricted to independent diagnostic
    roles, with the runtime score remaining manual?
23. Does the plan distinguish deterministic same-finite-scalar error from
    Monte Carlo/reference uncertainty clearly enough?

### E. Proposal, Measure, And Tail Closure

24. Can the passed T1 exact interval-mass sampler legitimately serve as a warm
    start for a staged frozen proposal chain, given its finite-law rather than
    author-CDF classification?
25. Does the plan accidentally treat high proposal ESS as score correctness?
26. Is sequential per-horizon tuning required by the repository policy and by
    the scientific problem, or can any scopes safely share a frozen tuning
    artifact?
27. The historical full 36D APF proposal failed T1 ESS, while the later
    retained sampler passed a small T1 correction screen. Does the plan
    adequately distinguish these laws/scopes, or does it assume the latter
    repairs the former?
28. What proposal-quality veto should be frozen before claim execution? Assess
    whether ESS/entropy/tail diagnostics can be hard vetoes without becoming
    claims of method superiority.
29. Examine the T2 signed-log value repair. Is the plan right that a zero-value
    certificate does not automatically establish a zero derivative
    contribution? State the correct limiting/finite-program analysis needed.
30. Are support, clipping, latent-preclip representation, Jacobian, and
    ancestor-law requirements sufficient to prevent a wrong-measure score?
31. Identify the earliest cheap test that would show T20 proposal closure is
    infeasible, so the campaign does not spend the full budget on a doomed
    ladder.

### F. Parameter Domain And Statistical Evidence

32. Is the historical `[-0.5,0.5]^3` box properly downgraded to a calibration
    hypothesis rather than an HMC prior/default, and is the predeclared nested
    half-width ladder `{0.03,0.10,0.25,0.50}` defensible?
33. Is the proposed axis/corner/mixed/untouched design adequate to support a
    compact-domain score claim? Recommend a smaller or stronger design if
    necessary.
34. Can domain selection on calibration points followed by untouched interior
    points avoid post-selection bias for the stated claim?
35. Is at least two frozen branches meaningful evidence, or should the minimum
    branch/seed count be higher? Tie the recommendation to a predeclared
    uncertainty calculation rather than a generic replication preference.
36. What statistical evidence is required before comparing the T20 score with
    GenUT, SGQF, UKF, or an independent importance authority?
37. Is the plan sufficiently direct that passing means only viability and
    same-program correctness, not superiority?

### G. XLA, TensorFlow, And Runtime Architecture

38. Is it feasible to express the entire manual score recursion with batched
    TensorFlow and XLA `tf.while_loop`, with no NumPy import in new claim-owned
    code and no Python numerical loops?
39. Which existing local operations are most likely to introduce `PyFunc`,
    eager decisions, unsupported dynamic shapes, or sample-wise mapping?
40. Are the graph-inspection, CPU-hidden parity, trusted-GPU smoke, XLA default,
    and 6,144 MiB logical-device-cap gates proportionate and sufficient?
41. Does compiling the runtime recursion avoid the higher-order
    `XlaDynamicUpdateSlice` gradient failure that blocked optimizer JVP? Is any
    hidden higher-order differentiation still required?
42. Should runtime use FP64 or FP32/TF32 for the terminal claim? The plan
    currently treats FP64 as reference and FP32/TF32 as production target;
    assess whether score precision requires an explicit FP64 GPU confirmation
    arm.
43. Identify any Python loop allowed by the plan that would actually influence
    numerical results and violate the XLA contract. In particular, verify that
    the existing Python RK4-substep loop cannot remain in the new claim path.

### H. Evidence Ladder, Budget, And Stop Logic

44. Are T1, T2, T3, T5, T10, T20 the right discriminating horizons? Could a
    different sequence expose failure earlier at less cost?
45. Does every phase produce an artifact that answers its question, or does
    any phase rely on a proxy as promotion evidence?
46. Are continuation vetoes distinguished correctly from candidate rejection
    and repair triggers?
47. Is the total attempt/GPU-hour budget proportionate to the uncertainty and
    likely proposal difficulty? Recommend exact changes if not.
48. Is one infrastructure-only retry for T20 adequate and appropriately
    distinguished from scientific retuning?
49. Are the terminal statuses sufficiently precise, including the possibility
    of a valid finite score but inadequate physical-likelihood evidence?
50. Does the definition of done close every gap it claims to close without
    smuggling HMC, production, default, posterior, or source-faithfulness
    claims into the result?
    Also check whether the three proposed runner interfaces and `conda run -p`
    command shapes are sufficiently explicit and reproducible without
    authorizing package mutation.

### I. Silent Defaults And Hostile Review

51. List every material default or convenience choice the plan has not fully
    justified. At minimum inspect domain, N=1008, two branches, tolerance,
    horizon ladder, ESS veto, rank/basis, branch-combination rule, dtype, and
    memory cap.
52. For each, classify it as baseline, warm start, hypothesis, convenience
    choice, reviewed default, or unsupported.
53. Ask what could make every command pass while the scientific conclusion is
    still wrong. Does the pre-mortem catch the strongest cases?
54. Identify the single strongest alternative plan architecture. In
    particular, compare the selected frozen-branch score with a joint
    `(x_t,theta,x_{t-1})` TT followed by parameter-marginal differentiation.
    Would that alternative close more of the source gap, and at what
    feasibility/target cost?
55. State whether the selected architecture is the simplest defensible path
    to the user's requested score computation, or whether the plan should be
    revised before execution.

## Required Deliverable

Write one review note at:

```text
docs/plans/bayesfilter-zhao-cui-austria-sir-score-completion-fable-review-2026-08-XX.md
```

It must contain:

1. An executive verdict and blocking findings first.
2. Answers to A1-I55, grouped by section. Concise answers may share a table,
   but no numbered question may be omitted.
3. For every technical verdict, one of: `correct`, `wrong relative to the
   stated target`, `unsupported`, `not checked`, or `heuristic only`.
4. Exact paper, author-code, local-code, or artifact anchors for checked
   claims.
5. An independent derivation of the scalar score recursion and multi-branch
   combination.
6. A route-classification verdict: whole route, each source-grounded
   operation, and the external parameter-score extension.
7. A table of silent defaults with recommended disposition.
8. A table separating hard veto evidence, descriptive evidence, and
   statistical evidence.
9. A statement of what was exhaustively checked, sampled, and not checked.
10. Exact required plan changes, separated from optional suggestions.
11. A final verdict line in the required format.

If the verdict is `AGREE-WITH-CHANGES` or `REVISE`, quote the smallest exact
replacement text or mathematical correction needed for every blocking item.
Do not approve on internal consistency alone: source-status approval requires
the listed paper and author-code inspection, and mathematical approval
requires re-deriving the finite scalar score.

## Non-Goals For The Review

- Do not execute the plan or benchmark the GPU.
- Do not review unrelated Zhao-Cui, LEDH, NeuTra, GenUT, SGQF, UKF, or HMC
  campaigns.
- Do not demand exact-zero or bitwise core replay; the active material rule is
  intentional and gauge-invariant.
- Do not require retired approval tokens, hash-bound natural-language claims,
  or multi-review ceremony.
- Do not treat review disagreement about file naming or prose style as a
  scientific blocker.
- Do not authorize T20 or HMC execution. The review evaluates the plan only.
