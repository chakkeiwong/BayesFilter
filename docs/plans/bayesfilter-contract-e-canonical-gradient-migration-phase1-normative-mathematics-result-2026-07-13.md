# Phase 1 Result: Normative Contract E Mathematics And Evidence Design

Date: 2026-07-13

Status: `PASSED_CLOSED_WITH_EXPLICIT_LATER_PHASE_PROMOTION_BLOCKERS`

Program ID: `contract-e-canonical-gradient-migration-20260713`

## Outcome

Phase 1 uniquely specifies the owner-selected finite Contract E--Chol reset and
its complete local pullback. The fixed-ridge manual VJP matches TensorFlow
autodiff and directional central differences on deterministic float64 tiny
fixtures, including direct source-moment, direct weight, residual-design, dense-
transport, and ridge adjoints. The row-quotient JVP/VJP and normalized-weight to
logit pullback were checked independently.

The phase did **not** derive scientifically defensible production tolerances for
finite Sinkhorn convergence, quotient conditioning, raw ridge bias, GPU kernel
backward error, chunk drift, or five-seed LGSSM equivalence. Earlier arbitrary
floating-point multipliers were removed. Each unresolved adequacy question is an
explicit blocker before the later phase that first consumes it for promotion;
none blocks Phase 2 schema-only work.

## Claimed Target And Quantity Checked

The claimed target is the deterministic finite filter conditional on prepared
inputs and fixed randomness:

```text
proposal -> LEDH flow -> corrected normalized weights
-> current likelihood increment -> finite positive transport row quotient
-> fixed-residual, fixed-ridge Contract E-Chol reset
-> next-time equal-weight cloud.
```

The checked quantity is the local derivative of the fixed-ridge Contract E reset
and its required coordinate composition. It is equal to TensorFlow autodiff and
central-directional derivatives on the checked smooth tiny chart. It is not yet
the implemented full-filter production derivative.

## Mathematical Verdicts

| Claim | Verdict | Evidence |
| --- | --- | --- |
| Likelihood increment precedes reset | `correct` for the canonical target | Normative specification, Section 1 |
| Covariance convention is population `1/N` | `correct` | Specification Sections 3 and 6 plus focused tests |
| Production transport output is `Y_i=Q_i/M_i` | `correct`; numerator-only is wrong relative to the target | Specification Section 4 and quotient duality test |
| Contract E pullback is `Y+`-only | `wrong relative to the target` | Direct moment/weight VJP derivation and nonzero-path tests |
| Canonical source derivative is direct moments plus transport | `correct` | Specification Section 8 |
| Canonical weight derivative may mix probability/log coordinates directly | `wrong relative to the target` | Section 2/8 normalization pullback and logit test |
| Nonzero-ridge reset exactly restores raw covariance | `wrong relative to the target` | Exact residual `lambda*(I-AA^T)` and test |
| Nonzero-ridge reset exactly satisfies the ridged identity | `correct` in exact arithmetic; production backward error not checked | Specification Section 7 and float64 test |
| Stopped candidate-dependent adaptive ridge is a total derivative | `wrong relative to the target` | Fixed-input route definition |
| `0.05*sqrt(p)` is a 95% confidence rule | `wrong`; it is owner-directed heuristic-only FD screening | Design freeze |
| Five-seed Bonferroni intervals unconditionally establish exact 95% simultaneous coverage | `unsupported` | Exact at-least-95% statement requires stated iid-normal marginal model; otherwise nominal |

## Review And Repair Loop

| Iteration | Verdict | Material repair |
| --- | --- | --- |
| 1 | `REVISE` | Removed unjustified `8/32/64/256` hard thresholds; blocked quotient/ridge/conditioning/chunk adequacy; made FD heuristic explicit; corrected equivalence and coverage language; completed VJP including ridge. |
| 2 | `REVISE` | Corrected Bonferroni “exact” to “at least”; made representable FD endpoint, actual-ratio Richardson, coarsest, run, tie, and even selection rules executable. |
| 3 | `AGREE` | No material findings; prior blockers remain explicit and Phase 2 schema-only work may proceed. |

Claude was not retried because the platform already blocked repository
disclosure. Fresh bounded Codex reviewers were used as the approved read-only
substitute.

## Checks

- JSON design parse: passed.
- CPU-hidden focused suite: `14 passed in 3.91s`.
- Python compile for helper/test: passed.
- `git diff --check`: passed at the pre-close checkpoint.
- Review convergence: `VERDICT: AGREE` on iteration 3.

The CPU choice was deliberate and recorded in the run manifest. No GPU, XLA,
TF32, HMC, nonlinear, or leaderboard command ran in Phase 1.

## Decision Table

| Decision | Primary criterion status | Veto diagnostic status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Close normative mathematics | Pass | No algebra/coordinate veto fired | Tiny chart only | Begin schema-v2/factory Phase 2 | Production implementation correctness |
| Admit canonical artifact | Not eligible | V1 remains revoked; v2 absent | Production callables and evidence absent | Keep public factory inert until later phases | Admission/default/HMC readiness |
| Accept production numerical gates | Blocked | Unjustified constants removed | Executed kernels and downstream budgets | Resolve at Phases 3--5 before each promotion use | GPU numerical adequacy |
| Accept LGSSM equivalence | Blocked | Nonfinite remains hard veto | Five-seed model, gradient margins | Freeze stronger design/margins before Phase 8 results | Kalman-gradient equivalence |

## Inference-Status Table

| Inference | Status |
| --- | --- |
| Hard veto screen | Tiny fixed-ridge chart finite; Cholesky factors existed; quotient masses were positive. This is local only. |
| Statistically supported ranking | None; no stochastic candidate comparison ran. |
| Descriptive-only differences | None relevant; test residuals were pass/fail engineering checks. |
| Default-readiness | Not established; all v1 routes remain historical and no v2 production route exists. |
| Next evidence needed | Factory-bound schema, production cloud module, streaming composition, one-graph FD, trusted GPU feasibility, and reviewed LGSSM equivalence design/results. |

## Blocker Handoff

| Blocker | Must be resolved before |
| --- | --- |
| Residual centering, mean, ridged-identity kernel error limits | Phase 3 promotion result |
| Raw ridge-bias and conditioning requirement plus ridge domain adequacy | Phase 3 promotion result |
| Row-mass conditioning and row/column finite-Sinkhorn convergence | Phase 4 promotion result |
| Chunk forward/gradient accumulation error budget | Phase 4 promotion result |
| Normalized-weight kernel error and callable endpoint/score FD error bounds | Phase 5 promotion result |
| LGSSM normal-model justification or stronger pre-result replication design | Phase 8 equivalence result |
| LGSSM gradient relative and near-zero margins | Before Phase 8 material results |

## Required Artifacts

- Normative specification:
  `docs/plans/bayesfilter-contract-e-canonical-gradient-migration-phase1-normative-mathematics-spec-2026-07-13.md`.
- Design freeze:
  `docs/plans/bayesfilter-contract-e-canonical-gradient-migration-phase1-numerical-statistical-design-freeze-2026-07-13.json`.
- Focused tests: `tests/test_contract_e_phase1_normative_math.py`.
- Diagnostic helper repair: `docs/benchmarks/contract_e_reset_tf.py`.
- Check log and run manifest under
  `docs/plans/logs/contract-e-canonical-gradient-migration-2026-07-13/phase1/`.
- Three review records under `docs/reviews/`.
- Phase 2 subplan:
  `docs/plans/bayesfilter-contract-e-canonical-gradient-migration-phase2-schema-v2-factory-subplan-2026-07-13.md`.

## Post-Run Red Team

Strongest alternative explanation: the manual VJP and autodiff could share the
same helper-level conceptual mistake. Independent directional derivatives and
matrix identities reduce that risk on the tiny chart, but only later dense/cloud
and full-composition tests discriminate implementation wiring errors.

What would overturn the close decision: a derivation error in the normative
adjoints, a reproducible mismatch on another valid fixed-ridge chart, or evidence
that the stated finite target is internally inconsistent.

Weakest evidence: production numerical adequacy and the five-seed statistical
design. Both are explicitly blocked, not treated as passed.

## Handoff

Phase 2 may begin after its drafted subplan passes a bounded consistency and
boundary-safety review. Phase 2 is limited to identity/schema mechanics; it may
not issue an admitted canonical artifact or claim that Contract E is implemented.
