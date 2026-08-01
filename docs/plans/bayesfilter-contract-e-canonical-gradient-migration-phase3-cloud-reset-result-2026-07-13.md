# Phase 3 Result: Contract E Cloud Reset

Date: 2026-07-13

Program ID: `contract-e-canonical-gradient-migration-20260713`

Status:
`EXACT_ENGINEERING_CERTIFICATE_PASSED_GENERAL_PARITY_AND_PROMOTION_BLOCKED`

## Outcome

The repository now has an owned TensorFlow Contract E-Chol cloud primitive with
XLA-on public forward, manual JVP, and manual VJP APIs. It consumes source
particles, normalized probabilities, an already transported cloud, a realized
fixed residual design, and a prepared fixed ridge. It accepts no dense transport
matrix and contains no NumPy, autodiff, adaptive ridge, `rho`, eigendecomposition,
or explicit inverse.

The implementation is **correct on the frozen bounded exact charts** relative
to the Phase 1 finite program: all five input JVPs/VJPs, dense composition,
duality, a noncommuting factor chart, a nonzero transported-covariance branch,
and a two-batch chart pass bitwise certificate checks. The frozen general chart
is finite and reproducible, but general numerical parity is **not checked to an
adequate criterion** because no justified general forward-error bound exists.

The production-reset promotion claim is **unsupported**. Six pre-result
numerical/scientific requirements remain unresolved, so the production factory
stays empty and no v2 artifact is issued.

## Claimed And Computed Quantities

| Item | Classification |
| --- | --- |
| Claimed Phase 3 target | Fixed-`rho=1`, fixed-`Xi`, fixed-ridge Contract E-Chol cloud reset and its five-input local derivative |
| Quantity computed | Frozen exact-chart forward/JVP/VJP/dense-composition certificates plus one descriptive general chart |
| Equality status | Correct on the checked exact charts; general-chart adequacy not checked |
| Full transport/filter total gradient | Not computed; the transported-cloud adjoint is intentionally uncomposed until Phase 4 |
| Production numerical adequacy | Unsupported while the six blockers below remain |
| Evidence anchors | Phase 1 normative specification, three Phase 3 JSON artifacts, focused tests, manifest, and review records |

## Repair Loop

| Iteration | Verdict | Repair |
| --- | --- | --- |
| Implementation 1 | `REVISE` | Added exact noncommuting factors and nonsymmetric affine coverage. |
| Implementation 2 | `REVISE` | Added an exact nonzero transported-covariance branch activating `plus_cov -> gap_chol`. |
| Implementation 3 | `AGREE` | No material formula defect remained. |
| Closeout 1 | `REVISE` | Added shared dense intermediates, certificate integrity/exactness checks, `B=2`, explicit ridged-identity scale, and recomputable persisted diagnostics. |

Claude was not retried because the platform repository-disclosure boundary had
already blocked that route. Fresh bounded Codex reviewers were the authorized
substitute.

## Checks

- Phase 3 focused suite: `16 passed, 2 warnings`.
- Final Phase 0-3 compatibility suite: `134 passed, 2 warnings in 9.93s`.
- Deliberate CPU-XLA public-wrapper smoke: passed inside the focused suite.
- Python compilation, three JSON parses, certificate hashes/exactness checks,
  reference hashes, and scoped `git diff --check`: passed.
- No GPU, HMC, full filter, nonlinear, leaderboard, or long command ran.

## Six Unresolved Promotion Blockers

| Blocker | Status |
| --- | --- |
| Residual-design centering error requirement | Unresolved promotion blocker |
| Mean-restoration error requirement | Unresolved promotion blocker |
| Executed-kernel ridged-identity backward-error requirement | Unresolved promotion blocker |
| Raw ridge-bias scientific requirement | Unresolved promotion blocker |
| Conditioning/downstream-error budget | Unresolved promotion blocker |
| Ridge magnitude/domain adequacy | Unresolved promotion blocker |

The observed general-chart residuals cannot be used to choose these boundaries
after the fact. In particular, the raw covariance residual is descriptive and
reflects the documented fixed-ridge identity; its scientific acceptability is
not established.

## Decision Table

| Decision | Primary criterion status | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Close bounded cloud implementation | Pass | No exact-chart, formula, XLA-smoke, or owned-module boundary veto remains | General kernel accuracy | Begin reviewed Phase 4 engineering composition | General numerical parity |
| Pass Phase 3 production-reset evidence gate | Blocked | Six promotion blockers unresolved | Ridge bias and executed-kernel/downstream adequacy | Preserve blockers; do not register route | Production reset admission |
| Treat cloud VJP as full gradient | Ineligible | Transported-cloud adjoint is uncomposed | Row quotient and transport coordinates | Compose in Phase 4 | Full filter total derivative |
| Issue v2/default/HMC artifact | Ineligible | Production factory remains empty | Phases 4-9 remain | Keep fail closed | Default, HMC, or leaderboard readiness |

## Inference-Status Table

| Inference | Status |
| --- | --- |
| Hard veto screen | Frozen exact charts are finite; Cholesky diagonals are positive; exact identities, autodiff checks, duality, and dense composition pass. |
| Statistically supported ranking | None; no stochastic comparison ran. |
| Descriptive-only differences | General-chart absolute/ULP differences, residuals, raw covariance bias, and condition proxies. |
| Default-readiness | Not established; no production route is registered. |
| Next evidence needed | Row-quotient streaming composition, dense/stream parity, then independently justified feasibility and numerical adequacy gates. |

## Artifacts

- Owned implementation: `bayesfilter/highdim/ledh_contract_e_reset_tf.py`.
- Focused tests: `tests/highdim/test_ledh_contract_e_cloud_reset_phase3.py`.
- Exact certificates under the Phase 3 program prefix.
- Parity diagnostics:
  `docs/plans/bayesfilter-contract-e-canonical-gradient-migration-phase3-parity-diagnostics-2026-07-13.json`.
- Check log and manifest under
  `docs/plans/logs/contract-e-canonical-gradient-migration-2026-07-13/phase3/`.
- Four Phase 3 review records under `docs/reviews/`.

## Phase 4 Handoff

Phase 4 may proceed as engineering repair/composition work only. The smallest
streaming design appends a constant-one feature to the particle payload so one
existing streaming pass produces numerator `Q` and row mass `M`, with the same
JVP/VJP yielding `dQ,dM` and accepting `barQ,barM`. The quotient is then applied
without a floor, and the cloud reset's direct probability-weight adjoint is
converted before addition to the streaming normalized-log-weight adjoint.

## Post-Run Red Team

Strongest alternative explanation: exact charts can miss general kernel error.
That is why the result does not claim general parity and preserves the six
promotion blockers.

What would overturn this close: a reproducible exact-chart mismatch, a missing
input adjoint, a dense allocation in the owned module, or source drift without
artifact regeneration.

Weakest evidence: production numerical adequacy. It was deliberately not
inferred from the small observed residuals.
