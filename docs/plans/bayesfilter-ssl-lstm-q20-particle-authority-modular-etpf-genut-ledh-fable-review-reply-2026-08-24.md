# SSL-LSTM q=20 Particle Authority and Modular ETPF/GenUT/LEDH Plan: Fable Review Reply

Review date: 2026-08-25 (handoff dated 2026-08-24)

Status: `READ_ONLY_BOUNDED_REVIEW_COMPLETE`

Auditor: Fable (claude-fable-5), independent reviewer

Handoff request:
`docs/plans/bayesfilter-ssl-lstm-q20-particle-authority-modular-etpf-genut-ledh-fable-handoff-2026-08-24.md`

Audited artifact (primary):
`docs/plans/bayesfilter-ssl-lstm-q20-particle-authority-modular-etpf-genut-ledh-plan-2026-08-24.md`

## 1. Scope statement and inspected state

Read-only inspection at commit
`14e4618749c9e04e8c4d2398becadb0206b30599` (the plan and its cited anchors are
untracked/uncommitted files on top of that commit; the exact file contents
reviewed are those present on 2026-08-25). Inspected beyond the primary path,
strictly because the plan's source ledger cites them:

- Acevedo--de Wiljes--Reich local text, lines `59-64`, `170-256`, `268-320`,
  `355-391`, `639-646` (all read directly);
- Li--Coates local text, lines `140-179`, `390-457` (read; `267-327` covered
  in the prior 2026-08-24 review, verdict carried forward);
- Ebeigbe GenUT local text, lines `24-26`, `165-181` (read; `114-164` covered
  in the prior review);
- prior replay note
  `bayesfilter-ssl-lstm-q20-adaptive-replay-neutra-mathematical-note-2026-08-21.md`,
  lines `318-335` (SMC-U contract, eq. (19)) and `741-759`;
- the Acevedo PDF checksum: recomputed SHA-256 equals the plan's stated
  `3e729ca967486163dd0cbdfde90baaedcc6ef76c1df111bad4550b831ebc80e1` — match.

Arithmetic checks performed: the phase budget table sums to the stated
`64800 s` cap (1800 + 7200 + 14400 + 18000 + 18000 + 5400). The carried
numerical evidence values (`3/100000`, latent separation `23.7`, covariance
residual `11.6264`--`24.8663`) were **not** re-derived from their source
result notes; the plan cites them with descriptive-only labels, which is the
correct evidence class, so their exact values are `not checked` here and
nothing in this review depends on them.

No file was edited except this reply. No command that samples, trains, or
touches a GPU was run. No agent was launched.

## 2. Verdicts

- **Plan verdict: VERDICT: AGREE.**
- **Mathematical-contract verdict: VERDICT: AGREE**, with two `minor` repairs
  recommended below (F1, F2). Neither is blocking, because the plan's own hard
  vetoes ("stale frozen-law hashes", "missing density terms") already
  fail-close the ambiguous cases; the repairs remove the ambiguity rather than
  change any decision rule.

The hybrid decision is coherent as posed. The authority change and the
modular-arm separation are logically independent and the plan keeps them
independent: withdrawing authority from the six-bank normalized replay is
justified by the already-reviewed ratio-bias classification (prior note,
Proposition 6 lineage) and does not presuppose that any of M1--M4 works;
conversely, each arm's admissibility is defined by its own contract against
M0, not against the withdrawn replay. Specifically:

- **Finite-cloud covariance matching is not promoted to density or mode
  correctness.** The `finite_moment_transform` tag, the promotion veto list,
  and the M1 row's "no IID claim" hold that line. The Acevedo anchors support
  every ETPF statement made: second-order accuracy is defined as agreement of
  the equal-weight analysis mean/covariance with the importance-sampling
  estimates (source eqs. 11--13, class `D2` in eq. 26); the correction is a
  Riccati-equation route (source eqs. 42--44); and the source explicitly
  reports second-order-accurate transformed samples leaving the prior range
  and "violating Bayes' law" (lines 639--646). The plan's claim that the
  second-order correction can produce negative transform entries is in fact
  understated relative to the source, which states `D1+ ∩ D2 = ∅`
  generically (see F3).
- **The contracts are stated with the right assumptions.** The flow contract
  is the correct change-of-variables pair; Li--Coates Lemma IV.3/IV.4
  (step-size condition `ε_j < 2r(λ_j)/(p̄h̄²)`, determinant product, source
  eq. 33) supports "invertible under step/regularity conditions", and the
  plan correctly places any later reset outside the determinant. The
  defensive-mixture block carries `0 < ε_min ≤ ε_t ≤ 1`, the support
  condition, and the second-moment check demanded by Hesterberg's boundary.
  The M0 fixture identity is the same conditional-unbiasedness contract as
  the prior note's eq. (19) at the cited lines, correctly presented as an
  obligation, not a property of any existing runner.
- **The baseline ladder and rules answer the q=20 question.** C0 (naive/tuned
  classical fresh SMC) → M0 (plain proof-bearing candidate) → M1--M3
  (one-factor) → M5 (enhanced, gated behind components) satisfies the
  baseline-ladder policy; M4 is correctly quarantined as an approximate-filter
  comparator. Promotion vetoes are not silently upgradable to continuation
  vetoes: the continuation veto is limited to common-support failure or a
  source-identity contradiction on an exact fixture, and poor covariance/ESS
  is named a repair trigger.
- **Numeric choices carry provenance.** The N ladder (measured historical
  `100`/`600`, proposed `300`), 2+4 seed policy, Sinkhorn grid `[0.1, 1, 10]`,
  batch `64` warm start, and the 18-hour cap are each labeled hypothesis,
  warm start, calibration-only, or cap — none is presented as a reviewed
  default. The budget table is arithmetically consistent and excludes HMC.
- **Prior-review continuity is faithful.** The three evidence gaps carried
  forward (per-proposal density identity, defensive-tail second moment,
  replay metadata parity) match the 2026-08-24 Fable reply exactly.

## 3. Findings ordered by severity

No `blocking` or `major` findings.

### F1. Adaptive tempering vs the frozen-protocol M0 identity

| Field | Content |
|---|---|
| Location | Default/assumption audit row "Adaptive tempering"; Phase 2 first paragraph; "Mathematical role boundaries," M0 fixture identity |
| Severity | `minor` |
| Classification | `project_derivation` |
| Claim checked | M0's fixture identity `E[gamma_hat_t(f) | frozen SMC protocol] = ∫ tilde_pi_t f` can be claimed for the executed authority route |
| Reason | Phase 2 requires "a predeclared tempering schedule selected on calibration data," but the audit row promotes "adaptive tempering" to avoid a fixed beta schedule. If the beta schedule (or ESS-triggered resampling) is adapted online from the claim-run particles, the protocol is not frozen and the conditional-unbiasedness identity does not automatically apply — the same class of adaptivity caveat the Cornuet source states for AMIS and the prior note states after its Proposition 7. The plan currently contains both readings. |
| Repair | State explicitly: adaptivity is confined to calibration partitions; every claim-bearing M0 run executes a schedule (stages, triggers, mutation controls) frozen and hashed before its draws; otherwise the M0 fixture must test the actual adaptive protocol, and until it does, an adaptively-tempered run is a C0-class descriptive comparator, not an authority candidate. |

### F2. ETPF constraint block conflates definition and constraint

| Field | Content |
|---|---|
| Location | "Mathematical role boundaries," first code block and the sentence "The second-order ETPF constraints require the last two rows to agree" |
| Severity | `minor` |
| Classification | `source_faithful` |
| Claim checked | The displayed equations state the Acevedo first/second-order LETF constraints |
| Reason | As written, `bar_y = sum_i w_i x_i` and `Cov(nu_hat_D) = sum_i w_i (x_i - bar_y)(x_i - bar_y)^T` read as definitions, and "the last two rows agree" has no second member to agree with. The source's constraints (eqs. 11--13, 18--20, 26) are: equal-weight transformed mean equals the weighted forecast mean, `(1/N) Σ_j y_j = Σ_i w_i x_i`, and equal-weight transformed covariance equals the weighted forecast covariance, `(1/N) Σ_j (y_j − ȳ)(y_j − ȳ)^T = Σ_i w_i (x_i − x̄)(x_i − x̄)^T`, under the column/row-sum conventions `D^T 1 = 1`, `D 1 = N w`. |
| Repair | Replace the block with the two labeled constraint equations above (left side: transformed equal-weight moments; right side: weighted forecast moments) and add the two `D` sum conventions. No decision rule changes. |

### F3. The negative-entry claim can cite the source's stronger structural statement

| Field | Content |
|---|---|
| Location | "Mathematical role boundaries," sentence "the second-order correction can allow negative entries and leave the forecast range" |
| Severity | `editorial` |
| Classification | `source_faithful` |
| Claim checked | Second-order accuracy is generically incompatible with nonnegative (convex-combination) transforms |
| Reason | The source states `D1+ ∩ D2 = ∅` generically (text near line 269) — i.e., leaving the forecast convex hull is structurally forced by exact second-order accuracy, not an occasional numerical artifact. Citing this strengthens the plan's bridge-artifact risk row from "can happen" to "generically must be possible," which is the correct severity for the Phase 1 two-mode fixture. |
| Repair | Optional: add the `D1+ ∩ D2 = ∅` anchor beside the sentence. |

### F4. Historical-metric values are cited but not rederivable from this plan

| Field | Content |
|---|---|
| Location | "Evidence carried forward" bullets |
| Severity | `editorial` |
| Classification | `project_derivation` |
| Claim checked | `3/100000`, `23.7`, `11.6264`--`24.8663` |
| Reason | These are quoted from the cited result notes with correct descriptive-only labels; this bounded review did not re-derive them (`not checked`). No plan decision depends on their exact values, only on their sign/magnitude class, so this is a provenance note, not a defect. |
| Repair | None required. |

## 4. Decision table

| Item | Status |
|---|---|
| Decision | Plan approved for documentary purposes; both verdicts AGREE with F1/F2 repairs recommended before the implementation-phase plan is drafted |
| Authority status | Six-bank normalized replay: authority correctly withdrawn (descriptive-only context). M0 fresh tempered SMC-U: candidate authority only; its eq.-(19)-class identity is an unmet obligation for a runner that does not yet exist |
| Modular-arm status | M1 (`finite_moment_transform`), M2 (`sigma_point_quadrature`), M3 (eligible for exact PF-PF status only via the affine identity and determinant lifecycle), M4 (approximate-filter comparator), M5 (gated behind component arms) — all correctly scoped; none promotable by covariance/whitening/ESS/loss alone |
| Continuation vetoes | Correctly narrow: common-support failure of the fresh authority, or a source-identity contradiction on an exact fixture. Poor proxy metrics are repair triggers. No silent promotion-to-continuation upgrade found |
| Main uncertainty | Whether the implemented M0 route (especially with any adaptivity, F1) satisfies the frozen-protocol identity; and whether fresh proposals/mutation reach the negative-half-space mode at feasible particle counts |
| What is not concluded | No arm ranking, no posterior correctness, no mode-coverage guarantee, no HMC readiness, no default change — matching the plan's own nonconclusions |

## 5. Unestablished assumptions (as requested by the handoff)

Current q=20 artifacts establish none of the following; each is an obligation
the plan correctly assigns to a future phase:

1. The SMC-U conditional-unbiasedness identity for the implemented authority
   route (no runner exists; Phase 0/1 obligation).
2. Mutation-kernel invariance for every bridge target actually used.
3. A concrete `r_safe` with a finite score-class second moment for the q=20
   forward score class.
4. Replay metadata sufficiency: recomputable historical log densities for any
   retained block (parity test still `not checked`).
5. LEDH invertibility for the q=20 model under the actual step schedule
   (Li--Coates step condition is model-dependent through `p̄`, `h̄`, `r`).
6. Any density statement for the ETPF/GenUT transformed clouds (none is
   claimed; the tags must survive implementation).
7. Mode reachability of the negative half-space by fresh proposals plus
   mutation at the proposed particle counts.

## 6. Cheapest next artifact

Phase 0 item 3 plus the first two Phase 1 fixtures, in this order:

1. the machine-checkable M0--M4 contract stubs that fail closed (no science,
   pure wiring);
2. the one-dimensional affine-flow PF-PF density identity (discriminates the
   M3 contract for pennies);
3. the M0 known-density unnormalized-mass fixture on a tractable target with
   the schedule-freezing rule from F1 written into the fixture itself — this
   is the single cheapest artifact that can falsify the authority hypothesis
   rather than a proxy.

## 7. Non-authorization statement

This reply is a documentary review only. It authorizes no runtime execution of
any kind: no sampler implementation run, no fixture run, no particle
generation, no replay replacement, no GPU or XLA work, no NeuTra training, no
HMC, and no default or policy change. Runtime work requires the separate
implementation-phase plan the reviewed plan itself demands, under the
repository's evidence-contract, per-scope tuning, GPU memory-growth, and
batch-native training rules.
