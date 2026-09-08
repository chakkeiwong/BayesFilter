# Direct-Factor SR-UKF Remaining-Gaps Closure and Hypothesis Plan

Date: 2026-08-17  
Scope: the six `blocked` rows in the superseding direct-factor inventory, plus
the deliberate analytical/numerical gaps recorded by the preceding closure
campaign.  SSL-LSTM remains excluded by the owner instruction.

## 1. Objective and evidence boundary

The objective is to determine, with reproducible tests, which remaining gaps
can be closed without changing the statistical target and which require a new
reviewed contract or remain mathematically unresolved.  This is not a plan to
turn every repository model into an SR-UKF.  An SR-UKF is admissible only for a
model with a fixed additive-Gaussian state/observation contract, a declared
sigma-point rule, a declared parameter chart, and a value/score program whose
measure and derivatives are explicit.

The active direct-factor route remains the block-QR recursion:

\[
 A^-_t = [F_tG_{t-1}\;G_{q,t}],\qquad
 (A^-_t)^\mathsf T = Q_tR_t,\qquad G^-_t=R_t^\mathsf T,
\]

and, for an observation residual stack, the joint factor

\[
 J_t=\begin{bmatrix} S_t & 0\\ C_t & G_{r,t}\end{bmatrix},\qquad
 J_t^\mathsf T=Q_tR_t,
\]

with the conditional filtered factor and gain read from the corresponding
blocks of `R`.  No temporal covariance matrix is formed.  A rectangular factor
`G\in\mathbb R^{n\times r}` is allowed for value calculations when
`P=GG^\mathsf T` is positive semidefinite.  A rank-changing branch is not an
analytical-score chart: it must return value-only status or fail closed.

The campaign may close an adapter, but it may not claim exact nonlinear
Bayesian filtering, universal SR-UKF applicability, posterior correctness,
HMC readiness, GPU production readiness, or score validity through a pivot,
rank, repeated-singular-value, or angular-branch event without separate
evidence.

## 2. Current inventory and classification

| Row | Current blocker | Classification for this plan | Permitted action |
|---|---|---|---|
| `SVX-SGQF` | no frozen SGQF level passed admission | known target-family repair; not SR-UKF | rerun the existing SGQF level ladder only; preserve blocked if no level passes |
| `KSC-UKF` | dense-reference/filter admission failed historically | bounded numerical repair hypothesis | test the mass-preserving Gaussian-sum UKF route at frozen T20 and then require fresh scope-specific tuning; do not equate it with exact SV or direct-factor SR-UKF |
| `PP-ZC` | no batch-native posterior adapter/chart admitted | source-route/target-contract hypothesis | test whether the existing fixed-variant Zhao–Cui route can expose a batch-native value/score contract without changing its source semantics; no SR-UKF promotion |
| `STR-ZC` | extension target absent | unknown extension hypothesis | design-only probe and registry guard; implementation requires a new reviewed target derivation |
| `SIR-ZC` | observed-data parameter-score closure absent | partially known route gap | test existing observed-data score components for finite same-program derivatives and explicitly check target identity; do not use latent/teacher score as closure |
| `SVX-ZC` | same-program score exists but XLA/HMC capability is not admitted | bounded numerical/capability repair hypothesis | test eager/XLA parity, own finite differences, rank/floor telemetry, then retain blocked unless capability and fresh tuning gates pass |

The non-blocked classifications are not reopened: multiplicative SV,
domain-constrained SIR, PP/SIR-SGQF, the generic covariance-provider adapter,
legacy actual-SV panel, SIR-UKF, and SSL-LSTM remain respectively
`not_applicable_contract`, `historical_only`, or `owner_excluded`.

## 3. Hypotheses and falsification tests

### H1: an admitted KSC approximation can be made operational without target drift

**Claim under test.** The deterministic mass-preserving Gaussian-sum UKF
implementation can retain all normalized component mass, produce finite value
and same-program score on the frozen KSC surrogate, and agree with the dense
reference/finite-difference gates at the declared T20 scope.

**Tests.** Use audit points `[-1,-1],[-1,1],[0,0],[1,-1],[1,1]` and the frozen
truth point, float64, component caps `7,16,32,64,128,256`, permutation reversal,
and the existing dense orders `401,601`. Check finite value/score, normalized
mass, positive component variances, cap-to-reference value and score gaps, and
permutation invariance. A passing diagnostic is not a direct-factor SR-UKF
claim; it only permits a separately scoped KSC Gaussian-sum target admission.
Failure of any dense, score, mass, or permutation gate falsifies H1 for that
cap and leaves `KSC-UKF` blocked.

### H2: the exact-SV fixed-SGQF ladder has a reproducible admitted level

**Claim under test.** One candidate sparse level can satisfy the existing
prefix dense-value, prefix finite-difference score, full-level convergence,
and status gates at the frozen dataset/horizon.

**Tests.** Run the repository SGQF admission script with candidate levels
`10,12,16,20,24` and reference `32`, retaining the exact observation/state
hashes, XLA setting, memory policy, and per-level diagnostics. If no candidate
passes, the hypothesis is falsified for this scope and the row remains blocked;
switching to KSC or a UKF is prohibited.

### H3: PP-ZC can expose a batch-native target without becoming a UKF

**Claim under test.** A fixed-variant Zhao–Cui predator-prey route can provide
batch-native value and score with a frozen chart/Jacobian and source anchors,
while preserving the author's transition/observation semantics.

**Tests.** First require paper/source anchors and a stable target signature.
Then test batched shape, finite value/score, own finite differences, XLA parity,
and independent value/score recomposition. Compare only to the same fixed
Zhao–Cui target, never to the SR-UKF likelihood. If no source-faithful
batch-native adapter exists, record `extension_or_invention` and falsify H3 as a
closure; preserve the row blocked.

### H4: STR-ZC is closable by reusing the structural UKF initializer

**Claim under test.** The structural UKF factor geometry can be used as a
frozen initializer for a Zhao–Cui target without changing the target program.

**Tests.** Require a concrete target program, parameter chart, observed-data
score, and source operation anchors before coding. A design-only probe checks
that the current registry has no such target and that no initializer hash can
be mistaken for a target identity. The hypothesis is falsified (and the row
stays blocked) until all four artifacts exist and pass finite-difference,
batch/XLA, and source-faithfulness tests.

### H5: SIR-ZC's existing observed-data lane supplies the missing score

**Claim under test.** The Austria SIR observed-data route can be bound to the
same target value program as its score, with no latent-teacher substitution,
and can pass batch/XLA/finite-difference checks.

**Tests.** Exercise the smallest source-anchored observed-data prefix first,
then T1/T2 where available. Check value-score recomposition, parameter order,
boundary/clip status, finite differences, and XLA parity. Teacher, latent, or
proposal scores are explicitly negative controls. A mismatch in target identity
or measure falsifies H5; no SIR-ZC admission follows.

### H6: SVX-ZC can be admitted as an XLA/HMC-capable same-program diagnostic

**Claim under test.** The existing fixed adjacent-state squared-TT program has
stable value and score across eager and XLA execution on its frozen T10 scope,
with finite differences, no rank/floor veto, and a reproducible capability
identity.

**Tests.** Use the existing adapter at a three-row batch and at the truth point;
check all status fields, own finite differences at two step sizes, eager/XLA
parity, batch permutation, and initializer/core hashes. Then run only the
scope-specific offline tuning/readiness gates. A finite score alone does not
set `xla_hmc_ready`; the capability flag is promoted only by the repository
admission factory after all gates pass. Prior self-convergence or one-seed HMC
artifacts are negative controls, not admission evidence.

### H7: singular/rank-changing analytical scores can be made globally valid

**Claim under test.** One score formula remains valid through a rank transition,
repeated singular values, QR pivot/sign changes, or an angular branch cut.

**Expected outcome.** This is a deliberately skeptical hypothesis. Matrix
factor gauges are only locally differentiable on fixed-rank/fixed-pivot/sign
charts; a singular Gaussian changes its reference measure, and circular means
are undefined at zero resultant. The falsification suite therefore checks that
the implementation returns finite value-only output plus an explicit invalid
score status at those events. A test that produces a finite autodiff number is
not evidence of mathematical validity.

## 4. Execution phases

1. **Freeze and review.** Record the inventory checksum, source anchors, exact
   commands, seed, dtype, device, XLA and memory-policy settings. Review this
   plan for target separation, mathematical measure, numerical gates, and
   testing coverage before execution.
2. **Implement guards and probes.** Add tests for explicit blocked-cell
   hypotheses, source/target separation, KSC mass/score diagnostics, SVX-ZC
   capability non-promotion, and singular/rank/branch value-only behavior.
   Add a bounded diagnostic runner that writes a fresh versioned artifact root
   and never edits the registry automatically.
3. **Run the campaign.** Run CPU diagnostics with GPU hidden only for small
   reference tests. Run existing GPU/XLA admission scripts only when the
   required visible GPU and memory-growth policy are available. The campaign
   budget is one attempt per hypothesis plus one localized repair retry; no
   training or HMC is launched by this plan.
4. **Decide each row.** A row may move to an admitted status only through its
   repository-owned route factory and a fresh artifact with all declared gates.
   Otherwise preserve `blocked` and record the falsified hypothesis or the
   missing contract.
5. **Document.** Write per-hypothesis JSON, a Markdown result memo, a
   superseding inventory (if any status changes are justified), and update the
   LaTeX survey with the result/nonclaim boundary. Compile the survey twice.
6. **Verify.** Run focused unit/integration tests, JSON/schema checks, source
   checksum checks, and the existing direct-factor regression suite. Report
   warnings and unexecuted GPU/HMC gates explicitly.

## 5. Promotion and stop rules

- Never promote a SGQF, Zhao–Cui, KSC surrogate, or latent/teacher route as a
  direct-factor SR-UKF adapter without a reviewed contract change.
- Never promote a finite value-only singular route to an analytical score row.
- Any nonfinite value/score, failed dense or finite-difference gate, target hash
  drift, XLA mismatch, rank/pivot/branch veto, or stale tuning artifact is a
  hard failure for that hypothesis.
- A clean negative result is useful: it closes the question for the tested
  scope and records the exact next derivation or contract needed for reentry.

## 6. Required artifacts

Use the fresh root
`docs/plans/artifacts/direct-factor-srukf-remaining-gaps-hypotheses-20260817/`.
It must contain `campaign_manifest.json`, one result JSON per hypothesis,
`coverage_summary.json`, `result.md`, `commands_and_environment.md`, and
SHA-256 checksums. Existing historical roots are read-only evidence.

