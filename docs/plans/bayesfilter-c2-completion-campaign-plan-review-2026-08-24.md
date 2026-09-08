# C2 Completion Campaign Plan — Independent Material Review — 2026-08-24

Target: `bayesfilter-c2-completion-campaign-plan-2026-08-24.md`
(status `DRAFT_FOR_REVIEW`). Scope: the six assigned audit items plus
the reviewer's own sweep. Classifications per the repo plain-language
rule. Finding namespace `CF#` (the `F#` namespace belongs to the C2
derivation-note review).

Sources inspected (not narrated from memory):

- The plan itself, whole.
- `bayesfilter-c2-gaussian-reference-derivation-note-2026-08-24.md`
  (REVIEWED spec), §§1–3b, incl. §3 declared bar 2.5e-3 nats/step and
  τ_max := bar/25; §3b engine findings F-ENG-1..3, rung-4b redefinition.
- `bayesfilter-c2-derivation-note-review-2026-08-24.md` (F1–F7 and
  discharges; the F1 tail-variance derivation is reused below).
- `bayesfilter-squared-tt-program-reset-memo-2026-08-24.md` (D1–D3,
  UPDATE 2 engine-block state, environment directive).
- `bayesfilter-p1b-attempt04-plan-2026-08-21.md` and
  `...-attempt04-result-2026-08-22.md` (tolerance 2.5e-3/step; runner
  pattern; the 9-axis XLA compile-timeout record).
- `bayesfilter/highdim/squared_tt_engine_gaussian_tf.py` (row-law
  weights multiplied across axes at :157; `_clamped_tau` at :188;
  fit path) and `squared_tt_engine_v0_tf.py:204` (fit_diag contents:
  `worst_condition`, `weighted_fit_rms` — no ESS emitted).
- `docs/benchmarks/check_c2_hermite_rowlaw_mechanism_20260824.py`
  (the Gram/ESS evidence is a SINGLE-AXIS ℓ=13 measurement).
- `tests/highdim/test_c2_gaussian_engine_oracle.py` (certified suite
  tops out at n=2 / 4-plus-branch axes: closure n=1, degree-0 T=120 at
  n=2, stress rung n=2 ℓ=13 r=6 T=12).

## Findings by severity

### CF1 (major, blocking) — the accuracy bar is circular in Phase C and
contradicted by the plan's own reviewed spec

Gate D states "this campaign plan does not pre-commit any accuracy bar"
(plan :128), yet: (i) C3's reference validity veto is "MC error exceeds
bar/5" and the §3 table sizes reference particle N by the "MC error ≤
bar/5 rule" — both executed in Phase C, before D1 exists, so Phase C is
ill-posed without a bar; (ii) the REVIEWED derivation note §3 already
declares the question at 2.5e-3 nats/step, attempt04 ran the same
2.5e-3/step tolerance, and the reviewed Class-C clamp value
τ_max = 1e-4 := bar/25 is derived from that bar. The bar is therefore
already program-committed; "not pre-committed" is wrong relative to the
governing spec, and a D1 re-declaration would silently detach the
reviewed τ_max provenance. Repair (small): declare the working bar =
2.5e-3 nats/step inherited from the reviewed note (naming the per-step
aggregation, total gap/T as attempt04 read it), use it for C3 sizing,
and state that any D1 bar change re-opens τ_max = bar/25 and the C3
sizing rather than being free.

### CF2 (major, blocking) — no n=4-scale engine shakedown before D; the
reviewed ladder's item 5 is silently dropped

The reviewed note's §2 validation ladder is titled "before any C2
claim" and its item 5 is a conditioning diagnostic at n ∈ {4, 8}. The
campaign plan's phases contain no n>2 gate anywhere before attempt05:
Gate A's two parity configs and the certified suite all sit at n ≤ 2.
This matters concretely, not generically:

- Row-law dimension scaling. The F-ENG-1 evidence (Gram 1.32,
  ESS 1400/2048) is a single-axis ℓ=13 measurement (mechanism script);
  the engine applies the half-mixture weights as a product across all
  axes (`squared_tt_engine_gaussian_tf.py:157`). For independent axes
  the population identity ESS_d/N = (ESS₁/N)^d applies:
  0.68^d gives ≈ 0.21·N at d=4 (the certified n=2 scale — passed) but
  ≈ 0.03–0.05·N ≈ 65–95 rows at d = 8–9, the n=4 claim scale —
  approaching the "fits starved" regime the note itself recorded
  (ESS 22). Estimate, not measurement — which is the point: it is a
  minutes-scale pre-D measurement, and the plan schedules it only as
  "Gram/ESS diagnostics logged per fit" during attempt05, where
  starvation would masquerade as rank infeasibility. The §4 pre-mortem
  names fitter floor and hint quality as infeasibility masquerades but
  not row-law starvation — the one with an unexamined exponential
  dimension dependence.
- Compile scale. attempt04 measured XLA compile timeouts precisely at
  rank-10 9-axis graphs (n=4). Gate A's compile battery at n=2
  (5 axes) does not observe that regime; discovering it at D burns
  claim-run budget.
- Row count N is absent from the §3 audit table although it must scale
  with axis count if the ESS estimate above is even directionally
  right — a material unexamined default (item 4).

Repair (small, cheap): reinstate ladder item 5 in post-F-ENG-1 form as
a pre-D gate — one n=4 LGSSM rung on the ported lane (degree ≥ 1, e.g.
the stress config at n=4) recording Gram condition, ESS, fitter-floor
gap against the Kalman oracle, compile time, and eager-vs-XLA parity in
one artifact; add a "row count N vs axes" row to the audit table; and
emit per-fit ESS (weights are computed at :157–158 and discarded;
fit_diag carries none — Class-A adopt-by-default, and the audit table's
own early diagnostic depends on it).

### CF3 (major, blocking) — the claim lane's parity evidence excludes
the claim configuration

Gate A's two parity configs are both LGSSM with λ ≡ 1. attempt05 runs
SV kernels + Student-t floor + SV hints inside the XLA hot path (model
kernels are inside the fit-target evaluation, as in the attempt04
port). No gate anywhere in the plan requires eager-vs-jitted parity or
a fresh-process compile battery for that configuration: Gate C has no
XLA item, and D1 is not required to add one. The plan's own
continuation veto treats "C2 parity fixtures cannot be made green" as
engine invalidity, yet the parity fixture family never covers the
configuration the claim runs use. Per the Implementation Audit
Call-Chain Rule, executable parity on the claim-bearing configuration
is the required evidence; attempt04's own discipline was "parity-gated
... before any ladder cell runs". Repair (small): add a post-C, pre-D
gate item — eager-vs-XLA parity ≤ 1e-12 plus compile battery on an
SV + Student-t + hints config (can be the same rung as CF2's shakedown
once the SV pieces exist, or a separate small config).

### CF4 (material, blocking) — C3 reference validity is variance-only
and under-specified for the campaign's comparator

The MC-error veto controls replicate spread; it does not by itself
control particle log-likelihood bias, and the §4 pre-mortem claims the
veto plus the n=1 anchor counter the under-resolved-reference risk.
The honest statement: E[log Ẑ] − log Z ≈ −Var(log Ẑ)/2 in the
CLT/lognormal regime, so once replicate std ≤ bar/5 the bias is
≈ bar²/50 ≈ 1.3e-7 — negligible — but exactly this argument fails in
the weight-degeneracy regime (heavy-tailed Ẑ; few-replicate sample std
underestimates; the bias formula is invalid), which is the regime an
n=4 particle reference actually risks and which the n=1 anchor (an
implementation check, not a regime check) cannot exclude. Also
unspecified: the replicate count and the estimator ("MC error" = std of
what, pooled-mean SE over R replicates?), and "veto, attempt04 style"
misattributes provenance — attempt04 had an exact Kalman reference and
no such rule; bar/5 is new here and owes its one-line justification.
Repair (a few sentences in C3): state the bias-variance argument and
its degeneracy failure mode; add a reference-PF ESS/degeneracy
diagnostic and a minimum replicate count to the validity gate (Class B,
adopt-by-default); define the error estimator; give bar/5 its
provenance line. Optional strengthener, not required: an n=2
dense-grid cross-check is feasible (2-dim state) and would anchor the
particle machinery beyond n=1.

### CF5 (material, blocking) — "ν by domination-margin derivation" is
under-determined as stated

For the SV tail class established by finding F1, log F(u) = αu²/2 +
O(u) along the whitened +x_c ray with α = 1 − s²/σ_f² ∈ (0, 1). For a
product-t floor, log λ_μ(u) = u²/2 − ((ν+1)/2)·log(1 + u²/ν) + const
per axis, so log(F/λ_μ) = −(1−α)u²/2 + ((ν+1)/2)·log(1+u²/ν) + O(u)
→ −∞ for EVERY finite ν. Domination sup F/λ < ∞ therefore holds for
all ν and cannot select one; the margin
log M(ν, α) (maximized at u*² = (ν+1)/(1−α) − ν) is monotone
increasing in ν, so a margin-only criterion drives ν → smallest —
exactly the "too heavy: bulk dilution" failure the plan's own audit
table names. ν is a Class-C-adjacent shape parameter (it alters the
represented density); the derivation is the plan's justification
mechanism, and as specified it cannot produce one. Two secondary
gaps: the derivation depends on α, i.e. on the hint whitening variance
s², but C4 (hints) runs after C2 — the derivation must be parametric
in the declared hint class with the D1 run gate instantiating actual
hints; and Gate C's criterion "domination pre-check derivation
written" is a completion checkbox, not an acceptance criterion.
Repair (small): C2 must declare the two-sided criterion — a required
margin cap M(ν, α_declared) ≤ declared value (tied to its consumer,
the ratio guard at τ ≥ τ_min) AND a bulk-retention floor (e.g. λ_μ on
the whitened bulk relative to 1) — parametric in s², with the LGSSM
no-fire check as the no-harm side; Gate C gates on that criterion, not
on "written". The framing in §1 ("domination margin inadequate →
declared repair: Student-t ν adjustment") should also match §2 C2's
correct posture that Student-t is the expected SV configuration
(F1), with ν adjustment as the within-scope repair.

Correctly carried from the prior review (verified): Student-t as the
expected SV configuration, not a contingency; the retained-floor-term
coverage assigned to the C2 derivation section (earlier than the F1
text required — good); domination pre-check as a run gate at D1;
"never tuned on claim data". The closed-form claims check: ∫λ dμ = 1
and per-axis retention marginals are exact for the product-t as
μ-density, and the defensive-corrected oracle gate is λ-independent,
so the F7 gate survives the escalation. The previous-block direction
of the retained t-floor is plausibly benign (polynomial tails are
affine-stable, unlike Gaussian variance mismatch) but that is the C2
derivation's to prove, as the plan already requires.

### CF6 (moderate) — row-law status overstated; C1 is fetch-only

The §3 table calls the half-mixture row law "reviewed default
(post-review amendment)". Wrong relative to the stated status: the
material review predates §3b; the row law was adopted afterwards under
campaign-repair rules as a measured engineering repair, its cited
basis (Cohen & Migliorati 2017) is not yet fetched or read (C1 exists
to fetch it), and the implemented object is a defensive HALF-mixture
modification of the cited law (the pure law starved fits, ESS 22), so
the citation does not automatically cover it. Repair: relabel the
status (measured repair, single-axis evidence, citation pending); make
C1 "fetch AND read the stability sections, reconcile the half-mixture
variant (bounded per-axis weights ≤ 2, hence ≤ 2^d after product)
against the paper's conditions, record deltas". The scaling half of
this risk is CF2's.

### CF7 (moderate) — one continuation veto is over-broad

§1 lists "the C2 parity or adjoint fixtures cannot be made green
(engine invalid)" as a campaign continuation veto, while §2 declares
Phase B non-blocking for D. An adjoint fixture failure blocks the
score-path deliverable (later HMC/MLE), not the value-side r*(n)
claim: the value engine is independently oracle-certified. As written
this is a promotion-scope failure upgraded to a campaign stop — the
exact misclassification the Research Question Guardian rule forbids.
Repair: split the veto (value-side parity fixtures → continuation;
adjoint fixture → blocks Phase B's deliverable only).

### Minor findings

- CF8: the domination pre-check appears both as a promotion veto (§1)
  and a repair trigger (§1 failure modes). Dual roles are allowed only
  when declared explicitly — add the one clause.
- CF9: Gate C evidence (reference sizing, hints, ν margin) is scoped
  to a pinned SV parameterization and horizon, which C3/C4 must
  consume before D1 "audits fixture params". State that C3/C4 pin the
  paper's values now and that a D1 parameter change re-opens Gate C
  (per-scope logic, LEDH-style), otherwise the D1 audit can silently
  invalidate Phase C evidence.
- CF10: reference-building compute (long-particle N sized to bar/5
  with replicates at n=4) shares the 12 h budget but is unsized —
  add a pre-mortem line; state the reference backend/dtype (NumPy is
  permitted for independent references per the backend rule; f64
  required at bar/5 precision) in the C3 artifact.
- CF11: A1/A2 GPU measurements must record the verified memory-growth
  policy in their manifest per the TensorFlow GPU Memory Rule; the
  plan names the env directive but not this manifest field.

## Assigned audit items — verdicts

1. Phase ordering and gates: two load-bearing gaps. Gate A's two
   parity configs are sufficient in degree/rank structure but
   insufficient in axis scale (no 9-axis compile/parity observation —
   CF2) and in kernel coverage (no SV/Student-t parity anywhere —
   CF3); the reviewed ladder's n ∈ {4, 8} item 5 is dropped (CF2).
   B non-blocking for D is correct (ALS value fits need no score
   path). Gate A's XLA-unreachable fallback correctly follows the TF
   policy's documented-exception route.
2. Reference ladder: design sound in outline, under-specified at the
   points that carry the terminal claim — the bar/5 veto is not
   well-posed in Phase C because the bar is deferred to D1 (CF1), and
   validity is variance-only with no degeneracy diagnostic, replicate
   floor, or estimator definition (CF4). n=1 grid anchor: correct as
   an implementation check; it does not certify the n=4 regime and
   should not be described as countering that risk alone.
3. Student-t floor: the plan carries F1 correctly (expected
   configuration; retained-floor coverage in C2; pre-check as run
   gate), but "ν by domination-margin derivation" is under-determined
   — domination holds for every finite ν, the margin is monotone in ν,
   and the derivation needs a declared two-sided criterion parametric
   in the hint whitening variance (CF5).
4. Default audit table: missing material rows — fit row count N vs
   axis count (CF2), replicate count and error estimator (CF4);
   one silent promotion — the row law labeled "reviewed default"
   (CF6); the bar silently assumed by C3 (CF1). The τ clamp row's
   "reviewed" status is conditional on bar = 2.5e-3 (CF1). Device,
   dtype, hints, fixture params, fitter budget: adequately tabled.
5. Budget/stop conditions: the 3-session/12 h budget is actionable
   (countable, manifest-measurable, checkpoint on exhaustion); the
   continuation vetoes are genuine except the adjoint clause (CF7);
   reference compute is unsized within the budget (CF10).
6. Two-stage structure: deferring degree/rank grids, seeds, and the
   formal evidence contract to D1 is sound. Deferred-but-needed-now:
   the accuracy bar (CF1 — C3 cannot size or veto without it), the
   claim-configuration parity gate's existence (CF3 — an engine-
   validity gate, not an attempt05 detail), and the pinned-parameter
   scope statement (CF9).

## Verified sound (no finding)

Research intent ledger role classifications (fit rms/ESS/Gram/hint
residuals explanatory or pathology-veto only; no proxy promoted);
expected-failure-mode pre-classification with declared repairs (the
Research Question Guardian structure is genuinely present); honors
decided questions D1–D3 and the chunk/NeuTra rules (N/A here); fresh
versioned artifact roots, never-overwrite; A1 framed as a
declared-choice diagnostic rather than a benchmark claim; owner
touchpoint at D1 proportionate to governance; out-of-scope list clean
(no C3-compactified work, no n=8, no HMC claims); non-conclusions
correct (no superiority, no transfer, no unrun n).

All blocking findings are small edits (a sentence to a short
paragraph, plus one cheap pre-D rung); none invalidates the campaign
direction or the certified engine.

VERDICT: DISAGREE — blocking findings CF1 (accuracy bar circular in
Phase C and contradicted by the reviewed spec's 2.5e-3/τ_max=bar/25),
CF2 (no pre-D n=4-scale shakedown; reviewed ladder item 5 dropped;
row-law ESS scaling unexamined at the claim scale), CF3 (no parity/
compile gate for the SV+Student-t claim configuration on the XLA
lane), CF4 (reference validity variance-only, no degeneracy
diagnostic/replicate floor/estimator definition), CF5 (ν
selection criterion under-determined: domination alone holds for every
finite ν). Repair and re-review of the diff suffices; no re-derivation
of the underlying program is required.

## Re-verification of repairs (2026-08-24, commit 431689c8)

Checked the committed plan text (`git show 431689c8`, whole file), the
engine diff, the A1 artifact, the fetched paper, and re-ran the
affected tests. Not narrated from the commit message.

- **CF1 discharged.** §2 preamble declares the working bar 2.5e-3
  nats/step with reviewed provenance (C2 note §3 + attempt04) AND the
  aggregation (total defensive-corrected gap over the horizon / T);
  Phase C sizes against it; the D1 bar-change rule explicitly re-opens
  τ_max = bar/25 and C3 sizing; Gate D reworded ("NOT D1's to
  invent"); the τ-clamp audit row is now conditional on the declared
  bar.
- **CF2 discharged.** Phase A3 reinstates ladder item 5 in
  post-F-ENG-1 form: n=4 stress config (degree 12, rank 6, 8 axes plus
  branch — the attempt04 compile-blowup regime) on the ported lane,
  one artifact carrying parity ≤ 1e-12, compile time, per-fit Gram
  condition, per-fit row ESS, and the defensive-corrected gap vs the
  exact Kalman oracle; ESS gate ≥ 5× the widest ALS design width with
  the declared repair (raise N per axis count) — at n=4 this floor
  (≈ 5×468) exceeds the n=2 heritage N=2048, so the rung forces the
  sizing decision fail-closed rather than letting starvation surface
  inside attempt05. Audit table gains the "fit row count N per axis
  count" row with the (ESS₁/N)^d failure mode; the pre-mortem now
  names row-law starvation as a third infeasibility masquerade.
  Engine change verified in the diff: `_christoffel_rows` returns
  row_ess = 1/Σw̄² over the already-normalized weights (correct ESS
  formula given `weights` sums to 1 at :158–159), emitted into the
  per-step diagnostics; both call sites updated and grep confirms no
  other caller of the changed signature. Re-ran
  `test_c2_gaussian_engine_oracle.py -k "degree0 or u_ret"` under
  tftwogpu with `CUDA_VISIBLE_DEVICES=-1` (deliberate CPU-only
  diagnostic): 3 passed in 14.4 s (U-RET-1, closure n=1, degree-0
  n=2 T=120 oracle) — the "re-verified green" claim checks.
- **CF3 discharged.** Phase C5: eager-vs-XLA parity ≤ 1e-12 plus
  fresh-process compile battery on an SV + Student-t + SV-hints
  configuration; Gate C requires "C5 parity green". The call-chain
  requirement is now met by an executable gate on the claim-bearing
  configuration.
- **CF4 discharged.** C3 defines the estimator (one PF log-likelihood
  per replicate; reference = mean over R ≥ 10 independent replicates;
  MC error = SE of that mean), adds the particle-degeneracy screen
  (min per-step ESS recorded; a degenerate replicate invalidates the
  cell's reference — Class B) as part of validity, states the bias
  posture correctly (CLT-regime E[log Ẑ] − log Z ≈ −Var/2, negligible
  under the SE gate; the argument fails under weight degeneracy, which
  the screen exists to catch; the n=1 anchor certifies implementation,
  not the n=4 regime), gives bar/5 its own provenance (new rule of
  this plan, no longer "attempt04 style"), adds the n=2 dense-grid
  cross-check, fixes the reference backend (independent NumPy f64
  under the diagnostic-reference exception), and the audit table gains
  the replicate-count/estimator row with the heavy-tail failure mode.
- **CF5 discharged.** The C2 criterion is now two-sided and
  well-posed: since log M(ν, α) is monotone increasing in ν, "the
  LARGEST ν with log M(ν, α_max) ≤ the declared cap" exists and
  simultaneously optimizes the bulk-dilution side (lightest admissible
  tails), so no separate bulk floor is needed; the criterion is
  parametric in the hint whitening over the declared hint class with
  D1 instantiating actual hints (resolves the C2→C4 ordering); the
  cap is tied to its consumer (ratio guard at τ ≥ τ_min) and Gate C
  gates on the criterion being SATISFIED at declared cap values, not
  on "derivation written"; the LGSSM no-fire check is the non-harm
  side; §1's failure-mode framing now matches F1 (Student-t as the
  expected SV configuration, ν recomputation as the within-scope
  repair, never on claim data).
- **CF6 discharged.** Audit row relabeled "measured repair, NOT yet a
  reviewed default" with single-axis evidence and variant status
  stated; C1 is fetch AND read + reconcile with deltas recorded as
  findings; the fetch is verified on disk
  (`.localresources/papers/cohen-migliorati-2017-optimal-weighted-ls`
  .pdf/.txt; arXiv 1608.00512 is the correct paper).
- **CF7–CF11 discharged.** Continuation veto now names value-side
  parity fixtures only, with the adjoint failure explicitly descoped
  to Phase B's deliverable (CF7); the domination pre-check's dual role
  is declared per the guardian rule (CF8); C3 pins the SV
  parameterization now and Gate C carries the re-open rule for D1
  parameter changes (CF9); reference-compute sizing is in C3 and the
  pre-mortem (CF10); Gate A requires memory-growth policy in A1/A2 GPU
  manifests (CF11).

Residual notes, non-blocking (both covered by gates that now exist):
(i) the committed A1 proxy artifact (`phase_a1/kernel_timings.jsonl`)
is honestly labeled a proxy in the audit table ("measured-proxy →
confirmed at Gate A", failure mode stated) and its numbers match the
table, but the jsonl itself lacks self-describing manifest fields
(kernel/shape, commit, escalation, memory-growth policy) — acceptable
only because the device policy is declared at Gate A on the real
kernel, where the CF11 manifest requirement binds; Gate A must not
repeat the omission. (ii) The C3 degeneracy screen records min
per-step ESS but its numeric invalidation threshold is left to the C3
artifact — analogous to the declared-at-C3 particle-N sizing;
declare it there before first use.

VERDICT: AGREE (after repairs, 2026-08-24)
