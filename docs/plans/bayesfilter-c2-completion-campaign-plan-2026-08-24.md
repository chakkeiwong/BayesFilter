# C2 Completion Campaign — Execution Plan — 2026-08-24

Status: `REVIEWED — EXECUTABLE` (skeptical self-audit §8; independent
material review `bayesfilter-c2-completion-campaign-plan-review-2026-08-24.md`:
initial DISAGREE with CF1–CF11, all repaired in `431689c8`,
re-verified "VERDICT: AGREE (after repairs, 2026-08-24)". Two carried
non-blocking obligations: Gate A artifacts must carry the memory-growth
manifest fields the A1 proxy jsonl lacks; C3's artifact must declare
its degeneracy threshold. attempt05's own plan gets a separate audit at
D1 per the two-stage structure below).

Governs the remaining program from the oracle-certified C2 value engine
(`d9a5481f`, suite 10/10) to the campaign goal: a valid r*(n) feasibility
curve at n ≥ 4 on a genuinely non-Gaussian target. Supersedes the
roadmap fragments in the 2026-08-24 reset memo updates; does not reopen
any decided question (construction C2, τ/λ policy D3, ladder
redeclaration D2, tftwogpu/4080-first environment directive).

## 1. Research intent ledger

- Main question: the smallest TT rank r*(n) meeting the declared
  per-step accuracy on the stochastic-volatility family under the C2
  program, as n grows — the feasibility evidence the master program
  needs.
- Candidate under test: the C2 Gaussian-reference engine (Hermite,
  half-mixture row law, clamped τ, frozen deterministic hints) at
  declared degree/rank/sweeps budgets on the XLA claim lane.
- Expected failure modes, classified in advance:
  - degree/rank demand grows fast on SV → that IS the answer the curve
    exists to measure (result, not harness failure);
  - fitter floor binds before the accuracy bar → declared repair:
    fitter-budget escalation within scope (F-ENG-2), not scope creep;
  - the Student-t floor (the EXPECTED SV configuration per F1) fails
    its declared margin cap under measured hints → declared repair:
    recompute ν per the C2 two-sided criterion, within scope, never on
    claim data;
  - compile/resource blowups → per-cell resource rule (attempt04
    pattern: cell recorded, arm continues).
- Promotion criterion (campaign level): attempt05 cells pass their
  declared accuracy bar against a validated reference with all vetoes
  clean; r*(n) read off per the attempt05 contract.
- Promotion vetoes: reference invalid or under-resolved; domination
  pre-check failed for the run's λ; non-finite/flagged cells; fitter
  rms ceiling breached (pathology detector).
- Continuation vetoes (stop the campaign): the VALUE-side parity
  fixtures cannot be made green (engine invalid); the SV reference
  ladder cannot be validated at n=1 (no trustworthy comparator);
  campaign budget exhausted. (CF7: an adjoint-fixture failure blocks
  Phase B's score-path deliverable only — the value-side r* claim is
  independently oracle-certified and proceeds.)
- Dual-role declaration (CF8): the domination pre-check is BOTH a
  promotion veto (a run that fails it yields no claim) AND a repair
  trigger (recompute ν per the C2 criterion) — declared here per the
  guardian rule.
- Explanatory diagnostics (never gates): ESS of row weights, Gram
  condition, per-step ε̂, wall time, compile time.
- Must not be concluded: HMC/posterior readiness; statistical
  superiority over any other method; transfer of SV tunings to other
  models; any n not actually run.

## 2. Phases, gates, budgets

Working accuracy bar (CF1 repair): **2.5e-3 nats per step**, inherited
from the reviewed C2 note §3 and attempt04 (read as attempt04 read it:
total defensive-corrected gap over the horizon divided by T). Phase C
sizes references against this bar. A D1 bar change is NOT free: it
re-opens the reviewed τ_max = bar/25 clamp and the C3 reference sizing,
and must say so explicitly.

Total campaign attempt budget: 3 working sessions beyond this one, or
~12 h wall of runs, whichever first; then a mandatory checkpoint memo
and owner sync. Every run writes under
`docs/benchmarks/artifacts/c2_completion_20260824/<phase>/` (fresh
versioned subdirs, never overwrite). Environment: tftwogpu; GPU runs on
the 4080 first per the standing directive; CPU-only diagnostics say so.

### Phase A — XLA claim lane (blocking for D)

- A1 (smallest first): device/dtype measurement on the oracle fixture —
  step time for eager-CPU (have), XLA-CPU, and float64-GPU on the 4080
  (consumer f64 throughput is poor; the bounded program's claim lane was
  XLA-CPU). OUTPUT: measured table + declared claim-lane device policy
  with manifest fields. This is a declared-choice diagnostic, not a
  benchmark claim. Budget: ≤ 1 h.
- A2: port the Gaussian value engine's hot path to the XLA lane
  mirroring `squared_tt_engine_xla_tf` (tf.function, stable input
  signatures; provenance/IO outside kernels per the TF policy; fresh-
  process compile batteries).
- Gate A (recalibrated 2026-08-25 on measured attribution evidence;
  `gate_a/gate_a_verdict.json`): per-step lane parity ≤ 1e-12 on the
  degree-0 T=120 oracle (certifies every non-fit component; measured
  7.06e-14) AND U-SOLVE-PARITY backend equivalence ≤ 1e-10 on a single
  stress-conditioned solve under real Christoffel weights (measured
  7.9e-14) — these are the correctness criteria. Swamp-regime
  stress-config lane gaps are floor noise, not correctness evidence
  (eager-vs-eager with a 1e-7 ridge perturbation diverges to
  4.13e-4/step; lanes at sweeps 16 converge to 4.07e-7/step, tracking
  the F-ENG-2 floor): gated only at a 10×-floor ceiling with each
  lane's own defensive-corrected oracle gap (rung-4b bound) as the
  scientific criterion. Compile time recorded and
  bounded (per-cell timeout rule inherited from attempt04). Failure →
  repair within budget; if XLA is unreachable, eager-CPU becomes the
  documented-exception lane (TF policy) and D's budgets are re-derived
  before proceeding. A1/A2 GPU manifests record the verified
  memory-growth policy (CF11).
- A3 (CF2 repair — reinstates the reviewed ladder's item 5 in
  post-F-ENG-1 form): **n=4-scale shakedown on the ported lane**, one
  LGSSM rung at the stress config (degree 12, rank 6, 2n = 8 axes plus
  branch — the attempt04 compile-blowup regime), short horizon,
  recording in one artifact: eager-vs-XLA parity, compile time, per-fit
  design-Gram condition, per-fit row ESS, and the defensive-corrected
  gap against the exact Kalman oracle. Gate values: parity ≤ 1e-12;
  ESS ≥ 5× the widest ALS design width, else the row count N is raised
  for that axis count per the §3 audit row (starvation must be caught
  here, not inside attempt05 where it masquerades as rank
  infeasibility). Budget: ≤ 1 h of runs.

### Phase B — adjoint (score) port (NOT blocking for D)

- B1: implement per the C2 note's node inventory (B = I removes mass
  nodes; derivative recurrence Hē_k' = √k Hē_{k−1}; η-ratio factor
  θ-free under frozen maps; M2/M3/M1-DETACHED contracts unchanged).
- Gate B: adjoint-vs-forward-JVP fixture ≤ 1e-12 at n ∈ {1, 2}
  (I-P2-4 pattern). Runs in parallel with C/D or after; blocking only
  for the later HMC/MLE stage.

### Phase C — SV prerequisites (blocking for D)

- C1 (CF6 repair): fetch AND read Cohen & Migliorati (2017) — done for
  the fetch (`cohen-migliorati-2017-optimal-weighted-ls`, arXiv
  1608.00512) — reading the stability sections and reconciling the
  implemented defensive HALF-mixture variant (bounded per-axis weights
  ≤ 2, hence ≤ 2^d after the product) against the paper's conditions;
  record any delta as a finding, not a footnote.
- C2 (CF5 repair): Student-t defensive floor: product-t λ (log-space
  evaluation, closed-form ∫λ dμ = 1 and per-axis retention marginals),
  unit tests (quadrature parity as U-HERM-1/U-RET-1 were), plus the
  retained-floor-term coverage flagged open in the C2 review (F1).
  ν selection is TWO-SIDED and parametric in the hint whitening
  (α = 1 − s²/σ_f² over the declared hint class, instantiated with
  actual hints at the D1 run gate): domination alone holds for every
  finite ν and its margin log M(ν, α) is monotone in ν, so the
  criterion is — take the LARGEST ν (lightest tails, least bulk
  dilution) such that log M(ν, α_max) ≤ the declared cap, the cap tied
  to its consumer (the ratio guard at τ ≥ τ_min); non-harm side: the
  LGSSM oracle suite must be unchanged with the t-floor active
  (no-fire check). Never tuned on claim data.
- C3 (CF4 + CF10 repair): SV adapter + reference ladder in the paper's
  log-volatility coordinates, pinned NOW to the paper's SV
  parameterization and horizon (CF9). n = 1: deterministic dense-grid
  filter (near-exact); n = 2: dense-grid cross-check where feasible
  PLUS long-particle; n = 4: long-particle. Reference estimator
  DEFINED: per replicate, one particle-filter log-likelihood estimate;
  reference value = mean over R ≥ 10 independent replicates; MC error
  = the standard error of that mean. Validity gate per cell: MC error
  ≤ bar/5 (provenance: new rule of this plan, so reference uncertainty
  is a minor fraction of the declared bar and cell verdicts attribute
  to the engine) AND a particle-degeneracy screen (minimum per-step
  ESS over the run recorded; a degenerate replicate invalidates the
  cell's reference — Class B, adopt by default). Bias posture stated
  honestly: in the CLT regime E[log Ẑ] − log Z ≈ −Var/2, negligible
  once the SE gate holds; that argument FAILS under weight degeneracy,
  which is exactly what the degeneracy screen exists to catch — the
  n=1 anchor certifies implementation, not the n=4 regime. Reference
  backend: an independent NumPy f64 implementation is permitted (and
  preferred) per the backend rule's diagnostic-reference exception.
  C3 must size the reference compute (wall estimate) before launching
  it; it shares the campaign budget.
- C4: SV moment hints: deterministic companion moment recursion
  (model-specific), validated by a hint-quality diagnostic (whitened
  target moment residuals at probe steps vs the n=1 reference).
  Hints are frozen scope inputs; hint quality is explanatory for
  degree/rank demand, never a correctness gate.
- C5 (CF3 repair): **claim-configuration parity**: eager-vs-XLA parity
  ≤ 1e-12 plus a fresh-process compile battery on an SV + Student-t +
  SV-hints configuration on the ported lane (small horizon suffices);
  the call-chain rule requires parity evidence on the configuration
  the claims actually run, not only on LGSSM/λ ≡ 1.
- Gate C: floor unit tests green incl. the LGSSM no-fire check; the ν
  criterion SATISFIED at its declared cap values (not merely
  "derivation written"); n=1 (and n=2 where feasible) reference
  cross-checks green with the degeneracy screen; hint diagnostic
  recorded; C5 parity green. Scope pin: C2–C5 evidence is bound to the
  pinned SV parameterization; any D1 fixture-parameter change re-opens
  Gate C (per-scope rule, LEDH-style).

### Phase D — attempt05 (the science)

- D1: write the attempt05 experiment plan as its own artifact under the
  existing templates: full evidence contract; the domination pre-check
  as a run gate; per-scope tuning partition (tuning data disjoint from
  claim data, LEDH-style); the fitter budget (sweeps/init/rounding) as
  a declared tuned control with provenance (F-ENG-2/3); degree/rank
  grids; ≥ 3 seeds per cell; per-cell subprocess isolation, timeout
  tolerance, skip-established-rank, accumulator + monitor (attempt04
  runner pattern); artifact root; pre-mortem. The owner sees this plan
  before compute is launched (stated touchpoint; a plain go-ahead
  suffices per governance).
- D2: runs → result note with decision table and inference-status
  table; monograph arc completion (Defect 2→3→C2 ending; the row-law
  finding as a short monograph addendum); reset memo; commits.
- Gate D: per the attempt05 contract. Degree/rank grids, seeds, and
  the cell-level contract are D1's to declare with their own audit;
  the accuracy bar is NOT D1's to invent — it is fixed above (§2
  preamble), and changing it re-opens the named reviewed provenances.

## 3. Default and assumption audit (material choices)

| Choice | Provenance | Justification | Failure mode | Early diagnostic | Status |
|---|---|---|---|---|---|
| Claim-lane device/dtype | Gate A CONFIRMED (2026-08-25): 4080-f64, degree-0 kalman identical to CPU (1.78e-9, no downcast); stress sweeps-16 kalman 2.38e-7 in 168 s vs CPU-XLA 726 s at sweeps 8 | 4-8x faster at equal-or-better accuracy with the sweep budget as control | sweeps-8 stress-class fits excurse with the fitter floor (measured 5.73e-4) — claim runs use sweeps ≥ 16 | per-run kalman/oracle checks where an oracle exists; fitter budget declared in D1 | measured-confirmed: 4080 GPU claim lane |
| Fit row count N per axis count | n=2 heritage (2048/8192) | worked at 2n=4 axes | per-axis ESS fraction compounds ~(ESS₁/N)^d → starvation at 2n=8–9 axes masquerading as rank infeasibility | A3 measured ESS with the ≥ 5× design-width floor; per-fit `row_ess` emitted | hypothesis, sized at A3 (CF2) |
| sweeps = 8 engine default | measured floor (F-ENG-2) | 7e-6/step on representable targets | insufficient for SV accuracy bar | rung-4b regression bound; D1 tuning scope | baseline, tuned in D |
| Half-mixture row law | measured engineering repair (F-ENG-1; SINGLE-AXIS ℓ=13 evidence); citation fetched, reading pending (C1); half-mixture is a variant of the cited law | Gram 1.32 + ESS 1400/2048 at one axis | dimension scaling unexamined; variant may miss the paper's conditions | A3 multi-axis measurement; C1 reconciliation | measured repair, NOT yet a reviewed default (CF6) |
| τ clamp [1e-6, 1e-4] | reviewed (F2 repair); τ_max := bar/25 with bar = 2.5e-3 | inherits fixed-policy no-harm + bar-derived cap | provenance breaks if the bar moves | τ_t in diagnostics; §2 bar-change rule | reviewed default, conditional on the declared bar (CF1) |
| Reference replicate count & estimator | R ≥ 10, SE-of-mean estimator (C3, this plan) | SE gate needs a defined estimator and floor | few-replicate SE understates under heavy tails | degeneracy screen (min per-step ESS) per replicate | declared, validated at Gate C (CF4) |
| Student-t ν | to be derived (C2) | domination margin vs declared SV tails | too heavy: bulk dilution; too light: domination fails | domination margin computation + no-fire check on LGSSM | hypothesis |
| SV fixture family/params | paper's SV (ZC24 §6) | source-anchored | fixture too easy/hard distorts r* reading | n=1 reference behavior + pre-mortem in D1 | paper-anchored, audited at D1 |
| Reference particle N | to be sized (C3) | MC error ≤ bar/5 rule | under-resolved reference biases r* | replicate error bars at n=1 gate | hypothesis → validated |
| Frozen SV hints | C4 construction | paper 5.2 uses estimated moments; ours must be deterministic (derivative program) | poor hints inflate degree demand (mistaken for infeasibility) | hint-quality diagnostic vs n=1 reference | hypothesis, validated C4 |

## 4. Pre-mortem

- Passes misleadingly: an under-resolved particle reference at n=4
  could certify wrong r* — countered by the MC-error veto and n=1
  anchor. A too-easy SV parameterization could make r* trivially small
  — countered by the D1 audit of fixture params against the paper's
  settings.
- Fails misleadingly: fitter floor, hint quality, OR row-law ESS
  starvation at 8–9 axes (CF2 — the one with an exponential dimension
  dependence) could masquerade as rank infeasibility — countered by
  the declared repair triggers (fitter escalation; hint diagnostic),
  the A3 shakedown with its ESS floor, per-fit `row_ess` emission, and
  by reading r* only under the attempt05 contract's veto-clean cells.
- Budget burn: the C3 reference build (long-particle at n=4, R ≥ 10,
  SE ≤ bar/5) is itself compute — C3 sizes it before launch and it
  draws from the same 12 h budget (CF10).
- Infrastructure: XLA compile blowups recur at 9-axis scale — per-cell
  timeout + resource rule already proven in attempt04.

## 5. What is out of scope

HMC/NeuTra integration; any default-policy change outside this program;
n = 8+ runs unless attempt05's contract declares them; C3-compactified
fallback work (dormant unless a continuation veto fires on C2).

## 6. Artifacts and reporting

Every phase gate writes its evidence under the campaign artifact root;
result-bearing notes follow the decision-table/inference-table
templates; the reset memo is refreshed at each phase gate; commits at
coherent checkpoints as throughout this campaign.

## 7. Review points

This plan: one independent material review before execution (below).
attempt05 plan: its own skeptical audit + owner touchpoint at D1.
Terminal: one result review of the attempt05 result note.

## 8. Skeptical self-audit (pre-execution, per policy)

- Wrong baseline? The r* comparator is a validated resolved reference,
  not a weak baseline; the reference itself carries a validity gate
  (C3). PASS.
- Proxy promoted? Fit rms, ESS, Gram condition, hint residuals are all
  classified explanatory or pathology-veto, never promotion. PASS.
- Missing stop conditions? Continuation vetoes and a total budget are
  declared (§1, §2). PASS.
- Unfair comparison? No cross-method claims are in scope; r* is a
  within-program feasibility curve. PASS.
- Hidden assumptions? The material ones are tabled in §3 with
  diagnostics; the known-weakest (reference N, hints, ν) are gated
  before D. PASS with the note that §3's "measured" statuses must be
  filled, not narrated.
- Environment mismatch? A1 exists precisely to measure rather than
  assume the claim lane; tftwogpu/4080 directive honored. PASS.
- Commands answer the question? Each gate's artifact is named next to
  its criterion. PASS.
