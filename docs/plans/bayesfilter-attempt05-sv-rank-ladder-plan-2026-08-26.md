# attempt05 — SV Rank Ladder under C2 — Experiment Plan — 2026-08-26

Status: `AWAITING_OWNER_GO` (campaign plan Phase D1; the declared owner
touchpoint — ladder compute launches only on a plain go-ahead).
Skeptical self-audit in §8. Governed by
`bayesfilter-c2-completion-campaign-plan-2026-08-24.md`
(REVIEWED-EXECUTABLE; Gates A, A3, C all PASS).

## 1. Question and evidence contract

- Question: r*(n) — the smallest TT rank meeting the declared accuracy
  on the SV family under the C2 program — at n ∈ {2, 4}. This is the
  redeclared feasibility question (decision D2): LGSSM is the oracle,
  SV carries the rank claim.
- Accuracy bar: 2.5e-3 nats per step (inherited; campaign plan §2
  preamble; τ_max = bar/25 provenance intact — unchanged, so no
  re-opens fire).
- Cell pass rule: |engine defensive-corrected total − reference total|
  / T ≤ 2.5e-3, with the cell's reference VALID and all engine vetoes
  clean. r*(n) = the smallest rank whose cells pass at ALL seeds at the
  declared degree.
- Comparators: n=2 — the exact dense-grid filter (resolution- and
  width-converged; zero MC error; Gate C certified). n=4 — bootstrap
  particle under the Gate C policy: R = 10 replicates, SE-of-mean; a
  cell's reference is valid iff SE_total ≤ T·bar/5, min per-step
  normalized ESS ≥ 0.05 on every replicate, and the N-doubling
  consistency check holds (two adjacent N agreeing within 2× joint SE).
  n=4 reference N ladder: 400k and 800k (sized from the measured n=2
  bias decay; the doubling check is the no-grid substitute).
- Promotion vetoes (cell invalid as rank evidence, not campaign stops):
  invalid reference; non-finite anything; row-ESS floor breach
  (5× design width); τ_t at τ_max; per-cell timeout.
- Explanatory only, never gates: fit rms, Gram condition, row ESS
  values above floor, hint residuals, wall time.
- Must not be concluded: HMC/posterior readiness; transfer of any
  tuning to other models; capacity claims from the degree screen; any
  n not run; statistical superiority over any other method. Single
  fixture family (coupled-A SV extension) is a declared scope limit.

## 2. Scope pins (all Gate-C certified; changing any re-opens Gate C)

- Fixture: `sv_fixture_c2_20260826.sv_model(n, seed)` — ZC24 Example 1
  synthetic values (γ=0.6, σ=1, β=0.4), coupled-A vector extension.
  Fixture-family audit (campaign CF9/D1 obligation): the coupling
  0.1·randn/(n−1) mirrors the attempt04 LGSSM family; independent
  components would factorize the target and trivialize the rank
  question — the coupling is what makes r*(n) informative. Accepted.
- Hints: `sv_gh_hint_factory` (GH 9-point), frozen per step. Measured
  n=1 quality 1.3e-2/2.8e-2; α measured 0.67.
- Defensive floor: Student-t, ν = 27.62 = criterion(α_max=0.8,
  cap=12); α_max=0.8 covers the measured 0.67 with headroom. Runtime
  α re-check: if hint diagnostics at any cell imply α > 0.8, the cell
  is flagged and ν recomputed per the criterion (declared repair, not
  scope change; never on claim data).
- τ policy: clamp(ε̂², 1e-6, 1e-4) (reviewed).
- Row law: β(d) = 0.5 (d ≤ 4) / 0.10 (d > 4), N = 8192 both n
  (margin over the certified 2048 at n=2; A3-sized at n=4). Degrees
  stay ≤ 8, far from the ℓ=13@9-axis boundary; approaching it would
  require new row-law evidence first (A3 rule).
- Fitter budget: sweeps = 32 (tuned control; A3 evidence: oracle-exact
  at 9 axes with s=32; GPU cost makes the margin cheap). One declared
  repair: a cell whose rms exceeds 10× the fitter floor may be retried
  once at 2× sweeps, recorded.
- Lane: XLA on the 4080 (CUDA_DEVICE_ORDER=PCI_BUS_ID,
  CUDA_VISIBLE_DEVICES=1), memory growth verified in the manifest;
  5080 fallback per the standing directive.

## 3. Grid and execution

- Horizon T = 20 (matches the validated reference runs); observation
  seeds {42, 142, 242} per cell (attempt04 pattern), fixture model
  seed fixed per n (52-family) so all cells share the model and
  references are computed once per (n, seed).
- Stage 1 (degree screen, explanatory): degree ∈ {2, 4, 6} at rank 6,
  one seed — selects the working degree as the smallest whose gap
  plateaus (the O3 expectation: low degree suffices; the screen makes
  the choice measured, not assumed).
- Stage 2 (rank ladder, the claim): rank ∈ {1, 2, 3, 4, 6} at the
  working degree, all three seeds, both n. Skip-established-rank: once
  r passes all seeds, higher ranks are not run (attempt04 refinement).
- Per-cell subprocess isolation with 60-min timeout; append-only
  accumulator JSONL; heartbeat lines + coverage-filtered monitor;
  every cell records the full manifest (commit, config, device,
  memory-growth, seeds, τ/ν/β/N, wall).
- Artifact root:
  `docs/benchmarks/artifacts/c2_completion_20260824/attempt05/`.

## 4. Budget

References: n=2 grids free; n=4 PF 2N×R ≈ 30–60 min (NumPy, CPU,
parallel to GPU cells). Cells: n=2 ≈ 2–4 min, n=4 ≈ 8–15 min + one
408 s compile per branch signature; ≈ 36 cells worst case before
skip-rank pruning → ≈ 2.5–4.5 h total, inside the remaining ≈ 6.5 h
campaign budget. Stop conditions: campaign budget exhaustion; reference
validity unachievable at n=4 within the N ladder (then r*(4) is
reported unmeasurable-under-budget, which is a result, not a failure).

## 5. Pre-mortem

- Pass-for-wrong-reason: too-easy fixture (guarded: coupling audit
  above; the degree screen will show if degree 2 suffices — reported
  as a property of the family, not suppressed); reference bias at n=4
  (guarded: screen + doubling consistency; the n=2 stage runs against
  an exact grid so cross-checks the engine independently).
- Fail-for-wrong-reason: fitter floor (guarded: s=32 + declared retry);
  hint quality (explanatory diagnostic recorded per cell; a hint-driven
  failure appears as degree-screen non-plateau, triggering the α/ν
  re-check, not a rank conclusion); row starvation (ESS floor veto).
- Infrastructure: compile timeouts at 9 axes (per-cell timeout +
  fresh-process isolation, proven in A3).

## 6. Deliverables

Accumulator + per-cell manifests; result note with the decision table
and inference-status table (hard vetoes / viable / no statistical
ranking claims / descriptive-only / next evidence); monograph ch38 arc
completion and reset memo update; one terminal result review.

## 7. What a verdict looks like

r*(2) and r*(4) as all-seed-passing smallest ranks at the declared
degree, with veto-clean cells and valid references — the campaign's
goal evidence. A clean "no rank ≤ 6 passes" is equally a verdict
(feasibility boundary of the family at this budget), reported with the
degree-screen context and without silent escalation.

## 8. Skeptical self-audit

Wrong baseline? References are Gate-C certified with per-cell validity
gates; n=2 is exact. PASS. Proxy promotion? rms/ESS/condition/hint
residuals all classified explanatory or veto; the pass rule uses only
the reference comparison. PASS. Stop conditions? §4. PASS. Unfair
comparison? No cross-method claims in scope. PASS. Hidden assumptions?
The scope pins are enumerated with their certification provenance; the
one new choice (T=20, seeds, grid) is declared here and consistent
with the reference validation scope. PASS. Environment? Claim lane and
fallback pinned; manifests record device + memory growth. PASS.
Commands answer the question? The pass rule is computed from exactly
the artifacts each cell writes. PASS.
