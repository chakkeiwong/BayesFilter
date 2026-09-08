# Fable Final-Release Audit Reply: Differentiable Singular SR-UKF

Date: 2026-08-18
Reviewer: Fable (Claude, Opus 5)
Handoff: `docs/plans/bayesfilter_differentiable_singular_srukf_fable_final_release_handoff_2026_08_18.md`
Audit type: read-only final mathematical, numerical, documentation, and release audit.
Evidence root: `docs/plans/artifacts/differentiable-singular-srukf-fable-final-audit-20260818/`

## 0. Provenance and method

- Commit at audit: `58292a6a491ce3bbf7befeacb89f1ffd0ad6a02b`
  ("Promote direct-factor SR-UKF as default"); worktree has staged changes.
  SHA-256 of the handoff, plan, plan review, and execution result are in
  `source_hashes.txt` under the evidence root.
- All checks were bounded to the handoff's listed paths, expanded only into
  cited dependencies (`srukf_route_guard.py`, `srukf_backend_policy.py`,
  `stack_qr_tf.py` internals, `tests/test_factor_srukf_tf.py`,
  `docs/preamble.tex`). No repository file was edited; new files exist only
  under the evidence root and `/tmp`.
- Independent evidence produced (all archived with raw output):
  1. `independent_rank_two_authority_check.py` — a **harder** authority than
     the repository's: rank-2 observation support inside a 3-dimensional
     ambient space (their rank-two test is non-diagonal but full ambient
     rank), non-diagonal mixing, **moving support basis**, dense
     Moore–Penrose + eigen authorities, renormalized ε-limit at three ε,
     centered FD at `h` and `h/2`, off-support rejection. **All pass**:
     gain/mean/posterior-factor vs dense authority ≤ 4.5e−16; support
     likelihood vs eigen authority 4.4e−16; score/`d_increment`/`d_factor`
     vs FD ≤ 1.8e−10; ε-limit converges at the documented rate; off-support
     → `−inf` value, `on_support=False`.
  2. `xla_assert_behavior_check.py` / `xla_telemetry_check.py` — execution-mode
     probes of the fail-closed contract (finding F1 below).
  3. `focused_pytest_output.txt` — the documented focused command re-run.
  4. `pdflatex_build_success.log` — documentation build reproduction (F3).

## 1. Section-by-section answers (handoff §1–§8)

**§1 Plan and execution contract — SUPPORTED with two evidence findings.**
The implementation stays inside the reviewed bounded scope: fixed-rank/
fixed-chart score, value-only SVD discovery, no production/HMC claims. The
plan-review addendum's three repaired rank-two defects are consistent with
the shipped code (the `R^{-1}` gain solve, index-consistent projector, and
dot-product support-coordinate derivatives are all present and verified
independently). Gates map to artifacts/tests except the two recorded in F2
(terminal 143-campaign command) and the GPU gate (correctly recorded as
unresolved). Nonclaims are sufficient; two statements need correction before
release (F1 wording, F3 stale TeX claim).

**§2 Canonical documentation — SUPPORTED with one wording repair (F1) and a
nit (F5).** All five chapters are byte-identical to the
`docs/fable-rewrite/monograph/chapters/` mirror (verified with `cmp`), and
both `main.tex` files input them (lines 36/38/43/44/81 in each). The chapters
correctly define the block-QR algorithm (ch17 §Implemented Direct-QR), the
rectangular fixed-chart extension (ch12 §Fixed-Rank Rectangular QR Charts),
the affine-support measure and complete score including support-basis motion
(ch12 eq. `bf-affine-support-score`: "A log-determinant-only derivative is
incomplete"), the renormalized ε-limit with the correct sign
(`ℓ_ε + (n−r)/2·log(2πε)`, ch14 eq. `bf-validation-renormalized-epsilon-limit`
and ch18 §Current Role), SVD's value-only role (ch18 l.1243–1259), and the
branch-boundary veto (ch23 §Rank, Chart, and Affine-Support Boundaries). I
found no undefined symbols, no dimension ambiguity, no transpose error
(ch17's `S=R_yy'R_yy`, `K=R_yx'R_yy^{-T}`, `P_f=R_xx'R_xx` re-derived below),
and no statement implying a score across a branch boundary. The one behavior
mismatch is ch17 l.406 ("rejects") vs XLA behavior — F1.

**§3 Full-rank direct block QR — SUPPORTED (math), one material behavior
finding (F1).** Independent derivation of the block identity: for
`A=[Y;X]`, `A' = Q[[R_yy,R_yx],[0,R_xx]]` with `Q'Q=I`,
`AA' = R'R` gives the three Gram blocks
`S = YY' = R_yy'R_yy`, `P_xy = XY' = R_yx'R_yy`, and
`XX' = R_yx'R_yx + R_xx'R_xx`; hence
`K = P_xy S^{-1} = R_yx'R_yy^{-T}` and
`P_f = XX' − K S K' = R_xx'R_xx` — exactly the Schur complement with no
covariance formed. The code matches: `block_qr_conditional_tf.py:107–110`
slices `Ly, Lxy, Lf` and computes the gain by the right lower solve
`gain·Ly' = Lxy` (= `R_yx'R_yy^{-T}`); the gain derivative
`d_gain·Ly' = d_Lxy − gain·d_Ly` (l.117–119) is the correct differentiated
solve. `stack_qr_tf.py` never materializes a covariance in the runtime path
(`compute_covariance_diagnostics=False` in the temporal route switches all
residual diagnostics to direct-stack residuals, `factor_srukf_tf.py:261,357`);
the QR sign convention is deterministic positive-diagonal (`stack_qr_tf.py:44–46`);
zero pivots, NaN/Inf, and the relative pivot floor are asserted
(`stack_qr_tf.py:38–43,87–91`). The signed-weight boundary is explicit in
ch17 (eqs. `bf-srukf-weighted-moment-factor`, `bf-srukf-filtered-factor`, and
the DZ5 nonnegative-weight statement l.426–428). Test
`test_block_qr_reconstructs_gain_and_conditional_schur_identity` provides the
dense authority; FD covers all three derivative blocks.

**§4 Rectangular QR mathematics and derivatives — SUPPORTED.** Dimensions
and orientations all check out: `B=AΠ'` handling via `tf.gather` on rows with
`argsort` inverse restoration (`rectangular_factor_tf.py:113–137`);
`B₁=QR₁₁`, `R₁₂=Q'B₂`, chart residual `E₂=(I−QQ')B₂` computed as
`tail − Q·R₁₂` (l.138); `G=Π[R₁₁ R₁₂]'` reconstructs `AA'=GG'` on a valid
chart. The QR-branch derivative (l.54–78) is the standard fixed-sign-branch
identity, and `dR₁₂ = dQ'B₂ + Q'dB₂` (l.125–127) is the plain product rule.
The conditional update matches the plan §3.4 math: with `Y=U R V'` on the
fixed chart, `gain = (X V R^{-1}) U'` via the right lower solve
`solved·R' = XV` (l.466–470 — this is the corrected `R^{-1}`, not `R^{-T}`),
and the posterior loading is the fixed-chart QR factor of `X(I−VV')`
(l.456, 461). The derivative includes every required term: support-coordinate
motion (`d_v`, `d_u_obs`), innovation (`de`), gain (`d_solved`, `d_u_obs`
terms l.475–484), mean increment (l.485), and posterior factor
(`d_residual_x` with all three projector terms l.457–460).
The support likelihood is presented and computed as the r-dimensional density
(`rank`-scaled `log 2π`, l.382–386; diagnostics
`likelihood_measure=affine_support_gaussian_fixed_qr`), never as a raw
ambient limit; the score includes `d_q` (support motion) and `de` (l.390–398).
**Rank-two non-diagonal authority independently verified** — and strengthened:
my check runs rank 2 inside ambient dimension 3 (genuinely singular *and*
non-diagonal, with a moving support), against dense pseudo-inverse and eigen
authorities: agreement to 4.5e−16 and FD to 1.8e−10 at two step sizes.
The renormalized ε-limit `ℓ_ε+(n−r)/2·log(2πε) → ℓ_supp` verified at
ε ∈ {1e−4, 1e−6, 1e−8} with the expected O(ε) convergence; off-support
ambient → −∞ verified.

**§5 Temporal fixed-rank route — SUPPORTED.** Sigma points are generated in
the retained latent rank (`latent_rank = filtered_rank + r_q`,
`rectangular_srukf_tf.py:370–372`), not padded ambient width; the initial
factor width is validated against the frozen filtered rank (l.344–347). The
`tf.while_loop` carries only static shapes; permutations are validated as
static bijections before tracing (`rectangular_factor_tf.py:100–105`,
`TFRectangularSRUKFFixedBranch.__post_init__`). The score call graph closure
is QR-only: verified both by reading every callee and by
`test_fixed_rectangular_score_primitive_closure_has_no_spectral_factorization`
(source scan of the 7-function closure). No hidden SVD/eig/covariance-to-
factor/dynamic-rank in the score path; SVD appears only in the value-only
discovery functions with `value_only=True` diagnostics. Branch identity is a
content-hashed `fixed_rank_row_pivot_qr_v1:<sha256>` over ranks,
permutations, and tolerances (l.141–156) — a caller cannot self-attest.
Pivot/chart/support telemetry is aggregated per step (l.438–440). Fail-closed
score invalidity is **tensor-based** (`chart_valid ∧ conditional_chart_valid ∧
on_support` → `score=NaN` via `tf.where`, l.432–434, 454), which survives
XLA — including the NaN-input case, since NaN comparisons evaluate false and
propagate into `score_valid=False`. Eager/XLA CPU parity is tested to 1e−11
(`test_fixed_rank_temporal_score_matches_fd_and_cpu_xla`), and off-support
returns `−inf` value + NaN score + `score_valid=False` without poisoning the
value (`test_fixed_rank_temporal_off_support_invalidates_score_without_nan_value`).

**§6 Numerical stability and branch semantics — SUPPORTED except F1.**
Scale-aware pivot policy present with 1e−12 float64 default
(`stack_qr_tf.py:14`, `rectangular_factor_tf.py:14–15`, branch dataclass
defaults); chart residual and support tolerances relative to
`max(1,‖·‖)`; the QR sign convention deterministic; NaN/Inf, zero rank
(value-only `rank_zero_value_only` branch), rank changes/anisotropic
near-rank changes (deterministic `chart_valid=False` at retained scale
1e−14), repeated singular values, duplicate permutations, and off-support
observations all covered by tests I re-ran. No silent nugget or jitter
anywhere in the new modules; the rank-discovery SVD carries `value_only=True`
and is never in the score closure. The single behavioral gap is F1: the
assert-based half of the fail-closed contract does not survive
`jit_compile=True` on the **full-rank** route.

**§7 Test coverage and independent authorities — SUPPORTED with F2.**
The focused command reproduces exactly: **25 passed, 1 warning** (the h5py
environment warning, as documented — warnings are not being counted as
passes). Coverage spans values and scores, eager/XLA parity, dense
covariance authorities, centered FD, ε-limit renormalization, on/off support,
rank-zero value-only, malformed inputs (bad bijection, rank > min(N,K),
NaN), and route closure. The **143-pass terminal campaign is not
reproducible from the record**: no exact command or selection is given, and
plausible selections collect 43/87/625 tests (F2). The missing release test
identified by the handoff — a GPU/XLA execution artifact — remains missing
and is correctly recorded as such (see §8).

**§8 GPU and documentation release gates — recorded correctly for GPU;
inaccurate for TeX (F3).** The gate script verifies everything the handoff
lists: `TF_FORCE_GPU_ALLOW_GROWTH` set before the TensorFlow import
(l.14→16), repository memory-growth helper with `require_gpu=True`, exactly
one visible logical GPU, CPU parity at 1e−11, `jit_compile=True`, allocator
telemetry, refusal to overwrite the artifact root, versioned JSON +
checksums, and explicit nonclaims. The documented command pins
`CUDA_VISIBLE_DEVICES=3`; the 3→2→1→0 fallback preference is procedural
(recorded in `physical_gpu_selection`), not automated — acceptable, worth a
sentence. The gate was **not executed** (I verified no artifact root exists)
and the execution result records the HTTP 502 authorization failure and makes
no GPU claim — this is the correct handling; I likewise did not execute it,
since producing the release artifact is outside a read-only audit. The TeX
claim, however, did not reproduce (F3 below).

## 2. Acceptance criteria

| Criterion | Result |
|---|---|
| M1 dimensional consistency under the factor orientation | PASS (independent re-derivation of block and rectangular identities) |
| M2 rank-two non-diagonal update vs dense authority + FD | PASS (repo test at full ambient rank; my independent singular rank-2-in-3 check to 4.5e−16 / 1.8e−10) |
| M3 correct affine-support measure and ε-limit sign | PASS (docs, code, tests, and independent numerics agree) |
| M4 total derivatives of the same fixed value program incl. moving support and innovation | PASS (verified with a moving support basis) |
| M5 no derivative claim across rank/pivot/support/chart/sign/signed-weight boundaries | PASS (docs and code consistently local; explicit vetoes) |
| N1 no covariance-to-factor decomposition in the admitted score path | PASS (source closure + guard + reading every callee) |
| N2 SVD confined to value-only discovery/diagnostics | PASS |
| N3 static-shape/XLA validity on the tested route | PASS for the rectangular score route (tensor gating); **QUALIFIED** for the full-rank default (F1) |
| N4 invalid branches → finite/value-only where defined, clearly invalid score, no silent repair | PASS on the rectangular route; **QUALIFIED** under XLA on the full-rank route (F1) |
| N5 exports/guards/docs/tests agree on the canonical route identity | PASS (`direct_qr_block_conditional` consistent across policy, diagnostics, ch17, plan) |

## 3. Findings, ordered by severity

### F1 — material: fail-closed asserts do not survive `jit_compile=True` on the default full-rank route

Anchors: `bayesfilter/linear/stack_qr_tf.py:38–43,87–91`;
`bayesfilter/nonlinear/factor_srukf_tf.py:460–461` (default `jit_compile=True`
wraps the whole recursion); `docs/chapters/ch17_square_root_sigma_point.tex:406`
("A scale-aware relative pivot floor of 1e-12 **rejects** a numerically
unresolved full-rank chart"); plan §4 Phase A.2 ("Do not set a nonzero
threshold inside an XLA graph").

Evidence (`xla_assert_behavior_output.txt`, `xla_telemetry_output.txt`): a
stack with relative pivot ≈9.4e−15 (below the 1e−12 floor) raises
`qr_relative_pivot_below_tolerance` in eager and graph mode, but under
`jit_compile=True` XLA compiles the `tf.debugging` assertions away: the
default route returns a **finite, plausible value and score** with no error.
NaN-input assertions are likewise dropped (NaN then propagates visibly, so
that case is detectable from outputs). The telemetry is intact — the returned
`relative_qr_pivot` diagnostic correctly reports the violated floor — but
nothing gates on it, unlike the rectangular route, whose
`score_valid`/`chart_valid`/`on_support` tensor logic survives XLA and is the
right pattern.

Blocks: the full-rank direct-QR default's *fail-closed documentation claim*
(ch17 l.406 and the execution-result pivot-policy row). Does **not** block
the fixed-rank score claim (tensor-gated) or the mathematics.

Required repair (either suffices; (a) is one line of prose, (b) is the better
fix and matches the plan's own Phase A.2 instruction):
(a) reword ch17 l.406 and the execution-result gap table to "rejects in
eager/graph execution; under XLA the violation is recorded in
`relative_qr_pivot` and callers must gate on it", or
(b) add a tensor-based `score_valid = relative_qr_pivot ≥ tolerance ∧ finite`
to `TFFactorSRUKFResult` mirroring the rectangular route, and NaN the score
via `tf.where`. Closing test: the below-floor fixture from my probe run under
`jit_compile=True`, asserting either the documented recording semantics or
`score_valid=False`.

### F2 — release evidence: the 143-pass terminal campaign is not reproducible from the record

Anchor: execution result §Verification ("Terminal direct-factor/model
regression campaign … **143 passed, 3 warnings** in 420.77 seconds") — no
command, no test selection. The focused command reproduces exactly
(25 passed, 1 warning — re-run archived), but I could not reconstruct any
selection collecting 143 tests (candidate selections collect 43, 87, or 625).
Two plan-§Phase-E items whose evidence would live in that campaign are
therefore unverifiable from the record: the temporal ill-conditioning sweep
(condition numbers 1e4–1e28) and batch-permutation coverage for the new
routes (duplicate-row invariance is verified in
`tests/test_factor_srukf_tf.py`; a permutation case is not in the files I
audited).

Blocks: release evidence only. Required repair: record the exact terminal
command and selection (and pin it in the result memo), or re-run and record.
Warnings handling is fine — they are counted separately, not as passes.

### F3 — release evidence: the TeX/PDF blocker claim is stale or environment-specific

Anchor: execution result §GPU and document-build status ("Final PDF emission
remains blocked by the pre-existing TeX installation error `pdfTeX error:
Font tcrm1200 at 600 not found`").

Evidence (`pdflatex_build_success.log`): in the current environment
`tcrm1200.tfm` resolves via kpathsea and `mktexpk` generates the 600-dpi PK
font on demand. Reproducing the result's own workaround (read-only
`algorithm.sty`/`algpseudocode.sty` shims — both packages are genuinely
absent, that part of the record is accurate; my shims live in `/tmp` only,
nothing installed), `pdflatex docs/main.tex` **completes and emits a 490-page
main.pdf with no font error**. Note `main.tex` must be built from `docs/`
(it inputs `preamble` by relative path). The font failure was most plausibly
a font-cache-write/permission artifact of the earlier restricted execution
context, not a TeX installation defect; it self-heals when `mktexpk` can
write its cache.

Blocks: documentation release evidence only. Required repair: correct the
execution-result sentence (the actual blocker is the two missing
`algorithm`/`algpseudocode` packages, already disclosed), and produce the
release PDF with real packages rather than shims before publishing.

### F4 — advisory: route-guard hardening around the rectangular closure

Anchors: `bayesfilter/nonlinear/srukf_route_guard.py:32–37,76–79`;
`tests/test_factor_srukf_route_guard.py:30–43`.
`ADMITTED_DIRECT_FACTOR_SRUKF_FILES` intentionally excludes
`rectangular_factor_tf.py` (its value-only half legitimately contains SVD),
so the rectangular score closure is protected by the function-source scan —
correct design, but that scan checks only three tokens and omits `cholesky`
(currently absent from the closure; add it for parity with the file guard).
Separately, the guard's legacy carve-out (l.78–79) exempts the `cholesky`
pattern by **file name** `srukf_factor_tf.py`; a name-keyed exemption is
spoofable in principle. Suggest keying it to the exact repository-relative
path. Also suggest adding a singular rank-two-in-ambient-three fixture to the
repository suite (the current rank-two authority is non-diagonal but full
ambient rank; my audit check can be adopted directly).

### F5 — advisory (documentation nits)

- `docs/chapters/ch17_square_root_sigma_point.tex:440` cites the absolute
  path `/home/chakwong/python/src/dsge_hmc/filters/CUTSRUKF.py` — a personal
  home-directory reference in a canonical monograph chapter; replace with a
  repository-relative or archival citation before release.
- GPU gate: the 3→2→1→0 device preference is procedural (documented command)
  rather than automated in `run_fixed_rank_srukf_gpu_gate_20260818.py`; one
  sentence in the result memo should say the fallback is manual.

## 4. Unresolved release gates (correctly recorded, still open)

1. **GPU/XLA execution artifact** — gate script is sound (§1.§8 above), was
   not executed due to the HTTP 502 authorization failure, and no GPU claim
   is made. This remains an open release evidence gate; it must be run under
   proper authorization before any GPU support claim.
2. **Release PDF** — see F3; buildable today, but with `/tmp` shims standing
   in for two absent packages.

## 5. Verdict rationale

The mathematics is sound and independently verified beyond the repository's
own evidence: the block-QR conditional identity, the rectangular fixed-chart
construction, the rank-two non-diagonal conditional update on a genuinely
singular geometry, the affine-support measure with the correctly signed
renormalized ε-limit, and total derivatives including moving support
coordinates all pass dense authorities and two-step finite differences to
machine-level residuals. The rectangular fixed-rank score route's fail-closed
semantics are tensor-based and genuinely XLA-robust. The route guards, SVD
confinement, branch identity hashing, and nonclaims are consistent across
code, tests, and the five byte-identical chapter pairs.

But a final-release audit must also certify the documented behavior and the
release evidence, and three items fail that bar: the default full-rank
route's fail-closed pivot/NaN assertions are silently compiled away under its
own default XLA mode while the chapters say "rejects" (F1); the terminal
143-pass campaign has no reproducible command (F2); and the recorded TeX
blocker does not reproduce, so the documentation-status record is wrong in
the other direction (F3). All three are small, closable repairs — none
undermines the fixed-rank score claim itself.

VERDICT: REVISE
