# Reset Memo: Squared-TT Program — Reference-Domain Pivot (2026-08-24)

Status: `CHECKPOINT` (supersedes as entry point:
`bayesfilter-squared-tt-program-reset-memo-2026-08-19.md`; its
continuation queue is fully executed and its n=4 structural-suspects
question is now answered — the suspect was the domain).

## What is now true (verified, committed)

- attempt04 CLOSED (`e56d7528`): r*(2) = 6 confirmed under the corrected
  design (margins 3-8x, truncation ratios <= 0.11); all n=4 cells
  invalidated as rank evidence by the pre-declared truncation-ratio flag
  (ratios 0.36-1.85; in 3 of 6 cells more step mass outside the box than
  inside). r*(4) is unmeasurable inside the bounded-box program.
- Root cause chain complete and documented as monograph Defect 3: the
  containment constraint (retained polynomial defined only on its own
  box) pins the effective previous-block half-width at kappa* ~ 2.0-2.2
  sigma (measured twice, kappa-invariant); truncated mass is then
  1-(2*Phi(kappa*)-1)^n — geometric in n; conditioning costs exactly that
  mass in TV even at zero fit error. No rank/degree repair exists.
- Reference-domain selection ANSWERED BY DERIVATION, not preference
  (`55e80c7b`, `891ac16e`, `c9c5ba5c`,
  `docs/plans/bayesfilter-reference-domain-selection-derivation-note-2026-08-23.md`):
  C1 bounded-box ELIMINATED (three grounds: containment/R2, geometric
  truncation + bias floor, defensive-domination hypothesis violated
  intrinsically). C2 Gaussian-reference/Hermite SELECTED (degree 0 at the
  Gaussian calibration point vs 21-63 per axis for C3-compactified,
  measured; identity mass matrices; cost channels). C3 recorded fallback
  (reuses incumbent algebra; the authors' own code pays ell=33/axis).
- Monograph documentation COMPLETE (`347c5fd5`): ch38 sections 40.8
  (Defect 3) and 40.9 (Reference-Domain Selection) in proposition-proof
  form — containment lemma, box-mass identity, truncation growth,
  conditioning-TV identity, Hermite mass identity (generating-function
  proof), C3 algebra invariance, defensive mixture-bias bound
  tau/(1+tau), domination-failure proposition, C2 degree collapse, C3
  no-geometric-rate theorem (Laplace bound + Bernstein ellipse + identity
  theorem + 4th-power trick for the branch cut). Defect 2 closing
  paragraph updated. `main.tex` compiles clean: 507 pp, zero undefined
  refs/cites. Diagnostic script:
  `docs/benchmarks/check_reference_domain_selection_20260823.py`.
- MathDevMCP audit run on the new labels (`audit-derivation-v2-label`,
  root docs/): ZERO mismatches/refutations on every label audited;
  mechanical certification `inconclusive:source_label_missing` on all
  obligations — a parser label-preservation limitation of the tool on
  this repo's LaTeX (known quirk, see memory note on the display-bracket
  macro masking repair), whose recommended path is human review. Per
  review proportionality this is recorded as a tool limitation, not a
  finding; the planned material review of the C2 derivation note is the
  human certification step.
- Local paper copies fetched and text-extracted
  (.localresources/papers/): `cui-dolgov-2022-deep-composition-sirt`
  (arXiv 2007.06968v3 = FoCM 22:1863-1922) and
  `cui-dolgov-zahm-2023-self-reinforced-approx` (arXiv 2303.02554).
  These carry the delegated unbounded/weighted approximation theory
  (Zhao-Cui Prop. 9 remark, txt:1372).

## Decisions recorded (owner, 2026-08-23)

1. D1: the n>=4 line PROCEEDS on C2 (Gaussian-reference program).
2. D2: ladder redeclared — LGSSM becomes the machine-precision
   exactness/parity oracle under C2 (degree-collapse proposition makes
   r*(LGSSM) trivial by construction); the rank-feasibility question
   r*(n) moves to the SV / non-Gaussian arm (or degraded hints).
3. D3: defensive-term policy — the paper's coupling
   tau_t <= ||phi_t - sqrt(q_t)||^2_L2 (their Prop. 11) is adopted for
   the C2 program (fixed tau=1e-6 remains for smokes only); the
   defensive density lambda defaults to the reference with the
   domination condition sup q/lambda < inf re-verified per declared
   target tail class, escalating to a heavier-tailed product lambda
   where needed.

## Continuation queue (in order)

UPDATE 2 (2026-08-24, engine block): C2 value engine BUILT and
oracle-certified. Commits eb25d325 (HermiteBasis1D + U-HERM-1),
36516d5a (gaussian engine), fa8446e2 (U-RET-1 + closure + degree-0
T=120 gate at 1e-8), d9a5481f (Christoffel half-mixture row law after
the stress rung exposed the raw-row Gram catastrophe; ALS floor
quantified 3/8/16 sweeps -> 1e-3/7e-6/1e-7 per step; warm start
evaluated + rejected; rung 4b = fitter-floor regression bound sweeps 8
T=12 gate 2e-4; full suite 10/10). Findings + amendments in the C2
note sec 3b. REMAINING before attempt05: XLA port + parity rung (claim
lane), adjoint engine port + I-P2-4-style fixture, attempt05 SV plan
(fitter budget = declared tuned control; Student-t floor expected per
F1; domination statement incl. retained floor term).

UPDATE (later 2026-08-24): items 1-3's reading/writing/review halves are
DONE — papers read (caveat disposition: both Hermite objections are
sampling-side CDF inversion or the O7 domination condition; CD22 Prop 2
anchors retention with M_k = I), C2 derivation note written and
MATERIALLY REVIEWED to "VERDICT: AGREE (after repairs)" through three
rounds (F1: SV domination corrected — Student-t floor is the EXPECTED SV
configuration, tail variance sigma_f^2 vs whitened s^2 < sigma_f^2; F2:
tau = clamp(eps_rel^2, 1e-6, 1e-4) with provenance; F3-F6 minors; F7:
oracle gate compares the defensive-corrected sum
sum_t(log Z_t - log(1+tau_t)) vs Kalman). Commits 90cd98c5, 37d94faa,
3e88f959. NEXT: engine implementation against the reviewed note (item 3
second half), then item 4 diagnostics, then attempt05.

1. READ (postdoc standard, not skim) the two fetched papers' technical
   sections: Cui-Dolgov 2022 — squared-IRT construction, weighted bases,
   error propagation; Cui-Dolgov-Zahm 2023 — self-reinforced
   approximation for concentrated densities (bears directly on the
   preconditioned-ratio fits). Anchor equations for the C2 note.
2. WRITE the C2 derivation note
   (`bayesfilter-c2-gaussian-reference-derivation-note-<date>.md`):
   normalized-Hermite bases and B=I Gram chains; row law (Sobol through
   Phi^{-1}); retention re-derivation under eta (reference-typed
   convention carries over; no boxes, no containment); defensive term
   under D3; conversion/log-det terms for the triangular maps (A14
   machinery unchanged); manual-adjoint node inventory vs the certified
   bounded program (expected isomorphic; B=I simplifies); XLA kernel
   plan (three-term recurrence, same shape as Legendre); LGSSM oracle
   gate design (machine-precision pass expected); SV question
   declaration with evidence contract.
3. One MATERIAL REVIEW of the C2 note (this is also the human
   certification step the MathDevMCP audit defers to), then engine
   implementation, unit ladder (U-MAP-MOM-1 analogues), and the
   attempt05 plan under the redeclared question.
4. Engineering diagnostics owed before any C2 claim run: Hermite
   conditioning at n>=8 rows (measured tail factor ~26 at |z|=4, k<=16
   — bounded, verify in the design matrix), XLA compile behavior of
   Hermite kernels.

## Environment notes

- OWNER DIRECTIVE (2026-08-24): engine/test/benchmark runs use the
  `tftwogpu` conda env
  (`/home/chakwong/anaconda3/envs/tftwogpu/bin/python`,
  TF 2.20.0-dev self-built). Two GPUs: prefer the RTX 4080 SUPER
  (`CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1`, probe-verified
  mapping), fall back to the RTX 5080 (`CUDA_VISIBLE_DEVICES=0`) when
  the 4080 is unavailable. Memory growth mandatory before GPU init
  (fail closed, record in manifests). Supersedes earlier `tf-gpu`
  references. U-HERM-1 re-verified green under tftwogpu (6/6, CPU-only
  diagnostic mode).
- latexmk absent; build with `pdflatex; bibtex; pdflatex x2` in docs/.
- MathDevMCP CLI at PYTHONPATH=/home/chakwong/MathDevMCP/src, ~3 min per
  label audit; summary-only output still includes full obligation JSON.
- Scratchpad dir of this session was uncreatable (permission); temp
  outputs went to /tmp.
