# GenUT Score Audit — Follow-Up Handoff Brief

Date: 2026-07-30
Audience: the next agent (Claude or Codex worker) acting on the GenUT score audit
Source audit: `docs/plans/bayesfilter-genut-score-computation-audit-result-2026-07-30.md`
Status: `HANDOFF_OPEN_WORK_ITEMS_NO_EXECUTION_STARTED`

## One-Paragraph Context

The 2026-07-30 audit re-derived the full recursive forward-sensitivity (JVP)
chain of the staged GenUT candidate
(`transition -> likelihood -> Sinkhorn row quotient -> Contract E-Chol ->
diagonal/pairwise higher-moment correction`) against
`docs/chapters/ch32c_entropic_ot_sinkhorn.tex` and found the **derivative
mechanics correct**: the score is the total derivative of the executed finite
scalar on the fixed branch (37/37 focused tests pass, float64 JVP parity
~1e-10 per stage, same-scalar FD gate). What remains are documentation gaps,
one interface risk, and open empirical score problems. This brief lists the
work items, their acceptance criteria, and the guardrails you must not cross.

Audit basis was the **dirty working tree at commit `fb9a0679`** (uncommitted
edits to `cubature_genut_filter.py`, `higher_moment_contract_e.py`,
`cubature_genut_adapters.py`, `cubature_genut_candidate.py`). If those files
change before you start, re-run the verification command below and re-check
the line anchors; do not assume the audit transfers.

## Verification Command (Rerun Before And After Any Change)

```bash
CUDA_VISIBLE_DEVICES=-1 python -m pytest \
  tests/highdim/test_higher_moment_contract_e.py \
  tests/highdim/test_cubature_genut_filter.py \
  tests/highdim/test_cubature_genut_candidate.py \
  tests/highdim/test_cubature_genut_adapters.py -q
# expected: 37 passed (CPU-only is deliberate; GPU work requires escalated permissions)
```

## Work Queue (Priority Order)

### W1 — Extend the chapter to cover the executed pairwise stage (Finding F1, doc-only, no runtime change)

The pairwise empirical-target correction is implemented, claim-bearing
(both 2026-07-30 trials), and JVP-parity-tested, but the chapter's algorithm
statement and score proposition cover only the diagonal stage.

Edit `docs/chapters/ch32c_entropic_ot_sinkhorn.tex`:

1. In section `sec:bf-eot-higher-moment-contract-e`, add a subsection stating
   the pairwise map actually executed by
   `bayesfilter/highdim/higher_moment_contract_e.py:217-337`:
   - residuals on ordered co-skew `E[z_i^2 z_j]` and unordered co-kurtosis
     `E[z_i^2 z_j^2]` (off-diagonal, masked);
   - direction = gradient of
     `1/2 [ sum_{ordered i != j} R3^2 + sum_{unordered i<j} R4^2 ]`
     (this convention is why the co-kurtosis term carries `2 z (R4 z^2)` and
     not `4 z (R4 z^2)` — do not "fix" the factor; it is absorbed by the
     tuned strength after RMS normalization);
   - projection removing the empirical mean component and the
     covariance-tangent component (the projected direction's cross-moment
     with `z` is antisymmetric, so `dCov = 0` at first order);
   - RMS normalization with floor, fixed strength step, full
     restandardization each iteration.
2. Extend Prop. `bf-eot-hm-score` (lines 2342–2371): add
   `pairwise_correction_steps`, `pairwise_strength`, `pairwise_floor`, and
   the mask tensors to the fixed-control list, and the pairwise direction,
   projection, RMS, and restandardization tangents to the enumerated tangent
   contents. Alternative (weaker but acceptable): scope the proposition
   explicitly to `pairwise steps = 0` and add a separate proposition for the
   pairwise stage.
3. Reword lines 2216–2218 ("Mixed third- and fourth-order tensors are
   deliberately not formed"): with pairwise controls on, `d x d`
   co-skew/co-kurtosis matrices ARE formed; only dense 3-/4-tensors are not.
4. The step enumeration at lines 2320–2335 must mention the optional pairwise
   loop between the diagonal loop and the map-back, matching
   `higher_moment_contract_e.py:722-756`.

Acceptance: `cd docs && latexmk -pdf -interaction=nonstopmode -halt-on-error
-outdir=/tmp/bayesfilter-latex main.tex` passes; no new exactness claim is
introduced (the pairwise stage gets the same "same-scalar total derivative,
not exact-posterior score" boundary); MathDevMCP audit optional but useful.

### W2 — Make the fixed-design score contract explicit (Finding F2, docstring + doc, no behavior change)

`_restore_cloud_jvp_core` hard-zeros the residual-design and ridge tangents
(`bayesfilter/highdim/cubature_genut_filter.py:371-372`). Exact for the
implemented constant designs; silently a **partial derivative** if anyone
ever passes a theta-dependent design (the chapter's adaptive GenUT residual,
lines 1786–1790, is exactly such a variant and is NOT implemented).

1. Add to the `finite_value_score` docstring: "The residual design and ridge
   are contractually theta-independent constants; their tangents are fixed to
   zero. Passing a parameter-dependent design yields a partial derivative and
   is outside the score claim."
2. Add one sentence to the chapter near Prop. `bf-eot-hm-score` and one to
   the GenUT residual paragraph (1786–1790) noting the implemented route
   restricts to fixed designs and the adaptive variant would need design
   tangent inputs.
3. Same caveat class applies to the explicit teacher-target hooks of
   `higher_moment_shape_jvp`: total-score correctness requires the caller to
   supply **total** target tangents (chapter already says this at
   2867–2874 for the TT teacher; mirror it in the docstring).

Acceptance: focused tests still 37 passed; no runtime behavior change.

### W3 — Austria SIR value/score tradeoff ladder (Finding F3, EXPERIMENT — requires a plan first)

Do NOT start runs without writing an experiment plan under `docs/plans/`
using the repo templates. This is a serious campaign under the global policy
(research-decision run, GPU, multi-seed).

Framing from the audit and the 2026-07-30 result
(`bayesfilter-austria-sir-pairwise-moment-genut-score-trial-result-2026-07-30.md`):

- Question: does a pairwise setting near steps 4, strength 0.01–0.02 retain
  the large score-variance reduction while keeping the mean finite value
  within the predeclared baseline-SE gate, and where does the
  `log_kappa_scale` discrepancy localize?
- Known state: diagonal-only score SDs `3435.6 / 1272.4 / 302.0`; pairwise
  (4, 0.02) reduced SDs `94.1x / 71.6x / 14.6x` but shifted the value by
  `1.260 ~ 7.9` baseline SEs and its `log_kappa_scale` interval excludes the
  SGQF diagnostic.
- Mechanism hypothesis (descriptive): replicated cubature residual rows have
  zero off-diagonal co-kurtosis versus Gaussian one; the injected cross-moment
  shape error recycles through resets and destabilizes a tangent mode.
- Mandatory plan content: evidence contract; promotion criterion (variance
  reduction retained AND value gate passes) versus promotion vetoes (value
  shift, nonfinite, OT/reset gates); continuation veto separate from
  promotion veto; fresh disjoint tuning data (LEDH per-scope tuning rule: a
  changed strength/steps grid is a new tuning scope; never tune on claim
  seeds/observations); N=1008, 16 untouched claim seeds convention; fresh
  versioned output root; run manifest.
- SGQF and Zhao–Cui are same-target diagnostics, NOT oracles. Do not use
  them as tuning objectives or as score-accuracy proof. A stronger reference
  (independent online SIR score teacher) is worth costing but only if its
  target/event order provably matches.

Boundary: if the plan implies materially expanded compute or a new reference
implementation, stop and ask the owner before executing.

### W4 — Generalize the design-selection helper (Finding F8, small code fix)

`docs/benchmarks/run_moment_retuned_genut_whole_leaderboard.py:91-102`
special-cases `dim >= 18`, but positive Gaussian-moment GenUT replication is
infeasible for every `d >= 4` (`w0 = 1 - d/3 < 0`); for `4 <= d <= 17` the
helper raises at setup instead of falling back to the cubature design.
Replace the threshold with a positivity/representability check (attempt
`replicate_positive_genut`, fall back to `cubature_design` on the
signed-weight `ValueError`, and label `design_family` accordingly). Behavior
must be bitwise-unchanged for the currently exercised dims {1, 2, 3, 18}.

Acceptance: existing leaderboard artifacts untouched; a small unit test for
d in {1,3,4,17,18} of the helper's design family; focused tests still pass.

### W5 — Check the legacy reporting repair (Finding F6, read-only check)

The 2026-07-21 review (verdict REVISE) found HMC-chain-scaled relative score
errors presented as physical-score intervals in
`docs/benchmarks/run_lgssm_recursive_score_matched_comparison.py` /
`run_lgssm_cubature_genut_fp32.py` reporting. Verify whether the relabeling
or raw-score promotion ever landed; record the answer (one paragraph in a
short note or as an addendum to the audit file). Do not rewrite historical
artifacts.

## Hard Guardrails (Do Not Cross)

1. **Revoked evidence:** never cite the 2026-07-21 exact-SV score
   bias/variance ladder (or `cubature_genut_model_claim_20260721/`,
   `cubature_genut_exact_sv_n1000_20260721/`,
   `cubature_exact_sv_score_ladder_20260721/`) as SV scientific evidence.
   Those observations were non-DGP; see
   `bayesfilter-exact-sv-nondgp-fixture-demotion-correction-2026-07-22.md`.
   Only derivative-mechanics/engineering evidence survives from them.
2. **No promotion:** pairwise correction stays opt-in. No default,
   leaderboard, HMC, or NAWM promotion from any of this work.
3. **Statistical language:** score-variance reductions are scope-specific
   (fixed data, 16 particle seeds, declared bootstrap). Do not write
   "better/superior/improved score accuracy" anywhere — no exact Austria
   score oracle exists. Use "passed the screen / viable / descriptively
   favorable".
4. **Backend:** TensorFlow/TFP only on runtime paths; NumPy is
   diagnostic-only (tests, references). No autodiff or finite differences on
   the claim score path (`test_generic_core_replays_and_has_no_autodiff_or_runtime_fd`
   enforces this).
5. **Artifacts:** never overwrite 2026-07-23 or 2026-07-30 artifacts; every
   new launch gets a fresh versioned output directory.
6. **GPU:** any GPU probe/run needs escalated permissions; non-escalated GPU
   failures are sandbox evidence only. CPU-hidden runs must set
   `CUDA_VISIBLE_DEVICES=-1` and say so in the artifact. Enable and verify
   TF memory growth before GPU initialization; fail closed otherwise.
7. **Do not re-tune on claim data** and do not relax the value gate after
   seeing results; a failed claim triggers fresh scope-specific tuning under
   the campaign budget (LEDH per-scope tuning rule in `CLAUDE.md`).
8. **Do not "fix" the pairwise factor-of-2** (see W1); it is the declared
   ordered/unordered convention, and the executed map is what the JVP and
   the 2026-07-30 claims differentiate. Changing it would silently change
   the finite estimator and orphan the existing tuning artifacts.

## What Is Already Settled (Do Not Redo)

- Chain-rule completeness of the executed route (weights, cost-scale max
  branch, Sinkhorn recursion, row quotient, Contract E forward/JVP, diagonal
  and pairwise corrections, map-back, adapters incl. Austria RK4 tangents):
  audited line-by-line, `correct` on the fixed branch.
- JVP parity and same-scalar FD gates: passing as of this handoff.
- GenUT positivity boundary: Gaussian GenUT replication infeasible for
  `d >= 4`; Austria d=18 correctly uses the cubature design.
- KSC-SV pairwise structural no-op at `d = 1`: exact, test-pinned.

## Open Questions For The Owner (Ask Before Acting)

1. Budget and priority for W3 (Austria tradeoff ladder) versus documentation
   items W1/W2 — W1/W2 are cheap and unblock the chapter's claim hygiene;
   W3 is the scientifically material one.
2. Whether an independent SIR score teacher/reference implementation is in
   scope (cost and event-order matching are nontrivial).
3. Whether the uncommitted working-tree edits to the four audited files
   should be committed before follow-up work (the audit anchors assume them).
