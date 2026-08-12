# GenUT Score Computation Audit

Date: 2026-07-30
Status: `AUDIT_COMPLETE_DERIVATIVE_MECHANICS_CONSISTENT_DOC_GAPS_AND_OPEN_EMPIRICAL_ISSUES`

## Scope And Object Audited

This note audits the score (parameter-derivative) computation of the
"GenUT algorithm" as it actually exists in this repository: the opt-in staged
candidate

```text
transition -> likelihood increment -> entropic OT row quotient
-> Contract E-Chol reset -> GenUT-motivated diagonal (and optional pairwise)
   higher-moment correction -> equal-weight carry
```

with a manual recursive forward-sensitivity (JVP) score and no runtime
autodiff or finite differences. Anchors:

- LaTeX: `docs/chapters/ch32c_entropic_ot_sinkhorn.tex`
  - GenUT design/positivity: lines 1681–1790
  - staged value/variance/score boundary: Prop. `bf-eot-both-residual-boundary`, lines 1792–1830
  - stopped-normalization partial-derivative warning: Prop. `bf-eot-stopped-normalization-partial`, lines 1916–1965
  - higher-moment Contract E candidate: Sect. `bf-eot-higher-moment-contract-e`, lines 2062–2380
  - total-score proposition: Prop. `bf-eot-hm-score`, lines 2342–2371
  - TT-teacher pairwise targets and total score: lines 2740–2874
- Code: `bayesfilter/highdim/cubature_genut_filter.py` (value/score core),
  `bayesfilter/highdim/higher_moment_contract_e.py` (diagonal + pairwise JVP),
  `bayesfilter/highdim/ledh_contract_e_reset_tf.py` (Contract E forward/JVP),
  `bayesfilter/highdim/cubature_genut_candidate.py` (designs, route identity),
  `bayesfilter/highdim/cubature_genut_adapters.py` (model value/tangent pairs).

The classical GenUT sigma-point filter of Ebeigbe et al. is *not* the executed
object; the chapter says this explicitly (lines 2066–2074). GenUT supplies the
selected standardized diagonal (and now pairwise) moment targets and the
residual-design motivation.

Audit basis: working tree at commit `fb9a0679` **with uncommitted
modifications** to `cubature_genut_filter.py`, `higher_moment_contract_e.py`,
`cubature_genut_adapters.py`, `cubature_genut_candidate.py` (the audited state
is the working tree, not HEAD).

## Evidence Contract

| Item | Status |
|---|---|
| Question | Is the computed GenUT-route score the total derivative of the executed finite likelihood scalar, as the chapter claims, and what score problems remain? |
| Baseline | Chapter propositions and differentiation contracts in `ch32c` |
| Primary criterion | Line-level derivation match between each chapter equation and the executed JVP, plus existing parity/FD gates |
| Veto diagnostics | A dropped chain-rule term on the claim path, a value/tangent program mismatch, or an unsupported exactness claim |
| Explanatory diagnostics | Focused test suite, historical campaign artifacts, review notes |
| Nonclaims | No exact-posterior score, no unbiasedness, no method superiority, no default/HMC promotion is established or revoked by this audit |
| Artifact | This note |

## What The Document Claims (Inventory)

1. **Same-scalar total score, not exact-posterior score.** Prop.
   `bf-eot-hm-score` (2342–2371): with innovation rows/ordering, residual
   design, Sinkhorn/balance counts, `epsilon`, `lambda`, `rho`, `delta`, `S`
   fixed, SPD/positivity validity holding, and the max branch in the cost
   scale locally fixed, the recursive tangent that includes the weights,
   OT kernel/scalings, target moments, directions `q3,q4`, Jacobians `J_a`,
   damped coefficients, and both standardization maps is the total derivative
   of the same finite scalar `sum_t l_t^N(theta)`. Explicitly *not* an exact
   filtering likelihood/score theorem.
2. **Stopped normalization is a partial derivative.** Prop.
   `bf-eot-stopped-normalization-partial` (1916–1965): treating the transport
   cost mean/scale as constants drops mean and scale terms and is wrong
   relative to a total-score claim.
3. **GenUT residual candidate score.** Lines 1786–1790: the *adaptive* GenUT
   residual's total score requires differentiating the empirical third/fourth
   moments, square-root gauge, point coordinates, weights, and restoration.
4. **Positivity boundary.** Prop. `bf-eot-genut-positivity` (1754–1773): a
   negative central weight `w0` makes GenUT a signed rule, unusable as a
   positive OT marginal; exact equal-weight replication needs representable
   masses.
5. **TT-teacher pairwise targets.** Lines 2740–2874: ordered co-skew
   `E[z_i^2 z_j]`, symmetric co-kurtosis `E[z_i^2 z_j^2]`, mask semantics, and
   a total-score proposition requiring teacher-target tangents.

## Code Audit: Verified Chain-Rule Ledger

Each stage was re-derived in the repo's notation and compared line-by-line.
Classification: `correct` = derivation reproduced and matches the executed
map on the fixed branch.

| Stage | Code anchor | Doc anchor | Verdict |
|---|---|---|---|
| Log-weights, increment `logsumexp`, normalized-weight tangent `dα_i = α_i(v̇_i − Σα v̇)` | `cubature_genut_filter.py:533–544` | eq. 2082–2092 | correct |
| Squared-distance cost tangent | `cubature_genut_filter.py:84–89` | eq. 2102–2105 | correct |
| Cost scale `max(mean C, 1e-3)` with active-branch tangent (mean term retained; zero on floored branch) | `cubature_genut_filter.py:90–96` | Prop. 1916–1965; fixed-max assumption 2346–2348 | correct (branch-conditional, as declared) |
| Kernel exponent quotient rule incl. scale term | `cubature_genut_filter.py:97–104` | same | correct |
| Unrolled Sinkhorn + terminal balance iterations, `+1e-7` floors as constants, full left/right tangent recursion | `cubature_genut_filter.py:112–172` | eq. 2107–2113 | correct |
| Coupling product rule; row-mass quotient barycenter tangent | `cubature_genut_filter.py:173–190` | eq. 2118–2128 | correct |
| Fail-closed marginal gates (`m_i > 1e-7`, post-quotient column TV ≤ 1e-4) | `cubature_genut_filter.py:194–207` | 2124–2128 | correct |
| Contract E forward: `G=sym(P_w−P_+)`, `B=chol(G+λI)`, realized injected covariance (cross terms included), `A=L_w L_E^{-1}`, reset mean `μ_w` | `ledh_contract_e_reset_tf.py:44–166` | eq. 2133–2156, Prop. 2158–2185 | correct |
| Contract E JVP: weighted/uniform moment tangents, symmetric Cholesky JVP `L·Φ(L^{-1} dA L^{-T})`, `dA=(dL_w−A dL_E)L_E^{-1}`, full particle tangent | `ledh_contract_e_reset_tf.py:263–344` | Prop. `bf-eot-hm-score` item list | correct |
| Design and ridge tangents passed as zeros for the fixed design/ridge | `cubature_genut_filter.py:371–372` | "Fix … residual design, λ" 2343–2345 | correct **for fixed designs only** (Finding F2) |
| Diagonal targets `s_a,k_a` from weighted cloud with un-ridged `C_w`, weight-tangent included; right-solve JVP | `higher_moment_contract_e.py:528–544` | eq. 2206–2214 | correct |
| Directions `q3=u²−1−m₃u`, `q4=u³−m₃−m₄u`; zero mean/inner-product property | `higher_moment_contract_e.py:372–384` | eq. 2229–2233, Prop. 2260–2286 | correct |
| 2×2 moment Jacobian rows `3⟨u²q_j⟩, 4⟨u³q_j⟩`; damped solve `c=ρ(JᵀJ+δI)^{-1}Jᵀd`; solve-tangent `ċ=ρN^{-1}(ṙ−Ṅ N^{-1} r)` | `higher_moment_contract_e.py:386–458` | eq. 2234–2248 | correct |
| Per-iteration re-centering/rewhitening JVP; map-back `x=μ_w+u C_wᵀ` with `C_w` tangent | `higher_moment_contract_e.py:459–483, 757–762` | eq. 2252–2258, Prop. 2288–2309 | correct |
| Zero-step passthrough (route identical to Contract E when `S=0` and pairwise steps 0) | `higher_moment_contract_e.py:665–699` | 2073–2074 | correct (test-pinned equality) |
| Pairwise direction = gradient of ½[Σ_{ordered i≠j} R3² + Σ_{unordered i<j} R4²]; first-order mean/covariance-neutral projection (residual cross term reduced to its antisymmetric part, so `dCov = 0` at first order); RMS normalization tangent; restandardization | `higher_moment_contract_e.py:217–337` | index semantics 2748–2761; map equations **not in the chapter** (Finding F1) | correct as an executed map; doc gap |
| Score accumulation = Σ_t increment tangents; NaN fail-closed when any validity gate fails | `cubature_genut_filter.py:657–659, 786–791` | Prop. 2187–2202, 2342–2371 | correct |
| Model adapters: exact-SV, KSC, generalized SV, diagonal LGSSM (incl. stationary-init and `q·noise` parameter terms), reduced SIR, Austria SIR d=18 (RK4 half-step-k4 source variant with stage-consistent tangents; κ/ν/observation-variance parameter terms), predator–prey | `cubature_genut_adapters.py` (LGSSM 318–423, Austria 632–800) | adapter contract 2082–2085 | correct (re-derived; FD-gated) |

Two details that superficially look like bugs but are not:

- The pairwise co-kurtosis term contributes `2·z⊙row4`, not `4·z⊙row4`.
  This is the exact gradient under the *unordered-pair* co-kurtosis
  convention the chapter declares (one requested unordered pair supplies both
  matrix entries, lines 2748–2756); the ordered co-skew part carries both its
  row and column terms. Any residual constant factor is absorbed by the tuned
  `pairwise_strength` after RMS normalization.
- `transition_value` for RK4 models recomputes the state pass discarded by
  `transition_tangent`; wasteful but value/tangent-consistent.

## Verification Run (This Audit)

```text
CUDA_VISIBLE_DEVICES=-1 python -m pytest \
  tests/highdim/test_higher_moment_contract_e.py \
  tests/highdim/test_cubature_genut_filter.py \
  tests/highdim/test_cubature_genut_candidate.py \
  tests/highdim/test_cubature_genut_adapters.py -q
# 37 passed, 14.13s   (deliberate CPU-only; GPU hidden by CUDA_VISIBLE_DEVICES=-1)
```

These include float64 `tf.autodiff.ForwardAccumulator` parity at
`atol/rtol ≈ 1e-10..2e-10` for the Sinkhorn JVP, the composed
restore (OT + Contract E) JVP, the diagonal higher-moment JVP, and the
pairwise JVP; a same-scalar central-difference gate for the end-to-end
`finite_value_score`; Austria SIR adapter tangent FD gates; no-autodiff and
no-Python-loop source gates. GPU/XLA claim-path behavior is separately
evidenced by the 2026-07-30 trusted smokes recorded in the trial artifacts
and is not re-established here.

## Findings

### F1 (doc–code gap, medium): the pairwise correction stage is executed and claim-bearing but absent from the chapter's algorithm and score statement

Prop. `bf-eot-hm-score` fixes controls `(ε, λ, ρ, δ, S)` and enumerates only
the diagonal-stage tangent terms (`q3, q4, J_a, c_a`, standardizations). The
implemented and trial-exercised route (Austria SIR and cross-model trials,
2026-07-30) adds `pairwise_correction_steps`, `pairwise_strength`,
`pairwise_floor`, and ordered/symmetric masks, with its own map
(gradient direction over pair residuals, mean/covariance-tangent projection,
RMS normalization, restandardization). The chapter documents pairwise
*targets* and mask semantics in the TT-teacher section (2748–2761) but never
states the empirical-target pairwise map or extends the total-score
proposition to it. Additionally the sentence "Mixed third- and fourth-order
tensors are deliberately not formed" (2216–2218) is stale relative to the
module: with pairwise controls on, `d×d` co-skew/co-kurtosis matrices *are*
formed (still no dense 3-/4-tensors).

Verdict: the executed pairwise score is `correct` as a manual JVP of the
executed map (parity-tested), but the chapter's total-score claim, as
written, covers the executed route only when `pairwise_correction_steps=0`.
Repair: add the pairwise map equations, its controls to the fixed-control
list, and its tangent terms to Prop. `bf-eot-hm-score` (or scope the
proposition explicitly); reword line 2218.

### F2 (interface risk, medium): hard-zero design/ridge tangents make a θ-dependent residual design silently produce a partial derivative

`_restore_cloud_jvp_core` passes `tf.zeros_like(design_batch)` and zero ridge
tangent unconditionally (`cubature_genut_filter.py:371–372`). This is exact
for every implemented design (replicated cubature; replicated positive
Gaussian-moment GenUT — both host-constant). But the chapter's adaptive GenUT
residual variant (1786–1790) builds points from the *empirical* `s_a, k_a`,
which are θ-dependent; if such a design tensor were ever passed to
`finite_value_score`, the returned score would silently drop the design
chain-rule terms — exactly the class of error Prop. 1916–1965 warns about.
No runtime guard can detect θ-dependence of an input tensor. Repair: state
the constant-design contract in the docstring and route identity; the
adaptive-design variant, if ever implemented, needs design tangent inputs.
The same caveat applies to the explicit teacher-target hooks in
`higher_moment_shape_jvp`: correctness of the total score depends on the
caller supplying *total* target tangents (the TT-teacher chapter section and
Prop. `bf-eot-tt-teacher-total-score` state this; the interface cannot check it).

### F3 (open empirical issue, high): recursive-score variance instability on Austria SIR; pairwise repair reduces variance but fails the value gate

From `bayesfilter-austria-sir-pairwise-moment-genut-score-trial-result-2026-07-30.md`
(16 common particle seeds, d=18, T=20, N=1008): diagonal-only score SDs were
`3435.6 / 1272.4 / 302.0` for
`(log_kappa_scale, log_nu_scale, log_observation_noise_scale)`; the pairwise
arm reduced SDs by `94.1x / 71.6x / 14.6x` (paired bootstrap aggregate
variance ratio `0.000468`, CI `[0.000082, 0.063166]`, below one). Promotion
nevertheless failed: the mean finite value shifted by `1.260` (≈ `7.9`
baseline MC SEs), and the `log_kappa_scale` interval `[-35.8, 3.2]` excludes
the SGQF diagnostic value `28.7` (SGQF is a same-target diagnostic, not an
oracle). Mechanism (descriptive, supported by the cross-model note): the
replicated cubature residual rows `±sqrt(d) e_i` have zero off-diagonal
co-kurtosis versus the Gaussian value one, so every reset injects a
cross-moment shape error that recycles through later weights, OT maps, and
increment tangents; an unstable tangent mode results. This is an
*algorithmic score-variance/bias problem of the finite route*, not a
chain-rule defect: same-scalar FD and parity gates pass throughout.
Open questions: value-shift versus score-variance tradeoff, `log_kappa`
bias localization, and a stronger reference/teacher.

### F4 (transfer boundary, supported): pairwise repair is not a universal variance reducer

From `bayesfilter-pairwise-moment-genut-lgssm-ksc-predator-prey-trial-result-2026-07-30.md`:
LGSSM T=50 — every nonzero pairwise arm increased at least one validation
score-coordinate variance (empirical noisy targets `O(N^{-1/2})` replace the
known Gaussian values `(0,1)`; a post-run Kalman diagnostic also showed
larger score RMSE for every nonzero arm); KSC-SV — structural no-op at d=1
(exact, test-pinned); predator–prey — aggregate variance ratio CI includes
one. The Austria repair addresses a *severe omitted-cross-moment
instability*; it is not a default.

### F5 (revoked history, must not be cited): exact-SV "score bias" conclusions from 2026-07-21 are non-DGP and revoked

`bayesfilter-exact-sv-nondgp-fixture-demotion-correction-2026-07-22.md`: the
2026-07-21 exact-SV score bias/variance ladder (including the familywise
score-gate failure and antithetic accuracy interpretation) used observations
drawn directly from `Normal(0,1)`, not the SV DGP. Those score-accuracy
conclusions are wrong relative to an SV scientific claim and revoked; only
derivative-mechanics and engineering evidence (FD parity, residual gates,
GPU placement) survives. Any future audit or plan citing a "persistent GenUT
SV score bias" from those artifacts is citing revoked evidence.

### F6 (reporting-layer, historical, repair not checked): HMC-chain-scaled relative errors were presented as physical-score intervals

The 2026-07-21 review
(`bayesfilter-lgssm-cubature-genut-score-variance-claude-review-result-2026-07-21.md`,
verdict REVISE) found the LGSSM matched-comparison headline tables reported
HMC-chain-scaled relative score errors as the primary score object while raw
physical-score intervals lived only in the JSON; relative blow-ups (e.g.
`phi3` at T=50) were denominator instability (Kalman raw score ≈ `0.302`),
not score failure. Current package-route campaigns report raw scores.
Whether the legacy matched-comparison script itself was relabeled: not
checked in this audit.

### F7 (precision boundary, engineering): claim path is FP32+TF32; derivative parity is certified in float64/CPU and FP32 FD gates at coarser tolerance

Parity at `1e-10` is float64/CPU. The claim path runs FP32 with TF32 GEMMs
and XLA (register-spill warnings observed; one XLA GEMM autotuner layout
failure was repaired by an algebraically identical contraction rewrite,
parity re-checked — cross-model result, Engineering Evidence). The FP32
recursive-vs-FD calibration gate tolerance is `0.05` (observed ≤ `0.0094`).
The total-derivative statement is exact-arithmetic; FP32/TF32 drift is an
accepted, manifest-recorded engineering tolerance, not a proof of agreement
at float64 precision on GPU.

### F8 (minor robustness): `_genut_design(dim)` in `run_moment_retuned_genut_whole_leaderboard.py` special-cases `dim >= 18`

Positive Gaussian-moment GenUT replication is infeasible for every `d ≥ 4`
(`w0 = 1 − d/3 < 0`), not just `d ≥ 18`; for `4 ≤ d ≤ 17` the helper would
raise at setup (fail-closed, so not a correctness bug) instead of falling
back to the cubature design. The Austria d=18 row correctly uses the
replicated cubature design and labels `design_family="cubature"`; the label
"GenUT" for that row refers to the higher-moment correction, not the
residual design. Worth renaming or generalizing the threshold to a
positivity/representability check.

## Separation Of Claim Classes

- **Proved by the chapter and reproduced here (fixed branch, exact
  arithmetic):** the recursive score equals the total derivative of the same
  executed finite scalar for the diagonal route with fixed designs; reset
  mean `μ_w`; ridged covariance identity; exact post-correction first two
  moments after rewhitening.
- **Verified engineering (tests/artifacts):** manual JVP = forward-autodiff
  JVP at float64 for every stage including pairwise; same-scalar FD
  agreement; fail-closed validity; score-increment additivity.
- **Empirical and open:** score variance instability and its pairwise
  repair tradeoff (F3), transfer limits (F4), FP32/TF32 drift magnitude on
  GPU claim paths (F7).
- **Not established anywhere (and not claimed):** exact filtering
  likelihood, exact posterior score, unbiasedness, ranking of GenUT versus
  Contract E/Cubature on score accuracy, HMC readiness.
- **Not checked in this audit:** the Zhao–Cui TT moment-teacher JVP
  internals (`zhao_cui_moment_teacher*.py`; separate tests exist), the
  structural-UKF and reduced-SIR adapter tangents line-by-line (FD tests
  exist), GPU-side numerical drift measurements, and the F6 legacy-report
  repair status.

## Decision Table

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Same-scalar score mechanics: `correct` on the fixed branch for the executed route | Line-level derivation match at every stage + 37 focused tests passing | No dropped chain-rule term found on the claim path | FP32/TF32 GPU drift not re-measured here | none required | not an exact-posterior score |
| Chapter score claim: incomplete w.r.t. executed pairwise route | Prop. `bf-eot-hm-score` control list and tangent enumeration exclude the pairwise stage | Doc-completeness veto (F1) | none | extend the chapter (map equations, controls, score terms) or scope the proposition | no math error in the pairwise JVP itself |
| Fixed-design contract: implicit | zero design tangent is exact only for constant designs (F2) | none today | future adaptive GenUT design | document the contract in docstring + route identity | adaptive-design route neither implemented nor claimed |
| Austria score-variance problem: open | value-shift gate failed despite 14.6–94.1x SD reduction (F3) | promotion veto fired | bias vs variance tradeoff, `log_kappa` discrepancy | fresh disjoint tradeoff ladder around steps 4, strength 0.01–0.02 with value-stability and `log_kappa` diagnostics; stronger reference if affordable | no accuracy or superiority claim |
| Historical SV score-bias narrative: revoked | non-DGP fixture (F5) | hard veto on citing it | none | keep demotion note authoritative | no SV score-bias conclusion exists |

## Nonclaims

No exact higher-moment projection, exact nonlinear likelihood, unbiasedness,
exact posterior score, statistical method ranking, default promotion,
leaderboard promotion, HMC readiness, or NAWM conclusion. Score-variance
reductions cited here are scope-specific (fixed data, 16 particle seeds,
declared bootstrap) and are not evidence of lower score bias.
