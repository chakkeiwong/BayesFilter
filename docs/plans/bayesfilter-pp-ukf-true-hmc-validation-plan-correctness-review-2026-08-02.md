# PP-UKF True HMC Validation Plan Correctness Review

Date: 2026-08-02

Reviewed document:
`docs/plans/bayesfilter-pp-ukf-true-hmc-validation-plan-2026-07-22.md`
(including its Execution Status, Partial Execution Result, and Terminal
Closure sections).

Verdict: `CORRECT_AT_DECLARED_SCOPE_WITH_MINOR_FINDINGS`

The plan's design, execution claims, and terminal closure are supported by the
inspected code and artifacts. Every quantitative claim I checked reproduced
exactly. Five minor findings are recorded below; none invalidates the plan's
declared conclusions, and none of the plan's nonclaims is exceeded by its own
text.

## Scope of this review

This review checked four things: (1) internal soundness of the plan's evidence
contract and gates; (2) whether the shared controller and diagnostics actually
implement the declared policy; (3) whether the terminal artifacts support the
plan's execution and closure claims; (4) whether the posterior-validation
methodology supports the `L=12`/`L=17` equivalence statement at its declared
scope. It did not re-derive the PP-UKF filter target itself (see Not checked).

Evidence inspected:

- Plan, terminal result note (`...true-hmc-validation-terminal-result-2026-07-30.md`),
  posterior-validation plan and result notes (2026-07-30).
- `bayesfilter/inference/neutra_hmc.py` (sequential controller and
  continuation), `bayesfilter/inference/hmc_convergence.py` (R-hat/ESS),
  `bayesfilter/inference/batched_value_score.py` (target/score binding),
  `docs/benchmarks/run_pp_ukf_true_hmc_validation_20260722.py` (campaign
  driver), `docs/benchmarks/run_pp_ukf_posterior_validation_20260730.py`
  (posterior harness).
- Artifacts: attempt-11 `public_result.json`, `run_manifest.json`,
  `artifact_hashes.json`; posterior-validation attempt-06/attempt-07
  `public_result.json` and `run_manifest.json`; the July-16 affine plain-HMC
  reference `result.json`.

Checks executed for this review (CPU-only diagnostic context; GPU devices not
needed because no sampling or GPU probe was run):

- Independent SHA-256 re-hash of the two terminal posterior-validation files:
  both match the hashes recorded in the result note.
- Independent re-hash of 4 randomly sampled cumulative retained `.tftensor`
  archives referenced by attempt-11: all match (the artifact's own
  `archive_validation` records 332/332 with zero mismatches).
- Recount of all 300 posterior-validation checks from attempt-07 JSON:
  reproduces the note's per-candidate table exactly (26/30, 27/30, 30/30,
  24/30, 25/30, 30/30, 19/30, 23/30, 21/30, 20/30; zero disagreements; all
  point estimates within tolerance).
- Attempt-6 vs attempt-7 payload equality excluding `finished_utc` and
  `wall_seconds`: exactly `True`, as the harness-repair ledger claims.
- Focused harness tests re-run today with `CUDA_VISIBLE_DEVICES=-1`:
  `5 passed`, matching the result note.
- Budget arithmetic: `42,746.868394 + 8,629.763232442 = 51,376.631626442 s`
  equals the recorded aggregate, under the `86,400 s` cap; prior-tuning
  headroom `14,400 − 13,750.560450 = 649.44 s` matches "about 649 s".

## Claim-by-claim verification

| Plan claim | Evidence inspected | Verdict |
| --- | --- | --- |
| Sequential policy: 4 chains, warmup min 2,000 with recent-window R-hat `<=1.05`, retained min 1,000 / max 10,000 with modern R-hat `<=1.01`, cumulative growth | `SequentialNeuTraHMCConfig` defaults and loop in `neutra_hmc.py`; all ten artifact rows show warmup exactly 2,000 and retained gates on cumulative draws | correct |
| Modern R-hat definition | `hmc_convergence.py`: Blom normal scores of pooled ranks, split chains, folded variant via pooled-median absolute deviation, final `max(rank, folded)`; matches Vehtari et al. (2021) construction | correct |
| Bulk/tail ESS gates (`>=1000` / `>=400`) declared before the full run | Thresholds frozen in `RankNormalizedHMCThresholds`, recorded in artifact `thresholds`, declared in the plan's Execution Status before the terminal campaign | correct |
| ESS formula parity after the real-FFT repair | `_real_fft_cross_chain_ess` implements vâr⁺ = (N−1)/N·W + B/N with Geyer initial-positive-pair truncation on split chains; repair note records max direct difference from TFP `3.64e-11`; pass status reproduced by the posterior harness | correct |
| Non-finite log-acceptance is a hard veto; finite extreme values are explanatory only | Health gate in `_summarize_batched_hmc_output` requires finiteness; extreme count labeled `explanatory_only_not_a_veto_or_divergence` | correct |
| Native divergence recorded as unavailable, never zero | TFP `HamiltonianMonteCarlo` kernel results expose no `has_divergence` field (a NUTS field); artifact records `not_exposed_by_tfp_hamiltonian_monte_carlo` | correct |
| Coverage candidates inherit parent epsilon bit-for-bit | L=12,14 equal L=13's epsilon; L=17,19 equal L=18's; L=24 equals L=25's, to full float precision in the artifact | correct |
| Identity bindings | Target signature `d3ed745b...` and transport SHA-256 `b7a558db...` identical across plan, driver constants, attempt-11 artifact, and the reference's `mathematical_target_signature`/scope | correct |
| Fresh execution partition, honestly scoped | Partition block records new initial-state seed/hash and warmup/retained seed roots with `tuning_draws_reused: false`; plan explicitly disclaims a new observation-data split | correct as declared; seed disjointness accepted at artifact level (see Not checked) |
| Continuation mechanics preserve chain law | `run_retained_neutra_hmc_continuation` resumes from the last prefix latent state, enforces `prefix_count == chunk_index * chunk_size`, verifies prefix equals concatenated chunk archives by hash and value, and recomputes gates on full cumulative draws (final diagnostics at 10,000/6,500/4,500 draws for L=9/12/17) | correct |
| Terminal candidate table | All ten rows (epsilon, retained draws, max R-hat, min bulk/tail ESS) recomputed from attempt-11 JSON match the terminal note exactly | correct |
| Budget compliance | Aggregate accounting chain is monotone and internally consistent (attempt-09 prior 17,424.7 + wall 24,978.8 ≈ attempt-11 prior less attempt-10); worst-case reconstruction stays far below 86,400 s | correct |
| Posterior closure: `L=12`,`L=17` equivalent on 30/30; eight inconclusive; zero disagreement | Recount of attempt-07 JSON; three-way classification logic in the harness matches the stated rule (interval inside margin ⇒ established; wholly outside ⇒ disagreement; else inconclusive) | correct at declared screen scope |

## HMC correctness decomposition

Per-component analysis of the sampling claim:

- **Proposal generation.** TFP leapfrog with fixed step size, fixed leapfrog
  count, and identity mass matrix (default standard-normal momentum), matching
  the fixed-identity-metric contract. Each leapfrog sub-step is a shear, so
  the proposal map is volume-preserving and, by the palindromic structure,
  reversible under momentum flip — regardless of whether the force field is an
  exact gradient.
- **Target evaluation.** `reviewed_value_score_target_fn` binds the adapter's
  value as `target_log_prob`. The sampled invariant distribution is therefore
  exactly the density defined by the adapter's value path. The attempt-11
  `recomposition` check reports maximum value and score error `0.0` against
  the frozen kernel construction at the checked point.
- **Leapfrog force.** The adapter score enters via `tf.custom_gradient`. Given
  the two properties above, any score inaccuracy degrades acceptance and
  mixing only; it cannot bias the invariant distribution, because the MH
  ratio uses exact value evaluations.
- **MH correction.** Standard TFP `HamiltonianMonteCarlo`
  (Metropolis-corrected); `state_gradients_are_stopped=True` affects only
  gradients through sampling outputs, not kernel dynamics.
- **Transformed-target composition.** Whether each frozen NeuTra pullback
  includes the correct log-determinant is not re-derived here; the
  same-mathematical-target agreement between the flow-transported candidates
  and the affine plain-HMC reference in model coordinates is end-to-end
  evidence consistent with a correct composition, within the screen's
  statistical resolution (an omitted log-det term would generically move the
  model-coordinate distribution far outside 0.10–0.15 SD margins).

The complex-cast warning history is correctly localized: the cast occurred in
TFP's FFT-backed ESS diagnostic, not in the target or transition path, so it
could not have altered the sampled density; the terminal note's own red-team
alternative on this point is resolved by that source localization plus the
post-run parity repair.

## Findings

1. **Missing promised sensitivity report (minor, unfulfilled diagnostic).**
   The posterior-validation plan's default-and-assumption audit lists
   "sensitivity report" as the early diagnostic for the block-bootstrap
   choice; no block-length sensitivity analysis exists in the harness or
   artifacts (zero occurrences). The block rule was predeclared and fixed, so
   the screen stands, but the `L=12`/`L=17` equivalence statements are
   conditional on the declared block rule `max(20, floor(sqrt(n)))`; longer
   blocks would widen intervals and could, in principle, demote 30/30 rows to
   inconclusive. A cheap one-off sensitivity rerun (e.g., doubled block
   length) would close this.
2. **Four-cluster chain bootstrap is coarse (minor, inherent).** Resampling 4
   chains with replacement gives a very coarse between-chain component. This
   is mitigated by the R-hat `<=1.01` gate (small between-chain variation) and
   by the method's declared status as a compatibility screen, not a proof.
3. **Optional-stopping selection on retained diagnostics (minor, inherent to
   the owner policy).** Retained sampling stops at the first checkpoint where
   noisy R-hat/ESS pass, which mildly inflates pass probability relative to a
   fixed-length test. The plan never upgrades screen passage to convergence
   proof, and stopping was driven by convergence diagnostics only — not by
   posterior agreement — so the later equivalence screen is not directly
   selected on. Acceptable at the declared evidence class; worth remembering
   if a screen result is ever cited as convergence evidence.
4. **Shared reference bootstrap replicates across candidates (note only).**
   All ten comparisons difference against the same 1,000 reference replicates
   (seed offset +100). Each comparison is marginally valid; outcomes are
   dependent across candidates, which would matter only for cross-candidate
   simultaneous claims that the notes correctly do not make.
5. **Superseded statement in the 2026-07-22 partial note (documentation).**
   The partial-result statement that the three preserved warmup prefixes "are
   not rerun" was overtaken by the reboot: the terminal artifact contains full
   fresh 2,000-draw warmups for all ten candidates. The plan's Terminal
   Closure records the repair, and the superseded prefixes were never claim
   evidence, so this is a documentation-chain note, not a defect.

Additional nits: `tf32_enabled: true` with `float64` tensors is inert for the
HMC path (TF32 affects float32 matmul only) and the contract required only
that the setting be recorded, which it is. The equivalence margins
(0.10/0.15 × reference SD) are correctly registered as "Hypothesis/screen" in
the audit table and never promoted to a correctness criterion.

## Decision table

| Decision | Primary criterion status | Veto status | Main uncertainty | Next justified action | What is not concluded |
| --- | --- | --- | --- | --- | --- |
| Accept the plan and its terminal closure as correct at declared scope; retain `L=12`,`L=17` as an unranked compatible set | All checked plan claims reproduced from artifacts and code; controller, diagnostics, and harness match the declared policy | No review veto; findings 1–5 are minor | Block-length sensitivity of the two 30/30 equivalence rows (Finding 1); comparator is not an exact oracle | If the equivalence result is ever load-bearing for a promotion, run the missing block-length sensitivity check; otherwise no rerun | No exact-posterior correctness, no ranking, no default/production readiness, no cross-target robustness — same nonclaims as the plan |

## Inference status of this review

| Evidence class | Status |
| --- | --- |
| Reproduced quantitative claims | Candidate tables, thresholds, budget arithmetic, signatures, hashes, attempt-6/7 identity, focused tests — all reproduced |
| Verified by code reading plus derivation | Sequential policy, modern R-hat/ESS construction, health/energy-veto semantics, continuation chain law, HMC invariance under inexact score |
| Verified at artifact level only | Tuning-seed disjointness flag; 332-file hash sweep (4 independently re-hashed, remainder per artifact receipt); carry-in wall times of crashed attempts 07/08/10 |
| Not checked | PP-UKF filter/target implementation math; frozen transport log-det composition by derivation; the July-16 reference campaign's own internals |

## Post-review red team

The strongest way this review could be wrong: the mathematical target
signature could bind a *wrong* PP-UKF implementation consistently across
candidates and reference, in which case every check above still passes — the
plan itself states this limitation and the closure claims only same-target
compatibility. Second: both the candidates and the reference could share a
subtle transport/log-det convention error in the same direction; only an
independent exact or otherwise-justified posterior reference would detect it.
Weakest evidence relied on here: tail-quantile equivalence for the two 30/30
candidates under a single predeclared block rule (Findings 1–2). The review
verdict would be overturned by a hash-valid rerun of the posterior screen
under a reasonable alternative block length that moves any `L=12`/`L=17`
interval wholly outside its margin, or by a demonstrated mismatch between an
adapter's value path and the signed target construction.
