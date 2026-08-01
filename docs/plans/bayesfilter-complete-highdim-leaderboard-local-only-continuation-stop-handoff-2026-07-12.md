# Complete High-Dimensional Leaderboard Local-Only Stop Handoff

Date: 2026-07-12

Run ID: `complete-highdim-leaderboard-local-20260712-134906`

Status: `BLOCKED_INVALID_EVIDENCE`

Blocker: `BLOCK_CANONICAL_TARGET_CONTRADICTION_SIR_OBSERVATION_IDENTITY`

## Outcome

The local-only runbook safely advanced through P1-A implementation and
adversarial validation, then stopped before any benchmark-harness edit or GPU
run. The canonical encoder is deterministic and its independent checker
passes, but the SIR bytes it materialized contradict the reviewed Phase 0
authority statement. A deterministic hash of an unauthorized target cannot be
promoted into a canonical target signature.

The frozen harness remains byte-identical to its Phase 0 SHA-256:
`2bd7c4c62773657213ccd488c9e55b96f3f7d6d4a3b00a7aaf2a8fb070031d58`.

## Decision Table

| Field | Result |
| --- | --- |
| Decision | Stop before P1-B closure and before P1-C implementation. |
| Primary criterion | Failed: P1-A requires no contradiction with the Phase 0 target identity. |
| Hard veto | The current fixed-SIR constructor reads `_sir_dataset(81103)`, while Phase 0 says the main SIR row has fixed source observations and explicitly declares no synthetic target-generation seed. |
| Repair trigger | Owner chooses the authoritative SIR observation identity; Phase 0/P1-A bindings and all affected artifacts are then refreshed. |
| Main uncertainty | Whether “fixed source observations” was intended to mean exact author `rng(1)` observations or the already-used fixed BayesFilter seed-81103 bytes. |
| Next justified action | Obtain the target decision and explicit extension approvals below; do not edit or run the six-row harness first. |
| Not concluded | No evaluator correctness, P1-B pass, Zhao-Cui source-faithfulness, GPU/XLA evidence, cell admission, complete leaderboard, ranking, HMC/posterior correctness, or scientific validity. |

## P1-A Evidence

Created artifacts:

- `docs/benchmarks/build_complete_highdim_phase1_canonical_targets.py`;
- `docs/benchmarks/check_complete_highdim_phase1_canonical_targets.py`;
- `tests/highdim/test_complete_highdim_phase1_canonical_targets.py`;
- `docs/plans/artifacts/complete-highdim-leaderboard/phase1-canonical-targets-2026-07-11.json`;
- `docs/plans/artifacts/complete-highdim-leaderboard/phase1-p1a-gate-receipt-2026-07-11.json`.

Checks:

| Check | Result |
| --- | --- |
| Canonical builder deterministic `--check` | `PASS_PHASE1_CANONICAL_TARGET_BUILDER_CHECK` |
| Independent reconstruction, framed digest, semantics, and source-hash check | `PASS_PHASE1_CANONICAL_TARGET_INDEPENDENT_CHECK` |
| Adversarial target checks | `7 passed` |
| Python compile | Pass |
| JSON parse | Pass |
| Scoped diff check | Pass |

These checks prove that the candidate bytes are encoded consistently and match
the current constructors. They do not prove that those constructors implement
the target authorized by Phase 0.

The diagnostic candidate canonical-target artifact SHA-256 is
`ba3d93521d8372acb2312447a70a6ff0dbf90890b945995527162e4f79e61cfd`.
The fail-closed P1-A receipt SHA-256 is
`3299d3b797aa41b028fe77ec4d3aabb639fa176da73767fe3b4905a2d614ff67`.

## SIR Contradiction

Phase 0 authority states:

- `docs/plans/bayesfilter-complete-highdim-leaderboard-phase0-boundary-freeze-subplan-2026-07-11.md:64` assigns explicit target-generation seeds to five rows;
- the same statement says fixed Austria SIR has fixed source observations and
  no synthetic dataset seed.

Current computation states:

- `docs/benchmarks/benchmark_p8p_parameterized_sir_gradient.py:324` builds
  fixed-SIR inputs through `_build_actual_sir_tensors`;
- that route obtains observations from the BayesFilter `_sir_observations()`;
- `scripts/filtering_value_gradient_benchmark_run_p8d_numeric.py:317` maps
  `_sir_observations()` to `_sir_dataset(81103)`;
- `scripts/filtering_value_gradient_benchmark_generate_p8_datasets.py:214`
  generates that deterministic synthetic dataset.

The older forward artifact proves the existing numerical lane used a
source-shaped fixed-SIR target and the three-coordinate human amendment, but it
does not embed the observation tensor or a canonical observation digest. It
cannot resolve which wording is authoritative.

Recommended repair: retain the current deterministic seed-81103 observation
bytes as the fixed main-row observations, amend Phase 0's prose to bind that
exact identity, and regenerate/review the Phase 0 freeze plus P1-A artifacts.
This preserves the evaluator/data actually used by prior BayesFilter work. It
must be an explicit owner target decision because it changes the reviewed
authority record. The alternative is to reconstruct exact pinned-author
`rng(1)` observations, change the current harness target, and invalidate all
affected prior artifacts.

## Preliminary P1-B Findings

P1-B was not closed because P1-A failed. The primary paper and pinned author
source were nevertheless inspected far enough to identify the next likely
vetoes. These classifications are preliminary availability findings, not final
cell admissions.

| Main row | Paper/source comparison | Preliminary classification |
| --- | --- | --- |
| `benchmark_lgssm_exact_oracle_m3_T50` | Paper Section 6.1 and author `eg1_kalman` use `m=n=3,T=50` but infer only `(a,d)` in a stationary scalar-coefficient model. BayesFilter evaluates five `(phi1,phi2,phi3,q_scale,r_scale)` coordinates with a different observation matrix and target generator. | `extension_or_invention`; not source-faithful for this exact row. |
| `zhao_cui_sv_actual_nongaussian_T1000` | Paper Example 1/Section 6.2 and author `models/sv` support a synthetic `T=1000` raw-return SV model with fixed sigma and two unknowns. The author coordinate maps both physical gamma and beta through `0.1 + 0.8*Phi(z)`; BayesFilter uses `gamma=Phi(z)`, `beta=exp(z)` and an exact log-square transformed target. The physical truth matches, but the exact parameterization/target scalar does not. | Source-anchored model family with `extension_or_invention` exact target/coordinate. |
| `zhao_cui_sv_ksc_gaussian_mixture_surrogate_T1000` | No KSC seven-component target appears in the Zhao-Cui paper or pinned companion code. | `extension_or_invention`. |
| `zhao_cui_spatial_sir_austria_j9_T20` | Paper Section 6.3 and author `eg3_sir` use `d=0` with fixed `(kappa,nu,sigma_obs)=(0.1,18,10)`. The three log-scale inference coordinates are absent from author code. | `extension_or_invention`, already explicitly human-amended on 2026-07-06 without a source-faithful inference-theta claim. Observation identity remains blocked separately. |
| `zhao_cui_predator_prey_T20` | Paper Section 6.4 and author `eg4_predatorprey` match `T=20`, six physical parameters, truth `(0.6,114,25,0.3,0.5,0.5)`, Gaussian covariances `4I`, initial `(50,5)`, RK4 interval `2`, and internal step `0.1`. Author code uses an equivalent bounded Gaussian-CDF coordinate internally; BayesFilter exposes physical coordinates. | Candidate `fixed_hmc_adaptation` only if the coordinate-change derivation and fixed randomness are bound; exact model support is available. |
| `zhao_cui_generalized_sv_synthetic_from_estimated_values` | Paper Section 6.2 and author `svmodels` cover the generalized eight-parameter S&P500 model and allow active parameter subsets. The leaderboard instead uses synthetic `T=1008` prior-center data, fixes five parameters, and labels the third coordinate `mu`; author `ftt2true.m` maps the third active transformed coordinate to physical `mu` by multiplying it by `tau`. Values coincide at zero, but derivatives do not. | Source-anchored family with `extension_or_invention` synthetic target/coordinate unless repaired to the exact author transformed coordinate. |

Primary paper:
`.local_sources/highdim_nonlinear_filtering/zhao_cui_tt_sequential_learning_jmlr_23-0743.pdf`,
SHA-256
`c547b9af2e407c7a0d28bf49ca594fed65d9794d4f37ca605edebd91f9755e35`.

Pinned source provenance:
`third_party/audit/zhao_cui_tensor_ssm_p10/MANIFEST.yml`, upstream commit
`80034dccb99eb1d86284a1839b4a12067d13b9da`.

No network metadata lookup was needed for this exact row-mapping gate. Prior
source-support ledgers report no local quarantine/retraction signal; live
publication-status lookup was not performed and is not being claimed.

## Owner Decisions Needed

To resume the same master program without changing current numerical bytes, the
owner must explicitly authorize both statements:

1. The main SIR row's authoritative fixed observations are the current
   deterministic `_sir_dataset(81103)` bytes. Phase 0 may be amended and
   re-reviewed to say so; exact author MATLAB `rng(1)` reproduction is not the
   target.
2. For the Zhao-Cui algorithm cells, the exact LGSSM, actual-SV, KSC-SV, and
   generalized-SV leaderboard rows may use explicitly labeled
   `extension_or_invention` target adapters grounded in the generic Zhao-Cui
   Algorithm 2 route. These cells must never be called source-faithful, and the
   extension approval does not establish correctness, admission, ranking, HMC
   readiness, or scientific validity. Fixed-SIR retains its already-approved
   log-scale extension classification; predator-prey remains subject to the
   exact coordinate/fixed-route check.

If either statement is not approved, the master program must be revised rather
than executing P1-C or any GPU phase.

## Inference Status

| Evidence class | Status |
| --- | --- |
| Hard veto screen | Failed on SIR target authority; no numerical candidate was evaluated. |
| Statistically supported ranking | None. |
| Descriptive-only differences | None; P1-A was deterministic metadata work. |
| Default-readiness | Not evaluated. |
| Next evidence needed | Owner target/extension decisions, refreshed Phase 0/P1-A, then completed P1-B exact route ledger. |

## Post-Run Red Team

- Strongest alternative explanation: “fixed source observations” was imprecise
  prose intended to refer to the already-fixed seed-81103 bytes. That is
  plausible, but it does not erase the reviewed no-seed statement; the authority
  must be repaired visibly.
- Result that overturns the blocker: a reviewed artifact predating Phase 0 that
  binds the main SIR observation bytes and proves the two descriptions are the
  same identity, or an explicit owner choice followed by a refreshed freeze.
- Weakest evidence: `.mlx` anchors are extracted from their embedded
  `matlab/document.xml`; final P1-B should preserve bounded extracts or line-
  addressable normalized source records before claiming detailed parity.
