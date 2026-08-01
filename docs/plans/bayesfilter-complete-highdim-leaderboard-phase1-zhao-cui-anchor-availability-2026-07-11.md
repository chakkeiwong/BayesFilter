# Complete High-Dimensional Leaderboard Phase 1 Zhao-Cui Anchor Availability

Date: 2026-07-12

Run ID: `complete-highdim-leaderboard-local-20260712-134906`

Gate: `P1-B Zhao-Cui availability`

Status: `PASS_P1B_APPROVED_EXTENSIONS_WITH_LATER_ROUTE_REPAIRS`

## Decision

All six exact leaderboard rows differ from the checked Zhao-Cui paper and/or
pinned companion code in a way that requires `extension_or_invention`. The
owner explicitly approved that classification for all six rows, including the
predator-prey mismatch, in:

`docs/plans/bayesfilter-complete-highdim-leaderboard-phase0-owner-authority-amendment-2026-07-12.md`.

P1-B therefore passes the early source-availability gate. This pass means only
that the sources, mismatches, approval, and plausible target-specific repair
lanes are identified before implementation. It is not source-faithfulness,
algorithm correctness, row admission, or numerical evidence.

## Source Support Ledger

| Field | Binding |
| --- | --- |
| Seed paper | Yiran Zhao and Tiangang Cui, "Tensor-Train Methods for Sequential State and Parameter Learning in State-Space Models," JMLR 25 (2024), paper 23-0743 |
| Paper classification | `DIRECT_METHOD` |
| Local full text | `.local_sources/highdim_nonlinear_filtering/zhao_cui_tt_sequential_learning_jmlr_23-0743.pdf` |
| Paper SHA-256 | `c547b9af2e407c7a0d28bf49ca594fed65d9794d4f37ca605edebd91f9755e35` |
| Inspected technical text | Example 1; Sections 6.1-6.4; equations (36)-(38); model setup, coordinate, horizon, noise, and data statements |
| Pinned author source | `third_party/audit/zhao_cui_tensor_ssm_p10/source` |
| Pinned upstream commit | `80034dccb99eb1d86284a1839b4a12067d13b9da` |
| Snapshot manifest | `third_party/audit/zhao_cui_tensor_ssm_p10/MANIFEST.yml` |
| Snapshot manifest SHA-256 | `59ccbf9e368292a76ee0a3264ce07a6579a5d4d0ad47a5257a49ae25ac5cafc4` |
| Binary-source audit aid | `docs/plans/artifacts/complete-highdim-leaderboard/phase1-zhao-cui-mlx-normalized-extracts-2026-07-12.md` |
| Audit-aid SHA-256 | `8424876cace3b22b7959247407a4de5955df73e232d5c69f355049b1c806dcd8` |
| Publication/retraction status | Published JMLR full text inspected; no retraction/erratum check was possible through network metadata in this local-only lane |
| Allowed claim | The cited paper/source contains the stated model family or example operation at the anchors below |
| Forbidden claim | Any exact BayesFilter row or adapter is source-faithful, a paper reproduction, or an author-code reproduction |

No network/API lookup was needed to decide exact local source availability and
was not authorized by the local-only runbook. Citation counts, venue metrics,
forward citations, live errata/retraction metadata, and forward snowballing are
recorded as `not available`, never as zero or clean. They are not required for
this narrow implementation-identity gate.

## Exact Row Ledger

### LGSSM `benchmark_lgssm_exact_oracle_m3_T50`

- Paper anchor: Section 6.1, equation (36), published pages 29-30. The paper
  uses `m=n=3`, `T=50`, fixes `mu=0`, constrains `a^2+b^2=1`, and estimates
  only `(a,d)`.
- Author entrypoint:
  `third_party/audit/zhao_cui_tensor_ssm_p10/source/eg1_kalman/main_script.m:14`
  through the `d=2,m=3,n=3,T=50` setup at lines 14-17.
- Author model anchors:
  `third_party/audit/zhao_cui_tensor_ssm_p10/source/models/kalman/setup.m:3`,
  `third_party/audit/zhao_cui_tensor_ssm_p10/source/models/kalman/st_process.m:4`,
  and
  `third_party/audit/zhao_cui_tensor_ssm_p10/source/models/kalman/ob_process.m:4`.
- Source hashes: entrypoint
  `60db748d0b358da44a31fbfbf71b2b4772c323f84dcd18c553e10d5eaa3c9d65`;
  setup `035bbe616782a5e43618832b2573a0a40228d3a68d34f841e2502f19e0973048`;
  transition `f5b1bb0afe83d78f14c3f9cbe3bae0b16f2a7b3c02525753713c124cdcc67630`;
  observation `6cb23b493c28a62c05412f95316b7bf2b03cf650e0ca3f96e99055b6bb53c063`.
- Exact mismatch: BayesFilter evaluates five physical coordinates
  `(phi1,phi2,phi3,q_scale,r_scale)` with a different transition/observation
  model and seed-81100 observations. It is not the paper's `(a,d)` posterior.
- Classification: `extension_or_invention`.
- Candidate route: the existing differentiated Kalman exact-oracle adapter at
  `docs/benchmarks/benchmark_two_lane_highdim_leaderboard.py:1417`, subject to
  fresh canonical-target/dependency and admission checks in Phase 4.
- Availability: `available_existing_target_specific_extension`.

### Actual SV `zhao_cui_sv_actual_nongaussian_T1000`

- Paper anchors: Example 1, published page 2, defines the raw-return SV model;
  Section 6.2, published page 34, fixes `sigma=1`, estimates `(gamma,beta)`,
  uses `T=1000`, and generates at `(0.6,0.4)`.
- Author entrypoint:
  `third_party/audit/zhao_cui_tensor_ssm_p10/source/eg2_sv/mainscript.m:14`
  through lines 14-28.
- Author model anchors:
  `third_party/audit/zhao_cui_tensor_ssm_p10/source/models/sv/setup.m:7`,
  `third_party/audit/zhao_cui_tensor_ssm_p10/source/models/sv/st_process.m:6`,
  and `third_party/audit/zhao_cui_tensor_ssm_p10/source/models/sv/ob_process.m:5`.
- Source hashes: entrypoint
  `cb3b7c7138f0c8f031d95cff414b547b0bbd31ddaba22cf0be4e89de47d9b2a5`;
  setup `e473e0bb894ab681dd0fccbcbdb1e1f671b7419544001c91cce2dcb6d11f1e42`;
  transition `9ea713d0e197a6c329944f408cdb84e00938f37c13fbf1f801cbd993e0a257bf`;
  observation `2c9b45656ec345e90a4c4625512d98bc44f6c7549747349edb4b65d70a890c81`.
- Exact mismatch: author code maps both physical coordinates through
  `0.1+0.8*Phi(z)`. BayesFilter uses `gamma=Phi(z_gamma)`,
  `beta=exp(log_beta)` and evaluates an exact log-square transformed
  observation density. The physical truth/model family matches, but the exact
  coordinates and scalar target do not.
- Classification: `extension_or_invention`.
- Candidate route: target-specific exact-transformed scalar fixed-branch TT
  adapter at `docs/benchmarks/benchmark_two_lane_highdim_leaderboard.py:776`,
  subject to Phase 4 re-admission and no source-faithful label.
- Availability: `available_existing_target_specific_extension`.

### KSC SV `zhao_cui_sv_ksc_gaussian_mixture_surrogate_T1000`

- Paper/source availability: no KSC seven-component log-chi-square mixture
  target was found in the full paper or pinned author snapshot. The full-text
  and source-tree searches covered `KSC`, `Kim`, `Shephard`, `Chib`,
  `seven-component`, `7-component`, and `mixture`; the only Shephard hits in
  the paper concern auxiliary particle filters.
- Source-family anchor: the underlying raw-return SV family is the Actual-SV
  paper/source route above. It does not support the KSC target.
- Exact mismatch: the mixture observation density, transform offset, and
  target scalar are BayesFilter additions.
- Classification: `extension_or_invention`.
- Candidate route: target-specific KSC scalar fixed-branch TT adapter at
  `docs/benchmarks/benchmark_two_lane_highdim_leaderboard.py:814`, with mixture
  definition in `bayesfilter/highdim/sv_mixture_cut4.py`; Phase 4 must bind the
  canonical KSC target and dependency hashes.
- Availability: `available_existing_target_specific_extension_source_absent`.
- Forbidden claim: the KSC row is present in or derived faithfully from the
  Zhao-Cui paper/code.

### Fixed SIR `zhao_cui_spatial_sir_austria_j9_T20`

- Paper anchor: Section 6.3 and equation (37), published pages 38-39, fix
  `kappa_j=0.1`, `nu_j=18`, use `J=9`, `T=20`, the Austrian adjacency map,
  infectious-only Gaussian observations, and state inference with `d=0`.
- Author entrypoint:
  `third_party/audit/zhao_cui_tensor_ssm_p10/source/eg3_sir/mainscript.m:12`
  through lines 12-28 (`d=0,m=18,n=9,T=20`, `rng(1)`). SHA-256
  `3b0460ad170847b70504f03982d2e683fafee7b83ee95fd7de37fdfdd92cd14a`.
- Author `.mlx` anchors: normalized audit aid, Austria SIR Setup lines 3-15,
  ODE lines 4-24, and RK4 Step lines 1-18. Original archive hashes are bound
  in that audit aid.
- Exact mismatch: BayesFilter adds three log-scale coordinates for kappa, nu,
  and observation noise and uses fixed `_sir_dataset(81103)` bytes. Those
  bytes are explicitly not an author `rng(1)` reproduction.
- Classification: `extension_or_invention`.
- Candidate route: adapt the specialized fixed-TTSIRT/source-route machinery
  in `bayesfilter/highdim/source_route.py` to the exact parameterized target.
  The current `d=0` author-SIR route cannot itself admit this row.
- Availability: `repair_required_target_specific_fixed_variant_extension`.
- Later gate: Phase 5 must build and validate this exact target-specific route;
  the parameterized complete-data sidecar and generic retained-grid routes are
  not substitutes.

### Predator-Prey `zhao_cui_predator_prey_T20`

- Paper anchor: Section 6.4 and equation (38), published pages 40-41, define
  parameter order `(r,K,a,s,u,v)`, physical truth
  `(0.6,114,25,0.3,0.5,0.5)`, `Delta t=2`, RK4 step `0.1`, Gaussian process
  and observation covariance `4 I2`, initial mean `(50,5)`, and `T=20`.
- Author entrypoint:
  `third_party/audit/zhao_cui_tensor_ssm_p10/source/eg4_predatorprey/mainscript.m:14`
  through lines 14-34. SHA-256
  `e132d7fc9eec1237dc62f49fbf5c60338102033c1dd9cccd3ae0a0e752808133`.
- Author `.mlx` anchors: normalized audit aid, Predator-Prey Setup lines 3-11,
  ODE lines 2-9, and RK4 Step lines 17-30. Original archive hashes are bound
  in that audit aid.
- Confirmed source contradiction: pinned `setup.mlx` stores
  `[0.6,1.2,0.5,0.3,0.5,0.5]`; pinned `odefun.mlx` interprets the order as
  `(r,s,u,v,K,a)` and uses physical denominators `90+20*K` and `20+10*a`.
  Thus the pinned code does not compute the paper/BayesFilter equation (38)
  target at physical `(K,a)=(114,25)`.
- Classification: `extension_or_invention` under explicit owner approval.
- Current route veto: the historical implementation calls
  `multistate_nonlinear_fixed_design_tt_score_path` at
  `docs/benchmarks/benchmark_two_lane_highdim_leaderboard.py:881`. This is the
  generic retained-grid multistate route and is diagnostic/historical only.
- Candidate route: Phase 6 must build a target-specific fixed-variant route
  using the approved paper/BayesFilter physical target and must bind the
  source discrepancy. It may reuse general source-route primitives but may not
  relabel the pinned code or retained-grid result as faithful/admitted.
- Availability: `repair_required_replace_forbidden_retained_grid_extension`.

### Generalized SV `zhao_cui_generalized_sv_synthetic_from_estimated_values`

- Paper anchor: Section 6.2, published pages 35-36, uses `T=1008` S&P 500
  returns and estimates `(gamma,sigma,beta)` in the Example-1 raw-return model.
- Author entrypoint:
  `third_party/audit/zhao_cui_tensor_ssm_p10/source/eg2_sv/mainscriptSP500.m:19`
  through lines 19-59. SHA-256
  `5fcadb361b14faa21c5123644b95e261263ec5c66130b66169924734d6b2919b`.
- Author generalized-model anchors:
  `third_party/audit/zhao_cui_tensor_ssm_p10/source/models/svmodels/setup.m:6`,
  `third_party/audit/zhao_cui_tensor_ssm_p10/source/models/svmodels/st_process.m:4`,
  `third_party/audit/zhao_cui_tensor_ssm_p10/source/models/svmodels/ob_process.m:8`,
  and
  `third_party/audit/zhao_cui_tensor_ssm_p10/source/models/svmodels/ftt2true.m:6`.
- Source hashes: setup
  `8298dee448ef83d68e195673fcdcc64fcdc97a470dee02b003defed0a8b3ce61`;
  transition `5b21ec0de0a6af8b0e24faa1d4926f8a9595558143cf20d19753723c69948d2c`;
  observation `b59a7549d2cb7ee5acadea04accab90bb343d17866468a942d1f6c8c54175274`;
  coordinate map `3c3b6678503e5f886fc9545472b67f126e2cde6900acf68b75382d31e8869b84`.
- Exact mismatch: BayesFilter evaluates a seed-81105 synthetic prior-mean
  generalized-SV target at three coordinates `(z_gamma,log_tau,mu)` with
  fixed context. It is neither the paper's S&P-return Example-1 target nor an
  exact author `svmodels` run.
- Classification: `extension_or_invention`.
- Current route caution: the historical scalar fixed-design result at
  `docs/benchmarks/benchmark_two_lane_highdim_leaderboard.py:912` is not
  source-faithful and must be freshly checked against the production-route
  boundary.
- Candidate route: Phase 7 target-specific fixed-variant extension using the
  checked `svmodels` operation family and the canonical BayesFilter target.
- Availability: `repair_or_readmission_required_target_specific_extension`.

## Claim Support Ledger

| Claim | Support class | Result |
| --- | --- | --- |
| Paper contains the four example families | `PRIMARY_TECHNICAL_SUPPORT` | Sections 6.1-6.4 and equations (36)-(38) checked |
| Pinned source contains matching entrypoint families | `IMPLEMENTATION_EVIDENCE` | Entry points and model operations checked at hashes above |
| Exact BayesFilter rows equal paper/source targets | `SOURCE_GAP_BLOCKER` | False for all six exact rows; extensions are required |
| Predator paper and pinned code agree | `IMPLEMENTATION_EVIDENCE` | False; exact parameter/order/scaling mismatch checked |
| KSC target exists in paper/source | `SOURCE_GAP_BLOCKER` | Not found after full local search |
| Exact extensions may proceed | owner authority | Approved by the Phase 0 amendment; correctness still unproved |

## Snowball And Omission Ledgers

- Backward snowballing: not needed to decide whether these exact six local
  adapters occur in the seed paper and pinned code. References relevant to SV
  and filtering context were not promoted into implementation support.
- Forward snowballing: not run; local-only lane forbids network/API calls.
- Citation/venue metadata: not collected; unavailable metadata is not evidence.
- Omission risk: a later scientific literature claim about KSC, generalized
  SV, or algorithm superiority would require a separate full audit. P1-B makes
  no such claim.
- Retraction/errata risk: live status not checked. This does not prevent the
  local byte-comparison gate, but it forbids a claim of complete scholarly
  coverage.

## Route Feasibility And Vetoes

| Row | Early feasibility | Required later action |
| --- | --- | --- |
| LGSSM | existing target-specific extension | Phase 4 fresh re-admission |
| Actual SV | existing target-specific extension | Phase 4 fresh re-admission |
| KSC SV | existing target-specific extension; no paper/source KSC anchor | Phase 4 fresh re-admission with extension label |
| Fixed SIR | specialized source-route primitives exist, exact parameterized route absent | Phase 5 build/validate exact extension |
| Predator-prey | current generic retained-grid route forbidden | Phase 6 replace with target-specific fixed-variant extension |
| Generalized SV | model-family and scalar route pieces exist | Phase 7 repair/re-admit exact extension |

P1-B does not require these later implementations to exist already; it
requires that missing/contradictory anchors and unapproved inventions stop
before implementation. The anchors are present or explicitly absent, every
invention is now approved and labeled, and the missing routes are assigned to
the phases designed to repair them. No continuation veto remains at P1-B.

## Decision Table

| Field | Result |
| --- | --- |
| Decision | Pass P1-B as an approved-extension availability screen |
| Primary criterion | Six exact row mappings, checked paper/source anchors, classifications, gaps, and candidate routes recorded |
| Hard veto status | No unapproved extension remains; source-faithful claims remain vetoed |
| Repair triggers | Fixed SIR, predator-prey, and generalized SV require later target-specific route work; all historical cells require fresh admission |
| Main uncertainty | Numerical feasibility and correctness of later repaired routes are not evaluated in P1-B |
| Next justified action | Issue hash-bound P1-B receipt; continue to P1-C only after superseding P1-A also passes |
| Not concluded | Source faithfulness, evaluator correctness, cell admission, ranking, HMC/posterior correctness, or scientific validity |

## Hostile Review

- Strongest alternative explanation: owner approval could be mistaken for
  evidence that the extensions implement the paper. Control: every row is
  explicitly labeled `extension_or_invention` and the exact mismatches are
  retained.
- Result that overturns this gate: a missing bound source, hidden target
  substitution, source-faithful label, or later discovery that no
  target-specific non-retained-grid route can be constructed.
- Weakest evidence: absence of KSC is supported by exhaustive local text/code
  search, not a theorem; it is recorded as `not found`, not universal absence.

## Nonclaims

- no row or cell is admitted;
- no historical numerical value or score is promoted;
- no generic retained-grid route is production-admissible;
- no adapter is source-faithful or a reproduction;
- no GPU/XLA, HMC, posterior, ranking, superiority, or scientific-validity
  evidence is produced.

