# q=20 Factor-Route Fresh Tuning and Numerical Admission Result

Date: 2026-09-06  
Plan: `docs/plans/bayesfilter-ssl-lstm-q20-factor-route-fresh-tuning-admission-plan-2026-09-04.md`  
Parent: `docs/plans/bayesfilter-ssl-lstm-q20-tempered-rkl-transport-ensemble-master-program-2026-09-02.md`  
Status: `PROMOTED_Q20_PHASE9_NUMERICAL_BACKEND_PENDING_PHASE9B`  
Decision: admit `tensorflow_eigh_strict_factor_cached` for the q=20 Phase 9B candidate lane; do not change the generic API default or claim posterior/whitening readiness.

## Executive Result

The source-synchronized factor campaign completed all six `(chart, beta)` scope handoffs, all eight declared `(epsilon, L)` pairs in every scope, held-out verification, and the Phase 9A replica-exchange mechanics controller. The independent auditor returned `PASS_FACTOR_SOURCE-SYNC_AUDIT`.

Together with the earlier paired strict/factor value, score, status, spectral, and transition evidence, this satisfies the narrow numerical-backend admission criterion. The factor route is admitted as the q=20 backend for a separately planned Phase 9B candidate run, with strict remaining the explicit comparator and fallback.

This is not promotion of the sampler, NeuTra transport, HMC tuning policy, or repository-wide default. The fresh tuning chains were deliberately short, and their diagnostics expose substantial unresolved tuning and chart-quality risk.

## Attempt Ledger

| Attempt | Artifact | Outcome | Classification |
|---|---|---|---|
| A1 canary | `canary-attempt-20260904T150551Z/` | Eight-pair factor canary passed in `1511.658` s | Numerical-route canary pass under the original source closure |
| A2 first full | `full-attempt-20260904T185852Z/` | Failed at scope 1 after `1617.631` s | Localized proposal/chart failure plus raw NaN/Inf receipt serialization |
| R2 terminal attempt | `r2-attempt-20260905T095452Z/` | All six scopes completed, then final manifest hash rejected raw `NaN` after `3334.376` s | Artifact hash repair required |
| Serialization replay | `r2-attempt-20260905T190500Z/` | Stopped before scope tuning after `941.644` s | Source-closure provenance veto; no bypass |
| Source-synchronized full | `source-sync-20260905T203000Z/` | All six scopes, complete grids, handoffs, held-out checks, and transition passed in `3551.299` s | Terminal admissible numerical-backend result |

No failed attempt was merged with a later attempt, and no old handoff was reused. The measured aggregate ledger remained within the amended campaign budget.

## Authoritative Artifacts

| Item | Value |
|---|---|
| Attempt directory | `docs/plans/artifacts/ssl-lstm-q20-factor-route-fresh-tuning-2026-09-04/source-sync-20260905T203000Z/` |
| Manifest | `run_manifest.json`, `PASS_PHASE9A_SCOPE_PREFLIGHT` |
| Independent audit | `source-sync-audit.json`, `PASS_FACTOR_SOURCE-SYNC_AUDIT` |
| Manifest SHA-256 | `e0225382192ceb9da1e075cb9c7a91ed424e2c5c67adcfec7a0f34c05003a4e1` |
| Audit SHA-256 | `9848779a1fe1611f3b3acfb666bec8b08bef6a42296fe30e84d3762d892c5933` |
| Git commit | `2dae412e450a5b44f46e375b810a7ad81aa78aeb` (dirty state recorded in manifest) |
| Target/backend | `9a86e60081f1b9cd288dbdb1dcbe1e9a5b5e23d9b5ef97afdb72ee95c23d7278` / `tensorflow_eigh_strict_factor_cached` |
| Runtime | GPU0, TensorFlow `2.20.0`, XLA and TF32, memory growth verified |
| Allocator peak | `1,402,668,544` bytes, below 4 GiB |
| Wall time | `3551.299326295033` s, below the 4,000 s source-sync cap |
| Seeds | initialization `980xx`, preflight `981xx`, training `982xx`, tuning `983xx`, transition `98401`, reliability `98501` |

## Scope Evidence

| Scope | Selected `(epsilon,L)` | Selected acceptance | Held-out acceptance | Max selection folded R-hat | Interpretation |
|---:|---|---:|---:|---:|---|
| 0, chart 0, beta 0 | `(0.060,3)` | `0.99995` | `0.99991` | `7.41` | finite/moving handoff; high-acceptance repair trigger |
| 1, chart 0, beta 0.5 | `(0.055,3)` | `0.64486` | `0.69688` | `6.37` | selected and held-out gates passed |
| 2, chart 0, beta 1 | `(0.060,3)` | `0.23315` | `0.11995` | `15.04` | finite handoff; low-acceptance repair trigger |
| 3, chart 1, beta 0 | `(0.070,3)` | `0.99987` | `0.99980` | `7.82` | finite/moving handoff; high-acceptance repair trigger |
| 4, chart 1, beta 0.5 | `(0.075,3)` | `0.27253` | `0.29056` | `8.80` | finite handoff; low-acceptance repair trigger |
| 5, chart 1, beta 1 | `(0.055,3)` | `0.49934` | `0.38859` | `17.54` | finite handoff; low-acceptance repair trigger |

Every scope measured all eight pairs and issued a factor-bound handoff. Rejected pairs with hard vetoes remain in the evidence tables; the admission rule applies hard vetoes to the selected pair, held-out verification, and transition, as predeclared by `measured_joint_grid_v1`.

## Mechanics And Chart Diagnostics

The transition controller passed with four warmup and four retained draws per chain, finite states/statuses, one compiled transition trace, and no hard controller veto. Its maximum folded R-hat was `3.3187` under the deliberately permissive Phase 9A threshold `100`; four draws per chain are not convergence evidence.

The reliability receipt passed round-trip, log-determinant, score-finite, and conditioning checks, but chart-quality diagnostics were large:

| Chart | Centered log-density RMS | Largest pullback-score RMS/coordinate |
|---|---:|---:|
| chart 0 | `535.98` | `1438.64` |
| chart 1 | `661.94` | `2551.14` |

These values block any IID-Gaussian whitening or NeuTra quality claim. They do not invalidate the narrower eigensystem backend admission. TensorFlow also emitted retracing warnings while constructing independent trainer instances. Per-scope reusable HMC and transition trace gates passed, but the run must not be described as globally retracing-free; this is a performance repair trigger.

## Decision And Inference Tables

| Decision | Primary criterion | Veto status | Next action | Not concluded |
|---|---|---|---|---|
| q=20 factor numerical backend | Complete six-scope grids, handoffs, transition, identity, and resource gates | Passed; independent audit clean | Require factor backend in a reviewed Phase 9B candidate plan; retain strict fallback | No generic superiority or scaling |
| Scope tuning quality | Finite selected/held-out calls and movement | Acceptance extremes and short-chain R-hat are repair triggers | Retune with long warmup and independent per-scope controls | No final `(epsilon,L)` claim |
| Transport quality | Reliability mechanics finite and round-trip exact | No hard numerical reliability veto | Repair/train charts before posterior interpretation | No IID Gaussian whitening |
| Posterior/HMC readiness | Not tested here | Blocked by short chains, chart residuals, R-hat/ESS, and missing downstream checks | Write and review Phase 9B sequential validation plan | No convergence, mode discovery, posterior correctness, or HMC superiority |
| Repository default | Not part of this admission | Strict remains generic default/fallback | Keep factor q=20-specific until downstream gates pass | No global default change |

| Evidence class | Status | Interpretation |
|---|---|---|
| Hard veto screen | Passed for selected handoffs, held-out checks, transition, identity, and resources | Narrow numerical backend admission is supported |
| Statistically supported ranking | Not available | Short stochastic runs do not rank candidates or backends |
| Descriptive-only differences | Acceptance, R-hat, ESS/gradient, residuals, runtime | Explain risk; do not promote correctness |
| Default readiness | Not ready | Generic default and posterior admission remain closed |
| Next evidence | Defined | Long sequential warmup, retained cold samples, modern R-hat/ESS/MCSE, chart gates, comparators, and uncertainty analysis |

## Post-Run Red Team

The strongest alternative explanation is that the four-dimensional C5 target and short mechanics schedule do not expose failures that appear in a longer cold chain. The large pullback residuals and extreme acceptance values support that concern. A source-synchronized Phase 9B run with failed convergence, poor cold-posterior agreement, or mode loss would overturn any broader factor claim while leaving this narrow numerical admission intact. The weakest evidence is statistical: two short selection replications, four-draw mechanics streams, and no independent posterior reference.

## Terminal Audit

The plan and result were re-audited after execution for source closure, fresh seeds, complete scope/grid coverage, selected/held-out vetoes, identity binding, target/bridge signatures, GPU0/XLA/memory-growth provenance, allocator peak, manifest self-hash, and claim boundaries. The audit returned `PASS_FACTOR_SOURCE-SYNC_AUDIT`. The preceding source-closure mismatch was preserved as a provenance veto rather than bypassed.

The result is complete for the narrow numerical-backend question. The parent scientific program remains `PHASE9B_BLOCKED_PENDING_REVIEWED_PLAN`.
