# GenUT Three-Model Simple Feasibility Plan

Date: 2026-07-22
Status: `retired_wrong_actual_suite_composition`
Historical execution order: artificial reduced-SIR fixture, generalized SV, KSC Gaussian-mixture SV

Current execution scope: `N=1008`; the earlier `N=96` attempt is mechanics-only and superseded for feasibility interpretation.
Current result: `docs/plans/bayesfilter-genut-three-model-simple-feasibility-n1008-result-2026-07-22.md`
Historical `N=96` result: `docs/plans/bayesfilter-genut-three-model-simple-feasibility-result-2026-07-22.md`

Correction: `docs/plans/bayesfilter-genut-actual-model-suite-correction-2026-07-22.md`.
The reduced-SIR phase is an artificial mechanics fixture and was wrongly given
an actual-model slot. The existing Chapter 18b structural target was omitted.
This plan must not be relaunched or used as an actual-model suite result.

## Research intent ledger

| Field | Declaration |
|---|---|
| Main question | Can the current non-fused positive-OT GenUT candidate emit finite, internally differentiated value and score estimates for three additional nonlinear targets, and how do those estimates compare with existing same-target value/score calculations at a small fixed scope? |
| Candidate | `cubature_genut_nonfused_positive_ot_candidate_v1`, using a model-specific hand-derived recursive tangent of the identical finite value program. |
| Expected failure mode | A model adapter can have the wrong first-observation timing, parameter chart, mixture target, or tangent; alternatively the finite reset may produce a materially different value or high-variance score even when the implementation is internally correct. |
| Promotion criterion | Diagnostic feasibility only: finite GenUT value and score, finite transport diagnostics, recursive score consistent with representative same-scalar finite differences, and at least one finite same-target value/analytical-or-manual-score comparator. |
| Promotion veto | Non-finite GenUT output, recursive-score/same-scalar-FD scaled discrepancy above 5%, transport residual above `1e-4`, target/timing mismatch, use of runtime autodiff/FD score, or failure to execute on GPU with FP32, TF32, XLA, and verified memory growth. |
| Continuation veto | Corrupt observations, unresolved target identity or first-observation timing, an invalid shared GenUT kernel, or a resource failure that prevents safe execution. A large method discrepancy is not a continuation veto; it is evidence to record for the relevant model. |
| Repair trigger | Adapter tangent mismatch, non-finite calculation, comparator target mismatch, or infrastructure failure under the unchanged campaign budget. |
| Explanatory diagnostics | Per-coordinate value/score differences, finite-difference residuals, transport marginal/mean residuals, wall time, and TensorFlow allocator current/peak bytes. |
| Must not be concluded | No method ranking, accuracy certification, leaderboard admission, default promotion, HMC readiness, canonical Austria-SIR result, or broad generalized-SV/KSC validity follows from one seed at `T=10`, `N=96`. |

## Exact targets and comparisons

### Historical Phase 1: artificial reduced preclip SIR mechanics fixture

- Target: `artificial_reduced_preclip_sir_j1_mechanics_fixture_v1`, state `(S,I)`, parameter order `(log_kappa_scale, log_nu_scale, log_obs_noise_scale)`.
- Scope: seed `97001`, `T=10`, `N=1008`, initial-observation-first.
- Primary comparator: the existing dense split-Gauss-Legendre value and manual score (`order=29`, radius `7`). Boundary mass is a veto diagnostic.
- This was a planning error: it is a boundary-stress fixture, not an actual model, and is ineligible for suite evidence. It does not substitute for either Austria SIR or the existing Chapter 18b structural target.

### Phase 2: generalized SV prior-mean source row

- Target: `GeneralizedSVPriorMeanSSM`, state `x`, raw observation `y`, parameter order `(z_gamma, log_tau, mu_over_tau)`, with `gamma=Phi(z_gamma)`, `tau=exp(log_tau)`, and `mu=mu_over_tau*tau`.
- Scope: canonical synthetic seed `81105`, first `T=10` observations, `N=1008`, initial-observation-first.
- Comparator: the existing fixed-branch scalar TT value and analytical complete parameter-score route on the identical prefix. It is a Zhao-Cui diagnostic comparator, not an oracle.

### Phase 3: KSC Gaussian-mixture transformed SV

- Target: `z_t=log(y_t^2+1e-8)` with the pinned seven-component KSC Gaussian mixture, parameter order `(z_gamma, log_beta)`.
- Scope: canonical SV seed `81101`, first `T=10` observations, `N=1008`, initial-observation-first.
- Comparators: fixed SGQF and principal-square-root UKF value and analytical score on the identical raw-observation prefix. Both implement the declared KSC surrogate and neither is an exact native-SV oracle.

## Evidence contract

- Exact baseline/comparator: the phase-specific same-target routes above. Comparators are labeled according to their actual role; Zhao-Cui is never an oracle.
- Primary pass/fail criterion: every phase satisfies its diagnostic feasibility criterion and records all value and score coordinates.
- Veto diagnostics: target/timing checks, finiteness, recursive-score FD check, reset residuals, dense SIR boundary mass, GPU/XLA/device/memory-growth provenance.
- Explanatory only: observed GenUT-minus-comparator differences and runtimes. One seed gives no statistical ranking.
- Preserved evidence: `docs/benchmarks/artifacts/genut_three_model_simple_feasibility_20260722/attempt02_n1008/` with per-phase JSON, aggregate JSON/Markdown, and run manifest. Attempt 01 at `N=96` is retained as historical mechanics evidence only.

## Defaults and assumptions audit

| Choice | Provenance | Justification | Failure mode | Early diagnostic | Status |
|---|---|---|---|---|---|
| `T=10` | User requested a simple pre-leaderboard test; prior GenUT ladders used short prefixes | Long enough to expose recursive accumulation while keeping repair cheap | Prefix behavior may not predict full horizon | Record per-time increments and forbid full-horizon claims | Convenience scope |
| `N=1008` | Owner directive requires `N>1000`; Gaussian GenUT weights require divisibility by 6 | Smallest convenient count above 1000 that exactly represents one- and two-dimensional positive GenUT masses | Quadratic transport can increase runtime or memory | Record allocator peak; stop on resource failure; use sequential models | Required feasibility scope, not a tuned default |
| `epsilon=2`, Sinkhorn steps `8`, ridge `1e-5` | Prior LGSSM/SV/predator-prey warm starts | Known finite starting point for mechanics | Cross-model transfer can be poorly tuned | Record residuals and classify discrepancies as tuning candidates | Warm-start hypothesis only |
| One seed per model | Bounded feasibility request | Answers executable-path question cheaply | Cannot estimate uncertainty or rank methods | Explicit descriptive-only inference status | Convenience scope |
| FP32 + TF32 + XLA | Repository production-target policy and user request history | Tests the intended GPU numerical lane | Comparator FP64 differences can mix numerical and method error | Record dtype per route and do not claim pure algorithmic attribution | Reviewed execution target |
| Initial-observation-first | Dataset/model definitions and existing comparator implementations | Matches `y0` generated from the stationary initial state | A transition-before-`y0` shift invalidates comparison | Explicit `transition_before_first_observation=False` and target manifest | Reviewed target requirement |
| Same-scalar central FD at the test theta | Existing score-audit policy | Audits the recursive tangent without making FD the runtime score | Step-size cancellation or truncation | Model-scaled steps and scaled error report | Diagnostic only |

## Skeptical plan audit

The plan was checked for wrong baselines, proxy promotion, stop-condition mistakes, unfair comparisons, stale context, environment mismatch, and uninformative artifacts.

- Wrong-baseline check: passed after replacing historical retained-grid Zhao-Cui as the SIR accuracy anchor with the dense manual-score reference. Generalized-SV Zhao-Cui remains explicitly diagnostic. KSC SGQF/UKF are same-surrogate comparators, not exact-SV oracles.
- Target check: passed only with explicit initial-observation-first semantics. The reduced SIR target is not substituted for the canonical Austria row.
- Proxy/promotion check: passed. FD, residuals, and one-seed differences are diagnostic; no leaderboard/default decision is permitted.
- Defaults check: passed with inherited controls downgraded to warm-start hypotheses, `N=1008` justified by the owner minimum and exact GenUT replication, and `T=10`/one seed retained as convenience scopes.
- Stop-condition check: passed. A candidate discrepancy rejects only that diagnostic arm; it does not invalidate the harness or stop later phases unless a true continuation veto fires.
- Environment check: passed conditionally on escalated GPU preflight, verified memory growth before device initialization, TF32 enabled, and XLA device evidence in the artifact.
- Artifact sufficiency: passed. Per-phase scalar, score coordinates, comparator differences, FD checks, residuals, timing, memory, target identity, observations hash, command, commit, environment, and nonclaims are required.

## Implementation and review steps

1. Add KSC-mixture and generalized-SV `CandidateModelAdapter` implementations with explicit value and tangent equations; do not use runtime autodiff or finite differences.
2. Add focused CPU-hidden tests for shapes, finiteness, parameter charts, and local value/tangent consistency.
3. Build one sequential runner that writes each phase immediately, then an aggregate result and manifest.
4. Review the runner for target timing, chart order, XLA closure purity, absence of NumPy/runtime autodiff, and artifact completeness.
5. Run focused CPU-hidden tests.
6. Run escalated `nvidia-smi` and an escalated TensorFlow GPU/memory-growth probe.
7. Execute the three phases sequentially with escalated trusted GPU access.
8. Interpret results as diagnostic feasibility only and write a result note with a decision and inference-status table.

## Budget and commands

- Attempt budget: at most three campaign launches under the unchanged target and scope, including localized harness repairs.
- Compute budget: at most 20 GPU-minutes total and at most 15 minutes for any single launch.
- Stop if a continuation veto fires or the budget is exhausted.
- Planned commands:

```bash
CUDA_VISIBLE_DEVICES=-1 python -m pytest -q tests/test_genut_three_model_adapters.py
nvidia-smi
TF_FORCE_GPU_ALLOW_GROWTH=true python docs/benchmarks/run_genut_three_model_simple_feasibility.py
```
