# Phase 7 Subplan: KSC-SV Compact Score Port

Date: 2026-07-08

Status: `DRAFT_READY_FOR_REVIEW`

## Phase Objective

Build a compact forward-sensitivity score route for the KSC-SV surrogate row, or write a blocker result if the score cannot be derived without changing the target scalar, target likelihood, or coordinate system.

The target is the admitted finite-`N` LEDH value scalar:

```text
observed_data_log_likelihood_estimator
```

reported as:

```text
log_likelihood
```

for row:

```text
zhao_cui_sv_ksc_gaussian_mixture_surrogate_T1000
```

The target observation policy is:

```text
ksc_log_chi_square_gaussian_mixture_surrogate
```

The active score coordinates are:

```text
gamma_unconstrained, log_beta
```

under:

```text
synthetic_unconstrained
```

## Entry Conditions Inherited From Previous Phase

- Phase 1 shared contract blocks historical `manual_total_vjp*` routes from full score admission.
- Phases 2-6 established compact tiny routes for LGSSM, actual-SV, fixed-SIR, predator-prey, and generalized-SV.
- KSC-SV has an admitted value artifact:
  `docs/plans/ledh-phase7-ksc-sv-forward-scalar-artifact-2026-07-07.json`.
- The KSC-SV value runner lives at:
  `docs/benchmarks/benchmark_ledh_same_target_ksc_sv_value.py`.
- Generalized-SV and actual-SV score evidence is model-specific and cannot admit KSC-SV.

## Required Artifacts

- New or updated KSC-SV score implementation:
  `docs/benchmarks/benchmark_ledh_same_target_ksc_sv_score.py`
- New KSC-SV score tests:
  `tests/highdim/test_ledh_ksc_sv_score_phase7_contract.py`
- Updated score contract if a compact KSC-SV provenance constant is added:
  `bayesfilter/highdim/ledh_score_contract.py`
- Tiny compact score diagnostic artifact:
  `docs/plans/bayesfilter-ledh-compact-score-default-phase7-ksc-sv-tiny-compact-score-2026-07-08.json`
- Phase 7 result or blocker result:
  `docs/plans/bayesfilter-ledh-compact-score-default-phase7-ksc-sv-result-2026-07-08.md`
- Phase 8 integration subplan:
  `docs/plans/bayesfilter-ledh-compact-score-default-phase8-integration-subplan-2026-07-08.md`
- Review bundle:
  `docs/reviews/bayesfilter-ledh-compact-score-default-phase7-ksc-sv-review-bundle-2026-07-08.md`

## Required Checks, Tests, And Reviews

Precheck commands:

```bash
python -m py_compile \
  docs/benchmarks/benchmark_ledh_same_target_ksc_sv_value.py
```

Focused CPU-only checks after implementation:

```bash
CUDA_VISIBLE_DEVICES=-1 MPLCONFIGDIR=/tmp python -m pytest \
  tests/highdim/test_ledh_ksc_sv_score_phase7_contract.py \
  tests/highdim/test_ledh_score_contract_phase1.py -q
```

Tiny diagnostic command must be drafted after the compact KSC-SV score entry point exists. It must use fixed randomness, small `N,T`, both active coordinates, manual streaming finite transport value+JVP, and no autodiff.

Review:

- Claude read-only review of implementation, tests, Phase 7 result, and Phase 8 subplan when available.
- Use the probe ladder on timeout.
- If Claude is unavailable or policy-blocked, write a Codex substitute review.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Can KSC-SV compute the same finite-`N` KSC finite-mixture surrogate LEDH `log_likelihood` score in `synthetic_unconstrained` coordinates using compact forward sensitivity? |
| Baseline/comparator | Admitted KSC-SV value artifact, KSC-SV value runner, previous compact model ports, and tiny same-scalar finite differences. |
| Primary criterion | KSC-SV default score route carries particles/log weights/tangents/log-likelihood tangents forward, emits compact provenance, matches the value scalar at the tiny gate, passes all-coordinate FD, and does not substitute exact actual-SV, generalized-SV, or raw Gaussian target semantics. |
| Veto diagnostics | Wrong target scalar, exact actual-SV target substitution, generalized-SV raw target substitution, raw Gaussian callback promoted as KSC target, KSC mixture used as proposal-only rather than correction target, wrong parameter coordinate/order, stopped partial derivative, tape/autodiff, reverse-record score default, nonfinite score, or tiny FD failure. |
| Explanatory diagnostics | Runtime, memory, per-coordinate FD error, chunk-size invariance, route parity against value-only diagnostics. |
| Not concluded | Full `N=10000,T=1000` KSC-SV score admission, exact native actual-SV likelihood, HMC readiness, posterior correctness, runtime ranking, or scientific superiority. |
| Artifact | Tiny compact score JSON, Phase 7 result or blocker, tests. |

## Required Implementation Steps

1. Inspect the KSC-SV value route and admitted artifact:
   - `docs/benchmarks/benchmark_ledh_same_target_ksc_sv_value.py`
   - `docs/plans/ledh-phase7-ksc-sv-forward-scalar-artifact-2026-07-07.json`
   - `tests/highdim/test_ledh_phase7_ksc_sv_forward_scalar_artifact.py`
2. Freeze the exact target identity:
   - transformed observation `z_t = log(y_t^2 + 1e-8)`;
   - KSC finite-mixture target likelihood;
   - `target_observation_policy = ksc_log_chi_square_gaussian_mixture_surrogate`;
   - parameter order `gamma_unconstrained, log_beta`;
   - theta values `[0.2533471031357997, -0.916290731874155]`.
3. Derive compact forward sensitivities for:
   - `gamma` and `beta` from `StochasticVolatilitySSM.physical_parameters`;
   - stationary variance and initial particles;
   - initial and positive-time proposal variance;
   - transition mean `gamma * x`;
   - LEDH proposal surface `x + 2 log(beta)`;
   - target KSC mixture log density via component posterior responsibilities;
   - LEDH flow, correction weights, normalization, and streaming transport value+JVP.
4. Add compact KSC-SV provenance only after the route is implemented.
5. Add compact artifact normalization with compact provenance and `claims_exact_native_actual_sv_likelihood = false`.
6. Add static tests against compact default symbols for:
   `records.append`, `reversed(records)`, reverse transport pullback, `GradientTape`, `ForwardAccumulator`, and `stop_gradient`.
7. Add tiny same-scalar value match and all-coordinate FD tests.
8. Write Phase 7 result or blocker, draft Phase 8 integration subplan, and review the handoff.

## Forbidden Claims And Actions

- Do not substitute exact actual-SV log-chi-square target evidence for KSC-SV.
- Do not substitute generalized-SV raw-y density for KSC-SV.
- Do not use raw Gaussian SV callback as the KSC target likelihood.
- Do not treat the KSC finite mixture as exact native actual-SV likelihood.
- Do not change row ID, target scalar, target observation policy, theta coordinate system, parameter order, observations, transform offset, or value artifact.
- Do not run full `N=10000,T=1000` before tiny compact correctness passes and a reviewed amendment authorizes the full run.
- Do not use tape/autodiff, stopped partial derivatives, hidden autodiff, or reverse-record score as default.

## Exact Next-Phase Handoff Conditions

Phase 8 integration may start only if:

- KSC-SV default score route is compact or KSC-SV is explicitly blocked with a result artifact;
- tiny same-scalar FD either passes or the blocker result explains the smallest unresolved issue;
- Phase 8 integration subplan exists and is reviewed;
- Codex records review findings and confirms no unresolved boundary issue.

## Stop Conditions

Stop and ask for direction or write a blocker result if:

- KSC-SV compact JVP cannot be derived without changing the KSC finite-mixture target scalar;
- target likelihood ambiguity cannot be resolved from the value artifact and value code;
- tiny same-scalar FD fails and cannot be localized in this phase;
- implementation would require unrelated dirty-worktree changes;
- memory/runtime behavior suggests even tiny compact KSC-SV is not viable;
- Claude and Codex review do not converge after five rounds on the same material blocker.

## Skeptical Audit Before Execution

Risks checked before launch:

- Wrong baseline: the baseline is the admitted KSC-SV LEDH value artifact, not exact actual-SV or generalized-SV.
- Proxy metrics: tiny FD may pass but still cannot admit a full row without full-row memory evidence.
- Hidden assumption: KSC is a surrogate likelihood row; it must not claim exact native actual-SV likelihood.
- Environment mismatch: CPU-only tests must hide GPU before TensorFlow import; later GPU runs require trusted execution.
- Useless artifact risk: any result must state KSC target policy, transform offset, parameter order, and non-claims explicitly.

Audit status: ready for read-only review after Phase 6 result review.
