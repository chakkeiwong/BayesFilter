# Phase 15 Subplan: Manual-Score LGSSM XLA Compile Gate

Date: 2026-07-08

## Phase Objective

Test whether the current no-`GradientTape` LGSSM affine NeuTra objective can be
compiled and executed with `jit_compile=True` on trusted GPU, and record compile
time plus compilation-size proxies.

This phase replaces the earlier Phase 15 GPU rerun plan. No `jit_compile=false`
run is allowed in this phase.

## Entry Conditions Inherited From Previous Phase

- Phase 14A passed the no-tape LGSSM target and fixed affine transport repair.
- Current target signature:
  `275bdd37a82d8c09d2c1b1935b6481f18224644ac659830a921c7958b6ed9038`.
- Current adapter signature:
  `d89bdedcf759566f490ce5222ef753cc8c0c8ea39805d68c064c12d2bec07900`.
- Old Phase 10/11 artifacts with target signature
  `290a91d2a8f90d5b29243965b258b1ec6fd965aa46ffca69dcb78f7fa1ecabcb`
  are historical diagnostics only.
- All runtime checks in this phase must use `jit_compile=True`.
- Trusted GPU execution is required for runtime evidence.
- No optimizer update, NeuTra training loop, HMC sampling/tuning, or external
  sample generation is allowed.

## Required Artifacts

- A compile-only diagnostic helper under `bayesfilter/testing/` using the
  current manual-score LGSSM target.
- A trusted GPU/XLA diagnostic JSON under `docs/plans/` or
  `docs/plans/artifacts/`, below repository size limits.
- The diagnostic JSON must record:
  - target signature;
  - adapter signature;
  - `jit_compile=true`;
  - trusted GPU manifest;
  - first-call wall time;
  - second-call wall time;
  - compile-time proxy: first call minus second call when both calls pass;
  - graph concrete-function serialized byte size when available;
  - compiler IR/HLO text byte size when available;
  - finite value/gradient checks;
  - no training, no optimizer update, no HMC, and no sample generation.
- Phase 15 result or blocker:
  `docs/plans/bayesfilter-lgssm-first-neutra-hmc-phase15-manual-score-xla-compile-gate-result-2026-07-08.md`.
- Next subplan only after the XLA compile gate passes or writes a blocker.

## Required Checks/Tests/Reviews

- Trusted `nvidia-smi` before GPU/XLA execution.
- GPU TensorFlow/XLA command must run with escalated/trusted permissions.
- CPU-hidden focused tests may validate config/source guards only; they are not
  runtime evidence.
- `python -m py_compile` for changed helpers/tests.
- Focused pytest for config/source guard behavior.
- `python -m json.tool` on the diagnostic JSON if produced.
- Source scan preserving no `GradientTape`, `batch_jacobian`, or `tape.` in
  admitted runtime paths.
- `git diff --check`.
- Bounded read-only review of Phase 15 result and next subplan before any
  training, HMC sampling/tuning, or readiness claim.

## Evidence Contract

| Field | Contract |
| --- | --- |
| Question | Can the no-tape LGSSM affine NeuTra objective compile and execute under trusted GPU XLA with `jit_compile=True`, and what are the compile-time/size diagnostics? |
| Baseline/comparator | Phase 14A manual-score target signatures and previous Phase 13 taped-XLA blocker as stale failure context only. |
| Primary criterion | Trusted GPU diagnostic compiles with `jit_compile=True`, executes two calls with finite value/gradient diagnostics, and records compile-time and size proxies; or writes a parseable blocker with exact error/provenance. |
| Veto diagnostics | Any `jit_compile=false` runtime run, CPU runtime evidence, hidden optimizer update/training, hidden HMC sampling/tuning, hidden sample generation, stale signature reuse, nonfinite diagnostics, unparseable blocker, unsupported readiness/scientific/product claims. |
| Explanatory diagnostics | First/second call timing, compile-time proxy, concrete graph byte size, compiler IR/HLO byte size, device manifest, target/adapter signatures. |
| Not concluded | HMC convergence, posterior correctness, sampler quality, transport superiority, production readiness, default readiness, broad XLA readiness beyond this exact compile gate, or scientific validity. |
| Artifact | Phase 15 diagnostic JSON, result/blocker, helper/tests, and next subplan. |

## Forbidden Claims/Actions

- Do not run `jit_compile=false` runtime diagnostics.
- Do not run NeuTra training or optimizer updates.
- Do not run HMC sampling or tuning.
- Do not generate external samples.
- Do not use old taped-signature artifacts for promotion.
- Do not use DSGE/c603.
- Do not change default policy.
- Do not claim broad XLA, HMC, production, or scientific readiness.

## Exact Next-Phase Handoff Conditions

The next phase may begin only if:

- Phase 15 records pass/blocker with a result artifact;
- every runtime command used for evidence records `jit_compile=true`;
- any pass records trusted GPU execution, current target/adapter signatures,
  finite diagnostics, first/second call timing, and size proxies;
- any blocker records exact error/provenance and no fallback run;
- no optimizer update, training loop, HMC sampling/tuning, or external sample
  generation occurred;
- old artifact staleness remains recorded;
- next subplan states whether it proceeds to GPU training with
  `jit_compile=true` only, another XLA repair, or a bounded no-runtime design
  phase.

## Stop Conditions

Stop if:

- trusted GPU access is unavailable;
- any runtime evidence would require `jit_compile=false`;
- target or adapter signatures change unexpectedly;
- the diagnostic would need optimizer updates, training, HMC, or sample
  generation;
- diagnostics are nonfinite;
- artifact JSON is malformed or too large;
- review does not converge after five rounds.

## Skeptical Plan Audit

| Risk | Control |
| --- | --- |
| Wrong baseline | Current manual-score signatures are the baseline; old artifacts are stale. |
| Proxy promotion | Compile success and finite diagnostics do not imply HMC/posterior/production readiness. |
| Missing stop conditions | `jit_compile=false`, CPU runtime evidence, training, HMC, and sample generation are vetoes. |
| Environment mismatch | Runtime evidence requires trusted GPU execution. |
| Artifact mismatch | The diagnostic must record timing and size proxies, not just pass/fail. |

Audit status: ready for implementation and trusted GPU execution.
