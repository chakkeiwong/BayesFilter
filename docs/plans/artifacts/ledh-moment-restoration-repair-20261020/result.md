# SIR affine-restoration failure: diagnosis, repair and limits

20 October 2026. Branch `ledh-refactor-with-policy-fix`, base commit
`4d6915cc66e701c4c7b657e094306d298fdd3e76`, with unrelated dirty work preserved.
Plan: [reviewed numerical repair](../../ledh-moment-restoration-repair-20261020.md).
This is numerical debugging, not a guide-efficacy or admission result.

## Root cause

The exact failed unguided endpoint uses SIR d18, T5/N64, dataset 262023017,
particle seed 274000034, FP64 GPU/XLA with TF32 off. Observations, latent
labels and explanatory score labels match the saved campaign inputs exactly.
OT remains epsilon1/five Sinkhorn steps, balance32 and K=N.

At time4 the largest weight rounds to one, although all 64 weights are
positive. The weighted source covariance trace is 4.771005654186e-29 and
the largest mean coordinate is about 433.68. The capped, untranslated
offsets retain covariance trace 2.349868257435e-29. Adding the mean rounds
all positions to the same FP64 vector. Affine restoration recomputes zero
current covariance and its relative ridge is also zero. Current-covariance
Cholesky is the first nonfinite operation, contaminating both particles
and analytical tangents. This is a representability defect, not an OT
iteration failure or a dispute about a 1e-13 acceptance tolerance.

Evidence: `diagnostic-01/` and `diagnostic-02/`, especially the saved
`restore-input-4.json`, `restore-intermediates-4.json` and `trace.json`.
The pre-cap RMS near 9.57e39 is a separate shape-conditioning warning;
it is not the first nonfinite operation. Why this importance distribution
concentrates so severely is not established by this repair.

## Implemented repair and mathematical review

`higher_moment_contract_e.py::higher_moment_shape_jvp` retains post-cap
offsets and their analytical tangents and passes them to the shared
`affine_restore_cloud_jvp`. For P_i=mu+O_i, real arithmetic gives
P_i-mean(P)=O_i-mean(O), including total directional derivatives. The
current covariance, its existing trace-scaled ridge and ridge derivative,
target factor, triangular solves and final target mean are unchanged.
Optional offsets require their own tangent. No new ridge, fallback,
clipping, invented spread or autodiff score path was introduced.

This is an equivalent real-arithmetic evaluation, not bitwise equality
of floating-point programs. Local non-harm and downstream derivative checks
were required before acceptance. Proposition44 in the updated manuscript
supplies the derivation, including positive-definite factor conditions.
The existing ridge remains a numerics-altering protection, not a pure
fail-closed guard.

Checked consumer chain: Campaign.evaluate -> make_moment_endpoint ->
build_backward_guided_ledh_kernel -> _value_and_analytical_score_impl ->
batched_higher_moment_shape_jvp -> higher_moment_shape_jvp ->
affine_restore_cloud_jvp. The batch wrapper calls the shared single-cloud
routine. Executable batch/single parity tests and actual GPU endpoint replay
verify this chain for the exercised scope; other consumers are not newly
certified. Covariance diagnostics and shape availability reach the endpoint.

## Numerical results

Final evidence: `validation-02/result.json`, `local.json`, `endpoints.json`,
per-endpoint traces and `manifest.json`. Validation-01 is preserved separately.

| Check | Final result | Interpretation |
|---|---|---|
| Focused CPU regression | 100 passed, 2 dependency warnings, 95.89s | Engineering checks; GPU intentionally hidden |
| Saved pre-failure clouds, times1--3 | Maximum scaled particle/tangent change 9.81818e-11 | Pass declared 1e-10 local non-harm tolerance; close to limit |
| Failing local restoration | Previously invalid; repaired states/tangents finite | Identified factorization defect fixed |
| Failed and healthy full GPU endpoints | All three analytical score directions valid | Finite numerical endpoint in this scope |
| Central differences: two endpoints, three coordinates, two steps | 12/12 pass; max normalized error 7.53520e-5 | Below predeclared 1e-3 limit; checks the finite calculation, not the true score |
| Entire trace finite | FAIL at failed endpoint time4 | Four standardized shape residual arrays remain undefined |
| Explicit unavailable-shape flag | PASS; false at failed time4 only | Observability, not a waived moment criterion |

Healthy comparator: dataset262023001/particle274000002. Its loss changes
from 1.3049529827777666 to 1.3049529827768303. This is a same-input
numerical comparison, not a stochastic quality ranking.

## Remaining precision limit

Final addition of the target mean still collapses the exceptional output
cloud. Its materialized covariance trace is zero despite target trace
4.771005654186e-29. The absolute maximum covariance residual is only
1.322810303908e-29, but variance loss is 100 percent. The existing normalized
residual uses a unit lower scale bound and must not be interpreted as
relative moment fidelity in this regime.

Skewness, kurtosis and pairwise standardized shape residuals are undefined.
They remain nonfinite with `materialized_shape_available=false`, not replaced
with zero. Numerical endpoint `valid` is narrower than scientific moment
validity. Future comparisons must inspect shape availability and covariance
fidelity, not admit these results using `valid` alone.

The full moment-restoration criterion does NOT pass. Persistent centered
state representation or a separately derived resolution policy changes the
numerical scope and needs early invariance/downstream checks before a fresh
comparison. No larger ridge or synthetic spread is justified here.

## Decision and inference status

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Retain narrow numerical repair | Finite endpoint, local non-harm and derivative checks pass | No numerical-parity veto in tested scope | Third-step change near tolerance; only two endpoint cases | Keep regression fixtures and offsets representation | Universal numerical robustness |
| Do not resume or promote efficacy comparison | Complete moment fidelity fails | Zero materialized covariance and undefined shape diagnostics remain a promotion veto | Behavior outside the failing cloud | Review representation-level repair under refreshed scope, then fresh holdout data | Improved moments, score accuracy, guide quality, default readiness or HMC validity |

| Inference item | Status |
|---|---|
| Hard veto screen | NaN state/score failure removed; scientific shape/covariance veto remains |
| Statistically supported ranking | None; no new stochastic efficacy comparison |
| Descriptive-only differences | Healthy loss/score changes and two-endpoint numerical checks |
| Default-readiness | Not established; no numerical controls promoted |
| Next evidence needed | Representation fidelity, fresh scope-specific tuning and independent paired efficacy evaluation |

## Execution, budget and terminal review

Four GPU launches consumed 251.675993930 seconds: diagnosis01 45.423792154,
diagnosis02 43.902874564, validation01 80.472817006, validation02 81.876510206.
Stage remaining: 6948.324006070 seconds of7200; four of eight launches used.
Inherited campaign allowance remaining: 27010.281125770 seconds.
RTX4080SUPER, growth verified before device initialization; TensorFlow
2.20.0-dev0+selfbuilt; final allocator peak44,549,632 bytes. Per-run manifests
preserve commands, controls, seeds, environment and selected source hashes.
Those hashes are targeted provenance, not a complete dependency closure.

CPU tests: CUDA_VISIBLE_DEVICES=-1 TF_CPP_MIN_LOG_LEVEL=2, tftwogpu Python,
`-m pytest -q` with test_affine_restoration_offsets, test_ledh_unified_correction,
test_higher_moment_contract_e, test_moment_safety_tf, test_moment_training_tf,
test_moment_training_analysis_diagnostic, test_supervised_guidance_tf and
test_ledh_backward_guide_tf under tests/highdim. Full output: `tests-final.log`.
Earlier regression: 53 passed (`tests-01.log`).

Exact final commands, from the execution worktree (logs redirected to this
artifact directory):

```sh
CUDA_VISIBLE_DEVICES=-1 TF_CPP_MIN_LOG_LEVEL=2 /home/chakwong/anaconda3/envs/tftwogpu/bin/python -m pytest -q tests/highdim/test_affine_restoration_offsets.py tests/highdim/test_ledh_unified_correction.py tests/highdim/test_higher_moment_contract_e.py tests/highdim/test_moment_safety_tf.py tests/highdim/test_moment_training_tf.py tests/highdim/test_moment_training_analysis_diagnostic.py tests/highdim/test_supervised_guidance_tf.py tests/highdim/test_ledh_backward_guide_tf.py
CUDA_VISIBLE_DEVICES=0 TF_FORCE_GPU_ALLOW_GROWTH=true TF_CPP_MIN_LOG_LEVEL=2 /home/chakwong/anaconda3/envs/tftwogpu/bin/python -m bayesfilter.score_study.moment_restoration_validation_diagnostic --diagnosis docs/plans/artifacts/ledh-moment-restoration-repair-20261020/diagnostic-02 --source docs/plans/artifacts/ledh-moment-training-20261020/campaign-01 --output docs/plans/artifacts/ledh-moment-restoration-repair-20261020/validation-02
```

These record the commands actually run, not instructions to overwrite the
existing output directory. Any rerun requires a fresh versioned directory.

Terminal skeptical self-review: PASS for the identified numerical repair;
FAIL for complete moment restoration. No external reviewer was launched.
The strongest alternative explanation is that the repair merely hides the
collapse: explicit zero materialized covariance and unavailable diagnostics
show why that concern still vetoes scientific promotion. Finite scores do
not override it. New derivative failures, healthy changes above tolerance
or bypass of the shared routine would overturn narrow acceptance. Weakest
evidence: two-endpoint coverage and near-limit third-step tangent comparison.

The result does not invalidate the simulated-label risk identity, labels,
saved data or optimizer implementation. It identifies a numerical failure
and a remaining representation limitation, not rejection of guide families.
No remaining phase of this bounded plan repairs whole-filter representation.
The original193/768-row comparison remains preserved and incomplete. No
survivor-only ranking, training rerun, commit or push occurred.

Manuscript: earlier table, equations and claims preserved; the unresolved
factorization paragraph replaced by diagnosis, proof and limited result.
Baseline source: `manuscript-before/`. Two pdflatex passes produced92 pages
with a clean warning scan. Pages85--86 rendered and visually checked.
latexmk was unavailable; installed pdflatex used without package changes.
Human readability review remains pending.
