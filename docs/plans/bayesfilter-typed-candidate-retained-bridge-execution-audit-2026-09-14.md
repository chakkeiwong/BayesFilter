# Numerical candidate-set and retained replay execution audit

Date: 2026-09-14. Baseline: `4990085e50829d1050b8cf03636f8136ce91350f`.
Plan: `bayesfilter-typed-candidate-retained-bridge-plan-2026-09-14.md`.

## Verdict and scope

The implemented numerical bridge satisfies the engineering contract. The shared
controller can now use BayesFilter's actual TF/TFP evaluator, retain all verified
members, and issue a frozen runner for an explicit member. Geometry, numerical
settings, verification evidence and final state survive export and reload.
Retained continuation uses the preceding archive's final active state. It does
not retune, include tuning draws, or turn R-hat into a tuning requirement.

This closes the typed numerical evaluator/retained bridge gap identified by
MacroFinance. It does not complete every historical P2/P3/P5 migration, remove
the older numerical configuration types, qualify MacroFinance's value/score
target, or establish posterior convergence. The automatic ordinary runtime's
upstream TF/TFP repair remains intact; old NumPy-blocked artifacts stay blocked.

## Audit of the implementation

| Issue | Checked behavior and evidence |
| --- | --- |
| Numerical authority | Callback observations and caller `qualification_status` labels cannot supply the required numerical evidence. The bridge validates repository binding, complete result, member identity, actual traces and recomputed acceptance/health decisions. |
| Acceptance versus convergence | The evaluator uses `evaluate_hmc_acceptance_evidence` with the declared policy. A finite .99 probability fails the default band. High/nonfinite R-hat diagnostics cannot change a passing decision. No stochastic ranking is performed. |
| Repair lifecycle | Inconclusive repaired children remain unverified. Live replay uses durable validation. A real numerical same-L repair child passes fresh verification, bridges and reloads with its own epsilon/L; the failed parent rejects. Pure-controller regressions also accept valid later descendants after earlier failed repairs and reject forged IDs or unfinished children. |
| Frozen kernel | Every run supplies the member's exact epsilon and L to the same independent scalar-chain TFP runner. No alternate retained kernel or adaptation is introduced. The cache fixes L and draw count while epsilon and seed are runtime arguments. |
| Warmup and health | TFP hidden burn-in is zero; all declared warmup is traced before exclusion. Checks cover accepted/proposed states and target values, endpoint score finiteness, momenta, first-transition displacement, Metropolis state consistency and declared accepted/proposed target status. Missing required traces fail closed. |
| Ordinary geometry | A real operational warmup fixture with a nonidentity initial mass validates both affine layers, composed value/score, finite-difference score agreement, raw/active coordinate round trip and durable reconstruction. |
| Frozen transports | Complete affine-diagonal and dense-IAF artifacts pass value/Jacobian/score checks, numerical tuning and durable retained replay. Unsupported artifacts and fallback score authority reject. |
| Drift and corruption | Tests reject changed target values under an unchanged label, candidate settings, source, start bank, dtype/backend/XLA, missing evidence, corrupt tensors, wrong endpoints and changed geometry. Exact geometry state catches even a Gaussian-preserving reflection that start-value probes alone would miss. |
| Durable continuation | Same-process and fresh-interpreter reload tests continue from the predecessor's endpoint. The complete predecessor chain, checksums, member and seed history are checked; seed reuse, broken ancestry and failed-health predecessors reject. |
| Authority roles | The two mechanics conveniences share one implementation. The claim-eligible convenience additionally requires GPU/XLA execution and exact-target capability. Every retained object still denies posterior-convergence authority. |
| Backend/policy | New runtime modules contain no NumPy computation, pfor or alternate HMC algorithm. GPU memory growth and actual device policy are checked. Numerical source dependencies and TF/TFP versions are persisted. |

The API uses ordinary checksums and versioned files for accidental corruption in
a trusted research workspace. It does not authenticate against a malicious
caller who can rewrite Python and every artifact. Target/data/prior completeness
still depends on a correct consumer signature, lineage and source list. Target
probes detect some mutable-target drift; they are not a proof of target math.
Intermediate target regularization that the target never reports is not audited
by endpoint telemetry.

## Numerical evidence and failure interpretation

The final GPU example (`gpu-xla-02`) passed in 65.38 seconds on the idle physical
GPU 1, an RTX 4080 SUPER, with TensorFlow/TFP, float64, memory growth, and XLA.
The log records actual XLA compilation. Two of six declared Gaussian fixture
candidates were freshly verified. The representative used epsilon 1.3 and L=3;
this is a choice for exercising replay, not an estimated optimum.

The same-backend independent runner produced exactly equal retained draws.
Using the captured momentum, the independent CPU Gaussian leapfrog reference
agreed to maximum absolute errors of 4.44e-16 for position, 8.88e-16 for momentum,
and 1.78e-15 for log acceptance. Each of the twelve inspected cached scalar
runner functions traced once. TensorFlow's global repeated-function warning
does not contradict those per-function counters. The recorded allocator peak
was 581120 bytes; this is an allocator observation for the tiny fixture, not a
claim about compiler memory or target-scale capacity.

The first GPU attempt compiled and ran but failed a cross-XLA/non-XLA same-seed
draw comparison. Its 55.33-second failure is preserved in `gpu-xla-01`; it is
not counted as a passing validation. The comparator assumed aligned random
innovations across different execution modes without establishing them. The
repair compares draws within one execution mode and independently checks the
deterministic leapfrog and energy quantities using captured innovations.
This is an engineering comparison repair, not a relaxed numerical tolerance.
GPU time was 120.71 seconds over the two authorized attempts, below the original
ten-minute budget. No MacroFinance run or transport training was launched.

Early CPU fixture failures covered a missing public policy export, invalid
zero repair reservation, an unchecked evidence-inventory hash, incorrect test
module imports, an unsuitable transferred epsilon grid, an incorrect fixture
scale convention, an incomplete dense-IAF fixture payload, and TFP's singleton
momentum-parts shape. They were localized and repaired; the final tests run the
actual numerical evaluator. CPU validation consumed well under the sixty-minute
budget. `cpu-reference-01` is an earlier passing mechanics check;
`cpu-reference-02` preserves the momentum-shape failure;
`cpu-reference-03` is the final 14.93-second CPU example with the deterministic
leapfrog/energy reference. The zero-test `regression.xml` records an erroneous
test filename; `regression-02.xml` is the completed suite.

## Commands and preserved artifacts

Artifacts are under `docs/plans/artifacts/hmc-typed-retained-bridge-2026-09-14/`.
Every example manifest records the source commit, exact script arguments,
Python environment, source hashes, random configuration, elapsed time and
artifact paths. GPU manifests also preserve memory and device policy. Unit-test
seeds and targets are fixed in the cited test sources; CUDA was intentionally
hidden for all CPU tests. The Python executable was
`/home/ubuntu/anaconda3/envs/tfgpu/bin/python`.

The main final regression command was:

```sh
CUDA_VISIBLE_DEVICES=-1 TF_FORCE_GPU_ALLOW_GROWTH=true BAYESFILTER_TEST_DEVICE_SCOPE=cpu /home/ubuntu/anaconda3/envs/tfgpu/bin/python -m pytest -q --disable-warnings tests/test_hmc_candidate_set_execution.py tests/test_hmc_candidate_set_tuning.py tests/test_hmc_candidate_set_artifacts.py tests/test_hmc_candidate_set_adapters.py tests/test_hmc_tuning_dispatch.py tests/test_hmc_tuning_contract.py tests/test_hmc_tuning_documentation_contract.py tests/test_hmc_step_bound_handoff.py tests/test_hmc_kernel_tuning_public_api.py tests/test_hmc_tuning_artifacts.py tests/test_hmc_retained_sample_archive_runner.py --junitxml=docs/plans/artifacts/hmc-typed-retained-bridge-2026-09-14/validation-r1/regression-02.xml
```

It passed 238 tests in 381.78 seconds. With the same CPU environment,
`tests/test_fixed_transport_hmc_tuning.py tests/test_hmc_verification.py` passed
122 tests in 29.32 seconds (`transport-verification.xml`), and
`tests/test_hmc_trace_target_status.py tests/test_nonlinear_ssm_phase4_full_chain_hmc.py`
passed 33 tests in 10.34 seconds (`runner-compatibility.xml`). The final
candidate-execution suite passed 37 tests in 82.90 seconds, including the added
real numerical repair-child regression (`candidate-execution-final.xml`). There
are 394 distinct passing test cases across the final affected suites; repeated
development runs are not added to that count.

The final standalone commands were:

```sh
CUDA_VISIBLE_DEVICES=-1 TF_FORCE_GPU_ALLOW_GROWTH=true /home/ubuntu/anaconda3/envs/tfgpu/bin/python docs/examples/hmc_candidate_set_retained.py --cpu-reference --output-dir docs/plans/artifacts/hmc-typed-retained-bridge-2026-09-14/cpu-reference-03
CUDA_VISIBLE_DEVICES=1 TF_FORCE_GPU_ALLOW_GROWTH=true timeout 300 /home/ubuntu/anaconda3/envs/tfgpu/bin/python docs/examples/hmc_candidate_set_retained.py --output-dir docs/plans/artifacts/hmc-typed-retained-bridge-2026-09-14/gpu-xla-02
```

Both wrote `validation_manifest.json`; shell output is preserved in the
corresponding `.log` files. The GPU command used trusted/escalated execution.
The route inventory check passed with no stale or unclassified routes. The
generated capability tables passed their documentation contract. `git diff
--check` passed. All twelve unrelated tracked dirty files matched their
pre-task checksums and remain outside this task's commits.

The guidebook was built from `docs/` with:

```sh
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir=/tmp/bayesfilter-typed-retained-guide-20260914 main.tex
```

The 557-page PDF built successfully. The changed tuning section was inspected
in rendered form (PDF pages 411–413); identifiers remained readable and the
new procedure distinguished numerical eligibility from posterior admission.
Three unresolved OBC citations elsewhere in the book are pre-existing.

## Decision and remaining evidence

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Deliver the numerical bridge | Actual tune, explicit member, export, reload and predecessor continuation pass | Corruption, drift, inconclusive-child and numerical-health negative controls reject | Consumer target identity and target-specific compilation/math | MacroFinance integrates the new binding and performs its own target checks | MacroFinance posterior validity or convergence |
| Preserve all verified candidates | Controller keeps all verified IDs; no implicit nominee | Failed or incomplete members cannot replay | Acceptance estimates remain finite-sample tuning evidence | Further retained diagnostics under the consumer's existing plan | A best candidate or statistical superiority |

| Inference status | Finding |
| --- | --- |
| Hard veto screen | Exercised by focused negative controls; healthy Gaussian numerical execution passes. |
| Statistically supported ranking | None attempted or established. |
| Descriptive-only differences | Candidate acceptance, count of passing members, runtime and allocator observations. |
| Default readiness | Supported numerical API for the stated contracts; target-specific scientific/default promotion remains separate. |
| Next evidence needed | MacroFinance's exact target/value-score and GPU/XLA checks, unchanged campaign policy, then its sequential retained convergence and scientific checks. |

The strongest alternative explanation for success is that the Gaussian fixtures
are much easier than MacroFinance's target. The nonidentity preparation and
nonlinear transport tests reduce geometry risk but do not remove that
limitation. A target-specific value/score mismatch, compilation failure,
unreported target invalidity or broken lineage would overturn eligibility for
that target. It would trigger localized repair, not retrospective promotion of
old callback-only results.
