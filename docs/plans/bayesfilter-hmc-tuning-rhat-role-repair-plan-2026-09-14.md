# HMC tuning R-hat role repair plan

Date: 2026-09-14  
Status: complete; implementation, focused regressions, and guidebook checked

## Question and intended policy

Does ordinary HMC tuning need rank-normalized split/folded R-hat to be at or
below 1.01 before it may issue a frozen-kernel handoff? No. Ordinary tuning
must retain R-hat values, threshold metadata, finite/nonfinite counts, and cap
status as explanatory diagnostics, but R-hat must not reject a candidate,
create an epsilon repair, force a verification-only retry, or consume the
tuning budget until the threshold is reached. Tuning handoff remains governed
by target and score validity, finite state/runtime/acceptance evidence,
movement and divergence/telemetry checks, minimum declared evidence, and the
candidate-specific epsilon qualification rules.

R-hat, ESS, and MCSE remain available as separately declared posterior-validity
checks after handoff. This repair makes no posterior-convergence or sampler-
superiority claim.

## Baseline inspected

At baseline commit `b603363df95ce5eb18b28202544dafabbc58be56`, ordinary Phase 7 constructed
`SequentialRHatHMCVerificationConfig`, labelled R-hat a tuning handoff gate, and
classified an otherwise healthy candidate with high R-hat as
`verification_rhat_repair_trigger`. `HMCTuneVerifyRepairAttempt` also rejected
a passed attempt unless the R-hat gate is true. The TensorFlow proposal-field
route already records fresh R-hat as outside tuning handoff. The shared
candidate-set controller has typed observations and does not intrinsically
require R-hat.

## Skeptical audit before execution

The plan was checked for wrong baselines, proxy promotion, missing stop rules,
hidden route divergence, stale guidebook claims, and accidental posterior
policy changes. The material flaw was the previous unification plan's use of
R-hat as a promotion/continuation veto. The repair below keeps the verifier's
R-hat calculation and direct posterior-gate default, and changes only the
ordinary tuning role. No long HMC run or GPU campaign is needed; focused unit,
contract, import, and documentation checks answer the implementation question.

## Implementation

1. Add an explicit `rhat_role` to the sequential verifier configuration. Keep
   `posterior_gate` as the default for direct posterior/diagnostic consumers.
   Ordinary Phase 7 passes `tuning_explanatory_only`.
2. In explanatory mode, stop/pass on valid declared tuning evidence and the
   minimum retained count; still compute and serialize R-hat normally.
3. Disable R-hat-only classification and retry/repair branches in ordinary
   Phase 7. Keep non-R-hat health, target, acceptance, divergence, telemetry,
   callback, and candidate-data vetoes unchanged.
4. Require passed ordinary attempts to declare `tuning_explanatory_only` and
   consistent `passed`/cap fields. The direct verifier's `posterior_gate` role
   is separate and cannot stand in for ordinary tuning evidence.
5. Update ordinary route metadata, public summaries, generated route tables,
   tests, and the guidebook/reference text to say that R-hat is explanatory in
   tuning and a posterior gate only where explicitly declared.

## Evidence contract and stop conditions

The primary criterion is that a healthy in-band ordinary candidate with
R-hat above threshold still produces a mechanics handoff, while a finite,
acceptance, target, telemetry, callback, or evidence-validity failure still
blocks handoff. R-hat must remain present in the diagnostics and must not
create a repair trigger. A focused test failure, import/compile failure, or
documentation contradiction stops implementation until repaired. No result
will be interpreted as evidence of posterior convergence or statistical
superiority.

## Validation

Run focused Python compilation and the HMC tuning, candidate-set, dispatch,
and documentation-contract tests, followed by the HMC route inventory and
`git diff --check`. Record the exact commands and outcomes in the execution
note after the code and documentation changes.

## Final source audit and assumptions

The final audit found one remaining inconsistency: the ordinary classifier
required the explanatory role, but the passed-attempt constructor also allowed
`posterior_gate`. The constructor now agrees with the classifier, with
regressions rejecting an incompatible role or inconsistent completion flags. The historical
R-hat-only retry hooks are disabled; their compatibility plumbing remains
dormant rather than being removed in a large outer-loop refactor.

| Choice | Provenance and role | Failure mode and smallest check |
| --- | --- | --- |
| R-hat is explanatory in ordinary tuning | Current owner request; intended tuning policy | A hidden threshold gate delays or rejects an otherwise qualified candidate; separated-chain and undefined-R-hat fixtures must hand off without extra chunks or attempts. |
| Direct verifier retains `posterior_gate` as its default | Inherited verifier behavior; compatibility choice for direct consumers | Tuning repair accidentally weakens a separately declared R-hat check; the same separated-chain fixture must fail that direct verifier at its cap. |
| Diagnostic threshold 1.01 | Inherited exported constant, not newly calibrated or used for tuning decisions | A copied number regains admission authority; check capability text, serialized role, and ordinary classifier. |
| Acceptance evidence, minimum draws, health, and budgets | Existing configured policies; unchanged hypotheses | Removing R-hat accidentally removes movement/target/acceptance checks; retain negative-control regressions and minimum-draw checks. |
| CPU-only small tests | Explicit reference/mechanics exception; `tests/conftest.py` hides GPUs before framework import | Mistaken GPU/XLA or posterior claim; tests establish control flow only, with no campaign or performance comparison. |

The repair is scoped to the public ordinary route, its shared verifier, result
reporting, and current tuning documentation. It does not revise the separate
NeuTra retained-sampling policy or rewrite frozen July LGSSM campaign criteria
in `bayesfilter/testing/deterministic_lgssm_hmc_phase7_tf.py`. Those campaign
checks are not the current public tuning procedure. Existing numerical backend
debt and the unification plan's P2/P3/P5 qualification work remain open; this
repair does not promote the ordinary route to claim-bearing backend status.

## Execution result

The repair is complete in the working tree. Ordinary verification sets
`rhat_role="tuning_explanatory_only"`; high or undefined R-hat remains visible
in the reported maximum, finite/nonfinite counts, threshold result, and role.
It creates no tuning veto, epsilon repair, verification-only retry, or extra
sampling requirement. Acceptance evidence, minimum draws, movement, divergence,
target/score health, callbacks, and runtime validity still control handoff.
Direct verifier consumers retain their separately declared `posterior_gate`.

The ordinary result constructor and classifier now require the same role and
completion semantics. The guide chapter, reference interface, generated route
tables, and candidate-set plan agree on this distinction. Historical tuning
artifacts are not relabelled as passing under the new policy.

Final focused regression: **294 passed**, 1,962 TensorFlow/TFP/gast deprecation
warnings, pytest elapsed 92.56 seconds (process wall time 95.058 seconds).
Environment: `/home/ubuntu/anaconda3/envs/tfgpu/bin/python`, Python 3.13.13,
TensorFlow 2.20.0, tfp-nightly 0.25.0, pytest 9.0.3. GPUs were intentionally
hidden before imports. The exact command, environment, source hashes and log
are preserved in the [validation manifest](artifacts/hmc-tuning-rhat-role-repair-2026-09-14/validation-20260914T082352Z/manifest.json).

```bash
CUDA_VISIBLE_DEVICES=-1 TF_FORCE_GPU_ALLOW_GROWTH=true BAYESFILTER_TEST_DEVICE_SCOPE=cpu \
  /home/ubuntu/anaconda3/envs/tfgpu/bin/python -m pytest -q --disable-warnings \
  tests/test_hmc_fixed_size_chunk_runner.py \
  tests/test_hmc_kernel_tuning_outer_loop.py \
  tests/test_hmc_kernel_tuning_public_api.py \
  tests/test_hmc_tuning_documentation_contract.py \
  tests/test_hmc_tuning_dispatch.py \
  tests/test_hmc_tuning_contract.py \
  tests/test_hmc_candidate_set_tuning.py \
  tests/test_hmc_candidate_set_artifacts.py \
  tests/test_hmc_candidate_set_adapters.py

python -m py_compile bayesfilter/inference/hmc.py \
  bayesfilter/inference/hmc_kernel_tuning.py bayesfilter/inference/tuning_contract.py \
  tests/test_hmc_fixed_size_chunk_runner.py tests/test_hmc_kernel_tuning_outer_loop.py \
  tests/test_hmc_kernel_tuning_public_api.py tests/test_hmc_tuning_documentation_contract.py
python scripts/inventory_hmc_tuning_routes.py --check
git diff --check
```

Compilation and diff checks pass; inventory reports no unclassified or stale
routes. The documentation contract checks generated files against the registry.
The full guide was built from `docs/` with:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error \
  -outdir=/tmp/bayesfilter-hmc-rhat-book main.tex
```

The [556-page PDF](artifacts/hmc-tuning-rhat-role-repair-2026-09-14/validation-20260914T082352Z/book/main.pdf)
builds successfully. The revised tuning paragraph and failure table were
visually inspected on PDF pages 415 and 427 (printed pages 397 and 409); both
render correctly. Three unresolved citations remain in the untouched OBC
chapter (`Afshar2015`, `Gorinova2020`, `Pakman2014`); these do not affect the
changed tuning text. The build also reports layout warnings elsewhere.

The resumed checks first passed 229 and then 260 tests before the final
constructor consistency repair. Setup-only mistakes were corrected: a command
named nonexistent `tests/test_hmc.py`, the first LaTeX invocation used the repo
root instead of `docs/`, and a metadata probe looked for the release TFP
distribution rather than installed `tfp-nightly`. None of those commands
produced validation evidence. The final commands above are the evidence.

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Complete the ordinary R-hat repair | High/undefined R-hat cannot delay an otherwise qualified handoff; no extra attempt or repair slot is consumed | Focused regressions pass, including unchanged health and directional-acceptance negative controls | Target-specific tuning outcomes have not been rerun | Use the repaired ordinary route for fresh tuning under the caller's existing evidence requirements | No posterior convergence or sampler ranking |
| Preserve separate posterior checking | The same separated-chain fixture fails with the direct verifier's `posterior_gate` role | That threshold remains enforced | Posterior validity needs target-specific retained evidence | Apply the declared posterior checks after handoff | Tuning acceptance does not establish posterior validity |
| Close documentation repair | Chapter, reference, registry, generated tables and active plan agree | Documentation checks and book build pass | Broader unification and backend qualification remain open | Continue P2/P3/P5 under their existing plan | This repair does not complete the full numerical candidate-set migration |

The final skeptical review checked the alternative failure mode that R-hat
could still control an outer-loop retry after its threshold test was removed.
The real sequential-verifier fixtures, ordinary classification tests, outer-loop
attempt counts, public artifact test, and disabled retry hooks jointly cover
that path. The weakest remaining evidence is downstream target execution:
MacroFinance P18 was not rerun. No stochastic comparison or statistically
supported ranking was performed.
