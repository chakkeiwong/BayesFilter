# N8 consumer and documentation audit result — 17 October 2026

The bounded engineering slice of the [N8 plan](bayesfilter-modern-hmc-n8-consumer-audit-2026-10-17.md)
passes. The modern methods remain experimental. This audit does not qualify a
posterior, rank samplers, change a default or complete N8 release work. It uses
deliberate CPU diagnostics with GPUs hidden; it is separate from the running
GPU/XLA SNAPER cohort and its immutable source.

## Evidence and execution

Artifact root `A`:
`docs/plans/artifacts/modern-hmc-implementation-2026-10-07/N8/consumer-audit-20261017-01/`.
Checkout: `preserve/shared-main-before-fab-20260926`, HEAD
`139cd11fe7fea37e971f2197a544cb26ccd39c44`, with existing shared-worktree
changes preserved. Source hashes in the manifests identify the tested bytes.

The baseline ran the ten modules in the plan using
`/home/ubuntu/anaconda3/envs/tfgpu/bin/python -m pytest -q` through
`A/run_checks.py`: **171 tests passed**, 12,827 warnings, pytest time 507.64
seconds. Runner-inclusive occupied time was 510.8821383100003 seconds.
`A/attempt-02/manifest.json`, `checks-01.json` and `pytest-01.log` preserve the
exact command, package providers, environment and source hashes. No watched
runtime, test or guide file changed during this baseline.

The tests cover both public tuning entry points, native tiny numerical
pipelines, every-member export/reload/direct-kernel parity, full-model adapter
wiring, interruption/resource recovery, peer preservation after posterior
failure, invalid or degenerate reference uncertainty, and settlement negative
controls. The exclusion test inspects and rejects NUTS execution routes; it
does not run NUTS. Full-model construction/wiring checks are not full-model
statistical validation.

After that baseline, the operational reference and official tuning chapter
were clarified. They now distinguish five core nonlinear targets and twenty
planned method–target cells from completed evidence; describe the generic
accumulation adapter and the four full-data driver adapters; and record the
unfinished growth, volatility and actual MacroFinance/DSGE qualifications.
Finite-filter agreement and single-target/seed feasibility have explicit
limits. Posterior failure leaves tuning membership unchanged.

The revised documents then passed **15 documentation-contract tests** and the
generated route-table check. These repeat a subset of the baseline tests;
there are not 186 distinct tests. Commands and revised source hashes are in
`A/documentation-check-01/manifest.json` and `result.json`, with the exact
runner in `A/run_documentation_checks.py`. No watched file changed during
these checks. The environment deliberately set `CUDA_VISIBLE_DEVICES=-1`,
`BAYESFILTER_TEST_DEVICE_SCOPE=cpu`, `TF_FORCE_GPU_ALLOW_GROWTH=true`, two
TensorFlow/OMP threads, disabled plugin autoload and disabled bytecode writes.
This is an explicit CPU diagnostic exception, not GPU/default qualification.

From `docs/`, `latexmk` built an **82-page tuning/modern-method chapter
excerpt** with the existing preamble and bibliography. The PDF is
[tuning-and-modern-methods.pdf](artifacts/modern-hmc-implementation-2026-10-07/N8/consumer-audit-20261017-01/documentation-check-01/chapter-build/tuning-and-modern-methods.pdf).
Local TeX input hashes are recorded in `result.json`. PDF pages 13–14 were
rendered and visually inspected; the inserted material is legible, including
the wrapped source-route identifier. The approximately 3.64-point overfull
line in the added paragraph remains readable without clipping. It is retained
as a minor layout limitation rather than changing the scientific wording.
The complete excerpt has 17 overfull-box and 11 underfull-page warnings, and
one unresolved reference to the omitted `ch:bf-diagnostics` chapter. It has no
undefined-citation warning. This is not a full-monograph build or a new
mathematical/source-faithfulness audit.

Warnings are not being treated as test failures or ignored as absent. The
baseline warning groups comprise TensorFlow Probability's deprecated
`distutils` use, `gast` AST-constructor deprecations under Python 3.13 and an
existing invalid-escape SyntaxWarning encountered by source-discovery tests.
The later documentation subset emitted two dependency warnings. No environment
or dependency change was made. Full warning logs remain available.

## Cost and live campaign

`A/preparation-failure-01.json` preserves an initial metadata-only failure:
the installed TensorFlow Probability provider is `tfp-nightly`, so assuming
the wheel name `tensorflow-probability` was wrong. The runner now resolves
installed providers with `packages_distributions()`; no package was installed.
The failed check plus metadata diagnostic cost 0.889204445 seconds.

`A/accounting-01.json` settles the measured sequential engineering work once:
522.8989839983948 seconds total, including baseline, revised documentation,
chapter build and page rendering, below the 1800-second cap. The unallocated
campaign pool is now **233891.35098051262 seconds**, before eventual SNAPER
observer active-time charges. The separate **180000-second SNAPER numerical
reservation** remains intact. The already-held 3600-second reserve and
historical 1579.5929542018566-second balance are not added or charged twice.
No SNAPER numerical settlement has been repeated.

At 17 October 02:52 China time, SNAPER tuning was complete: all six roots
qualified, no repair descendants, 480 qualification numerical chunks. The
first posterior member was running and five were pending; no final posterior
delivery is claimed. Numerical, progress and terminal-audit services were
active with zero restarts and no reported error. Binding/source checks and
the recorded GPU memory-growth policy passed. The existing closeout observer
will audit and settle the attempt at terminal exit. This observation is not a
terminal integrity or posterior result.

## Decision and remaining program

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | What is not concluded |
| --- | --- | --- | --- | --- | --- |
| Accept this bounded N8 engineering slice | 171 baseline tests, 15 revised-document checks, generated tables and chapter build pass | No failing check or source drift; layout/reference limitations recorded | CPU checks do not cover every model/backend or statistical failure | Keep experimental scope; preserve receipts | N8 completion, posterior validity or release readiness |
| Continue the existing SNAPER cohort | All six tuning-qualified identities preserved; posterior in progress | No reported service/source/memory-policy error at inspection | Delivery, count limits, reference agreement and terminal audit unfinished | Monitor existing services; inspect terminal closeout once | Completion, ranking or fresh adaptation reliability |
| Keep broader qualification open | Scope inventory now matches current consumers | Missing growth/eligible-SV/external-consumer evidence prevents corresponding qualification | Multi-model/seed reliability and sequential coverage | Prepare priced complete-fit/model evidence after the active cohort; continue independent calibration | Twenty completed cells or default promotion |

| Inference status | Finding |
| --- | --- |
| Hard veto screen | No engineering test failure or watched-source drift. No new terminal sampler assessment. |
| Statistically supported ranking | None. This suite is not a stochastic comparison. |
| Descriptive-only differences | Test/build times and live progress counts are operational observations. |
| Default-readiness | Not established; modern profiles remain experimental. |
| Next evidence needed | Broader method–model/independent-fit validation, actual-stopping calibration, and actual MacroFinance/DSGE consumer qualification. |

The strongest alternative explanation for a passing audit is that tiny
fixtures and adapter wiring omit difficult full-model behavior. Existing
native GPU checks and SGQF cohorts reduce some engineering uncertainty but
do not remove this limitation. A reproducible public-consumer failure or
source/target mismatch would overturn this scoped engineering conclusion.
The weakest evidence is generalization beyond the tested contracts and
fixtures. The scientific direction is not rejected; the next numerical and
consumer phases remain the appropriate discriminating work.
