# Zhao-Cui Audit Stage 03, 2026-09-11

Status: bounded derivation and call-chain audit complete, verdict REVISE.
The implementation remains ineligible for claim-bearing C2/GPU evidence.
Focused repair diagnostics may continue. Runtime source was not changed.

## Saved-session diagnosis

The inspected rollout is
/home/chakwong/.codex/sessions/2026/09/10/rollout-2026-09-10T23-22-12-01a08be9-85ba-7f21-8e53-a2dbf566ab7d.jsonl.
The structured analysis is in session-context-summary.json beside this file.
It contains 219 records, 28 tool calls, and 28 matching outputs; there are no
orphan calls or outputs and no successful compaction record. The saved tool
outputs total 657,799 characters, seven are explicitly truncated, and the
largest four outputs are 48,116, 45,010, 44,515, and 42,229 characters.
Recovery contributes 331,385 characters and the later algorithm audit 326,414.
These are decoded saved-output sizes, not tokenizer counts. The four largest
outputs came from recovery calls mentioning saved sessions, using three to six
commands per batch. One batch requested 31,600 output tokens across six
commands. Parallelism itself is not the fault: printing broad results from all
calls adds their output to the retained conversation.

The recovery phase ended at 142,331 context tokens; the later audit began at
148,657. The runtime counter rose by 101,231 before compaction was triggered.
That increase includes more than tool output, so the character counts cannot be
used to assign an exact share of tokens to tools.

The runtime counter reached 249,888 tokens against a compaction trigger of
244,800, while the full context limit was not reached. Earlier ordinary model
responses had already failed below that trigger. The observed failure is in the
configured gateway/upstream response stream; local evidence does not identify
whether request size, gateway capacity, timeout, or an upstream defect caused
it. The large retained history and broad multi-command reads explain why
compaction was required, but do not prove they caused the gateway failure.
Practical controls are a fresh thread from the compact checkpoint, exact file
ranges, per-command limits plus an outer exec limit, selected JSON fields, and
saved full logs with concise returned summaries. The existing launcher uses
an earlier compaction threshold of 120,000 and a 2,000-token output limit;
neither value is a measured gateway capacity. No provider, credential, or
global Codex configuration was changed. Official documentation lookup again
returned HTTP 503, so this diagnosis rests on local rollout and runtime logs.

This continuation initially repeated the same excessive-output pattern:
duplicate full diagnostic reads, broad repository searches, truncated combined
results, and reads from the wrong checkout. Those were avoidable tool-use
errors. Subsequent inspection used exact paths and bounded summaries.

## Source and derivation audit

The local Zhao-Cui paper was checked at the Algorithm 2 and Proposition 2
material around lines 590-730 and Algorithm 3/equations (20)-(23) around
lines 730-924. The author implementation was checked at
@TTSIRT/marginalise.m:25-85, @TTSIRT/eval_cirt_reference.m:43-153, and
AbstractIRT.m:217-270. These sources support full adjacent-TT mass
contraction, upper conditional inversion, identity particle continuation, the
f*g/qhat correction, and the physical Jacobian contribution. They do not
support the local sigma-point guide as source-faithful; that guide remains an
extension.

The active manuscript's frozen finite-score recurrence is algebraically
consistent with its declared fixed coefficients, charts, states, and proposal
densities. Its chart-closure proof gives a valid sufficient condition and a
counterexample to direct contraction with an eliminated-state-dependent map;
it is not a theorem ruling out every possible special-case reprojection.
The positive sigma-point covariance proof is valid under positive weights and
affine spanning. The zero population cross covariance for signed C2
observations follows from E[Y|X]=0 and finite required moments.
The active Lean file passed a fresh check
with Lean 4.30.0-rc2, exit code 0, in 30.59 seconds. The proof scope is finite
algebra only; it does not certify TT fitting, numerical CDF inversion,
TensorFlow execution, or Monte Carlo accuracy.

## Executable checks

The independent CPU result is
[derivation-code-audit-rerun.json](../zhao-cui-audit-20260911-02/derivation-code-audit-rerun.json).
CUDA was hidden, JIT was disabled, and TensorFlow 2.19.1 was used. All six
runtime hashes in that artifact matched at closeout. Execution took 2.056
seconds inside the script, excluding framework startup.

| Check | Result | Interpretation |
| --- | --- | --- |
| Coupled two-dimensional upper conditional | max error 6.7e-16 | Proposal CDF, density, marginal, and affine Jacobian agree on the fixture. |
| Numerical CDF versus smooth TT density | cell-slope error 2.5e-11; smooth-density gap 3.68e-3 | The reported density matches the interpolated CDF's derivative on tested cell interiors; it differs from the smooth TT density. |
| Real C2 model and generic compiler | value error 0; score error 2.0e-10 | Frozen score parity passes on this mechanics fixture; no fit-quality or campaign claim follows. |
| Consumer inventory | no non-test consumer for the guide or compile_algorithm3 | The claim-bearing end-to-end call chain is absent. Tests alone do not close it. |

## Blocking findings

1. Rank activation remains defective. The initializer at
   squared_tt_engine_v0_tf.py:136-148 writes the identity only into basis
   index zero. The Algorithm 2 preparation calls it at
   zhao_cui_algorithm2_preparation_tf.py:122-123. For h(x,y)=1+0.25xy, the
   inherited start stays effective rank one with RMS 0.234375; an explicitly
   active rank-two start reaches RMS 1.03e-10. This is an
   initialization/optimization defect. It blocks any rank-two preparation
   claim until a scoped repair and regression are added.

2. Nonfinite aggregate output is accepted. The Algorithm 3 evaluator at
   zhao_cui_algorithm3_tf.py:171-200 checks per-step factors and normalized
   logs but does not validate accumulated increments, returned weight sums, or
   ESS before returning. The recorded extreme finite-input fixture returns
   valid=true, log_likelihood=-inf, and weight sums 2.0. This blocks value,
   score, and downstream experiment claims until a fail-closed aggregate guard
   has a healthy-case no-fire regression.

3. The active manuscript overstates numerical-CDF equivalence. At
   attempt05_n4_failure_analysis.tex:2788-2790, the finite-grid CDF and
   bisection are described as a local approximation “not a new probability
   law.” The measured 3.68e-3 gap shows that statement is wrong relative to
   the smooth TT density. The manuscript must define the piecewise-linear CDF
   and its cell-slope density as the numerical proposal, and label the smooth
   TT density as nominal/reference where both are reported. Finite bisection
   adds inversion error; an exact-inverse theorem does not literally prove the
   law of floating-point samples. The fixture checks CDF residuals and density
   consistency under that numerical approximation, not exact sampling.

4. The claim-bearing endpoint is not wired. The named-consumer inventory finds
   only tests for compile_algorithm3 and prepare_algorithm2_proposals, and no
   runtime consumer for likelihood_weighted_sigma_point_chart. The
   implementation claim is therefore unsupported at the end-to-end call-chain
   level even though the isolated mechanics pass.

Additional scope findings: master section 5 freezes schedules without
explicitly listing coefficients and sampled states, although the active code
and LaTeX freeze both. The master starts from joint parameter/state particles;
the implemented evaluator varies one common parameter on precompiled paths.
It must be described as that fixed-parameter adaptation. Master section 7
allows an ESS veto while its bounded-entry ledger calls ESS explanatory; the
active bounded-entry role is explanatory. Positive Gaussian defense supplies
support on R^d, but neither a bounded target/proposal density ratio nor the
paper's tau-versus-L2-error hypothesis has been established for these fits.
The source paper's guarantees therefore cannot be transferred wholesale.

The named-consumer inventory is an AST check of Python call sites under
bayesfilter, tests, and docs/benchmarks, excluding artifact diagnostics. It is
not proof against arbitrary dynamic dispatch. The exact-name search also found
no production consumer. No executed author-MATLAB parity, filtering-quality,
GPU/XLA, or HMC result was produced. The bounded-grid preparation is host/eager;
the boolean jit_compile metadata alone cannot establish whole-route XLA.

## Decision table

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Do not launch claim-bearing C2/GPU runs from current checkout | identity, finite-output, and consumer gates must pass | blocked by findings 1-4 | scoped repair design and endpoint ownership | repair initialization and validity checks; reconcile manuscript; wire one generic runner | no rejection of Zhao-Cui or UKF-guided research direction |
| Preserve frozen-score route as diagnostic | same-scalar finite-difference parity | passes tested fixtures | broader model/path coverage | retain as reference after repairs | not an adaptive-total or exact-model score |
| Treat guide as hypothesis | observation response and downstream reference agreement | no promotion evidence | no runtime consumer | wire guide through generic endpoint after baseline repair | not source-faithful, production-ready, or HMC-ready |

There is no stochastic comparison in this stage. Hard veto evidence is the
accepted nonfinite result, rank-initialization failure, and missing consumer
call chain. The numerical errors and score differences are descriptive
mechanics evidence, not a ranking. No default, filtering accuracy, posterior
correctness, HMC readiness, or scientific superiority is established.

| Inference status | Result |
| --- | --- |
| Hard veto screen | Current initializer and evaluator fail their stated mechanics; implementation cannot be accepted. |
| Statistically supported ranking | None; no stochastic method comparison was run. |
| Descriptive-only differences | Recorded numerical errors describe exact-fixture agreement; exploratory initialization errors do not select a default. |
| Default-readiness | Not established. |
| Next evidence needed | Focused repair regressions and consumer wiring, followed by a separately specified downstream comparison. |

Post-run red team: passing low-degree interior fixtures could conceal
higher-rank instability, tail-CDF cancellation, support loss, or poor fitted
proposals. The initializer counterexample distinguishes representational
capacity from optimization failure, but does not choose a general initializer.
The extreme nonfinite counterexample is a validity failure even if ordinary C2
data do not reach it. More source-faithful numerical parity and downstream
evidence could expand the permitted claims after the concrete defects are
repaired. Publisher corrections and forward citation coverage remain unchecked
because metadata access failed; this is not a completed literature survey.

## Commands and limits

The corrected diagnostic command, run from /home/chakwong/BayesFilter, was:

```sh
MPLCONFIGDIR=/tmp/zhao-cui-audit-mpl-20260911 CUDA_VISIBLE_DEVICES=-1 BAYESFILTER_TEST_DEVICE_SCOPE=cpu BAYESFILTER_PRELOAD_CUSTOM_OP=0 PYTHONDONTWRITEBYTECODE=1 TF_CPP_MIN_LOG_LEVEL=3 timeout 180 /home/chakwong/anaconda3/envs/tf-gpu/bin/python /home/chakwong/BayesFilter/docs/plans/artifacts/zhao-cui-audit-20260911-02/derivation_code_audit.py --output /home/chakwong/BayesFilter/docs/plans/artifacts/zhao-cui-audit-20260911-02/derivation-code-audit-rerun.json
```

Two commands failed before Python execution because of the wrong script path
and wrong Python path. The first actual diagnostic completed calculations but
failed JSON serialization on a NumPy boolean, leaving a zero-byte
derivation-code-audit.json. A type-inspection rerun followed, then an explicit
Python bool conversion fixed serialization and the command above succeeded.
The empty failed artifact remains preserved and is not result evidence. A later
serializer-order fix encodes before opening the output, avoiding another empty
artifact. The successful run's exact script is preserved separately as
derivation_code_audit.executed.py; its hash must match the result.

Two additional exploratory CPU commands tested constant perturbations of the
initializer on two- and three-dimensional positive polynomial fixtures. They
showed rank activation was possible, but were not saved as standalone manifests
and do not establish a general repair or a selected perturbation scale. Those
two commands exceeded the written three-invocation continuation allowance;
this is an execution-accounting error, not new campaign authority. No further
numerical checks were launched. Numerical subprocess wall times for failed and
exploratory calls were not preserved; they must not be invented.

The Lean command, environment paths, source hash, exit status, and wall time
are in lean-check.json. It used the installed binary directly and performed no
toolchain download. The session measurement command was:

```sh
PYTHONDONTWRITEBYTECODE=1 python docs/plans/artifacts/zhao-cui-audit-20260911-03/session_context_audit.py --output docs/plans/artifacts/zhao-cui-audit-20260911-03/session-context-summary.json
```

No GPU, HMC, training/tuning, or C2 campaign was launched. The research checkout
remains at 47176bce7cdaa91ddd4466c39f90a11ad005c800 with preserved uncommitted
implementation. Synthetic fixtures are defined in the saved scripts; no external
data version applies. Seeds/configuration for the earlier diagnostic are in
its result JSON; the new coupled-law and C2 fixtures are deterministic.
