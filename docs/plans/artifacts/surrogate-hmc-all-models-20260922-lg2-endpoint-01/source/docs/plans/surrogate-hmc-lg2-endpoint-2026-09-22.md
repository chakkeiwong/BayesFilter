# LG2 claim-endpoint and complete inference-cost qualification

The question remains whether full canonical analytical DCTR, used as an
imperfect position-only proposal field, supports endpoint-corrected HMC
parameter inference. This preparation stage checks the original LG2 claim
consumer and measures the complete cost of one prospective B/R test. It does
not tune HMC, retain samples, consume a statistical look, or establish any of
the B/R, C/D, D/R or C/R comparisons.

The [precision audit](surrogate-hmc-inference-precision-result-2026-09-22.md)
shows that the forecast below prices a bounded 2,000-draw pilot, not an
adequately justified equivalence design: even ideal IID median-CDF intervals
have nominal radius .02146 against a .02 margin before reference error.
No inference is authorized by this qualification. A later inference plan must
justify its diagnostic scope and precision, then reserve the complete cost of
that prospectively chosen schedule. Active or closed looks remain fixed.

The master reporting path will preserve this qualification through a checked
LG2 evidence reader. It verifies archived source and input hashes, all required
checks and the complete cost calculation. It keeps cumulative family balance
separate from the one-attempt qualification reservation, and reports all four
posterior comparisons as NOT_RUN. Focused reporting and accounting regressions
use the existing60-second implementation-check allocation; they do not grant
authority for a new inference attempt.

Preserve LG2's two states and observations, T=20, N=K=1024, transition before
observation, parameter-dependent stationary initial law, F=a[[.7,.2],[.1,.6]],
Q=exp(2b)[[1,.3],[.3,.8]], H=I and R=.5I. The priors remain independent
Uniform(.5,1.2) for a and Uniform(-1,1) for b, in bounded-logit coordinates.
Use the original attempt1/replicate0 claim data, rounded once to FP32 and
promoted exactly for the independent FP64 Kalman endpoint. The independently
calibrated protection multiplier remains0.01. No setting is selected using
claim posterior discrepancies.

The comparator is the checked matrix-Kalman implementation and its independently
qualified full-prior parameter grid, preserved under the September20 LG2
reference root. Inspect and verify their data/source hashes before use. Direct
shared-callable parity, bitwise replay, row independence, permutation,
position-only binding, force/endpoint signs and reversible proposals must pass
on the actual claim consumer. Original force statuses and numerical endpoint
validity must remain healthy; an invalid field is not an endpoint support
exclusion. Probe logit positions(0,0),(-2,-2),(-2,2),(2,-2),(2,2),(-4,0),
(4,0),(0,-4),(0,4), padded to fixed four-row batches.

Retained diagnostics must preserve thirteen physical-parameter functionals:
standardized means, centered second moments and cross moment; marginal CDFs
at the reference's10/50/90 percentiles; and the two Kalman conditional
probabilities Pr(y21,j>0|theta,y1:20). Their order, reference allowances and
interpretation follow the existing qualified reference. Check the entire
fixed250x4 draw-chunk evaluator against separate pointwise reference values.
No posterior claim follows from this fixed-position check.

Measure five complete four-chain L1 transitions: one including compilation,
then four steady measurements. Also measure five complete retained-diagnostic
chunks, each containing1000 parameter positions. Use the largest steady CPU
and wall costs independently. The prospective workload is3586 transitions
(32+64 mass windows,128 adaptation,64 nomination verification,48 trajectory
probes,250 discarded warmup,1000 fresh verification,2000 retained draws),
eight cached diagnostic chunks, a compilation allowance of at least240 seconds,
and90 seconds analysis, all multiplied by1.5. For CPU and wall time separately,
increase the compilation allowance if the sum of the positive first-call
excesses over each kernel's largest steady time exceeds240 seconds. Observed
compilation overhead must never disappear behind a fixed allowance.
This supersedes the old optimistic screen, which omitted
nomination and used the fastest measured transition. It remains a cost forecast,
not evidence that L1 will mix or that a later statistical test is affordable.

| Material choice | Reason and status | Failure mode and earliest check |
|---|---|---|
| Existing claim law and selected0.01 controls | Frozen LG2 scope; independently selected proposal hypothesis | Data, chart or control drift; provenance and canonical-callable parity |
| FP64 matrix Kalman B endpoint | Same-law reference previously checked against joint Gaussian calculations | Incorrect covariance lifecycle or prior chart; existing source hashes and claim point checks |
| Fixed250x4 diagnostic chunk | Prospective retained archive layout, preserving chain/time axes | Transposition, wrong physical units or predictive law; direct evaluation parity |
| L1, epsilon.02 and identity mechanics metric | Small reversible map used only for qualification and cost measurement | Misleading cost or healthy tiny steps mistaken for mixing; complete workload accounting and explicit no-inference status |
| Largest steady cost, factor1.5 and explicit overhead | Conservative measured allocation hypothesis | Compilation/long-run variance; later job watchdog and fresh tuning remain mandatory |

The qualification pass criterion is that every declared engineering and
numerical check passes and that complete positive CPU/wall measurements are
preserved. Source/data drift, failed parity/reversal, invalid original status,
nonfinite values, missing measurements or budget exhaustion veto qualification.
Timing and force/reference differences are explanatory only. A failed cost
screen defers inference and does not reject surrogate HMC.

The previously constructed cheap adversaries are exact Kalman inference,
random-walk Metropolis and a Gaussian posterior approximation, evaluated in
central and marginal-tail regions. Their downstream comparison remains NOT_RUN;
efficiency, superiority and default promotion are unavailable.

Skeptical audit before implementation: the existing reference covers this
two-parameter target, and the claim observations and chart must be shared
exactly. Prior consumer and control checks do not substitute for a claim-endpoint
check or retained inference. The old forecast cannot authorize a new run.
This stage therefore measures the complete prospective workload before any
allocation change. No inherited control or proxy diagnostic is promoted.

Provenance audit: the earlier calibration-consumer report predates the
selection issuer's September20 repair from hardcoded LG1 attempt4 to explicit
model/attempt validation. The archived source matches its recorded hash and
its `protection_controls` function is unchanged. The selected LG2 controls
were issued after that repair; their complete sources still match current
files. Retain the archived consumer source as historical evidence and check
the unchanged control function explicitly. The new claim-consumer checks
establish current wiring and mechanics; the older report is not current
admission. All other reference, consumer and control dependencies must match.

Reserve350 CPU-worker and450 GPU-wall seconds for one qualification, plus60
CPU seconds for focused implementation checks and30 for orchestration, within
the existing cumulative LG2 allocation of1800 CPU/7200 GPU seconds. Recompute
remaining family/global spending before launch. Only one serious GPU job runs:
launch after the active SV20 inference exits. Use GPU1
GPU-68251639-fe82-8f81-3ccc-2953c32e805b, the tftwogpu environment, verified memory
growth, TF32/XLA and two CPU threads. CPU checks explicitly hide GPU devices.
No additional inference allocation is authorized by this qualification.

Mechanics use transition seeds[20260922,37100+i] for i=0,...,4. Reversal
tolerance is32 FP32 epsilons times max(1,|position|,|momentum|), carried from
the earlier linear-consumer check to allow rounding in the FP32 proposal
field. Endpoint chart parity uses absolute tolerance1e-10; FP64 predictive
batch parity uses1e-11. These are numerical equivalence vetoes, not accuracy
or posterior promotion margins. Peak allocator usage above4 GiB stops this
small qualification; it does not alter numerical results. The separate
FP32 proposal chart need not equal the FP64 endpoint chart.

Run `scripts/run_surrogate_hmc_program.py --stage P3-LG2-endpoint --scope LG2 --output
docs/plans/artifacts/surrogate-hmc-all-models-20260922-lg2-endpoint-01` with
the environment above; `--dry-run` checks evidence and budgets without GPU
initialization. Focused checks are in `tests/test_surrogate_hmc_linear_endpoint.py`.

Use a fresh `surrogate-hmc-all-models-20260922-lg2-endpoint-01` output root and
preserve exact commands, environment, seeds, input/source hashes and snapshot,
memory policy, complete logs, timings, pass/fail checks and incremental
accounting. Record the decision, remaining budget and next justified action
in a result note. Preparation may nominate a later B/R plan; it cannot complete
the other linear scopes or any finite-DCTR posterior comparison.

Implementation audit before GPU execution: seven focused CPU checks passed,
charging7.585041 CPU seconds, including physical chart/moment/CDF ordering,
time/chain axes, full1000-position Kalman prediction parity, complete-cost
accounting and cumulative family spending. GPU devices were intentionally
hidden. Evidence and successful dry-run are under
`docs/plans/artifacts/surrogate-hmc-all-models-20260922-lg2-metrics-tests-01`.
The dry-run validates12 frozen inputs and retains the full350 CPU/450 wall
qualification allowance. GPU XLA mechanics and measured cost remain untested
until the planned qualification executes after SV20.

The master dispatch now reaches the same bounded qualification directly and
rejects incompatible scope or ignored overrides. The SV20 run has completed;
the LG2 qualification is the next serialized GPU action. The pre-run skeptical
audit preserves the independent endpoint, original claim data, full canonical
proposal field, frozen control choice, complete pilot-cost forecast and strict
separation from inference. No posterior comparison or HMC tuning occurs here.
