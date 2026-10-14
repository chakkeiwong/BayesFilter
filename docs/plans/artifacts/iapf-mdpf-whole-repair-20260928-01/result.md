# iAPF / MDPF repair execution result — 28 September 2026

The bounded repair tranche is complete. Both scientific replications remain
**INCOMPLETE**. The strongest new positive result is that the paper-target
iAPF value recursion matches the independent R reconstruction on all ten
paired cases in CPU, GPU graph and GPU/XLA execution. This covers frozen
twists; it does not cover fitting, adaptation, parameter scores or the paper's
complete experiment.

## What was executed

The governing plan is [the whole repair plan](../../iapf-mdpf-whole-repair-plan-20260928.md).
The executable authority is [run_replication_master.py](../../../../scripts/run_replication_master.py).
Sixteen versioned launch attempts were recorded. Five failed and were
diagnosed before retry. Ten implemented phase identifiers are now complete;
the original fixture phase was superseded by a corrected v2 fixture.
Process completion and scientific conclusions are separate fields.

The master, source specification, phase contracts and source snapshots preserve
the tested question. Every attempt directory contains its exact command,
environment, timestamps, resource charge, source hashes, log and decision
where produced. The checkout was surrogate-hmc at base commit
89065bc6354801cb368d5e163ba49fd9c4372d10, with relevant dirty sources snapshotted.
Unrelated working-tree changes were preserved.

## iAPF: what is and is not established

The target is Guarniero, Johansen and Lee's iAPF, including the twisted
initial distribution, Gaussian-plus-floor proposals, ESS-based resampling,
retained weights between resampling times, backward finite fitting, adaptive
particle/iteration controller and a fresh final likelihood run.
The local [publisher PDF](../../../../.localresources/papers/guarniero-johansen-lee-2017-iapf-publisher-version.pdf)
is JASA 112(520), 1636–1647, DOI 10.1080/01621459.2016.1222291.

The source reconciliation establishes a reconstruction family, not the
original executable recipe. The final article describes a regularized version
of Eq. (15); the thesis describes finite local BFGS with no explicit penalty
for Part III and discusses Tikhonov regularization separately for Part IV.
The original initializer, coordinates, optimizer controls, positive-floor
formula, final regularizer and seeds remain unrecovered. The explicit
A_raw and B_scaled recipes in [execution-spec.md](execution-spec.md) are
reconstruction hypotheses.

Twenty controls passed in [attempt01](attempt01_iapf_controls/controls.csv).
They cover the joint Eq. (15) loss gradient, covariance validity,
proposal/mixture identities, the floor-probability bound, exact-twist Kalman
agreement at three ESS thresholds, constant-twist BPF equivalence and the
adaptive controller's fresh final call. These are implementation controls,
not twenty stochastic replications.

[Attempt02](attempt02_iapf_pilot/runs.csv) ran two fresh d=5, T=100 datasets,
four repetitions and two finite-fit recipes. A_raw completed 8/8; B_scaled
completed 5/8 and rejected three nonpositive-amplitude endpoints. All eight
runs for each BPF, fully adapted APF and observation-twist comparator completed.
The pilot has no predeclared uncertainty analysis supporting a ranking.

| Recipe | Completed / planned | Main fitting evidence | Interpretation |
| --- | ---: | --- | --- |
| A_raw | 8/8 | 2,416 of 4,800 recorded fits reached the 100-iteration limit; minimum log-density-norm change about -24.09 | Finite filtering is feasible for these cases; an optimizer limit is not an automatic likelihood-validity veto |
| B_scaled | 5/8 | Three nonpositive-amplitude endpoints; minimum recorded log-density-norm change about -35.45 | This reconstruction has an observed endpoint failure; completed runs do not erase those failures |

The vanishing-density/amplitude escape remains visible. Removing the former
universal stationarity veto allowed the intended finite-optimizer question to
be tested; it did not cure that objective geometry. Stationarity, shape loss,
ESS and optimizer codes are explanatory here. Proposition 1's likelihood
identity for eligible positive twists does not require a stationary
least-squares solution. None of this establishes that the paper's authors used
either local recipe, or that the paper's identity is wrong.

The failed B_scaled endpoints are reproducible from saved seeds and sources,
but their final failing coefficient vectors were not saved. Their complete
optimization trajectories have therefore not been root-caused. Retaining
those vectors is required before the next fitting diagnostic.

## The paper-target TensorFlow value port

The new shared [value kernel](../../../../bayesfilter/score_study/iapf_paper_tf.py)
uses explicit model matrices, the paper's initial-state convention,
Gaussian-plus-floor normalization and ESS-dependent retained weights.
Numerical time recursion uses TensorFlow control flow; particles use batched
tensor operations. The kernel has stable input signatures, no NumPy
computation, Python callbacks or pfor. All tested signatures traced once.

The paired fixtures contain nine d=2, N=32, T=6 cases (constant, observation
plus floor, and exact twists at kappa=0, 0.5, 1) and one frozen pilot
d=5, N=2000, T=100 case. The latter replays already fitted twists from
A_raw; it does not run a TensorFlow fitter.

All seven outputs were checked: terminal likelihood, clouds, weights,
ancestors, ESS, resampling decisions and prefix likelihoods. Floating-point
tolerance was 1e-8 times (1 + absolute reference); integer/Boolean outputs
had to match exactly. Exact-twist cases also match the Kalman likelihood.

| Lane | Cases passed | Maximum absolute discrepancy across floating outputs | Frozen pilot first / repeated call (s) | Peak host RSS / GPU allocator (bytes) |
| --- | ---: | ---: | ---: | ---: |
| [CPU FP64](attempt11_iapf_value_cpu/comparisons.json) | 10/10 | 6.37e-12 | 0.531 / 0.106 | 763174912 / N/A |
| [GPU graph FP64](attempt12_iapf_value_gpu_graph/comparisons.json) | 10/10 | 6.82e-12 | 1.894 / 1.407 | 1320710144 / 78120960 |
| [GPU XLA FP64](attempt13_iapf_value_gpu_xla/comparisons.json) | 10/10 | 6.37e-12 | 1.695 / 0.608 | 1317961728 / 77176064 |

These are single recorded timing pairs, not a benchmark ranking. GPU/XLA
was slower than CPU in this small FP64 case. First-call time includes tracing
and compilation where applicable. Both GPU runs used GPU:0 (RTX 4080 SUPER),
TF32 disabled and verified memory growth before initialization on both visible
devices, including the RTX 5080. TensorFlow was 2.20.0-dev0+selfbuilt.
This is not FP32/TF32 parity, a complete fitter/controller port, downstream
score validation or evidence of a GPU speed advantage.

An endpoint repair was required in the R reference: Cauchy–Schwarz gives
ESS=(sum w)^2/sum(w^2) <= N, so kappa=1 means resampling at every later
time. Floating-point ESS slightly above N had suppressed three of five
expected resampling events in an exact-twist fixture. R and TensorFlow now
handle the kappa=1 endpoint explicitly. The kappa=0.5 adaptive pilot is
unaffected. The original fixtures remain archived; attempt09 is authoritative.

## Why the old consumer was misleading

The [actual consumer audit](attempt11_iapf_value_cpu/legacy-consumer-audit.json)
traced iapf_adapter.execute_iapf to make_fitted_twist_kernel and evaluated
that same callable body on deterministic inputs. Its model has six score-study
parameters, an extra initial transition and compulsory resampling. The
checked initial-law and resampling discrepancies against the paper-target
recursion were approximately 1.028 and 0.378, respectively.

That consumer is **wrong relative to a claim that it replicates this paper
experiment**. Its eager body matches its own legacy equations exactly on the
fixture. This does not establish that those equations are mathematically
invalid for their separately defined model. Earlier component parity could
not close this call-chain and target mismatch.

Separately, the tiny legacy graph returned three empty tensors, while the
same body in an explicitly diagnostic eager call returned the expected scalar,
six-vector and particle clouds. [Attempt14](attempt14_legacy_graph_localization/decision.csv)
ruled out the simple unwritten-TensorArray explanation: writing every optional
array did not fix the graph, and small written/unwritten-array controls passed.

[Attempt16](attempt16_legacy_optimizer_isolated/cases.json) used nine fresh
CPU processes with the same deterministic d=1, N=4, T=2 input and one changed
optimizer option per process:

| Isolated graph setting | Legacy output valid and matches its eager body |
| --- | --- |
| Default | No: three empty tensors |
| Meta optimizer disabled | Yes |
| Arithmetic optimization disabled | Yes |
| Dependency optimization disabled | Yes |
| Constant folding disabled | Yes |
| Loop, function or shape optimization disabled; remapping disabled (separate processes) | No |

This localizes an interaction with TensorFlow graph optimization in the
tested self-built version. It does not identify a unique faulty transformation
or establish a legacy GPU/XLA failure. Attempt15's later single-option
comparisons were confounded because TensorFlow updates an existing optimizer
option dictionary rather than replacing it; attempt16 resolves that ambiguity
with isolated processes. No global optimizer default, framework installation
or legacy production path was changed. The new paper-target kernel passes
under default graph optimizations.

## MDPF / KDM-IWSG

The target is Younis and Sudderth's MDPF: kernel-density mixture filtering and
the importance-weighted sample gradient (IWSG), with the paper's learning and
tracking experiments. It is distinct from iAPF.
The official source checkout is
[local author code](../../../../.localresources/code/younis-mdpf-neurips-2023)
at b0e2fd54db7b6c36d70e8e701ddc6a3f3d5dee18.

One previously completed author-source NLL training trial and its untouched
evaluation are reused as evidence; the lost supervisor exit is not invented
or repaired by retraining. The current analysis checks the saved checkpoint
on 5,000 trajectories of length 150, with 25 particles.

[Attempt05](attempt05_mdpf_saved_screen/result.json) reproduced the author's
RMSE aggregation to maximum absolute discrepancy 1.182e-5 against a 1e-4
tolerance. Source RMSE was 3.175109826 and NLL 5.925723451. The formula is
mean over trajectories of sqrt(mean over time of sum of squared state errors),
not a mean or root of the serialized tensor under an assumed interpretation.

Three cheap point predictors were constructed using training-only information:
initial-cloud persistence, constant-heading propagation at training median
displacement, and the training marginal center with circular heading.
The learned checkpoint had no observed conditional mean point-error loss to
these predictors across 12 prespecified strata: four time windows crossed
with all/ordinary/unusual bearing residuals. Thresholds and heuristic
coefficients were frozen from training data. See
[conditional.csv](attempt05_mdpf_saved_screen/conditional.csv).

This passes a cheap point-error falsification screen. It does not pass the
full normalized-density baseline gate, compare independent training seeds,
establish statistically supported superiority, reproduce Appendix B learning,
complete MSE refinement or validate a TensorFlow training consumer.
The source metric audit and heuristic aggregation are explicit independent
CPU diagnostics in the author environment.

The source toy_problem_dataset is a different nonlinear model and cannot be
substituted for Appendix B's controlled linear-Gaussian dynamics and bimodal
observation experiment. The remaining Appendix B parameter/initialization
specification must be resolved or explicitly labeled as reconstruction before
that learning experiment runs.

## Failed attempts and evidence corrections

| Attempt | Finding | Repair / disposition |
| --- | --- | --- |
| 03 | Syntax error in new saved-output analysis | Corrected literal; dispatcher now checks Python syntax before launching |
| 04 | Wrong interpretation of serialized source error tensor | Read source loss/evaluation code and reproduced the actual aggregation |
| 06 | R fixture sampler tried assigning into a locked base binding | Explicit global sampler shadow; prior output preserved |
| 08 | kappa=1 rounding changed exact-twist resampling branches | Mathematical endpoint handling in both reference and candidate; regenerated v2 fixtures |
| 10 | All ten new value cases passed, but legacy graph returned empty outputs | Record the legacy veto separately and investigate; no production fallback |
| 15 | Later optimizer-option comparisons could inherit earlier settings | Fresh-process isolation in attempt16; no causal attribution from confounded rows |

[Metadata errata](metadata-errata.json) correct the interpretation of the
seed field in attempts14–16: those experiments use fixed deterministic
tensors and no random draws. The historical manifests are preserved. The
dispatcher now emits the correct seed classification for future such runs.

## Decisions and inference status

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Accept frozen-twist value parity within tested scope | PASS: all outputs, three lanes, ten cases | No checked numerical/branch/trace veto | Untested shapes, fitting, scores and downstream consumers | Port the frozen finite fitter/controller and test the actual endpoint | Full iAPF replication or default readiness |
| Retain A_raw as a viable reconstruction arm | 8/8 pilot runs complete | No invalid endpoint in these eight; density escape remains | Few datasets/runs; original recipe unknown | Preserve recipe; retain full fit failure diagnostics in the port | Best recipe or recovered author implementation |
| B_scaled has a recorded fitting failure | 5/8 complete | Three nonpositive-amplitude endpoints | Exact failing trajectories not saved | Capture failed coefficients/steps before a focused repair | Rejection of the iAPF theory or all scaled objectives |
| Preserve MDPF source trial and point screen | Source metric parity and 12 conditional screens pass | Full density gate and independent-run evidence absent | One learned checkpoint | Appendix B specification plus matched UKF/BPF density baselines | MDPF replication or ranking |
| Quarantine legacy path from paper-target claims | Checked call-chain and model/time/resampling differences | Paper-target mismatch; graph empty-output failure | Unique optimizer transformation not isolated | Use shared paper-target implementation; keep legacy reproducer diagnostic | A global TensorFlow/XLA defect |
| Whole replication remains open | Full experiment protocols absent | Required comparisons and inference missing | Cost, source ambiguity, repeat variability | Implement the next bounded phase under the existing plan | Completion merely from successful commands |

| Inference category | Status |
| --- | --- |
| Hard veto screen | Three B_scaled endpoints fail their declared validity check; legacy consumer fails paper-target identity and graph output validity; five launch failures preserved |
| Statistically supported ranking | None |
| Descriptive-only differences | Pilot likelihood error/runtime, conditional one-checkpoint heuristic means, all recorded timing pairs and fit-geometry summaries |
| Default-readiness | Not established for either replicated method; new kernel is a value candidate |
| Next evidence needed | Actual fitter/controller and training consumer checks, normalized-density/oracle comparisons, untouched scoped experiments and independent repetitions with predeclared uncertainty |

## Review and remaining resources

Twelve focused dispatcher regressions passed (2.799 seconds), including actual
child failures, timeouts, inherited locking after supervisor loss, syntax
preflight, budget/dependency checks, correct nonzero CLI status and a test that
value parity cannot mark scientific replication complete. Review is
self-review; no independent agent or external reviewer was invoked.

The strongest alternative explanation for the favorable value result is its
restricted frozen-twist, linear-Gaussian, FP64 scope. A fitting/controller or
gradient mismatch would overturn a broader implementation claim without
invalidating these value results. R and TensorFlow share the mathematical
specification, so exact Kalman controls and source inspection remain necessary
independent checks. For MDPF, easy point predictors and one saved seed are
weak adversaries; a matched density comparator or independent run may reverse
the favorable descriptive impression. Missing original iAPF settings still
prevent an exact-author-executable claim.

The final ledger charges 851.648256517 CPU seconds of 1800 and
42077.277523910 GPU seconds of 173700, including inherited spending and
failed attempts. Remaining: **948.351743483 CPU seconds and
36.561867355 GPU hours**. GPU wall-time charging follows the inherited
campaign definition; it is not a new allocation.

No campaign job is running. This tranche's sixteen launch slots are accounted
for. The next required work is implementation/specification, not another blind
retry: finite BFGS plus adaptive controller and actual endpoint parity for
iAPF; Appendix B plus fair normalized-density baselines for MDPF. Subsequent
bounded phases can be specified under the existing authorization and remaining
compute. Full paper-scale repetition cannot be promised inside the current
balance: ten additional runs at the observed 8.589-hour MDPF training cost
would require about 85.89 hours for that arm alone.
