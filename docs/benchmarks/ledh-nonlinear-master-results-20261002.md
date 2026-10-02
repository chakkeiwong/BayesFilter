# Nonlinear master tester: implementation and bounded checks

The master program now evaluates predator–prey and SIR d=18 through the shared
canonical LEDH value and recursive analytical total-score executor. Its main
output is actual likelihood and score coordinates, with optional same-target
reference errors and conditional comparisons. No nonlinear accuracy campaign,
scope-specific tuning or exact nonlinear oracle was established in this task.

Three integration defects were corrected: SIR has 9 observations for 18 states;
the predator–prey adapter needed dtype plumbing; and SIR's fixed topology and
initial-mean metadata needed eager validation outside traced evaluation. Scalar
log constants now use Python math rather than NumPy in the touched runtime.
The filtering, correction and analytical score algorithms remain shared.

## Executable evidence

49 focused CPU reference regressions passed, covering model-source fidelity,
existing SQMC mechanics and the new driver. After the final reporting changes,
all 11 driver tests passed, including one additional conditional-comparison test
(50 distinct checks across those runs). Transition tangents were checked against
finite differences. Wiring tests prove both model endpoints call the shared
executor with pairwise correction and the guard enabled. The controller test
checks timeout cleanup, evidence preservation, nonzero exit and overwrite refusal.
CPU runs intentionally hid GPU devices. Full logs are under `checks/` below.

The GPU checks used TensorFlow 2.20.0-dev0+selfbuilt, XLA, verified memory growth,
and an RTX 4080 SUPER. FP32 runs enabled TF32. FP64 is an explicit reference arm.
The four attempts consumed 442.01 seconds, including worker compilation, within
the 1,800-second implementation budget. The separate device probe used an RTX
5080; actual numerical artifacts identify the RTX 4080 SUPER correctly.

| Attempt | Scope | Outcome |
|---|---|---|
| gpu-smoke-01 | T1/N36, reduced controls, guarded FP32, both models | Predator–prey finite; SIR invalid likelihood |
| gpu-reference-02 | SIR T1/N72, reduced controls, guarded FP64 | Finite, trace/value/score parity |
| gpu-smoke-03 | T2/N72, reduced controls, original and guarded FP32 | Both predator–prey arms finite; both SIR arms invalid at step 2 |
| gpu-smoke-04 | SIR T2/N1008, declared controls, guarded FP32 | Finite, trace/value/score parity |

Reduced controls are exactly recorded in `checks/smoke-controls.json`; they use
one flow, marginal and pairwise step, four Sinkhorn and two balancing steps.
The final SIR check uses the driver's declared warm-start controls, not those
reduced controls. Since count, precision and controls change between attempts,
these checks do not isolate the cause of the small-fixture failures. The guard
cannot repair invalid inputs to its correction. All failed attempts are kept.

Representative actual outputs (one dataset and one design each):

| Model / scope | Arm | Log likelihood | Analytical score |
|---|---|---:|---|
| Predator–prey, T2/N72, FP32 | original | -15.533039093 | [-39.202477, -0.763752, -0.037413, 7.892341, 3.008602, -4.144073] |
| Predator–prey, T2/N72, FP32 | guarded pairwise | -15.617095947 | [-39.973198, -0.771630, -0.039007, 7.547935, 3.529061, -4.820899] |
| SIR, T1/N72, FP64 | guarded pairwise | -36.389353258 | [-0.246543, -0.047089, 5.559299] |
| SIR, T2/N1008, FP32 | guarded pairwise | -81.812324524 | [-143.582336, 52.071945, 38.652504] |

These are finite-program outputs, not accuracy rankings. Missing reference
values remain unavailable. Invalid SIR evaluations are stored with validity
false and null nonfinite likelihood; their zero masked scores are not usable
score estimates. At T1 the final reset cannot affect the current likelihood;
T2 checks that corrected particles reach the next step.

Evidence root: `docs/plans/artifacts/ledh-nonlinear-master-20261002/`. Every attempt
has a campaign file, source hashes/Git state, exact command, worker configuration,
data/observation hash, seed list, device/memory policy, wall time, rows, traces
where requested, and completion status. The final reporting-only revision adds
worker elapsed-time/environment/output fields and conditional simple-arm error
comparisons; numerical smokes predate that reporting revision. No numerical
rerun is needed to support its unit-tested reporting behavior.

## Decision and inference

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Deliver master program | Shared call-chain, model dimensions, valid representative two-step execution, failure preservation | Small SIR failures retained; no silent promotion | Untuned scopes and absent same-target nonlinear oracle | Scope-specific reference and calibration plan before accuracy comparisons | Nonlinear accuracy or default readiness |
| Preserve guarded option | Existing safety checks plus actual nonlinear execution | Invalid upstream correction inputs remain failures | Scope dependence and selected-branch smoothness | Evaluate guarded/unprotected pair on matched, calibrated scopes | Universal safety or HMC readiness |
| Include actual KSC evidence in monograph | Values, coordinate scores and uncertainty transcribed from saved results | Dataset-B likelihood error deterioration explicit | Two fixed datasets, eight designs, exploratory intervals | Broader parameter-neighborhood tests | Population-wide superiority |

| Inference status | Finding |
|---|---|
| Hard veto screen | Small reduced-control SIR FP32 cases are invalid; they cannot support accuracy or runtime rankings |
| Statistically supported ranking | None for these nonlinear smokes |
| Descriptive-only differences | Every nonlinear value/score and moment-loss difference in this delivery |
| Default-readiness | Not established; existing production direction is unchanged |
| Next evidence needed | Scope-specific tuning, independent same-target references with uncertainty, untouched replications and conditional simple-arm comparisons |

Post-run skeptical review: the strongest alternative explanation for differences
is the deliberately different numerical scope, not a change in the underlying
model or evidence of filter quality. Matched references and replicated errors
could overturn any favorable visual impression. The weakest evidence is a
single dataset/design at T2. Engineering execution, numerical validity in a
specific fixture, and statistical accuracy are kept separate.

## Manuscript

The full monograph compiled in three passes to 605 pages, with no undefined
references/citations or duplicate labels. Both chapter copies are identical.
PDF pages 216–219 were visually inspected: the two T120 tables, derivations and
new discussion are readable. Existing global overfull boxes and a later long
source-path overflow remain; this is not a claim that the whole monograph's
layout is clean. The build manifest and verification JSON preserve the PDF hash.
Human prose review remains pending.


## Remote-main integration

Development commit `1ccf9375e` was merged with remote main `3673aebf1`.
The two conflicts were resolved by preserving upstream native RK loops, the
fixed SIR adjacency helper and compiled input preparation while retaining dtype
support, SIR's nine-observation signature, the selected richer design and the
moment-safety option. The SIR topology no longer needs the local eager-validation
workaround because upstream supplies a graph-compatible constant helper.
The richer design remains fixed configuration, constructed outside XLA and
captured identically by value/score and trace programs. An executable assertion
checks that the nonlinear consumer actually receives that design.

Merged validation passed 73 focused CPU reference tests and then 13 endpoint and
driver checks (11 repeated driver checks, 75 distinct checks overall). The latter
checks preserve upstream compiled-kernel cache behavior and input parity through
both IID and permutation public score endpoints. The driver's explicit JIT option
now also reaches the upstream input-preparation API. All CPU runs intentionally
hide GPU devices; logs and manifests are in `merged-validation/`.

Both models passed a fresh GPU/XLA/TF32 check at T=2, N=1008, one dataset and one
design, using the declared controls and guarded pairwise arm. Predator–prey's
log likelihood was -15.615931510925293; SIR's was -81.81239318847656. Every score
coordinate was finite. Trace likelihood and first score coordinate agreed
exactly with ordinary evaluation for both models. Complete scores and settings
are in `merged-gpu-smoke-05/rows.json`. This merged attempt used 97.903 s,
bringing GPU smoke wall time to 539.916 s of the 1800 s allowance.
These are execution checks; small floating-point differences from premerge
artifacts do not establish accuracy changes or a ranking.

The merged monograph also compiled in three passes to 605 pages, with no
undefined references/citations or duplicate labels. The first attempt exposed
five unchanged figures omitted by the sparse checkout; those exact tracked
figures were materialized and a fresh build succeeded. A verification parser
was adjusted for a wrapped log line without rerunning the successful build.
The updated table page was visually inspected. Both chapter copies remain
identical. The final build and hash are in `merged-monograph-build-02/`;
the failed build and dependency-materialization record are preserved.
