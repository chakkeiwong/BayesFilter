# Handoff to Claude: review the surrogate-HMC master program

Updated 2026-09-22. Workspace:
`/home/chakwong/BayesFilter-ledh-graceful-failure-execution-20260917`.
Branch `ledh-graceful-failure-execution-20260917`; source baseline
`8a4cffa2fd085f43ab5710c24d6e7b48770aea0d`, with recorded working-tree changes.
This memo has not been sent to Claude and no new Claude review has been launched.

The latest [master report](artifacts/surrogate-hmc-all-models-20260922-master-reporting-03/result.md)
and [reporting review](surrogate-hmc-master-reporting-review-2026-09-22.md)
preserve completed results and remaining blockers. No experiment is active.
The review note records a 1.600458-second reporting subreservation overrun;
the full charge remains inside the authorized campaign ceiling.

## Goal and first review

**Test whether endpoint-corrected surrogate HMC driven by an imperfect full
canonical analytical DCTR proposal field gives reliable model-parameter
inference, and at what cost, across the repository's concrete model scopes.**
The goal is posterior inference. Filtering accuracy, force accuracy, mechanics
checks, reference qualification and control calibration cannot substitute for
retained surrogate-HMC comparisons.

DCTR means LEDH-PFPF-OT with GenUT, dual-cap trust-region correction, the UKF
per-particle covariance lifecycle, Contract E–Chol and analytical recursive
scores. The force and acceptance endpoint may use different frozen bindings.
The [Opus reply](surrogate-hmc-all-models-opus-reply-2026-09-18.md) withdrew
its earlier contrary requirement. Endpoint evaluations must remain consistent
with their own binding; the force must remain a deterministic position-only
field under its separate binding. Check the actual consumer call chain.

Start with this one-path prompt:

```text
READ-ONLY BOUNDED REVIEW. Review exactly this path and nothing else unless the
file itself explicitly asks you to inspect a cited line:
docs/plans/surrogate-hmc-all-models-master-program-2026-09-18.md.
Do not edit, run commands, launch agents, or review the whole repo.
Question: does the program correctly test endpoint-corrected surrogate HMC
across the repository's model scopes, preserve B/R, C/D, D/R and C/R, and
separate actual inference from preparation, incomplete scope and budget limits?
End with VERDICT: AGREE or VERDICT: REVISE.
```

Request the next exact source path/range only when needed for a concrete
finding. A design verdict does not establish implementation correctness or
posterior validity. The owner has authorized execution within the campaign
budget; review is advisory and introduces no new approval gate.

## Evidence to preserve

| Comparison | Question |
|---|---|
| B/R | Does full-DCTR-force HMC sample the qualified model-reference endpoint? |
| C/D | Does surrogate HMC sample its own frozen finite-DCTR endpoint? |
| D/R | How does that finite endpoint change model-parameter inference? |
| C/R | Does the practical DCTR-endpoint sampler recover the model reference? |

A D/R discrepancy does not cancel C/D. An unavailable D reference does not
block a separately funded B/R test. A failed candidate or an inconclusive
short run does not reject surrogate HMC as a research direction. Keep true
support exclusion, numerical endpoint failure and a qualified deterministic
force extension separate.

The completed scope results are deliberately partial:

| Scope | Preserved result | Main limitation |
|---|---|---|
| LG1 | B/R, C/D and C/R INCONCLUSIVE; D/R DISCREPANCY | Finite-likelihood effect is distinct from sampling validity |
| KSC T2 | B/R, C/D and C/R INCONCLUSIVE; D/R EQUIVALENT | Empirical reference refinement; retained MCSE diagnostic veto |
| M10 T2 | B/R INCONCLUSIVE after 2,000 draws per chain | Final median-CDF MCSE screen failed; C/D, D/R and C/R NOT_RUN |
| SV20 | B/R INCONCLUSIVE after 2,000 draws per chain | Both looks failed functional MCSE checks; all chunks and original-force checks healthy; C/D, D/R and C/R NOT_RUN |
| LG2 | All 16 claim-endpoint and complete-cost checks pass | B/R pilot forecast 15,327.99 CPU seconds exceeds 1,024.00 remaining; inference NOT_RUN; prospective precision design also required |
| C01 Gaussian controls | Exact, half-strength and rotational fields INCONCLUSIVE; zero force TUNING_REJECTED | Mechanism controls provide no DCTR model inference evidence |

See the [LG1 result](surrogate-hmc-all-models-execution-result-2026-09-19.md),
[KSC result](surrogate-hmc-ksc-finite-result-2026-09-22.md),
[M10 result](surrogate-hmc-m10-inference-result-2026-09-22.md),
[SV20 result](surrogate-hmc-sv20-inference-result-2026-09-22.md), and
[LG2 qualification result](surrogate-hmc-lg2-endpoint-result-2026-09-22.md).
These link the exact commands, archived sources, seeds, tuning, retained
traces and incremental accounting. Do not repeat closed statistical looks.

The [precision audit](surrogate-hmc-inference-precision-result-2026-09-22.md)
shows that the 13-function/four-comparison 2,000-draw pilot has an ideal IID
median-CDF interval radius of 0.02146 against a 0.02 margin, before reference
error. This does not prove that every realized interval must fail, but the
schedule lacks a convincing prospective equivalence-precision justification.
Future scopes need a justified diagnostic policy and complete funded schedule.
The earlier binary-MCSE repair passed only its declared Gaussian calibration;
it has not been promoted to runtime or validated for these 13 functions.
Active and completed runs retain their original rules.

A separate [SV20 force-health audit](surrogate-hmc-sv20-force-health-audit-2026-09-22.md)
checked the completed run's original-force condition: zero fallbacks in
warmup, verification and all eight retained chunks. The generic public tuner
intentionally permits a qualified deterministic field extension. The repaired
consumer enforces the stricter scope condition before posterior functionals;
the history reader verifies inline or archived-audit evidence without adding
looks. Forty-nine focused tests pass, including the actual worker call chain.

Two concrete missing-scope findings must stay visible. The
[structural audit](surrogate-hmc-structural-scope-result-2026-09-22.md) reproduces
the canonical batch consumer's failure on the AR2 singular process covariance.
Adding noise to the deterministic lag state would change the model. The
[range–bearing audit](surrogate-hmc-range-bearing-scope-result-2026-09-22.md)
shows that its nearest-alias density is not the simulator's normalized wrapped
Gaussian law over a general noise-scale parameter space. The default narrow
noise check agrees numerically; no posterior error at that default was shown.
Neither finding rejects the imperfect-force mechanism.

## Review questions

1. Are the model posterior, frozen finite posterior and proposal field defined
   separately, with correct priors, Jacobians, observation timing and fixed
   randomness? Check per-chain potential-plus-kinetic acceptance, cache
   consistency, reversal and the limits of the detailed-balance argument.
2. Does every implementation claim follow the consumer call chain into the
   full shared algorithm? Inspect executable wiring/parity evidence when
   requested; the existence of a general function is insufficient. Preserve
   the FP64-integrator/FP32-DCTR precision distinction for each arm.
3. Are scope, reference and failure classifications honest? Distinguish model
   variants, restricted parameter tasks and controls; retain uninstantiated
   upstream applications. SIR-UKF remains owner-excluded. An approximate
   filter posterior is not a same-law model-posterior oracle.
4. Are summary functions, physical margins, numerical-reference allowances,
   uncertainty multiplicity, diagnostic scope and retained length justified
   before execution? Identify any convenient inherited setting or proxy that
   could produce a misleading pass or an avoidable inconclusive result.
5. Do conditional cheap adversaries, independent tuning, mode/tail checks and
   uncertainty support the precise claim? No stochastic efficiency ranking or
   default promotion is supported by the present partial results.
6. Does the executable master preserve prior outcomes and charge failed
   attempts once, reserve the complete workload, and report unresolved work
   within the unchanged 63 GPU-hour/15 CPU-worker-hour global limits? A
   preparation pass must leave the corresponding inference comparisons NOT_RUN.

Return substantive findings with severity, exact path/line, the violated
quantity or assumption, why it matters and the smallest concrete correction.
Distinguish design defects, implementation gaps and empirical questions.
Use VERDICT: REVISE for an unresolved material flaw. AGREE means agreement
with the stated design and explicit limitations; it does not certify all-model
validation, general feasibility or default readiness. State what was not
checked and do not overwrite old evidence.

The [active checkpoint](artifacts/surrogate-hmc-all-models-20260919-review-tests/checkpoint.md)
contains the next authorized action. Historical reviews remain available for
provenance and do not replace an independent reading.
