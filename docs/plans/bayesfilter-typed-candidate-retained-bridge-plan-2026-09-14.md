# Typed HMC candidate evaluation and retained replay repair

Date: 2026-09-14  
Baseline: `4990085e`  
Status: implemented and audited; validation passed

## Question and delivery

Can BayesFilter measure and freshly verify every candidate through its own
TF/TFP evaluator, then replay an explicitly chosen verified member with exactly
the same target, geometry, epsilon, L, and execution settings after process
exit? The delivery is a usable numerical candidate-set adapter and retained
bridge, with durable continuation and a concrete integration reply for
MacroFinance. It completes this missing portion of the existing unification
plan; it does not substitute a new selector or select a scientific winner.

The motivating review is
`bayesfilter-typed-candidate-retained-bridge-review-2026-09-14.md`. The pinned
MacroFinance snapshot predates the ordinary TF backend repair. Preserve the
current backend and R-hat repairs rather than reinstating their old blockers.

## Evidence contract and intent ledger

| Role | Requirement |
| --- | --- |
| Primary engineering criterion | A real deterministic-target TF/TFP tune produces verified members; an explicit member bridges, exports, reloads with the same numerical state/settings, and continues from the preceding retained endpoint without retuning. |
| Comparator | Existing TF/TFP fixed HMC transition and validated affine transforms, using the same state, epsilon, L, and seed; the controller's old callback route remains the scheduling-only comparison. |
| Promotion veto | Invalid or inconclusive acceptance evidence; failed target/score/state/proposal health; no movement or declared recurrence/divergence veto; missing or mismatched candidate, scope, evidence, geometry, source, backend, or state. |
| Continuation veto | Invalid target/execution binding, corrupted evidence, failed invariant, or exhausted validation budget. A rejected candidate alone does not stop the planned repairs or tests. |
| Repair trigger | Directional acceptance evidence creates a fresh same-L child; an inconclusive child remains unverified. Infrastructure or fixture failures trigger localized repair and a focused rerun. |
| Explanatory diagnostics | R-hat, acceptance summaries beyond the declared decision, runtime, and compiler/device observations. R-hat never gates ordinary tuning. Retained acceptance does not become a posterior-convergence test. |
| Not concluded | No MacroFinance posterior validity, target-specific XLA qualification, sampler superiority, or universal numerical defaults from small known-target tests. |
| Preserved evidence | Versioned validation directories under `docs/plans/artifacts/hmc-typed-retained-bridge-2026-09-14/`, this execution note, and the Markdown reply. |

## Implementation

1. Repair controller state transitions: both inconclusive decisions remain
   unverified for repair children; failed children have failed states. Validate
   live member replay with the same complete record/receipt checks as durable
   replay. Preserve valid multi-repair descendants even when an ancestor did
   not pass.
2. Add a repository-issued executable binding. Bind the actual exact
   value/score adapter, frozen affine mass layers or an explicitly supported
   frozen transport, actual start bank, preparation lineage, acceptance
   policy, numerical budgets, backend/dtype, XLA mode, and source files.
   Provide an ordinary preparation entry point that consumes and revalidates
   BayesFilter's existing windowed-mass handoff, preserving its two affine
   layers and start-bank convention. Unsupported coordinates/forces fail
   closed. Source hashes and ordinary durable manifests suffice; no new
   approval-token or security protocol is needed.
3. Implement numerical observations by reusing the repository TF/TFP fixed
   transition and `evaluate_hmc_acceptance_evidence`. Every proposal is
   evaluated with its own epsilon and L; measurement and verification use
   distinct stateless streams. Capture health for discarded warmup as well as
   measured transitions. Freeze verification endpoint and evidence by member.
   Caller callback decisions and `qualification_status` cannot issue this
   numerical evidence. Explicitly bind XLA and numerical evidence identities
   without relabelling historical controller artifacts.
4. Expose the three requested retained builders through one validation and
   execution implementation. The result/member and repository binding must
   agree, and actual numerical verification evidence must pass. Keep mechanics
   and claim-eligible entry points explicit; eligibility never establishes
   scientific correctness or posterior convergence. Non-XLA execution requires
   an explicit scoped exception and cannot silently become GPU/XLA evidence.
5. Export/load the binding and member with frozen numerical geometry and the
   verified endpoint. The caller supplies the actual model adapter on reload;
   BayesFilter reconstructs and validates geometry and kernel state. Persist
   retained samples, diagnostics, final state, settings, and parent archive
   identity. Continuation uses the predecessor's final active state and a fresh
   declared seed. Never include tuning draws in retained samples or apply the
   tuning acceptance band as a posterior convergence gate.
6. Update public exports, capability/status discovery, the reference guide,
   and guidebook. Include an executable integration example. The MacroFinance
   reply must identify the API, call graph, identity and continuation contract,
   tests, commit, limitations, and replacement for its finite-acceptance-only
   callback. Do not edit or launch the MacroFinance campaign in this task.

## Defaults and assumptions

| Choice | Provenance / justification | Failure mode and early check |
| --- | --- | --- |
| Four chains, float64, dependence-aware acceptance policy | Existing repository runner and `HMCAcceptancePolicy` contract, not newly calibrated | Wrong shape/dtype or insufficient evidence; constructor checks and the existing minimum of four blocks of at least sixteen draws. |
| Explicit measurement, verification, warmup, repair, and retained budgets | Caller contract; no new scientific count defaults | Short chains silently pass; reject budgets below the policy minimum and preserve inconclusive decisions. |
| XLA enabled by default | Owner backend policy | Unsupported target silently falls back; explicit signatures, backend capability checks, CPU reference followed by trusted GPU/XLA smoke, and recorded exceptions. |
| Existing TF/TFP transition and affine conventions | Checked repository implementation comparator | A reconstructed mass layer changes the target/score or effective kernel; nonidentity two-layer value/score, coordinate, and same-seed transition parity tests. |
| Source and target identity supplied through validated adapters | Existing trusted research model contract | A changed model/data/closure retains a label; bind actual source files and preparation content and reject drift; target mathematics itself still needs the consumer's evidence. |
| Tiny known-target tests | Engineering fixtures only | Passing mechanics mistaken for posterior evidence; fixtures and artifacts explicitly exclude scientific promotion. |

## Skeptical audit before execution

The baseline and active upstream backend changes were inspected. The original
memo's all-ancestors-must-pass condition is wrong and is replaced by valid
lineage plus passing selected-child verification. Merely wrapping callback
receipts is insufficient: the new binding must own numerical evaluation and
state capture. A live verified-ID check is also insufficient because the
controller currently mishandles inconclusive repaired children. Both flaws
are addressed before exposing a bridge.

Retained execution must reuse the same transition and geometry rather than
switching from independent scalar-chain tuning to an unrelated archive
kernel. Tuning and retained sample counts may differ; frozen kernel settings
must not. Health evidence must cover discarded transitions, and diagnostics
must not silently promote R-hat or retained acceptance to an unintended gate.
Existing XLA restrictions cannot be cleared by a label; qualify the supported
implementation with numerical parity and real compilation, and document
remaining target-specific requirements. These checks address wrong baselines,
proxy promotion, unfair comparisons, stale context, environment mismatch,
hidden defaults, and successful commands that would fail to answer the question.
The revised plan passes this audit within the bounded engineering scope.

## Validation and execution limits

Run pure-controller regressions first, then CPU-only numerical tests with
`CUDA_VISIBLE_DEVICES=-1`, `TF_FORCE_GPU_ALLOW_GROWTH=true`, and the `tfgpu`
Python environment. Add meaningful negative controls for out-of-band finite
acceptance, hard vetoes, inconclusive repairs, forged live IDs, valid two-repair
descendants, changed candidate/target/mass/start/source/backend/XLA settings,
changed persisted tensors, missing state, and invalid continuation. Test every
supported preparation/coordinate mode and ensure unsupported modes reject.

Run a trusted GPU/XLA known-target smoke only after the CPU checks pass. Verify
memory growth before device initialization and record device/compiler policy,
seeds, git source, elapsed time, and artifact paths. Convenience validation
limits are sixty CPU minutes across test/debug invocations and ten GPU minutes
across at most two smoke attempts; these bound engineering cost and are not
scientific thresholds. Stop and record a limitation if those limits are reached.
No long MCMC comparison or MacroFinance run is authorized by this plan.

Finally run the affected tuning, replay, dispatch, and documentation suites,
route inventory, compilation, diff checks, and guidebook build/render inspection.
Audit the actual implementation against the contract and test matrix, repair
material findings, record execution results, and then commit/push only this
task's changes. Preserve all unrelated dirty work.

## Implementation audit before the GPU check

The numerical evaluator, both preparation paths, all three retained builders,
and durable reload/continuation are implemented. The first complete focused
suite passed 34 tests in 64.79 seconds; the standalone CPU reference example
passed in 14.38 seconds. GPU devices were intentionally hidden for CPU checks.
These are engineering checks, not posterior or performance comparisons.

The audit verified that numerical decisions come from the repository acceptance
policy, every fixed candidate supplies its own epsilon and L, fresh verification
has a distinct stream, discarded warmup remains in health traces, and R-hat
is recorded without influencing admission. The numerical binding preserves the
actual two-layer preparation and reconstructs supported frozen affine and dense
IAF transports. Retained replay uses the same runner, validates receipt and
repair ancestry, and continues from a checked predecessor's endpoint. A caller's
qualified label cannot supply missing numerical evidence.

Testing exposed and repaired an unchecked unused-evidence checksum and the
fixed-transport dispatcher's forwarding of the typed adapter in place of the
base target. The underlying runner now permits zero hidden burn-in, and the
finite-check and candidate-health trace wrappers compose. The preparation test
uses the rotated Gaussian's actual transformed maximum frequency to construct
its trial epsilon grid; transporting a convenient unit-Gaussian grid did not
provide a passing verification. Frozen-affine fixtures use the codec's checked
exponential scale convention, and dense-IAF fixtures include its required hash
fields. These fixture repairs changed no scientific default or target.

The GPU smoke will compare the same fixed transition with a reference at
the same initial state, epsilon, L, seed and compilation/device mode, and check actual compilation,
bounded tracing, memory growth, export/reload and continuation. The example's
wider acceptance band and small grid are explicit mechanics fixtures. Passing
does not qualify MacroFinance's target or justify a statistical ranking.

### Comparator repair after the first GPU attempt

`gpu-xla-01` compiled with XLA, verified two members and completed retained
export/reload/continuation in 55.33 seconds, then failed the proposed cross-mode
same-seed draw comparison (maximum difference 3.73876). That comparison had an
unexamined assumption: identical integer seeds need not produce identical
random innovations across XLA and non-XLA paths. This failure invalidates the
draw-equality comparator; it does not, by itself, diagnose incorrect dynamics.
The original attempt remains preserved and is not a passed validation result.

The repaired comparator checks equality against the existing runner with the
same device and compilation mode, then separately checks every GPU proposal
against a deterministic CPU Gaussian leapfrog reference using the actual
captured initial momentum. For the identity-mass standard Gaussian,
`p <- p - epsilon*q/2`, `q <- q + epsilon*p`, `p <- p - epsilon*q/2`,
repeated L times, must reproduce proposed q, final p and
`log_accept_ratio = (||q0||^2 + ||p0||^2 - ||q1||^2 - ||p1||^2)/2`.
This directly tests the numerical quantity without treating random-seed
agreement across implementations as a mathematical requirement. The comparison
tolerance 1e-10 is inherited from the repository's float64 reference checks.

Capturing momenta also adds finite-momentum checks. A CPU fixture invocation
exposed TFP's singleton momentum-parts list; the trace now unwraps the one
tensor state explicitly and rejects unsupported multipart capture. The repaired
CPU example passed in 14.93 seconds: same-backend draws matched exactly;
position, momentum and log-acceptance reference errors were at most 1.78e-15.
The second GPU attempt remains within the original two-attempt/ten-minute
budget. Re-audit passes because the revised artifacts answer transition parity
directly; neither acceptance nor compiler success substitutes for that check.

## Completion

The second GPU/XLA attempt passed in 65.38 seconds. Same-backend retained
draws matched exactly; deterministic CPU position, momentum and acceptance
energy errors were below 1.8e-15. The final affected tests comprise 394 distinct
passing cases, including a real numerically repaired child, fresh-interpreter
continuation, two nonidentity affine layers, both supported transport codecs,
identity/corruption controls, and high-R-hat reporting-only behavior. The final
candidate-execution suite passed 37 tests in 82.90 seconds. The guidebook builds
and the changed pages were inspected. The tested GPU source closure matches
the delivered implementation.

The detailed execution audit, commands, failure classifications, evidence-role
tables and limitations are in
`bayesfilter-typed-candidate-retained-bridge-execution-audit-2026-09-14.md`.
The MacroFinance response records the public API, integration changes and
implementation commit. No MacroFinance campaign was run or authorized by
these engineering tests.
