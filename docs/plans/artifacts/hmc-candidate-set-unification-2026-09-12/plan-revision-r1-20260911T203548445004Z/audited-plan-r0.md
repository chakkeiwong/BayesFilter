# HMC candidate-set unification and guidebook rewrite

Date: 2026-09-12.
Status: proposed implementation plan; ready for a thorough Claude audit.
Inspected BayesFilter commit: `9be4b8fe7bad711deea61e915c6f95bc0d37649f`.
Owner request: plan one sensible tuning procedure that explores a broad set of
leapfrog counts, retains every viable candidate, and carries those candidates
through further tuning and verification; include rewriting the guidebook.

This change set contains the plan and its audit handoff only. It does not change
the running tuner, rewrite the manuscript yet, or launch experiments. The plan
proposes the successor to the September 1 and September 5 repairs. Their code
remains the current implementation until this migration is implemented; their
historical results and documents remain preserved.

Audit instructions: [Claude handoff](bayesfilter-hmc-candidate-set-unification-claude-audit-handoff-2026-09-12.md).

## 1. Intended outcome and scope

BayesFilter should expose one recommended tuning entry point, one candidate
lifecycle, and one set-valued result. Ordinary coordinates and frozen nonlinear
transports require different preparation, but the subsequent search, measured
refinement, retention, verification, budgeting, and reporting must use the same
controller. A compatibility name must not own a second selection procedure.

The proposed entry point is the existing `tune_hmc_kernel`, extended with typed
preparation for exact ordinary and exact transformed targets. Keep
`tune_fixed_transport_hmc_kernel` as an explicitly classified compatibility
wrapper into that entry point. It must not independently select, stop, repair,
or issue a different kind of result. This is an intentional public-interface
migration, not an assertion that the two current functions are interchangeable.

The typed deterministic proposal-field branch must use the shared lifecycle
where it is supported, while retaining its mechanics-only status and exact
endpoint-potential requirements. It must never become an exact-score route by
dispatch convenience. Migration is incomplete if a package-reachable active
branch still silently performs a powers-of-two/first-pass search.

Scope includes the active tuning dependency paths, result/replay interfaces,
registry, repository-owned consumers, examples, and the tuning-related content
of the guidebook built from `docs/main.tex`. It does not require rewriting
unrelated filtering chapters or expanding this into NUTS, learned-transport
training, or a new posterior study.

## 2. Checked starting point

The following are source observations, not numerical comparisons. Line numbers
refer to the inspected commit; use the named function if they move.

| Evidence | Exact source to inspect | Observation and consequence |
| --- | --- | --- |
| E1 | `docs/reference/hmc-tuning-interface.md:128–170` | Ordinary tuning measures a primary/refinement grid but starts at most two fresh verifications per attempt and stops on a pass. Preserving candidate records does not ensure continued candidate evaluation. |
| E2 | `bayesfilter/inference/hmc_kernel_tuning.py::_select_joint_l_epsilon_candidate`, lines 22211–22291 | Both selection branches prioritize distance from target acceptance among eligible candidates. That order is not an efficiency ranking. |
| E3 | `bayesfilter/inference/hmc_kernel_tuning.py::_run_phase7_direct_candidate_queue`, lines 25148–25468 | The queue records unstarted candidates and stops at `first_admission`; its start quota can leave eligible candidates unverified. |
| E4 | `bayesfilter/inference/fixed_transport_hmc_tuning_tf.py:79–106, 842–1094, 1425–1592, 1992–2075` | Measured joint-grid tuning owns a separate selector and verifier. Out-of-band acceptance is not a validity veto under the measured policy. Audit the actual selection and held-out logic, not only its policy name. |
| E5 | `bayesfilter/inference/fixed_transport_candidate_selection.py:121–219`; `docs/chapters/ch21b_hmc_tuning_interfaces.tex:446–507` | The helper retains all validated candidates, but callers first tune each frozen transport. It is not already an all-pairs tuner or an authoritative replay builder. |
| E6 | `bayesfilter/inference/tuning_contract.py:939–1081`; `bayesfilter/inference/hmc_tuning_dispatch.py:29–100` | The registry describes two active tuner names and different preparation, epsilon, and verification rules. Typed force dispatch is another conditional mechanics branch. |
| E7 | `bayesfilter/hmc_ordinary_selection_policy.py:1–105` | The current ordinary grid, cap of 25, and single midpoint-refinement barrier are centralized but specific to the ordinary policy. |
| E8 | `docs/chapters/ch21_hmc_for_state_space.tex:55–143`; `docs/chapters/ch22_mass_matrices.tex:36–101` | The earlier HMC chapter still selects a “best pair” and describes a different refinement sequence. The mass chapter presents budget formulas whose coefficients still need provenance. Updating only chapter 21b would leave conflicting guidance. |
| E9 | `docs/chapters/ch26b_neutra_transport_hmc.tex:316–367` | NeuTra describes another selection/validation sequence. It must distinguish transport training restarts from multiple HMC settings per frozen transport. |
| E10 | `tests/test_hmc_kernel_tuning_outer_loop.py:3760–3900, 4160–4235`; `tests/test_fixed_transport_candidate_selection.py:1–80` | Tests currently protect both first-admission/quota behavior and all-survivor retention in different interfaces. Migration requires changing the intended assertions, not preserving both as active defaults. |

The motivating M4 file is outside this checkout, at
`/home/ubuntu/python/MacroFinance/results/hmc/daily_asset_midas_phase14_end_to_end_master_20260911_attempt03/tuning_attempt01/tuning_summary.json`.
Its run manifest records BayesFilter commit
`d2124d425b0ea0ae0e3e5f4246bd6b03ff8a2170`; do not pretend it ran the current
HEAD. At JSON path
`bayesfilter_payload.tune_verify_repair_loop.attempts[0].verification_diagnostics.phase7_direct_candidate_queue`,
the recorded counts are three eligible candidates, two starts, and one unstarted
candidate with reason `start_quota_exhausted`. Attempt 1 starts its two eligible
candidates; attempt 2 has no eligible handoff and starts no verification. The
top-level result is `budget_exhausted`, `passed=false`, with an empty
`hard_vetoes` list. These observations establish incomplete tuning and repair
requests, not valid posterior sampling, invalidity of every candidate, or a
ranking of the reported pairs. Preserve this as a motivating control-flow
example; use a small synthetic fixture for regression tests.

## 3. Research intent and evidence contract

| Item | Contract |
| --- | --- |
| Main question | Can one explicit procedure carry all viable measured candidates through bounded refinement and fresh verification without coordinate-specific changes in retention semantics? |
| Mechanism under test | One shared controller and typed preparation adapters, with staged allocation to every survivor, candidate-local repair, set-valued replay, and explicit incomplete status. |
| Expected failure mode | A renamed facade hides old first-pass selectors, starves later candidates, collapses epsilon alternatives within an L, or presents unverified records as admitted kernels. |
| Comparator | Exact current code at the commit above, including the two-start queue, acceptance-distance order, separate fixed-transport selection, and existing set-validation helper. Compare control-flow obligations and the same frozen transition mechanics. The M4 artifact is not a performance baseline. |
| Primary engineering pass criterion | Deterministic adversarial tests demonstrate complete survivor retention, fair stage allocation, correct scope-bound replay for each verified member, unchanged target/transition contracts, bounded repairs/resume, and matching guide/examples/registry. |
| Promotion veto | Lost viable candidate; unmeasured or unverified member exported as verified; target/score mismatch; invalid scope reuse; missing required diagnostics; unsupported backend; divergent active procedures; unsupported scientific wording; failing relevant tests or documentation contradictions. |
| Continuation veto | Shared target/coordinate/geometry corruption, invalid execution environment, corrupted or cross-wired evidence, or exhausted total budget. Local kernel failure and insufficient mixing evidence are not shared invalidity. |
| Repair trigger | Candidate-local numerical failure with an identified repair, poor or inconclusive mixing, acceptance needing exploration, inadequate grid coverage, or localized infrastructure failure. Each trigger has a bounded next action. |
| Explanatory diagnostics | Acceptance, trajectory length, geometry spectrum, movement, runtime, compilation cost, and descriptive ESS summaries, except where the declared validation design assigns a stronger role. |
| Unsupported conclusions | No universal optimal L range, epsilon, mass, acceptance target, convergence guarantee, sampler superiority, or M4/NeuTra posterior readiness follows from controller tests or short smokes. |
| Preserved result | Versioned test and smoke outputs, exact command/environment manifest, candidate tables, code diff, rewritten source/PDF, result note, and Claude audit findings. |

This is primarily an engineering migration. It does not need an expensive
sampler-ranking experiment to prove that the controller retains its members.
Default API activation nevertheless requires the numerical/compilation checks
below; fixing the orchestration must not conceal the ordinary route's known
NumPy/runtime-policy debt. Any later assertion that the procedure finds better
kernels requires a separate target-specific comparison with uncertainty.

## 4. The single procedure

### 4.1 Prepare and freeze a scope

Validate the target measure, matching exact score or explicitly typed proposal
field, data identity, dimensions, dtype/backend, coordinate transforms, and
telemetry. Ordinary HMC may perform qualified windowed mass adaptation. A frozen
transport supplies its exact Jacobian-corrected target and declared latent mass.
Do not silently add ordinary mass adaptation to identity-mass NeuTra.

Freeze the resulting geometry and an explicit dispersed chain bank before
comparing epsilon/L candidates. Bind each candidate to this scope. Reuse the
same declared start-bank design and stage budgets within a scope, with distinct
recorded candidate/stage/replication random streams. Sharing physical target
identity across transports does not make their latent states interchangeable.

### 4.2 Declare broad coverage and a finite proposal budget

One common configuration declares the primary L grid, per-L epsilon proposal
budget, permitted refinement L values or bounded expansion rule, stage ladder,
repair limit, and total compute limits. Serious calls require these choices and
their provenance explicitly. Named convenience fixtures remain nonclaiming.
There is no hidden universal cap of 25 and no separate ordinary versus NeuTra
search algorithm selected by a config type.

The existing six-value L grid is an inherited starting hypothesis. A scope may
reuse it with justification or provide broader coverage within its declared
budget. Positive distinct integers are necessary but do not alone establish
that a grid is broad enough for the target. Record coverage limitations and
predeclare how boundary evidence can propose additional L values. No expansion
may exceed the declared maximum or total budget without a revised plan.

For every primary L, independently qualify epsilon with the same bounded pilot
proposal routine. Dual averaging may propose settings; it does not verify them.
Explicit per-L epsilon proposals may seed that routine, including the existing
fixed-transport grid. Every resulting pair must be measured with a frozen
kernel before it can enter the survivor set. The same epsilon number may arise
at different L values, but its qualification never transfers across L.

Preserve multiple viable epsilon values at one L. Do not collapse them because
one has acceptance closest to a target. Enumerate permitted refinement around
all survivors, deduplicate exact scope/settings matches, and give newly proposed
pairs their own measurement and validation. Epsilon proposals and adaptation
updates are hypotheses; do not assume fixed-L acceptance is monotone or treat
an unmeasured multiplied/divided epsilon as qualified.

### 4.3 Separate health, adequacy, and repair

| Observation | Role and next action |
| --- | --- |
| Exact target/score mismatch or shared coordinate/geometry corruption | Shared continuation veto; stop the affected scope and repair its definition. |
| Nonfinite candidate trajectory, declared divergence, invalid target status, or missing required telemetry | Veto that candidate's promotion. Distinguish a local step/geometry problem from shared implementation failure before scheduling a measured repair. |
| Finite acceptance outside a target band | Tuning/efficiency diagnostic and possible repair proposal. Neither an upper-band crossing nor closeness to 0.70 defines correctness or superiority. |
| High acceptance with absent movement | Movement/health failure under the declared screen; acceptance cannot rescue it. |
| Early R-hat/ESS/MCSE insufficient or unavailable because evidence is too short | Pending evidence or candidate repair. No verified handoff; extend/repair under the declared ladder. Do not reject the L family merely for failing an early short screen. |
| Final declared R-hat, bulk/tail ESS, MCSE, or target-specific mixing screen fails | Promotion veto for that candidate at that evidence budget. Record whether longer evidence or new settings are allowed; do not stop unrelated candidates. |
| Runtime/device failure | Infrastructure classification; preserve the failed attempt and retry locally if the scope and remaining budget permit. |
| Total campaign budget exhausted | Stop new work, preserve completed work and pending candidates, and report incomplete coverage. No scientific rejection is inferred. |

Choose numeric mixing thresholds and any target-specific mode/observable checks
before execution. Modern rank-normalized split/folded R-hat, bulk/tail ESS, and
MCSE must be computed in the coordinates and observables relevant to the claim,
including model parameters when latent diagnostics do not answer that question.
Record mean Metropolis probability separately from realized binary acceptance.

### 4.4 Advance every survivor through declared stages

Use a stage barrier or equivalent persisted fair queue: give every current
survivor its next declared allocation before advancing any survivor to a longer
stage. A first pass does not end the campaign. The old two-start quota becomes
an execution-batch limit at most; it must not erase the rest of the queue or
force another mass adaptation before they receive service.

The default fairness contract is comparable per-chain transition allocations
within a stage, with measured gradient counts and time recorded. Longer L costs
more and that cost consumes the total budget. Preflight checks must budget the
full declared cohort and reserve final validation work. If the minimum coverage
cannot fit, report an under-budgeted design before launch; do not silently run
the first few candidates. Runtime overruns leave an explicit partial result.

A local epsilon/L repair creates a new candidate with parent lineage; it does
not overwrite the old evidence or discard other healthy candidates. Local
repair does not rerun mass adaptation without a geometry-specific reason. A
geometry change creates a new scope and requires fresh qualification; old
verification cannot be transferred to it.

Within a frozen validation sequence, longer rungs may continue from saved chain
states. Record cumulative draw ranges so they are not counted twice or called
independent replications. After adaptation or any kernel change, restart the
declared warmup/validation sequence and use fresh validation streams. Keep all
adaptation, discovery, comparison, and validation draws out of later posterior
estimation. Preserve them as separately labeled evidence.

Every survivor must face the declared fresh final verification, independent of
the data used to choose its settings. A failure can trigger a bounded repair
with a new reserved verification stream; the failed holdout remains recorded.
Do not repeatedly inspect a fixed-size confidence interval at arbitrary stops
and claim nominal coverage. Predeclare assessment rungs and use a justified
sequential uncertainty method if inferential optional stopping is intended.

### 4.5 Return a set; nominate separately

The proposed versioned result, `HMCTuningCandidateSetResult`, contains:

- all candidate records and their complete scope/settings/parent identities;
- mechanically screened survivors, pending repairs, rejected candidates, and
  candidates left incomplete by budget;
- `verified_candidates`, each with its own frozen settings, final states,
  diagnostic evidence, and replayable repository-issued handoff;
- separate campaign coverage/completion and per-candidate evidence status;
- optional `nominee_id`, nomination criterion, and uncertainty status; and
- total work, remaining budget, resume state, source/environment identity, and
  explicit posterior/scientific authority fields.

`viable_candidates` must not mean both screened and verified in different
adapters. Define it as screened, non-vetoed candidates still eligible for further
work; expose `verified_candidates` separately. A result can have verified
members while the campaign remains incomplete. Report that fact without
claiming the full search completed. Replay requires an explicit verified member
ID; neither list position nor an implicit acceptance winner selects a kernel.

Default behavior returns the set without a nominee. Optional efficiency
nomination uses predeclared observables, budgets, independent replications, and
an explicit objective such as minimum relevant ESS per measured target-gradient
evaluation or per wall time. These objectives are not interchangeable. A
descriptive nominee may be useful when uncertainty is inconclusive, but its
choice never removes other verified members or establishes superiority. A
statistical ranking requires an appropriate uncertainty and multiplicity
analysis for the declared candidate family.

Selecting a verified kernel is separate from posterior estimation. Existing
sequential NeuTra retained-sampling requirements remain applicable. This change
does not introduce adaptive switching or pooling across kernels inside a
retained chain. Such a mechanism would need its own sampling argument.

## 5. Default and numerical-assumption audit

Numbers below are inherited observations or proposed bounded engineering
choices, not calibrated scientific defaults.

| Choice | Provenance and status | Failure mode | Early check or decision |
| --- | --- | --- | --- |
| L = (3,5,9,13,18,25), current upper bound 25 | September 5 owner-directed ordinary policy; inherited coverage hypothesis for the new procedure | Missed useful integration scales; accidental universal cap | Explicit scope grid, boundary/coverage report, measured permitted extensions; remove route-specific hardcoding from the new controller. |
| Target acceptance 0.70 and band [0.65,0.75] | Existing repository settings; tuning heuristics | Noisy band crossing mistaken for invalidity; poor movement at high acceptance | Separate acceptance and movement, uncertainty, and final mixing evidence; no acceptance-distance elimination. |
| Four-chain bank | Existing ordinary/validation convention; inherited baseline | Common starts conceal poor exploration or modes | Require explicit dispersed start provenance and appropriate observable/mode diagnostics; do not claim four chains suffice universally. |
| R-hat 1.01; helper ESS 400/200 and MCSE/SD 0.10 | Existing interface/helper screens; not newly calibrated here | Insufficient evidence or inappropriate observable coverage | Scope-specific validation specification, recorded limits, and declared extension/repair behavior. Do not silently copy helper constants into a new universal policy. |
| Fixed-transport 4+16 transition convenience screens | Current config, short diagnostic budgets | Short chains mistaken for mixing or efficiency proof | Keep mechanics-only fixtures explicit; serious configurations require their own ladder. |
| Single survivor-midpoint refinement barrier | Existing ordinary engineering rule; proposed shared initial refinement hypothesis | Sparse grid or resonance missed; repeated refinements explode cost | Explicit complete proposal bound, multiple epsilon retention, and target-specific coverage review. |
| Per-stage equal transition allocation | Proposed fairness convention | Expensive L values consume budget; finish order biases nomination | Preflight whole-cohort cost and final-verification reserve; record actual gradients/time and incomplete work. |
| Mass freezing and identity mass in declared latent coordinates | Existing target/geometry contracts | Comparing changed kernels; transferred stale epsilon bounds | Same-scope signatures and same-kernel replay/parity fixtures; requalify after geometry changes. |
| TensorFlow/TFP, GPU, XLA, memory growth | Applicable repository policy | Nominal shared controller still calls NumPy or uncompiled kernels | Dependency/runtime checks, stable signatures, trusted small GPU/XLA checks; preserve explicit non-admission until blockers are repaired. |
| Scientific tolerances and random seeds | Not selected for a new scientific campaign by this planning task | Arbitrary constants acquire scientific authority | Preserve existing named fixture values where justified; future target-specific protocols must state their own derivation or provenance before running. |

## 6. Implementation work packages and acceptance

| Phase | Work and principal files | Completion evidence |
| --- | --- | --- |
| P0: freeze the migration surface | Use `scripts/inventory_hmc_tuning_routes.py`, `scripts/audit_ordinary_hmc_migration_surface.py`, registry/exports, examples, benchmark callers, and replay consumers. Read the bounded active dependency closure, including `hmc_kernel_tuning.py`, `hmc_tensorflow_tuning.py`, `hmc_tuning_state.py`, and existing artifact helpers. Save the current guide sources/PDF identities. | One inventory classifies each reachable route as migrate, compatibility wrapper, diagnostic, or historical; lists NumPy/runtime debt and existing failing tests separately. No whole-repository cleanup. |
| P1: common policy and result | Implement the controller's pure state machine/configuration, shared candidate identity, stage/repair budget accounting, resumable queue, and set result. Proposed new modules are `bayesfilter/inference/hmc_candidate_set_tuning.py` and `hmc_candidate_set_artifacts.py`; names may be adjusted without changing semantics. | Stubbed adversarial tests prove lifecycle and accounting before numerical integration. No one-use launch tokens or custom authorization system. |
| P2: typed numerical adapters | Extract/reuse ordinary preparation and fixed-transport target construction; migrate the touched ordinary NumPy execution path to TF/TFP and host bookkeeping to Python standard types. Route all supported active config branches through P1, including the mechanics-only force branch. | Same frozen target, scores, coordinate transforms, and transitions agree with independent eligible references; callable dependency checks show no diagnostic NumPy runtime import. Kernels have stable `tf.function` signatures and default XLA; no pfor. |
| P3: replay and consumers | Update public dispatch, `inference/__init__.py`, `tuning_contract.py`, `hmc_route_contract.py`, artifact/replay builders, wrappers, and in-repo consumers. Version semantics; consume verified member IDs. Preserve existing authority restrictions until actually resolved. | One controller is executable through all supported wrappers; every verified member can replay its own kernel; failed/pending/stale members cannot. Old payloads stay readable in their declared legacy/mechanics role and are not relabeled. |
| P4: guidebook and examples | Rewrite the narrative and align all active cross-references listed below; regenerate registry tables and update example/test contracts. | Consistent source and rendered book, runnable examples, complete substantive comparison to the protected source baseline, and recorded reader-review limitation. |
| P5: bounded integration and closeout | Run focused tests, known-target checks, compilation/device checks, downstream static audit, and documentation builds. Resolve scoped findings and publish a local result/reset note with the final audit disposition. | All engineering criteria pass or unresolved blockers are stated directly. No new default activation while an active adapter silently retains a different procedure or violates runtime policy. |

No automatic edits, lock updates, or M4 reruns in MacroFinance/dsge_hmc are part
of this BayesFilter implementation plan. Produce concrete migration instructions
and static compatibility findings for those repositories. Their owners choose
when to adopt the versioned API. Existing experiments cannot change procedure
mid-run because the library's default changed.

## 7. Guidebook rewrite specification

Rewrite the tuning explanation as scientific exposition, followed by a compact
API reference. The reader should understand why geometry, epsilon, integration
length, mixing, and computation cost are separate choices before seeing config
fields. Explain one common procedure once; use ordinary and transformed targets
as examples of its preparation step.

| Surface | Required rewrite |
| --- | --- |
| `docs/chapters/ch21_hmc_for_state_space.tex` | Replace the stale “best pair”/local-refinement algorithm with the set procedure and a worked candidate table. Preserve exact target/score requirements, same-L qualification, mass-change requalification, and Metropolis mechanics. |
| `docs/chapters/ch21b_hmc_tuning_interfaces.tex` | Make this the complete procedure: preparation, broad coverage, independent per-L epsilon proposals, measured pairs, all-survivor stages, repairs, fresh verification, set result, optional nomination, resume, and posterior separation. Replace the route menu with one entry point and typed preparation examples. |
| `docs/chapters/ch22_mass_matrices.tex` | Explain geometry preparation/freezing and when a genuine geometry repair invalidates old tuning. Audit budget formulas/coefficients; preserve justified mathematics while labeling heuristic allocations accurately. |
| `docs/chapters/ch25_diagnostics.tex` | Align acceptance, movement, divergence, modern R-hat/ESS/MCSE, observable coordinates, uncertainty, and candidate-versus-campaign failures. Explain why an early short-chain failure can require more evidence. |
| `docs/chapters/ch26b_neutra_transport_hmc.tex` | Preserve the change-of-variables mathematics and sequential retained sampling. Show the hierarchy frozen transport -> geometry scope -> multiple (epsilon,L) kernels, retaining all viable kernels for each transport. Remove duplicate hidden tuning rules. |
| `docs/chapters/ch26c_hnn_surrogate_hmc.tex` | Repair tuning references only; preserve the exact endpoint-potential and proposal-mechanics assumptions and the branch's weaker authority. |
| `docs/reference/hmc-tuning-interface.md`, `docs/generated/hmc_tuning_route_table.{md,tex}` | A concise operational reference generated/aligned with the new registry, including compatibility mapping, candidate IDs, incomplete status, and replay semantics. Generated tables remain an inventory, not an alternative procedure menu. |
| `docs/examples/hmc_tuning_*.py`, `docs/examples/fixed_transport_candidate_selection.py` | Demonstrate ordinary/frozen preparation into the same call, multiple retained pairs at one L, optional nomination, budget-limited resume, and explicit verified-member replay. Clearly mark illustrative configuration values. |
| Other active book/API cross-references discovered in P0 | Correct only confirmed contradictory guidance, including claims about default XLA and publicly visible tuning summaries. Full local reports must expose the candidate set; externally redacted reports may preserve actual privacy boundaries. |

Required worked example: start with several L values and multiple measured
epsilon values, show more than one survivor, let the first candidate pass while
later candidates still receive their stages, show a local repair and an
unfinished budget-limited member, then return the verified set. Use explicitly
synthetic values or captured fixture output. Do not disguise invented numbers
as the M4 run, combine attempts across changed geometries, or call a nominee
the best sampler without uncertainty support.

Before making mathematical/literature claims, inspect the relevant method,
theory, and appendix sections and official implementation where available.
Check the existing Neal/Hoffman-Gelman/Betancourt citations and the sources for
modern diagnostics and any acceptance-optimality claims actually used. Keep
local tractable paper copies and exact source/equation anchors. Acceptance
targets derived for a specific asymptotic setting must not become universal
finite-target rules. This plan itself makes no uninspected optimality claim.

Protect the actual current chapter sources, bibliography, examples, and PDF
identity before rewriting. Track every removed equation, derivation, citation,
assumption, finding, or qualification and explain its disposition. Inspect the
rendered PDF for equation explanations, tables, captions, cross-references,
transitions, and stale policy language; successful compilation is insufficient.
Naturalness and final reader acceptance remain subject to human feedback;
provisional drafting and technical integration may continue before it arrives.

The separate `docs/fable-rewrite/monograph` tree exists and currently does not
include chapter 21b. Preserve it as a separate manuscript unless its active
status is established in P0. Do not silently copy the rewrite into protected
historical/staging manuscripts. Identify which PDF and source tree the delivery
actually updates; `docs/main.pdf` is not assumed to match today's source.

## 8. Verification, budget, and commands

### Focused engineering checks

Create meaningful controller/replay tests in the proposed
`tests/test_hmc_candidate_set_tuning.py` and
`tests/test_hmc_candidate_set_artifacts.py`. Required adversarial cases are:

1. The first candidate passes, a later candidate also passes, and both complete
   the declared stages and remain replayable.
2. More candidates survive than the old two-start quota; later members receive
   work before an earlier member advances to a longer stage.
3. Multiple epsilon values at one L survive; refinement includes every survivor
   and no pair is admitted without being measured.
4. Out-of-band finite acceptance alone does not reject a candidate; high
   acceptance with no movement does not pass its declared health screen.
5. Early insufficient R-hat/ESS evidence takes a permitted extension/repair path;
   a final screen failure blocks that member while unrelated work continues.
6. Local failure/repair preserves other candidates and frozen geometry; shared
   invalidity stops the affected scope; changed geometry requires fresh tuning.
7. Budget exhaustion/resume preserves partial progress, RNG/seed lineage,
   endpoint states, draw ranges, work charges, and pending order without
   repeating completed chunks or granting verification to unrun members.
8. Every verified member replays its own settings; wrong ID/scope, stale source,
   changed transport/mass/backend, or pending/failed status is rejected.
9. All supported entry names use the same controller; compatibility wrappers,
   typed force bindings, diagnostics, and legacy payloads cannot gain authority
   by relabeling. Default examples cannot accidentally select an old procedure.
10. Discovery/validation draws never enter posterior estimates, and uncertainty
    summaries do not treat continuation chunks as independent replications.

Use the existing public API, outer-loop, fixed-transport, selection, dispatch,
route, replay-authority, and documentation tests as regression sources. Replace
old first-admission/quota assertions for the new version; preserve them only
where they test an explicitly historical reader. Do not leave contradictory
active behaviors merely to keep old assertions passing.

Known-target checks should exercise a small exact Gaussian and a smooth
nonlinear target with an explicit invertible transform, in ordinary and frozen
coordinates. Compare the same frozen finite transition/value/score to eligible
independent references. Add controlled resonance/no-movement and candidate-local
failure fixtures. These establish mechanics and diagnostics, not a stochastic
ranking or reliable posterior estimation from a short run. Choose dimensions,
seeds, transition counts, and numeric tolerances from inspected fixtures or
checked error arguments in the P0 execution note before launching them.

### Proposed execution envelope

The planning task uses no HMC or GPU work. For later execution of this plan,
the proposed engineering envelope is 60 minutes cumulative CPU-command wall
time, 20 minutes cumulative GPU-process wall time, and at most four GPU smoke
launches including retries, on one existing local GPU at a time. These are
convenience-chosen cost ceilings for interface qualification, not statistical
sample-size claims. Compilation and failed attempts consume the same envelope.
Limit each GPU smoke to five minutes, derived from the total ceiling divided
by the launch bound. A run that cannot answer its engineering check within the
remaining budget stops as under-budgeted; do not weaken its check to fit.

Before those GPU launches, P0 must record the exact smoke cases, commands,
settings, and tolerances to the execution note under this contract. This is a
bounded specification task, not approval-token machinery. No serious M4,
NeuTra training, posterior, or sampler-ranking run is included. Such runs need
their own target-specific tuning and statistical evidence budgets.

Use a fresh output root for every execution/repair, proposed as
`docs/plans/artifacts/hmc-candidate-set-unification-2026-09-12/<UTC-run-id>/`.
Record Git commit and relevant dirty-file diff, command, environment, device and
memory-growth provenance, XLA/dtype settings, seeds, actual wall time, outputs,
plan/result paths, and remaining budget. Use ordinary checksums and atomic
checkpoint replacement where needed; do not introduce one-use authority files.

Future commands below are templates, not commands already run. Set the
task-specific `HMC_UNIFY_RUN_ROOT` to a fresh absolute output root and use the
inspected TensorFlow environment
`/home/ubuntu/anaconda3/envs/tfgpu/bin/python`; verify its versions before tests.
The new test files must exist before their commands are used.

```bash
CUDA_VISIBLE_DEVICES=-1 TF_FORCE_GPU_ALLOW_GROWTH=true /home/ubuntu/anaconda3/envs/tfgpu/bin/python scripts/inventory_hmc_tuning_routes.py --check
CUDA_VISIBLE_DEVICES=-1 TF_FORCE_GPU_ALLOW_GROWTH=true /home/ubuntu/anaconda3/envs/tfgpu/bin/python scripts/audit_ordinary_hmc_migration_surface.py --downstream-root /home/ubuntu/python/MacroFinance --downstream-root /home/ubuntu/python/dsge_hmc --output-dir "${HMC_UNIFY_RUN_ROOT:?set a fresh output root}/consumer-audit"
CUDA_VISIBLE_DEVICES=-1 TF_FORCE_GPU_ALLOW_GROWTH=true /home/ubuntu/anaconda3/envs/tfgpu/bin/python -m pytest -q tests/test_hmc_candidate_set_tuning.py tests/test_hmc_candidate_set_artifacts.py tests/test_hmc_tuning_dispatch.py tests/test_hmc_tuning_contract.py tests/test_hmc_tuning_documentation_contract.py tests/test_hmc_tuning_policy_replay_authority.py
CUDA_VISIBLE_DEVICES=-1 TF_FORCE_GPU_ALLOW_GROWTH=true /home/ubuntu/anaconda3/envs/tfgpu/bin/python -m pytest -q tests/test_hmc_kernel_tuning_public_api.py tests/test_hmc_kernel_tuning_outer_loop.py tests/test_fixed_transport_hmc_tuning.py tests/test_fixed_transport_candidate_selection.py
CUDA_VISIBLE_DEVICES=-1 TF_FORCE_GPU_ALLOW_GROWTH=true /home/ubuntu/anaconda3/envs/tfgpu/bin/python scripts/render_hmc_tuning_interface_docs.py --check
```

CPU commands intentionally hide GPU devices. Tests importing TensorFlow must
establish that setting before import. Default GPU checks require trusted tool
execution and `TF_FORCE_GPU_ALLOW_GROWTH=true` before import plus verified
memory growth before device initialization. Use the installed read-only GPU
probe for readiness, then a separate bounded smoke launcher. No nontrusted GPU
failure proves that the hardware or method is broken. Pure numerical kernels
need stable graph signatures and XLA compatibility, equivalence, memory, and
compile/steady-state observations; short checks do not establish performance
superiority. No pfor or unreviewed backend substitution is permitted.

Regenerate tables before their check. Build the guide from working directory
`docs`, using an absolute output directory outside the protected PDF path:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error -outdir="${HMC_UNIFY_RUN_ROOT:?set a fresh output root}/book" main.tex
```

Inspect the resulting PDF and compare it with the saved
source baseline. Record pre-existing unrelated book build failures separately
and repair only those required for this delivery; do not quietly report a
chapter-only build as a successful full guidebook build.

## 9. Skeptical audit and pre-mortem

This plan was audited before writing implementation or experiment commands.
The naive “reuse the candidate-set helper” plan fails: the helper consumes
already tuned transports and cannot issue per-pair tuning handoffs. Merely
removing the early `break` also fails: the start cap, result schema, replay
selection, local repairs, budget reservations, and documentation still assume
one chosen pair. Both errors are addressed by P1–P4.

| Risk examined | Resolution or remaining limitation |
| --- | --- |
| Wrong baseline or stale context | Pin current code and the distinct M4 commit; compare control flow and same-kernel mechanics, not historical speed or acceptance outcomes. |
| Proxy metrics promoted | Acceptance affects proposals/description; mixing screens have declared roles; optional efficiency ranking needs its own uncertainty evidence. |
| Missing stop conditions | Candidate, scope, and total-budget failures are separate; incomplete members survive in the result/resume state. |
| Unfair comparison | Stage allocation precedes further allocation to earlier candidates; bind geometry/start design/budgets; report differing cost and compilation explicitly. |
| Hidden numerical defaults | Inventory above exposes inherited grids, thresholds, chain counts, and screen budgets; serious scope values require provenance. |
| Environment mismatch | Plan includes the NumPy dependency migration, CPU-hidden tests, and separate trusted GPU/XLA checks. A wrapper around the old ordinary runtime is insufficient. |
| Artifacts cannot answer the question | Tests must inspect execution traces and replay every verified member. A candidate list in JSON or one successful selected kernel is insufficient. |
| Unlimited retention means unlimited computation | Preserve all records; allocate additional work only under the declared proposal/rung/repair and total budgets. Budget exhaustion is explicit incompleteness. |
| Guide rewrite loses rigor | Compare substantive source changes and inspect the rendered full book; preserve target/score, transform, uncertainty, and qualification arguments. |

Audit disposition: coherent enough for independent plan audit and bounded
engineering specification. Scientific fixture settings and GPU commands remain
to be frozen in P0; they are not inferred from the cost ceilings above. This is
not a claim that Claude has audited the plan or that implementation has passed.

A misleading success would be two adapters returning the same schema while
retaining different early-stop behavior. The cheapest discriminator is a shared
injected outcome schedule with several survivors exercised through each public
entry. A misleading failure would be GPU compilation timeout or short-chain
R-hat failure being blamed on candidate retention. Distinguish these with pure
controller tests and same-kernel reference checks before broadening runtime.

## 10. Completion and reset note

Completion requires one shared active procedure; all declared survivors either
receive their required work or are explicitly incomplete; all verified members
can be replayed; repair/resume works without lost evidence; legacy and typed
authority boundaries remain truthful; eligible TF/TFP execution passes the
scoped checks; and the book, reference, registry, and examples tell the same
story. A nominee is optional. Posterior inference remains separate.

The implementation result note must include a decision table with primary
criterion status, veto status, main uncertainty, next action, and unsupported
conclusions. Any stochastic summaries also need an inference-status table:
hard vetoes, viable candidates, supported ranking (normally none for smokes),
descriptive differences, default-readiness limits, and next evidence needed.
Keep engineering correctness, numerical validity, and scientific interpretation
separate. Add the strongest alternative explanation and evidence that would
overturn each material conclusion.

One material Claude plan audit and one terminal implementation/result audit
are the intended independent reviews. Findings must identify concrete source,
mathematical, numerical, cost, or migration risks. Review is advisory under the
current repository governance; formatting disputes or reviewer unavailability
do not create new launch authority requirements.

Reset for the next agent: this is the planned successor to “one tuner per target
class.” Do not execute a third parallel procedure, rerun M4, overwrite old
results, or rewrite unrelated dirty work. Start with the audit handoff, then P0
after implementation is requested. Current unrelated dirty files concern the
SSL-LSTM recovery runtime and sigma-point work; none were edited for this plan.
