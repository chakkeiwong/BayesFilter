# HMC tuning procedure repair plan

Date: 2026-09-14. Baseline: `9329cadf`. This executes the owner's request to
repair the nine findings in `bayesfilter-hmc-whole-procedure-gap-review-2026-09-14.md`.
Earlier numerical bridge tests establish their tested mechanics, not completion
of the whole tuning procedure. Unrelated working-tree changes are protected.

Completed locally on 2026-09-15 after the skeptical plan audit and terminal
execution audit. See
[the execution audit](bayesfilter-hmc-whole-procedure-repair-execution-audit-2026-09-14.md)
for G1–G9 dispositions, 314 passing latest regression results, the final GPU/XLA
mechanics check, rebuilt guide, preserved attempts, and target-specific limits.
The implementation sequence and pre-run contract below are preserved as the
plan under which the work executed. No real-data campaign or commit/push is
included in this task.

## Intended behavior and evidence contract

One controller owns broad L coverage, exact-pair measurement, fresh verification,
bounded repair, further evidence, retention, completion, and restart. Preparation
and transition implementations may differ with the target and coordinates; they
must not select a different scheduling or admission procedure. Every verified
member remains available by explicit identity. No acceptance-distance, ESS, or
R-hat winner removes another viable candidate.

Acceptance and valid numerical evidence qualify an exact frozen kernel. A
promotion veto can reject that kernel while permitting a smaller-epsilon child.
Shared execution corruption invalidates the whole scope. Inconclusive evidence
either has declared future work or an explicit terminal disposition at its cap.
R-hat is reporting-only during tuning, including when its computation fails.
Posterior warmup, cumulative R-hat/ESS, and scientific assessment remain separate.

The engineering comparator is the specification above and the nine reproduced
gaps, not the old tests' first-admission behavior. Pass requires public-call
regressions, durable restart through numerical replay, consistent route/guide
claims, and bounded actual numerical checks. Corrupt evidence, scope drift,
replay of an unverified member, missing observations, or lost work veto completion.
Acceptance and runtime in small numerical checks are descriptive only. This work
does not qualify MacroFinance targets, rank samplers, prove convergence, or launch
a posterior campaign.

## Implementation sequence

1. **Roles, lifecycle, and validation (G2/G3/G6).** Preserve evidence validity,
   promotion vetoes, repair eligibility, and diagnostic alerts in observations
   and receipts. Permit supported directional repairs from measurement and
   verification, with immutable same-L children. Reject repair factors <=1,
   nonintegral/bool integer settings, and duplicate proposals. On a direction
   reversal propose an unvisited interior epsilon, explicitly as a measured
   hypothesis. Add finite evidence rungs and terminal `inconclusive_at_cap`;
   unfinished rungs must survive reload. Rungs use fresh independently seeded
   runs of the unchanged kernel. Their interval decisions are operational
   screens, without a nominal repeated-look coverage claim.
2. **Durability and resource accounting (G4/G7).** Persist every observation,
   execution specification, tensor evidence, completed work, attempts, and
   controller ledger at bounded checkpoints, including zero-verified runs.
   Numerical evidence files are immutable; checkpoint files may be atomically
   replaced. Refuse final-output collisions before work. Public resume checks
   source, target, geometry, policy, tensors, and queue identities. Resource
   failures pause with evidence; unexpected exceptions preserve state and
   propagate. Optional reporting errors cannot erase numerical evidence.
   Reserve at least measurement plus verification and permit explicit bounded
   top-ups for extensions/retries. Keep call accounting separate from transition,
   leapfrog/gradient-work, and elapsed-time accounting. Numerical work uses bounded
   chunks; a wall limit is checked between chunks and includes reported preparation
   time. A single native call/compilation cannot be preempted by a Python deadline;
   document that limit.
3. **Shared search policy (G5).** Retain explicit per-L epsilon grids as initial
   hypotheses. Add an optional per-L pilot that proposes but never qualifies a
   pair. Freeze proposals before measurement. Refine around all surviving
   families, retain multiple epsilons per L, deduplicate exact pairs, and explore
   only declared L/epsilon domains with finite round/candidate limits. Trajectory
   alerts may propose new declared L values; they never silently mutate an
   existing candidate. Metric-derived epsilon bounds are proposal safety bounds,
   not cross-L qualification.
4. **Public migration (G1/G8/G9).** Translate supported ordinary/fixed-transport
   preparation options into the shared controller and common result. Keep
   position-field transition authority explicit while sharing its scheduler.
   Unsupported legacy customization must fail with an actionable migration
   error before work; no silent fallback to a first-winner procedure. Historical
   implementations/readers remain explicitly classified and cannot issue new
   default tuning authority. Update active library consumers and examples; do
   not reactivate historical experiments. Align preparation's XLA default with
   owner policy, keeping explicit diagnostic exceptions. Remove optional tuning
   R-hat gates/tie-breakers; preserve posterior assessment.
5. **Guide and terminal audit.** Rewrite normative reference and book passages
   around the final executable procedure. Update registry and agent guidance.
   Check public signatures, imports, consumers, source inventory, tests, and
   rendered affected book pages. Record each finding's disposition and any
   remaining target-specific limits in an execution audit.

## Default and assumption audit

| Choice | Provenance/status | Justification and failure diagnostic |
| --- | --- | --- |
| Broad L grid | Existing user-reviewed ordinary grid; target-specific hypothesis | Preserve coverage; test that every primary member executes before refinement. |
| Supplied epsilon values / warmup epsilon | Warm starts only | Independently measure every pair; deliberately oversized Gaussian checks repair. |
| Repair factor 2 | Inherited proposal heuristic, not monotonicity theorem | Strict direction checks and deduplication; opposite evidence tests interior proposals. |
| Finite evidence rungs | Explicit caller design; any library sequence is a convenience default | Check independent streams, unchanged identity, cap disposition, and resume. No superiority or nominal sequential-coverage claim. |
| Acceptance/health policy | Existing reviewed evaluator | Preserve actual evidence roles; synthetic corruption and movement tests distinguish scope failure from repair. |
| GPU/TFP/XLA/memory growth | Owner execution policy | Focused actual GPU/XLA smoke plus existing target preparation tests; no claim that all targets are qualified. |
| CPU/non-XLA checks | Explicit mechanics/reference exception | Hide GPUs before import; artifacts state the exception. |
| Resource ceilings | Engineering convenience limits | Verify preflight and ledger; no time/gradient count used to rank candidates. |

The library evidence multipliers `(1, 2, 4)`, reciprocal refinement factors
`(0.8, 1.25)`, one automatic pilot and refinement round, 256-draw chunks, and
automatic call budget `3 * max_candidates` are convenience choices for bounded
execution. They are serialized and configurable search hypotheses, not
scientifically qualified defaults for arbitrary targets. The first 64 draws
come from the existing acceptance policy's four blocks of sixteen decisions;
they do not guarantee that an acceptance decision will be conclusive.

On a directional proposal crossing the declared epsilon domain, measure its
unvisited boundary before declaring the direction exhausted. Opposite evidence
may propose a geometric interior point; neither rule asserts monotonicity.
The direct operational-preparation binding preserves the final coordinate and
metric signatures with its epsilon proposal bound and intersects the requested
domain with that bound.

Validation budget: up to 60 CPU minutes of focused tests/builds and 10 GPU minutes
of tiny Gaussian/adapter mechanics checks, no real-data sweep, training, package
mutation, or long MCMC. These are convenience ceilings for regression validation,
not scientific sample-size justifications. Use
`/home/ubuntu/anaconda3/envs/tfgpu/bin/python`; CPU checks set
`CUDA_VISIBLE_DEVICES=-1 TF_FORCE_GPU_ALLOW_GROWTH=true BAYESFILTER_TEST_DEVICE_SCOPE=cpu`.
GPU checks use trusted execution and verified growth before device initialization.
Preserve exact commands, Git/diff identity, versions, seeds, device policy,
wall times and outputs under
`docs/plans/artifacts/hmc-whole-procedure-repair-2026-09-14/`, with fresh run
subdirectories. Stop an invalid diagnostic, preserve it, repair locally, and
retry within budget. Stop for a changed target/method, exhausted budget, or an
irreducible correctness flaw; a failed candidate alone does not stop repairs.

## Skeptical audit before implementation

The audit passes after making three limits explicit. First, a policy decision
string alone cannot fix G2: adapter, controller, receipt, and replay must agree
on roles. Second, changing draw counts during resume would change the bound
policy, so the complete rung sequence must be frozen before execution. Third,
legacy config translation is a real API migration: unsupported options require
an explicit error, and active consumers need common-result tests. Merely adding
a wrapper or relabeling historical execution would leave G1 open.

Wrong baselines and stale MacroFinance context are excluded by using the current
public call chains. Short-chain acceptance, R-hat, and timing are not promotion
proxies for posterior correctness. Fairness means equal evidence allocation
within each cohort; longer L consumes more work without losing its draws.
Budget exhaustion yields a partial search rather than pretending completion.
Unexpected target/programming errors must remain visible instead of becoming
candidate failures. Checkpoint tests must reconstruct actual numerical evidence
and produce retained members; controller-only reload is insufficient. The
smallest discriminating checks are the review's reproduced failures, followed
by complete public preparation/search/restart/replay fixtures.

The terminal audit also checks observation-to-tensor agreement on checkpoint
reload, position-field persistence within a longer stage, elapsed time at chunk
checkpoints, and engineering preflight health independent of candidate admission.
These are localized completion checks under the same evidence contract and
validation budget; no additional campaign or scientific default is introduced.
