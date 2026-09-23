# M28 execution review

The active design is [the post-M27 plan](bayesfilter-hmc-post-m27-next-phase-2026-09-23.md).
The opening commit is `47ae8836c`; remote main matches it. Unrelated Q20,
training and agent-policy edits are excluded from this phase's source snapshot.
The opening allowance is 76517.1963238087 CPU and 82259.94093647867 GPU
worker-seconds. This tranche reserves at most 1800 CPU and 4800 GPU seconds,
with four 1000-second fit attempts, 980-second child limits, and at most two
numerical workers. Failed attempts count. Nested sequential children are not
charged twice.

## Pre-execution skeptical audit

The question is whether a supplied map with bounded residual conditional scale
avoids M25's tail defect while the public tuner retains all verified pairs and
the separate posterior procedure measures model-coordinate precision. The
exact noncentered map, independent fixed-count posterior runs, and exact model
means/medians are the declared comparators. Algebra, starts, receipts,
retention, archives/restart and normal exits are engineering criteria;
posterior health/readiness/precision determine each member's posterior result.
R-hat, ESS and MCSE never determine tuning membership or epsilon repair.

M25's actual native supplied-map designs used L=(3,5,9,13,18,25), epsilon .5,
128 measurement and verification draws, four chains and 8 discarded probe
warmup draws. Search used per-L pilots, one refinement round, at most 100
candidates, 300 work units and 40 reserved repair units. M28 preserves these
search hypotheses and serializes their resolved defaults. It does not transfer
the later, hand-supplied intermediate epsilon grid. These controls are inherited
development allocations, not proof that the search is exhaustive. The complete
resolved controller, execution and transport configuration is saved per fit.

The material changes are the new residual map, two members selected by distinct
increasing verified L before posterior sampling, longer explicit counts, and
lugsail mean MCSE. The residual amplitude .5 and width twice scale are
convenience hypotheses with analytically checked curvature bounds. The exact
control and residual use the same model starts. The fixture includes the base
chart's Jacobian; bounded child curvature does not bound its entire Hessian.

The generated suite must be checked against the plan's literal count table
before launch, including 30000 maximum warmup and 60000 maximum retention.
This prevents recurrence of M27's count mismatch. The isolated CLI worker must
report verified GPU growth and XLA placement. CPU checks deliberately hide GPUs
and make no production-performance claim. Source and outputs are versioned.

The review passes for a bounded development experiment. The earliest falsifiers
are independent density/score/Jacobian and inverse-start tests. Wrong target,
corrupt archives, unavailable trusted device or exhausted budget stop the
affected cell. A valid member's failed posterior check triggers diagnosis and
preservation of the second predeclared member, not repeated seed optimization.
Two seeds cannot establish coverage, ranking, general funnel success, learned
transport quality or default readiness. Funnel child means can remain imprecise
even when the exact latent target is Gaussian.

## Execution record

Exact commands, environments, source hashes and wall times are recorded under
`artifacts/hmc-repair-master-2026-09-16/m28-r1/`. The terminal result will include
separate candidate and posterior decisions, cost reconciliation and a refreshed
next phase.

The first GPU launch was stopped before numerical execution: a dsge_hmc worker
acquired GPU 1 after readiness. Its preflight receipt charges 0.128 seconds.
GPU 2 has the same RTX 4080 SUPER hardware class and only the system remote
desktop process. The meter now records and permits the two named system display
processes while rejecting any other compute process. A fresh trusted GPU 2
probe precedes the four numerical fits in fresh r2 directories. This localized
resource repair changes no target, design, seeds or statistical criterion.
The initial unmetered GPU 1 readiness check is conservatively charged 60 seconds
(the bounded readiness allocation), not misreported as a measured runtime.

CPU regression on the frozen source passed all 36 collected tests; the separate
documentation contract suite passed 15. The initial curvature test incorrectly
differentiated a first-order adapter whose base score is deliberately frozen.
It now differences the actual score and passes the analytic curvature oracle.
This was a test defect, not a change to the sampling target or score.

Two additional setup failures occurred before numerical GPU work: the installed
readiness helper supports only devices 0/1, so GPU 2 used the repository's
trusted worker readiness check; an overbroad launcher path replacement pointed
at a nonexistent source-r2 snapshot. Fresh r3 launch directories use the checked
source-r1 snapshot and the unchanged designs. Every failure is charged.

The official book was rebuilt with bibliography resolved and pages 418/419
visually inspected. A recursive TeX input path initially traversed campaign
artifacts; the owned build was stopped and repeated without recursion. BibTeX
then required its output directory as cwd. Both attempts and costs are kept
in guide-build.json. No old equations, sources or qualifications were removed.
