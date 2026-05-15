# Supervisor audit: DPF monograph implementation-evidence master and subplans

## Date

2026-05-16

## Scope

This audit covers the reviewer-grade DPF monograph implementation-and-evidence
planning lane only:

- `docs/plans/bayesfilter-dpf-monograph-implementation-evidence-master-program-2026-05-16.md`;
- IE0--IE8 subplans dated 2026-05-16;
- reviewer-grade reset memo continuity text.

It does not cover student DPF baseline plans, controlled student-baseline
experiments, production `bayesfilter/`, or monograph chapter edits.

## Codex Initial Inspection

Codex inspected:

- the reviewer-grade reset memo;
- the reviewer-grade final readiness report;
- Chapter 26 debugging-crosswalk diagnostics;
- existing reviewer-grade P0--P13 planning state.

Interpretation:

- P0--P13 reviewer-grade writing/audit work is already complete;
- the next missing work is implementation evidence, not another prose rewrite;
- the correct planning artifact is a new non-production implementation-evidence
  program with phase subplans.

## Created Plan Set

Codex created:

- IE0 preflight, inventory, and plan audit;
- IE1 source-review intake or explicit deferral;
- IE2 Chapter 26 diagnostic harness design;
- IE3 linear-Gaussian PF/EDH recovery;
- IE4 affine-flow PF-PF density/log-det checks;
- IE5 soft-resampling and Sinkhorn controlled tests;
- IE6 learned-OT teacher/student/OOD residual tests;
- IE7 same-scalar HMC value-gradient checks;
- IE8 posterior sensitivity and research evidence note.

The proposed evidence root is:

`experiments/dpf_monograph_evidence/`.

The plan forbids production `bayesfilter/` edits/imports, student-baseline
evidence use, real DPF-HMC/DSGE/MacroFinance target chains, GPU/network/API
runs, broad dependency installs, and bank-facing/model-risk/production claims.

## Claude Review Iteration 1

Claude was launched as a bounded read-only worker through:

```text
bash /home/ubuntu/python/claudecodex/scripts/claude_worker.sh --cwd /home/ubuntu/python/BayesFilter --name dpf-implementation-evidence-plan-review-1 --model sonnet --effort high "<critical review prompt>"
```

Claude returned: `REJECT`.

Must-fix themes:

- missing research-run governance artifacts;
- ambiguous clean-room versus production-import semantics;
- incomplete IE2 schema and run-manifest contract;
- unsafe stochastic PF interpretation;
- affine-flow and Sinkhorn over-reading risks;
- missing learned-OT artifact/provenance gate;
- HMC/posterior safety gaps;
- IE8 trusted-reference and naming overreach;
- source-support propagation gaps;
- missing promotion/continuation/repair classification.

Codex agreed these were execution-readiness blockers.

## Repairs Applied

Codex revised the master, subplans, and reviewer-grade reset memo to add:

- clean-room no-production-import policy;
- mandatory research-run governance artifacts;
- IE2 schema fields for comparator id, source-support class, promotion and
  continuation status, repair triggers, environment, command, artifacts,
  CPU/GPU policy, and uncertainty status;
- boundedness defaults for CPU-only execution, wall-clock caps, no hidden
  retries, cache policy, seed policy, and compiled/eager parity;
- PF stochastic uncertainty and descriptive-only rules;
- affine-only non-implications;
- Sinkhorn trusted marginal-residual comparator;
- learned-OT artifact/provenance entry gate and teacher-origin class;
- HMC same-scalar, repeatability, and fixed-target reversibility/energy smoke
  boundaries;
- IE8 restriction to analytic or controlled fixtures unless a later actual-
  target-instantiation phase is accepted;
- source-support propagation into Markdown and JSON result summaries;
- phase-local promotion/pass, promotion-veto, continuation-veto, repair-trigger,
  and explanatory-only classifications.

## Claude Review Iteration 2

Claude was launched again as a bounded read-only worker through:

```text
bash /home/ubuntu/python/claudecodex/scripts/claude_worker.sh --cwd /home/ubuntu/python/BayesFilter --name dpf-implementation-evidence-plan-review-2 --model sonnet --effort high "<second critical review prompt>"
```

Claude returned: `ACCEPT`.

Claude's rationale:

- prior blockers were repaired;
- clean-room import semantics are unambiguous;
- IE2 now provides the schema/run-manifest gate;
- stochastic, affine, Sinkhorn, learned-OT, HMC, source-support, and IE8
  trusted-reference constraints are explicit;
- IE3--IE8 contain promotion/continuation/repair classifications.

Claude's residual risks:

- IE1 should default to deferral if ResearchAssistant authorization is
  ambiguous;
- IE8 result wording must avoid over-reading controlled-fixture evidence;
- reset-memo continuity remains conservative.

Codex repaired the IE8 wording by replacing `research-credible` with
`controlled-fixture-supported`.

## Codex Verdict

Codex verdict: `ACCEPT`.

The master program and IE0--IE8 subplans are good to go as plans.  This is plan
readiness only.  No implementation-evidence phase has been executed yet.

Next executable phase:

`docs/plans/bayesfilter-dpf-monograph-implementation-evidence-phase-ie0-preflight-plan-2026-05-16.md`

Execution must begin with IE0 and must not skip directly to harness
implementation.
