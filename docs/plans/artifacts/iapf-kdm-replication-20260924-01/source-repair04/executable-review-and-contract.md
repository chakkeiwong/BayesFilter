# Executable replication master: skeptical review and bounded execution

Date: 2026-09-24. Governing plan: ../../iapf-kdm-replication-master-program-2026-09-24.md.
User instruction: review the executable program thoroughly and execute; continued explicitly.
Code checkout: /home/chakwong/BayesFilter/.localresources/worktrees/ledh-iapf-execution-20260922.
Canonical planning/evidence checkout: /home/chakwong/BayesFilter.
Base commit verified: 89065bc6354801cb368d5e163ba49fd9c4372d10.

## Review and corrections before implementation

The existing master is an evidence plan, not an executable supervisor. The old
Equation (15) supervisor has a September 21 deadline and fixed historical output
paths. Do not launch it. Its 59-launch investigation already contains 32 failed
full-filter attempts; blindly repeating them cannot close the original fit
choice. The raw RDS failures and MDPF author checkout exist in the planning
checkout, not at the same relative locations in the numerical checkout. The new
dispatcher must require explicit evidence-root paths and record their hashes.

The correct MDPF entry selects KDEParticleFilter with ImportanceSampling; the
similarly named discrete comparator is wrong for this replication. The author's
generated 11-trial YAML is absent, training switches are disabled, and pretrained
models must be supplied or trained. The current interpreter lacks torchviz,
pynvml and tqdm. Configuration extraction and component-only AST checks cannot
establish that the full training endpoint executes. Attempt the real import
without stubbing dependencies and record any failure. Do not call the author's
main entry, which selects a GPU and can overwrite output.txt in old save paths.

No ranking or learned-method promotion is contemplated in this tranche. Algebra,
saved-failure integrity, source/configuration identity, dependencies and a small
GPU/XLA smoke can answer engineering questions only. Full-run cost estimates
remain unavailable until the real source training chain and fit policy work.

## Evidence contract and research intent

Question: can one executable entry point safely dispatch the verified Stage 0
checks, preserve their provenance, distinguish check completion from replication,
and identify the next actual prerequisites?
Comparators: unchanged local R equation/analytic checks; preserved 32 RDS
failures; pinned author source b0e2fd54db7b6c36d70e8e701ddc6a3f3d5dee18;
TensorFlow CPU analytic identity versus the same small GPU/XLA diagnostic.
Primary engineering criterion: timeout/attempt/budget/no-overwrite checks pass;
each child leaves a log and structured outcome; the selected source/configuration
is recorded and the dispatcher never treats a missing prerequisite as success.
Promotion veto: failing algebra, wrong method, missing dependency, unresolved
source settings, or absent executable consumer evidence blocks the dependent
replication stage. No claim-bearing paper replication runs here.
Continuation veto: corrupt saved evidence, invalid source identity, process
cleanup failure, or exhausted budget. Expected dependency and candidate failures
are repair triggers; independent checks continue.
Explanatory diagnostics: original failure counts, device/compiler visibility,
compile/warm timings and trace count. They do not estimate full training cost.
No statistical ranking, replication completion, posterior accuracy, default
readiness, or end-to-end XLA claim follows from these checks.
Heuristic adversaries and conditional comparisons remain specified in the master
for later scientific experiments; they are not relevant promotion criteria for
this engineering/source-closure diagnostic.
Evidence: a fresh numerical-checkout docs/plans/artifacts/iapf-kdm-replication-
20260924-NN/stage0/attemptNN directory per child, command logs, source checksums,
Git/dirty-source provenance, budget ledger and result.json/result.md.

## Defaults and assumptions

| Choice | Provenance and justification | Failure and smallest check | Status |
| --- | --- | --- | --- |
| Stage 0: 1800 CPU-worker seconds, 900 GPU-worker seconds, 12 launches, 300 seconds/launch | Existing master; user authorized execution | Cumulative ledger, reserve timeout before launch; count failures | Authorized ceiling |
| One worker at a time | Simpler cleanup and attribution, within master maximum | Process-group timeout and interrupt test | Engineering convenience |
| Existing exact R tolerances/seeds | Prior diagnostic source, independently derived identities | Reexecute exact checked source; preserve failures | Reference-only |
| Original generated 11 trials and YAML defaults | Pinned source generator and main loader | Save expanded and merged configs; preserve disabled stages and missing paths | Source defaults, not locally promoted |
| Explicit cross-checkout source/evidence root | Files actually present there | Existence, SHA-256 and commit checks | Required path repair |
| CPU-hidden author import; original PyTorch/NumPy source | Master independent-reference exception | Actual module import; no dependency stubs | Reference-only |
| Tiny TensorFlow GPU/XLA smoke, fixed signature | Device/compiler localization only, memory growth before initialization | CPU identity, repeat values, trace count and allocator record | Diagnostic exception; not method timing |

## Implementation and test scope

Implement a small standard-library supervisor and diagnostic worker, not a new
numerical method. Refuse existing campaign output unless explicitly resuming.
Persist attempts before starting children; bounded process groups receive cleanup
on timeout and interruption. A resumed interrupted attempt is conservatively
charged its reserved timeout. Completed checks and scientific prerequisites have
separate statuses. Only Stage 0 is runnable; later stages remain UNIMPLEMENTED /
NOT_BUDGETED. No environment mutation, package install, full training or historical
result overwrite belongs to this launch.

Focused integration tests must cover timeout descendant cleanup, no overwrite,
budget/attempt refusal and exit-zero with an incomplete prerequisite. Review
source/configuration semantics and reports after the run. The revised bounded
plan passes this skeptical audit; full replication remains gated by the unresolved
fit policy, author endpoint and measured compute.

## Localized reference-environment repair after the first execution

Seven launches are recorded. The TensorFlow environment's PyTorch is CPU-only;
the existing tf-gpu2 interpreter passes an escalated PyTorch CUDA 13.0 device
probe. Its actual author imports stop at torchviz and pynvml. Inherited unrelated
PYTHONPATH projects have already been excluded using python -I.

Revise the initial no-install scope only for a new project-local virtual
environment based on the existing CUDA interpreter. This is a reversible local
prerequisite repair authorized by the requested execution; it changes no shared
environment and is not broad package/environment mutation. Use system-site-
packages to reuse the verified numeric stack. Inspect package resolution before
installing; exclude upgrades/replacements of PyTorch, TensorFlow, NumPy, SciPy and
CUDA packages. If those are required, stop before crossing that boundary.
Record exact resolved versions, commands and installation report. Dependency
preparation is environment setup, not numerical evidence or a method experiment.

The source pin, target, data, method, hardware class, 12-attempt ceiling and
remaining CPU/GPU budget do not change. Numerical source-import retries remain
inside the existing dispatcher ledger. Passing imports or model construction
still cannot establish a complete forward/loss/backward/optimizer call chain or
paper replication. Do not run the author's main training entry or write to its
historical save/data paths. Full-run budgets remain NOT_BUDGETED.
