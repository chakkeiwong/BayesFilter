# M13 FP32 repeatability localization — 2026-09-24

Current status: optional diagnostic, deferred under the
[owner's warning policy](surrogate-hmc-fp32-warning-policy-2026-09-24.md).
No comparison process has been launched. The prepared batch is no longer a
prerequisite for the master or a pending approval that blocks it. Its code and
earlier 100-test preparation remain available if later evidence warrants
localization. The latest policy review passed 125 focused CPU tests.
The original comparison design below is preserved for that optional use.

## Research intent and scope

The master tests surrogate HMC: whether canonical analytical DCTR forces can
support valid posterior sampling with a separately declared endpoint acceptance
target. This optional diagnostic investigates the saved-run warning. The latest
same-scope M13 FP32 replay differed from the saved values and scores by up to
0.00497578 and 0.0138047 after scaling each difference by one plus the
corresponding absolute saved quantity. These are not posterior errors.
FP64 analytical derivative and trace checks passed.

Question: does the FP32 difference track the source revision, or do fresh
processes produce different results from unchanged source? The candidate under
test is the current default consumer against the archived consumer-03 source.
Neither source is tuned or admitted for M13 posterior sampling. No likelihood
law, numerical controls, observations, tolerance, prior, or endpoint is changed.
This is part of the existing master, not a change in scientific direction.

Expected failure: repeat compilation or a changed output graph can affect FP32
numerics. Another possibility is an unintended source or input mismatch. Those
explanations must be separated before choosing a repair. A difference from one
saved process alone does not establish its cause.

## Evidence contract

Comparator: the N256, FP32, T20, nine-point arm and source dependency snapshots
under artifacts/surrogate-hmc-closure-20260922-m13-consumer-03. This is a
post-reset September 23 artifact, not a pre-August-21 LEDH baseline.

Run four fresh, sequential processes in this fixed order: archived-1, current-1,
archived-2, current-2. Each creates the same target, compiles the nine-point
batch once with XLA and original five scalar correction diagnostics, and calls
the same compiled graph twice. Archived workers copy the checksummed dependency closure into their own fresh
diagnostic source directory and import it from there. The archived snapshot
lacks transport_chunk_policy.py; the current copy exactly matches its recorded
SHA-256 and is recovered with that check. Every source origin is recorded, and
any hash mismatch stops the worker. The live checkout is not
restored or replaced. Each factory must issue its own source identity; the
harness may not stamp one. Validate all non-source manifest fields, fixed
inputs, parameter points, and TensorFlow version against the saved arm.
Record actual loaded files and verify the consumer-to-batch-to-score callable
bindings during every numerical worker.

Primary engineering criterion: per-point value and each score component agree
within the unchanged 3e-5 scaled limit, for archived fresh repeats, current fresh
repeats, the two matched source pairs, and each fresh run against the saved
baseline. Retain bitwise equality checks within each compiled graph. Report
all nine points and both score components separately.

The result classifies observed behavior: within-process variability, archived
fresh-process variability, current fresh-process variability, source-associated
difference, saved-baseline mismatch, or all observed comparisons matching.
All comparisons remain visible even when more than one differs. Two fresh
replicates can expose variability; agreement cannot establish general
repeatability or prove a unique cause. No stochastic ranking is attempted.

Continuation vetoes: invalid/non-finite finite program, wrong data/control/source
scope, unexpected callable binding, changed reference or archived source,
TensorFlow version mismatch, missing GPU placement or memory-growth policy,
missing artifacts, failed focused tests, or exhausted batch/global budget.
A valid comparison exceeding the tolerance is a repair finding, not a veto
on the remaining declared comparisons or on surrogate-HMC research.

Explanatory diagnostics: ESS, covariance and correction status tensors, compiler
logs, wall time, and the known untuned likelihood error. Persist all returned
status tensors separately so the compact report does not discard observability.
No tolerance relaxation, control selection, likelihood-accuracy certification,
full-support reference qualification, HMC, posterior conclusion, or numerical
default promotion follows from this batch.

Cheap adversaries, constructed for this question: (1) the saved same-scope
finite program detects regression; (2) unchanged archived source in separate
processes detects variability predating the edit; (3) unchanged current source
in separate processes tests current variability; (4) repeated calls to one
compiled graph distinguish runtime variability from fresh-process effects.
Condition comparisons on every saved noise-scale setting, including the
low-noise points. This is not an inference-method efficiency comparison.

## Assumptions and skeptical audit

| Choice | Provenance and status | Failure mode | Earliest check |
| --- | --- | --- | --- |
| N256/T20, nine points and fixed designs | Frozen consumer-03 baseline | Changed inputs masquerade as regression | Hashes and exact non-source manifest equality |
| Existing correction/reset controls | Untuned hypothesis retained solely for localization | Weight collapse magnifies small arithmetic differences | Per-point validity and retained status tensors; no quality promotion |
| Two processes per source | Bounded diagnostic allocation | Sparse replication misses intermittent variability | Report limited inference and every individual result |
| Alternating source order | Convenience order to reduce simple time confounding | Shared host/device state still differs | Device/version provenance; no causal proof from two pairs |
| Fresh FP32-only workers | Deliberate process isolation | Original preceding FP64 compilation could affect saved result | Compare both sources to each other and to saved result; leave compilation-history explanation open |
| 3e-5 scaled limit and exact within-graph check | Existing engineering criteria, unchanged | Scaling can be misread as posterior error | Preserve absolute and scaled differences; no posterior claim |
| GPU/XLA with memory growth | Repository execution policy | CPU/eager or different dependency code could hide failure | Trusted worker, device checks, call-chain identity, source hashes |
| Saved source importer | Diagnostic comparator only | Cached current modules cause a mixed implementation | Reject preloaded numerical dependencies; inspect loaded paths and actual callable globals |

Pre-run skeptical audit: source imports must be isolated before numerical
dependencies load; merely possessing archived files is insufficient. Both arms
must execute the same frozen scope and default output contract. Qualification
of the nine-point likelihood reference does not certify the finite program or
posterior. A passing regression test is not surrogate-HMC success. Prior
consumer attempts remain preserved and their limits remain enforced.

Pre-mortem: all four processes could agree yet differ in an untested tail; all
could fail because imported code is mixed. Exact scope/binding checks address
the second explanation. The fixed nine-point scope and two replicates cannot
address the first, so this stage is diagnostic only.

## Bounded execution

One batch, at most four numerical subprocesses, one per declared case. Each
has a 300 GPU-wall-second timeout and 300 CPU-second allowance. Required
review uses at most 240 CPU seconds; summary uses 60 CPU seconds; parent uses
120 CPU seconds. Total cap: 1,200 GPU-wall seconds and 1,620 CPU seconds.
The entire batch is proposed upfront, with no approval between its workers.
No automatic numerical retry is added.

At preparation start the campaign has charged 42,299.956468 CPU seconds and
33,604.547754150 GPU seconds, leaving 11,700.043532 and 193,195.452245850
under the unchanged 54,000/226,800 ceilings. CPU preflight is charged separately
before approval. This reservation does not expand the campaign total.

The preceding replay plan allowed one numerical replay and no automatic retry;
that attempt is used. Approval of this batch adds these four specific diagnostic
processes to the master without reopening LG1 or authorizing M13 HMC.

Exact command, from the repository root:

```text
/home/chakwong/anaconda3/envs/tftwogpu/bin/python scripts/run_surrogate_hmc_program.py --stage closure --closure-stages m13-repeatability --output docs/plans/artifacts/surrogate-hmc-closure-20260922-m13-repeatability-01
```

Use escalated GPU access and the existing absolute-interpreter/master-script
allow-list prefix. The same exact command with --dry-run must validate the stage
graph, reservation and fresh output root. No shell environment assignments,
pipes or redirects are needed; the master configures each subprocess.
Persist receipts, manifests, source snapshots, focused-test logs, four worker
reports and status tensors, and a comparison report in this versioned root.
The master records source changes during execution as invalid evidence.

## Decisions and interpretation

| Decision | Primary criterion | Veto status | Main uncertainty | Next action in master | Not concluded |
| --- | --- | --- | --- | --- | --- |
| All observed repeats and pairs match | Original tolerance satisfied in this batch | All scope/validity checks required | Intermittent variability and original failed replay remain unexplained | Explain the original failure before promotion | General repeatability, tuned M13 likelihood or HMC |
| Archived or current repeats differ | Repeatability failure observed | Numerical validity assessed separately | Compilation/process sensitivity | Isolate unstable operation or compilation history | Source edit is uniquely responsible |
| Repeats match but source pairs differ | Source-associated difference observed | All scope/validity checks required | Which graph change drives it | Minimal source-change isolation | Unique causal proof or numerical default readiness |
| Invalid scope, target or artifact | Comparison invalid | Continuation veto | Harness versus numerical failure | Repair the concrete invalidity | Evidence against surrogate HMC |

| Inference status | Bound |
| --- | --- |
| Hard veto screen | Source, inputs, callable, validity, device, artifact and budget checks |
| Statistically supported ranking | None |
| Descriptive-only differences | Four process results, same-graph repeats, runtime and status tensors |
| Default-readiness | No promotion from this batch |
| Next evidence needed | Specific repair evidence, scoped tuning, then target-specific posterior validation |

Post-run review must consider compilation history and device/compiler
nondeterminism before attributing a difference to source. The weakest evidence
is only two fresh processes per source. Preserve the failure even if subsequent
replays match.

CPU preflight audit decision: the source isolation and diagnostic-only design
answer the stated implementation question. Proceed with bounded CPU import,
fixture and orchestration tests; no GPU numerical stage is authorized by
this preflight. Preserve the measured charge and any failed test.

## Completed review and command preflight

The first CPU preflight found two harness errors: a missing archived
transport-chunk policy file and JSON stringification of integer seed keys.
The source importer now recovers a missing snapshot file only when its
recorded SHA-256 matches, into a new directory. Manifest comparison uses the
same JSON representation on both sides while retaining every non-source
scope field. Neither repair changes numerical calculations.

The second preflight passed all 100 focused tests, including actual fresh
CPU processes that import, prepare and verify the archived/current consumers,
record GPU hiding, and check the compiled function's signature without running
the filter. Tests also verify real saved-dependency dispatch, rejection of a
mixed callable, rejection of changed source and scope, all comparison outcomes,
complete stage selection, budget totals, dependency blocking and the one-batch
attempt ceiling. git diff --check passed.

Artifacts:
- surrogate-hmc-closure-20260924-repeatability-preflight-01: preserved failure,
  34.892992 CPU seconds, zero GPU seconds.
- surrogate-hmc-closure-20260924-repeatability-preflight-02: 100 tests pass,
  35.717382 CPU seconds, zero GPU seconds.
- surrogate-hmc-closure-20260924-repeatability-command-preflight-01: exact
  master dry run and five command-policy checks pass, 0.285527 CPU seconds,
  zero GPU seconds. Full checked command arrays and policy matches are in
  report.json; measured charges are in accounting.json.

Current cumulative charge is 42,370.852369 CPU and 33,604.547754150 GPU seconds.
Remaining global allocation is 11,629.147631 CPU and 193,195.452245850 GPU
seconds. The dry run confirms the separate 1,620 CPU / 1,200 GPU reservation.
The final receipt includes the command preflight's small cost after the dry
run inspected earlier receipts; the master re-reads them at actual launch.

Final skeptical review: the checked consumer call chain uses the recorded
source in the archived arm and the checkout source in the current arm.
The recovered dependency has the exact original hash. Fixed inputs, law,
design, controls, point set, dtype, TF32 and XLA settings must match. The
standalone FP32 compilation differs from the original multi-arm process
history by design and remains a competing explanation if both source variants
differ from the saved baseline. No unresolved implementation finding blocks
requesting this bounded diagnostic batch. Numerical repeatability itself
remains untested until execution.

NEXT within the existing master: run the single four-process batch above
after approval. The command and budget are concrete and checked. Execution is
paused at the preceding replay plan's explicit one-attempt boundary; the
global budget remains available. Do not mistake this preparation result for a
completed M13 surrogate-HMC test.
