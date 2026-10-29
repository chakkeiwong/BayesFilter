> Historical launch/cost-stage note. The Figure4 and queued K jobs have now
> finished; the sequential consumer was subsequently validated. Current outcomes,
> budgets and remaining work are in the
> [terminal execution review](controlled-smc-kdm-terminal-execution-result-20260929.md).

# Controlled-SMC execution expansion: cost, batching and full source matrix

The first validation tranche passed; its immutable evidence remains in
controlled-smc-kdm-first-execution-result-20260929.md. The next reviewed phases
measured the actual Figure 4 source configurations and validated repetition
batching before starting the full matrix. This note records those completed
engineering results. Live execution state is in controlled-smc-kdm-checkpoint.md
and the runner manifests.

## Checked results

| Attempt | Scope | Result |
|---|---|---|
| 13, c3_cost_pilot | H3000, cSMC N128/I3 and BPF N5529; two repetitions at q=.01,.10,.20 | PASS; 12 valid method/repetition rows, one trace per invoked signature; 54.867681 GPU-charged seconds |
| 14, c2_batch | Shared scalar/batched kernel versus the preserved scalar snapshot; independent short replicas plus full-horizon consumers | PASS, 137/137 checks; 88.950966 GPU-charged seconds |

Sequential observed combined warm costs were at most 7.683035 seconds per
cSMC/BPF repetition. The predeclared conservative projection was 23180.232891
seconds (6.44 hours) for the complete 20-by-100 source experiment. The source
BPF count was matched to the authors' original CPU costs, and does not match
our GPU cost.

The batched controlled filter's warm time was 2.426090 seconds per replica
(batch ten), compared with 7.257858 seconds for the sequential middle-noise
pilot. BPF batching was slower: .821205 seconds per replica (batch four),
compared with .425177 seconds in that pilot. These are descriptive measurements
from two batches, not statistically established timing rankings. The selected
execution uses ten independent controlled repetitions per batch and a scalar
BPF. No numerical policy, fitting objective or source setting changed.

Batch validation included every short replica, full-horizon controlled forward
and fit outputs at every refinement, and all four full-horizon BPF replicas
against the immutable scalar implementation. A scalar wrapper check also passed.
The current scalar endpoint and batched endpoint call the same time-loop and
linear-algebra implementation. There is no per-replica map or reduced numerical
fork. Checks used the frozen 1e-7 absolute plus 1e-7 relative tolerance and exact
ancestry equality. A source-hash check confirms the validated core is identical
to the core snapshotted for the full experiment.

Peak TensorFlow allocator usage in the batch-validation process was
4,070,136,320 bytes. Both devices had memory growth configured before
initialization. Computation was on GPU:0, FP64, TF32 off, XLA on; stable
signatures traced once. There was no observed XLA failure in these checks.
This does not diagnose every historical iAPF execution issue.

## Decision and uncertainty

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Unsupported conclusion |
|---|---|---|---|---|---|
| Accept batching for the bounded Figure 4 source run | Preserved-scalar value/fit/ancestry checks PASS | No numerical veto in tested scope | Rare trajectory problems and full-grid costs remain possible | Run frozen source matrix with per-call validity checks and cap | General multidimensional algorithm or universal speedup |
| Run full Figure 4 right matrix | Source settings recovered; executable driver and disjoint streams frozen | Budget and validity stops implemented | Empirical curves and source agreement not yet available at launch | Complete 4000 method/repetition outputs and uncertainty tables | Completion of all C/K replications |
| Implement exact K mixture reference | Appendix B equations give an exact tractable recursion for H5 | Numerical validation queued, not yet established at launch | Author numerical settings/initial conditions remain unrecovered | Execute independent branching/Kalman/gradient controls after C job | Appendix B training replication |

| Inference status | Verdict |
|---|---|
| Hard veto screen | Cost pilot and 137 batch parity checks passed; future invalid fits/weights stop the relevant run |
| Statistically supported ranking | None from these timing pilots |
| Descriptive-only differences | Warm/cold timings, memory and per-stream likelihood/ESS diagnostics |
| Default-readiness | Not established; selected source-comparison execution only |
| Next evidence | Complete Figure 4 source matrix, analyze uncertainty and source differences; finish guided/temporal heuristic comparisons |

## Execution and accounting

The full experiment is attempt15_c3_figure4, with its exact command, input/source
hashes, environment, seed, process identity and time cap in its manifest.
It runs 20 variances with 100 independent repetitions each for cSMC and BPF,
all 3000 observations. Its output root is
artifacts/controlled-smc-kdm-replication-20260928/attempt15_c3_figure4/.
Rows are appended after each completed batch; progress.json identifies the
current cell, method and completed count. No early results may be used to
change tuning, particle counts or stopping criteria.

The validated Figure 4 driver records all fit diagnostics and fitted policies,
rather than discarding computed conditioning, residual and projection arrays.
It computes the source statistic log10(var(logZ)/mean(logZ)^2), not variance(Z).
Its automatic report provides 3999-resample pointwise bootstrap intervals.
A confidence interval overlapping zero is not proof of equivalence.

The full source run has an aggregate cap of 10800 GPU seconds over at most two
attempts. The current allocation is 12000 GPU seconds including prior checks,
drawn from the inherited campaign balance. The next exact K reference check is
queued with a separate 120-second aggregate cap. CPU budget remains
680.3388495521278 seconds. Read runner status for the changing GPU balance.

## Source and review limits

The strongest alternative explanation for a favorable source curve is finite
Monte Carlo error or a source-comparison mismatch. Exact/source checks,
independent streams, the preserved source arrays and uncertainty tables make
that discrepancy inspectable. The weakest remaining evidence is full learned
KDM: only primitive source parity has passed. The complete sampling-to-optimizer
consumer, exact Appendix B settings and adequate independent training runs are
still missing.

The source released as toy_problem_dataset.py implements the nonlinear Cappe
example, not Appendix B's equations (20)--(22). A scoped search of the pinned
repository found no mixture-Kalman or matching linear/bimodal experiment.
The exact reference therefore uses explicitly constructed coefficients for
validation and must not be labelled recovery of author training settings.

Review was a source/code self-audit; no independent reviewer was launched.
The master still requires broader method/heuristic checks, other selected
experiments and a terminal scientific review. iAPF remains incomplete/deferred.
