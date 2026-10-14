# Gap-resolution revision: pre-run review and diagnostic contract

Date: 2026-09-24. Authority: the master and owner's request to update, review and
execute. Numerical checkout remains
/home/chakwong/BayesFilter/.localresources/worktrees/ledh-iapf-execution-20260922.
Output remains docs/plans/artifacts/iapf-kdm-replication-20260924-01.
Nine launches consumed 17.157416536007 CPU and 5.8910572070017 GPU-worker seconds.
Limits remain 1800 CPU / 900 GPU seconds, 12 launches, 300 seconds per child.
No budget reset or full campaign is included. Source review, implementation and
focused harness tests are routine development work, recorded separately.

## Skeptical audit before implementation

Reusing PASS by job name after source/interpreter changes is a material
provenance flaw. Repair dependency-aware reuse before more launches.
The previous MDPF fixture bypassed trainer construction, loaders and optimizer
updates. The new check must call the author's setup_training endpoint and
record diagnostic size overrides. Original training paths must receive no writes.
New PyTorch versus old author code makes scheduler keywords and dataset
deserialization concrete compatibility risks. Inspect and record narrow repairs;
package-wide changes are unnecessary.

For iAPF, increasing variance and shrinking amplitude can lower the residual
without recovering a useful twist. A different optimizer is a reconstruction
hypothesis. Check its Jacobian, unchanged objective, shape/scales and exact controls.
Do not add a new floor, clipping, parameter bounds or log-residual objective.
The revised conceptual plan passes with these corrections; implementation tests
must also pass before launch. Later handlers, full experiments, method-level XLA
and scientific success remain absent. The two lanes may proceed independently.

## Evidence contract and research intent

Question: can bounded repairs close the real author training endpoint and
nominate a usable local reconstruction of the unchanged iAPF fit?
Comparators: pinned MDPF source b0e2fd54db7b6c36d70e8e701ddc6a3f3d5dee18;
saved original Eq15 failures; independent Gaussian/Jacobian identities.
Primary engineering criteria: actual setup/constructor/loader/optimizer/validation/
checkpoint calls with finite outputs and parameter changes; unchanged Eq15
residual/Jacobian and exact-control preservation. Fit nomination additionally
requires stationarity without amplitude-collapse/domain/underflow vetoes.
Promotion vetoes: wrong target, unrecorded changes, nonfinite arithmetic,
unexecuted endpoint, collapsed shape, stale evidence or a missing required check.
Continuation vetoes: corrupt inputs, wrong source identity, unsafe output paths,
failed process cleanup or exhausted budget.
Repair triggers: API incompatibility, local solver nonconvergence and other
localized failures; independent checks may continue.
Explanatory only: fixture losses/timing/memory, objective trajectories and counts
of old clouds admitting a candidate. No learned quality, statistical ranking,
unbiased-gradient claim, full replication or production promotion follows.
Heuristic comparisons remain mandatory in the full experiments; they cannot
assess the short wiring and solver diagnostics.

## Default and assumption audit

| Choice | Provenance and role | Failure mode / earliest check |
| --- | --- | --- |
| Author PyTorch/NumPy GPU | Existing independent-reference exception, local venv | Probe this actual interpreter before training; record device/version |
| Small source data / short epochs | Diagnostic convenience, source T=17/N=25/batch sizes retained | Trace real loaders/updates, record overrides; no quality inference |
| One persistent loader worker | Bounded resource choice; zero conflicts with persistent_workers=True | Actual iteration must work; throughput remains preliminary |
| Seeds 924011–924013 | Reproducibility choices, not recovered paper seeds | Record data/model/sampling seeds and hashes |
| Scheduler/deserialization compatibility | Version adaptation only | Preserve exact patch and test scheduling and data/weight round trips |
| Damped Gauss–Newton | Local solver hypothesis, damping alters the step not the objective | Analytic/centered-difference Jacobian and no movement on exact fits |
| Solver thresholds/damping | Numerical reconstruction, not recovered author defaults | Record trust-model agreement, rejected steps, gradients, relative residual and scale drift; scalar-loss-only success forbidden |
| Full-run cost | Small source runs provide component timing | Include generation, pretraining, repeats, baselines/evaluation/retries; leave unknown components unbudgeted |

## Execution

Use the existing approved absolute interpreter and explicit numerical checkout.
GPU jobs require escalation. CPU-reference children hide GPUs before imports.
Run focused unittest checks first. Select jobs explicitly with --jobs; invalidated
historical smokes must not consume the remaining launches just to repeat unrelated
checks. Resume the same ledger with --repair-note. Record source snapshots,
dependency identities, actual commands, wall time and remaining budget.
The terminal review must distinguish executed phases from planned/unimplemented
ones and distinguish successful short training from experimental reproduction.

## Implementation review before launch

Eleven focused dispatcher/configuration tests passed in development-review07
in the numerical campaign. Python syntax and R parsing passed. Tests cover
budget/timeout cleanup, unchanged dependency reuse, source/environment
invalidation, legacy PASS classification, independent failure continuation and
YAML integer-dimension restoration across JSON. No scientific result follows
from these tests.

The training diagnostic calls source setup_training -> problem/dataset ->
trainer -> optimizers/loss -> train/validation -> checkpoint. It records actual
calls and compares generated/reloaded datasets. Narrow compatibility patches
are applied only to an isolated source copy. Original data/checkpoints remain
untouched. The exact/full Gaussian controls and analytical Jacobian are included
in the iAPF numerical launch, not presumed passed from parsing.

The local solver nomination thresholds (scaled gradient <=1e-8 and profiled
relative residual <=1e-6) are conservative diagnostic hypotheses, not the paper's
fit tolerances. Their failure does not invalidate the method or prove a local
minimum cannot exist. Initial damping 1e-3 is a trial step regularization:
trust-model agreement and accepted/rejected steps are preserved for evaluating
it, with exact/near Gaussian controls as the non-harm checks. No setting is
eligible as a runtime default from this diagnostic.

Decision: execute only mdpf_source_training and iapf_eq15_local_solver under
the remaining three-launch budget. Preserve the final launch for a localized
repair. Full learning experiments and paper replication remain unbudgeted.

## Attempt 10–11 findings and reserved repair

Attempt 10 passed its actual-interpreter CUDA check but failed before training:
Matplotlib required CXXABI_1.3.15 after Torch loaded the older system libstdc++.
Use only the reference child's base-environment lib directory first in
LD_LIBRARY_PATH; bind library hashes and record the effective child environment.
Shared environments and source remain unchanged.

The focused CPU regression then exposed the source scheduler's old is_better
method name. Its installed equivalent is _is_better. The isolated source patch
now adapts that name as well as dropping the removed logging-only verbose
argument. Twelve harness tests, imports, both model constructors and scheduler
cadence/state parity across min/max and relative/absolute modes pass in
development-review09. These are CPU-only compatibility checks, not training.

Attempt 11 passed unchanged Eq15 value/gradient/Jacobian and exact/near-Gaussian
controls. Six of 32 stored failure records satisfy the nomination screen;
26 hit amplitude collapse. These are stored records, not 32 independent trials.
The solver remains a reconstruction hypothesis and is not ready for a full
filter. The iAPF direction remains open; the current candidate is insufficient.

Decision: use the one remaining launch for mdpf_source_training with the
localized loader/scheduler repairs, unchanged diagnostic design and total budget.
No additional numerical launch is allowed afterward in this campaign.
