# Executable replication master: execution and terminal review

The Stage 0 dispatcher is executable and its bounded diagnostic tranche finished.
Both scoped paper replications remain **INCOMPLETE**. Stage 0 itself remains
**INCOMPLETE** because fit reconstruction, complete training wiring and reliable
full-experiment cost are unresolved.

Code: [dispatcher](../../../../scripts/run_replication_master.py),
[diagnostic worker](../../../benchmarks/diagnose_replication_stage0.py),
[focused tests](../../../../tests/test_replication_master.py).
Authority and pre-run contract live in the planning checkout:
[master](/home/chakwong/BayesFilter/docs/plans/iapf-kdm-replication-master-program-2026-09-24.md),
[review/contract](/home/chakwong/BayesFilter/docs/plans/artifacts/iapf-kdm-replication-master-20260924-01/executable-review-and-contract.md).
All numerical commands used the explicit numerical checkout above and escalated
permissions. Deliberate CPU children hid GPUs before framework import.

## What was executed

Nine numerical/reference diagnostic launches, including environment repairs,
consumed **17.157 CPU-worker seconds** and
**5.891 GPU-worker seconds**. Remaining:
**1782.843 CPU-worker seconds**,
**894.109 GPU-worker seconds**, and **3 / 12
launches**. Each child was limited to 300 seconds; runs were serial with one
compute thread. Source reading, configuration inspection, environment preparation
and routine development tests are recorded separately from numerical campaigns.

The supervisor returns exit code **2** when prerequisite checks remain incomplete.
The children completed; that return code is an incomplete-stage signal,
not an execution hang. See [machine result](result.json),
[attempt/budget ledger](ledger.json) and versioned manifests/source snapshots.

| Check | Evidence | Finding |
| --- | --- | --- |
| iAPF Equation (15) algebra/gradients/mechanics | [attempt01](stage0/attempt01/command.log) | PASS for the reference identities and short controller fixture |
| Saved original-objective failures | [attempt02](stage0/attempt02/outcome.json) | All 32 RDS failures preserved: 25 optimizer nonconvergence, 5 floating-point domain boundary, 2 density underflow |
| Original MDPF configuration | [attempt09](stage0/attempt09/resolved-configuration.json) | Correct KDEParticleFilter/ImportanceSampling; 55 generated experiments across 11 trials; no active training; 14 distinct absent checkpoint paths |
| Original author import/factory | [attempt09](stage0/attempt09/result.json) | Original main and KDE modules import; factory constructs KDEParticleFilter |
| Source sequence/loss/backward endpoint | [executed calls](stage0/attempt09/source-sequence-wiring.json) | PASS on diagnostic B=2,T=9,N=25, Normal/Normal/Von-Mises, labels/BPTT every four steps; 31 parameter tensors have finite gradients |
| TensorFlow GPU/XLA environment | [attempt04](stage0/attempt04/result.json) | PASS: fixed signature, one trace, CPU identity, saved HLO, verified memory growth on both visible GPUs |
| PyTorch reference GPU environment | [attempt07](stage0/attempt07/result.json) | Existing tf-gpu2 PyTorch 2.11.0+cu130 runs on RTX 4080 SUPER |

The sequence fixture calls the original FullSequenceTrainer.do_forward_pass,
model forward/resampling/density and original loss. It deliberately bypasses the
trainer constructor and dataset loaders, uses random weights, and performs no
optimizer update. Finite nonzero gradients establish executable wiring only;
they do not establish correct or unbiased gradient estimation, learned quality,
training reproduction or the complete training entry. Seeds: model 924001,
sampling 924002. Its loss and timing are diagnostic values, not comparisons.
The source trace and executed source-file hashes are preserved.

## Problems found and repairs

The old Equation (15) supervisor has a fixed expired September 21 deadline and
historical output root. It is unsuitable as the new entry point. The new runner
uses unique output, cumulative CPU/GPU/attempt accounting, versioned source
snapshots, process-group cleanup and explicit incomplete verdicts. It implements
Stage 0 only; later stages are rejected as unimplemented and unbudgeted.
No approval-token system was added.

The saved 32 iAPF failures already existed; this execution read them and did not
repeat the failed full-filter campaign. Their equation-level checks passing
does not repair the original fit. The positive floor/solver/source choices and
full adaptive learned runs remain unresolved. The failure is in the current
fit candidate; it does not reject iAPF as a research direction.

An inherited PYTHONPATH caused Python to select MacroFinance.models instead of
the author's namespace package. The reference now uses isolated Python (-I),
disables bytecode writes (-B), and checks the actual package path before import.
The old failed attempts are retained.

The TensorFlow GPU environment contains CPU-only PyTorch 2.11.0+cpu. The existing
tf-gpu2 environment provides working PyTorch CUDA. Four missing utility packages
were installed into a new project-local virtual environment based on tf-gpu2:
torchviz 0.0.3, graphviz 0.21, nvidia-ml-py 13.610.43, prettytable 3.18.0. A dry-run
allowed only these additions, and installation disabled dependency changes.
Shared environments and the numeric stack were unchanged. Exact resolver,
installation and command reports are in [environment-setup](environment-setup/commands.json).
Reference interpreter:
/home/chakwong/BayesFilter/.localresources/envs/mdpf-reference-20260924/bin/python.
The CUDA probe was run on its underlying tf-gpu2 interpreter; the final local
environment's source fixture was explicitly CPU-only.

The author source requires weight pretraining (up to 400 epochs), then full
training (up to 800 epochs) before evaluation. Required cached datasets and
checkpoints are absent. The generator's 11 trials and actual merge/override
semantics were executed into new diagnostic files; the source's disabled training
flags and original paths were preserved. These inspection files are not training
launch files. Future training needs reviewed activation and all outputs,
datasets and checkpoints relocated to a new experiment directory.

A working small XLA kernel is evidence that basic CUDA/XLA works. It cannot
explain or clear the previous large scientific graph problems, prove that the
methods use stable graphs, or replace their future full-sequence checks.
Python host loops in this dispatcher and the independent author reference are
not evidence of retracing in a BayesFilter TensorFlow kernel.

## Decision and inference

| Decision | Primary criterion | Veto status | Main uncertainty | Next justified action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Use this dispatcher for bounded Stage 0 work | Focused execution tests and recorded launches pass | No overwrite, budget bypass or dangling timed-out child observed | Later phases unimplemented | Preserve this ledger for remaining repairs | Full replication controller exists for all stages |
| Continue iAPF fit reconstruction | Algebra/mechanics pass; original full-filter success absent | Candidate promotion blocked by original fit failures | Unspecified floor/solver behavior | Formulate the next target-preserving I2 repair from saved failures | iAPF direction rejected or replicated |
| Continue MDPF source training closure | Configuration/import/factory and narrow sequence wiring pass | Pretraining/data/full trainer/optimizer evidence absent | Training API compatibility, full-run costs and experiment seeds | Test source trainer/optimizer compatibility on an explicit tiny dataset fixture, then cost the original protocol | Learned MDPF or Appendix B reproduced |
| Defer full campaigns and integration | No full experimental evidence/cost yet | Full budgets NOT_BUDGETED | Pretraining, retries, comparator and repeat costs | Complete prerequisites, then present one measured full budget | Permission to shrink or skip paper protocols |

| Inference item | Status |
| --- | --- |
| Hard veto screen | Saved iAPF fit failures; missing MDPF trained artifacts and full-training evidence block replication promotion |
| Statistically supported ranking | None; no scientific comparison launched |
| Descriptive-only differences | Tiny compile/runtime and fixture loss/gradient values |
| Default readiness | Not established; no numerical default changed |
| Next evidence needed | Original fit reconstruction, exact learning protocol closure, full reference experiments with uncertainty, then eligible TensorFlow GPU/XLA consumer conformance |

## Terminal skeptical review

The strongest alternative explanation for future MDPF failure is unfinished
environment/trainer/data preparation or absent pretraining, not failure of IWSG.
For iAPF, the source's unspecified numerical choices and objective degeneracy
remain stronger explanations than a general algorithm rejection. The narrow
diagnostic passing is not a proxy for experimental reproduction. The weakest
evidence is the untested full trainer/optimizer call chain and absent full-scale
cost, which this result explicitly leaves open.

Seven focused tests cover prior-output preservation, timeout descendant cleanup,
cumulative charging/recovery, attempt/resource refusal, wrong discrete-method
rejection, unbudgeted stage refusal, and exit-zero/incomplete-science separation.
Final code/source hashes, configuration values, failure hashes and scoped
whitespace checks are recorded in final-validation.json. This is a self-review;
no independent reviewer was launched.

## Using the executable

Working directory:
/home/chakwong/BayesFilter/.localresources/worktrees/ledh-iapf-execution-20260922

Read-only status command:
    /home/chakwong/anaconda3/envs/tftwogpu/bin/python -I -B scripts/run_replication_master.py status --output docs/plans/artifacts/iapf-kdm-replication-20260924-01

Plan without experiments:
    /home/chakwong/anaconda3/envs/tftwogpu/bin/python -I -B scripts/run_replication_master.py plan

After a reviewed local repair, resume this same ledger with run, --resume,
--retry-incomplete, --repair-note and --evidence-root /home/chakwong/BayesFilter.
Do not create a fresh campaign merely to reset the remaining budget.
