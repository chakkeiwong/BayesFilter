# Native validation selection: completed diagnostic, open trained-quality question

13 October 2026. [Plan](controlled-smc-kdm-native-selection-plan-20261013.md).
[Master](controlled-smc-kdm-replication-master-program-20260928.md).

The full native validation experiment completed. The two saved TF-trained MDPF
checkpoints remain descriptively worse under the unchanged author evaluator.
Ordinary reevaluation uncertainty on these checkpoints is much smaller than the
observed gap. Nearby checkpoints exchange selection order; this does not
reconstruct historical selection or explain the large final-checkpoint gap.

## Execution and checked target

Command: /home/chakwong/anaconda3/envs/tftwogpu/bin/python -I -B docs/benchmarks/run_kdm_native_selection.py.
Attempt01 completed in 354.036491439 CPU seconds, zero GPU seconds, under the
1800 s attempt cap. GPU was deliberately hidden before framework import.
The original PyTorch author reference is an explicit independent CPU exception,
not a BayesFilter implementation or speed measurement.

The code executes AST-extracted, unchanged numerical bodies of the pinned author
Trainer.do_training_epoch, Trainer.do_validation_epoch and
FullSequenceTrainer.do_forward_pass. The native batch-weighted validation NLL
equals the independently accumulated calls to that same source forward method
exactly under restored RNG state; terminal RNG hashes also match. Actual
consumer call chains were exercised, not merely inspected.

There are 96 full validation evaluations, each with 1000 rows, T=17, N=25 and native
batches 256/256/256/232. The worker passed 2638 validity checks; the independent
terminal audit passed 479 checks. It verifies source/input/snapshot
hashes, raw batch coverage and weighting, finite losses, checkpoint preservation,
ledger charge, and saved-window identities. No failed or missing run is hidden.

The two source training epochs restart from the complete unit01 TF-best state.
Both process 5000 rows in 79 batches, with final batch 8. Adam steps advance from
21646 to 21725 and 21804; moments and saved learning rates are retained. Complete
intermediate states and batch histories are saved. Rates stay fixed for this
bounded counterfactual; it is not native completed training or a recreation of
the historical optimization/selection path.

## Native validation findings

Eight repetitions are conditional evaluations of each fixed checkpoint, not
eight independent training units. All differences below are descriptive.

| Unit | RNG arm | Author NLL mean | TF NLL mean | TF minus author | SD of repeated difference |
|---|---|---:|---:|---:|---:|
|01|Independent initial states|5.89918|6.66014|+0.76096|0.05002|
|01|Shared initial state|5.88927|6.64741|+0.75814|0.03288|
|02|Independent initial states|6.19384|6.74730|+0.55346|0.05346|
|02|Shared initial state|6.17626|6.73541|+0.55915|0.06160|

Individual-checkpoint NLL SDs range 0.02966–0.05304. This directly measures the
native 1000-row validation uncertainty; no scaling from the old 64-row panel is
used. RMSE and 150-step heldout test quality are outside this experiment.

In the unit01 nearby window, the independent-draw mean NLLs are 6.66014,
6.65589 and 6.65213 for the starting checkpoint and the next two epochs.
Across eight draws, they win selection 2,2 and 4 times. With shared initial RNG,
the counts are 3,2 and 3. The heldout four-draw evaluation of choices from the
first four draws gives observed regret 0–0.02012 nat for independent initial
states and 0–0.00605 nat for shared initial state. These tiny panels do not
establish that seed sharing improves selection. Their role is to show that
selection among nearby states is noisy on the actual native validation size.

The master previously called for common draws without accounting for the
native rejection sampler. Initial RNG states really match, but 11 of 16
unit/repetition groups end at different RNG states across checkpoints.
Even identical terminal RNG state would not prove identical intermediate
draws. The repaired description is shared initial RNG state, not common
realized base variates.

## What remains unresolved

The saved checkpoint gaps persist after replacing the reduced panel with full
native validation. This weakens ordinary evaluation noise as the sole
explanation. It does not distinguish optimization variability, accumulated
floating-point and resampling sensitivity, or historical checkpoint selection.
Two low-rate epochs near a saved checkpoint cannot reconstruct hundreds of
earlier epochs. The current evidence does not identify another translation
operation that should be changed.

The earlier gradient follow-up separately localized the selected arithmetic
flags: pretraining drift clears with full author state, and matching all raw
angle pairs removes 98.52% of the selected trained gradient discrepancy without
changing ancestors. These local successes cannot certify learned quality.
There is no evidence here that the paper is fundamentally wrong.

## Independent-unit precision feasibility

The accepted criteria remain 0.01 nat NLL and 1% RMSE. Recomputed from all three
existing development pairs, the original illustrative80%-power calculation
at true equality, with 95% Bonferroni coverage across four endpoints, is:

| Variant | Paired NLL SD | Illustrative NLL pairs | Illustrative RMSE pairs |
|---|---:|---:|---:|
|MDPF|0.60808|35,147|2,991|
|A-MDPF|0.13427|1,714|53|

These are unstable three-pair, normal-approximation planning illustrations,
not validated requirements or a proof that all possible designs are impossible.
The current MDPF mean difference is already+0.46245 nat; a calculation at true
equality is optimistic about the scientific effect as well as about precision.

Actual source/TF full-pair CPU costs were 11.24 and 8.43 hours; TF GPU costs were
2.44 and 1.85 hours. With60.26 CPU hours and 24.54 active GPU hours remaining,
these observed costs allow roughly5–7 fresh complete pairs before evaluation
overhead. That budget does not justify the present qualification design.
At seven pairs, the illustrative target NLL SD for 80% power is about 0.00858 nat,
far below either observed paired SD.

No new full ladder was launched. A future affordable design must demonstrate
substantial variance reduction at the independent-training-unit level, or the
owner must explicitly change the qualification scope. More evaluations of the
same checkpoints do not solve this problem. No margin was relaxed.

## Decisions and inference

| Decision | Primary criterion status | Veto diagnostics | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
|Close native selection diagnostic|Native endpoint, full row coverage and window completed|All validity checks pass|Historical selection remains unavailable|Preserve result and freeze this phase|Noise explains the final gap|
|Keep KDM trained quality open|0.01 nat/1% noninferiority unestablished|No invalid run; unfavorable finite checkpoint outcomes|Independent-unit variability and training divergence|Require an affordable precision design before full qualification|Replication complete or population ranking|
|Preserve implementation|Selected gradient flags localized; no new source deviation found|No numerical repair triggered here|Untested full optimization trajectories|Change code only after a source-grounded mismatch|All training behavior is equivalent|
|Preserve cSMC bounded verdict|Unchanged tested-input and failure fidelity|Author-intrinsic failures remain acceptable|Population continuous-outcome inference|No rerun under withdrawn 1%/5% rule|Paper-wide or population certification|

| Inference status | Result |
|---|---|
|Hard veto screen|No execution, source-identity, finite-value, batching or artifact failure|
|Statistically supported ranking|No population training-method ranking established|
|Descriptive-only differences|Native NLL gaps, noise SDs, nearby selection counts and regrets|
|Default-readiness|Unchanged; no new numerical/runtime default|
|Next evidence needed|A demonstrated affordable independent-training-unit precision design, or a newly localized source deviation and its repair|

Post-run red team: the strongest alternative explanation is divergence between
valid stochastic training trajectories, possibly compounded by historical
selection. The short window cannot distinguish that explanation from a
remaining long-horizon implementation problem. The weakest evidence is still
the three independent training pairs. A well-powered untouched comparison, or
a matched-state source deviation that explains later quality, could overturn
the present interpretation.

## Preserved evidence

Phase root:
artifacts/controlled-smc-kdm-replication-20260928/native-selection-20261013.
attempt01/diagnostic contains raw evaluations, all intermediate states, batch
histories, controls and input/source hashes. The phase root contains
terminal-audit.json and precision-feasibility.json. The run manifest, command,
environment and snapshots are under attempt01. Full logs are preserved.
The phase is CLOSED; no experiment remains running.

## Additional 48-hour budget assessment

The owner's conditional offer was evaluated as 48 additional CPU hours and
48 additional GPU hours. At observed costs, that adds 4–5 complete training
pairs. Including the existing balance of 60.26 CPU and 24.54 GPU hours gives
9–12 fresh pairs, before evaluation overhead. Under the existing planning
approximation, even 12 pairs would require an NLL paired SD near 0.01124 nat
at true equality for 80% power; the three-pair estimates are 0.60808 and
0.13427 nat. These estimates remain uncertain, but they do not justify
expecting this extra budget to complete qualification. A reliable completion
budget cannot yet be quoted. Additional repetitions also do not repair an
actual difference in trained quality.

The calculation is preserved in
artifacts/controlled-smc-kdm-replication-20260928/native-selection-20261013/budget-scenario-plus48-20261013.json.
No conditional renewal was added to the ledger and no new training run was
launched. Existing compute remains available for a discriminating diagnostic
once its design identifies evidence that would change the next action.
