# Surrogate-HMC master-program checkpoint — 2026-09-18

Superseded by the [active execution checkpoint](../surrogate-hmc-all-models-20260919-review-tests/checkpoint.md).
The remaining text is preserved authoring history, not current launch status.

ACTIVE UPDATE 2026-09-19: owner authorized revision and execution, including
63 additional GPU hours / 15 CPU-worker hours and LG1 inference attempts 4,5.
Opus withdrew F1. Master revised for separate frozen bindings, precision and
proper synthetic-prior assumptions. Current execution note:
../../surrogate-hmc-all-models-execution-2026-09-19.md.
No new numerical job has run yet. Next: implement driver + full LG1 consumer,
test GPU/XLA binding, replay, row independence and measured transition cost.
Material source finding: TensorFlow public tuner is diagnostic-only and rejects
XLA; legacy admitted tuner uses NumPy. Resolve before retained inference.
Budget remains unchanged. This update supersedes the old permission/status
statements in the preserved authoring checkpoint below.

Historical authoring request: assess the Opus verdict on the completed master program.
Assessment complete: agree with the design endorsement and useful clarifications;
reject F1's force/endpoint input-identity requirement. Separate frozen force and
endpoint bindings are mathematically valid. Derivation and source checks:
../../surrogate-hmc-all-models-opus-response-2026-09-18.md.
The goal remains testing surrogate HMC with an imperfect full-DCTR force.
The master and Opus verdict were preserved; no experiment/reviewer was launched.

Checkout: /home/chakwong/BayesFilter-ledh-graceful-failure-execution-20260917;
branch ledh-graceful-failure-execution-20260917; HEAD 8a4cffa2. Preserve the
pre-existing execution-result edit, LG1 audit files, and recovered documents.

Stage: authoring and local skeptical review complete. Deliverables:
../../surrogate-hmc-all-models-master-program-2026-09-18.md,
../../surrogate-hmc-all-models-master-review-2026-09-18.md, and
../../surrogate-hmc-all-models-claude-handoff-2026-09-18.md.
The master is a research/implementation program; the proposed all-model runner
does not yet exist. Claude has not been launched or contacted.

Checked: recovery evidence; public tuning interface/capability; shared HMC
proposal, transition and typed status/force APIs; LG1 fixture; model registries
and additional range-bearing, quadratic structural, banana and mixture sources.
Repaired: target/reference distinctions, four direct posterior comparisons,
support/tail bounds, force semantics, row independence, predictive comparability,
LG1 precision/prior/initialization, and explicit cost/attempt limits. The minimum
register has M01-M14 and C01-C03; source-based closure is still an execution task.
No full-DCTR consumer call chain or all-model inference result is certified.

Evidence: ../../ledh-surrogate-hmc-recovery-result-2026-09-18.md and
../../ledh-surrogate-hmc-model-coverage-2026-09-18.md. Conservative unspent
containment accounting: 63.5687 GPU hours / 15.1016 CPU-worker hours, with the
three LG1 attempts exhausted. These figures do not authorize new attempts.
No additional compute or attempt budget has been consumed by this planning.

The proposed future cap is 63 GPU-hours / 15 CPU-worker-hours and two additional
LG1 inference attempts, explicitly subject to measured feasibility and a future
execution instruction. Current document work consumed no numerical allocation.

Documentation checks passed: 56 local links, 16 Markdown tables, 13 source
symbols, 17 minimum-register IDs and allocation arithmetic against the saved
accounting. Results are in validation.json; they are not numerical evidence.
Next action: when revising or executing the program, use the response's
independent endpoint/force binding checks, not Opus F1's equality constraint.
Useful precision reporting and prior/budget clarifications are identified.
Future execution still begins with P0/P1 consumer implementation and scope
closure before LG1 inference. Do not replace inference tests with force accuracy.
