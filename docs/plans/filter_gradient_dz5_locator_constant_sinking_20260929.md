# DZ5 locator constant-sinking intervention

Adaptive iAPF is deferred by the owner. The next question is whether XLA's
while-loop constant-sinking pass affects the specific floating-point difference
identified in the frozen historical DZ5 locator. This is a CPU reference
diagnostic within the existing repair campaign, not a production flag change.

The exact controls are original04656 and candidate04657, whose complete
one-iteration callbacks and results reproduce04635/04636. The prior inspection
found four candidate gradient fusions with two loop-carried copies of constant
2.4, versus embedded constants in the original. See
`filter_gradient_dz5_locator_optimized_hlo_result_20260928.md` and installed
TensorFlow `include/xla/service/while_loop_constant_sinking.h:25-58`. The header
defines the transformation and its pass name; it does not prove the pass caused
the observed difference.

Run the same frozen original and candidate twice each: normal compiler settings
and the isolated diagnostic flag
`--xla_disable_hlo_passes=while-loop-constant-sinking`. Set and record the flag
before TensorFlow import in each isolated child. Keep source, input, resource
dtypes, dispatch, precision, random streams and objective unchanged. Reuse the
existing one-iteration harness and optimized-HLO capture. Do not edit live
MacroFinance, runtime implementations, defaults or policy allowances.

Both normal-setting controls must reproduce all saved callback bytes and the
complete short records exactly before interpreting either intervention. Check
snapshot/dispatch/source identities, device/environment, one trace and absence
of host callbacks. Record every intervention's full results and callbacks,
including any changed validity, callback count, failure or convergence fields.
Compare all records and first identical-input scores without a tolerance waiver.
Parse exported optimized HLO with the tested reader and count the previously
identified214/217-instruction fusion signatures; retain structural differences
and hashes. Export overhead remains diagnostic and ineligible for cost claims.

The primary diagnostic is whether disabling this named pass changes the
identified fusions and the original/candidate numerical difference together.
The strongest competing explanation is another affected loop or compiler pass:
this disables a pass module-wide and is not a surgical edit of four fusions.
Therefore even joint changes support only pass/context sensitivity. If the
fusions do not change, or numerics move independently, reject this intervention
as an isolation of the proposed mechanism. Use the result to choose a smaller
source-grounded intervention; do not try unrelated flags or counter dtypes.
No outcome alone qualifies a runtime remedy or optimizer convergence.

Allocate at most6 serialized workers and4200 CPU process-seconds, zero GPU:
four900-second export workers (870-second child deadlines), one300-second
combined saved-evidence/reader/policy check, and one300-second localized
harness repair if necessary. These maxima fit inside the existing remaining
24.999432 CPU /24.770371 GPU process-hours and do not expand the56/52-hour cap.
Use groups `dz5_locator_sinking_{original,candidate}_{control,disabled}_cpu`
and `dz5_locator_sinking_readback_cpu` with the stable campaign runner. All
outputs go to new run directories under the existing raw artifact root.

Stop this unit on control/source drift, unsupported compiler/reader behavior,
lost evidence, timeout or its budget limit. Preserve failures and charge every
attempt. A negative causal result closes the experiment, not the locator gap.
Full trajectories require a separately justified portable remedy and unchanged
numerical/status gates. Both prior full optimizers remain unconverged.

Skeptical pre-execution review by the primary agent: the baseline is historical
and intentionally reproduces the existing discrepancy; it cannot admit the
current source or historical derivative machinery. Exact fresh controls protect
against stale environment/context. The narrow named-pass intervention answers
a compiler-mechanism question, with explicit module-wide and independently
exported-IR limitations. No independent reviewer is used. Other numerical,
memory/performance, consumer-audit and integration gaps remain open.
