# Restore the owner's 70% HMC acceptance target

Owner instruction, 19 October 2026: target 70% acceptance. The completed
range-bearing run targeted95%, selected in the 18 October extension as a
conservative response to its former energy-error stop. The saved nomination
manifest confirms0.95, and TensorFlow Probability dual averaging receives that
value directly. Resumption retained the frozen kernel. Consequently94% observed
acceptance is consistent with the configuration; it does not show that the
adaptation failed. The choice of target was wrong for the owner's requested
tuning policy. The energy-stop repair should have separately identified this
remaining efficiency issue.

Evidence contract: restore0.70 for fresh range-bearing commands and test the
actual queue command builder and saved-state resume commands. Comparator:
the pre-change command explicitly requested0.95. Pass criterion: new fresh
range-bearing commands request0.70; resumed kernels retain their saved settings
and prohibit contradictory retuning flags. No target, numerical HMC kernel,
particle design, energy-warning policy, or convergence criterion changes.
This local controller repair needs no new GPU experiment. Acceptance and ESS
remain descriptive; no efficiency ranking follows from the target change.

Default audit:0.70 comes directly from the owner and agrees with the existing
five-model make_tuning_config default. It is a tuning objective, not a guaranteed
realized rate or a new convergence requirement. The initial step seed0.005
remains unchanged in this narrow correction; dual averaging adapts it for each
candidate. Fresh adaptation remains necessary to obtain a new frozen kernel.
The completed95%-target samples are preserved with their actual configuration;
they cannot be relabeled as70%-target results.

Skeptical pre-edit audit: PASS. The mismatch is in the range-specific launch
override, not a failure to pass the target into dual averaging. The reusable
five-model default and the active SIR configuration already request0.70.
The running supervisor has loaded the old source and has already passed range
bearing; this edit therefore affects future fresh launches only. Archive the
pre-edit source because the original campaign manifest pins it. Do not rewrite
that manifest or edit the active SIR settings. No active health failure or
scientific reason requires interrupting SIR.

Execution scope: one command-value change, its focused wiring regression, and
the queue tests. Use CPU-only tests with CUDA_VISIBLE_DEVICES=-1. No new GPU
attempt is launched: SIR currently occupies the last of the six authorized
attempts. A fresh range-bearing tuning run is subsequent work, not part of the
active queue; the original queue records range bearing as sampling-complete.
The existing campaign deadline remains20 October16:00:41 Hong Kong.

Artifacts: docs/plans/artifacts/surrogate-hmc-acceptance-target-repair-20261019-01.
Preserve source-before, test log, change patch and a result note. Diagnostic
vetoes are command/config mismatch or any changed resume semantics. Stop the
edit if a live numerical consumer would be changed. No new stochastic method
comparison or default-readiness claim is made.
