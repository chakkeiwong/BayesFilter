# Acceptance target correction

The completed range-bearing run targeted95%, as its saved nomination manifest
confirms. The18 October queue explicitly supplied --hmc-target-accept0.95;
the five-model worker passed it into make_tuning_config, and the public tuner
passed config.target_accept_prob into TFP DualAveragingStepSizeAdaptation.
Resumption preserved that frozen kernel. Actual chain acceptance93.8–94.9%
is consistent with the configured target. The former energy-error stop motivated
the conservative override; its later removal did not revise that override.

The owner now specifies70%. The range-specific fresh-launch command has been
changed to0.70, matching the existing five-model default. The initial step-size
seed0.005 is preserved and remains subject to adaptation. The completed samples
and their95% configuration are preserved. The live SIR run already has
target_accept_prob0.70; its current kernel was not modified.

Validation:24 focused CPU-only tests passed in4.14 seconds. These exercise the
actual queue command construction, saved-state resume commands, restart guards,
budget handling and the public configuration. The modified command regression
asserts0.70 for fresh range tuning. No new GPU tuning or sampling was performed.
The pre-edit source and checksums are saved locally because the active historical
campaign manifest records the previous supervisor source hash.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Correct fresh target to70% | Queue passes requested target;24 tests pass | No command/resume regression | Realized acceptance and efficiency after new adaptation | Keep active70%-target SIR running; fresh range retuning is later work | Existing range result was tuned to70%, or70% is demonstrably faster |

| Inference status | Finding |
|---|---|
| Hard veto screen | No new sampling evidence; existing range health checks passed |
| Statistically supported ranking | None |
| Descriptive-only differences | Saved range acceptance93.8–94.9% versus configured95% |
| Default-readiness | Owner-directed tuning configuration corrected; performance not evaluated |
| Next evidence needed | Fresh70%-target adaptation and subsequent sampler diagnostics |

Post-edit review: the mismatch was a deliberate configuration override, not a
broken dual-averaging argument. Target acceptance is a tuning objective; finite
adaptation and changing explored regions need not give exactly that realized
acceptance. The95%-target samples still satisfy the declared sampling rule.
Acceptance alone neither proves efficient exploration nor invalidates those
samples. No efficiency ranking is established by this configuration change.

The current master is executing SIR on the sixth and final allowed attempt.
A new range-bearing70%-target run is not in that queue and has not been launched;
it would require a fresh tuning phase and an additional attempt allocation.
The live run and existing20 October16:00:41 Hong Kong deadline are unchanged.
