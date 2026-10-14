# KDM source-training variability result — 8 October 2026

Phase COMPLETED. Both replications remain incomplete.

Plan: [source-only calibration](controlled-smc-kdm-source-variability-20261008.md); [supervisor recovery](controlled-smc-kdm-source-variability-recovery-20261008.md).
Summary: [structured result](artifacts/controlled-smc-kdm-replication-20260928/kdm-source-variability-20261008-01/summary.json).

The reference uses unchanged author training, two predeclared new seed units,
the same training/validation data and the same development author evaluator.
Both new units passed constructor/data checks, the independent terminal audit
and exact replay of the fixed TF checkpoint evaluation. Audit counts were
88516 and 80186, with zero failures; each evaluation passed all 16 fixed-TF
replay controls.

| Source unit | Stage seed base | Variant | NLL | RMSE |
|---|---:|---|---:|---:|
| original | 202610072 | mdpf | 6.41509175 | 3.45501196 |
| unit01 | 202610101 | mdpf | 6.14145184 | 3.25393462 |
| unit02 | 202610111 | mdpf | 6.44162869 | 3.30710316 |
| original | 202610072 | am_dpf | 6.02694178 | 3.38561046 |
| unit01 | 202610101 | am_dpf | 5.80121613 | 3.2093333 |
| unit02 | 202610111 | am_dpf | 5.87295341 | 3.29723859 |

Each value averages two filter seeds within one trained unit.
The independently trained sample size is three per variant. Conditional
metrics and per-step values remain in the linked unit evaluations.

| Variant / metric | Source SD | Planning units per arm |
|---|---:|---:|
| mdpf/nll | 0.166177 | 3415 |
| mdpf/log_rmse | 0.0310171 | 121 |
| am_dpf/nll | 0.115333 | 1645 |
| am_dpf/log_rmse | 0.026736 | 90 |

The sample-size figures use the predeclared equal-variance normal
approximation and are uncertain planning estimates, not validated power
calculations. TF training variance and paired covariance remain unknown.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
|---|---|---|---|---|---|
| Calibration complete | Two full audited source units evaluated | Float32 screen remains open | Training-level precision and TF variability | Localize accumulated TF training drift with saved checkpoints and matched inputs, preserving failed float32 screens; assess paired-comparison variance before proposing untouched qualification. The present unpaired precision estimate exceeds the remaining budget; controlled-SMC fitting remains open. | Replication or noninferiority |

| Inference class | Status |
|---|---|
| Hard veto screen | Preserve all failed numerical checks; see per-unit validity/audit status |
| Statistically supported ranking | None |
| Descriptive-only differences | Source training variation and fixed TF comparison |
| Default-readiness | Not established |
| Next evidence needed | Reviewed untouched trained-unit qualification and numerical accuracy checks |

Post-run review: the strongest alternative explanation is development-data-specific
or arithmetic-dependent variability. Three source units provide an unstable
variance estimate; more independent training units or new data could change it.
This phase does not explain the existing TF gap causally or repair the author.

Charged CPU seconds: 55948.498548287 (summed child wall time plus 300 routine seconds). GPU charge: zero.
Exact environments, seeds, code hashes, commands and wall times are in the manifests.
The live checkpoint carries the remaining allocation.

Supervision recovery: the original supervisor exited 143. Unit 1's complete
artifacts were independently audited; its process exit code remains unobserved.
Its CPU charge conservatively includes launch to first verified process absence,
so it overstates measured training time. Original manifests and helper edits are
preserved in recovery-20261008-01. No training was repeated or seed replaced.
Recovery plan: [bounded recovery](controlled-smc-kdm-source-variability-recovery-20261008.md).

Terminal review: [verification and descriptive comparisons](artifacts/controlled-smc-kdm-replication-20260928/kdm-source-variability-20261008-01/terminal-review.json).

Averaging the two filter seeds within each trained unit, the fixed TF checkpoints
have higher NLL than all three source checkpoints, by 0.300–0.600 nat, and higher
RMSE by 2.79–9.73%. These are descriptive comparisons of one TF training unit
with three source units, not a statistically established population ranking.
The additional source runs therefore do not resolve the development quality gap.

The predeclared unpaired planning approximation suggests 1645–3415 units per arm
for the 0.01-nat NLL margin. With only three source units and unknown TF variance,
these are unstable estimates, not a validated sample-size requirement. They do
show that this unpaired design has no current precision justification within
the remaining compute. The next work is matched-input training-drift localization
and assessment of paired variance before freezing untouched qualification;
do not spend the remaining budget on an unpowered independent training ladder.
