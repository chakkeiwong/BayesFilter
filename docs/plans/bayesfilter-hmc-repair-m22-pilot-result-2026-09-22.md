# M22 null and complete-fit mutation pilots

Both pilots completed. The sequential test ran two independent complete
experiments each for baseline and no-op, with four valid outcomes and no
rejections. The complete-fit pilot ran two independently simulated datasets
times three independent full fits for each of baseline, no-op and a half-SD
posterior translation: 18 fits. Every fit completed its public preparation,
tuning and posterior checks. None of the three tiny rank experiments detected
a discrepancy. These pilots establish execution and cost, not size or power.

The diagnostic mutation evaluates both the prior and likelihood at
`q - d*s`, where `s^2 = 1/(tau^-2 + n*sigma^-2)` and `d=0.5`. Thus the fitted
normal-conjugate posterior is translated by half its posterior standard
deviation. The data generator and reference law are unchanged. Law, score,
curvature, no-op and public preparation tests check activation independently.
The control changes the mathematical target deliberately and remains confined
to diagnostic validation code. It is not an eligible runtime model variant.

Results are preserved under `artifacts/hmc-repair-master-2026-09-16/m22-r1/`:
`null-pilot-cpu-r1` and `full-fit-pilot-cpu-r1`. Their source is the frozen
pre-late-merge `source-r1`; `source-manifest-r1.json` hashes each executable
file. The separate `source-r2` pins the same executable files' commit metadata.
Neither snapshot includes the incoming ESS repair. All workers used the tfgpu
Python environment, hidden GPUs and one intra/inter-op thread. Exact commands,
seeds, runtime policy and evidence paths are in the run indices and manifests.

The full-fit workers consumed 1117.1194969150238, 1157.9721191899152 and
1100.529405809939 seconds (3375.621021914878 total); the null pilot consumed
26.38678574701771. Coordinator waiting/wall time is not charged again. The
full-fit average is about 188 worker-seconds per fit. At that measured rate,
one 128-dataset, three-fit SBC experiment costs about 20 CPU hours per arm.
Repeated experiments needed for a power interval exceed M22's 12-hour
allocation even before adding baseline/no-op arms. Reporting a single smaller
SBC nonrejection would not close the subtle-defect-power gap.

| Decision | Primary criterion | Veto status | Main uncertainty | Next action | Not concluded |
| --- | --- | --- | --- | --- | --- |
| Continue sequential null confirmation | Four pilot experiments valid | None | Tiny pilot denominator | Freeze 512 fresh experiments per arm on merged source | Exact nominal size |
| Accept full-fit mutation engineering check | All 18 fits complete and activation tests pass | None | Only two ranks per control | Preserve measured cost; design an affordable sensitivity study before launch | Calibrated defect power |
| Keep whole-fit power open | Repeated 128-dataset study unaffordable in phase allocation | Budget prevents this inventory, not other work | Small effects and three-rank resolution | Use analytic sensitivity planning to choose a justified future inventory; do not shrink it silently | Complete-pipeline calibration |

| Inference status | Finding |
| --- | --- |
| Hard veto screen | No incomplete fit or invalid pilot experiment |
| Statistically supported ranking | None |
| Descriptive-only differences | Durations and rank p-values from two datasets |
| Default readiness | No numerical default or testing threshold changed |
| Next evidence needed | Independent null-size confirmation; separately funded/resolved full-fit power inventory |

The strongest alternative to "no defect" is that this pilot has almost no
power to detect the injected shift. Independent target/score checks establish
that the mutation was active. The nonrejection therefore provides no evidence
that the pipeline is calibrated. Remote-source integration changes the next
confirmation source and requires a new inventory/output root; it does not
relabel or erase these completed pilots.
