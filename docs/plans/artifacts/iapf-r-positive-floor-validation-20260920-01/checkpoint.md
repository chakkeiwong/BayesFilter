# Active R iAPF positive-floor validation

2026-09-20; /home/chakwong/BayesFilter; surrogate-hmc;
HEAD6fbcf3147660c40d5d5644bbcbcc9fadbcb06aef. Preserve unrelated edits.
Owner requested program refresh and execution. Plan:
docs/plans/iapf-r-positive-floor-validation-2026-09-20.md.
Skeptical review passed; source trace confirms fresh final draw and caps20/16000.

Question: frozen optional log_quadratic/power8 R iAPF likelihood accuracy on
two untouched d80 datasets, with d40 regression control, all T100/32 repeats.
Each conditional bootstrap95 interval must lie in[.9,1.1]; also require valid
fits, Gaussian-limit tail margins and ordinary/large-innovation heuristic checks.
No author Eq15/paper-scale replication, default/TF/KDM/LEDH/score/HMC conclusion.

Budget:4000 worker-seconds (3800 replication,200 post-run), mechanics120;
three scientific launches, at most one localized infrastructure retry. CPU
intentional, GPUs hidden, one BLAS/OpenMP thread, at most two R workers.
Planned data seeds80000080/81000080/80000040; repetition IDs501:532/601:632/701:732.
Previous repair stage is closed in ../iapf-r-log-fit-validation-20260920-01.

Current stage COMPLETE: all96 repetitions PASS the per-dataset accuracy,
tail and heuristic screens. A: mean1.020140718, bootstrap95[.975003908,1.073665459];
B: mean.981627897,[.927681137,1.036751856]; d40:1.041180777,[1.005135661,1.078378009].
The d40 interval excludes1, although inside the declared10% tolerance. This is
an explanatory discrepancy, not evidence of unbiasedness or a new veto.
All64100 fits and9600 tail checks pass, minimum relative margin.301641955.
d80 final N2000,7..9 passes; d40 N1000/2000,7/8 passes. Max final floor p9.485e-20.
Ten pytest cases pass3.36sec; complete evidence accepted and six corruptions
rejected. Core unchanged. Result.md/summary.json and per-attempt manifests,
inspection.json/diagnostic-manifest.json preserve evidence and captured sources.

Charged3146.026614/4000 seconds (3085.985672 replication+60.040942 post-run);
853.973386 remain. All3 scientific launches complete, no scientific retry,
no process running, no further launch allocated. One pre-launch report wrapper
read a running diagnostic as final; fixed timeout reservation, charged1sec.
See report-repair-note.md. Mechanics conservatively10/120sec before closeout.

Exact next action: plan/review an additional untouched d40 control and bounded
scaling of the optional reference. Keep original-paper objective/solver/floor/
author provenance reconciliation separate:1000 repeats of log_quadratic cannot
close Eq.15 conformance. Do not tune on these datasets. A later TF comparison
must name this qualified alternative; no paper replication/default/score/HMC/
KDM/LEDH claim. Current master and reference documentation refreshed.
