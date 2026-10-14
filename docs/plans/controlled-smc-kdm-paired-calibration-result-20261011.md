# Repaired KDM paired calibration result

Updated 12 October 2026. Calibration completed at 04:28:11 HKT after
4h34m46s elapsed. Both new TensorFlow training runs completed all three stages,
independent terminal audits and common-author evaluations. The terminal review
passed 72/72 completion, source/input hash, aggregation and budget checks.
KDM replication remains incomplete: the two new MDPF models have substantially
unfavorable observed outcomes relative to their paired author models.

See [the phase plan](controlled-smc-kdm-paired-calibration-20261011.md).
All paths below are under
`artifacts/controlled-smc-kdm-replication-20260928/paired-calibration-20261011/calibration-01/`.
The exact evidence is `manifest.json`, `decision.json`, each unit's `audit.json`
and `evaluation/result.json`, and `terminal-review-20261012.json`.

The original pair's evaluation replayed exactly. Each new full-data smoke and
terminal audit passed. Source/input hashes remain unchanged, all ten supervised
steps exited zero, and each resource charge appears once in the campaign ledger.
The recovered author unit retains its previously audited, unobserved exit code;
this does not invent a successful original process exit.

Differences below are TensorFlow minus author NLL and the percentage change in
RMSE. NLL is averaged over the two fixed evaluation seeds. RMSE percentages
convert the mean log RMSE ratio, so they are geometric averages of the two
ratios. The independent training sample size is three, not six.

| Training pair | MDPF NLL difference (nat) | MDPF RMSE change | A-MDPF NLL difference (nat) | A-MDPF RMSE change |
|---|---:|---:|---:|---:|
| Original repaired pilot | -0.23771 | -2.72% | -0.10452 | -1.97% |
| New unit 01 | +0.85829 | +26.59% | +0.09102 | +2.62% |
| New unit 02 | +0.76678 | +36.03% | +0.15265 | +1.19% |

These are finite unfavorable development outcomes, not crashes or a population
inferiority finding. They cannot be discarded, replaced with favorable seeds,
or excused as intrinsic author failures: the author comparators completed and
were valid. The earlier native TensorFlow endpoint parity audit establishes
mechanics on its checked inputs; it does not establish learned quality for new
training runs. Current evaluation uses the common pinned author filter for
both checkpoints, so a difference between native inference implementations is
not responsible for the reported comparison.

The MDPF gap is already visible at the chosen validation checkpoint. Source/TF
best validation losses are 5.83935/6.56454 in unit 01 and 6.11279/6.62254 in
unit 02. Weight-pretraining losses are much closer, 4.70610/4.71163 and
4.70144/4.70005, respectively. Both frameworks followed the audited source
stopping controller. This localizes an observed gap to the trained MDPF stage;
it does not prove whether preceding weights, native random streams, accumulated
FP32 update discrepancies, or optimization variability caused it.

Paired NLL standard deviations are 0.60808 nat for MDPF and 0.13427 nat for
A-MDPF. With the unchanged 0.01 nat / 1% criteria, the predeclared normal
approximation at true equality suggests 35,147 / 1,714 units for NLL and
2,991 / 53 for RMSE. These are unstable n=3 plug-in illustrations, not validated
sample sizes. They provide no defensible affordable qualification design within
the current allocation. Do not launch such a ladder or silently relax criteria.

Next, inspect and replay the saved stage boundaries before another full run:
verify that the actual MDPF consumer receives the intended weights/bandwidth,
optimizer state and draw schedule; then compare both implementations on common
saved states, batches and random draws to distinguish a translation mismatch
from divergence between valid training trajectories. Existing unresolved FP32
proposal-gradient differences are a hypothesis to test, not an established
explanation for this larger learned-quality gap. Preserve all three outcomes;
any repair must follow the author operation and precede fresh qualification.

| Decision | Primary criterion | Veto diagnostics | Main uncertainty | Next justified action | Not concluded |
|---|---|---|---|---|---|
| Calibration complete; replication incomplete | Three audited pairs and required evaluations complete | No infrastructure, artifact or nonfinite veto; new MDPF outcomes unfavorable | Cause and population frequency of training gap; uncertainty calibration | Bounded localization on saved training states before another full campaign | No population noninferiority, supported ranking or paper-wide replication |

| Inference status | Finding |
|---|---|
| Hard veto screen | Terminal/comparator checks pass; no execution failure |
| Statistically supported ranking | None from three development pairs |
| Descriptive-only differences | All NLL/RMSE differences, validation losses, SDs and planning counts above |
| Default-readiness | Not established by this phase |
| Next evidence needed | Source-preserving localization, then a defensible untouched comparison design |

Post-run red-team: the strongest alternative to a translation defect is
ordinary optimization variability under independent native random streams.
Shared initialization and data do not couple training draws or guarantee lower
paired variance; observed MDPF source/TF NLL covariance is negative. The weakest
evidence is the three-unit estimate of population variability. Common-state
update replay or a source-preserving boundary correction would discriminate
mechanisms; a valid independent training comparison could overturn the apparent
pattern. A short replay alone cannot prove population no-worse behavior.

Command: `/home/chakwong/anaconda3/envs/tftwogpu/bin/python docs/benchmarks/run_kdm_paired_calibration.py --execute`.
Checkout `/home/chakwong/BayesFilter`, branch `surrogate-hmc`, commit
`89065bc6354801cb368d5e163ba49fd9c4372d10`; uncommitted source/input hashes
are preserved in the manifest. GPU/XLA, TF32 off, verified memory growth and
one host thread; CPU author references intentionally hide GPU devices.
New TF seed bases were 1100000000 and 1200000000; evaluation seeds were
202610082 and 202610083. Data and constructor identities are in the manifest.
Resource charges: GPU 15,789.925614 s; CPU 183.865753 s. Remaining campaign
tranche: GPU 91,743.507420 s (25.48h), CPU 220,555.559930 s (61.27h).
The larger aggregate GPU balance is not another allocation. No run is active.
