# Controlled-SMC/KDM C0 source-matrix lock — 2026-10-03

This is the first executable phase of the replication-first master program:
[`controlled-smc-kdm-replication-first-execution-20261003.md`](controlled-smc-kdm-replication-first-execution-20261003.md).
It records the author repository, paper copy, drivers, data, runtime, and
source-versus-blocked status before any translated TensorFlow run.

## Research intent

The question is whether each selected controlled-SMC experiment has a source
driver and source inputs that can be executed without silently replacing the
author method. C0 does not test numerical parity and does not promote a
reconstruction. It distinguishes a runnable source matrix from a matrix that
needs a named provenance repair.

The source repository is pinned at commit
`2696f6d611aae172d10102cb7bfbfd314bb6b88f`. The paper evidence is the local
author-hosted manuscript and supplement at
`.localresources/papers/heng-controlled-smc-author-manuscript-and-supplement.pdf`.
The checked source mapping is preserved in
`docs/reference/controlled-smc-remaining-source-audit-20260929.md`.

## Evidence contract

| Field | C0 definition |
|---|---|
| Question | Are the selected source matrices and their provenance executable or explicitly blocked? |
| Comparator | The pinned author source tree and the cited manuscript/supplement, not a reconstructed TensorFlow driver |
| Pass criterion | Repository commit, required source/data files, paper copy, and source runtime are recorded; every matrix is `READY_FOR_SOURCE_COST_PILOT` (or a named generative-design limitation) or has a named missing item |
| Vetoes | Pinned commit mismatch; missing required source/data file; missing paper copy; unavailable source runtime; hidden driver/data discrepancy |
| Explanatory diagnostics | File hashes, source line settings, existing Lorenz pilot record, observation-row counts, and missing-driver detection |
| Nonclaims | C0 does not establish source output parity, TensorFlow parity, convergence, statistical superiority, or scientific validity |
| Artifact | Campaign `attempt*_c0_source_lock/summary.json`, `source-lock.json`, `manifest.json`, and `command.log` |

The cheap heuristic set is not applicable to this provenance lock. No filter is
being promoted or compared here; source execution is the prerequisite for
constructing the later conditional comparison table.

## Locked matrix

* **Lorenz Figure 6:** `cSMC_VaryF.m` and `APF_VaryF.m`, dimension 8,
  `T=100` (101 observations), forcing grid `linspace(2.5,8.5,40)`, 100
  repetitions, cSMC `N=512`, up to four adaptive refinements, and APF particle
  counts selected from the cSMC refinement count. The supplied
  `SimulatedData.mat` is the fixed Figure 6 data. The six-case native source
  pilot is already recorded; the full 40×100 matrix remains open.
* **Neuroscience Figure 4 right:** `NormConst_cSMC.m` and `NormConst_BPF.m`,
  supplied `Data_NeuroScience.mat`, 20 values of `sigmasq`, 100 repetitions,
  cSMC `N=128, I=3`, and BPF `N=5529, I=0`. The driver and data are present;
  source execution is the next C1 action.
* **Multimodal Table 2:** the algorithm and supplied data are present, but the
  named `Repeat.m` driver is absent. The source files use `T=100` and 101
  rows, while Appendix F.2 states `T+1=100`. This matrix is blocked until the
  author driver or an explicitly labelled reconstruction is recovered. No
  101-row run may be called a literal 100-observation source replication.

* **Lorenz Table 1:** all four `Repeat_Dim*.m` drivers are retained. They
  generate a new dataset for each variance. A seeded rerun can replicate this
  generative design; the exact original unseeded datasets remain unavailable.

## Skeptical audit before execution

The old master dispatcher still contained the historical repair ladder and
could launch those phases even though the revised master classifies them as
historical. C0 therefore adds a dedicated executable phase and the dispatcher
will reject historical-phase launches. The default manifest plan is changed to
the replication-first plan. The old attempts remain preserved as historical
diagnostics and are not prerequisites for C0 or C1.

The C0 script is metadata-only and uses the Python standard library. It does
not import NumPy, TensorFlow, or the candidate filter, so no XLA, pfor, or GPU
claim is hidden in this phase. It runs with CUDA hidden and records the Octave
runtime for the later source lane.

## Exact command and stop rule

```text
/home/chakwong/anaconda3/envs/tftwogpu/bin/python -B scripts/run_controlled_smc_kdm.py run --phase c0_source_lock
```

The phase is bounded to at most two local infrastructure attempts and 30 seconds in total. Stop if the source
commit, required file set, paper copy, or Octave runtime check fails. A
multimodal `BLOCKED_MISSING_AUTHOR_DRIVER` row is an expected C0 outcome and
does not invalidate the source lock; it blocks only that matrix's C1 source
run.

## Next action after a passing lock

Write the C1 source-run driver specification with exact output statistics and
seeds, then run the unchanged author neuroscience driver and complete the
Lorenz source matrix in bounded source-native tranches. Keep source output
conventions and normalized likelihood conventions separate. Only after C2
source verdicts may P0/P1 translate the same locked inputs to TensorFlow.
Any damping, projection, precision change, covariance floor, KL control, or
other numerics-altering route remains E0 and cannot repair a replication gap.

The revised allocation draws only on the original unspent CPU grant (about
286 seconds at review start); it does not convert GPU hours into CPU hours.
C0 plus a separate <=210-second source cost pilot may use at most four attempts
in total. Full source matrices require a costed CPU allocation before launch.
All selected MATLAB files are checked against the pinned Git blobs and copied
with their relative paths; a matching HEAD alone is insufficient provenance.
