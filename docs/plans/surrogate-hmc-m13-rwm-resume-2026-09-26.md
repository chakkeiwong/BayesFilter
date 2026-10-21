# M13 interrupted comparator continuation

This is a local repair within the authorized surrogate-HMC master, not a new
scientific direction. The question remains whether the frozen DCTR proposal
force yields valid posterior sampling when the separate frozen full-support
endpoint supplies MH correction. RWM is the same-endpoint classical comparator.

RWM attempt01 reached its 3495-worker-CPU watchdog after preserving 512+512
adaptive warmup, 500 fresh fixed-kernel warmup and the first 250 retained draws
per chain. The total receipt is 3497.907818 CPU seconds. Source checks passed.
This is a budget/forecast failure, with no posterior conclusion. Do not repeat
the warmup or include it among retained draws.

Resume the complete archived chunks in a fresh output directory. Reconstruct
the frozen covariance from the exact original warmup formula, reload the
frozen scale, and continue fold-in seeds 11 through17 after retained chunk00.
Validate tensor shapes, finiteness, health, state continuity, frozen scales,
and saved/source identity before computing. Preserve the prior tensors and
their new checksum inventory. The original source is saved in
`docs/plans/artifacts/surrogate-hmc-m13-rwm-resume-audit-20260926-01` and matches
the interrupted manifest. Compare the numerical kernel and constants with it.

The exact target, data1704/T20, prior, endpoint4096x8/FP64, four chains,
proposal, seeds, looks1000/2000 and all assessment margins are unchanged.
Coverage/reference02 remains the independent comparator. Promotion requires
the previously declared posterior equivalence assessment; health failures
veto qualification. Acceptance and timing remain descriptive. No HMC tuning
authority, convergence claim from warmup, or speed ranking follows from RWM.

Use at most6000 additional CPU seconds, one resumed attempt, from the existing
10000-CPU repair allocation. The materiality repair already consumed
1386.822697 CPU seconds, leaving8613.177303 before this attempt. Charge focused
tests to the remaining shared baseline allocation; record every receipt once.
No GPU work is added. The CPU-only reference exception explicitly hides GPUs
before TensorFlow import. A second budget stop or invalid archive is reported
and cannot silently extend this allowance.

Skeptical audit: original allocations underestimated comparator cost. Restarting
all warmup would waste the reserve, while shortening the retained run would
change the evidence contract. State continuation preserves the actual Markov
chain and independent seed sequence. A Gaussian interrupted-versus-uninterrupted
continuation test and corrupted/inconsistent archive rejection are required.
The main risk is mixing archived and current targets; unchanged numerical
source checks and exact reference/coverage linkage address it. The independent
GPU public tuning snapshot is unaffected by these repair files.

Fix the retained supervisor to recognize a terminal baseline watchdog failure
instead of waiting until timeout. It will then wait on the repaired baseline
path and the ongoing public tuning, and launch the already authorized retained
pilot only after both prerequisites qualify for their declared roles.

The bounded launcher (which records the repair receipt and enforces the
6000-CPU allowance) is:
```
/home/chakwong/anaconda3/envs/tftwogpu/bin/python scripts/run_surrogate_hmc_m13_rwm_resume.py --previous docs/plans/artifacts/surrogate-hmc-m13-assessment-rwm-20260926-01 --reference docs/plans/artifacts/surrogate-hmc-m13-assessment-reference-20260926-02/worker/reference/reference.json --coverage docs/plans/artifacts/surrogate-hmc-m13-assessment-reference-20260926-02/worker/coverage.json --output docs/plans/artifacts/surrogate-hmc-m13-assessment-rwm-20260926-06
```

The launch-only retries using `...rwm-20260926-02` through
`...rwm-20260926-05` are preserved as failed infrastructure/metadata/code-path
attempts; the fresh directory above is required so that each receipt and result
remains immutable.
